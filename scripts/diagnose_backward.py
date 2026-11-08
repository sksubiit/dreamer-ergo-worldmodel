"""Read-only matched-reset rollouts; writes only a new diagnostic directory.

Policy uses unchanged upstream eval sampling with seed/RNG replay, not a mean
actor substitution. No train(), optimizer updates, or physical changes.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from ruamel.yaml import YAML

from dreamer import embodied, fingerprint, make_agent, verify_upstream
from ergocub_env import ErgoCubEnv
from ergocub_physics import PROJECT, PHYSICS_DT, SUBSTEPS
from control_feasibility import gravity_reference
from diagnose_physics import contact_metrics, sole_pad_envelope


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(env):
    r=env.robot; m,d=r.model,r.data
    contacts=contact_metrics(r)[0]
    envelope=sole_pad_envelope(r)
    root=d.xmat[r.root_id].reshape(3,3)
    chest=d.xmat[m.body("chest").id].reshape(3,3)
    pitch=lambda rot: float(np.degrees(np.arctan2(rot[0,2],rot[2,2])))
    requested=m.actuator_gainprm[:,0]*d.ctrl+m.actuator_biasprm[:,1]*d.qpos[r.qadr]+m.actuator_biasprm[:,2]*d.qvel[r.vadr]
    effort=np.abs(d.actuator_force)/m.actuator_forcerange[:,1]
    hull=np.array(contacts["support_hull"])
    xy=d.subtree_com[r.root_id,:2]
    return dict(time=float(d.time),root=d.qpos[:3].tolist(),root_velocity=d.qvel[:3].tolist(),
        root_pitch_deg=pitch(root),torso_pitch_deg=pitch(chest),upright=float(root[2,2]),
        com=d.subtree_com[r.root_id].tolist(),contact=contacts,sole_envelope=envelope,
        sagittal_loaded_margin=(float(min(xy[0]-hull[:,0].min(),hull[:,0].max()-xy[0])) if len(hull) else None),
        target=d.ctrl.tolist(),q=d.qpos[r.qadr].tolist(),dq=d.qvel[r.vadr].tolist(),
        torque=d.actuator_force.tolist(),unclipped_servo_torque=requested.tolist(),
        effort_fraction=effort.tolist(),saturated=(effort>=.999).tolist(),
        clearance=env.clearance(),height_guard=bool(d.qpos[2]<env.task["fall_height_fraction"]*r.qreset[2]),
        tilt_guard=bool(root[2,2]<env.task["fall_upright_cosine"]),
        clearance_guard=bool(env.clearance()<env.task["nonfoot_clearance"]))


def rollout(case,task,config,output,agent=None,rng_state=None):
    from embodied.agents.dreamerv3 import train as upstream
    env=ErgoCubEnv(seed=0,evaluation=True,task=task)
    wrapped=upstream.wrap_env(env,config)
    obs=wrapped.step(dict(reset=True,action=np.zeros(23,np.float32)))
    r=env.robot
    reference,gravity=gravity_reference(r)
    fixed=np.clip((reference-r.nominal)/r.scale,-1,1).astype(np.float32)
    rows=[snapshot(env)]; controls=[]
    original=env.fallen
    def observed_fall():
        result=original()
        rows.append(snapshot(env))
        return result
    env.fallen=observed_fall
    if agent is not None:
        agent.rng.bit_generator.state=copy.deepcopy(rng_state)
        carry=agent.init_policy(1)
    for _ in range(task["max_steps"]):
        if agent is None:
            raw=fixed if case=="gravity" else np.zeros(23,np.float32)
        else:
            acts,carry=agent.policy({k:v[None] for k,v in obs.items()},carry,mode="eval")
            raw=acts["action"][0]
        previous=env.previous.copy(); start=r.data.time
        # Match upstream wrapper's ClipAction and NormalizeAction exactly.
        clipped=np.clip(raw,-1,1)
        mapped=(clipped+1)/2*2-1
        target_unclipped=r.nominal+r.scale*mapped
        obs=wrapped.step(dict(reset=False,action=raw))
        applied=env.previous.copy()
        dist=float(np.linalg.norm(r.data.site_xpos[env.palm]-env.goal))
        upright=float(r.data.xmat[r.root_id].reshape(3,3)[2,2])
        balance=float(np.clip((upright-.5)/.5,0,1)*np.exp(-((r.data.qpos[2]-r.qreset[2])/task["height_width"])**2))
        reach=float(np.exp(-(dist/task["reach_width"])**2))
        components=dict(balance=balance*task["balance_weight"],reaching=balance*task["reach_weight"]*reach,
            effort=-task["effort_cost"]*float(np.mean((r.data.actuator_force/r.model.actuator_forcerange[:,1])**2)),
            action_change=-task["action_change_cost"]*float(np.mean((applied-previous)**2)),
            base_speed=-task["base_speed_cost"]*float(np.sum(r.data.qvel[:2]**2)))
        recomputed=sum(components.values())*(r.data.time-start)/(PHYSICS_DT*SUBSTEPS)
        if obs["is_terminal"]:
            recomputed-=task["fall_penalty"]
        assert abs(recomputed-float(obs["reward"]))<1e-5
        controls.append(dict(time=float(r.data.time),raw_action=raw.tolist(),applied_action=applied.tolist(),
            unclipped_target=target_unclipped.tolist(),target=r.data.ctrl.tolist(),
            reward=float(obs["reward"]),reward_components=components,distance=dist,
            is_terminal=bool(obs["is_terminal"]),is_last=bool(obs["is_last"]),
            success=bool(obs["success"]),balance=balance,reach=reach))
        if obs["is_last"]:
            break
    for suffix,data in (("substeps",rows),("controls",controls)):
        (output/f"{case}_{suffix}.jsonl").write_text("".join(json.dumps(x)+"\n" for x in data))
    peak=np.max([x["effort_fraction"] for x in rows],axis=0)
    target_clip=np.array([x["unclipped_target"] for x in controls])-np.array([x["target"] for x in controls])
    result=dict(duration=rows[-1]["time"],termination=[k for k in ("height_guard","tilt_guard","clearance_guard") if rows[-1][k]],
        initial=rows[0],final=rows[-1],at_100ms=min(rows,key=lambda x:abs(x["time"]-.1)),
        at_200ms=min(rows,key=lambda x:abs(x["time"]-.2)),
        support_exit=next((x["time"] for x in rows if x["contact"]["com_support_margin"] is None or x["contact"]["com_support_margin"]<0),None),
        sagittal_support_exit=next((x["time"] for x in rows if x["sagittal_loaded_margin"] is None or x["sagittal_loaded_margin"]<0),None),
        max_effort_fraction=float(peak.max()),saturated_joints=[n for n,v in zip(r.names,peak) if v>=.999],
        position_target_clipped_joints=[n for n,v in zip(r.names,np.max(np.abs(target_clip),axis=0)) if v>1e-6],
        action_clip_fraction=float(np.mean(np.abs([x["raw_action"] for x in controls])>1)),
        reward_sum=sum(x["reward"] for x in controls),success=env.success_ever,
        minimum_hand_distance=min(x["distance"] for x in controls),
        peak_slip_speed=max(x["contact"]["max_contact_slip_speed"] for x in rows),
        gravity_calculation=gravity if case=="gravity" else None)
    wrapped.close()
    return result,controls


def run(args):
    verify_upstream()
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    run_dir=args.checkpoint.resolve().parent
    # Protect all existing run/evaluation files and philosophy, not just checkpoint.
    protected=[p for base in (run_dir,PROJECT/"results/reach_seed0_eval",PROJECT/"grand_philosophy")
               for p in base.rglob("*") if p.is_file()]
    before={str(p.relative_to(PROJECT)):sha(p) for p in protected}
    config=embodied.Config(YAML(typ="safe").load((run_dir/"config.yaml").read_text())).update({
        "logdir":str(out),"jax.platform":args.platform,"jax.prealloc":False})
    task=json.loads((run_dir/"reach_task.json").read_text())
    for name in ("ergocub_physics.py","ergocub_env.py"):
        assert (PROJECT/"scripts"/name).read_bytes()==(run_dir/name).read_bytes(),name
    assert (PROJECT/"configs/servo_settings.json").read_bytes()==(run_dir/"servo_settings.json").read_bytes()
    cases={}
    for name in ("zero","gravity"):
        cases[name],_=rollout(name,task,config,out)
    print("Physics controls complete",{k:v["duration"] for k,v in cases.items()},flush=True)
    agent=make_agent(config)
    checkpoint=embodied.Checkpoint(); checkpoint.agent=agent
    checkpoint.load(str(args.checkpoint.resolve()),keys=["agent"])
    state_hash=fingerprint(agent.save()); rng=copy.deepcopy(agent.rng.bit_generator.state)
    cases["policy"],controls=rollout("policy",task,config,out,agent,rng)
    repeat,repeated=rollout("policy_repeat",task,config,out,agent,rng)
    assert controls==repeated and cases["policy"]==repeat
    assert fingerprint(agent.save())==state_hash and int(agent.updates)==0
    assert all(sha(PROJECT/p)==value for p,value in before.items())
    summary=dict(cases=cases,joint_order=list(cases["gravity"]["gravity_calculation"]["required_torque"]),
        deterministic_policy_repeat=True,policy_mode="Unchanged eval sampler, seed 0, restored inference RNG/carry",
        pitch_definition="degrees atan2(body world up-vector x,z): positive leans +x, negative -x; root/chest separately",
        optimizer_updates_performed=0,unchanged_agent_state=True,protected_files_unchanged=len(before),protected_hashes=before)
    # Explicit names and bounds for interpreting every recorded joint vector.
    e=ErgoCubEnv(evaluation=True); r=e.robot
    summary["joint_order"]=r.names
    summary["action_scale"]=r.scale.tolist()
    summary["joint_ranges"]=r.model.actuator_ctrlrange.tolist()
    summary["effort_limits"]=r.model.actuator_forcerange.tolist()
    e.close()
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    for name in ("diagnose_backward.py","ergocub_env.py","ergocub_physics.py","control_feasibility.py","diagnose_physics.py"):
        (out/name).write_bytes((PROJECT/"scripts"/name).read_bytes())
    print(json.dumps({k:{n:c[n] for n in ("duration","termination","support_exit","max_effort_fraction","position_target_clipped_joints","reward_sum")} for k,c in cases.items()},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,default=PROJECT/"runs/reach_seed0_short/final.ckpt")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--platform",choices=("gpu","cpu"),default="gpu")
    run(p.parse_args())
