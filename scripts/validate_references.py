"""Physics-only A/B validation of servo references; no agent is constructed."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np

from ergocub_env import ErgoCubEnv, embodied
from ergocub_physics import PROJECT, REFERENCE_MODES
from control_feasibility import gravity_reference
from diagnose_backward import snapshot
from embodied.agents.dreamerv3 import train as upstream


def limits(r):
    m,d=r.model,r.data
    limited=np.flatnonzero(m.jnt_limited)
    q=d.qpos[m.jnt_qposadr[limited]]
    violation=np.maximum(np.maximum(m.jnt_range[limited,0]-q,q-m.jnt_range[limited,1]),0)
    active=[]
    # Auto Jacobian is dense at nv=33; inspect unilateral joint-limit reactions.
    jac=d.efc_J.reshape(d.nefc,m.nv)
    for i in range(d.nefc):
        if d.efc_type[i]!=mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT:
            continue
        j=int(d.efc_id[i]); v=m.jnt_dofadr[j]
        active.append(dict(joint=m.joint(j).name,force=float(d.efc_force[i]),
                           generalized_force=float(jac[i,v]*d.efc_force[i])))
    return dict(max_joint_limit_penetration=float(violation.max()),limit_reactions=active,
                joint_limit_penetration={m.joint(j).name:float(v) for j,v in zip(limited,violation) if v>0},
                warning_counts=[int(w.number) for w in d.warning])


def rollout(mode,case,action,config,out):
    env=ErgoCubEnv(evaluation=True,reference_mode=mode)
    wrapped=upstream.wrap_env(env,config)
    wrapped.step(dict(reset=True,action=np.zeros(23,np.float32)))
    rows=[]; observations=[]
    def capture():
        row=snapshot(env); row.update(limits(env.robot))
        rotations=env.robot.data.geom_xmat[env.guard_ids].reshape(-1,3,3)
        centers=env.robot.data.geom_xpos[env.guard_ids]+np.einsum("nij,nj->ni",rotations,env.guard_centers)
        clear=centers[:,2]-np.einsum("ni,ni->n",np.abs(rotations[:,2]),env.guard_extents)
        g=int(env.guard_ids[np.argmin(clear)])
        row["lowest_nonfoot_body"]=env.robot.model.body(int(env.robot.model.geom_bodyid[g])).name
        rows.append(row)
    capture()
    fall=env.fallen
    def observed_fall():
        result=fall(); capture(); return result
    env.fallen=observed_fall
    for _ in range(env.task["max_steps"]):
        obs=wrapped.step(dict(reset=False,action=action.copy()))
        observations.append({k:v.tolist() for k,v in obs.items()})
        if obs["is_last"]:
            break
    r=env.robot
    assert np.isfinite(np.array([x["q"]+x["dq"]+x["torque"] for x in rows])).all()
    assert max(max(x["effort_fraction"]) for x in rows)<=1+1e-12
    assert not any(any(x["warning_counts"]) for x in rows)
    assert all(x["clearance"]>0 for x in rows),"Stopped after an absent body-ground impact"
    assert all(x["contact"]["effective_friction"]==[1.]*x["contact"]["contact_count"] for x in rows)
    summary=dict(duration=rows[-1]["time"],ending="fall" if obs["is_terminal"] else "timeout",
        termination=[k for k in ("tilt_guard","height_guard","clearance_guard") if rows[-1][k]],
        peak_effort_fraction=max(max(x["effort_fraction"]) for x in rows),
        peak_joint_limit_penetration=max(x["max_joint_limit_penetration"] for x in rows),
        peak_limit_sample=max(rows,key=lambda x:x["max_joint_limit_penetration"]),
        peak_limit_penetration_first_second=max(x["max_joint_limit_penetration"] for x in rows if x["time"]<=1),
        warning_counts=rows[-1]["warning_counts"],
        peak_slip_speed=max(x["contact"]["max_contact_slip_speed"] for x in rows),
        peak_contact_penetration=max(max(0.,-min(x["sole_envelope"]["bottom_corner_z"])) for x in rows),
        min_loaded_contacts=min(x["contact"]["contact_count"] for x in rows),
        max_loaded_contacts=max(x["contact"]["contact_count"] for x in rows),
        minimum_clearance=min(x["clearance"] for x in rows),
        knee_targets=rows[1]["target"][8:10],
        at_200ms=min(rows,key=lambda x:abs(x["time"]-.2)),final=rows[-1])
    (out/f"{mode}_{case}.jsonl").write_text("".join(json.dumps(x)+"\n" for x in rows))
    wrapped.close()
    return summary,rows,observations


def compiled_checks():
    a=ErgoCubEnv(reference_mode="joint_range"); b=ErgoCubEnv(reference_mode="unclipped")
    x,y=a.robot,b.robot
    differences=[]
    for name in dir(x.model):
        if name.startswith("_"):
            continue
        v,w=getattr(x.model,name),getattr(y.model,name)
        if isinstance(v,np.ndarray) and not np.array_equal(v,w):
            differences.append(name)
    assert differences==["actuator_ctrllimited"],differences
    for name in dir(x.model.opt):
        if name.startswith("_"):
            continue
        v,w=getattr(x.model.opt,name),getattr(y.model.opt,name)
        if isinstance(v,np.ndarray) or isinstance(v,(int,float)):
            np.testing.assert_array_equal(v,w)
    assert x.names==y.names
    for name in ("qreset","nominal","scale"):
        np.testing.assert_array_equal(getattr(x,name),getattr(y,name))
    assert np.all(x.model.actuator_ctrllimited) and not np.any(y.model.actuator_ctrllimited)
    assert all(y.model.jnt_limited[1:]) and not y.model.jnt_limited[0]
    assert not (y.model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_LIMIT))
    assert np.all(y.model.actuator_forcelimited)
    # Zero-time actuator-only stress probe: no integration, outside the normalized
    # environment interface. Verify actual force clipping in both directions.
    for sign in (-1,1):
        y.reset()
        caps=y.model.actuator_forcerange[:,1]
        y.set_targets(y.nominal+sign*2*caps/y.constants.STIFFNESS)
        mujoco.mj_forward(y.model,y.data)
        np.testing.assert_allclose(y.data.actuator_force,sign*caps,atol=1e-10)
    # Exercise both public APIs: the physics helper and the actual environment.
    ref,_=gravity_reference(y)
    x.step(ref); y.step(ref)
    np.testing.assert_allclose(x.data.ctrl[8:10],.12217304763960307,atol=1e-14)
    np.testing.assert_array_equal(y.data.ctrl,ref)
    # Normalized action clipping is unchanged even for out-of-range caller input.
    for env in (a,b):
        env.reset(); env.step(dict(reset=False,action=np.full(23,2,np.float32)))
        np.testing.assert_array_equal(env.previous,np.ones(23,np.float32))
    result=dict(changed_model_arrays=differences,limited_physical_joints=int(sum(y.model.jnt_limited)),
        limited_actuator_forces=int(sum(y.model.actuator_forcelimited)),
        physical_joint_ranges=y.model.jnt_range[1:].tolist(),effort_limits=y.model.actuator_forcerange.tolist(),
        reset_and_action_scale_identical=True,solver_options_identical=True,normalized_bounds_unchanged=True,
        physics_step_mode_checked=True,zero_time_force_caps_verified_both_signs=True)
    a.close(); b.close()
    return result


def knee_probes(gravity_action):
    result=[]
    for mode in REFERENCE_MODES:
        env=ErgoCubEnv(evaluation=True,reference_mode=mode); r=env.robot
        for baseline,base in (("zero",np.zeros(23,np.float32)),("gravity",gravity_action)):
            for i in (8,9):
                for delta in (-.05,0.,.05):
                    env.reset(); action=base.copy(); action[i]+=delta
                    assert np.abs(action).max()<=1
                    # Same affine arithmetic as upstream NormalizeAction.
                    applied=(action+1)/2*2-1
                    r.set_targets(r.nominal+r.scale*applied)
                    mujoco.mj_forward(r.model,r.data)
                    result.append(dict(mode=mode,baseline=baseline,joint=r.names[i],delta_action=delta,
                        target=float(r.data.ctrl[i]),actual=float(r.data.qpos[r.qadr[i]]),
                        actuator_force=float(r.data.actuator_force[i]),time=float(r.data.time)))
        env.close()
    for mode in REFERENCE_MODES:
        for baseline in ("zero","gravity"):
            for joint in ("l_knee","r_knee"):
                rows=[x for x in result if x["mode"]==mode and x["baseline"]==baseline and x["joint"]==joint]
                forces=[x["actuator_force"] for x in rows]
                if mode=="joint_range" and baseline=="gravity":
                    np.testing.assert_array_equal(forces,[forces[0]]*3)
                else:
                    np.testing.assert_allclose(np.diff(forces),[1.375]*2,atol=4e-6)
    return result


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout="constrained")
    for mode,color,style in (("joint_range","tab:blue","-"),("unclipped","tab:orange","--")):
        rows=[json.loads(l) for l in (out/f"{mode}_gravity.jsonl").read_text().splitlines()]
        t=[x["time"] for x in rows]
        axes[0,0].plot(t,[x["root"][2] for x in rows],color=color,linestyle=style,label=mode)
        axes[0,1].plot(t,[x["root_pitch_deg"] for x in rows],color=color,linestyle=style,label=mode+" root")
        axes[0,1].plot(t,[x["torso_pitch_deg"] for x in rows],color=color,alpha=.45,linestyle=style,label=mode+" chest")
        axes[0,2].plot(t,[x["contact"]["com_support_margin"] for x in rows],color=color,linestyle=style)
        for ax,i in zip(axes[1,:2],(8,9)):
            ax.plot(t,[x["q"][i] for x in rows],color=color,linestyle=style,label=mode+" actual")
            ax.plot(t,[x["target"][i] for x in rows],color=color,linestyle=":",label=mode+" target")
        axes[1,2].plot(t,[max(x["effort_fraction"]) for x in rows],color=color,linestyle=style)
    for ax,title,unit in zip(axes.flat,("Root height","Root and chest sagittal lean","COM vs loaded support hull","Left knee: actual and target","Right knee: actual and target","Maximum actuator force / cap"),("m","deg (+x forward)","m","rad","rad","fraction")):
        ax.set(title=title,xlabel="time (s)",ylabel=unit); ax.grid(alpha=.25)
    axes[0,0].legend(); axes[0,1].legend(fontsize=7); axes[1,0].legend(fontsize=7)
    axes[0,1].axhline(-60,color="black",linestyle=":")
    axes[0,2].axhline(0,color="black",linestyle=":")
    axes[1,2].axhline(1,color="black",linestyle=":")
    fig.suptitle("Same gravity-reference actions and reset; only servo-reference clipping differs")
    fig.savefig(out/"gravity_comparison.png",dpi=150)
    plt.close(fig)
    summary=json.loads((out/"summary.json").read_text())
    fig,axes=plt.subplots(1,3,figsize=(13,3.7),layout="constrained")
    for mode,color,style in (("joint_range","tab:blue","-"),("unclipped","tab:orange","--")):
        zero=[json.loads(l) for l in (out/f"{mode}_zero.jsonl").read_text().splitlines()]
        gravity=[json.loads(l) for l in (out/f"{mode}_gravity.jsonl").read_text().splitlines()]
        axes[0].plot([r["time"] for r in zero],[r["root"][2] for r in zero],color=color,linestyle=style,label=mode)
        probe=[r for r in summary["knee_probes"] if r["mode"]==mode and r["baseline"]=="gravity" and r["joint"]=="l_knee"]
        axes[1].plot([r["delta_action"] for r in probe],[r["actuator_force"] for r in probe],color=color,linestyle=style,marker="o",label=mode)
        axes[2].plot([r["time"] for r in gravity],[r["max_joint_limit_penetration"] for r in gravity],color=color,linestyle=style)
    for ax,title,xlabel,ylabel in zip(axes,("Zero action: identical trajectories","Left knee: gravity-reference perturbation","Gravity rollout: soft joint-limit penetration"),("time (s)","normalized action perturbation","time (s)"),("root height (m)","actuator force (Nm)","rad")):
        ax.set(title=title,xlabel=xlabel,ylabel=ylabel); ax.grid(alpha=.25)
    axes[0].legend(); axes[1].legend()
    fig.savefig(out/"validation_checks.png",dpi=150)
    plt.close(fig)


def run(args):
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    # Same upstream wrapper settings as the baseline, without requiring an old run.
    config=embodied.Config({"wrapper":{"discretize":0,"length":0,"reset":True,"checks":False}})
    env=ErgoCubEnv(evaluation=True); reference,gravity=gravity_reference(env.robot)
    action=np.clip((reference-env.robot.nominal)/env.robot.scale,-1,1).astype(np.float32)
    historical=PROJECT/"results/backward_fall_diagnosis_gpu/gravity_controls.jsonl"
    if historical.is_file():
        expected=json.loads(historical.read_text().splitlines()[0])["raw_action"]
        np.testing.assert_array_equal(action,expected)
    env.close()
    compiled=compiled_checks(); cases={}; zeros={}
    for mode in REFERENCE_MODES:
        for case,a in (("zero",np.zeros(23,np.float32)),("gravity",action)):
            result,rows,obs=rollout(mode,case,a,config,out)
            cases[mode+"_"+case]=result
            if case=="zero":
                zeros[mode]=(rows,obs)
    assert zeros["joint_range"]==zeros["unclipped"],"Zero trajectories/observations/rewards differ"
    for case,expected in (("zero",.686),("gravity",1.194)):
        assert abs(cases["joint_range_"+case]["duration"]-expected)<=.002
    np.testing.assert_allclose(cases["unclipped_gravity"]["knee_targets"],reference[8:10],atol=5e-8)
    summary=dict(compiled=compiled,cases=cases,gravity_action=action.tolist(),gravity_calculation=gravity,
        zero_trajectories_observations_rewards_bit_identical=True,original_fall_times_reproduced=True,
        knee_probes=knee_probes(action),agent_constructed=False,optimizer_updates_performed=0)
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    for p in (PROJECT/"scripts").glob("*.py"):
        if p.name in ("validate_references.py","ergocub_env.py","ergocub_physics.py","diagnose_backward.py","diagnose_physics.py","control_feasibility.py","dreamer.py"):
            shutil.copyfile(p,out/p.name)
    shutil.copyfile(PROJECT/"configs/servo_settings.json",out/"servo_settings.json")
    plot(out)
    print(json.dumps({k:{n:c[n] for n in ("duration","knee_targets","peak_effort_fraction","peak_joint_limit_penetration","peak_slip_speed","termination")} for k,c in cases.items()},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    run(p.parse_args())
