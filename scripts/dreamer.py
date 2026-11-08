"""Thin factories and readiness/evaluation commands for pinned upstream Dreamer.

Only the explicit `train` subcommand invokes the upstream training loop.
"""
import argparse
from contextlib import nullcontext
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

os.environ.setdefault("MUJOCO_GL","osmesa")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
from ergocub_env import ErgoCubEnv
from ergocub_physics import PROJECT, REFERENCE_MODES
import embodied
import numpy as np

REVISION="cb1189039151c8aadaaa987b442da54383c87fab"
UPSTREAM=PROJECT/"vendor/humanoid-bench"


def verify_upstream():
    actual=subprocess.check_output(["git","-C",str(UPSTREAM),"rev-parse","HEAD"],text=True).strip()
    if actual!=REVISION or subprocess.check_output(["git","-C",str(UPSTREAM),"diff","HEAD","--","dreamerv3"],text=True):
        raise ValueError("Expected clean pinned upstream Dreamer source")


def configuration(args):
    from embodied.agents.dreamerv3.agent import Agent
    if args.mode=="eval" or args.mode=="train" and args.checkpoint:
        from ruamel.yaml import YAML
        config=embodied.Config(YAML(typ="safe").load((args.checkpoint.parent/"config.yaml").read_text()))
    else:
        config=embodied.Config(Agent.configs["defaults"]).update(Agent.configs["humanoid_benchmark"])
        if args.profile=="small":
            config=config.update(Agent.configs["small"])
    # Old checkpoints lack this metadata; absent means the exact upstream path.
    mode=getattr(args,'prior_prediction',None) or config.get('prior_prediction','off')
    config=embodied.Config({**dict(config),'prior_prediction':mode})
    return config.update({
        "task":"ergocub_balance_left_reach_v0","method":"dreamer","seed":args.seed,
        "logdir":str(args.logdir),"jax.platform":args.platform,"jax.prealloc":False,
        "run.num_envs":1,"run.driver_parallel":False,"run.steps":args.steps,
        "run.log_every":10,"run.save_every":30,"run.eval_every":1000,
        "run.eval_eps":5,"run.usage.nvsmi":False,"run.wandb":False,
        "run.log_keys_video":["image"],"run.log_keys_avg":"log_(balance|reach|hand_distance|joint_error|effort|clearance)",
        "run.log_keys_max":"log_(episode_success|upright_seconds|action_clipped)",
        "replay_size":100000})


def make_agent(config):
    from embodied.agents.dreamerv3.agent import Agent
    mode=config.get('prior_prediction','off')
    if mode=='ten_step':
        from prior_prediction import Agent
    elif mode!='off':
        raise ValueError(f'Unknown prior-prediction mode: {mode}')
    env=ErgoCubEnv(seed=config.seed,evaluation=True)
    try:
        # Keep diagnostic success/log values out of policy and prediction heads.
        obs={k:v for k,v in env.obs_space.items() if not k.startswith("log_") and k!="success"}
        return Agent(obs,env.act_space,config)
    finally:
        env.close()


def gpu_check():
    import jax
    import jax.numpy as jnp
    device=jax.devices("gpu")[0]
    # Computation and synchronized host verification, not detection alone.
    x=jax.device_put(np.ones((1024,1024),np.float32),device)
    y=jax.jit(lambda a:a@a+3)(x)
    y.block_until_ready()
    value=np.asarray(jax.device_get(y))
    np.testing.assert_allclose(value,1027.)
    return dict(jax=jax.__version__,device=device.device_kind,
                output_devices=[str(d) for d in y.devices()],shape=list(value.shape),
                verified_value=float(value[0,0]),memory_stats=device.memory_stats())


def fingerprint(state):
    import jax
    h=hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(state):
        a=np.asarray(leaf)
        h.update(str((a.shape,str(a.dtype))).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def copy_resume_replay(checkpoint,logdir):
    """Replay chunks are external to the checkpoint; preserve the source run."""
    for name in ("train_replay","eval_replay"):
        source=checkpoint.resolve().parent/name
        if not source.is_dir():
            raise ValueError(f"Resumption requires the original replay directory: {source}")
        shutil.copytree(source,logdir/name)


def environment_checks(output,task=None):
    from ergocub_env import ErgoCubEnv
    import mujoco
    env=ErgoCubEnv(seed=7,evaluation=True,output=output/"environment",video=True,task=task)
    zero=dict(reset=False,action=np.zeros(23,np.float32))
    first=env.step(dict(zero,reset=True))
    assert first["is_first"] and not first["is_last"] and first["reward"]==0
    for k,v in first.items():
        assert v in env.obs_space[k],(k,v.shape,v.dtype)
    np.testing.assert_array_equal(first["vector"],env.step(dict(zero,reset=True))["vector"])
    # Reaching target is a kinematic check only, not a runtime controller.
    r=env.robot
    joint_indices=[i for i,n in enumerate(r.names) if n.startswith("l_") and any(x in n for x in ("shoulder","elbow"))]
    jac=np.zeros((3,r.model.nv))
    before=float(np.linalg.norm(env.goal-r.data.site_xpos[env.palm]))
    for _ in range(12):
        mujoco.mj_jacSite(r.model,r.data,jac,None,env.palm)
        error=env.goal-r.data.site_xpos[env.palm]
        arm_jac=jac[:,r.vadr[joint_indices]]
        # Damped inverse avoids singular straight-arm steps; diagnostic only.
        delta=arm_jac.T@np.linalg.solve(arm_jac@arm_jac.T+.001*np.eye(3),error)
        proposed=r.data.qpos[r.qadr[joint_indices]]+np.clip(delta,-.02,.02)
        lo=np.maximum(r.nominal-r.scale,r.model.actuator_ctrlrange[:,0])[joint_indices]
        hi=np.minimum(r.nominal+r.scale,r.model.actuator_ctrlrange[:,1])[joint_indices]
        r.data.qpos[r.qadr[joint_indices]]=np.clip(proposed,lo,hi)
        mujoco.mj_forward(r.model,r.data)
    kinematic_error=float(np.linalg.norm(env.goal-r.data.site_xpos[env.palm]))
    assert kinematic_error<env.task["success_distance"]
    env.step(dict(zero,reset=True))
    while not env.done:
        env.step(zero)
    env.video=False
    replay=embodied.replay.Replay(64,10000,directory=str(output/"replay"),seed=0)
    # Use Driver, preserving upstream's observation/action time alignment.
    driver=embodied.Driver([lambda:env],parallel=False)
    driver.on_step(replay.add)
    endings=[]
    driver.on_step(lambda tr,i:endings.append((bool(tr["is_last"]),bool(tr["is_terminal"]))))
    driver(lambda obs,carry:({"action":np.zeros((1,23),np.float32)},carry),steps=256)
    assert any(last and terminal for last,terminal in endings)
    assert len(replay)>0
    batch=next(replay.dataset(batch=2))
    assert batch["vector"].shape==(2,64,90) and batch["action"].shape==(2,64,23)
    assert np.all(batch["is_first"][:,0])
    replay.save(wait=True)
    timeout=ErgoCubEnv(evaluation=True,task={**(task or {}),"max_steps":1})
    timeout.step(dict(zero,reset=True))
    last=timeout.step(zero)
    assert last["is_last"] and not last["is_terminal"]
    assert timeout.step(zero)["is_first"]
    # Guard on deliberately lowered body; never step through a body impact.
    timeout.robot.data.qpos[2]=.1
    mujoco.mj_forward(timeout.robot.model,timeout.robot.data)
    assert timeout.fallen()
    timeout.close()
    a,b=ErgoCubEnv(seed=19,task=task),ErgoCubEnv(seed=19,task=task)
    for _ in range(3):
        np.testing.assert_array_equal(a.reset()["vector"],b.reset()["vector"])
    a.close();b.close()
    result=dict(observation_shape=[90],action_shape=[23],deterministic_eval_resets=True,
                seeded_training_resets=True,fall_and_timeout_flags=True,
                replay_items=len(replay),batch_shape=list(batch["vector"].shape),
                target_initial_distance=before,kinematic_target_error=kinematic_error,
                kinematic_note="Bounded left-arm joint pose, fixed base/legs, no dynamics or balance claim.")
    driver.close()
    return result


def readiness(config,output):
    import jax
    task=json.loads((output/"reach_task.json").read_text())
    summary=dict(environment=environment_checks(output,task),optimizer_updates_performed=0)
    (output/"environment_checks.json").write_text(json.dumps(summary["environment"],indent=2)+"\n")
    if config.jax.platform=="gpu":
        summary["gpu"]=gpu_check()
        (output/"gpu.json").write_text(json.dumps(summary["gpu"],indent=2,default=int)+"\n")
    start=time.monotonic()
    agent=make_agent(config)
    state=agent.save()
    before=fingerprint(state)
    integer_scalars=[int(np.asarray(v)) for v in jax.tree_util.tree_leaves(state)
                     if np.asarray(v).shape==() and np.issubdtype(np.asarray(v).dtype,np.integer)]
    assert integer_scalars and all(x==0 for x in integer_scalars),integer_scalars
    assert int(agent.updates)==0
    env=ErgoCubEnv(evaluation=True,task=task)
    obs=env.reset()
    actions,_=agent.policy({k:v[None] for k,v in obs.items()},agent.init_policy(1),mode="eval")
    assert actions["action"].shape==(1,23) and np.isfinite(actions["action"]).all()
    assert fingerprint(agent.save())==before
    checkpoint=embodied.Checkpoint(str(output/"initial.ckpt"))
    checkpoint.agent=agent
    checkpoint.step=embodied.Counter()
    checkpoint.save()
    checkpoint.load(keys=["agent"])
    assert fingerprint(agent.save())==before
    total=sum(np.asarray(v).nbytes for v in jax.tree_util.tree_leaves(state))
    summary.update(agent_initialization_seconds=time.monotonic()-start,
                   initialized_state_bytes=total,initialized_state_fingerprint=before,
                   zero_integer_state_counters=len(integer_scalars),policy_inference=True,
                   unchanged_parameters=True,checkpoint_roundtrip=True,
                   upstream_agent_updates=int(agent.updates),
                   device_memory=jax.devices(config.jax.platform)[0].memory_stats(),
                   memory_limitation="Initialization/inference measured; no training gradients executed. "
                   "Training activation/workspace peak remains unmeasured.")
    env.close()
    (output/"readiness.json").write_text(json.dumps(summary,indent=2,default=int)+"\n")
    print(json.dumps(summary,indent=2,default=int))


def run(args):
    verify_upstream()
    args.logdir=args.logdir.resolve()
    if args.prepared:
        if args.mode!="train" or not args.checkpoint:
            raise ValueError('--prepared requires train --checkpoint')
        from checkpointing import verify_prepared
        verify_prepared(args,configuration(args))
    else:
        args.logdir.mkdir(parents=True,exist_ok=False)
    if args.mode=="gpu":
        result=gpu_check()
        (args.logdir/"gpu.json").write_text(json.dumps(result,indent=2,default=int)+"\n")
        print(json.dumps(result,indent=2,default=int))
        return
    config=configuration(args)
    config.save(str(args.logdir/"config.yaml"))
    resume=args.mode=="train" and args.checkpoint is not None
    task=json.loads(((args.checkpoint.parent if args.mode=="eval" or resume else PROJECT/"configs")/
                     ("reach_task.json")).read_text())
    if resume and args.reference_mode and args.reference_mode!=task.get("reference_mode","joint_range"):
        raise ValueError("Resume must preserve the saved servo reference mode")
    task["reference_mode"]=args.reference_mode or task.get("reference_mode","joint_range")
    (args.logdir/"reach_task.json").write_text(json.dumps(task,indent=2)+"\n")
    for name in ("servo_settings.json",):
        source=PROJECT/"configs"/name
        if (args.mode=="eval" or resume) and (args.checkpoint.parent/name).is_file():
            if source.read_bytes()!=(args.checkpoint.parent/name).read_bytes():
                raise ValueError("Servo settings differ from checkpoint run")
        shutil.copyfile(source,args.logdir/name)
    if resume:
        for name in ("ergocub_env.py","ergocub_physics.py"):
            if (args.checkpoint.parent/name).read_bytes()!=(PROJECT/"scripts"/name).read_bytes():
                raise ValueError(f"Resume environment source differs from saved run: {name}")
        if not args.prepared:
            copy_resume_replay(args.checkpoint,args.logdir)
    for source in (Path(__file__),PROJECT/"scripts/ergocub_env.py",PROJECT/"scripts/ergocub_physics.py"):
        shutil.copyfile(source,args.logdir/source.name)
    if args.milestones:
        shutil.copyfile(PROJECT/"scripts/checkpointing.py",args.logdir/"checkpointing.py")
    if config.get('prior_prediction','off')!='off' or args.prepared:
        shutil.copyfile(PROJECT/"scripts/prior_prediction.py",args.logdir/"prior_prediction.py")
    (args.logdir/"manifest.json").write_text(json.dumps(dict(upstream_revision=REVISION,
        mode=args.mode,profile=args.profile,seed=args.seed,task=task,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__),PROJECT/"scripts/ergocub_env.py",PROJECT/"scripts/ergocub_physics.py")}),indent=2)+"\n")
    if args.mode=="check":
        readiness(config,args.logdir)
        return
    from embodied.agents.dreamerv3 import train as upstream
    args_run=embodied.Config(**config.run,logdir=config.logdir,batch_size=config.batch_size,batch_length=config.batch_length)
    envs=[]
    def make_env(index,evaluation=False):
        env=ErgoCubEnv(seed=config.seed+index,evaluation=evaluation,task=task,
                      output=args.logdir/("evaluation" if evaluation else "environment"),
                      video=args.video and evaluation)
        envs.append(env)
        return upstream.wrap_env(env,config)
    holder={}
    def agent_factory():
        holder["agent"]=make_agent(config)
        return holder["agent"]
    def logger_factory():
        holder["logger"]=upstream.make_logger(config)
        return holder["logger"]
    def replay_factory(name,evaluation=False):
        holder[name]=upstream.make_replay(config,name,is_eval=evaluation)
        # Commit a checkpoint only after its external replay chunks are on disk.
        holder[name].save=partial(holder[name].save,wait=True)
        return holder[name]
    try:
        if args.mode=="train":
            if args.checkpoint:
                args_run=args_run.update({"from_checkpoint":str(args.checkpoint.resolve())})
            from checkpointing import retain_milestones
            context=retain_milestones(embodied,holder,args.logdir,args.milestones) if args.milestones else nullcontext()
            with context:
                embodied.run.train_eval(agent_factory,lambda:replay_factory("train_replay"),
                    lambda:replay_factory("eval_replay",True),lambda i:make_env(i),
                    lambda i:make_env(i,True),logger_factory,args_run)
        elif args.mode=="eval":
            args_run=args_run.update({"from_checkpoint":str(args.checkpoint.resolve())})
            embodied.run.eval_only(agent_factory,lambda i:make_env(i,True),logger_factory,args_run)
    finally:
        if args.mode=="train" and "agent" in holder and "logger" in holder:
            final=embodied.Checkpoint(str(args.logdir/"final.ckpt"),parallel=False)
            final.agent=holder["agent"]
            final.step=holder["logger"].step
            for name in ("train_replay","eval_replay"):
                if name in holder:
                    setattr(final,name,holder[name])
            final.save()
        for env in envs:
            env.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("gpu","check","train","eval"))
    parser.add_argument("--logdir",type=Path,required=True)
    parser.add_argument("--platform",choices=("gpu","cpu"),default="gpu")
    parser.add_argument("--profile",choices=("benchmark","small"),default="benchmark")
    parser.add_argument("--steps",type=int,default=10000)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--video",action="store_true")
    parser.add_argument("--prior-prediction",choices=("off","ten_step"),default=None,
                        help="Experimental auxiliary model loss; default inherits checkpoint, otherwise off")
    parser.add_argument("--prepared",action="store_true",help="Launch a hash-verified prepared continuation directory once")
    parser.add_argument("--milestones",type=int,nargs="*",default=[],help="Retain full resume bundles at exact total driver steps (multiples of 10)")
    parser.add_argument("--reference-mode",choices=REFERENCE_MODES,
                        help="Servo setpoint bounds; saved with task. Old runs default to joint_range.")
    args=parser.parse_args()
    if args.mode=="eval" and not args.checkpoint:
        parser.error("eval requires --checkpoint")
    if args.steps<=0:
        parser.error("--steps must be positive")
    if args.milestones and (args.mode!="train" or any(s<=0 or s%10 or s>args.steps for s in args.milestones)):
        parser.error("Milestones require train and positive multiples of 10 up to --steps")
    run(args)
