"""Small embodied environment for the public ergoCub stance/reaching task."""
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
from ergocub_physics import PROJECT, PHYSICS_DT, SUBSTEPS, ErgoCubPhysics
sys.path.insert(0, str(PROJECT / "vendor/humanoid-bench/dreamerv3"))
import embodied
import mujoco
import numpy as np


class ErgoCubEnv(embodied.Env):
    def __init__(self, seed=0, evaluation=False, output=None, video=False, task=None,
                 reference_mode=None):
        self.task = json.loads((PROJECT / "configs/reach_task.json").read_text())
        if task:
            self.task.update(task)
        mode = reference_mode if reference_mode is not None else self.task.get("reference_mode", "joint_range")
        self.robot = ErgoCubPhysics(reference_mode=mode)
        self.rng = np.random.default_rng(seed)
        self.seed, self.evaluation = seed, evaluation
        self.output = Path(output) if output else None
        if self.output:
            self.output.mkdir(parents=True, exist_ok=True)
        self.video = video
        self.renderer = self.writer = None
        self.done, self.episode = True, -1
        self.previous = np.zeros(23, np.float32)
        self.goal = np.zeros(3)
        self.palm = self.robot.model.site("l_hand_palm").id
        # Conservative oriented bounding boxes around non-foot visual meshes.
        # This gives early impact termination without inventing body contacts.
        m = self.robot.model
        feet = {m.body_weldid[m.body(s + "_ankle_2").id] for s in ("l", "r")}
        self.guard_ids, self.guard_centers, self.guard_extents = [], [], []
        for i in range(m.ngeom):
            if m.geom_group[i] != 1 or m.body_weldid[m.geom_bodyid[i]] in feet:
                continue
            mesh = m.geom_dataid[i]
            if mesh < 0:
                continue
            start, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
            vertices = m.mesh_vert[start:start+count]
            lo, hi = vertices.min(0), vertices.max(0)
            self.guard_ids.append(i)
            self.guard_centers.append((lo+hi)/2)
            self.guard_extents.append((hi-lo)/2)
        self.guard_ids = np.array(self.guard_ids)
        self.guard_centers, self.guard_extents = map(np.array,(self.guard_centers,self.guard_extents))

    @property
    def obs_space(self):
        metrics = ("balance", "reach", "hand_distance", "upright", "height", "joint_error",
                   "effort", "upright_seconds", "episode_success", "clearance", "action_clipped")
        return dict(vector=embodied.Space(np.float32,(90,)), reward=embodied.Space(np.float32),
                    is_first=embodied.Space(bool),is_last=embodied.Space(bool),
                    is_terminal=embodied.Space(bool),success=embodied.Space(np.float32),
                    **{"log_"+k:embodied.Space(np.float32) for k in metrics})

    @property
    def act_space(self):
        return dict(action=embodied.Space(np.float32,(23,),-1,1),reset=embodied.Space(bool))

    def clearance(self):
        d = self.robot.data
        rotations = d.geom_xmat[self.guard_ids].reshape(-1,3,3)
        centers = d.geom_xpos[self.guard_ids] + np.einsum('nij,nj->ni',rotations,self.guard_centers)
        return float(np.min(centers[:,2]-np.einsum('ni,ni->n',np.abs(rotations[:,2]),self.guard_extents)))

    def fallen(self):
        r = self.robot
        return (r.data.qpos[2] < self.task["fall_height_fraction"]*r.qreset[2] or
                r.data.xmat[r.root_id].reshape(3,3)[2,2] < self.task["fall_upright_cosine"] or
                self.clearance() < self.task["nonfoot_clearance"])

    def reset(self):
        if self.writer:
            self.writer.close()
            self.writer = None
        r = self.robot
        r.reset()
        self.episode += 1
        self.steps, self.success_time, self.success_ever = 0, 0., False
        self.previous[:] = 0
        offset = np.array(self.task["target_offset"])
        if not self.evaluation:
            jitter = np.array(self.task["training_target_jitter"])
            offset += self.rng.uniform(-jitter,jitter)
        self.goal = r.data.site_xpos[self.palm] + r.data.xmat[r.root_id].reshape(3,3) @ offset
        self.done = False
        self.balance_sum, self.distance_sum, self.peak_effort = 0.,0.,0.
        if self.fallen():
            raise ValueError("Reset fails the fall/clearance guard")
        result = self.observe(0.,is_first=True)
        self.record_frame()
        return result

    def step(self, action):
        if bool(action["reset"]) or self.done:
            return self.reset()
        requested = np.asarray(action["action"],np.float32)
        if requested.shape != (23,) or not np.isfinite(requested).all():
            raise ValueError("Expected 23 finite normalized actions")
        applied = np.clip(requested,-1,1)
        r,m,d = self.robot,self.robot.model,self.robot.data
        r.set_targets(r.nominal+r.scale*applied)
        elapsed = 0.
        fall = False
        for _ in range(SUBSTEPS):
            mujoco.mj_step(m,d)
            mujoco.mj_forward(m,d)
            elapsed += PHYSICS_DT
            if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
                raise FloatingPointError("Invalid physics state; no replay transition produced")
            fall = self.fallen()
            if fall:
                break
        self.steps += 1
        dist = float(np.linalg.norm(d.site_xpos[self.palm]-self.goal))
        upright = float(d.xmat[r.root_id].reshape(3,3)[2,2])
        balance = float(np.clip((upright-.5)/.5,0,1)*np.exp(-((d.qpos[2]-r.qreset[2])/self.task["height_width"])**2))
        reach = float(np.exp(-(dist/self.task["reach_width"])**2))
        effort = float(np.mean((d.actuator_force/m.actuator_forcerange[:,1])**2))
        reward = (balance*(self.task["balance_weight"]+self.task["reach_weight"]*reach)
                  - self.task["effort_cost"]*effort
                  - self.task["action_change_cost"]*float(np.mean((applied-self.previous)**2))
                  - self.task["base_speed_cost"]*float(np.sum(d.qvel[:2]**2))) * elapsed/(PHYSICS_DT*SUBSTEPS)
        if fall:
            reward -= self.task["fall_penalty"]
        qualified = dist < self.task["success_distance"] and upright > np.cos(np.deg2rad(20)) and d.qpos[2]>.9*r.qreset[2]
        self.success_time = self.success_time+elapsed if qualified else 0.
        success = self.success_time >= self.task["success_hold_seconds"] and not self.success_ever
        self.success_ever = self.success_ever or success
        self.done = fall or self.steps >= self.task["max_steps"]
        self.previous = applied.copy()
        self.balance_sum += balance
        self.distance_sum += dist
        self.peak_effort = max(self.peak_effort,float(np.max(np.abs(d.actuator_force)/m.actuator_forcerange[:,1])))
        result = self.observe(reward,is_last=self.done,is_terminal=fall,success=success,
                              action_clipped=float(np.mean(requested!=applied)))
        self.record_frame()
        if self.done:
            if self.writer:
                self.writer.close()
                self.writer = None
            if self.output:
                row=dict(episode=self.episode,seed=self.seed,evaluation=self.evaluation,
                         duration=float(d.time),ending="fall" if fall else "timeout",
                         success=bool(self.success_ever),mean_balance=self.balance_sum/self.steps,
                         mean_hand_distance=self.distance_sum/self.steps,final_hand_distance=dist,
                         peak_effort_fraction=self.peak_effort,joint_error=result["log_joint_error"].item())
                with (self.output/"episodes.jsonl").open("a") as f:
                    f.write(json.dumps(row)+"\n")
        return result

    def observe(self,reward,is_first=False,is_last=False,is_terminal=False,success=False,action_clipped=0.):
        r,d,m = self.robot,self.robot.data,self.robot.model
        rotation = d.xmat[r.root_id].reshape(3,3)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,r.root_id,velocity,1)
        error = rotation.T @ (self.goal-d.site_xpos[self.palm])
        vector = np.r_[d.qpos[7:]-r.qreset[7:],d.qvel[6:],rotation.T@[0,0,-1],
                       velocity[3:],velocity[:3],d.qpos[2],error,self.previous].astype(np.float32)
        upright = float(rotation[2,2])
        distance = float(np.linalg.norm(error))
        balance = float(np.clip((upright-.5)/.5,0,1)*np.exp(-((d.qpos[2]-r.qreset[2])/self.task["height_width"])**2))
        values=dict(vector=vector,reward=reward,is_first=is_first,is_last=is_last,is_terminal=is_terminal,
                    success=success,log_balance=balance,log_reach=np.exp(-(distance/self.task["reach_width"])**2),
                    log_hand_distance=distance,log_upright=upright,log_height=d.qpos[2],
                    log_joint_error=np.max(np.abs(d.qpos[r.qadr]-d.ctrl)),
                    log_effort=np.max(np.abs(d.actuator_force)/m.actuator_forcerange[:,1]),
                    log_upright_seconds=d.time,log_episode_success=self.success_ever,
                    log_clearance=self.clearance(),log_action_clipped=action_clipped)
        return {k:np.asarray(values[k],dtype=s.dtype) for k,s in self.obs_space.items()}

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.robot.model,height=256,width=256)
        camera=mujoco.MjvCamera()
        camera.lookat[:]=[self.robot.data.qpos[0],self.robot.data.qpos[1],.65]
        camera.distance,camera.azimuth,camera.elevation=2.,135,-15
        options=mujoco.MjvOption()
        options.geomgroup[2]=0
        self.renderer.update_scene(self.robot.data,camera=camera,scene_option=options)
        scene=self.renderer.scene
        if scene.ngeom<scene.maxgeom:
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom],mujoco.mjtGeom.mjGEOM_SPHERE,
                               [.015]*3,self.goal,np.eye(3).ravel(),[1,.2,.1,1])
            scene.ngeom+=1
        return self.renderer.render().copy()

    def record_frame(self):
        if self.video and self.output:
            import imageio.v2 as imageio
            if self.writer is None:
                self.writer=imageio.get_writer(self.output/f"episode_{self.episode:04}.mp4",fps=50)
            self.writer.append_data(self.render())

    def close(self):
        if self.writer:
            self.writer.close()
            self.writer=None
        if self.renderer:
            self.renderer.close()
            self.renderer=None
