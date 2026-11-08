"""Public ergoCub URDF, experimental position servos and ground-aligned reset.

Standalone MuJoCo only. No learner, task reward, mjlab, or worker process.
"""
from pathlib import Path
import json
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
ASSET_REVISION = "6c255e258cd184e91765b607222b6b6877bf7459"
ROBOT_DIR = PROJECT / "assets/ergocub"
PHYSICS_DT = 0.002
SUBSTEPS = 10
REFERENCE_MODES = ("joint_range", "unclipped")


class ErgoCubPhysics:
    def __init__(self, pose="clear", reference_mode="joint_range"):
        if reference_mode not in REFERENCE_MODES:
            raise ValueError(f"Unknown servo reference mode: {reference_mode}")
        self.reference_mode = reference_mode
        # Numerical settings retained solely for comparison with local evidence.
        # They are experiment settings, not public hardware ratings.
        settings = json.loads((PROJECT / "configs/servo_settings.json").read_text())
        groups = tuple((tuple(g["joints"]), g["effort_limit"]) for g in settings["groups"])
        constants = SimpleNamespace(
            STIFFNESS=settings["kp"], DAMPING=settings["kd"],
            ARMATURE=settings["armature"], _ACTUATOR_SPECS=groups,
            ERGOCUB_ACTION_SCALE={n: .25 * effort / settings["kp"] for names, effort in groups for n in names})
        self.constants = constants

        robot = ET.parse(ROBOT_DIR / "model.urdf").getroot()
        # Explicit stance-only collision model: retain the four original sole
        # boxes. Preserve SN001 inertials instead of Gazebo's surrogate inertias.
        for link in robot.findall("link"):
            for collision in link.findall("collision"):
                if collision.find("./geometry/box") is None:
                    link.remove(collision)
        retained = set(constants.ERGOCUB_ACTION_SCALE) | {
            "neck_pitch", "neck_roll", "neck_yaw", "camera_tilt"}
        for joint in robot.findall("joint"):
            if joint.get("type") != "fixed" and joint.get("name") not in retained:
                joint.set("type", "fixed")  # Wrists/fingers locked at zero.
        for mesh in robot.findall(".//mesh"):
            uri = mesh.get("filename")
            if not uri.startswith("package://ergoCub/meshes/"):
                raise ValueError(f"Unexpected mesh source: {uri}")
            mesh.set("filename", str(ROBOT_DIR / uri.removeprefix("package://ergoCub/")))
        extension = ET.SubElement(robot, "mujoco")
        ET.SubElement(extension, "compiler", fusestatic="false", discardvisual="false", inertiafromgeom="false")
        spec = mujoco.MjSpec.from_string(ET.tostring(robot, encoding="unicode"))
        spec.body("root_link").add_freejoint(name="floating_base_joint")
        for geom in spec.geoms:
            geom.group = 2 if geom.contype or geom.conaffinity else 1
        for name in ("l_sole", "r_sole", "l_hand_palm", "r_hand_palm"):
            spec.body(name).add_site(name=name, size=[.002, .002, .002])
        spec.option.timestep = PHYSICS_DT
        spec.option.gravity = [0, 0, -9.81]
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.option.iterations = 100
        spec.option.ls_iterations = 50
        spec.worldbody.add_geom(
            name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[3, 3, 0.1], pos=[0, 0, 0],
            contype=1, conaffinity=63, condim=3,
            friction=[0.8, 0.005, 0.0001], rgba=[0.32, 0.35, 0.38, 1])
        spec.worldbody.add_light(pos=[1, -1, 3], dir=[-0.3, 0.3, -1])
        self.names = []
        for joints, effort in constants._ACTUATOR_SPECS:
            for name in joints:
                joint = spec.joint(name)
                joint.armature = constants.ARMATURE
                actuator = spec.add_actuator(
                    name=name + "_servo", target=name,
                    trntype=mujoco.mjtTrn.mjTRN_JOINT,
                    ctrllimited=True, ctrlrange=joint.range,
                    forcelimited=True, forcerange=[-effort, effort])
                actuator.set_to_position(kp=constants.STIFFNESS, kv=constants.DAMPING)
                # A servo reference is not a physical joint-position constraint.
                # Keep the original semantics selectable; force caps always apply.
                actuator.inheritrange = 0.0
                actuator.ctrllimited = reference_mode == "joint_range"
                self.names.append(name)
        self.spec = spec
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        assert (m.nq, m.nv, m.nu) == (34, 33, 23)
        self.joint_ids = m.actuator_trnid[:, 0].copy()
        self.qadr = m.jnt_qposadr[self.joint_ids]
        self.vadr = m.jnt_dofadr[self.joint_ids]
        self.scale = np.array([constants.ERGOCUB_ACTION_SCALE[n] for n in self.names])
        self.root_id = m.body("root_link").id
        self.ground_id = m.geom("ground").id
        self.foot_geoms = {}
        for side in ("l", "r"):
            body = m.body(side + "_ankle_2").id
            self.foot_geoms[side] = np.array([
                i for i in range(m.ngeom) if m.body_weldid[m.geom_bodyid[i]] == m.body_weldid[body]
                and (m.geom_contype[i] or m.geom_conaffinity[i])], dtype=int)

        # Existing bent-leg candidate, all other scalar joints (including camera) zero.
        self.qreset = m.qpos0.copy()
        self.qreset[:7] = [0, 0, 0.82, 1, 0, 0, 0]
        if pose not in ("bent", "straight", "clear"):
            raise ValueError(pose)
        if pose in ("bent", "clear"):
            for side in ("l", "r"):
                for joint, angle in (("hip_pitch", 0.2), ("knee", -0.4), ("ankle_pitch", -0.2)):
                    self.qreset[m.jnt_qposadr[m.joint(side + "_" + joint).id]] = angle
        if pose == "clear":
            # Retain the previous reset angle for model-to-model comparison.
            self.qreset[m.jnt_qposadr[m.joint("l_shoulder_roll").id]] = 0.25
            # Solve base roll using this model's actual collision surfaces.
            lo, hi = -0.003, 0.003
            for _ in range(40):
                angle = (lo + hi) / 2
                self.qreset[3:7] = [np.cos(angle / 2), np.sin(angle / 2), 0, 0]
                self.data.qpos[:] = self.qreset
                mujoco.mj_forward(m, self.data)
                if self.foot_bottom("l") > self.foot_bottom("r"):
                    hi = angle
                else:
                    lo = angle
        self.data.qpos[:] = self.qreset
        mujoco.mj_forward(m, self.data)
        # Align the lowest collidable foot surface, not an approximate sole site.
        lowest = min(self.foot_bottom(side) for side in ("l", "r"))
        self.qreset[2] -= lowest
        if pose == "clear":
            self.qreset[2] -= 1e-6  # 1 um contact preload avoids roundoff contact loss.
        self.nominal = self.qreset[self.qadr].copy()
        self.reset()
        if pose == "clear":
            assert abs(self.foot_bottom("l") - self.foot_bottom("r")) < 1e-9

    def foot_bottom(self, side):
        bottoms = []
        for i in self.foot_geoms[side]:
            rotation = self.data.geom_xmat[i].reshape(3, 3)
            if self.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX:
                extent_z = np.abs(rotation[2]) @ self.model.geom_size[i]
                bottoms.append(float(self.data.geom_xpos[i, 2] - extent_z))
            elif self.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = self.model.geom_dataid[i]
                start, count = self.model.mesh_vertadr[mesh], self.model.mesh_vertnum[mesh]
                vertices = self.model.mesh_vert[start:start + count]
                bottoms.append(float(self.data.geom_xpos[i, 2] + np.min(vertices @ rotation[2])))
            else:
                raise ValueError("Unsupported foot collision shape")
        return min(bottoms)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.qreset
        self.data.ctrl[:] = self.nominal
        mujoco.mj_forward(self.model, self.data)

    def set_targets(self, targets):
        targets = np.asarray(targets)
        if targets.shape != (23,) or not np.isfinite(targets).all():
            raise ValueError("Expected 23 finite joint targets in radians")
        if self.reference_mode == "joint_range":
            targets = np.clip(targets, *self.model.actuator_ctrlrange.T)
        self.data.ctrl[:] = targets

    def step(self, targets=None):
        self.set_targets(self.nominal if targets is None else targets)
        mujoco.mj_step(self.model, self.data, nstep=SUBSTEPS)
        mujoco.mj_forward(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError("Nonfinite physics state")

    def contacts(self):
        result = {"l_force": 0.0, "r_force": 0.0, "nonfoot_ground": 0,
                  "ground_contacts": 0, "max_penetration": 0.0}
        force = np.zeros(6)
        for i, contact in enumerate(self.data.contact):
            if self.ground_id not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == self.ground_id else contact.geom1
            result["ground_contacts"] += 1
            result["max_penetration"] = max(result["max_penetration"], -float(contact.dist))
            mujoco.mj_contactForce(self.model, self.data, i, force)
            side = next((s for s, ids in self.foot_geoms.items() if other in ids), None)
            if side:
                result[side + "_force"] += float(force[0])
            else:
                result["nonfoot_ground"] += 1
        return result

    def snapshot(self):
        return dict(time=float(self.data.time), root=self.data.qpos[:3].tolist(),
                    upright=float(self.data.xmat[self.root_id].reshape(3, 3)[2, 2]),
                    joint_error_max=float(np.max(np.abs(self.data.qpos[self.qadr] - self.data.ctrl))),
                    root_speed=float(np.linalg.norm(self.data.qvel[:3])),
                    effort_fraction_max=float(np.max(np.abs(self.data.actuator_force) /
                                                       self.model.actuator_forcerange[:, 1])),
                    **self.contacts())
