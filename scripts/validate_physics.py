"""No-learning ergoCub checks; writes measured results and optional video."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

# CPU rendering is portable and does not depend on JAX/CUDA or a display server.
os.environ.setdefault("MUJOCO_GL", "osmesa")
import imageio.v2 as imageio
import mujoco
import numpy as np

from ergocub_physics import ASSET_REVISION, PHYSICS_DT, SUBSTEPS, PROJECT, ErgoCubPhysics


def run(output, render=True, pose="clear"):
    output.mkdir(parents=True, exist_ok=False)
    robot = ErgoCubPhysics(pose=pose)
    m, d = robot.model, robot.data
    records = []
    renderer = mujoco.Renderer(m, height=480, width=640) if render else None
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, 0.7]
    camera.distance, camera.azimuth, camera.elevation = 2.3, 135, -15
    options = mujoco.MjvOption()
    options.geomgroup[2] = 0  # Keep collision boxes out of the robot rendering.
    writer = imageio.get_writer(output / "nominal_hold.mp4", fps=25, codec="libx264") if render else None

    def frame():
        camera.lookat[:] = [d.qpos[0], d.qpos[1], max(0.3, d.qpos[2] - 0.1)]
        renderer.update_scene(d, camera=camera, scene_option=options)
        return renderer.render().copy()

    inventory = []
    for i, name in enumerate(robot.names):
        assert m.joint(int(robot.joint_ids[i])).name == name
        assert m.actuator_ctrllimited[i] and m.actuator_forcelimited[i]
        np.testing.assert_allclose(m.actuator_gainprm[i, 0], robot.constants.STIFFNESS)
        np.testing.assert_allclose(m.actuator_biasprm[i, 1:3],
                                   [-robot.constants.STIFFNESS, -robot.constants.DAMPING])
        np.testing.assert_allclose(m.dof_armature[robot.vadr[i]], robot.constants.ARMATURE)
        effort = next(e for names, e in robot.constants._ACTUATOR_SPECS if name in names)
        np.testing.assert_allclose(m.actuator_forcerange[i], [-effort, effort])
        np.testing.assert_allclose(m.actuator_ctrlrange[i], m.jnt_range[robot.joint_ids[i]])
        inventory.append(dict(index=i, joint=name, qpos_address=int(robot.qadr[i]),
                              dof_address=int(robot.vadr[i]),
                              ctrl_range=m.actuator_ctrlrange[i].tolist(),
                              effort_range=m.actuator_forcerange[i].tolist(),
                              armature=float(m.dof_armature[robot.vadr[i]]),
                              passive_damping=float(m.dof_damping[robot.vadr[i]]),
                              action_scale=float(robot.scale[i])))
    initial = robot.snapshot()
    initial["foot_bottom"] = {s: robot.foot_bottom(s) for s in ("l", "r")}
    initial["com"] = d.subtree_com[robot.root_id].tolist()
    initial["mass_kg"] = float(m.body_subtreemass[robot.root_id])
    initial["root_quaternion_wxyz"] = d.qpos[3:7].tolist()
    initial["self_contacts"] = [
        dict(body1=m.body(m.geom_bodyid[c.geom1]).name,
             body2=m.body(m.geom_bodyid[c.geom2]).name, distance=float(c.dist))
        for c in d.contact if robot.ground_id not in (c.geom1, c.geom2)]
    initial_geometry_passed = (not any(c["distance"] < -1e-8 for c in initial["self_contacts"])
                               and initial["l_force"] > 0 and initial["r_force"] > 0
                               and initial["max_penetration"] <= 1e-5)
    # Repeated resets must reproduce the complete physical state exactly.
    reset_qpos, reset_qvel = d.qpos.copy(), d.qvel.copy()
    robot.step()
    robot.reset()
    np.testing.assert_array_equal(d.qpos, reset_qpos)
    np.testing.assert_array_equal(d.qvel, reset_qvel)
    if render:
        imageio.imwrite(output / "initial.png", frame())
    fall_time = None
    try:
        # Free-base, ordinary gravity: nominal targets for a full ten seconds.
        for step in range(500):
            robot.step()
            record = robot.snapshot()
            records.append(record)
            assert record["effort_fraction_max"] <= 1 + 1e-8
            if fall_time is None and (record["upright"] < 0.5 or
                                      record["root"][2] < 0.65 * initial["root"][2] or
                                      record["nonfoot_ground"]):
                fall_time = record["time"]
            if render and step % 2 == 0:
                writer.append_data(frame())
            if fall_time is not None:
                break  # Stance-only collisions do not model body/ground impact.
        if render:
            imageio.imwrite(output / "final.png", frame())
    finally:
        if writer:
            writer.close()
        if renderer:
            renderer.close()
    hold_warnings = {mujoco.mjtWarning(i).name: int(w.number)
                     for i, w in enumerate(d.warning) if w.number}
    np.testing.assert_allclose(d.time, len(records) * PHYSICS_DT * SUBSTEPS, atol=1e-10)

    # Same physical scene and reset, compare each pulse with an unperturbed
    # trajectory before the nominal pose collapses. This separates its response
    # from gravitational sag without claiming successful posture control.
    robot.reset()
    for _ in range(10):
        robot.step()
    baseline = d.qpos[robot.qadr].copy()
    grounded_pulses = []
    for i, name in enumerate(robot.names):
        for sign in (-1, 1):
            robot.reset()
            targets = robot.nominal.copy()
            targets[i] += sign * 0.03
            for _ in range(10):
                robot.step(targets)
                assert np.all(np.abs(d.actuator_force) <= m.actuator_forcerange[:, 1] + 1e-8)
            response = float(d.qpos[robot.qadr[i]] - baseline[i])
            grounded_pulses.append(dict(joint=name, requested_delta=sign * 0.03,
                                       applied_delta=float(d.ctrl[i] - robot.nominal[i]),
                                       response_relative_to_nominal=response,
                                       direction_passed=bool(sign * response > 1e-5),
                                       final=robot.snapshot()))

    # Isolate transmission direction from falling: temporarily disable gravity
    # and collisions for these tests, then restore the unchanged physical model.
    gravity, masks = m.opt.gravity.copy(), (m.geom_contype.copy(), m.geom_conaffinity.copy())
    m.opt.gravity[:] = 0
    m.geom_contype[:] = 0
    m.geom_conaffinity[:] = 0
    pulses = []
    try:
        for i, name in enumerate(robot.names):
            for sign in (-1, 1):
                robot.reset()
                targets = robot.nominal.copy()
                targets[i] += sign * 0.03
                displacement = []
                for _ in range(25):
                    robot.step(targets)
                    displacement.append(float(d.qpos[robot.qadr[i]] - robot.nominal[i]))
                peak = max(sign * x for x in displacement)
                pulses.append(dict(joint=name, target_delta=sign * 0.03,
                                   final_delta=displacement[-1], peak_in_command_direction=peak,
                                   passed=bool(peak > 0.0001)))
        # A large request must clip to position bounds; force must stay bounded.
        for sign, bound in ((-1, 0), (1, 1)):
            robot.reset()
            robot.step(np.full(23, sign * 1000.0))
            np.testing.assert_allclose(d.ctrl, m.actuator_ctrlrange[:, bound])
            assert np.all(np.abs(d.actuator_force) <= m.actuator_forcerange[:, 1] + 1e-8)
        limit_check = True
    finally:
        m.opt.gravity[:] = gravity
        m.geom_contype[:], m.geom_conaffinity[:] = masks
        robot.reset()

    summary = dict(
        asset_revision=ASSET_REVISION, python=sys.version, mujoco=mujoco.__version__,
        numpy=np.__version__, pose=pose, nq=m.nq, nv=m.nv, nu=m.nu,
        model="ergoCubSN001, four original sole-pad collision boxes", initial_geometry_passed=initial_geometry_passed,
        asset_manifest=json.loads((PROJECT / "assets/ergocub/provenance.json").read_text()),
        servo_settings=json.loads((PROJECT / "configs/servo_settings.json").read_text()),
        render_backend=os.environ["MUJOCO_GL"] if render else None,
        servo_kp=robot.constants.STIFFNESS, servo_kd=robot.constants.DAMPING,
        physics_dt=PHYSICS_DT, substeps=SUBSTEPS, control_dt=PHYSICS_DT * SUBSTEPS,
        initial=initial, hold_final=records[-1], first_fall_time=fall_time,
        hold_passed=fall_time is None, warnings=hold_warnings,
        root_displacement=(np.array(records[-1]["root"]) - initial["root"]).tolist(),
        max_ground_penetration=max(x["max_penetration"] for x in records),
        max_hold_effort_fraction=max(x["effort_fraction_max"] for x in records),
        grounded_pulse_checks=grounded_pulses,
        grounded_pulse_direction_passed=all(x["direction_passed"] for x in grounded_pulses),
        grounded_pulse_conditions="gravity and contacts on; 0.03 rad, 0.2 s; compare identical nominal rollout",
        pulse_passed=all(x["passed"] for x in pulses), pulse_checks=pulses,
        pulse_conditions="gravity off; collisions off; floating base remains free; 0.03 rad, 0.5 s",
        command_and_force_limits_passed=limit_check, actuators=inventory,
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [Path(__file__), Path(__file__).with_name("ergocub_physics.py")]})
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "hold.jsonl").write_text("".join(json.dumps(x) + "\n" for x in records))
    # Keep the exact small validation sources beside the measurements.
    for name in ("ergocub_physics.py", "validate_physics.py"):
        (output / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    (output / "servo_settings.json").write_bytes((PROJECT / "configs/servo_settings.json").read_bytes())
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("actuators", "pulse_checks", "grounded_pulse_checks", "source_sha256", "asset_manifest")}, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--pose", choices=("clear", "bent", "straight"), default="clear")
    args = parser.parse_args()
    result = run(args.output, render=not args.no_render, pose=args.pose)
    sys.exit(0 if (result["initial_geometry_passed"] and result["hold_passed"] and result["pulse_passed"] and
                   result["grounded_pulse_direction_passed"] and
                   not result["warnings"]) else 2)
