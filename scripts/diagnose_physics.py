"""Measure nominal servo sag and contact/balance; no baseline modifications.

Counterfactuals change one setting each: gravity, physics step, or ideal joint
constraints. The latter are diagnostic support, never a proposed controller.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import mujoco
import numpy as np

from ergocub_physics import ASSET_REVISION, PROJECT, ErgoCubPhysics


def hull(points):
    points = sorted(set(map(tuple, np.asarray(points).reshape(-1, 2))))
    if len(points) < 3:
        return np.array(points).reshape(-1, 2)
    def cross(a, b, c):
        return np.cross(np.subtract(b, a), np.subtract(c, a))
    def half(seq):
        out = []
        for p in seq:
            while len(out) > 1 and cross(out[-2], out[-1], p) <= 0:
                out.pop()
            out.append(p)
        return out
    return np.array(half(points)[:-1] + half(points[::-1])[:-1])


def margin(point, polygon):
    if not len(polygon):
        return None
    if len(polygon) < 3:
        a, b = polygon[0], polygon[-1]
        edge = b - a
        t = np.clip((point - a) @ edge / max(edge @ edge, 1e-30), 0, 1)
        return -float(np.linalg.norm(point - a - t * edge))
    edge = np.roll(polygon, -1, axis=0) - polygon
    return float(np.min(np.cross(edge, point - polygon) / np.linalg.norm(edge, axis=1)))


def contact_metrics(r):
    m, d = r.model, r.data
    points, bodies, world_forces = [], [], []
    force = np.zeros(6)
    jac = np.zeros((3, m.nv))
    metrics = dict(normal_force=[0., 0.], tangential_force=[0., 0.],
                   max_friction_utilization=0., max_contact_slip_speed=0.,
                   effective_friction=[], contact_count=0)
    for i, c in enumerate(d.contact):
        if r.ground_id not in (c.geom1, c.geom2):
            continue
        other = c.geom2 if c.geom1 == r.ground_id else c.geom1
        side = next((s for s in ("l", "r") if other in r.foot_geoms[s]), None)
        if side is None:
            continue
        mujoco.mj_contactForce(m, d, i, force)
        if force[0] < 1e-3:
            continue
        body = int(m.geom_bodyid[other])
        mujoco.mj_jac(m, d, jac, None, c.pos, body)
        slip = np.linalg.norm((jac @ d.qvel)[:2])
        tangent = float(np.linalg.norm(force[1:3]))
        metrics["normal_force"][side == "r"] += float(force[0])
        metrics["tangential_force"][side == "r"] += tangent
        metrics["max_contact_slip_speed"] = max(metrics["max_contact_slip_speed"], float(slip))
        metrics["max_friction_utilization"] = max(
            metrics["max_friction_utilization"], tangent / (c.friction[0] * force[0]))
        metrics["effective_friction"].append(float(c.friction[0]))
        points.append(c.pos.copy())
        bodies.append(body)
        f = c.frame.reshape(3, 3).T @ force[:3]
        world_forces.append(f if c.geom1 == r.ground_id else -f)
    polygon = hull(np.array(points)[:, :2] if points else [])
    metrics["contact_count"] = len(points)
    metrics["support_hull"] = polygon.tolist()
    metrics["com_support_margin"] = margin(d.subtree_com[r.root_id, :2], polygon)
    if points:
        weights = np.array(world_forces)[:, 2]
        metrics["cop"] = (weights @ np.array(points)[:, :2] / weights.sum()).tolist()
    else:
        metrics["cop"] = None
    return metrics, points, bodies


def sole_pad_envelope(r):
    """Projected pad geometry, distinct from the currently loaded contacts."""
    points = []
    for ids in r.foot_geoms.values():
        for i in ids:
            if r.model.geom_dataid[i] >= 0:  # Exclude mesh-fitted collision boxes.
                continue
            for x, y in itertools.product((-1, 1), repeat=2):
                local = r.model.geom_size[i] * [x, y, -1]
                points.append(r.data.geom_xpos[i] + r.data.geom_xmat[i].reshape(3, 3) @ local)
    polygon = hull(np.asarray(points)[:, :2])
    return dict(polygon=polygon.tolist(), bottom_corner_z=np.asarray(points)[:, 2].tolist(),
                com_margin=margin(r.data.subtree_com[r.root_id, :2], polygon))


def static_support(r, points, bodies):
    """Existence witness: nonnegative vertical tripod forces balancing gravity.

    Enumerates contact triples, not a dynamics controller or optimal load solver.
    Reports passive-joint torque residuals and position-target feasibility too.
    """
    m, d = r.model, r.data
    rest = mujoco.MjData(m)
    rest.qpos[:] = d.qpos
    mujoco.mj_forward(m, rest)
    com = rest.subtree_com[r.root_id]
    weight = m.body_subtreemass[r.root_id] * -m.opt.gravity[2]
    best = None
    jac = np.zeros((3, m.nv))
    for ids in itertools.combinations(range(len(points)), 3):
        xy = np.array([points[i][:2] for i in ids])
        matrix = np.vstack((xy.T, np.ones(3)))
        if abs(np.linalg.det(matrix)) < 1e-10:
            continue
        weights = np.linalg.solve(matrix, [*com[:2], 1])
        if weights.min() < -1e-9:
            continue
        external = np.zeros(m.nv)
        for i, w in zip(ids, weights):
            mujoco.mj_jac(m, rest, jac, None, np.asarray(points[i]), bodies[i])
            external += jac[2] * w * weight
        needed = rest.qfrc_bias - external
        torque = needed[r.vadr]
        target = rest.qpos[r.qadr] + torque / r.constants.STIFFNESS
        fraction = float(np.max(np.abs(torque) / m.actuator_forcerange[:, 1]))
        violation = np.maximum(m.actuator_ctrlrange[:, 0] - target, 0) + np.maximum(
            target - m.actuator_ctrlrange[:, 1], 0)
        # Prefer a witness achievable with position limits, then low effort.
        score = (float(violation.max()), fraction)
        if best is None or score < best[0]:
            passive = np.setdiff1d(np.arange(6, m.nv), r.vadr)
            best = score, dict(
                torque=torque.tolist(), required_position_target=target.tolist(),
                max_effort_fraction=fraction, max_target_limit_violation=float(violation.max()),
                floating_base_residual=float(np.max(np.abs(needed[:6]))),
                passive_joint_torque_residual=needed[passive].tolist(),
                supporting_points=[np.asarray(points[i]).tolist() for i in ids],
                normal_forces=(weights * weight).tolist())
    return None if best is None else best[1]


def run_case(case, output):
    r = ErgoCubPhysics()
    if case == "gravity_off":
        r.model.opt.gravity[:] = 0
        r.reset()
    if case == "half_timestep":
        r.model.opt.timestep = 0.001
        r.reset()
    if case == "ideal_joint_constraints":
        for name, q in zip(r.names, r.nominal):
            r.spec.add_equality(name="hold_" + name, type=mujoco.mjtEq.mjEQ_JOINT,
                                name1=name, data=[q] + [0] * 10,
                                solref=[0.004, 1.0])
        r.model = r.spec.compile()
        r.data = mujoco.MjData(r.model)
        r.reset()
    m, d = r.model, r.data
    dt = m.opt.timestep
    initial_height = d.qpos[2]
    sole_ids = [m.site(s + "_sole").id for s in ("l", "r")]
    sole0 = d.site_xpos[sole_ids].copy()
    data = {k: [] for k in ("time", "q", "dq", "target", "torque", "requested_torque",
                            "com", "root", "upright", "sole_xy_displacement",
                            "support_margin", "normal_force", "slip_speed", "friction_utilization")}
    contact_records = []
    static = {}
    initial = None
    for step in range(round(2.0 / dt) + 1):
        contacts, points, bodies = contact_metrics(r)
        if initial is None:
            initial = dict(**r.snapshot(), com=d.subtree_com[r.root_id].tolist(), **contacts,
                           sole_pad_envelope=sole_pad_envelope(r))
        if case == "baseline" and step in (0, round(.02 / dt), round(.1 / dt)):
            static[str(round(d.time, 3))] = static_support(r, points, bodies)
        q = d.qpos[r.qadr].copy()
        dq = d.qvel[r.vadr].copy()
        requested = r.constants.STIFFNESS * (d.ctrl - q) - r.constants.DAMPING * dq
        values = dict(time=d.time, q=q, dq=dq, target=d.ctrl.copy(),
                      torque=d.actuator_force.copy(), requested_torque=requested,
                      com=d.subtree_com[r.root_id].copy(), root=d.qpos[:3].copy(),
                      upright=d.xmat[r.root_id].reshape(3, 3)[2, 2],
                      sole_xy_displacement=d.site_xpos[sole_ids, :2] - sole0[:, :2],
                      support_margin=np.nan if contacts["com_support_margin"] is None
                      else contacts["com_support_margin"], normal_force=contacts["normal_force"],
                      slip_speed=contacts["max_contact_slip_speed"],
                      friction_utilization=contacts["max_friction_utilization"])
        for k, v in values.items():
            data[k].append(v)
        if step % max(1, round(.01 / dt)) == 0:
            contact_records.append(dict(time=float(d.time), **contacts))
        if d.xmat[r.root_id].reshape(3, 3)[2, 2] < .5 or d.qpos[2] < .65 * initial_height:
            break  # Stop before unsupported body/ground impact in this stance model.
        if step < round(2.0 / dt):
            mujoco.mj_step(m, d)
            mujoco.mj_forward(m, d)
            assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
    data = {k: np.asarray(v) for k, v in data.items()}
    np.savez_compressed(output / (case + ".npz"), **data)
    (output / (case + "_contacts.jsonl")).write_text(
        "".join(json.dumps(x) + "\n" for x in contact_records))
    def first(mask):
        ids = np.flatnonzero(mask)
        return None if not len(ids) else float(data["time"][ids[0]])
    errors = np.abs(data["q"] - data["target"])
    saturation = np.abs(data["torque"]) >= .999 * m.actuator_forcerange[:, 1]
    early = data["time"] <= .5 + 1e-9
    rows = []
    for t in (0, .01, .02, .04, .1, .2, .5, .7, 1.0, 2.0):
        i = round(t / dt)
        if i >= len(data["time"]):
            continue
        largest = np.argsort(errors[i])[::-1][:5]
        rows.append(dict(time=t, root_height=float(data["root"][i, 2]),
                         com=data["com"][i].tolist(), support_margin=(
                             None if np.isnan(data["support_margin"][i]) else float(data["support_margin"][i])),
                         normal_force=data["normal_force"][i].tolist(),
                         slip_speed=float(data["slip_speed"][i]),
                         upright=float(data["upright"][i]),
                         top_errors=[dict(joint=r.names[j], error=float(data["q"][i,j]-data["target"][i,j]),
                                          torque=float(data["torque"][i,j]),
                                          requested_torque=float(data["requested_torque"][i,j]),
                                          effort_limit=float(m.actuator_forcerange[j,1])) for j in largest]))
    return dict(initial=initial, static_support=static, sample_rows=rows,
                joint_order=r.names, physics_dt=float(dt),
                first_error_001_by_joint={n:first(errors[:,i] > .01) for i,n in enumerate(r.names)},
                first_saturation=first(saturation.any(axis=1)),
                first_positive_support_margin=first(data["support_margin"] > 0),
                max_requested_actual_torque_difference=float(np.max(np.abs(
                    data["requested_torque"] - data["torque"]))),
                max_tracking_error=float(errors.max()),
                first_tilt_1deg=first(data["upright"] < np.cos(np.deg2rad(1))),
                first_com_outside_after_002=first((data["time"] >= .02) & (data["support_margin"] < 0)),
                first_fall_threshold=first((data["upright"] < .5) |
                                          (data["root"][:,2] < .65 * initial_height)),
                first_half_second_max_slip_speed=float(data["slip_speed"][early].max()),
                first_half_second_max_sole_xy_displacement=float(np.linalg.norm(
                    data["sole_xy_displacement"][early],axis=2).max()),
                first_half_second_max_friction_utilization=float(data["friction_utilization"][early].max()),
                max_effort_fraction_before_05=float(np.max(np.abs(data["torque"][early]) /
                                                                          m.actuator_forcerange[:,1])),
                warnings={mujoco.mjtWarning(i).name:int(w.number) for i,w in enumerate(d.warning) if w.number})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = {name:run_case(name, args.output) for name in
             ("baseline", "gravity_off", "half_timestep", "ideal_joint_constraints")}
    summary = dict(asset_revision=ASSET_REVISION, mujoco=mujoco.__version__, cases=cases,
                   model="ergoCubSN001, four original sole-pad collision boxes",
                   servo_settings=json.loads((PROJECT / "configs/servo_settings.json").read_text()),
                   diagnostic_only="No baseline changes. Ideal constraints replace finite joint compliance; "
                   "they can supply additional torque and are not the robot's servo controller.",
                   source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                  (Path(__file__), Path(__file__).with_name("ergocub_physics.py"))})
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name in summary["source_sha256"]:
        (args.output / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    (args.output / "servo_settings.json").write_bytes((PROJECT / "configs/servo_settings.json").read_bytes())
    print(json.dumps({n:{k:v for k,v in c.items() if k.startswith("first_") or k=="warnings"}
                      for n,c in cases.items()},indent=2))
