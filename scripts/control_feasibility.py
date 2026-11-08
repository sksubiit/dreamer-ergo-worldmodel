"""Four declared, bounded open-loop references; no tuning or hidden controller."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from ergocub_physics import ErgoCubPhysics
from diagnose_physics import contact_metrics


def gravity_reference(r, straight=False):
    m = r.model
    d = mujoco.MjData(m)
    d.qpos[:] = r.qreset
    if straight:
        for i, name in enumerate(r.names):
            if any(key in name for key in ("hip_pitch", "knee", "ankle_pitch")):
                d.qpos[r.qadr[i]] = 0
    mujoco.mj_forward(m, d)
    points, bodies = [], []
    for ids in r.foot_geoms.values():
        for i in ids:
            for x in (-1, 1):
                for y in (-1, 1):
                    points.append(d.geom_xpos[i] + d.geom_xmat[i].reshape(3, 3) @
                                  (m.geom_size[i] * [x, y, -1]))
                    bodies.append(int(m.geom_bodyid[i]))
    points = np.array(points)
    matrix = np.vstack((np.ones(len(points)), points[:, :2].T))
    weight = m.body_subtreemass[r.root_id] * 9.81
    rhs = weight * np.r_[1, d.subtree_com[r.root_id, :2]]
    normal = matrix.T @ np.linalg.solve(matrix @ matrix.T, rhs)
    assert normal.min() >= 0
    external, jac = np.zeros(m.nv), np.zeros((3, m.nv))
    for point, body, force in zip(points, bodies, normal):
        mujoco.mj_jac(m, d, jac, None, point, body)
        external += jac[2] * force
    residual = d.qfrc_bias - external
    desired = d.qpos[r.qadr] + residual[r.vadr] / r.constants.STIFFNESS
    return desired, dict(required_torque=residual[r.vadr].tolist(),
                         floating_base_residual=residual[:6].tolist(),
                         normal_forces=normal.tolist(),
                         assumption="Minimum-norm nonnegative vertical load sharing over sole corners; "
                         "static illustration, not a unique force solution or feedback controller.")


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    r = ErgoCubPhysics()
    ff, ff_info = gravity_reference(r)
    straight_ff, straight_info = gravity_reference(r, straight=True)
    straight = r.nominal.copy()
    for i, name in enumerate(r.names):
        if any(key in name for key in ("hip_pitch", "knee", "ankle_pitch")):
            straight[i] = 0
    cases = {}
    for name, desired in (("nominal", r.nominal), ("gravity_reference", ff),
                          ("straight_reference", straight), ("straight_gravity_reference", straight_ff)):
        # Identical reset, same gains/effort limits/ranges; only reference changes.
        action = np.clip((desired - r.nominal) / r.scale, -1, 1)
        target = np.clip(r.nominal + r.scale * action, *r.model.actuator_ctrlrange.T)
        r.reset()
        sole_ids = [r.model.site(s+"_sole").id for s in ("l", "r")]
        sole0 = r.data.site_xpos[sole_ids].copy()
        records = []
        for _ in range(500):
            r.step(target)
            row = r.snapshot()
            row.update(com=r.data.subtree_com[r.root_id].tolist(),
                       q=r.data.qpos[r.qadr].tolist(), torque=r.data.actuator_force.tolist(),
                       tracking_error=(r.data.qpos[r.qadr]-target).tolist(),
                       sole_xy_displacement=np.linalg.norm(r.data.site_xpos[sole_ids,:2]-sole0[:,:2],axis=1).tolist(),
                       contact=contact_metrics(r)[0])
            records.append(row)
            if row["upright"] < .5 or row["root"][2] < .65 * r.qreset[2]:
                break
        cases[name] = dict(duration=records[-1]["time"], survived_10s=len(records)==500,
                           action=action.tolist(), unclipped_reference=desired.tolist(),
                           applied_reference=target.tolist(), reference_clipped_joints=[
                               n for n,a,b in zip(r.names,desired,target) if abs(a-b)>1e-9],
                           peak_effort_fraction=max(x["effort_fraction_max"] for x in records),
                           peak_joint_error=max(x["joint_error_max"] for x in records),
                           com_support_exit=next((x["time"] for x in records if
                               x["contact"]["com_support_margin"] is None or x["contact"]["com_support_margin"]<0),None),
                           peak_contact_slip_speed=max(x["contact"]["max_contact_slip_speed"] for x in records),
                           peak_friction_utilization=max(x["contact"]["max_friction_utilization"] for x in records),
                           at_200ms=records[9],
                           final=records[-1])
        (output / (name + ".jsonl")).write_text("".join(json.dumps(x)+"\n" for x in records))
    summary = dict(cases=cases,joint_order=r.names,gravity_calculation=ff_info,
                   straight_gravity_calculation=straight_info,
                   interpretation="These fixed analytic references are diagnostic trials only, "
                   "not used in the task environment or learner. No search or physical parameter change.")
    (output / "summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps({n:{k:c[k] for k in ("duration","survived_10s","peak_effort_fraction","reference_clipped_joints")} for n,c in cases.items()},indent=2))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    run(p.parse_args().output)
