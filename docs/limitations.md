# Limitations

- The V1/V2 empirical record follows one training lineage; it is not independent training-seed replication. The later SN001 walking study has two training seeds, a replication check rather than a population estimate. SN001 standing/reaching has one.
- Survival is censored at 10 seconds and the simulator uses a simplified contact/task setup.
- Evaluation panels, checkpoint choices, and some mechanism states were selected retrospectively; reused panels are labeled as such.
- Qualified reaching is a finite 0.5-second latch and must not be read as sustained reaching, standing, or recovery.
- E1 tests linear accessibility, E2 tests selected prediction behavior, and E3 tests a finite-window intervention. None proves physical understanding, calibrated uncertainty, actor optimality, or durable rescue.
- E4 was numerically censored and is not a successful whole-body-control comparator.
- V1/V2 cohorts and control/task packages differ; no gains-only causal explanation is supported.
- No hardware validation, sim-to-real transfer, disturbance robustness, all-direction balance, PPO/TD-MPC2 superiority, or cross-robot generalization is claimed.
- The public release omits private checkpoints, raw trajectories, replay, full reports, and restricted or machine-specific material.

## SN001 experiments

- **Survival is not learned balance.** On the SN001 reference plant, zero action also stands for 10 s, so 10 s survival there is not evidence of learned balance. The learned skill is near-target hand positioning toward one fixed target 5 cm away, achieved mostly by whole-base positioning.
- **No walking was demonstrated.** Both SN001 walking seeds learned a forward lean over planted feet. The cause is not isolated among reward incentives, plant/interface feasibility, learner coverage and contact termination. No other learner (PPO, TD-MPC2) and no reward variant was trained, so nothing is claimed about DreamerV3 relative to other methods.
- **The diagnostic foot lift is not a step.** It is stationary, produced by a privileged hand-designed controller, and rare (2 of 6,000 attempts). It is not a forward step, gait or learned behaviour.
- **Privileged observations.** Observations include simulator ground truth: root velocity, height above the floor, and the target vector for reaching. Contact sensing is absent from the walking observation.
- **Seed 0's walking lineage is confounded.** It was interrupted and resumed twice, which confounds seed with protocol.
- **Plant parameters are unvalidated.** Effort caps, contact parameters and servo gains are experimental simulation settings, not hardware-validated values.

## Future work (proposed, no results)

Hierarchical control is a separate proposed direction: a PPO low-level locomotion policy with a DreamerV3 high-level controller, motion retargeting for references, and a later TD-MPC2 comparison. It is not part of the closed walking study.
