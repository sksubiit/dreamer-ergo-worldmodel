# SN001 forward walking with DreamerV3: two-seed negative result

**Status:** CLOSED. **No walking was demonstrated.** Two independent DreamerV3 training runs both learned a controlled forward lean over planted feet. Neither learned to step.

Values are in [`analysis/sn001_results.json`](../analysis/sn001_results.json). Claim-to-evidence mapping is in [`analysis/release_manifest.json`](../analysis/release_manifest.json).

## Setup

| Item | Setting |
|---|---|
| Plant | Motor V2 contract on the official SN001 geometry: 23 position servos with Kp 25.266 and Kd 2.011, armature 0.01, effort caps 60/110/17/20 Nm, passive neck and camera. Integration is `implicitfast`, 2 ms × 10 substeps per command. **This differs from the stiff `sn001_sim_v1` plant used for standing/reaching** |
| Reward | Pays forward centre-of-mass speed, saturating at 0.2 m/s, while upright and loaded. Small effort and slip penalties; −1 on a fall. **No survival term** |
| Termination | Height < 0.8 of reset; tilt > 30°; a joint more than 0.01 rad beyond its range; any contact deeper than 5 mm. Numerical faults are handled separately as invalid states (none occurred). Cap of 500 commands |
| Observation | 88-D: joint angles and velocities, projected gravity, root-local linear and angular velocity, root height above the floor, speed command, previous action. Simulator ground truth; **no contact sensing** |
| Learner | DreamerV3 (HumanoidBench `cb118903`), unmodified, trained from scratch. No imitation, reference motion or pretrained gait |
| Budget | 200,000 environment steps per run, about 12,500 updates each |

**The two runs.**

| | Training seed 0 | Training seed 2 |
|---|---|---|
| Lineage | 0→10k pilot, then a continuation **interrupted externally at 80.1k**, recovered from its verified checkpoint to 200k (two resume boundaries) | Uninterrupted, from scratch |
| Shared data | — | No weights or replay inherited from seed 0 |
| Skipped actor updates | 9 | 10 |
| Invalid states | 0 | 0 |

The configurations are identical except for the seed. Seed 0's resume boundaries confound seed with protocol in its learning curve.

## Evaluation

30 stochastic episodes per policy, all reported. Settings: evaluation sampling seed 0 for both, deterministic reset, CPU, 0 updates, MuJoCo contact forces at every 2 ms substep. The training seed is a property of the weights; the sampling seed is a property of the evaluation.

| | Seed 0 | Seed 2 |
|---|---:|---:|
| Episode length | 66.9 commands (SD 13.0), ≈ 1.34 s | 69.1 commands (SD 4.3), ≈ 1.38 s |
| Return | 40.8 (SD 8.1) | 41.3 (SD 2.0) |
| Forward COM displacement | 0.229 m | 0.278 m |
| Mean forward COM speed | 0.170 m/s | 0.204 m/s |
| Both feet in contact (share of substeps, settling and final command excluded) | 99.92% | 99.93% |
| Of which in weight transfer (smaller foot carrying 10–30% of load) | 32.4% | 3.0% |
| Longest single support | 2 ms | 10 ms |
| Maximum sole clearance | 1.02 mm | 0.96 mm |
| Episodes with a swing and touchdown | 0 / 30 | 0 / 30 |
| COM outside the support polygon before the end (median) | 0.44 s | 0.46 s |
| Endings | joint range 22, tilt 8 | tilt 27, joint range 3 |

For reference, zero action collapses in 24 commands (≈ 0.48 s).

The two seeds differ in how they lean, not in whether they step:
- **Seed 0** shifts load onto the right foot and falls forward-right.
- **Seed 2** keeps the load near 50/50 and falls straight forward.

**Registered predictions for seed 2**, written before it was trained and scored unchanged:

| Prediction | Result | Verdict |
|---|---|---|
| P1: no single support ≥ 100 ms | longest 10 ms | holds |
| P2, episode means: return < 60 and displacement < 0.32 m | 41.3 and 0.278 m | holds |
| P2, episode maxima: same thresholds | 42.4 and 0.304 m | holds |
| P3: ≥ 3/30 upright torso-roll endings | 0/30 | falsified |

## Why no stepping: evidence and hypotheses

These are offline analyses of the frozen policies; no training was involved.

**Q1, reward (measured).**
- The lean earns about 0.60 reward per command, immediately.
- Weight transfer pays the same as double support: 0.604 against 0.613 per command for seed 0.
- A controlled *stationary* foot lift, produced by a hand-designed feedback controller, earns 0.085–0.10 per command, about half of it from sway.
- No precursor of stepping is paid more than the lean, and forward COM motion earns return without stepping.
- No forward step was constructed, so the return of a step is unknown.

**Q2, learned models (measured).**
- Both world models predict reward well on their own lean (correlation 0.93–0.95).
- Every single-support state is out of their training distribution.
- Their termination predictions do not separate a lift that survives from one that falls (AUROC 0.66 and 0.61, against a registered 0.75).
- Whether this caused the lean, or resulted from it, cannot be determined offline.
- Imagined action rankings were not measured.

**Q3, the 5 mm contact-penetration termination (measured).**
- It is a valid physical event: 0 numerical faults in 1.52 M diagnostic substeps.
- 0 of 1,313 controlled controller touchdowns crossed 5 mm.
- It ended one fast toe-corner touchdown of a policy.
- It ended only 6 (seed 0) and 13 (seed 2) training episodes as touchdowns. It is not a major obstruction at the observed incidence.

**Diagnostic lift (measured, limited).** A controller with privileged true-state feedback lifted the left foot 13 mm for 280 ms and recovered. It found this in 2 of 6,000 attempts on the unchanged plant and interface. This shows that a small *stationary* lift is feasible. It is **not a forward step, a gait, or a learned behaviour**.

**Attribution.**

| Hypothesis | Assessment |
|---|---|
| Plant/interface feasibility | Narrow but non-empty for a stationary lift; gait feasibility unknown |
| Task incentives | Strongest explanatory support; no task variant was trained |
| DreamerV3 learning | Coverage gap confirmed, but plausibly a consequence of the lean; optimisation healthy; **no other learner (e.g. PPO, TD-MPC2) was trained on this task** |
| Simulation infrastructure | No defect found; a small threshold effect of low incidence |

No single cause is established. These results do **not** show that DreamerV3 cannot learn to walk this robot, that another algorithm would do better, or that the simulator prevents walking.

## Clips

Each clip is a fresh rollout from saved weights, identical to registered evaluation episode 2. Rendering is visual-mesh only; physics is unchanged.

| Clip | Episode | Selection |
|---|---|---|
| [`assets/demo/sn001-walking-seed0-200k-forward-lean.mp4`](../assets/demo/sn001-walking-seed0-200k-forward-lean.mp4) | seed 0: 67 commands, return 44.60, ended on a joint-range event | closest to the cohort median length (72) |
| [`assets/demo/sn001-walking-seed2-200k-forward-lean.mp4`](../assets/demo/sn001-walking-seed2-200k-forward-lean.mp4) | seed 2: 70 commands, return 41.45, ended at the 30° tilt threshold | closest to the cohort median length (70) |

Among the rendered episodes, the clip was chosen by the rule shown, not by outcome.

Viewing notes:
- **Half speed.** The clips hold one frame per 20 ms command and play at 25 fps, so they run at **half real-time speed**: 1.34 s and 1.40 s of motion become about 2.7 s and 2.8 s of video.
- **Tracking camera.** The camera follows the pelvis, so forward travel looks smaller than the measured COM displacement of these two episodes (0.212 m for seed 0 and 0.290 m for seed 2).
- **No edits.** The clips have no cuts: frames equal commands + 1.
- **Orange feet.** The orange foot pieces are visual clones generated by the reduced model, not a physical change.

## Limitations

- Two training seeds are a replication check, not a population estimate.
- Confidence intervals cover policy sampling from one initial state.
- Seed 0's interrupted lineage confounds seed with protocol.
- Simulation only; the contact and actuator parameters are experimental and not hardware-validated.
- No comparator learner and no reward variant were trained.

The study is closed. Hierarchical control (a PPO low-level policy with a DreamerV3 high-level controller, motion retargeting, and a later TD-MPC2 comparison) is a separate, proposed experiment with no results.
