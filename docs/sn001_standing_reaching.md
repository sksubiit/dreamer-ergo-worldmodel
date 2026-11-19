# SN001 reference plant: standing and left-hand reaching (110k)

**Status:** observed, one training seed. The policy reaches the target reliably, but **learned balance is not shown**: zero action also stands for the full 10 s on this plant.

Values are in [`analysis/sn001_results.json`](../analysis/sn001_results.json). Claim-to-evidence mapping is in [`analysis/release_manifest.json`](../analysis/release_manifest.json).

## What was trained

| Item | Setting |
|---|---|
| Plant | `sn001_sim_v1`: official-source ergoCubSN001 (27 joints, 57.1045 kg), floating base. Position servos with Kp 1000, actuator Kv 0 and passive damping 2. Official box contacts. Simulation-only experimental effort caps. 500 Hz physics, 50 Hz policy |
| Task | Hold a fixed left-palm target **50 mm** in front of the reset palm position, with no target jitter. Success means the palm stays within 30 mm for 0.5 s while upright and above a height limit. Episodes are capped at 10 s (500 commands) |
| Observation | 90-D: 27 joint angles and 27 joint velocities, projected gravity, root-local linear and angular velocity, root height above the floor, target-minus-palm vector, previous action. These are simulator ground-truth quantities, privileged relative to on-robot sensing |
| Action | 23 normalized joint-position references |
| Learner | DreamerV3 (HumanoidBench `cb118903`), unmodified, trained from scratch with training seed 0 |
| Budget | 110,000 environment (driver) steps, 6,871 agent updates. Skipped mixed-precision updates: 1 model, 7 actor, 0 critic. All saved learning state is finite |
| Lineage | A first launch stopped at driver step 79 on a reporting diagnostic. A fresh relaunch from scratch produced every result here; the stopped run was used for neither initialization nor evaluation |

**Evaluated, not trained.** The 75k milestone and the 110k final checkpoint were evaluated as frozen policies. Each used 20 independent stochastic-policy seeds (1000–1019), the nominal reset, CPU execution and 0 optimizer updates. Checkpoint hashes were unchanged before and after.

## Results

| Measure | 75k | 110k | 110k − 75k (paired 95% CI) |
|---|---:|---:|---:|
| Episodes reaching the 10 s cap | 20/20 | 20/20 | — |
| Qualified reaching (≥ 0.5 s within 30 mm) | 20/20 | 20/20 | — |
| Qualified at episode end | 20/20 | 20/20 | — |
| Mean maximum continuous dwell | 9.343 s | 9.445 s | +0.102 s [−0.243, +0.435] |
| Mean hand-target error | 14.74 mm | 12.24 mm | −2.50 mm [−3.47, −1.55] |
| Hand error in the first second | 22.63 mm | 25.03 mm | +2.40 mm [+1.15, +3.66] |
| Mean return | 2149.5 | 2167.2 | +17.7 [+7.5, +27.8] |
| Both feet loaded (share of samples) | 1.000 | 0.999 | — |

For 20/20 outcomes, the Wilson 95% interval is 83.9–100%. The paired intervals are marginal and not multiplicity-adjusted; they cover policy sampling only, for one training seed.

## Baselines: why this is not evidence of learned balance

| Controller | 10 s survival | Qualified reaching | Mean hand error | Return |
|---|---|---|---|---|
| **Zero action** (servos hold the reset posture; one deterministic episode) | **yes (timeout)** | no (max dwell 0 s) | 70.57 mm | 1390.3 |
| Trained policy, 75k (mean of 20) | 20/20 | 20/20 | 14.74 mm | 2149.5 |
| Trained policy, 110k (mean of 20) | 20/20 | 20/20 | 12.24 mm | 2167.2 |

What the comparison shows:
- **Survival does not separate trained from untrained control.** The stiff servos already hold the robot upright. In training, only 4 of 110,000 recorded steps were terminal, all within roughly the first 12.6k steps, so even the untrained policy usually reached the cap.
- **Reaching does separate them.** The trained policy brings the palm to the target and keeps it there. Zero action never qualifies.
- **The reach is mostly whole-body positioning.** A kinematic decomposition attributes only 5.50 mm (75k) and 3.96 mm (110k) of mean palm displacement to joint motion relative to the base. The target is close (50 mm) and fixed, so this is not evidence of general articulated arm reaching or goal-conditioned feedback.
- **Small posture differences within the first 5 s.** Compared with a saved 5 s nominal-hold trial, root-pitch RMS was 2.435° for the hold against 1.842° / 1.982° for the policies. Peak COM drift was 33.67 mm against 22.19 / 25.30 mm. This is a measured posture difference within that window only. It does not separate learned from mechanical (servo) stabilization.

**Answer to "balancing, reaching, or survival?"** The policy shows nominal standing with successful and sustained near-target hand positioning for the full 10 s. Survival to timeout is shared with zero action. Active learned balancing is **not demonstrated**.

## 75k versus 110k

The extra 35k steps changed neither survival nor reaching incidence, since both were at the ceiling. They improved whole-episode hand error and return, and worsened first-second error. Neither checkpoint dominates.

The 110k final checkpoint is used as the demonstration model. The 75k milestone is equally defensible on survival and reaching.

## Checkpoints

| Checkpoint | Driver steps | Role |
|---|---:|---|
| Intermediate milestone | 75,000 | the only saved intermediate checkpoint; evaluated above |
| Final | 110,000 | end of training; evaluated above; used for the clip |

No SN001 standing/reaching run beyond 110k exists. The 200k SN001 runs are the [two walking seeds](sn001_walking_negative_result.md), which use a different plant and task.

**No balance milestone can be dated.** Training episodes reach the 10 s cap from the start. Training score rises until about 70–80k steps and then plateaus.

## Demonstration clip

[`assets/demo/sn001-110k-seed-1000-stand-reach.mp4`](../assets/demo/sn001-110k-seed-1000-stand-reach.mp4): SN001 reference 110k, evaluation seed 1000. The episode times out at 10 s with qualified reaching (max dwell 9.70 s).

How the clip was made:
- It is a **re-render of the recorded evaluation states** with `mj_forward` only. There is no new rollout and no policy call.
- A visualization-only model clone adds the four public foot meshes and the URDF colours as massless, contact-free visuals. Its physical arrays were verified equal to the evaluated model's.
- Red marker: fixed target. Green marker: palm frame.
- Playback is real time: 101 frames at 10 fps over the 10 s episode. The fixed camera framing is a visualization choice only. There are no cuts or edits.

Selection: seed 1000 is the first seed of the cohort. It also happens to be the best of 20 on hand error and return, so it is a favourable rather than typical episode.

## Limitations specific to this experiment

- One training seed, one fixed reset and one fixed nearby target.
- Survival is capped at 10 s, and zero action also survives.
- Simulation only, with experimental effort caps and contact parameters; no hardware validation.
- The observations are privileged (root velocity, height, target vector).
- No disturbance tests and no untrained or random-policy cohort, beyond the one zero-action episode and a 79-step untrained prefix.
- **Not comparable** with the walking runs, or with the V1/V2 plants, as a controlled experiment.
