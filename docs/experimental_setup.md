# Experimental setup

## Task and interface

The study uses the standard ergoCubSN001 model in MuJoCo for left-palm reaching with whole-body stabilization. The public environment exposes a 90-dimensional proprioceptive/task observation and 23 normalized joint position-reference actions at 50 Hz. Commands pass through the recorded servo interface; they are not direct torque actions.

The physics adapter preserves the bundled inertial model, four sole-pad collision boxes, joint limits, effort caps, and recorded servo settings. The public asset bundle records the source revision and SHA-256 manifest in `assets/ergocub/`.

## DreamerV3 path

DreamerV3 encodes observations into an RSSM with deterministic recurrent state and stochastic latent state. The world model predicts observation and reward consequences under actions. Actor and critic updates use imagined trajectories, while environment interaction updates the recurrent belief from observations. The pinned HumanoidBench source revision is checked by the launcher before agent construction.

## Evaluation conventions

Survival is capped at 10 seconds. A qualified reaching event requires the palm to remain within 3 cm of the target for 0.5 seconds, root tilt below 20 degrees, and root height above 90% of reset height. Reaching, survival, and sustained reaching are reported separately. Results come from one training lineage and named frozen-policy cohorts; cohort and checkpoint differences are retained in the result documents.

No hardware, sim-to-real transfer, or independent training-seed replication was evaluated for the V1/V2 study.

## Later SN001 experiments

Two later experiments use plants built from the official ergoCubSN001 sources at the same model revision. Both are described in [V1, V2 and SN001 platforms](v1_v2_comparison.md#sn001-platforms).

- **[Standing and reaching](sn001_standing_reaching.md).** Stiff reference servos, a 90-D observation and a fixed 5 cm palm target. One training seed to 110k; frozen evaluation of 75k and 110k on 20 policy seeds each.
- **[Forward walking](sn001_walking_negative_result.md).** Motor V2 servos, an 88-D observation and a forward-speed reward. Two independent training seeds (0 and 2) to 200k each; 30 stochastic episodes per seed, with contact forces measured at every physics substep.

Both use the unmodified pinned DreamerV3 learner and learn from scratch.
