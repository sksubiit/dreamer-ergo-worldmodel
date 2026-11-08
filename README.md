# DreamerV3 for ergoCub

Empirical investigation of DreamerV3, proprioception-dominant humanoid
control, and learned world models for a standard ergoCubSN001 in MuJoCo.

[![Standard ergoCub reaching and stabilization in MuJoCo](assets/demo/reaching-balance-500k.gif)](assets/demo/reaching-balance-500k.mp4)

The repository contains the public implementation surface, selected figures,
and a frozen record of results.

## Research system

The task combines left-palm reaching with whole-body stabilization. The
environment exposes a 90-dimensional proprioceptive and task-state vector and
23 normalized joint position-reference actions at 50 Hz. The physics adapter
builds the standard ergoCubSN001 URDF, preserves its inertial model, uses the
four original sole-pad collision boxes, and applies the recorded experimental
servo settings. No pretrained humanoid controller supplies the behavior.

DreamerV3 encodes observations into a recurrent state-space model (RSSM). The
RSSM combines a deterministic recurrent state with a stochastic latent state,
predicts observation and reward consequences under actions, and supports
imagination without stepping MuJoCo. An actor and critic are trained on
imagined trajectories; the actor selects actions and the critic estimates
future return. This is a model-based reinforcement-learning study, not a claim
that the latent state is a complete physical model.

The upstream DreamerV3 source is pinned to HumanoidBench revision
`cb1189039151c8aadaaa987b442da54383c87fab`. The public launcher verifies that
revision before constructing an agent. The robot model and meshes are copied
from ergoCub software revision
`6c255e258cd184e91765b607222b6b6877bf7459`, with its BSD-3-Clause notice
retained under `assets/ergocub/`.

## Frozen results

These are verified results from one training seed and matched frozen-policy
evaluation cohorts. They are reported for Empirical traceability, not as
generalization, safety, or continual-learning claims.

| Policy and total driver steps | Mean survival, capped at 10 s | Reached target | Survived to 10 s |
| --- | ---: | ---: | ---: |
| DreamerV3, 300k | 4.45 s | 100% | 0% |
| DreamerV3, 500k | 7.74 s | 100% | 42% |
| Prior10, 500k | 8.43 s | 100% | 56% |

The primary 50-episode comparison used matched policy seeds 50–99. In a
separate 200-episode panel (seeds 100–299), 500k control/Prior10 mean survival
was 8.256/8.606 s, with 47.0%/63.5% surviving to 10 s. The paired mean
difference was +0.350 s with 95% CI [-0.068, 0.770], so a mean-survival
advantage was not established. Prior10 also changed the failure distribution:
leftward falls increased from 1.0% to 21.5% in that panel.

The Prior10 intervention adds a 0.1-weighted, ten-step open-loop prior loss to
the world-model heads using recorded actions. It leaves the upstream network
and optimizer structure unchanged. In the 50-seed comparison, Prior10 minus
unchanged 500k was +0.69 s, 95% CI [-0.24, 1.63], and held-out action-ranking
agreement was 10/10 for Prior10 versus 1/10 for unchanged 500k. This small
open-loop diagnostic did not establish reliable action ranking or calibrated
values.

The Prior10 continuation was evaluated at 600k, 700k, 800k, 900k, and 1M
total steps on the reused 200-seed panel. Mean capped survival was respectively
9.339, 5.248, 7.732, 8.784, and 8.086 s. The 1M value remained censored by the
10-second cap and does not establish indefinite balance. These continuation
results require unavailable checkpoints and replay; they are not reproducible
from this checkout alone.

## Experiment 1: latent representation analysis

Experiment 1 is complete as a frozen-model analysis. It probes the RSSM latent
representation for information relevant to proprioceptive state, contact,
uprightness, reaching, and fall-related outcomes, while separating analysis
from policy training. Its private checkpoint- and trajectory-level artifacts
are intentionally not redistributed. The public code preserves the model,
physics, and evaluation interfaces needed for a future independently packaged
analysis. Experiment 2 remains ongoing and is not included.

![Matched survival, qualified dwell, and paired episode outcomes](assets/figures/survival-comparison.png)

![Frozen-model action-ranking diagnostic](assets/figures/action-ranking.png)

The figures are unchanged selected exports from the frozen evaluations. A
qualified reaching event means the palm is within 3 cm of the target for 0.5 s,
with root tilt below 20 degrees and root height above 90% of reset height.

## Reproducibility boundary

**Reproducible from this repository:** asset hashes and URDF references,
physics construction, reset and actuator checks, environment interfaces,
Prior10 loss unit tests, and JSONL-to-TensorBoard conversion tests. These are
lightweight checks and do not perform training or optimizer updates.

**Reproducible analysis with external inputs:** frozen-policy evaluation,
latent probes, action-ranking diagnostics, and result figures require the
corresponding checkpoints, replay or trajectory files, and analysis scripts.
Those private artifacts are deliberately absent.

**Not claimed:** a fresh training reproduction, independent-seed uncertainty,
hardware transfer, disturbance robustness, self-collision handling, or
long-horizon stability. The simplified contact model and the 10-second cap are
important limitations.

## Setup and checks

Install the pinned physics requirements for asset-only checks:

```bash
python3 -m pip install -r configs/physics_requirements.in
PYTHONPATH=scripts python3 scripts/verify_assets.py
PYTHONPATH=scripts python3 scripts/validate_physics.py --output /tmp/ergocub-physics --no-render
```

The physics validator intentionally reports exit status 2 when the unchanged
nominal hold falls; its geometry, reset, actuator-limit, and pulse checks still
record the measured failure and are the expected lightweight baseline result.

For Dreamer checks, obtain the pinned upstream HumanoidBench checkout at
`vendor/humanoid-bench` and make its `dreamerv3` directory available as shown
by the launcher. Install `configs/dreamer_requirements.lock`, then run:

```bash
PYTHONPATH=scripts python3 -m unittest scripts.test_prior_prediction
PYTHONPATH=scripts python3 -m unittest scripts.test_jsonl_to_tensorboard
PYTHONPATH=scripts python3 scripts/dreamer.py check --logdir /tmp/ergocub-readiness --platform cpu
```

The readiness command initializes and evaluates a policy without training. It
may still require a compatible JAX/Embodied installation. Do not place
checkpoints, replay, logs, virtual environments, or datasets in this checkout.

## References and licenses

- Hafner et al., *Mastering Diverse Domains through World Models*, DreamerV3:
  [project page](https://danijar.com/project/dreamerv3/).
- HumanoidBench DreamerV3 implementation:
  [pinned source](https://github.com/carlosferrazza/humanoid-bench/tree/cb1189039151c8aadaaa987b442da54383c87fab/dreamerv3).
- ergoCub software and model source:
  [pinned revision](https://github.com/mesh-iit/ergocub-software/tree/6c255e258cd184e91765b607222b6b6877bf7459/urdf/ergoCub/robots/ergoCubSN001).
- MuJoCo: [mujoco.org](https://mujoco.org/).

Original repository contributions are covered by [LICENSE](LICENSE). Third-party
licenses and attribution notices remain with their respective works.