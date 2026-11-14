# Scientific results

This page is the compact evidence-backed narrative. The public repository contains selected figures and sanitized summaries; private checkpoints, replay, trajectories, and full report trees are intentionally excluded.

## Learning and Prior10

Unchanged DreamerV3 improved matched mean ten-second restricted survival from 4.447 s at 300k to 7.739 s at 500k on policy seeds 50-99. Prior10's 500k mean was 8.43 s in the primary table, but its additional mean-survival benefit was not established: the matched difference was +0.692 s with 95% CI [-0.243, 1.626]. On a separate 200-episode panel, the mean difference was +0.350 s with 95% CI [-0.068, 0.770]. Prior10 changed failure directions, including leftward incidence from 1.0% to 21.5% in that panel.

Longer Prior10 continuation was non-monotonic on the reused panel: 600k, 700k, 800k, 900k, and 1M produced 9.339, 5.248, 7.732, 8.784, and 8.086 s mean capped survival. The 600k value is a panel result, not a general optimum.

## Mechanism studies

- **E1, representation:** 1M posterior MAE for loaded-contact COM support margin was 13.181 mm versus 14.278 mm for the current-observation baseline. The simultaneous 95% interval for the difference was [-2.069, -0.125] mm; only this posterior met the prespecified primary criterion. This establishes limited linear accessibility, not understanding or actor use.
- **E2, prediction:** action-conditioned prediction errors varied by phase and channel. Predicted continuation remained optimistic near actual falls. Posterior reconstruction and genuine future prediction are different tests.
- **E3, action and feedback:** imagined ranking could disagree with physical outcomes. A bounded negative ankle intervention improved one-second restricted return by +1.096630, paired 95% CI [0.377594, 1.843514], with support margin improving by 12.378 mm. All branches survived the one-second cap, so durable recovery was not demonstrated.
- **E4, explicit control:** an existing-library BLF balance chain was integrated, but numerical admission stopped the trial at 0.360 s and the trajectory deteriorated. This was not a successful WBC baseline or a fair controller comparison.

## V2

V2 validated named per-joint PD changes and improved nominal actuator tracking and hold behavior. Fresh training reached 550k with a complete frozen curve, but V2 did not demonstrate better combined standing-and-reaching learning than V1. V2 550k averaged 5.299 s survival with 18/20 qualified reaching episodes and 3/20 ten-second survivors; a separate V1 500k cohort averaged 7.739 s. The cross-version cohorts, control packages, and evaluation schedules are not matched or causally comparable, so actuator gains alone cannot explain the difference. Qualified or latched reaching is not sustained reaching.

## Interpretation

The completed evidence separates representation, prediction, action ranking, physical feedback, and learning. Improvements in one layer did not uniformly produce improvements in the others. See [representation, prediction, and control](representation_prediction_control.md) for the mechanism-level reading and [limitations](limitations.md) for the inference boundary.
