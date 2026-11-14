# Representation, prediction, and control

The investigation follows a chain:

`Representation -> Prediction -> Decision/action -> Physical feedback -> Learning`

Each link was tested with a different frozen-policy analysis. None is a substitute for the next.

## Representation

E1 used 120 physical episodes, four frozen policies, and held-out seed clusters to test linear access to current loaded-contact COM support margin. The current observation baseline was 14.278 mm MAE; the 1M posterior was 13.181 mm. The result is a constrained accessibility result. It does not show that the actor reads the feature, that the latent is physically meaningful in general, or that the policy will control better.

## Prediction

E2 evaluated action-conditioned forecasts across horizons, phases, channels, batches, and RSSM particles. Errors grew and changed with phase and signal. At actual-fall anchors, predicted continuation remained well above the target-zero continuation, showing optimistic terminal continuation under the tested filters. Posterior reconstruction can use later observations; it is therefore not evidence of genuine future prediction.

## Decision and physical feedback

Imagined action ranking sometimes reversed the physically useful order. In the follow-up physical test, a bounded negative ankle correction improved the restricted one-second return and several physical secondary metrics, while the mean loaded margin remained negative and every condition survived the one-second window. The actor changed subsequent actions under matched random streams, confirming feedback response, but the design does not identify durable rescue or actor optimality.

## Explicit control

The BLF path produced valid commands, but a downstream numerical admission guard rejected a candidate after 18 commands at 0.360 s. Root and COM heights were lower and pitch was more negative than the zero-action comparison. The stop was numerical censoring, not a physical fall, and the result cannot support a Dreamer-versus-WBC ranking.

## What the chain shows

The evidence is deliberately non-transitive: accessible state information did not guarantee reliable prediction; prediction diagnostics did not guarantee action ranking; local physical improvement did not establish sustained recovery; and improved nominal actuator tracking did not establish superior learned standing-and-reaching performance.
