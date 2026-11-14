# V1 and V2 comparison

V2 is an engineering and training comparison, not a controlled claim that PD gains caused a better learned policy.

| Item | V1 reference | V2 final cohort |
| --- | ---: | ---: |
| Training state | 500k unchanged DreamerV3 | 550k fresh DreamerV3 |
| Mean capped survival | 7.739 s | 5.299 s |
| Qualified reaching | 100% in the matched 50-seed panel | 18/20 at 550k |
| Ten-second survivors | 42% in the matched 50-seed panel | 3/20 |

V2's sampled reaching completion first became high at 440k (19/20), then was 18/20 at 550k. Mean survival across V2 checkpoints was non-monotonic: 2.114, 4.577, 3.945, 2.837, 4.154, and 5.299 s. Three final episodes reached the 10-second cap, while 17/20 fell.

The validated V2 control package included named per-joint PD settings and improved nominal tracking/hold behavior. The historical comparison uses different cohorts, evaluation schedules, task/control packages, and a Prior10 V1 branch elsewhere in the study. Therefore it does not isolate actuator gains, establish a learning-efficiency advantage, or prove that the simulator is defective. Better nominal tracking did not demonstrate superior combined standing-and-reaching learning.
