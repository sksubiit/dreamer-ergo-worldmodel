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

## SN001 platforms

The later SN001 experiments use two further plants built from the official SN001 sources. All four plants are different, and policies are not transferable between them. Any cross-row comparison is descriptive.

| Plant | Servo/actuation | Contacts | Zero-action outcome | Experiments |
| --- | --- | --- | --- | --- |
| V1 | Kp 25.27 position servos | four sole pads | falls (about 0.69 s nominal hold) | V1, Prior10, E1–E4 |
| V2 | V1 with knee/ankle-pitch Kp ×4, Kd ×2 | four sole pads | falls (about 1.08 s) | V2 to 550k |
| SN001 reference `sn001_sim_v1` | Kp 1000, actuator Kv 0, passive damping 2 | official box contacts | **stands for 10 s** | [standing/reaching 110k](sn001_standing_reaching.md) |
| SN001 Motor V2 contract | Kp 25.266, Kd 2.011, effort caps 60/110/17/20 Nm | official box contacts | collapses in 24 commands (≈ 0.48 s) under the walking task | [walking, two seeds, 200k](sn001_walking_negative_result.md) |

**Not a learning advantage.** The SN001 reference results (10 s survival, 100% reaching at 75k/110k) are descriptively stronger than V2 at the same budgets (2.114 s and 15% at 75k; 4.577 s and 30% at 110k). Much of that difference is provided by the plant itself: the stiff reference servos hold the robot without any policy. This comparison does not show a learning-efficiency advantage of SN001, of DreamerV3 on SN001, or of any gain setting.
