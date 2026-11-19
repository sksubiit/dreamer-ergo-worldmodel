# Public release checklist

Status: documentation and lightweight reproducibility pass for review. No merge, tag, or additional publication is authorized by this checklist.

- [x] README updated from ongoing-study wording to the completed E1-E4/V2 scope.
- [x] Research setup, scientific results, mechanism interpretation, V1/V2 comparison, reproducibility boundary, and limitations documented.
- [x] Existing sanitized result tables and figures retained; no fabricated frames, trajectories, intervals, or new empirical claims added.
- [x] Added sanitized source data: `analysis/learning_curves.csv`, `analysis/mechanism_summary.json`, and `analysis/video_manifest.json`.
- [x] Corrected Prior10 600k-1M survival rates and restored the authoritative mean-survival intervals; regenerated the learning-curve figure.
- [x] Added public artifact hashes and evidence identifiers in `analysis/public_provenance.json`.
- [x] Added generated figures and scripts: `assets/figures/learning-curves.png`, `assets/figures/mechanistic-findings.png`, `scripts/plot_learning_curves.py`, and `scripts/plot_mechanistic_findings.py`.
- [x] Added two V2 550k videos with checkpoint, seed, and outcome labels; one timeout after qualified reaching and one failure.
- [x] Asset provenance and third-party notices retained under `assets/ergocub/`.
- [ ] Run the documented CPU environment installation and readiness check on a clean machine; dependency availability is environment-dependent.
- [x] Run the asset verifier on the release branch; it passed for all 69 bundled files.
- [ ] Run the lightweight unit tests; the current system Python lacks `jax` and `tensorboard`.
- [x] Validate all local Markdown targets and inspect the two MP4 files as present files; full stream validation is blocked because `imageio` and `ffprobe` are unavailable.
- [x] Regenerate both new figures from the public sanitized data; both PNGs are non-empty and were visually inspected.
- [x] Deliberately excluded checkpoints, replay buffers, raw trajectories, private reports, credentials, internal notes, machine-specific paths, and proprietary or unverified assets.
- [x] Checked the public diff for private paths, credentials, raw experiment outputs, and unallowlisted large files; only the two selected V2 MP4s are included as new media.
- [x] Known limitations recorded, including one lineage, 10-second censoring, cohort differences, retrospective state selection, and absent hardware validation.

## My contributions

This repository presents an independent investigation built around SN001 simulation and actuator integration, DreamerV3 experimentation, the Prior10 implementation, representation/prediction analysis, physical action-ranking and closed-loop evaluation, V2 control-interface engineering, and reproducibility/statistical evaluation. Upstream DreamerV3, HumanoidBench, MuJoCo, and official ergoCub assets remain attributed to their original authors.

## SN001 update (release candidate, not merged)

Status: prepared on branch `release/sn001-results-and-walking-closeout` for review. Not pushed, merged or tagged.

- [x] Added [SN001 standing/reaching](sn001_standing_reaching.md) and the [two-seed walking negative result](sn001_walking_negative_result.md). Extended the README, setup, results, V1/V2/SN001 comparison, limitations and reproducibility pages. Historical V1/V2 and E1–E4 text is unchanged.
- [x] Every SN001 number was derived from private evidence by a script with no physics and no policy calls. Each public claim maps to an evidence identifier and a SHA-256 in `analysis/release_manifest.json`.
- [x] Checkpoint scope stated exactly: the intermediate SN001 standing checkpoint is 75k, the final is 110k, and there is no SN001 standing run beyond 110k.
- [x] Added three clips: one re-render of recorded 110k states, and two visual-mesh renders of registered walking evaluation episodes. Selection rules and hashes are in `analysis/video_manifest.json`. Two poster figures are copied frames, regenerable with `scripts/make_sn001_posters.py`.
- [x] Disclosed that the SN001 standing clip is the cohort's best episode, as well as its first seed.
- [x] No checkpoints, replay, trajectories, private reports, private paths or SN001 plant/runner code are included.
- [ ] Researcher review of wording, media and the LICENSE statement below.
- [x] **LICENSE wording corrected.** The root `LICENSE` previously said no robot model or meshes were included, although `assets/ergocub/` distributes them. It now separates MIT-licensed project contributions from the bundled ergoCubSN001 model and meshes, which remain under their BSD-3-Clause notice (iCub Tech, 2022) and are not relicensed. The MIT and BSD notice texts are unchanged.
