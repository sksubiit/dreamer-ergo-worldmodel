# Public release checklist

Status: documentation and lightweight reproducibility pass for review. No merge, tag, or additional publication is authorized by this checklist.

- [x] README updated from ongoing-study wording to the completed E1-E4/V2 scope.
- [x] Research setup, scientific results, mechanism interpretation, V1/V2 comparison, reproducibility boundary, and limitations documented.
- [x] Existing sanitized result tables and figures retained; no fabricated frames, trajectories, intervals, or new empirical claims added.
- [x] Added sanitized source data: `analysis/learning_curves.csv`, `analysis/mechanism_summary.json`, and `analysis/video_manifest.json`.
- [x] Added generated figures and scripts: `assets/figures/learning-curves.png`, `assets/figures/mechanistic-findings.png`, `scripts/plot_learning_curves.py`, and `scripts/plot_mechanistic_findings.py`.
- [x] Added two V2 550k videos with checkpoint, seed, and outcome labels; one timeout after qualified reaching and one failure.
- [x] Asset provenance and third-party notices retained under `assets/ergocub/`.
- [ ] Run the documented CPU environment installation and readiness check on a clean machine; dependency availability is environment-dependent.
- [x] Run the asset verifier on the release branch; it passed for all 69 bundled files.
- [ ] Run the lightweight unit tests; the current system Python lacks `jax` and `tensorboard`.
- [x] Validate all local Markdown targets and inspect the two MP4 files as present files; full stream validation is blocked because `imageio` and `ffprobe` are unavailable.
- [x] Regenerate both new figures from the public sanitized data; both PNGs are non-empty and were visually inspected.
- [x] Deliberately excluded checkpoints, replay buffers, raw trajectories, private reports, credentials, internal notes, machine-specific paths, and proprietary or unverified assets.
- [x] Checked the public diff for private paths, credentials, raw experiment outputs, and unallowlisted large files; only the two selected MP4s are included.
- [x] Known limitations recorded, including one lineage, 10-second censoring, cohort differences, retrospective state selection, and absent hardware validation.

## My contributions

This repository presents an independent investigation built around SN001 simulation and actuator integration, DreamerV3 experimentation, the Prior10 implementation, representation/prediction analysis, physical action-ranking and closed-loop evaluation, V2 control-interface engineering, and reproducibility/statistical evaluation. Upstream DreamerV3, HumanoidBench, MuJoCo, and official ergoCub assets remain attributed to their original authors.
