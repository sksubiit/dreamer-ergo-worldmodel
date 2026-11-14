# Public release checklist

Status: documentation and lightweight reproducibility pass for review. No merge, tag, or additional publication is authorized by this checklist.

- [x] README updated from ongoing-study wording to the completed E1-E4/V2 scope.
- [x] Research setup, scientific results, mechanism interpretation, V1/V2 comparison, reproducibility boundary, and limitations documented.
- [x] Existing sanitized result tables and figures retained; no fabricated frames, trajectories, intervals, or new empirical claims added.
- [x] Public figures and data have named local source files: `assets/figures/`, `assets/demo/`, and `analysis/`.
- [x] Asset provenance and third-party notices retained under `assets/ergocub/`.
- [ ] Run the documented CPU environment installation and readiness check on a clean machine; dependency availability is environment-dependent.
- [x] Run the asset verifier on the release branch; it passed for all 69 bundled files.
- [ ] Run the lightweight unit tests; the current system Python lacks `jax` and `tensorboard`.
- [x] Validate local Markdown targets; media rendering still needs a browser-side inspection before opening the pull request.
- [x] Deliberately excluded checkpoints, replay buffers, raw trajectories, private reports, credentials, internal notes, machine-specific paths, and proprietary or unverified assets.
- [x] Known limitations recorded, including one lineage, 10-second censoring, cohort differences, retrospective state selection, and absent hardware validation.

## My contributions

This repository presents an independent investigation built around SN001 simulation and actuator integration, DreamerV3 experimentation, the Prior10 implementation, representation/prediction analysis, physical action-ranking and closed-loop evaluation, V2 control-interface engineering, and reproducibility/statistical evaluation. Upstream DreamerV3, HumanoidBench, MuJoCo, and official ergoCub assets remain attributed to their original authors.
