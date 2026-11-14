# Reproducibility

## Level A: direct public checks

From a clean checkout, the following are intended to run without private artifacts or training:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip uv
.venv/bin/uv pip sync --python .venv/bin/python configs/dreamer_requirements.lock
git clone https://github.com/carlosferrazza/humanoid-bench.git vendor/humanoid-bench
git -C vendor/humanoid-bench checkout cb1189039151c8aadaaa987b442da54383c87fab
PYTHONPATH=scripts:vendor/humanoid-bench/dreamerv3 .venv/bin/python scripts/verify_assets.py
PYTHONPATH=scripts:vendor/humanoid-bench/dreamerv3 .venv/bin/python -m unittest scripts.test_prior_prediction scripts.test_jsonl_to_tensorboard
PYTHONPATH=scripts:vendor/humanoid-bench/dreamerv3 .venv/bin/python scripts/dreamer.py check --logdir /tmp/ergocub-readiness --platform cpu
PYTHONPATH=scripts .venv/bin/python scripts/plot_representation.py --output /tmp/experiment1-representation.png
PYTHONPATH=scripts .venv/bin/python scripts/plot_learning_curves.py --output /tmp/learning-curves.png
PYTHONPATH=scripts .venv/bin/python scripts/plot_mechanistic_findings.py --output /tmp/mechanistic-findings.png
```

The physics validator may return status 2 when the unchanged nominal hold falls; that is an expected recorded baseline result. The readiness check initializes and evaluates without training, but may require a compatible JAX/Embodied installation.

## Level B: external artifacts

Frozen-policy evaluation, E1 probes, E2 prediction, E3 ranking/feedback, E4 control analysis, and complete empirical plots require the matching checkpoints, replay or trajectory files, and private analysis inputs. The public JSON summaries support inspection and selected plot regeneration but are not sufficient to re-run those analyses.

## Level C: not reproducible from this release

Fresh end-to-end training, independent training-seed uncertainty, and raw evaluation campaigns are not claimed. Full checkpoints, replay buffers, private logs, credentials, machine-specific paths, and unpublished trajectories are deliberately excluded.

Do not put checkpoints, replay, logs, virtual environments, or datasets in this repository. Asset provenance is recorded in `assets/ergocub/provenance.json`; third-party notices remain in `assets/ergocub/NOTICE` and `assets/ergocub/LICENSE`. Public derived-artifact hashes and evidence identifiers are recorded in `analysis/public_provenance.json`.
