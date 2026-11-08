"""Plot the sanitized Experiment 1 summary without private artifacts."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "analysis" / "experiment1_summary.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    horizons = summary["prediction_horizons_seconds"]
    errors = summary["open_loop_errors"]
    values = summary["critic_values_raw_reward_units"]
    continuation = summary["terminal_continuation"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    axes[0].plot(horizons, errors["root_height_mae_mm"], marker="o", label="height MAE (mm)")
    axes[0].plot(horizons, errors["projected_gravity_angle_deg"], marker="o", label="gravity error (deg)")
    axes[0].set(title="Open-loop error", xlabel="horizon (s)")
    axes[0].legend(fontsize=8)
    axes[1].plot(horizons, values["imagined_endpoint"], marker="o", label="imagined")
    axes[1].plot(horizons, values["observed_posterior_endpoint"], marker="o", label="observed posterior")
    axes[1].set(title="Critic value", xlabel="horizon (s)", ylabel="raw reward units")
    axes[1].legend(fontsize=8)
    axes[2].plot(horizons, continuation["predicted"], marker="o", label="predicted")
    axes[2].plot(horizons, continuation["observed"], marker="o", label="observed")
    axes[2].set(title="Terminal continuation", xlabel="horizon (s)", ylim=(-0.05, 1.05))
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.25)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
