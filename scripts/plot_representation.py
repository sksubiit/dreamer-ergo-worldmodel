"""Plot the sanitized Experiment 1 representation comparison."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "analysis" / "representation_summary.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    labels = ["Observation", *summary["frozen_policies"]]
    values = [summary["current_observation_baseline_mae_mm"]]
    values.extend(summary["posterior_mae_mm"][label] for label in summary["frozen_policies"])

    fig, axis = plt.subplots(figsize=(7, 4), layout="constrained")
    bars = axis.bar(labels, values, color=["0.55", "tab:blue", "tab:blue", "tab:orange", "tab:red"])
    axis.set_ylabel("Held-out MAE (mm)")
    axis.set_title("Linear accessibility of loaded-contact support margin")
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.1, f"{value:.3f}",
                  ha="center", va="bottom", fontsize=8)
    interval = summary["one_m_posterior_minus_observation_simultaneous_95_percent_interval_mm"]
    axis.text(0.98, 0.96, f"1M - observation: [{interval[0]:.3f}, {interval[1]:.3f}] mm\n"
              f"Improvement: {summary['one_m_posterior_improvement_percent']:.1f}%",
              transform=axis.transAxes, ha="right", va="top", fontsize=8,
              bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
