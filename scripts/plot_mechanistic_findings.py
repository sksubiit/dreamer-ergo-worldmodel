"""Plot the sanitized central E1-E4 findings."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "analysis" / "mechanism_summary.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.data.read_text())

    figure, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")

    e1 = data["e1_representation"]
    labels = ["Observation", "300k", "500k", "600k", "1M"]
    values = list(e1["values"].values())
    axes[0, 0].bar(labels, values, color=["#777777", "#245a8d", "#245a8d", "#c46b27", "#b33f3f"])
    axes[0, 0].set_ylabel("Held-out MAE (mm)")
    axes[0, 0].set_title("E1: current support-margin access")
    axes[0, 0].tick_params(axis="x", rotation=20)
    axes[0, 0].text(0.98, 0.96, "1M - observation\n95% simultaneous CI: [-2.069, -0.125] mm",
                     transform=axes[0, 0].transAxes, ha="right", va="top", fontsize=8,
                     bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"})

    e2 = data["e2_prediction"]
    x = np.arange(len(e2["models"]))
    intervals = np.array(e2["simultaneous_or_bootstrap_intervals"])
    y = np.array(e2["values"])
    axes[0, 1].errorbar(x, y, yerr=[y - intervals[:, 0], intervals[:, 1] - y],
                        fmt="o", color="#b33f3f", capsize=4, linewidth=1.5)
    axes[0, 1].axhline(e2["target"], color="#333333", linestyle="--", linewidth=1)
    axes[0, 1].set_xticks(x, ["300k", "500k", "600k", "1M"])
    axes[0, 1].set_ylim(-0.05, 1.05)
    axes[0, 1].set_ylabel("Predicted continuation")
    axes[0, 1].set_title("E2: optimistic continuation at falls")
    axes[0, 1].grid(axis="y", alpha=0.22)

    e3 = data["e3_action_feedback"]
    axes[1, 0].errorbar([0], [e3["estimate"]],
                        yerr=[[e3["estimate"] - e3["paired_95_percent_interval"][0]],
                              [e3["paired_95_percent_interval"][1] - e3["estimate"]]],
                        fmt="o", color="#2d7a58", capsize=6, markersize=8)
    axes[1, 0].axhline(0, color="#333333", linewidth=1)
    axes[1, 0].set_xlim(-0.7, 0.7)
    axes[1, 0].set_xticks([0], ["Negative ankle correction"])
    axes[1, 0].set_ylabel("One-second restricted return difference")
    axes[1, 0].set_title("E3: finite-window physical feedback")
    axes[1, 0].text(0.02, 0.95, "Paired 95% CI; all branches survived 1 s",
                     transform=axes[1, 0].transAxes, va="top", fontsize=8)
    axes[1, 0].grid(axis="y", alpha=0.22)

    e4 = data["e4_explicit_control"]
    axes[1, 1].bar(["BLF commissioning"], [e4["duration"]], color="#777777", width=0.45)
    axes[1, 1].set_ylim(0, 0.5)
    axes[1, 1].set_ylabel("Seconds before numerical stop")
    axes[1, 1].set_title("E4: explicit-control integration")
    axes[1, 1].text(0, e4["duration"] + 0.02, "0.360 s\n18 commands",
                     ha="center", va="bottom", fontsize=9)
    axes[1, 1].text(0.02, 0.95, "Guard-censored; not a physical fall or WBC baseline",
                     transform=axes[1, 1].transAxes, va="top", fontsize=8)
    axes[1, 1].grid(axis="y", alpha=0.22)

    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Mechanistic findings from the completed E1-E4 investigation", fontsize=14, fontweight="bold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220)
    plt.close(figure)


if __name__ == "__main__":
    main()
