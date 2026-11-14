"""Plot sanitized V1, Prior10, and V2 frozen-policy learning curves."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "analysis" / "learning_curves.csv"
COLORS = {"v1_matched": "#245a8d", "prior10_progression": "#c46b27", "v2_curve": "#2d7a58"}
LABELS = {"v1_matched": "V1 unchanged control", "prior10_progression": "V1 Prior10", "v2_curve": "V2"}


def load_rows(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("driver_steps", "mean_survival_s", "ci95_low_s", "ci95_high_s",
                    "survival5_rate", "survival10_rate", "qualified_reaching_rate"):
            row[key] = float(row[key]) if row[key].strip() else None
    return rows


def plot_series(axis, rows, value_key, ylabel, percent=False):
    for series in ("v1_matched", "prior10_progression", "v2_curve"):
        points = [row for row in rows if row["series"] == series]
        if not points:
            continue
        x = [row["driver_steps"] / 1000 for row in points]
        y = [row[value_key] * 100 if percent else row[value_key] for row in points]
        axis.plot(x, y, marker="o", linewidth=2, label=LABELS[series], color=COLORS[series])
        if value_key == "mean_survival_s":
            lower = [row["ci95_low_s"] for row in points]
            upper = [row["ci95_high_s"] for row in points]
            if all(value is not None for value in lower + upper):
                axis.fill_between(x, lower, upper, color=COLORS[series], alpha=0.12)
    axis.set_xlabel("Training steps (thousands)")
    axis.set_ylabel(ylabel)
    axis.set_xlim(0, 1050)
    if percent:
        axis.set_ylim(-0.03 * 100, 1.03 * 100)
        axis.set_yticks([0, 25, 50, 75, 100])
        axis.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    axis.grid(alpha=0.22)
    axis.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(args.data)
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, layout="constrained")
    plot_series(axes[0, 0], rows, "mean_survival_s", "Mean survival (s)")
    axes[0, 0].set_title("Capped survival")
    plot_series(axes[0, 1], rows, "survival10_rate", "Episodes surviving 10 s", percent=True)
    axes[0, 1].set_title("Ten-second completion")
    plot_series(axes[1, 0], rows, "qualified_reaching_rate", "Qualified reaching", percent=True)
    axes[1, 0].set_title("Latched reaching event")
    plot_series(axes[1, 1], rows, "survival5_rate", "Episodes surviving 5 s", percent=True)
    axes[1, 1].set_title("Five-second survival")
    axes[0, 0].legend(loc="best", fontsize=9, frameon=False)
    figure.suptitle("Frozen-policy learning curves: within-series progression", fontsize=14, fontweight="bold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220)
    plt.close(figure)


if __name__ == "__main__":
    main()
