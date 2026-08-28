#!/usr/bin/env python3
"""Generate the benchmark comparison figure used on the project page."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures"
WEB_ASSET_DIR = ROOT / "docs" / "assets"

BASE = "#aab2bb"
BASELINE = "#c69214"
OURS = "#176f67"
INK = "#252a31"
GRID = "#dfe3e7"


def draw_panel(ax, title, labels, base, baseline, ours, baseline_names):
    y = np.arange(len(labels))
    height = 0.22

    ax.barh(y - height, base, height, color=BASE, label="Base Qwen2.5-7B")
    ax.barh(y, baseline, height, color=BASELINE, label="Best same-backbone baseline")
    ax.barh(y + height, ours, height, color=OURS, label="M2D-Chat")

    for row, values in enumerate(zip(base, baseline, ours, baseline_names)):
        base_score, baseline_score, our_score, baseline_name = values
        ax.text(base_score + 1.1, row - height, f"{base_score:.1f}", va="center", fontsize=8, color=INK)
        ax.text(
            baseline_score + 1.1,
            row,
            f"{baseline_score:.1f}  {baseline_name}",
            va="center",
            fontsize=8,
            color="#70520b",
        )
        ax.text(
            our_score + 1.1,
            row + height,
            f"{our_score:.1f}",
            va="center",
            fontsize=8.2,
            fontweight="bold",
            color=OURS,
        )

    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK, pad=10)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 101)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlabel("Accuracy (%)")
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#8d949c")
    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", colors="#5b636d")


def main():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "Helvetica Neue", "DejaVu Sans"],
            "font.size": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.75), gridspec_kw={"wspace": 0.34})

    draw_panel(
        axes[0],
        "(a) Personalization",
        ["PersonaMem-v1 MCQ", "PersonaMem-v2 MCQ", "PrefEval Gen", "PrefEval Cls"],
        [49.9, 32.0, 23.4, 62.0],
        [54.3, 39.8, 43.6, 78.5],
        [56.4, 42.0, 56.8, 78.9],
        ["Mem0", "Mem0", "HumanLM", "HumanLM"],
    )
    draw_panel(
        axes[1],
        "(b) Theory of Mind",
        ["ToMi", "BigToM Belief", "BigToM Action"],
        [80.5, 31.0, 23.5],
        [85.5, 40.3, 29.3],
        [82.3, 44.0, 31.0],
        ["AutoToM", "AutoToM", "AutoToM"],
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=3,
        frameon=False,
        fontsize=9,
        handlelength=1.8,
    )
    fig.subplots_adjust(top=0.82, bottom=0.17, left=0.16, right=0.98)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    WEB_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "benchmark_comparison.pdf")
    fig.savefig(WEB_ASSET_DIR / "benchmark_comparison.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
