# %% [markdown]
# # Milestone 2 Qwen Kernel Ablation Speedups
#
# This script visualizes the single-thread Milestone 2 Qwen preprocessing ablation.
# Runtime is normalized to Hugging Face fast for each workload, so values above
# 1.0x are faster than the HF fast baseline.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-whitegrid")

SCRIPT_DIR = Path.cwd() if Path.cwd().name == "visualizations" else Path.cwd() / "visualizations"
FIG_DIR = SCRIPT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

WORKLOADS = ["W2", "W3", "W4"]
IMPLEMENTATIONS = ["hf_fast", "v1", "v2", "v3"]
LABELS = {
    "hf_fast": "HF Fast",
    "v1": "v1 Naive",
    "v2": "v2 Pointwise Fusion",
    "v3": "v3 Full Fusion",
}
COLORS = {
    "hf_fast": "#4C78A8",
    "v1": "#B279A2",
    "v2": "#F58518",
    "v3": "#54A24B",
}

# Median single-thread runtime in ms/img from results/bench_kernels_results.md.
RUNTIME_MS_PER_IMG = {
    "W2": {"hf_fast": 120.409, "v1": 133.317, "v2": 113.972, "v3": 39.887},
    "W3": {"hf_fast": 32.304, "v1": 46.904, "v2": 39.235, "v3": 18.054},
    "W4": {"hf_fast": 1013.734, "v1": 1212.327, "v2": 997.460, "v3": 334.576},
}


def annotate_bars(ax, bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}x",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_ablation_speedups():
    x = np.arange(len(WORKLOADS))
    width = 0.19

    fig, ax = plt.subplots(figsize=(10.5, 5.8))

    for idx, impl in enumerate(IMPLEMENTATIONS):
        offset = (idx - (len(IMPLEMENTATIONS) - 1) / 2) * width
        values = [
            RUNTIME_MS_PER_IMG[workload]["hf_fast"] / RUNTIME_MS_PER_IMG[workload][impl]
            for workload in WORKLOADS
        ]
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=LABELS[impl],
            color=COLORS[impl],
            edgecolor="white",
            linewidth=0.8,
        )
        annotate_bars(ax, bars)

    ax.axhline(1.0, color="#333333", linewidth=1.1, linestyle="--", alpha=0.8)
    ax.text(
        len(WORKLOADS) - 0.52,
        1.04,
        "HF Fast baseline",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#333333",
    )
    ax.set_title("Qwen Kernel Ablation: Runtime Speedup vs HF Fast", fontsize=15, pad=14)
    ax.set_ylabel("Speedup over HF fast (x)")
    ax.set_xticks(x)
    ax.set_xticklabels(WORKLOADS)
    ax.legend(frameon=True, ncol=2)
    ax.margins(y=0.2)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "milestone_2_ablation_speedups.png", dpi=200, bbox_inches="tight")
    return fig, ax


plot_ablation_speedups()

