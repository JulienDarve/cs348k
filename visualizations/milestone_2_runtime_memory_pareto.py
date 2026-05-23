# %% [markdown]
# # Milestone 2 Runtime/Memory Pareto Plot
#
# This script plots median runtime against peak/output memory for the Qwen
# Milestone 2 implementations. Lower-left points are better.

# %%
from pathlib import Path

import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")

SCRIPT_DIR = Path.cwd() if Path.cwd().name == "visualizations" else Path.cwd() / "visualizations"
FIG_DIR = SCRIPT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

WORKLOADS = ["W2", "W3", "W4"]
IMPLEMENTATIONS = ["hf_legacy", "hf_fast", "hf_bilinear", "v1", "v2", "v3"]
LABELS = {
    "hf_legacy": "HF Legacy",
    "hf_fast": "HF Fast",
    "hf_bilinear": "HF Bilinear",
    "v1": "v1 Naive",
    "v2": "v2 Pointwise Fusion",
    "v3": "v3 Full Fusion",
}
COLORS = {
    "hf_legacy": "#4C78A8",
    "hf_fast": "#F58518",
    "hf_bilinear": "#E45756",
    "v1": "#B279A2",
    "v2": "#72B7B2",
    "v3": "#54A24B",
}
MARKERS = {
    "hf_legacy": "o",
    "hf_fast": "s",
    "hf_bilinear": "D",
    "v1": "^",
    "v2": "P",
    "v3": "*",
}

# Median single-thread runtime in ms/img and peak/output RSS from results/bench_kernels_results.md.
RESULTS = {
    "W2": {
        "hf_legacy": {"runtime": 98.685, "memory": 2.09},
        "hf_fast": {"runtime": 120.409, "memory": 3.87},
        "hf_bilinear": {"runtime": 120.357, "memory": 3.76},
        "v1": {"runtime": 133.317, "memory": 1.94},
        "v2": {"runtime": 113.972, "memory": 1.93},
        "v3": {"runtime": 39.887, "memory": 1.00},
    },
    "W3": {
        "hf_legacy": {"runtime": 37.572, "memory": 1.96},
        "hf_fast": {"runtime": 32.304, "memory": 2.29},
        "hf_bilinear": {"runtime": 32.125, "memory": 2.33},
        "v1": {"runtime": 46.904, "memory": 1.86},
        "v2": {"runtime": 39.235, "memory": 1.75},
        "v3": {"runtime": 18.054, "memory": 1.00},
    },
    "W4": {
        "hf_legacy": {"runtime": 836.589, "memory": 2.02},
        "hf_fast": {"runtime": 1013.734, "memory": 3.86},
        "hf_bilinear": {"runtime": 1011.957, "memory": 3.84},
        "v1": {"runtime": 1212.327, "memory": 2.19},
        "v2": {"runtime": 997.460, "memory": 2.06},
        "v3": {"runtime": 334.576, "memory": 1.06},
    },
}


def plot_runtime_memory_pareto():
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.9))

    for ax, workload in zip(axes, WORKLOADS):
        for impl in IMPLEMENTATIONS:
            point = RESULTS[workload][impl]
            size = 145 if impl == "v3" else 78
            ax.scatter(
                point["runtime"],
                point["memory"],
                s=size,
                marker=MARKERS[impl],
                color=COLORS[impl],
                edgecolor="white",
                linewidth=0.9,
                label=LABELS[impl],
                zorder=3 if impl == "v3" else 2,
            )
            ax.annotate(
                LABELS[impl],
                xy=(point["runtime"], point["memory"]),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=8.5,
            )

        ax.set_title(workload, fontsize=13, pad=10)
        ax.set_xlabel("Median runtime (ms/img)")
        ax.set_ylabel("Peak / output memory (x)")
        ax.set_ylim(0.82, 4.15)
        ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=True, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Qwen Preprocessing Runtime/Memory Pareto Frontier", fontsize=15, y=1.18)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "milestone_2_runtime_memory_pareto.png", dpi=200, bbox_inches="tight")
    return fig, axes


plot_runtime_memory_pareto()

