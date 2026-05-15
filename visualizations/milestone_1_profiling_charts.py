# %% [markdown]
# # Milestone 1 Profiling Visualizations
# 
# This notebook creates profiling-focused figures for the `MILESTONE_1.md` profiling section. The runtime and memory notebooks answer how slow and how memory-hungry preprocessing is; these figures answer what kind of work dominates.
# 
# The category values below are grouped from the top cProfile entries in `notes/full_benchmarks_results.md`. They are intentionally coarse and conservative: nested cumulative cProfile rows are not summed directly when that would double-count a parent and child call. The point is to visualize the dominant operation families, not to reconstruct an exact line-level profiler trace.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-whitegrid")

NOTEBOOK_DIR = Path.cwd() if Path.cwd().name == "visualizations" else Path.cwd() / "visualizations"
FIG_DIR = NOTEBOOK_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

categories = [
    "Resize",
    "Rescale / normalize",
    "Format conversion",
    "Patch / tile logic",
    "Stack / cat / reshape / pad",
    "Other",
]

colors = {
    "Resize": "#4C78A8",
    "Rescale / normalize": "#F58518",
    "Format conversion": "#B279A2",
    "Patch / tile logic": "#54A24B",
    "Stack / cat / reshape / pad": "#E45756",
    "Other": "#9D9D9D",
}

variants = [
    "Qwen legacy",
    "Qwen fast",
    "IV25 HF legacy",
    "IV25 HF fast",
    "IV25 manual",
    "LLaVA legacy",
    "LLaVA fast",
]

# Approximate grouped hotspot time in seconds. Values are derived from the W3/W4 cProfile traces.
# The categories are kept non-overlapping for plotting, so parent cumulative calls are not added on top
# of their child calls.
profile_seconds = {
    "W3 randomized images": {
        "Qwen legacy": {
            "Resize": 0.476,
            "Rescale / normalize": 0.476,
            "Format conversion": 0.309,
            "Patch / tile logic": 0.000,
            "Stack / cat / reshape / pad": 0.266,
            "Other": 0.255,
        },
        "Qwen fast": {
            "Resize": 0.185,
            "Rescale / normalize": 0.120,
            "Format conversion": 0.000,
            "Patch / tile logic": 0.000,
            "Stack / cat / reshape / pad": 0.801,
            "Other": 0.162,
        },
        "IV25 HF legacy": {
            "Resize": 0.395,
            "Rescale / normalize": 0.135,
            "Format conversion": 0.058,
            "Patch / tile logic": 0.000,
            "Stack / cat / reshape / pad": 0.000,
            "Other": 0.128,
        },
        "IV25 HF fast": {
            "Resize": 0.149,
            "Rescale / normalize": 0.029,
            "Format conversion": 0.038,
            "Patch / tile logic": 0.000,
            "Stack / cat / reshape / pad": 0.107,
            "Other": 0.000,
        },
        "IV25 manual": {
            "Resize": 1.409,
            "Rescale / normalize": 0.321,
            "Format conversion": 0.126,
            "Patch / tile logic": 0.209,
            "Stack / cat / reshape / pad": 1.094,
            "Other": 0.217,
        },
        "LLaVA legacy": {
            "Resize": 0.701,
            "Rescale / normalize": 0.466,
            "Format conversion": 0.181,
            "Patch / tile logic": 0.134,
            "Stack / cat / reshape / pad": 0.113,
            "Other": 0.110,
        },
        "LLaVA fast": {
            "Resize": 0.130,
            "Rescale / normalize": 0.087,
            "Format conversion": 0.000,
            "Patch / tile logic": 0.080,
            "Stack / cat / reshape / pad": 0.455,
            "Other": 0.000,
        },
    },
    "W4 A4-sized images": {
        "Qwen legacy": {
            "Resize": 2.215,
            "Rescale / normalize": 2.657,
            "Format conversion": 2.976,
            "Patch / tile logic": 0.000,
            "Stack / cat / reshape / pad": 1.953,
            "Other": 1.740,
        },
        "Qwen fast": {
            "Resize": 1.359,
            "Rescale / normalize": 0.889,
            "Format conversion": 0.000,
            "Patch / tile logic": 0.000,
            "Stack / cat / reshape / pad": 7.459,
            "Other": 1.833,
        },
        "IV25 HF legacy": {
            "Resize": 0.989,
            "Rescale / normalize": 0.054,
            "Format conversion": 0.300,
            "Patch / tile logic": 0.000,
            "Stack / cat / reshape / pad": 0.000,
            "Other": 0.031,
        },
        "IV25 HF fast": {
            "Resize": 0.629,
            "Rescale / normalize": 0.000,
            "Format conversion": 0.170,
            "Patch / tile logic": 0.000,
            "Stack / cat / reshape / pad": 0.368,
            "Other": 0.030,
        },
        "IV25 manual": {
            "Resize": 2.149,
            "Rescale / normalize": 0.047,
            "Format conversion": 0.028,
            "Patch / tile logic": 0.093,
            "Stack / cat / reshape / pad": 0.180,
            "Other": 0.000,
        },
        "LLaVA legacy": {
            "Resize": 1.759,
            "Rescale / normalize": 0.115,
            "Format conversion": 0.210,
            "Patch / tile logic": 0.227,
            "Stack / cat / reshape / pad": 0.000,
            "Other": 0.198,
        },
        "LLaVA fast": {
            "Resize": 0.275,
            "Rescale / normalize": 0.000,
            "Format conversion": 0.170,
            "Patch / tile logic": 0.052,
            "Stack / cat / reshape / pad": 0.073,
            "Other": 0.000,
        },
    },
}

memory_movement_categories = ["Format conversion", "Stack / cat / reshape / pad"]

def total_profile_time(workload, variant):
    return sum(profile_seconds[workload][variant].values())

def memory_movement_share(workload, variant):
    total = total_profile_time(workload, variant)
    movement = sum(profile_seconds[workload][variant][cat] for cat in memory_movement_categories)
    return movement / total if total else 0.0


# %% [markdown]
# ## Stacked Profile Breakdowns
# 
# These charts show the profiled time attributed to broad operation families. They are the main figures for the profiling section because they connect the benchmark results to the project argument: different models and workloads spend time in different schedules of resize, conversion, patching, and output assembly.

# %%
def plot_profile_breakdown(workload, filename):
    x = np.arange(len(variants))
    bottom = np.zeros(len(variants))

    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    for category in categories:
        values = [profile_seconds[workload][variant][category] for variant in variants]
        ax.bar(
            x,
            values,
            bottom=bottom,
            color=colors[category],
            edgecolor="white",
            linewidth=0.7,
            label=category,
        )
        bottom += values

    for xpos, total in zip(x, bottom):
        ax.annotate(
            f"{total:.2f}s",
            xy=(xpos, total),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title(f"{workload}: Profiled Runtime Breakdown", fontsize=15, pad=14)
    ax.set_ylabel("Grouped profiled time (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=25, ha="right")
    ax.legend(ncol=3, frameon=True, loc="upper left")
    ax.margins(y=0.14)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=200, bbox_inches="tight")
    return fig, ax

plot_profile_breakdown(
    "W3 randomized images",
    "w3_profile_runtime_breakdown.png",
);

plot_profile_breakdown(
    "W4 A4-sized images",
    "w4_profile_runtime_breakdown.png",
);


# %% [markdown]
# ## Memory-Movement Share
# 
# These charts collapse the grouped profile into the fraction of time spent in format conversion and output assembly/layout movement. This is the most direct visual support for a scheduling-oriented approach: high bars indicate time spent moving or re-laying out data rather than doing the semantic image operation itself.

# %%
def plot_memory_movement_share(workload, filename):
    x = np.arange(len(variants))
    shares = np.array([memory_movement_share(workload, variant) for variant in variants])

    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    bars = ax.bar(
        x,
        shares * 100,
        color="#E45756",
        edgecolor="white",
        linewidth=0.8,
    )

    for bar, share in zip(bars, shares):
        ax.annotate(
            f"{share * 100:.0f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title(f"{workload}: Time in Format Conversion + Assembly/Layout", fontsize=15, pad=14)
    ax.set_ylabel("Share of grouped profiled time (%)")
    ax.set_ylim(0, max(75, shares.max() * 115))
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=25, ha="right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=200, bbox_inches="tight")
    return fig, ax

plot_memory_movement_share(
    "W3 randomized images",
    "w3_memory_movement_share.png",
);

plot_memory_movement_share(
    "W4 A4-sized images",
    "w4_memory_movement_share.png",
);


# %% [markdown]
# ## Generated Figures
# 
# - `visualizations/figures/w3_profile_runtime_breakdown.png`
# - `visualizations/figures/w4_profile_runtime_breakdown.png`
# - `visualizations/figures/w3_memory_movement_share.png`
# - `visualizations/figures/w4_memory_movement_share.png`


