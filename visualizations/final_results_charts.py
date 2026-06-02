from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from visualizations.data import (
        get_hf_fast_memory_heatmap,
        get_hf_fast_pipeline_memory,
        get_hf_fast_pipeline_runtime,
        get_hf_fast_thread_scaling_heatmap,
        get_legacy_fast_thread_comparison,
        get_qwen_dsl_schedule_ablation,
        get_qwen_dsl_schedule_ablation_multi_thread,
        get_qwen_dsl_schedule_ablation_single_thread,
        get_qwen_multithread_headline,
        get_qwen_runtime_memory_pareto,
        get_qwen_singlethread_headline,
        get_qwen_thread_scaling,
    )
except ModuleNotFoundError:
    from data import (
        get_hf_fast_memory_heatmap,
        get_hf_fast_pipeline_memory,
        get_hf_fast_pipeline_runtime,
        get_hf_fast_thread_scaling_heatmap,
        get_legacy_fast_thread_comparison,
        get_qwen_dsl_schedule_ablation,
        get_qwen_dsl_schedule_ablation_multi_thread,
        get_qwen_dsl_schedule_ablation_single_thread,
        get_qwen_multithread_headline,
        get_qwen_runtime_memory_pareto,
        get_qwen_singlethread_headline,
        get_qwen_thread_scaling,
    )


plt.style.use("seaborn-v0_8-whitegrid")

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

COLORS = {
    "HF Legacy": "#4C78A8",
    "HF Fast": "#F58518",
    "HF Bilinear": "#E45756",
    "Hand v3": "#54A24B",
    "DSL v1": "#B279A2",
    "DSL v2": "#72B7B2",
    "DSL v3": "#2F855A",
    "Qwen2.5-VL Fast": "#4C78A8",
    "InternVL3.5 Fast": "#F58518",
    "LLaVA-NeXT Fast": "#54A24B",
    "Legacy 1T": "#4C78A8",
    "Fast 1T": "#F58518",
    "Legacy 8T": "#72B7B2",
    "Fast 8T": "#2F855A",
}


def _annotate_bars(ax, bars, fmt="{:.2f}x", fontsize=8.5):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def _style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.35)


def _plot_qwen_headline(data, title, filename):
    workloads = list(data["workloads"])
    series = data["series"]
    x = np.arange(len(workloads))
    width = 0.15
    max_value = max(max(values) for values in series.values())

    fig, ax = plt.subplots(figsize=(11.2, 6.0))
    for idx, (label, values) in enumerate(series.items()):
        offset = (idx - (len(series) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=label,
            color=COLORS[label],
            edgecolor="white",
            linewidth=0.8,
        )
        _annotate_bars(ax, bars)

    ax.axhline(1.0, color="#333333", linewidth=1.1, linestyle="--", alpha=0.8)
    ax.set_title(title, fontsize=15, pad=14)
    ax.set_ylabel("Speedup over HF Fast (x)")
    ax.set_xticks(x)
    ax.set_xticklabels(workloads)
    ax.set_ylim(0, max(1.35, max_value * 1.18))
    ax.legend(frameon=True, ncol=3)
    _style_axes(ax)

    fig.tight_layout()
    path = FIG_DIR / filename
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_qwen_singlethread_headline():
    return _plot_qwen_headline(
        get_qwen_singlethread_headline(),
        "Qwen Single-Thread Runtime Speedup vs HF Fast",
        "final_qwen_singlethread_speedup.png",
    )


def plot_qwen_multithread_headline():
    return _plot_qwen_headline(
        get_qwen_multithread_headline(),
        "Qwen Multi-Thread Runtime Speedup vs HF Fast",
        "final_qwen_multithread_speedup.png",
    )


def plot_qwen_thread_scaling():
    data = get_qwen_thread_scaling()
    workloads = list(data["workloads"])
    series = data["series"]
    x = np.arange(len(workloads))
    width = 0.23

    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    for idx, (label, values) in enumerate(series.items()):
        offset = (idx - (len(series) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=label,
            color=COLORS[label],
            edgecolor="white",
            linewidth=0.8,
        )
        _annotate_bars(ax, bars)

    ax.set_title("Qwen Thread Scaling from 1 Thread to 8 Threads", fontsize=15, pad=14)
    ax.set_ylabel("8-thread speedup over 1-thread runtime (x)")
    ax.set_xticks(x)
    ax.set_xticklabels(workloads)
    ax.set_ylim(0, 7.7)
    ax.legend(frameon=True, ncol=3)
    _style_axes(ax)

    fig.tight_layout()
    path = FIG_DIR / "final_qwen_thread_scaling.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_qwen_dsl_schedule_ablation():
    data = get_qwen_dsl_schedule_ablation()
    workloads = list(data["workloads"])
    panels = data["panels"]
    x = np.arange(len(workloads))
    width = 0.24

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.7), sharey=True)
    for ax, (panel, series) in zip(axes, panels.items()):
        for idx, (label, values) in enumerate(series.items()):
            offset = (idx - (len(series) - 1) / 2) * width
            bars = ax.bar(
                x + offset,
                values,
                width,
                label=label,
                color=COLORS[label],
                edgecolor="white",
                linewidth=0.8,
            )
            _annotate_bars(ax, bars)
        ax.set_title(panel, fontsize=13, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(workloads)
        _style_axes(ax)

    axes[0].set_ylabel("Speedup over DSL v1 (x)")
    axes[0].set_ylim(0, 8.8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("DSL Schedule Ablation: Same Qwen Algorithm, Different Schedules", fontsize=15, y=1.10)

    fig.tight_layout()
    path = FIG_DIR / "final_qwen_dsl_schedule_ablation.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_qwen_dsl_schedule_ablation_panel(data, title, filename, ylim=None):
    workloads = list(data["workloads"])
    series = data["series"]
    x = np.arange(len(workloads))
    width = 0.24
    max_value = max(max(values) for values in series.values())

    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    for idx, (label, values) in enumerate(series.items()):
        offset = (idx - (len(series) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=label,
            color=COLORS[label],
            edgecolor="white",
            linewidth=0.8,
        )
        _annotate_bars(ax, bars)

    ax.set_title(title, fontsize=15, pad=14)
    ax.set_ylabel("Speedup over DSL v1 (x)")
    ax.set_xticks(x)
    ax.set_xticklabels(workloads)
    ax.set_ylim(0, ylim or max_value * 1.18)
    ax.legend(frameon=True, ncol=3)
    _style_axes(ax)

    fig.tight_layout()
    path = FIG_DIR / filename
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_qwen_dsl_schedule_ablation_single_thread():
    return _plot_qwen_dsl_schedule_ablation_panel(
        get_qwen_dsl_schedule_ablation_single_thread(),
        "Qwen DSL Schedule Ablation, Single Thread",
        "final_qwen_dsl_schedule_ablation_single_thread.png",
        ylim=1.45,
    )


def plot_qwen_dsl_schedule_ablation_multi_thread():
    return _plot_qwen_dsl_schedule_ablation_panel(
        get_qwen_dsl_schedule_ablation_multi_thread(),
        "Qwen DSL Schedule Ablation, 8 Threads",
        "final_qwen_dsl_schedule_ablation_multi_thread.png",
        ylim=8.9,
    )


def plot_qwen_runtime_memory_pareto():
    points_by_workload = get_qwen_runtime_memory_pareto()
    markers = {
        "hf_legacy": "o",
        "hf_fast": "s",
        "hf_bilinear": "D",
        "dsl_v1": "^",
        "dsl_v2": "P",
        "v3": "*",
        "dsl_v3": "X",
    }
    offsets = {
        "HF Legacy": (5, 5),
        "HF Fast": (5, 5),
        "HF Bilinear": (5, -11),
        "DSL v1": (5, 5),
        "DSL v2": (5, -11),
        "Hand v3": (5, 5),
        "DSL v3": (5, -11),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.0))
    for ax, (workload, points) in zip(axes, points_by_workload.items()):
        for point in points:
            label = str(point["label"])
            model = str(point["model"])
            size = 145 if label in {"Hand v3", "DSL v3"} else 82
            ax.scatter(
                point["runtime"],
                point["memory"],
                s=size,
                marker=markers[model],
                color=COLORS[label],
                edgecolor="white",
                linewidth=0.9,
                label=label,
                zorder=3 if label in {"Hand v3", "DSL v3"} else 2,
            )
            ax.annotate(
                label,
                xy=(point["runtime"], point["memory"]),
                xytext=offsets[label],
                textcoords="offset points",
                fontsize=8.3,
            )
        ax.set_title(workload, fontsize=13, pad=10)
        ax.set_xlabel("Median runtime (ms/img)")
        ax.set_ylabel("Peak / output memory (x)")
        ax.set_ylim(0.82, 4.15)
        _style_axes(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=7, frameon=True, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Qwen Multi-Thread Runtime/Memory Pareto Frontier", fontsize=15, y=1.10)
    fig.tight_layout()
    path = FIG_DIR / "final_qwen_runtime_memory_pareto.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_heatmap(data, title, colorbar_label, filename, cmap, vmin=None, vmax=None):
    workloads = list(data["workloads"])
    models = list(data["models"])
    values = np.array(data["values"], dtype=float)

    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=15, pad=14)
    ax.set_xticks(np.arange(len(workloads)))
    ax.set_xticklabels(workloads)
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels(models)

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.text(
                col,
                row,
                f"{values[row, col]:.2f}x",
                ha="center",
                va="center",
                color="#111111",
                fontsize=11,
                fontweight="bold",
            )

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label)
    ax.grid(False)
    fig.tight_layout()
    path = FIG_DIR / filename
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_grouped_bar_series(
    groups,
    series,
    title,
    ylabel,
    filename,
    value_fmt="{:.1f}",
    legend_ncol=3,
    ylim=None,
):
    groups = list(groups)
    x = np.arange(len(groups))
    width = min(0.8 / len(series), 0.22)
    max_value = max(max(values) for values in series.values())

    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    for idx, (label, values) in enumerate(series.items()):
        offset = (idx - (len(series) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=label,
            color=COLORS[label],
            edgecolor="white",
            linewidth=0.8,
        )
        _annotate_bars(ax, bars, fmt=value_fmt, fontsize=8.2)

    ax.set_title(title, fontsize=15, pad=14)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, ylim or max_value * 1.18)
    ax.legend(frameon=True, ncol=legend_ncol)
    _style_axes(ax)

    fig.tight_layout()
    path = FIG_DIR / filename
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_hf_fast_pipeline_runtime_multithread():
    data = get_hf_fast_pipeline_runtime("multi")
    return _plot_grouped_bar_series(
        data["workloads"],
        data["series"],
        "HF Fast Multi-Thread Runtime by Model Pipeline",
        "Median runtime (ms/img)",
        "final_hf_fast_pipeline_runtime_multithread.png",
        value_fmt="{:.1f}",
        legend_ncol=3,
    )


def plot_hf_fast_pipeline_memory_multithread():
    data = get_hf_fast_pipeline_memory("multi")
    return _plot_grouped_bar_series(
        data["workloads"],
        data["series"],
        "HF Fast Multi-Thread Memory by Model Pipeline",
        "Peak / output memory (x)",
        "final_hf_fast_pipeline_memory_multithread.png",
        value_fmt="{:.2f}x",
        legend_ncol=3,
        ylim=4.15,
    )


def plot_workload_runtime_legacy_fast_threads(workload):
    data = get_legacy_fast_thread_comparison(workload, metric="runtime")
    return _plot_grouped_bar_series(
        data["models"],
        data["series"],
        f"{workload} Runtime: Legacy/Fast, Single/8 Threads",
        "Median runtime (ms/img)",
        f"final_{workload.lower()}_runtime_legacy_fast_threads.png",
        value_fmt="{:.1f}",
        legend_ncol=4,
    )


def plot_workload_memory_legacy_fast_threads(workload):
    data = get_legacy_fast_thread_comparison(workload, metric="memory")
    return _plot_grouped_bar_series(
        data["models"],
        data["series"],
        f"{workload} Memory: Legacy/Fast, Single/8 Threads",
        "Peak / output memory (x)",
        f"final_{workload.lower()}_memory_legacy_fast_threads.png",
        value_fmt="{:.2f}x",
        legend_ncol=4,
    )


def plot_hf_fast_memory_heatmap():
    return _plot_heatmap(
        get_hf_fast_memory_heatmap(),
        "HF Fast Multi-Thread Peak Memory Overhead",
        "Peak / output memory (x)",
        "final_hf_fast_memory_overhead.png",
        cmap="YlOrRd",
        vmin=1.0,
        vmax=4.0,
    )


def plot_hf_fast_thread_scaling_heatmap():
    return _plot_heatmap(
        get_hf_fast_thread_scaling_heatmap(),
        "HF Fast Thread Scaling Across Model Families",
        "8-thread speedup over 1-thread runtime (x)",
        "final_hf_fast_thread_scaling.png",
        cmap="viridis",
        vmin=1.0,
        vmax=4.0,
    )


def main():
    paths = [
        plot_qwen_singlethread_headline(),
        plot_qwen_multithread_headline(),
        plot_qwen_thread_scaling(),
        plot_qwen_dsl_schedule_ablation_single_thread(),
        plot_qwen_dsl_schedule_ablation_multi_thread(),
        plot_qwen_runtime_memory_pareto(),
        plot_hf_fast_pipeline_runtime_multithread(),
        plot_hf_fast_pipeline_memory_multithread(),
        *[plot_workload_runtime_legacy_fast_threads(workload) for workload in ("W2", "W3", "W4")],
        *[plot_workload_memory_legacy_fast_threads(workload) for workload in ("W2", "W3", "W4")],
        plot_hf_fast_memory_heatmap(),
        plot_hf_fast_thread_scaling_heatmap(),
    ]
    for path in paths:
        print(path.relative_to(SCRIPT_DIR.parent))


if __name__ == "__main__":
    main()
