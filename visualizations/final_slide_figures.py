from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np

try:
    from visualizations.data import (
        get_final_selected_grouped_data,
        get_llava_singlethread_memory_floor_summary,
        get_qwen_w4_schedule_axes,
        get_qwen_w4_thread_scaling,
    )
except ModuleNotFoundError:
    from data import (
        get_final_selected_grouped_data,
        get_llava_singlethread_memory_floor_summary,
        get_qwen_w4_schedule_axes,
        get_qwen_w4_thread_scaling,
    )


plt.style.use("seaborn-v0_8-whitegrid")

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

COLORS = {
    "HF Legacy": "#E6194B",
    "HF Fast": "#F58231",
    "HF Bilinear": "#4363D8",
    "DSL v1": "#911EB4",
    "DSL v2": "#42D4F4",
    "DSL v3": "#3CB44B",
    "Memory": "#FF6F61",
}


def _style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.28)


def _annotate_bars(ax, bars, fmt="{:.1f}", fontsize=9, dy=3):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, dy),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def _save(fig, filename):
    path = FIG_DIR / filename
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_memory_floor_panel(
    ax,
    family,
    thread,
    title,
    dsl_legend_label,
    legend_columns,
    show_ylabel=True,
):
    data = get_final_selected_grouped_data(
        family,
        thread,
        "memory",
        ("hf_legacy", "hf_fast", "hf_bilinear", "dsl_v3"),
    )
    workloads = list(data["workloads"])
    series = data["series"]
    x = np.arange(len(workloads))
    width = 0.18

    for idx, (label, values) in enumerate(series.items()):
        offset = (idx - (len(series) - 1) / 2) * width
        legend_label = dsl_legend_label if label == "DSL v3" else label
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=legend_label,
            color=COLORS[label],
            edgecolor="white",
            linewidth=0.9,
        )
        if label == "DSL v3":
            _annotate_bars(ax, bars, "{:.2f}x", fontsize=9)

    ax.axhline(1.0, color="#111827", linestyle=(0, (4, 3)), linewidth=1.2, alpha=0.8)
    ax.text(
        1.005,
        1.04,
        "output floor",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=10,
        clip_on=False,
    )
    ax.set_title(title, fontsize=14, pad=12)
    if show_ylabel:
        ax.set_ylabel("Peak / Output RSS (x)")
    ax.set_xticks(x)
    ax.set_xticklabels(workloads)
    ax.set_ylim(0, 4.35)
    ax.legend(frameon=True, ncol=legend_columns, loc="upper left", fontsize=9)
    _style_axes(ax)


def plot_slide_5_qwen_memory_floor():
    fig, ax = plt.subplots(figsize=(9.4, 5.7))
    _plot_memory_floor_panel(
        ax,
        "qwen",
        "single",
        "Qwen 1 Thread: Peak Memory Reaches the Output Floor",
        "DSL v3 (Qwen)",
        4,
    )
    fig.tight_layout()
    return _save(fig, "slide_5_qwen_memory_floor.png")


def plot_slide_5_llava_singlethread_memory_floor():
    fig, ax = plt.subplots(figsize=(9.4, 5.7))
    _plot_memory_floor_panel(
        ax,
        "llava",
        "single",
        "LLaVA-NeXT 1 Thread: Peak Memory Reaches the Output Floor",
        "DSL v3 (LLaVA)",
        4,
    )
    fig.tight_layout()
    return _save(fig, "slide_5_llava_singlethread_memory_floor.png")


def plot_slide_5_combined_memory_floor():
    fig, axes = plt.subplots(1, 2, figsize=(15.4, 5.8), sharey=True)
    _plot_memory_floor_panel(
        axes[0],
        "qwen",
        "single",
        "Qwen",
        "DSL v3 (Qwen)",
        2,
    )
    _plot_memory_floor_panel(
        axes[1],
        "llava",
        "single",
        "LLaVA-NeXT",
        "DSL v3 (LLaVA)",
        2,
        show_ylabel=False,
    )
    fig.suptitle("1 Thread Peak Memory Reaches the Output Floor", fontsize=16, y=1.02)
    fig.tight_layout()
    return _save(fig, "slide_5_qwen_llava_memory_floor.png")


def plot_slide_6_qwen_w4_thread_scaling():
    data = get_qwen_w4_thread_scaling()
    threads = list(data["threads"])
    x = np.arange(len(threads))
    series = data["series"]
    speedups = data["speedups"]

    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    ax.axvspan(1.05, 1.95, color="#FDE68A", alpha=0.28, zorder=0)
    ax.text(1.5, 188, "crossover", ha="center", va="center", fontsize=10, color="#78350F")

    for label, values in series.items():
        ax.plot(
            x,
            values,
            marker="o",
            markersize=8,
            linewidth=2.7,
            label=label,
            color=COLORS[label],
        )
        ax.annotate(
            f"{values[0]:.0f} ms",
            xy=(x[0], values[0]),
            xytext=(-14, 10),
            textcoords="offset points",
            ha="right",
            fontsize=9,
            color=COLORS[label],
        )
        ax.annotate(
            f"{values[-1]:.0f} ms",
            xy=(x[-1], values[-1]),
            xytext=(12, -3),
            textcoords="offset points",
            ha="left",
            fontsize=9,
            color=COLORS[label],
        )

    ax.text(
        0.25,
        155,
        f"HF scales {speedups['HF Bilinear']:.1f}x",
        color=COLORS["HF Bilinear"],
        fontsize=10,
    )
    ax.text(
        1.28,
        128,
        f"DSL scales {speedups['DSL v3']:.1f}x",
        color=COLORS["DSL v3"],
        fontsize=10,
    )
    ax.set_title("Qwen W4 Runtime Scaling: DSL Crosses HF at 8 Threads", fontsize=15, pad=14)
    ax.set_ylabel("Median runtime (ms/img, log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(threads)
    ax.set_yscale("log")
    ax.set_ylim(115, 1150)
    ax.set_yticks([140, 200, 300, 500, 800, 1000])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.legend(frameon=True, loc="upper right")
    _style_axes(ax)
    fig.tight_layout()
    return _save(fig, "slide_6_qwen_w4_thread_scaling.png")


def plot_slide_7_qwen_schedule_axes():
    data = get_qwen_w4_schedule_axes()
    variants = list(data["variant_labels"])
    x = np.arange(len(variants))

    fig, ax_fusion = plt.subplots(figsize=(9.8, 5.8))

    ax_fusion.plot(
        x,
        data["fusion_runtime"],
        marker="o",
        markersize=8,
        linewidth=2.8,
        color="#4363D8",
        label="Runtime",
    )
    for xi, value in zip(x, data["fusion_runtime"]):
        ax_fusion.annotate(
            f"{value:.0f}",
            xy=(xi, value),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#4363D8",
        )
    ax_fusion.set_title("Qwen W4 Fusion Axis: Runtime Moves Little, Memory Drops at v3", fontsize=15, pad=14)
    ax_fusion.set_ylabel("Runtime (ms/img)")
    ax_fusion.set_xticks(x)
    ax_fusion.set_xticklabels(variants)
    ax_fusion.set_xlabel("Schedule variant")
    ax_fusion.set_ylim(125, max(data["fusion_runtime"]) * 1.14)
    _style_axes(ax_fusion)

    ax_memory = ax_fusion.twinx()
    ax_memory.plot(
        x,
        data["fusion_memory"],
        marker="o",
        markersize=7,
        linewidth=2.4,
        color=COLORS["Memory"],
        label="Peak / Output RSS",
    )
    ax_memory.set_ylabel("Peak / Output RSS (x)")
    ax_memory.set_ylim(0.95, max(data["fusion_memory"]) * 1.08)
    ax_memory.spines["top"].set_visible(False)
    ax_memory.grid(False)
    for xi, value in zip(x, data["fusion_memory"]):
        ax_memory.annotate(
            f"{value:.2f}x",
            xy=(xi, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=COLORS["Memory"],
        )

    ax_fusion.text(
        0.02,
        0.93,
        f"runtime spread {data['runtime_spread']:.2f}x",
        transform=ax_fusion.transAxes,
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#D1D5DB"},
    )
    ax_memory.annotate(
        "memory drops at v3",
        xy=(2, data["fusion_memory"][-1]),
        xytext=(1.2, data["fusion_memory"][0] * 0.86),
        arrowprops={"arrowstyle": "->", "color": COLORS["Memory"], "lw": 1.4},
        fontsize=10,
        color=COLORS["Memory"],
    )
    handles_1, labels_1 = ax_fusion.get_legend_handles_labels()
    handles_2, labels_2 = ax_memory.get_legend_handles_labels()
    ax_fusion.legend(
        handles_1 + handles_2,
        labels_1 + labels_2,
        frameon=True,
        loc="lower left",
    )
    fig.tight_layout()
    return _save(fig, "slide_7_qwen_schedule_axes.png")


def plot_slide_8_llava_schedule_flat_runtime():
    data = get_final_selected_grouped_data(
        "llava",
        "multi",
        "runtime",
        ("hf_bilinear", "dsl_v1", "dsl_v2", "dsl_v3"),
    )
    memory_summary = get_llava_singlethread_memory_floor_summary()
    workloads = list(data["workloads"])
    raw_series = data["series"]
    baselines = raw_series["HF Bilinear"]
    series = {
        label: [
            value / baseline
            for value, baseline in zip(values, baselines)
        ]
        for label, values in raw_series.items()
    }
    x = np.arange(len(workloads))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    for idx, (label, values) in enumerate(series.items()):
        offset = (idx - (len(series) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=label,
            color=COLORS[label],
            edgecolor="white",
            linewidth=0.9,
        )
        if label in {"HF Bilinear", "DSL v3"}:
            _annotate_bars(ax, bars, "{:.2f}x", fontsize=8)

    ax.text(
        0.43,
        0.91,
        "DSL v1-v3 nearly identical within each workload",
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#D1D5DB"},
    )
    ax.annotate(
        "resize-bound:\nDSL is ~1.7x HF",
        xy=(x[2] + 1.5 * width, series["DSL v3"][2]),
        xytext=(x[2] - 0.72, 1.52),
        arrowprops={"arrowstyle": "->", "color": COLORS["HF Bilinear"], "lw": 1.3},
        fontsize=10,
        color=COLORS["HF Bilinear"],
    )
    memory_text = (
        "Single-thread memory floor\n"
        f"DSL: {memory_summary['dsl_min']:.2f}-{memory_summary['dsl_max']:.2f}x\n"
        f"HF: {memory_summary['hf_min']:.2f}-{memory_summary['hf_max']:.2f}x"
    )
    ax.text(
        0.22,
        0.72,
        memory_text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#D1D5DB"},
    )

    ax.set_title("LLaVA-NeXT 8-Thread Runtime Normalized to HF Bilinear", fontsize=15, pad=14)
    ax.set_ylabel("Runtime / HF Bilinear (x)")
    ax.set_xticks(x)
    ax.set_xticklabels(workloads)
    ax.set_ylim(0, 1.9)
    ax.legend(frameon=True, ncol=4, loc="upper left")
    _style_axes(ax)
    fig.tight_layout()
    return _save(fig, "slide_8_llava_schedule_flat_runtime.png")


def main():
    paths = [
        plot_slide_5_qwen_memory_floor(),
        plot_slide_5_llava_singlethread_memory_floor(),
        plot_slide_5_combined_memory_floor(),
        plot_slide_6_qwen_w4_thread_scaling(),
        plot_slide_7_qwen_schedule_axes(),
        plot_slide_8_llava_schedule_flat_runtime(),
    ]
    for path in paths:
        print(path.relative_to(SCRIPT_DIR.parent))


if __name__ == "__main__":
    main()
