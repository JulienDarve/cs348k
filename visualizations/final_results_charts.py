from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from visualizations.data import (
        get_final_grouped_bar_data,
        get_final_pareto_points,
        get_pareto_frontier,
    )
except ModuleNotFoundError:
    from data import (
        get_final_grouped_bar_data,
        get_final_pareto_points,
        get_pareto_frontier,
    )


plt.style.use("seaborn-v0_8-whitegrid")

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

COLORS = {
    "HF Legacy": "#4C78A8",
    "HF Fast": "#F58518",
    "HF Bilinear": "#E45756",
    "DSL v1": "#B279A2",
    "DSL v2": "#72B7B2",
    "DSL v3": "#2F855A",
}

MARKERS = {
    "HF Legacy": "o",
    "HF Fast": "s",
    "HF Bilinear": "D",
    "DSL v1": "^",
    "DSL v2": "P",
    "DSL v3": "X",
}

FAMILY_LABELS = {
    "qwen": "Qwen",
    "llava": "LLaVA-NeXT",
}

THREAD_LABELS = {
    "multi": "8 Threads",
    "single": "1 Thread",
}

THREAD_SLUGS = {
    "multi": "multithread",
    "single": "singlethread",
}

METRIC_LABELS = {
    "runtime": "Runtime",
    "memory": "Peak Memory",
}

Y_LABELS = {
    "runtime": "Median runtime (ms/image)",
    "memory": "Peak / output RSS memory (x)",
}

VALUE_FORMATS = {
    "runtime": "{:.1f}",
    "memory": "{:.2f}x",
}


def _style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.35)


def _annotate_bars(ax, bars, fmt, fontsize=7.6):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=90,
        )


def _plot_grouped_bars(family, thread, metric):
    data = get_final_grouped_bar_data(family, thread, metric)
    workloads = list(data["workloads"])
    series = data["series"]
    x = np.arange(len(workloads))
    width = min(0.82 / len(series), 0.14)
    max_value = max(max(values) for values in series.values())

    fig, ax = plt.subplots(figsize=(11.4, 6.2))
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
        _annotate_bars(ax, bars, VALUE_FORMATS[metric])

    title = f"{FAMILY_LABELS[family]} {THREAD_LABELS[thread]} {METRIC_LABELS[metric]}"
    ax.set_title(title, fontsize=15, pad=14)
    ax.set_ylabel(Y_LABELS[metric])
    ax.set_xticks(x)
    ax.set_xticklabels(workloads)
    ax.set_ylim(0, max_value * (1.22 if metric == "runtime" else 1.18))
    ax.legend(frameon=True, ncol=3)
    _style_axes(ax)

    fig.tight_layout()
    path = FIG_DIR / f"final_{family}_{THREAD_SLUGS[thread]}_{metric}.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_pareto():
    points_by_panel = get_final_pareto_points()
    row_keys = (
        ("qwen", "multi"),
        ("llava", "multi"),
        ("qwen", "single"),
        ("llava", "single"),
    )
    workloads = ("W2", "W3", "W4")

    fig, axes = plt.subplots(len(row_keys), len(workloads), figsize=(17.0, 15.4))
    for row_idx, (family, thread) in enumerate(row_keys):
        for col_idx, workload in enumerate(workloads):
            ax = axes[row_idx, col_idx]
            points = points_by_panel[(family, thread, workload)]
            frontier = get_pareto_frontier(points)

            for point in points:
                label = str(point["label"])
                ax.scatter(
                    point["runtime"],
                    point["memory"],
                    s=82,
                    marker=MARKERS[label],
                    color=COLORS[label],
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=3,
                )

            ax.plot(
                [point["runtime"] for point in frontier],
                [point["memory"] for point in frontier],
                color="#111111",
                linewidth=1.3,
                linestyle="--",
                alpha=0.75,
                zorder=2,
            )
            ax.set_title(f"{FAMILY_LABELS[family]} {THREAD_LABELS[thread]} {workload}", fontsize=11, pad=8)
            ax.set_xlabel("Runtime (ms/image)")
            ax.set_ylabel("Peak / output memory (x)")
            _style_axes(ax)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker=MARKERS[label],
            color="none",
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=8,
            label=label,
        )
        for label, color in COLORS.items()
    ]
    handles.append(
        plt.Line2D([0], [0], color="#111111", linestyle="--", linewidth=1.3, label="Pareto frontier")
    )
    fig.legend(handles=handles, loc="upper center", ncol=7, frameon=True, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Runtime and Peak-Memory Pareto Frontiers", fontsize=16, y=1.035)
    fig.tight_layout()

    path = FIG_DIR / "final_runtime_memory_pareto.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    paths = []
    for thread in ("multi", "single"):
        for metric in ("runtime", "memory"):
            for family in ("qwen", "llava"):
                paths.append(_plot_grouped_bars(family, thread, metric))
    paths.append(_plot_pareto())

    for path in paths:
        print(path.relative_to(SCRIPT_DIR.parent))


if __name__ == "__main__":
    main()
