from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

try:
    from visualizations.data import (
        get_llava_w4_profiling_data,
        get_qwen_w4_profiling_data,
    )
except ModuleNotFoundError:
    from data import get_llava_w4_profiling_data, get_qwen_w4_profiling_data


plt.style.use("seaborn-v0_8-whitegrid")

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

COLORS = {
    "Input conversion": "#B279A2",
    "HF resize": "#F2B134",
    "Rescale / normalize": "#E45756",
    "Patch layout": "#4C78A8",
    "Output assembly": "#72B7B2",
    "Tiling / assembly / setup": "#59A14F",
    "DSL v3 fully fused processing": "#F58518",
    "Other": "#BAB0AC",
}


def _style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.35)


def _plot_panel(ax, title, subtitle, categories, totals, comparison_text):
    variants = ("HF Bilinear", "DSL v3")
    x = np.arange(len(variants))
    bottom = np.zeros(2, dtype=np.float64)

    for category, values_raw in categories.items():
        values = np.array(values_raw, dtype=np.float64)
        bars = ax.bar(
            x,
            values,
            bottom=bottom,
            width=0.62,
            color=COLORS[category],
            edgecolor="white",
            linewidth=0.8,
        )
        for idx, (bar, value, base) in enumerate(zip(bars, values, bottom)):
            if value >= totals[idx] * 0.07:
                pct = value / totals[idx] * 100
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    base + value / 2,
                    f"{value:.0f} ms\n{pct:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="white" if category != "Other" else "#333333",
                    fontweight="bold",
                )
        bottom += values

    for xpos, total in zip(x, totals):
        ax.annotate(
            f"{total:.0f} ms",
            xy=(xpos, total),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    ax.set_title(title, fontsize=13, pad=20)
    ax.text(
        0.5,
        1.015,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    ax.text(
        0.5,
        0.94,
        comparison_text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9.5,
        color="#333333",
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.set_ylim(0, max(totals) * 1.22)
    _style_axes(ax)


def _plot_combined_breakdown():
    qwen = get_qwen_w4_profiling_data()
    llava = get_llava_w4_profiling_data()

    qwen_source = qwen["comparison_categories"]
    qwen_categories = {
        "Input conversion": (qwen_source["Input conversion"][0], qwen_source["Input conversion"][3]),
        "HF resize": (qwen_source["HF resize"][0], 0.0),
        "Rescale / normalize": (qwen_source["HF rescale / normalize"][0], 0.0),
        "Patch layout": (qwen_source["HF patch layout"][0], 0.0),
        "Output assembly": (qwen_source["HF output assembly"][0], 0.0),
        "DSL v3 fully fused processing": (0.0, qwen_source["DSL processing call"][3]),
        "Other": (qwen_source["Other"][0], 0.0),
    }
    qwen_totals = np.array(
        (qwen["comparison_profile_totals_ms"][0], qwen["comparison_profile_totals_ms"][3]),
        dtype=np.float64,
    )

    llava_source = llava["comparison_categories"]
    llava_categories = {
        "Input conversion": (llava_source["Input conversion"][0], llava_source["Input conversion"][3]),
        "HF resize": (llava_source["HF resize"][0], 0.0),
        "Rescale / normalize": (llava_source["Rescale / normalize"][0], 0.0),
        "Tiling / assembly / setup": (
            llava_source["Tiling / assembly / setup"][0],
            llava_source["Tiling / assembly / setup"][3],
        ),
        "DSL v3 fully fused processing": (0.0, llava_source["DSL tile-processing call"][3]),
        "Other": (llava_source["Other"][0], 0.0),
    }
    llava_totals = np.array(
        (llava["comparison_profile_totals_ms"][0], llava["comparison_profile_totals_ms"][3]),
        dtype=np.float64,
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.8))
    _plot_panel(
        axes[0],
        "Qwen2.5-VL",
        "Near-native resize, 356,000 patches, 1675 MB output",
        qwen_categories,
        qwen_totals,
        "DSL v3 is 16% faster",
    )
    _plot_panel(
        axes[1],
        "LLaVA-NeXT",
        "AnyRes downsampling, 40 tiles, 54 MB output",
        llava_categories,
        llava_totals,
        "DSL v3 is 59% slower",
    )
    axes[0].set_ylabel("Profiled time (ms/batch)")

    legend_order = [
        "Input conversion",
        "HF resize",
        "Rescale / normalize",
        "Patch layout",
        "Output assembly",
        "Tiling / assembly / setup",
        "DSL v3 fully fused processing",
        "Other",
    ]
    handles = [Patch(facecolor=COLORS[label], edgecolor="white", label=label) for label in legend_order]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=True,
        bbox_to_anchor=(0.5, 0.055),
        fontsize=9,
    )

    fig.suptitle(
        "W4 Profiling: HF Bilinear vs Fully Fused DSL v3",
        fontsize=16,
        y=0.98,
    )
    fig.text(
        0.5,
        0.015,
        (
            "Panels use independent y-axis scales. HF categories are measured call groups; "
            "DSL v3 compiled processing remains grouped because Python profilers cannot see "
            "inside Numba kernels."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.16, 1, 0.94), w_pad=3.5)

    path = FIG_DIR / "final_qwen_llava_w4_profile_comparison.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    path = _plot_combined_breakdown()
    print(path.relative_to(SCRIPT_DIR.parent))


if __name__ == "__main__":
    main()
