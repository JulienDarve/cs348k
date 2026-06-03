from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from visualizations.data import get_llava_w4_profiling_data
except ModuleNotFoundError:
    from data import get_llava_w4_profiling_data


plt.style.use("seaborn-v0_8-whitegrid")

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

PROFILE_COLORS = {
    "Input conversion": "#B279A2",
    "HF resize": "#F2B134",
    "DSL tile-processing call": "#F58518",
    "Rescale / normalize": "#E45756",
    "Tiling / assembly / setup": "#72B7B2",
    "Other": "#BAB0AC",
}

WORK_COLORS = {
    "Sampled output pixels": "#4C78A8",
    "Padded output pixels": "#BAB0AC",
    "Thumbnail source taps": "#4C78A8",
    "Grid source taps": "#F58518",
}


def _style_axes(ax, grid_axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, alpha=0.35)


def _plot_runtime_breakdown():
    data = get_llava_w4_profiling_data()
    variants = list(data["comparison_variants"])
    categories = data["comparison_categories"]
    profile_totals = np.array(data["comparison_profile_totals_ms"], dtype=np.float64)

    x = np.arange(len(variants))
    bottom = np.zeros(len(variants), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(11.8, 6.5))

    for category, values_raw in categories.items():
        values = np.array(values_raw, dtype=np.float64)
        bars = ax.bar(
            x,
            values,
            bottom=bottom,
            width=0.64,
            color=PROFILE_COLORS[category],
            edgecolor="white",
            linewidth=0.8,
            label=category,
        )
        for idx, (bar, value, base) in enumerate(zip(bars, values, bottom)):
            if value >= 15:
                pct = value / profile_totals[idx] * 100
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    base + value / 2,
                    f"{value:.0f} ms\n{pct:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8.8,
                    color="white" if category != "Other" else "#333333",
                    fontweight="bold",
                )
        bottom += values

    for xpos, total in zip(x, profile_totals):
        ax.annotate(
            f"profile total {total:.0f} ms",
            xy=(xpos, total),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title(
        "LLaVA-NeXT W4 Profiling: DSL Runtime Is Dominated by Tile Processing",
        fontsize=15,
        pad=28,
    )
    ax.text(
        0.5,
        1.015,
        (
            "The DSL profiler groups each schedule's Numba tile-processing call: "
            "v1 is staged, v2 fuses only the pointwise tail, and v3 is fully fused."
        ),
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#555555",
    )
    ax.set_ylabel("Profiled time (ms/batch)")
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.set_ylim(0, profile_totals.max() * 1.24)
    ax.legend(frameon=True, ncol=2, loc="upper left")
    _style_axes(ax, "y")

    fig.text(
        0.5,
        0.015,
        (
            "DSL bars have fewer visible colors because thumbnail/grid template calls are timed "
            "as a whole, not because v1 or v2 fully fuse resize; setup is only about 0.4 ms."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))

    path = FIG_DIR / "final_llava_w4_profile_breakdown.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def _draw_stacked_horizontal(ax, segments, colors, total_label, xlabel):
    left = 0.0
    for label, value in segments:
        ax.barh(
            [0],
            [value],
            left=[left],
            height=0.42,
            color=colors[label],
            edgecolor="white",
            linewidth=0.8,
        )
        ax.text(
            left + value / 2,
            0,
            f"{label}\n{value:.2f}M",
            ha="center",
            va="center",
            fontsize=8.8,
            color="white" if colors[label] != "#BAB0AC" else "#333333",
            fontweight="bold",
        )
        left += value

    ax.annotate(
        total_label,
        xy=(left, 0),
        xytext=(0, 22),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
    ax.set_xlim(0, left * 1.08)
    ax.set_ylim(-0.55, 0.7)
    ax.set_yticks([])
    ax.set_xlabel(xlabel)
    _style_axes(ax, "x")


def _plot_sampling_work():
    data = get_llava_w4_profiling_data()
    geometry = data["geometry"]

    sampled_m = geometry["sampled_output_pixels"] / 1e6
    padded_m = geometry["padded_output_pixels"] / 1e6
    thumbnail_taps_m = geometry["thumbnail_taps"] / 1e6
    grid_taps_m = geometry["grid_taps"] / 1e6

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.3))

    _draw_stacked_horizontal(
        axes[0],
        [
            ("Sampled output pixels", sampled_m),
            ("Padded output pixels", padded_m),
        ],
        WORK_COLORS,
        f"{geometry['output_pixels'] / 1e6:.2f}M output pixels",
        "Output tile pixels (millions)",
    )
    axes[0].set_title("Output Pixel Destinations", fontsize=12, pad=12)

    _draw_stacked_horizontal(
        axes[1],
        [
            ("Thumbnail source taps", thumbnail_taps_m),
            ("Grid source taps", grid_taps_m),
        ],
        WORK_COLORS,
        f"{geometry['source_taps'] / 1e6:.2f}M estimated source taps",
        "Estimated source taps (millions)",
    )
    axes[1].set_title("Scalar Sampling Work", fontsize=12, pad=12)

    input_h, input_w = geometry["input_hw"]
    fig.suptitle("LLaVA-NeXT W4: Small Output, Expensive Downsampling", fontsize=15, y=1.02)
    fig.text(
        0.5,
        0.91,
        (
            f"{geometry['images']} x {input_h} x {input_w} images -> "
            f"{geometry['tiles_total']} tiles, {geometry['output_mb']:.2f} MB output, "
            f"{geometry['avg_taps_per_sample']:.1f} source taps per sampled output pixel"
        ),
        ha="center",
        va="center",
        fontsize=10,
        color="#444444",
    )
    fig.text(
        0.5,
        0.015,
        (
            "Analytic counts use the exact AnyRes tile descriptors and bilinear sampling support "
            "used by the DSL kernels."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.88), w_pad=3.2)

    path = FIG_DIR / "final_llava_w4_sampling_work.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    paths = [
        _plot_runtime_breakdown(),
        _plot_sampling_work(),
    ]
    for path in paths:
        print(path.relative_to(SCRIPT_DIR.parent))


if __name__ == "__main__":
    main()
