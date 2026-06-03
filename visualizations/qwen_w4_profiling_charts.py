from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from visualizations.data import get_qwen_w4_profiling_data
except ModuleNotFoundError:
    from data import get_qwen_w4_profiling_data


plt.style.use("seaborn-v0_8-whitegrid")

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

PROFILE_COLORS = {
    "Input conversion": "#B279A2",
    "HF resize": "#F2B134",
    "HF rescale / normalize": "#E45756",
    "HF patch layout": "#4C78A8",
    "HF output assembly": "#72B7B2",
    "DSL processing call": "#F58518",
    "Other": "#BAB0AC",
}


def _style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.35)


def _plot_runtime_breakdown():
    data = get_qwen_w4_profiling_data()
    variants = list(data["comparison_variants"])
    categories = data["comparison_categories"]
    profile_totals = np.array(data["comparison_profile_totals_ms"], dtype=np.float64)
    geometry = data["geometry"]

    x = np.arange(len(variants))
    bottom = np.zeros(len(variants), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(12.6, 7.0))

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
            if value >= 45:
                pct = value / profile_totals[idx] * 100
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    base + value / 2,
                    f"{value:.0f} ms\n{pct:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8.6,
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
        "Qwen2.5-VL W4 Profiling: Resize Is Largest, but Layout and Assembly Also Matter",
        fontsize=15,
        pad=28,
    )
    ax.text(
        0.5,
        1.015,
        (
            f"{geometry['images']} near-native-resolution images, "
            f"{geometry['patches_total']:,} patches, {geometry['output_mb']:.0f} MB output; "
            "v1 -> v2 saves 142 ms, v2 -> v3 saves 7 ms"
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
    ax.set_ylim(0, profile_totals.max() * 1.25)
    ax.legend(frameon=True, ncol=2, loc="upper right")
    _style_axes(ax)

    fig.text(
        0.5,
        0.015,
        (
            "HF uses non-overlapping cProfile call groups. DSL Numba processing is opaque to "
            "cProfile; the non-parallel v1 diagnostic stage breakdown is intentionally excluded."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))

    path = FIG_DIR / "final_qwen_w4_profile_breakdown.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    path = _plot_runtime_breakdown()
    print(path.relative_to(SCRIPT_DIR.parent))


if __name__ == "__main__":
    main()
