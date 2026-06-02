"""Algorithm filter comparison: our kernels vs HF slow/fast × BICUBIC/BILINEAR.

Informational only — no pass/fail gate.  Run from repo root:
    python tests/test_filter_comparison.py

Prints a max_diff table for every (processor, resample) combination on both
noise and smooth images for LLaVA-NeXT (DSL-v1) and Qwen2.5-VL (hand-fused
v1 kernel).  The goal is to identify which HF algorithm our kernels match and
to quantify how large the algorithmic gap is for the others.

Our kernels implement an antialiased BILINEAR (scaled triangle) filter, i.e.
the same algorithm as PIL Image.BILINEAR when downsampling.  HF defaults
(from preprocessor_config.json) are BICUBIC for both models.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers.image_utils import PILImageResampling

from benchmarks.data import load_images_w3, load_images_smooth
from benchmarks.models import get_llava_hf_processor, load_processors, MODEL_ID_QWEN
from dsl.codegen import build_llava
from kernels.qwen_v1_naive import qwen_v1
from pipelines.llava import pipeline as llava_pipeline, sched_v1

RESAMPLE_PAIRS = [
    ("BICUBIC",  PILImageResampling.BICUBIC),
    ("BILINEAR", PILImageResampling.BILINEAR),
]


def _fmt(val):
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


# ---------------------------------------------------------------------------
# LLaVA-NeXT
# ---------------------------------------------------------------------------

def _llava_max_diff(pv_dsl, n_tiles, hf_pv_np):
    """Compare unpadded DSL output against padded HF pixel_values tile-by-tile.

    pv_dsl:   (N_total, 3, 336, 336) float32
    hf_pv_np: (B, max_tiles, 3, 336, 336) float32
    """
    B = len(n_tiles)
    tile_offset = 0
    max_diff = 0.0
    for b in range(B):
        n = int(n_tiles[b])
        dsl = pv_dsl[tile_offset:tile_offset + n].astype(np.float32)
        hf  = hf_pv_np[b, :n].astype(np.float32)
        max_diff = max(max_diff, float(np.max(np.abs(dsl - hf))))
        tile_offset += n
    return max_diff


def run_llava(images_noise, images_smooth):
    print("\n" + "=" * 68)
    print("LLaVA-NeXT  —  DSL-v1 (antialiased BILINEAR) vs HF slow/fast")
    print("=" * 68)

    print("  Loading HF processors...")
    proc_legacy = get_llava_hf_processor(use_fast=False)
    proc_fast   = get_llava_hf_processor(use_fast=True)

    print("  Building DSL-v1 kernel (Numba JIT, ~30 s first call)...")
    dsl_v1 = build_llava(llava_pipeline, sched_v1)

    print("  Running DSL-v1 on noise and smooth images...")
    pv_n, nt_n = dsl_v1(images_noise)
    pv_s, nt_s = dsl_v1(images_smooth)

    col = f"  {'processor':<10} {'resample':<10} {'noise':>8} {'smooth':>8}"
    print("\n" + col)
    print("  " + "-" * (len(col) - 2))

    for proc_name, proc_obj in [("legacy", proc_legacy), ("fast", proc_fast)]:
        for resample_name, resample_val in RESAMPLE_PAIRS:
            out_n = proc_obj(images=images_noise,  return_tensors="pt", resample=resample_val)
            out_s = proc_obj(images=images_smooth, return_tensors="pt", resample=resample_val)
            hf_n = out_n["pixel_values"].float().numpy()
            hf_s = out_s["pixel_values"].float().numpy()
            diff_n = _llava_max_diff(pv_n, nt_n, hf_n)
            diff_s = _llava_max_diff(pv_s, nt_s, hf_s)
            print(f"  {proc_name:<10} {resample_name:<10} {_fmt(diff_n):>8} {_fmt(diff_s):>8}")


# ---------------------------------------------------------------------------
# Qwen2.5-VL
# ---------------------------------------------------------------------------

def _qwen_max_diff(pv_kernel, hf_pv_np):
    """Compare kernel and HF pixel_values; returns float or an error string."""
    if pv_kernel.shape != hf_pv_np.shape:
        return f"shape mismatch kernel={pv_kernel.shape} HF={hf_pv_np.shape}"
    return float(np.max(np.abs(pv_kernel.astype(np.float32) - hf_pv_np.astype(np.float32))))


def run_qwen(images_noise, images_smooth):
    print("\n" + "=" * 68)
    print("Qwen2.5-VL  —  v1 kernel (antialiased BILINEAR) vs HF slow/fast")
    print("=" * 68)

    print("  Loading HF processors (slow + fast)...")
    proc_slow, proc_fast = load_processors(MODEL_ID_QWEN)
    kw = dict(min_pixels=proc_fast.min_pixels, max_pixels=proc_fast.max_pixels)

    print("  Running v1 kernel (Numba JIT, ~30 s first call)...")
    pv_n, _ = qwen_v1(images_noise,  **kw)
    pv_s, _ = qwen_v1(images_smooth, **kw)

    col = f"  {'processor':<10} {'resample':<10} {'noise':>8} {'smooth':>8}"
    print("\n" + col)
    print("  " + "-" * (len(col) - 2))

    for proc_name, proc_obj in [("slow", proc_slow), ("fast", proc_fast)]:
        for resample_name, resample_val in RESAMPLE_PAIRS:
            out_n = proc_obj(images=images_noise,  return_tensors="pt", resample=resample_val)
            out_s = proc_obj(images=images_smooth, return_tensors="pt", resample=resample_val)
            hf_n = out_n["pixel_values"].float().numpy()
            hf_s = out_s["pixel_values"].float().numpy()
            diff_n = _qwen_max_diff(pv_n, hf_n)
            diff_s = _qwen_max_diff(pv_s, hf_s)
            print(f"  {proc_name:<10} {resample_name:<10} {_fmt(diff_n):>8} {_fmt(diff_s):>8}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("Loading images (W3 noise + smooth, 4 images each)...")
    images_noise  = load_images_w3(n_images=4, seed=0)
    images_smooth = load_images_smooth(n_images=4, seed=0)

    run_llava(images_noise, images_smooth)
    run_qwen(images_noise, images_smooth)

    print("\nDone.")


if __name__ == "__main__":
    main()
