"""Correctness harness for hand-fused Qwen kernel versions.

Checks:
  1. smart_resize_dims matches HF fast for representative sizes.
  2. v1 output shape matches HF fast (same total_patches, same patch_dim).
  3. v1 image_grid_thw matches HF fast (same spatial patch counts).
  4. v1 pixel_values are numerically close to HF fast pixel_values.

Note on tolerance: HF fast uses BICUBIC + antialias; v1 uses bilinear (as
specified in kernel_implementation.md). On smooth (band-limited) inputs the
two methods agree to within ~1% of the normalized range (atol=0.1). On the
white-noise w3 inputs neighboring pixels are uncorrelated, which widens the
gap to ~0.7 — that's expected, not a kernel bug, so the w3 check uses
atol=0.7 as a sanity bound while the smooth-image check enforces atol=0.1.
For v2/v3 correctness against v1, use rtol=1e-4 (same interpolation method).

Run from the repo root:
    python tests/test_correctness.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.data import load_images_w3, load_images_smooth
from benchmarks.models import load_processors, MODEL_ID_QWEN
from kernels.qwen_v1_naive import qwen_v1, smart_resize_dims


def check(name, passed, msg=""):
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}" + (f": {msg}" if msg else ""))
    return passed


def main():
    print("Loading images and HF fast processor...")
    images = load_images_w3(n_images=4, seed=0)
    _, proc = load_processors(MODEL_ID_QWEN)

    all_passed = True

    # ------------------------------------------------------------------
    # 1. smart_resize_dims matches HF
    # ------------------------------------------------------------------
    print("\n--- smart_resize_dims ---")
    from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize as hf_smart_resize
    test_sizes = [(512, 512), (256, 768), (1024, 512), (300, 400), (800, 600)]
    for h, w in test_sizes:
        our = smart_resize_dims(h, w, min_pixels=proc.min_pixels, max_pixels=proc.max_pixels)
        hf  = hf_smart_resize(h, w, factor=proc.patch_size * proc.merge_size,
                               min_pixels=proc.min_pixels, max_pixels=proc.max_pixels)
        ok = check(f"smart_resize_dims({h},{w})", our == hf,
                   f"ours={our} HF={hf}" if our != hf else "")
        all_passed = all_passed and ok

    # ------------------------------------------------------------------
    # 2–4. v1 output vs HF fast
    # ------------------------------------------------------------------
    print("\n--- v1 vs HF fast ---")
    print("  Running HF fast processor...")
    hf_out = proc(images=images, return_tensors="pt")
    hf_pv   = hf_out["pixel_values"].numpy()
    hf_grid = hf_out["image_grid_thw"].numpy()

    print("  Running v1 (numba JIT compile on first call, ~30s)...")
    pv, grid = qwen_v1(images, min_pixels=proc.min_pixels, max_pixels=proc.max_pixels)

    ok = check("output shape", pv.shape == hf_pv.shape,
               f"v1={pv.shape} HF={hf_pv.shape}" if pv.shape != hf_pv.shape else "")
    all_passed = all_passed and ok

    ok = check("image_grid_thw", np.array_equal(grid, hf_grid),
               f"\n    v1={grid}\n    HF={hf_grid}" if not np.array_equal(grid, hf_grid) else "")
    all_passed = all_passed and ok

    max_diff = float(np.max(np.abs(pv - hf_pv)))
    ok = check("pixel_values atol=0.7 (noise input)", max_diff <= 0.7,
               f"max_diff={max_diff:.4f}")
    all_passed = all_passed and ok

    # ------------------------------------------------------------------
    # 5. v1 vs HF fast on smooth (low-frequency) images
    #    Bilinear and bicubic agree tightly when the input is band-limited,
    #    so this is the real correctness check on pixel values.
    # ------------------------------------------------------------------
    print("\n--- v1 vs HF fast (smooth images) ---")
    smooth_images = load_images_smooth(n_images=4, seed=0)
    hf_out_s = proc(images=smooth_images, return_tensors="pt")
    hf_pv_s   = hf_out_s["pixel_values"].numpy()
    hf_grid_s = hf_out_s["image_grid_thw"].numpy()

    pv_s, grid_s = qwen_v1(smooth_images,
                           min_pixels=proc.min_pixels, max_pixels=proc.max_pixels)

    ok = check("output shape", pv_s.shape == hf_pv_s.shape,
               f"v1={pv_s.shape} HF={hf_pv_s.shape}" if pv_s.shape != hf_pv_s.shape else "")
    all_passed = all_passed and ok

    ok = check("image_grid_thw", np.array_equal(grid_s, hf_grid_s),
               f"\n    v1={grid_s}\n    HF={hf_grid_s}" if not np.array_equal(grid_s, hf_grid_s) else "")
    all_passed = all_passed and ok

    max_diff_s = float(np.max(np.abs(pv_s - hf_pv_s)))
    ok = check("pixel_values atol=0.1", max_diff_s <= 0.1, f"max_diff={max_diff_s:.4f}")
    all_passed = all_passed and ok

    print(f"\n{'All checks passed.' if all_passed else 'Some checks FAILED.'}")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
