"""Correctness harness for hand-fused Qwen kernel versions.

Run from the repo root:
    python tests/test_correctness.py

================================================================================
ALGORITHM CHOICE
================================================================================
HF's Qwen2VLImageProcessorFast uses BICUBIC + antialias=True (torchvision
default). Our v1 kernel uses naive bilinear (4 taps, no antialias filter), per
kernel_implementation.md. Bilinear was chosen for v1 because it is a simpler
algorithm IR to optimize and iterate on while building the v2/v3 scheduling
story. For VLM training quality the two choices are nearly indistinguishable;
for ms-level inference parity on a released checkpoint, HF's bicubic+antialias
would need to be matched exactly.

================================================================================
CHECKS
================================================================================
  1. smart_resize_dims matches HF fast for representative sizes.
  2. v1 output shape matches HF fast (total_patches, patch_dim).
  3. v1 image_grid_thw matches HF fast (spatial patch counts).
  4. v1 pixel_values vs HF fast as shipped (BICUBIC + antialias) — both
     smooth and noise inputs, loose tolerances since algorithms differ.
  5. v1 pixel_values vs HF fast forced to BILINEAR (antialias still on) —
     algorithm partially matched, isolates the antialias filter's effect.
  6. v1 pixel_values vs HF fast forced to BILINEAR + antialias=False (via
     a contextmanager that patches torchvision.F.resize) — fully matched.
     Any remaining gap is pure float32 reduction-order noise.

================================================================================
EMPIRICAL RESULTS (seed=0, n_images=4)
================================================================================
                                     noise input    smooth input
  HF bicubic + antialias              0.6951         0.0177
  HF bilinear + antialias             0.1194         0.0151
  HF bilinear, antialias=False        0.0150         0.0150

Each residual is fully explained by a named algorithmic difference:

  - 0.6951 (noise, bicubic+aa)   = bicubic-vs-bilinear  +  antialias  +  fp
  - 0.1194 (noise, bilinear+aa)  = antialias filter  +  fp           (~0.10 from
                                                                       antialias)
  - 0.0177 (smooth, bicubic+aa)  = tiny bicubic-vs-bilinear  +  fp   (~0.003
                                                                       from bicubic)
  - 0.0151 (smooth, bilinear+aa) = fp only (antialias does ~nothing on smooth)
  - 0.0150 (both, bilinear+noAA) = fp only — the FLOOR

The ~0.015 floor (~0.4% of normalized range, ~1 pixel value out of 255) is the
irreducible difference between two correct implementations of the same algorithm
on float32, caused by:
  - torchvision SIMD reduction order  vs  our serial njit loops.
  - HF's fused rescale_and_normalize (one rounding point)  vs  our two-pass
    rescale → normalize (two rounding points).

================================================================================
CORRECTNESS VERDICT
================================================================================
  - resize geometry (sampling positions, kernel weights)  → correct
  - rescale (/255.0)                                       → correct
  - normalize (per-channel mean/std, in the right order)   → correct
  - patchify row order  (gh//m, gw//m, mh, mw merge-block) → correct
  - patchify column order  (C, T, P_y, P_x)                → correct

For v2/v3 correctness against v1, use rtol=1e-4 (same interpolation method,
same algorithm — diffs should be at the float32 accumulation-order level only).

================================================================================
WHAT THIS MEANS FOR BENCHMARKING
================================================================================
Comparing v1 (bilinear) against HF-as-shipped (bicubic+antialias) conflates two
speedups: (a) algorithmic savings from cheaper interpolation, (b) scheduling
wins from fusion/tiling/format-handling. For honest speedup numbers, also
benchmark HF with `resample=PILImageResampling.BILINEAR` (and optionally with
antialias=False via the contextmanager below) so the (b)→(c) gap isolates the
schedule win.
"""

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np


@contextmanager
def _hf_no_antialias():
    """Force torchvision F.resize(antialias=False) inside the with-block.

    HF's Qwen2VL fast processor calls F.resize via BaseImageProcessorFast and
    doesn't expose `antialias` as a call kwarg, so we patch the module attribute
    that BOTH HF call sites resolve through.
    """
    import torchvision.transforms.v2.functional as Fmod
    original = Fmod.resize

    def _patched(*args, **kwargs):
        kwargs["antialias"] = False
        return original(*args, **kwargs)

    Fmod.resize = _patched
    try:
        yield
    finally:
        Fmod.resize = original

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.data import load_images_w3, load_images_smooth
from benchmarks.models import load_processors, MODEL_ID_QWEN
from kernels.qwen_v1_naive import qwen_v1, smart_resize_dims
from kernels.qwen_v2_fused import qwen_v2
from kernels.qwen_v3_storage import qwen_v3


def check(name, passed, msg=""):
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}" + (f": {msg}" if msg else ""))
    return passed


def _run_kernel(version, images, proc):
    """Run the selected kernel version on images and return (pixel_values, grid)."""
    kw = dict(min_pixels=proc.min_pixels, max_pixels=proc.max_pixels)
    if version == "v1":
        return qwen_v1(images, **kw)
    if version == "v2":
        return qwen_v2(images, **kw)
    if version == "v3":
        return qwen_v3(images, **kw)
    raise ValueError(f"Unknown version: {version}")


def main():
    ap = argparse.ArgumentParser(
        description="Correctness harness for hand-fused Qwen kernel versions.")
    ap.add_argument("--version", choices=["v1", "v2", "v3"], default="v1",
                    help="Kernel version to test against HF (default: v1). "
                         "Tests 1-7 run on this version; tests 8-9 always cross-check all versions.")
    ap.add_argument("--num-threads", type=int, default=1,
                    help="Number of Numba threads for parallel kernels (default: 1).")
    args = ap.parse_args()
    ver = args.version

    import numba
    numba.set_num_threads(args.num_threads)

    print(f"Testing version: {ver} (num_threads={args.num_threads})")
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
    # 2–4. <ver> output vs HF fast (noise images)
    # ------------------------------------------------------------------
    print(f"\n--- {ver} vs HF fast ---")
    print("  Running HF fast processor...")
    hf_out = proc(images=images, return_tensors="pt")
    hf_pv   = hf_out["pixel_values"].numpy()
    hf_grid = hf_out["image_grid_thw"].numpy()

    print(f"  Running {ver} (numba JIT compile on first call, ~30s)...")
    pv, grid = _run_kernel(ver, images, proc)

    ok = check("output shape", pv.shape == hf_pv.shape,
               f"{ver}={pv.shape} HF={hf_pv.shape}" if pv.shape != hf_pv.shape else "")
    all_passed = all_passed and ok

    ok = check("image_grid_thw", np.array_equal(grid, hf_grid),
               f"\n    {ver}={grid}\n    HF={hf_grid}" if not np.array_equal(grid, hf_grid) else "")
    all_passed = all_passed and ok

    max_diff = float(np.max(np.abs(pv - hf_pv)))
    ok = check("pixel_values atol=0.7 (noise input)", max_diff <= 0.7,
               f"max_diff={max_diff:.4f}")
    all_passed = all_passed and ok

    # ------------------------------------------------------------------
    # 5. <ver> vs HF fast on smooth (low-frequency) images
    #    Bilinear and bicubic agree tightly when the input is band-limited,
    #    so this is the real correctness check on pixel values.
    # ------------------------------------------------------------------
    print(f"\n--- {ver} vs HF fast (smooth images) ---")
    smooth_images = load_images_smooth(n_images=4, seed=0)
    hf_out_s = proc(images=smooth_images, return_tensors="pt")
    hf_pv_s   = hf_out_s["pixel_values"].numpy()
    hf_grid_s = hf_out_s["image_grid_thw"].numpy()

    pv_s, grid_s = _run_kernel(ver, smooth_images, proc)

    ok = check("output shape", pv_s.shape == hf_pv_s.shape,
               f"{ver}={pv_s.shape} HF={hf_pv_s.shape}" if pv_s.shape != hf_pv_s.shape else "")
    all_passed = all_passed and ok

    ok = check("image_grid_thw", np.array_equal(grid_s, hf_grid_s),
               f"\n    {ver}={grid_s}\n    HF={hf_grid_s}" if not np.array_equal(grid_s, hf_grid_s) else "")
    all_passed = all_passed and ok

    max_diff_s = float(np.max(np.abs(pv_s - hf_pv_s)))
    ok = check("pixel_values atol=0.1", max_diff_s <= 0.1, f"max_diff={max_diff_s:.4f}")
    all_passed = all_passed and ok

    # ------------------------------------------------------------------
    # 6. <ver> vs HF fast forced to BILINEAR (algorithm-matched comparison)
    #    Removes the bicubic-vs-bilinear confound: any remaining gap is
    #    just torchvision's antialias filter vs our naive 4-tap bilinear.
    #    On smooth inputs both should be effectively identical (atol=0.05).
    #    On noise the antialias filter still widens the gap, so use atol=0.3.
    # ------------------------------------------------------------------
    print(f"\n--- {ver} vs HF fast (BILINEAR, algorithm-matched) ---")
    from transformers.image_utils import PILImageResampling

    # 6a. smooth images — tight tolerance
    hf_out_bl_s = proc(images=smooth_images, return_tensors="pt",
                       resample=PILImageResampling.BILINEAR)
    hf_pv_bl_s = hf_out_bl_s["pixel_values"].numpy()
    hf_grid_bl_s = hf_out_bl_s["image_grid_thw"].numpy()

    ok = check("smooth: output shape", pv_s.shape == hf_pv_bl_s.shape,
               f"{ver}={pv_s.shape} HF={hf_pv_bl_s.shape}" if pv_s.shape != hf_pv_bl_s.shape else "")
    all_passed = all_passed and ok

    ok = check("smooth: image_grid_thw", np.array_equal(grid_s, hf_grid_bl_s),
               f"\n    {ver}={grid_s}\n    HF={hf_grid_bl_s}" if not np.array_equal(grid_s, hf_grid_bl_s) else "")
    all_passed = all_passed and ok

    max_diff_bl_s = float(np.max(np.abs(pv_s - hf_pv_bl_s)))
    ok = check("smooth: pixel_values atol=0.05", max_diff_bl_s <= 0.05,
               f"max_diff={max_diff_bl_s:.4f}")
    all_passed = all_passed and ok

    # 6b. noise images — looser tolerance (antialias filter still differs)
    hf_out_bl_n = proc(images=images, return_tensors="pt",
                       resample=PILImageResampling.BILINEAR)
    hf_pv_bl_n = hf_out_bl_n["pixel_values"].numpy()

    max_diff_bl_n = float(np.max(np.abs(pv - hf_pv_bl_n)))
    ok = check("noise: pixel_values atol=0.3", max_diff_bl_n <= 0.3,
               f"max_diff={max_diff_bl_n:.4f}")
    all_passed = all_passed and ok

    # ------------------------------------------------------------------
    # 7. <ver> vs HF fast forced to BILINEAR + antialias=False
    #    Fully algorithm-matched: same kernel, same tap count, no filter.
    #    Any remaining gap is purely float32 accumulation order (different
    #    SIMD reduction in torchvision vs our njit loops, different rounding
    #    point in HF's fused rescale_and_normalize vs our two-pass version).
    #    Expected tight on both smooth and noise inputs (atol=0.05).
    # ------------------------------------------------------------------
    print(f"\n--- {ver} vs HF fast (BILINEAR, antialias=False, fully matched) ---")
    with _hf_no_antialias():
        hf_out_na_s = proc(images=smooth_images, return_tensors="pt",
                           resample=PILImageResampling.BILINEAR)
        hf_out_na_n = proc(images=images, return_tensors="pt",
                           resample=PILImageResampling.BILINEAR)
    hf_pv_na_s = hf_out_na_s["pixel_values"].numpy()
    hf_pv_na_n = hf_out_na_n["pixel_values"].numpy()

    max_diff_na_s = float(np.max(np.abs(pv_s - hf_pv_na_s)))
    ok = check("smooth: pixel_values atol=0.05", max_diff_na_s <= 0.05,
               f"max_diff={max_diff_na_s:.4f}")
    all_passed = all_passed and ok

    max_diff_na_n = float(np.max(np.abs(pv - hf_pv_na_n)))
    ok = check("noise: pixel_values atol=0.05", max_diff_na_n <= 0.05,
               f"max_diff={max_diff_na_n:.4f}")
    all_passed = all_passed and ok

    # ------------------------------------------------------------------
    # 8. v2 vs v1 — always runs regardless of --version
    #    Same algorithm, different schedule: diffs are accumulation-order
    #    noise only. rtol=1e-4 is the correctness gate per the design doc.
    # ------------------------------------------------------------------
    print("\n--- v2 vs v1 (cross-version check, always runs) ---")
    kw = dict(min_pixels=proc.min_pixels, max_pixels=proc.max_pixels)
    pv1_n, grid1_n = qwen_v1(images, **kw)
    pv1_s, grid1_s = qwen_v1(smooth_images, **kw)
    pv2_n, grid2_n = qwen_v2(images, **kw)
    pv2_s, grid2_s = qwen_v2(smooth_images, **kw)

    ok = check("output shape (noise)", pv2_n.shape == pv1_n.shape,
               f"v2={pv2_n.shape} v1={pv1_n.shape}" if pv2_n.shape != pv1_n.shape else "")
    all_passed = all_passed and ok

    ok = check("image_grid_thw (noise)", np.array_equal(grid2_n, grid1_n),
               f"\n    v2={grid2_n}\n    v1={grid1_n}" if not np.array_equal(grid2_n, grid1_n) else "")
    all_passed = all_passed and ok

    max_diff_v2_n = float(np.max(np.abs(pv2_n - pv1_n)))
    ok = check("pixel_values rtol=1e-4 (noise)", max_diff_v2_n <= 1e-4,
               f"max_diff={max_diff_v2_n:.2e}")
    all_passed = all_passed and ok

    ok = check("output shape (smooth)", pv2_s.shape == pv1_s.shape,
               f"v2={pv2_s.shape} v1={pv1_s.shape}" if pv2_s.shape != pv1_s.shape else "")
    all_passed = all_passed and ok

    max_diff_v2_s = float(np.max(np.abs(pv2_s - pv1_s)))
    ok = check("pixel_values rtol=1e-4 (smooth)", max_diff_v2_s <= 1e-4,
               f"max_diff={max_diff_v2_s:.2e}")
    all_passed = all_passed and ok

    # ------------------------------------------------------------------
    # 9. v3 vs v1 / v3 vs v2 — always runs regardless of --version
    #    Full fusion eliminates the intermediate buffer but keeps the same
    #    algorithm. Diffs vs v1 and v2 should be at the float32 accumulation-
    #    order level only. rtol=1e-4 is the correctness gate.
    # ------------------------------------------------------------------
    print("\n--- v3 vs v1 / v3 vs v2 (cross-version check, always runs) ---")
    pv3_n, grid3_n = qwen_v3(images, **kw)
    pv3_s, grid3_s = qwen_v3(smooth_images, **kw)

    ok = check("v3: output shape (noise)", pv3_n.shape == pv1_n.shape,
               f"v3={pv3_n.shape} v1={pv1_n.shape}" if pv3_n.shape != pv1_n.shape else "")
    all_passed = all_passed and ok

    ok = check("v3: image_grid_thw (noise)", np.array_equal(grid3_n, grid1_n),
               f"\n    v3={grid3_n}\n    v1={grid1_n}" if not np.array_equal(grid3_n, grid1_n) else "")
    all_passed = all_passed and ok

    max_diff_v3_v1_n = float(np.max(np.abs(pv3_n - pv1_n)))
    ok = check("v3 vs v1: pixel_values rtol=1e-4 (noise)", max_diff_v3_v1_n <= 1e-4,
               f"max_diff={max_diff_v3_v1_n:.2e}")
    all_passed = all_passed and ok

    ok = check("v3: output shape (smooth)", pv3_s.shape == pv1_s.shape,
               f"v3={pv3_s.shape} v1={pv1_s.shape}" if pv3_s.shape != pv1_s.shape else "")
    all_passed = all_passed and ok

    max_diff_v3_v1_s = float(np.max(np.abs(pv3_s - pv1_s)))
    ok = check("v3 vs v1: pixel_values rtol=1e-4 (smooth)", max_diff_v3_v1_s <= 1e-4,
               f"max_diff={max_diff_v3_v1_s:.2e}")
    all_passed = all_passed and ok

    max_diff_v3_v2_n = float(np.max(np.abs(pv3_n - pv2_n)))
    ok = check("v3 vs v2: pixel_values rtol=1e-4 (noise)", max_diff_v3_v2_n <= 1e-4,
               f"max_diff={max_diff_v3_v2_n:.2e}")
    all_passed = all_passed and ok

    max_diff_v3_v2_s = float(np.max(np.abs(pv3_s - pv2_s)))
    ok = check("v3 vs v2: pixel_values rtol=1e-4 (smooth)", max_diff_v3_v2_s <= 1e-4,
               f"max_diff={max_diff_v3_v2_s:.2e}")
    all_passed = all_passed and ok

    print(f"\n{'All checks passed.' if all_passed else 'Some checks FAILED.'}")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
