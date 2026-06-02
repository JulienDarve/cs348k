"""Correctness harness for the DSL LLaVA-NeXT pipeline (D4 gate).

Run from the repo root:
    python tests/test_dsl_llava_correctness.py

Checks (in order, all must pass to clear the D4 gate):

  0. Constants: LLAVA_MEAN, LLAVA_STD, GRID_PINPOINTS in pipelines/llava.py
     match the HF processor's stored values (fails fast if constants are wrong).

  1. classify_fusion_llava maps sched_v1/v2/v3 → "naive"/"pointwise"/"full".

  2. build_llava(llava_pipeline, sched_*) returns a callable that runs and
     produces the right output shape (N_total, 3, 336, 336) and n_tiles_per_image
     with correct sum and length.

  3. Cross-schedule consistency: v1, v2, v3 all produce identical pixel values
     within rtol=1e-4 — same algorithm, different schedule. This is the load-
     bearing claim for the DSL abstraction.

  4. DSL-v1 vs HF (BILINEAR): atol=0.05 — establishes that the naive schedule
     computes the correct algorithm against the HF baseline.

  5. DSL-v3 vs HF (BILINEAR): atol=0.05 — establishes that the full-fusion
     schedule still computes the correct algorithm (composed coordinates give
     the same result as materializing intermediate buffers).

HF baseline: LlavaNextImageProcessor (use_fast=False, legacy PIL-based) with
resample=BILINEAR. The legacy processor uses PIL resize which has no antialias
parameter — the comparison is algorithm-matched without patching torchvision.

Output format: HF returns (B, max_tiles, 3, 336, 336) padded with zeros.
DSL returns (N_total_tiles, 3, 336, 336) unpadded. Checks 4 and 5 compare
per-image, per-tile using n_tiles_per_image as the unpadding index.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.data import load_images, load_images_w3, load_images_smooth
from benchmarks.models import get_llava_hf_processor
from dsl.codegen import build_llava, classify_fusion_llava
from dsl.schedule import Schedule, StageSchedule
from pipelines.llava import (
    pipeline as llava_pipeline,
    sched_v1, sched_v2, sched_v3,
    LLAVA_MEAN, LLAVA_STD, LLAVA_SCALE, TILE_SIZE, GRID_PINPOINTS,
)


def check(name, passed, msg=""):
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}" + (f": {msg}" if msg else ""))
    return passed


def _compare_vs_hf(pv, n_tiles, hf_pv, label, atol=0.05):
    """Compare unpadded DSL output against padded HF output tile-by-tile.

    hf_pv: (B, max_tiles, 3, 336, 336) numpy array (may be zero-padded).
    pv:    (N_total, 3, 336, 336) numpy array (unpadded).
    n_tiles: (B,) int64 — number of valid tiles per image.
    """
    B = len(n_tiles)
    tile_offset = 0
    max_diff = 0.0
    for b in range(B):
        n = int(n_tiles[b])
        dsl_tiles = pv[tile_offset:tile_offset + n]    # (n, 3, 336, 336)
        hf_tiles  = hf_pv[b, :n]                       # (n, 3, 336, 336) unpadded slice
        max_diff = max(max_diff, float(np.max(np.abs(dsl_tiles.astype(np.float32) - hf_tiles.astype(np.float32)))))
        tile_offset += n
    ok = check(f"{label} atol={atol}", max_diff <= atol, f"max_diff={max_diff:.4f}")
    return ok


def main():
    ap = argparse.ArgumentParser(
        description="Correctness harness for the DSL LLaVA-NeXT pipeline (D4 gate).")
    ap.add_argument("--num-threads", type=int, default=1,
                    help="Number of Numba threads (default: 1).")
    args = ap.parse_args()

    import numba
    numba.set_num_threads(args.num_threads)

    print(f"Loading HF LLaVA processor (use_fast=False)... (num_threads={args.num_threads})")
    proc = get_llava_hf_processor(use_fast=False)
    from transformers.image_utils import PILImageResampling

    all_passed = True

    # ------------------------------------------------------------------
    # 0. Constants verification
    # ------------------------------------------------------------------
    print("\n--- Check 0: constants match HF processor ---")
    proc_mean = np.array(proc.image_mean, dtype=np.float32)
    proc_std  = np.array(proc.image_std,  dtype=np.float32)
    proc_gp   = proc.image_grid_pinpoints  # list of [h, w] or [w, h] pairs

    ok = check("LLAVA_MEAN matches HF",
               np.allclose(LLAVA_MEAN, proc_mean, atol=1e-6),
               f"ours={LLAVA_MEAN.tolist()} hf={proc_mean.tolist()}")
    all_passed = all_passed and ok

    ok = check("LLAVA_STD matches HF",
               np.allclose(LLAVA_STD, proc_std, atol=1e-6),
               f"ours={LLAVA_STD.tolist()} hf={proc_std.tolist()}")
    all_passed = all_passed and ok

    ok = check("GRID_PINPOINTS matches HF",
               GRID_PINPOINTS == proc_gp,
               f"ours={GRID_PINPOINTS} hf={proc_gp}")
    all_passed = all_passed and ok

    if not all_passed:
        print("\nConstant mismatch — fix pipelines/llava.py before running further checks.")
        print("All checks FAILED.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. classify_fusion_llava
    # ------------------------------------------------------------------
    print("\n--- Check 1: classify_fusion_llava ---")
    for label, sched, expected in [
        ("v1", sched_v1, "naive"),
        ("v2", sched_v2, "pointwise"),
        ("v3", sched_v3, "full"),
    ]:
        got = classify_fusion_llava(llava_pipeline, sched)
        ok = check(f"sched_{label} → {expected!r}",
                   got == expected,
                   f"got {got!r}")
        all_passed = all_passed and ok

    # ------------------------------------------------------------------
    # 2. Build + run DSL-v1/v2/v3 (primes Numba JIT cache).
    # ------------------------------------------------------------------
    print("\n--- Check 2: build_llava + run (Numba JIT compile on first call, ~30s each) ---")
    dsl_v1 = build_llava(llava_pipeline, sched_v1)
    dsl_v2 = build_llava(llava_pipeline, sched_v2)
    dsl_v3 = build_llava(llava_pipeline, sched_v3)

    # Use W2 uniform images (all 1024×1024 → 5 tiles each) for shape checks.
    images_w2 = load_images(n_images=4, img_size=(1024, 1024))
    pv1_w2, nt1_w2 = dsl_v1(images_w2)
    pv2_w2, nt2_w2 = dsl_v2(images_w2)
    pv3_w2, nt3_w2 = dsl_v3(images_w2)

    for label, pv, n_tiles in [("v1", pv1_w2, nt1_w2), ("v2", pv2_w2, nt2_w2), ("v3", pv3_w2, nt3_w2)]:
        ok = check(f"{label}: output rank == 4", pv.ndim == 4,
                   f"got ndim={pv.ndim}")
        all_passed = all_passed and ok
        ok = check(f"{label}: output tile shape == (3, 336, 336)",
                   pv.shape[1:] == (3, TILE_SIZE, TILE_SIZE),
                   f"got {pv.shape[1:]}")
        all_passed = all_passed and ok
        ok = check(f"{label}: n_tiles.sum() == pv.shape[0]",
                   int(n_tiles.sum()) == pv.shape[0],
                   f"sum={n_tiles.sum()} shape[0]={pv.shape[0]}")
        all_passed = all_passed and ok
        ok = check(f"{label}: len(n_tiles) == batch_size",
                   len(n_tiles) == len(images_w2),
                   f"len={len(n_tiles)} batch={len(images_w2)}")
        all_passed = all_passed and ok

    # ------------------------------------------------------------------
    # 3. Cross-schedule consistency (noise W3 images)
    # ------------------------------------------------------------------
    print("\n--- Check 3: cross-schedule consistency rtol=1e-4 ---")
    images_noise = load_images_w3(n_images=4, seed=0)
    pv1_n, nt1_n = dsl_v1(images_noise)
    pv2_n, nt2_n = dsl_v2(images_noise)
    pv3_n, nt3_n = dsl_v3(images_noise)

    ok = check("n_tiles v1 == v2", np.array_equal(nt1_n, nt2_n),
               f"v1={nt1_n.tolist()} v2={nt2_n.tolist()}")
    all_passed = all_passed and ok
    ok = check("n_tiles v1 == v3", np.array_equal(nt1_n, nt3_n),
               f"v1={nt1_n.tolist()} v3={nt3_n.tolist()}")
    all_passed = all_passed and ok

    for label, a, b in [
        ("v2 vs v1 (noise)", pv2_n, pv1_n),
        ("v3 vs v1 (noise)", pv3_n, pv1_n),
        ("v3 vs v2 (noise)", pv3_n, pv2_n),
    ]:
        max_diff = float(np.max(np.abs(a - b)))
        ok = check(f"{label} rtol=1e-4", max_diff <= 1e-4,
                   f"max_diff={max_diff:.2e}")
        all_passed = all_passed and ok

    # Also verify on smooth images
    images_smooth = load_images_smooth(n_images=4, seed=0)
    pv1_s, nt1_s = dsl_v1(images_smooth)
    pv2_s, nt2_s = dsl_v2(images_smooth)
    pv3_s, nt3_s = dsl_v3(images_smooth)

    for label, a, b in [
        ("v2 vs v1 (smooth)", pv2_s, pv1_s),
        ("v3 vs v1 (smooth)", pv3_s, pv1_s),
        ("v3 vs v2 (smooth)", pv3_s, pv2_s),
    ]:
        max_diff = float(np.max(np.abs(a - b)))
        ok = check(f"{label} rtol=1e-4", max_diff <= 1e-4,
                   f"max_diff={max_diff:.2e}")
        all_passed = all_passed and ok

    # ------------------------------------------------------------------
    # 4. DSL-v1 vs HF (BILINEAR) — noise and smooth
    # ------------------------------------------------------------------
    print("\n--- Check 4: DSL-v1 vs HF legacy BILINEAR ---")
    hf_n  = proc(images=images_noise,  return_tensors="np",
                 resample=PILImageResampling.BILINEAR)
    hf_s  = proc(images=images_smooth, return_tensors="np",
                 resample=PILImageResampling.BILINEAR)
    hf_pv_n = np.array(hf_n["pixel_values"])   # (B, max_tiles, 3, 336, 336)
    hf_pv_s = np.array(hf_s["pixel_values"])

    ok = _compare_vs_hf(pv1_n, nt1_n, hf_pv_n, "DSL-v1 vs HF (noise)", atol=0.05)
    all_passed = all_passed and ok
    ok = _compare_vs_hf(pv1_s, nt1_s, hf_pv_s, "DSL-v1 vs HF (smooth)", atol=0.05)
    all_passed = all_passed and ok

    # ------------------------------------------------------------------
    # 5. DSL-v3 vs HF (BILINEAR) — noise and smooth
    # ------------------------------------------------------------------
    print("\n--- Check 5: DSL-v3 vs HF legacy BILINEAR ---")
    ok = _compare_vs_hf(pv3_n, nt3_n, hf_pv_n, "DSL-v3 vs HF (noise)", atol=0.05)
    all_passed = all_passed and ok
    ok = _compare_vs_hf(pv3_s, nt3_s, hf_pv_s, "DSL-v3 vs HF (smooth)", atol=0.05)
    all_passed = all_passed and ok

    # ------------------------------------------------------------------
    # 5b. Parallel v1/v2 vs serial v1/v2 — verifies prange(1 thread) ≡ serial
    # ------------------------------------------------------------------
    print("\n--- Check 5b: parallel v1/v2 vs serial v1/v2 (max_abs_diff <= 1e-4) ---")
    _names = llava_pipeline.names
    sched_v1_serial = Schedule(
        stages={n: StageSchedule() for n in _names},
        preallocate_output=False,
    )
    sched_v2_serial = Schedule(
        stages={
            **{n: StageSchedule() for n in _names},
            "rescale":   StageSchedule(compute_at="inline"),
            "normalize": StageSchedule(compute_at="inline"),
        },
        preallocate_output=False,
    )
    dsl_v1_serial = build_llava(llava_pipeline, sched_v1_serial)
    dsl_v2_serial = build_llava(llava_pipeline, sched_v2_serial)

    for label, pv_par, pv_ser in [
        ("v1 parallel vs serial (noise)",  pv1_n, dsl_v1_serial(images_noise)[0]),
        ("v2 parallel vs serial (noise)",  pv2_n, dsl_v2_serial(images_noise)[0]),
        ("v1 parallel vs serial (smooth)", pv1_s, dsl_v1_serial(images_smooth)[0]),
        ("v2 parallel vs serial (smooth)", pv2_s, dsl_v2_serial(images_smooth)[0]),
    ]:
        max_diff = float(np.max(np.abs(pv_par - pv_ser)))
        ok = check(label + " <= 1e-4", max_diff <= 1e-4, f"max_diff={max_diff:.2e}")
        all_passed = all_passed and ok

    print(f"\n{'All checks passed.' if all_passed else 'Some checks FAILED.'}")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
