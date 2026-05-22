"""Kernel benchmark — Qwen2.5-VL: HF fast, HF legacy, HF fast w/ bilinear, v1, v2, v3.

Important note on memory measurement order:
  Memory benchmarks are run FIRST, before any timing loops or warmup. Timing
  loops (N_WARMUP=10, N_TIMED=100 per variant) cause the Python/numpy allocator
  to build up pools that inflate RSS for all subsequent measure_peak_rss() calls,
  making the peak delta look smaller or zero. Running memory benchmarks first —
  with only the controlled n_memory_warmup calls as pre-measurement activity —
  gives the cleanest possible RSS delta without restructuring the kernel code.

  For v1/v2 specifically, Numba JIT-compiles on the first call (~30s). We perform
  one unmeasured compilation call before any measurement, then rely on the normal
  n_memory_warmup mechanism for the warm-cache measurement.

Processor variants (Qwen2.5-VL only):
  hf_legacy:   AutoImageProcessor(use_fast=False)  — torchvision bicubic+antialias
  hf_fast:     AutoImageProcessor(use_fast=True)   — torchvision bicubic+antialias
  hf_bilinear: AutoImageProcessor(use_fast=True, resample=BILINEAR) — algorithm-matched
  v1:          kernels/qwen_v1_naive.py             — naive staged numba pipeline
  v2:          kernels/qwen_v2_fused.py             — rescale+normalize fused into resize
  v3:          kernels/qwen_v3_storage.py           — full fusion, pre-allocated output, parallel batch

Workloads:
  W2: 32 images, all 1024×1024 (uniform size baseline).
  W3: 32 images with random sizes and aspect ratio in [0.5, 2.0].
  W4: 8 images at 2480×3508 (A4 @ 300 dpi).

Protocol:
  Phase 1 — MEMORY (runs first, before timing pollutes allocator pools):
    measure_peak_rss() around one call per variant per workload.
    Optional n_memory_warmup unmeasured calls immediately before each measurement.
  Phase 2 — TIMING (runs second):
    W2/W3: N_WARMUP=10 warmup + N_TIMED=30 timed iterations.
    W4:    N_WARMUP_W4=2  warmup + N_TIMED_W4=16 timed iterations.

Usage:
  python bench_kernels.py
  python bench_kernels.py --num-threads N
  python bench_kernels.py --skip-warmup
  python bench_kernels.py --memory-only
  python bench_kernels.py --timing-only
"""
import gc
import os
import sys
import argparse
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.data import load_images, load_images_w3, load_images_w4
from benchmarks.measurement import time_fn, measure_memory, env_info
from benchmarks.models import load_processors, MODEL_ID_QWEN
from kernels.qwen_v1_naive import qwen_v1
from kernels.qwen_v2_fused import qwen_v2
from kernels.qwen_v3_storage import qwen_v3

from transformers.image_utils import PILImageResampling

N_WARMUP = 10
N_TIMED = 30
N_WARMUP_W4 = 2
N_TIMED_W4 = 16
N_MEMORY_WARMUP = 0


def _wrap_v1(images, proc):
    """Call v1 and return an HF-compatible dict for output_bytes/output_shape."""
    pv, grid = qwen_v1(images, min_pixels=proc.min_pixels, max_pixels=proc.max_pixels)
    return {"pixel_values": pv, "image_grid_thw": grid}


def _wrap_v2(images, proc):
    """Call v2 and return an HF-compatible dict for output_bytes/output_shape."""
    pv, grid = qwen_v2(images, min_pixels=proc.min_pixels, max_pixels=proc.max_pixels)
    return {"pixel_values": pv, "image_grid_thw": grid}


def _wrap_v3(images, proc):
    """Call v3 and return an HF-compatible dict for output_bytes/output_shape."""
    pv, grid = qwen_v3(images, min_pixels=proc.min_pixels, max_pixels=proc.max_pixels)
    return {"pixel_values": pv, "image_grid_thw": grid}


ALL_VARIANTS = ["hf_legacy", "hf_fast", "hf_bilinear", "v1", "v2", "v3"]


def make_calls(images, q_legacy, q_fast, proc_fast, variants=None):
    """Return an ordered list of (label, callable) for one workload.

    variants: iterable of variant names to include; None means all.
    """
    if variants is None:
        variants = ALL_VARIANTS
    selected = set(variants)
    all_calls = [
        ("hf_legacy",   lambda: q_legacy(images=images, return_tensors="np")),
        ("hf_fast",     lambda: q_fast(images=images, return_tensors="pt")),
        ("hf_bilinear", lambda: q_fast(images=images, return_tensors="pt",
                                       resample=PILImageResampling.BILINEAR)),
        ("v1",          lambda: _wrap_v1(images, proc_fast)),
        ("v2",          lambda: _wrap_v2(images, proc_fast)),
        ("v3",          lambda: _wrap_v3(images, proc_fast)),
    ]
    return [(name, fn) for name, fn in all_calls if name in selected]


# ---------------------------------------------------------------------------
# Memory phase
# ---------------------------------------------------------------------------

def run_memory_workload(label, images, q_legacy, q_fast, proc_fast, n_memory_warmup, variants):
    n = len(images)
    print(f"\n{'='*60}")
    print(f"[MEMORY] workload {label}: {n} images  (warmup={n_memory_warmup})")
    print(f"{'='*60}")

    results = {}
    for name, call in make_calls(images, q_legacy, q_fast, proc_fast, variants):
        ratio = measure_memory(f"{name} ({label})", call, n_memory_warmup)
        results[name] = ratio

    print(f"\n=== memory summary {label} ===")
    for name, ratio in results.items():
        print(f"  {name:<20} peak/output: {ratio:.2f}x")

    return results


# ---------------------------------------------------------------------------
# Timing phase
# ---------------------------------------------------------------------------

def run_timing_workload(label, images, q_legacy, q_fast, proc_fast, n_warmup, n_timed, variants):
    n = len(images)
    print(f"\n{'='*60}")
    print(f"[TIMING] workload {label}: {n} images  (warmup={n_warmup}, timed={n_timed})")
    print(f"{'='*60}")

    results = {}
    for name, call in make_calls(images, q_legacy, q_fast, proc_fast, variants):
        median_ms, p95_p50 = time_fn(call, n_warmup=n_warmup, n_timed=n_timed, desc=name)
        results[name] = median_ms
        print(f"\n--- {name} ({label}) ---")
        print(f"  median:    {median_ms:8.2f} ms/batch ({median_ms/n:.3f} ms/img)")
        print(f"  p95-p50:   {p95_p50:8.2f} ms")

    hf_fast_ms = results.get("hf_fast")
    print(f"\n=== timing summary {label} ===")
    for name, ms in results.items():
        if hf_fast_ms:
            ratio = ms / hf_fast_ms if hf_fast_ms > 0 else float("nan")
            print(f"  {name:<20} {ms:8.2f} ms/batch  ({ratio:.2f}x vs hf_fast)")
        else:
            print(f"  {name:<20} {ms:8.2f} ms/batch  ({ms/n:.3f} ms/img)")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Kernel benchmark: HF fast/legacy/bilinear vs v1/v2 on W2, W3, W4.")
    ap.add_argument("--num-threads", type=int, default=1, metavar="N",
                    help="OMP/MKL/torch intra-op thread count (default: 1).")
    ap.add_argument("--n-memory-warmup", type=int, default=N_MEMORY_WARMUP,
                    help="Unmeasured warmup calls before each memory measurement.")
    ap.add_argument("--skip-warmup", action="store_true",
                    help="Set n_memory_warmup=0.")
    ap.add_argument("--n-warmup", type=int, default=N_WARMUP)
    ap.add_argument("--n-timed", type=int, default=N_TIMED)
    ap.add_argument("--n-warmup-w4", type=int, default=N_WARMUP_W4)
    ap.add_argument("--n-timed-w4", type=int, default=N_TIMED_W4)
    ap.add_argument("--memory-only", action="store_true",
                    help="Skip timing phase.")
    ap.add_argument("--timing-only", action="store_true",
                    help="Skip memory phase (RSS results will be tainted by timing warmup).")
    ap.add_argument("--workloads", nargs="+", choices=["W2", "W3", "W4"],
                    default=["W2", "W3", "W4"], metavar="W",
                    help="Which workloads to run (default: W2 W3 W4).")
    ap.add_argument("--variants", nargs="+", choices=ALL_VARIANTS,
                    default=None, metavar="V",
                    help="Which processor variants to benchmark (default: all). "
                         f"Choices: {ALL_VARIANTS}.")
    args = ap.parse_args()

    if args.timing_only and args.memory_only:
        raise SystemExit("--memory-only and --timing-only are mutually exclusive")

    n_threads = args.num_threads
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    torch.set_num_threads(n_threads)

    n_memory_warmup = 0 if args.skip_warmup else args.n_memory_warmup

    print(env_info())

    # Load processors — use_fast=True twice: once as q_fast, once as proc_fast
    # (proc_fast is also used by _wrap_v1 to read min_pixels/max_pixels).
    q_legacy, q_fast = load_processors(MODEL_ID_QWEN)
    proc_fast = q_fast  # alias for clarity in _wrap_v1 calls
    print(f"\nLoaded: {MODEL_ID_QWEN} (legacy, fast)")

    # Load only the requested workloads
    selected = set(args.workloads)
    images_w2 = load_images(n_images=32, img_size=(1024, 1024)) if "W2" in selected else None
    images_w3 = load_images_w3()                                 if "W3" in selected else None
    images_w4 = load_images_w4()                                 if "W4" in selected else None
    if images_w2: print(f"\nW2: {len(images_w2)} images @ {images_w2[0].size}")
    if images_w3: print(f"W3: {len(images_w3)} images, sizes e.g. {[img.size for img in images_w3[:3]]}")
    if images_w4: print(f"W4: {len(images_w4)} images @ {images_w4[0].size}")

    variants = args.variants  # None means all

    # Trigger Numba JIT compilation before any measurement using the first available workload.
    # These calls are intentionally unmeasured; RSS from compilation is not meaningful.
    first_images = images_w2 or images_w3 or images_w4
    assert first_images is not None
    needs_v1 = variants is None or "v1" in variants
    needs_v2 = variants is None or "v2" in variants
    needs_v3 = variants is None or "v3" in variants
    if needs_v1 or needs_v2 or needs_v3:
        print("\nWarming up Numba JIT for v1/v2/v3 (first call compiles, ~30s)...")
        if needs_v1:
            _wrap_v1(first_images[:1], proc_fast)
        if needs_v2:
            _wrap_v2(first_images[:1], proc_fast)
        if needs_v3:
            _wrap_v3(first_images[:1], proc_fast)
        print("Numba JIT ready.")

    all_workloads = {
        "W2": (images_w2, args.n_warmup,    args.n_timed),
        "W3": (images_w3, args.n_warmup,    args.n_timed),
        "W4": (images_w4, args.n_warmup_w4, args.n_timed_w4),
    }
    workloads = [(label, *all_workloads[label]) for label in args.workloads]

    # -----------------------------------------------------------------------
    # Phase 1: Memory — MUST run before timing to avoid allocator pool buildup.
    # -----------------------------------------------------------------------
    if not args.timing_only:
        print("\n" + "#"*60)
        print("# PHASE 1: MEMORY  (running before timing to keep RSS clean)")
        print("#"*60)
        for label, images, _, _ in workloads:
            run_memory_workload(label, images, q_legacy, q_fast, proc_fast, n_memory_warmup, variants)

    # -----------------------------------------------------------------------
    # Phase 2: Timing — runs after memory so allocator warmup doesn't affect RSS.
    # -----------------------------------------------------------------------
    if not args.memory_only:
        print("\n" + "#"*60)
        print("# PHASE 2: TIMING")
        print("#"*60)
        for label, images, n_warmup, n_timed in workloads:
            run_timing_workload(label, images, q_legacy, q_fast, proc_fast, n_warmup, n_timed, variants)


if __name__ == "__main__":
    main()
