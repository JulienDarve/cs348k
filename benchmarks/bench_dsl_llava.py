"""DSL benchmark — LLaVA-NeXT: HF fast, HF legacy, dsl_v1, dsl_v2, dsl_v3.

Mirrors bench_dsl.py in protocol, flags, and output format, but targets
the LLaVA-NeXT pipeline (llava-hf/llava-v1.6-mistral-7b-hf) with AnyRes
tiling. No hand-tuned kernels exist for LLaVA.

Processor variants:
  hf_legacy:  AutoImageProcessor(use_fast=False) — PIL-based resize
  hf_fast:    AutoImageProcessor(use_fast=True)  — torchvision-based
  dsl_v1:     DSL build_llava(pipeline, sched_v1) — naive staged pipeline
  dsl_v2:     DSL build_llava(pipeline, sched_v2) — rescale+normalize+CHW fused
  dsl_v3:     DSL build_llava(pipeline, sched_v3) — full fusion, preallocated output

Output format:
  HF:  {"pixel_values": (B, max_tiles, 3, 336, 336)} — padded per batch
  DSL: {"pixel_values": (N_total_tiles, 3, 336, 336)} — unpadded concatenation
  output_bytes uses pixel_values.nbytes in both cases.

Workloads:
  W2: 32 images at 1024×1024 (uniform → 5 tiles each = 160 total tiles/batch).
  W3: 32 images with random sizes and aspect ratio in [0.5, 2.0].
  W4: 8 images at 2480×3508 (A4 @ 300 dpi).

Protocol:
  Phase 1 — MEMORY (runs first, before timing pollutes allocator pools):
    measure_peak_rss() around one call per variant per workload.
    Optional n_memory_warmup unmeasured calls before each measurement.
  Phase 2 — TIMING:
    W2/W3: N_WARMUP=10 warmup + N_TIMED=30 timed iterations.
    W4:    N_WARMUP_W4=2  warmup + N_TIMED_W4=16 timed iterations.

Memory story (expected):
  hf_legacy / hf_fast:  peak/output ~ 2-4x (multiple intermediate PIL/tensor allocs)
  dsl_v1:               peak/output ~ 3-4x (full-res + per-tile HWC intermediates)
  dsl_v2:               peak/output ~ 2x   (full-res only; per-tile HWC eliminated)
  dsl_v3:               peak/output ~ 1x   (zero intermediates, preallocated output)

Usage:
  python benchmarks/bench_dsl_llava.py
  python benchmarks/bench_dsl_llava.py --num-threads N
  python benchmarks/bench_dsl_llava.py --variants hf_fast dsl_v1 dsl_v3
  python benchmarks/bench_dsl_llava.py --workloads W2 W3
  python benchmarks/bench_dsl_llava.py --memory-only
  python benchmarks/bench_dsl_llava.py --timing-only
"""
import gc
import os
import sys
import argparse
from pathlib import Path

import numba
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.data import load_images, load_images_w3, load_images_w4
from benchmarks.measurement import time_fn, measure_memory, env_info
from benchmarks.models import get_llava_hf_processor
from dsl.codegen import build_llava
from pipelines.llava import pipeline as llava_pipeline, sched_v1, sched_v2, sched_v3

N_WARMUP    = 10
N_TIMED     = 30
N_WARMUP_W4 = 2
N_TIMED_W4  = 16
N_MEMORY_WARMUP = 0

ALL_VARIANTS = ["hf_legacy", "hf_fast", "dsl_v1", "dsl_v2", "dsl_v3"]


def _wrap_hf(proc, images):
    """Call HF processor and return output dict for output_bytes/output_shape."""
    return proc(images=images, return_tensors="pt")


def _wrap_dsl(fn, images):
    """Call a DSL-built callable and return dict with pixel_values numpy array."""
    pv, _n_tiles = fn(images)
    return {"pixel_values": pv}   # measurement.output_bytes handles np.ndarray via .nbytes


def make_calls(images, hf_legacy, hf_fast, dsl_fns, variants=None):
    """Return an ordered list of (label, callable) for one workload."""
    if variants is None:
        variants = ALL_VARIANTS
    selected = set(variants)
    dsl_fns = dsl_fns or {}
    all_calls = [
        ("hf_legacy", lambda: _wrap_hf(hf_legacy, images)),
        ("hf_fast",   lambda: _wrap_hf(hf_fast,   images)),
        ("dsl_v1",    lambda: _wrap_dsl(dsl_fns["dsl_v1"], images)),
        ("dsl_v2",    lambda: _wrap_dsl(dsl_fns["dsl_v2"], images)),
        ("dsl_v3",    lambda: _wrap_dsl(dsl_fns["dsl_v3"], images)),
    ]
    return [(name, fn) for name, fn in all_calls if name in selected]


# ---------------------------------------------------------------------------
# Memory phase
# ---------------------------------------------------------------------------

def run_memory_workload(label, images, hf_legacy, hf_fast, n_memory_warmup,
                        variants, dsl_fns=None):
    n = len(images)
    print(f"\n{'='*60}")
    print(f"[MEMORY] workload {label}: {n} images  (warmup={n_memory_warmup})")
    print(f"{'='*60}")

    results = {}
    for name, call in make_calls(images, hf_legacy, hf_fast, dsl_fns, variants):
        ratio = measure_memory(f"{name} ({label})", call, n_memory_warmup)
        results[name] = ratio

    print(f"\n=== memory summary {label} ===")
    for name, ratio in results.items():
        print(f"  {name:<20} peak/output: {ratio:.2f}x")
    return results


# ---------------------------------------------------------------------------
# Timing phase
# ---------------------------------------------------------------------------

def run_timing_workload(label, images, hf_legacy, hf_fast, n_warmup, n_timed,
                        variants, dsl_fns=None):
    n = len(images)
    print(f"\n{'='*60}")
    print(f"[TIMING] workload {label}: {n} images  (warmup={n_warmup}, timed={n_timed})")
    print(f"{'='*60}")

    results = {}
    for name, call in make_calls(images, hf_legacy, hf_fast, dsl_fns, variants):
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
        description="DSL LLaVA-NeXT benchmark: HF fast/legacy vs dsl_v1/v2/v3 on W2, W3, W4.")
    ap.add_argument("--num-threads", type=int, default=1, metavar="N",
                    help="OMP/MKL/torch intra-op thread count (default: 1).")
    ap.add_argument("--n-memory-warmup", type=int, default=N_MEMORY_WARMUP,
                    help="Unmeasured warmup calls before each memory measurement.")
    ap.add_argument("--skip-warmup", action="store_true",
                    help="Set n_memory_warmup=0.")
    ap.add_argument("--n-warmup",    type=int, default=N_WARMUP)
    ap.add_argument("--n-timed",     type=int, default=N_TIMED)
    ap.add_argument("--n-warmup-w4", type=int, default=N_WARMUP_W4)
    ap.add_argument("--n-timed-w4",  type=int, default=N_TIMED_W4)
    ap.add_argument("--memory-only", action="store_true", help="Skip timing phase.")
    ap.add_argument("--timing-only", action="store_true",
                    help="Skip memory phase (RSS results will be tainted by timing warmup).")
    ap.add_argument("--workloads", nargs="+", choices=["W2", "W3", "W4"],
                    default=["W2", "W3", "W4"], metavar="W")
    ap.add_argument("--variants", nargs="+", choices=ALL_VARIANTS,
                    default=None, metavar="V",
                    help=f"Variants to benchmark (default: all). Choices: {ALL_VARIANTS}.")
    args = ap.parse_args()

    if args.timing_only and args.memory_only:
        raise SystemExit("--memory-only and --timing-only are mutually exclusive")

    n_threads = args.num_threads
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    torch.set_num_threads(n_threads)
    numba.set_num_threads(n_threads)

    n_memory_warmup = 0 if args.skip_warmup else args.n_memory_warmup

    print(env_info())

    hf_legacy = get_llava_hf_processor(use_fast=False)
    hf_fast   = get_llava_hf_processor(use_fast=True)
    print(f"\nLoaded: llava-hf/llava-v1.6-mistral-7b-hf (legacy, fast)")

    selected_workloads = set(args.workloads)
    images_w2 = load_images(n_images=32, img_size=(1024, 1024)) if "W2" in selected_workloads else None
    images_w3 = load_images_w3()                                 if "W3" in selected_workloads else None
    images_w4 = load_images_w4()                                 if "W4" in selected_workloads else None
    if images_w2: print(f"\nW2: {len(images_w2)} images @ {images_w2[0].size}")
    if images_w3: print(f"W3: {len(images_w3)} images, sizes e.g. {[img.size for img in images_w3[:3]]}")
    if images_w4: print(f"W4: {len(images_w4)} images @ {images_w4[0].size}")

    variants = args.variants  # None → all

    # Trigger Numba JIT compilation before any measurement.
    first_images = images_w2 or images_w3 or images_w4
    assert first_images is not None

    needs_dsl_v1 = variants is None or "dsl_v1" in variants
    needs_dsl_v2 = variants is None or "dsl_v2" in variants
    needs_dsl_v3 = variants is None or "dsl_v3" in variants

    dsl_fns = {}
    if needs_dsl_v1 or needs_dsl_v2 or needs_dsl_v3:
        print("\nBuilding DSL variants and warming up Numba JIT (~30s each)...")
        if needs_dsl_v1:
            dsl_fns["dsl_v1"] = build_llava(llava_pipeline, sched_v1)
            _wrap_dsl(dsl_fns["dsl_v1"], first_images[:1])
        if needs_dsl_v2:
            dsl_fns["dsl_v2"] = build_llava(llava_pipeline, sched_v2)
            _wrap_dsl(dsl_fns["dsl_v2"], first_images[:1])
        if needs_dsl_v3:
            dsl_fns["dsl_v3"] = build_llava(llava_pipeline, sched_v3)
            _wrap_dsl(dsl_fns["dsl_v3"], first_images[:1])
        print("DSL JIT ready.")

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
            run_memory_workload(label, images, hf_legacy, hf_fast,
                                n_memory_warmup, variants, dsl_fns)

    # -----------------------------------------------------------------------
    # Phase 2: Timing — runs after memory so allocator warmup doesn't affect RSS.
    # -----------------------------------------------------------------------
    if not args.memory_only:
        print("\n" + "#"*60)
        print("# PHASE 2: TIMING")
        print("#"*60)
        for label, images, n_warmup, n_timed in workloads:
            run_timing_workload(label, images, hf_legacy, hf_fast,
                                n_warmup, n_timed, variants, dsl_fns)


if __name__ == "__main__":
    main()
