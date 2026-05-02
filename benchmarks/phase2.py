"""Phase 2 — second model, generalize the finding.

Benchmarks InternVL2.5-8B preprocessing in three configurations:
  1. HF AutoImageProcessor (use_fast=False) — generic CLIP stub, single 448×448 crop.
  2. HF AutoImageProcessor (use_fast=True)  — same, torchvision backend.
  3. Manual model-card pipeline             — dynamic tiling + thumbnail, as written
                                              in the OpenGVLab/InternVL2_5-8B Quickstart.

Comparing (1)/(2) against (3) isolates the cost of the full tiling logic vs. the
CLIP stub, and shows whether the Phase 1 bottleneck pattern holds across processor
families.

Cross-cutting protocol (per the implementation plan):
  - Single thread (OMP/MKL/torch all = 1).
  - 10 warmup + 100 timed iterations, time.perf_counter_ns().
  - gc disabled around timed loop.
  - Pre-decoded PIL images in RAM.

Outputs:
  - profiles/IV25_W2_hf_legacy.txt   (cProfile + memory header)
  - profiles/IV25_W2_hf_fast.txt
  - profiles/IV25_W2_manual.txt
  - stdout summary with timing, peak/output ratio, env info.

Memory caveat: tracemalloc tracks PyMem allocations and undercounts torch
tensors (which use their own allocator). The manual path's ratio is a lower bound.

Usage:
  python phase2.py                       # synthetic random images
  python phase2.py --img-dir /path/imgs  # real images (e.g. COCO val2017)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from pathlib import Path

import torch
torch.set_num_threads(1)

from data import load_images
from measurement import time_fn, profile_and_measure, env_info
from models import get_internvl_hf_processor, get_internvl_manual_processor

N_IMAGES = 32
IMG_SIZE = (1024, 1024)
N_WARMUP = 10
N_TIMED = 100


def run(name, call, n_images, profile_path, n_warmup, n_timed):
    median_ms, p95_p50 = time_fn(call, n_warmup=n_warmup, n_timed=n_timed, desc=name)
    peak, out_b, ratio = profile_and_measure(name, call, profile_path)

    print(f"\n--- {name} ---")
    print(f"  median:        {median_ms:8.2f} ms/batch ({median_ms/n_images:.3f} ms/img)")
    print(f"  p95 - p50:     {p95_p50:8.2f} ms")
    print(f"  output:        {out_b/1e6:.2f} MB")
    print(f"  peak alloc:    {peak/1e6:.2f} MB")
    print(f"  peak / output: {ratio:.2f}x")
    print(f"  profile:       {profile_path}")
    return median_ms


def main():
    ap = argparse.ArgumentParser(description="Phase 2: profile InternVL2.5 preprocessing on W2.")
    ap.add_argument("--img-dir", default=None,
                    help="Directory of source images (recursive). Falls back to synthetic if omitted.")
    ap.add_argument("--profiles-dir", default="profiles")
    ap.add_argument("--n-images", type=int, default=N_IMAGES)
    ap.add_argument("--img-size", type=int, nargs=2, default=list(IMG_SIZE), metavar=("W", "H"))
    ap.add_argument("--n-warmup", type=int, default=N_WARMUP)
    ap.add_argument("--n-timed", type=int, default=N_TIMED)
    args = ap.parse_args()

    img_size = tuple(args.img_size)
    print(env_info())
    images = load_images(args.img_dir, n_images=args.n_images, img_size=img_size)
    print(f"\nworkload W2: {len(images)} x {img_size} "
          f"({'real images from ' + args.img_dir if args.img_dir else 'synthetic'})")

    pdir = Path(args.profiles_dir)
    hf_slow = get_internvl_hf_processor(use_fast=False)
    hf_fast = get_internvl_hf_processor(use_fast=True)
    manual = get_internvl_manual_processor(max_num=12)

    print("Loaded processors: OpenGVLab/InternVL2_5-8B (HF legacy, HF fast, manual card)")

    hf_slow_ms = run(
        "InternVL2.5 HF Legacy (W2)",
        lambda: hf_slow(images=images, return_tensors="np"),
        len(images), pdir / "IV25_W2_hf_legacy.txt",
        args.n_warmup, args.n_timed,
    )
    hf_fast_ms = run(
        "InternVL2.5 HF Fast (W2)",
        lambda: hf_fast(images=images, return_tensors="pt"),
        len(images), pdir / "IV25_W2_hf_fast.txt",
        args.n_warmup, args.n_timed,
    )
    manual_ms = run(
        "InternVL2.5 Manual Card (W2)",
        lambda: manual(images),
        len(images), pdir / "IV25_W2_manual.txt",
        args.n_warmup, args.n_timed,
    )

    print(f"\n=== summary ===")
    print(f"  HF legacy / HF fast ratio:   {hf_slow_ms / hf_fast_ms:.2f}x")
    print(f"  manual / HF fast ratio:      {manual_ms / hf_fast_ms:.2f}x")


if __name__ == "__main__":
    main()
