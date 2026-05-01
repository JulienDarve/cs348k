"""Phase 1 — one model, real profile.

Builds on Phase 0 by adding cProfile (top 20 by cumulative time) and
tracemalloc peak-bytes measurement, plus a peak/output memory ratio.

Cross-cutting protocol (per the implementation plan):
  - Single thread (OMP/MKL/torch all = 1).
  - 10 warmup + 100 timed iterations, time.perf_counter_ns().
  - gc disabled around timed loop.
  - Pre-decoded PIL images in RAM.

Outputs:
  - profiles/Q25_W2_legacy.txt   (cProfile + memory header)
  - profiles/Q25_W2_fast.txt
  - stdout summary with timing, peak/output ratio, env info.

Memory caveat: tracemalloc tracks PyMem allocations and undercounts torch
tensors (which use their own allocator). The legacy NumPy/PIL path is well
covered; the fast/torchvision path's ratio is a lower bound.

Usage:
  python phase1.py                       # synthetic random images
  python phase1.py --img-dir /path/imgs  # real images (e.g. COCO val2017)
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
from models import load_processors

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
N_IMAGES = 32
IMG_SIZE = (1024, 1024)
N_WARMUP = 10
N_TIMED = 100


def run(name, processor, images, return_tensors, profile_path, n_warmup, n_timed):
    call = lambda: processor(images=images, return_tensors=return_tensors)

    median_ms, p95_p50 = time_fn(call, n_warmup=n_warmup, n_timed=n_timed)
    peak, out_b, ratio = profile_and_measure(name, call, profile_path)

    print(f"\n--- {name} ---")
    print(f"  median:        {median_ms:8.2f} ms/batch ({median_ms/len(images):.3f} ms/img)")
    print(f"  p95 - p50:     {p95_p50:8.2f} ms")
    print(f"  output:        {out_b/1e6:.2f} MB")
    print(f"  peak alloc:    {peak/1e6:.2f} MB")
    print(f"  peak / output: {ratio:.2f}x")
    print(f"  profile:       {profile_path}")
    return median_ms


def main():
    ap = argparse.ArgumentParser(description="Phase 1: profile Qwen2.5-VL preprocessing on W2.")
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
    slow, fast = load_processors(MODEL_ID)

    print(f"Loaded models: {MODEL_ID}")

    slow_ms = run("Qwen2.5-VL legacy (W2)", slow, images, "np", pdir / "Q25_W2_legacy.txt",
                  args.n_warmup, args.n_timed)
    fast_ms = run("Qwen2.5-VL fast (W2)",   fast, images, "pt", pdir / "Q25_W2_fast.txt",
                  args.n_warmup, args.n_timed)

    print(f"\n=== summary ===")
    print(f"  legacy / fast ratio: {slow_ms / fast_ms:.2f}x")


if __name__ == "__main__":
    main()
