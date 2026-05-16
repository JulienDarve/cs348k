"""Full memory baseline — Qwen2.5-VL, InternVL2.5, and LLaVA on W2, W3 and W4.

Workloads:
  W2: 32 images, all 1024×1024 (uniform size baseline).
  W3: 32 images with random sizes and aspect ratio in [0.5, 2.0].
  W4: 8 images at 2480x3508 (A4 @ 300 dpi).

Processor variants:
  Qwen2.5-VL:  legacy (use_fast=False) and fast (use_fast=True).
  InternVL2.5: HF legacy, HF fast, and manual model-card pipeline (dynamic tiling).
  LLaVA-NeXT:  legacy (use_fast=False) and fast (use_fast=True).

Protocol:
  - Thread count: 1 by default; use --num-threads N to enable multi-thread.
  - Optional unmeasured warmup before each memory measurement.
  - measure_peak_rss() around one processor call, no timing loop, no cProfile.
  - Pre-decoded PIL images in RAM, no GPU.

Usage:
  python full_memory_benchmark.py
  python full_memory_benchmark.py --num-threads N
  python full_memory_benchmark.py --skip-warmup
"""
import os

import argparse

import torch

from data import load_images, load_images_w3, load_images_w4
from measurement import env_info, measure_memory
from models import (
    load_processors,
    get_internvl_manual_processor,
    MODEL_ID_QWEN,
    MODEL_ID_INTERNVL,
    MODEL_ID_LLAVA,
)

N_MEMORY_WARMUP = 1


def run_workload(label, images,
                 q_slow, q_fast,
                 iv_hf_slow, iv_hf_fast, iv_manual,
                 llava_slow, llava_fast,
                 n_memory_warmup):
    n = len(images)

    print(f"\n{'='*60}")
    print(f"workload {label}: {n} images  (memory warmup={n_memory_warmup})")
    print(f"{'='*60}")

    q_slow_ratio = measure_memory(
        f"Qwen2.5-VL legacy ({label})",
        lambda: q_slow(images=images, return_tensors="np"),
        n_memory_warmup,
    )
    q_fast_ratio = measure_memory(
        f"Qwen2.5-VL fast ({label})",
        lambda: q_fast(images=images, return_tensors="pt"),
        n_memory_warmup,
    )
    iv_slow_ratio = measure_memory(
        f"InternVL2.5 HF Legacy ({label})",
        lambda: iv_hf_slow(images=images, return_tensors="np"),
        n_memory_warmup,
    )
    iv_fast_ratio = measure_memory(
        f"InternVL2.5 HF Fast ({label})",
        lambda: iv_hf_fast(images=images, return_tensors="pt"),
        n_memory_warmup,
    )
    iv_manual_ratio = measure_memory(
        f"InternVL2.5 Manual Card ({label})",
        lambda: iv_manual(images),
        n_memory_warmup,
    )
    llava_slow_ratio = measure_memory(
        f"LLaVA legacy ({label})",
        lambda: llava_slow(images=images, return_tensors="np"),
        n_memory_warmup,
    )
    llava_fast_ratio = measure_memory(
        f"LLaVA fast ({label})",
        lambda: llava_fast(images=images, return_tensors="pt"),
        n_memory_warmup,
    )

    print(f"\n=== memory summary {label} ===")
    print(f"  Qwen legacy peak/output:        {q_slow_ratio:.2f}x")
    print(f"  Qwen fast peak/output:          {q_fast_ratio:.2f}x")
    print(f"  IV25 HF legacy peak/output:     {iv_slow_ratio:.2f}x")
    print(f"  IV25 HF fast peak/output:       {iv_fast_ratio:.2f}x")
    print(f"  IV25 manual peak/output:        {iv_manual_ratio:.2f}x")
    print(f"  LLaVA legacy peak/output:       {llava_slow_ratio:.2f}x")
    print(f"  LLaVA fast peak/output:         {llava_fast_ratio:.2f}x")


def main():
    ap = argparse.ArgumentParser(
        description="Full memory baseline: Qwen, InternVL, LLaVA on W3 and W4.")
    ap.add_argument("--n-memory-warmup", type=int, default=N_MEMORY_WARMUP,
                    help="Number of unmeasured warmup calls before each memory measurement.")
    ap.add_argument("--skip-warmup", action="store_true",
                    help="Skip the unmeasured memory warmup calls.")
    ap.add_argument(
        "--num-threads", type=int, default=1, metavar="N",
        help="OMP/MKL/torch intra-op thread count (default: 1 = single-thread baseline).",
    )
    args = ap.parse_args()

    n_memory_warmup = 0 if args.skip_warmup else args.n_memory_warmup
    if n_memory_warmup < 0:
        raise SystemExit("--n-memory-warmup must be >= 0")

    n_threads = args.num_threads
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    torch.set_num_threads(n_threads)

    print(env_info())

    q_slow, q_fast = load_processors(MODEL_ID_QWEN)
    iv_hf_slow, iv_hf_fast = load_processors(MODEL_ID_INTERNVL, trust_remote_code=True)
    iv_manual = get_internvl_manual_processor(max_num=12)
    llava_slow, llava_fast = load_processors(MODEL_ID_LLAVA)

    print(f"\nLoaded: {MODEL_ID_QWEN} (legacy, fast)")
    print("Loaded: OpenGVLab/InternVL2_5-8B (HF legacy, HF fast, manual card)")
    print(f"Loaded: {MODEL_ID_LLAVA} (legacy, fast)")

    images_w2 = load_images(n_images=32, img_size=(1024, 1024))
    images_w3 = load_images_w3()
    images_w4 = load_images_w4()

    print(f"\nW2 image size: {images_w2[0].size} x {len(images_w2)} images")
    sizes_w3 = [img.size for img in images_w3]
    print(f"W3 sizes (first 4): {sizes_w3[:4]} ...")
    print(f"W4 image size: {images_w4[0].size} x {len(images_w4)} images")

    run_workload(
        "W2", images_w2,
        q_slow, q_fast, iv_hf_slow, iv_hf_fast, iv_manual, llava_slow, llava_fast,
        n_memory_warmup=n_memory_warmup,
    )
    run_workload(
        "W3", images_w3,
        q_slow, q_fast, iv_hf_slow, iv_hf_fast, iv_manual, llava_slow, llava_fast,
        n_memory_warmup=n_memory_warmup,
    )
    run_workload(
        "W4", images_w4,
        q_slow, q_fast, iv_hf_slow, iv_hf_fast, iv_manual, llava_slow, llava_fast,
        n_memory_warmup=n_memory_warmup,
    )


if __name__ == "__main__":
    main()
