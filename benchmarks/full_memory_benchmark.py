"""Full memory baseline — Qwen2.5-VL, InternVL2.5, InternVL3.5, and LLaVA on W2, W3 and W4.

Important note:
  Do not use benchmarks/full_benchmark.py for memory numbers. That script is
  the runtime/cProfile benchmark, and the earlier memory measurements taken
  from it were invalid because they were mixed into the same long-lived process
  after repeated warmup, timing, and profiling calls. RSS is process-wide, so
  allocator pools and intermediate buffers retained by previous variants or
  workloads contaminated the baseline and could make the peak delta look too
  small, too large, or even zero.

  This file exists as a separate benchmark so memory is measured in the cleanest
  protocol we can use without changing the processor implementations: run only
  one processor call under measure_peak_rss(), avoid the timing loop and
  cProfile overhead, and optionally perform a controlled unmeasured warmup
  immediately before the measurement. This keeps the runtime benchmark and the
  memory benchmark separate, reproducible, and easier to interpret.

Workloads:
  W2: 32 images, all 1024×1024 (uniform size baseline).
  W3: 32 images with random sizes and aspect ratio in [0.5, 2.0].
  W4: 8 images at 2480x3508 (A4 @ 300 dpi).

Processor variants:
  Qwen2.5-VL:  legacy (use_fast=False) and fast (use_fast=True).
  InternVL2.5: manual model-card pipeline (dynamic tiling).
  InternVL3.5: HF GotOcr2ImageProcessorFast with crop_to_patches=True.
  LLaVA-NeXT:  legacy (use_fast=False) and fast (use_fast=True).

Protocol:
  - Thread count: 1 by default; use --num-threads N to enable multi-thread.
  - Optional unmeasured warmup before each memory measurement.
  - measure_peak_rss() around one processor call, no timing loop, no cProfile.
  - Pre-decoded PIL images in RAM, no GPU.

Usage:
  python full_memory_benchmark.py [--num-threads N] [--model MODEL]
  python full_memory_benchmark.py --skip-warmup
  MODEL choices: all (default), qwen, internvl25, internvl35, llava
"""
import os

import argparse

import torch

from data import load_images, load_images_w3, load_images_w4
from measurement import env_info, measure_memory
from models import (
    load_processors,
    get_internvl_manual_processor,
    get_internvl35_hf_processor,
    MODEL_ID_QWEN,
    MODEL_ID_INTERNVL35,
    MODEL_ID_LLAVA,
)

N_MEMORY_WARMUP = 1


def run_workload(label, images,
                 q_slow, q_fast,
                 iv_manual,
                 iv35,
                 llava_slow, llava_fast,
                 n_memory_warmup, model_filter="all"):
    n = len(images)

    print(f"\n{'='*60}")
    print(f"workload {label}: {n} images  (memory warmup={n_memory_warmup})")
    print(f"{'='*60}")

    results = {}

    if model_filter in ("all", "qwen"):
        results["q_slow"] = measure_memory(
            f"Qwen2.5-VL legacy ({label})",
            lambda: q_slow(images=images, return_tensors="np"),
            n_memory_warmup,
        )
        results["q_fast"] = measure_memory(
            f"Qwen2.5-VL fast ({label})",
            lambda: q_fast(images=images, return_tensors="pt"),
            n_memory_warmup,
        )

    if model_filter in ("all", "internvl25"):
        results["iv_manual"] = measure_memory(
            f"InternVL2.5 Manual Card ({label})",
            lambda: iv_manual(images),
            n_memory_warmup,
        )

    if model_filter in ("all", "internvl35"):
        results["iv35"] = measure_memory(
            f"InternVL3.5 HF ({label})",
            lambda: iv35(images),
            n_memory_warmup,
        )

    if model_filter in ("all", "llava"):
        results["llava_slow"] = measure_memory(
            f"LLaVA legacy ({label})",
            lambda: llava_slow(images=images, return_tensors="np"),
            n_memory_warmup,
        )
        results["llava_fast"] = measure_memory(
            f"LLaVA fast ({label})",
            lambda: llava_fast(images=images, return_tensors="pt"),
            n_memory_warmup,
        )

    print(f"\n=== memory summary {label} ===")
    if "q_slow" in results:
        print(f"  Qwen legacy peak/output:        {results['q_slow']:.2f}x")
    if "q_fast" in results:
        print(f"  Qwen fast peak/output:          {results['q_fast']:.2f}x")
    if "iv_manual" in results:
        print(f"  IV25 manual peak/output:        {results['iv_manual']:.2f}x")
    if "iv35" in results:
        print(f"  IV35 HF peak/output:            {results['iv35']:.2f}x")
    if "llava_slow" in results:
        print(f"  LLaVA legacy peak/output:       {results['llava_slow']:.2f}x")
    if "llava_fast" in results:
        print(f"  LLaVA fast peak/output:         {results['llava_fast']:.2f}x")


def main():
    ap = argparse.ArgumentParser(
        description="Full memory baseline: Qwen, InternVL2.5, InternVL3.5, LLaVA on W2, W3 and W4.")
    ap.add_argument("--n-memory-warmup", type=int, default=N_MEMORY_WARMUP,
                    help="Number of unmeasured warmup calls before each memory measurement.")
    ap.add_argument("--skip-warmup", action="store_true",
                    help="Skip the unmeasured memory warmup calls.")
    ap.add_argument(
        "--num-threads", type=int, default=1, metavar="N",
        help="OMP/MKL/torch intra-op thread count (default: 1 = single-thread baseline).",
    )
    ap.add_argument(
        "--model", default="all",
        choices=["all", "qwen", "internvl25", "internvl35", "llava"],
        help="Run only a specific model's variants (default: all).",
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

    model_filter = args.model

    q_slow = q_fast = iv_manual = iv35 = llava_slow = llava_fast = None

    if model_filter in ("all", "qwen"):
        q_slow, q_fast = load_processors(MODEL_ID_QWEN)
        print(f"Loaded: {MODEL_ID_QWEN} (legacy, fast)")

    if model_filter in ("all", "internvl25"):
        iv_manual = get_internvl_manual_processor(max_num=12)
        print("Loaded: OpenGVLab/InternVL2_5-8B (manual card)")

    if model_filter in ("all", "internvl35"):
        iv35 = get_internvl35_hf_processor()
        print(f"Loaded: {MODEL_ID_INTERNVL35} (HF GotOcr2ImageProcessorFast, crop_to_patches=True)")

    if model_filter in ("all", "llava"):
        llava_slow, llava_fast = load_processors(MODEL_ID_LLAVA)
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
        q_slow, q_fast, iv_manual, iv35, llava_slow, llava_fast,
        n_memory_warmup=n_memory_warmup, model_filter=model_filter,
    )
    run_workload(
        "W3", images_w3,
        q_slow, q_fast, iv_manual, iv35, llava_slow, llava_fast,
        n_memory_warmup=n_memory_warmup, model_filter=model_filter,
    )
    run_workload(
        "W4", images_w4,
        q_slow, q_fast, iv_manual, iv35, llava_slow, llava_fast,
        n_memory_warmup=n_memory_warmup, model_filter=model_filter,
    )


if __name__ == "__main__":
    main()
