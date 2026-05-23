"""Full benchmark baseline — Qwen2.5-VL, InternVL2.5, InternVL3.5, and LLaVA on W2, W3 and W4.

Workloads:
  W2: 32 images, all 1024×1024 (uniform size baseline).
  W3: 32 images with random sizes [512, 2048] and aspect ratio in [0.5, 2.0].
  W4: 8 images at 2480×3508 (A4 @ 300 dpi). Fewer iterations due to size.

Processor variants:
  Qwen2.5-VL:  legacy (use_fast=False) and fast (use_fast=True).
  InternVL2.5: manual model-card pipeline (dynamic tiling).
  InternVL3.5: HF legacy (use_fast=False) and fast (use_fast=True), both with crop_to_patches=True.
  LLaVA-NeXT:  legacy (use_fast=False) and fast (use_fast=True).

Protocol:
  - Thread count: 1 by default; use --num-threads N to enable multi-thread.
  - W3: 10 warmup + 100 timed iterations.
  - W4: 2 warmup + 8 timed iterations (large images; keep wall time reasonable).
  - time.perf_counter_ns(), gc disabled during timed loop.
  - Pre-decoded PIL images in RAM, no GPU.
  - Optional torch.compile: use --torch-compile to wrap each processor with
    torch.compile(dynamic=True). Recommend --n-warmup >= 20 so compilation
    completes before the timed loop.

Profile outputs (one .txt per variant, grouped by workload subdirectory):
  profiles/W2/{Q25_legacy,Q25_fast,IV25_manual,IV35_hf_legacy,IV35_hf_fast,LLaVA_legacy,LLaVA_fast}.txt
  profiles/W3/{Q25_legacy,Q25_fast,IV25_manual,IV35_hf_legacy,IV35_hf_fast,LLaVA_legacy,LLaVA_fast}.txt
  profiles/W4/{Q25_legacy,Q25_fast,IV25_manual,IV35_hf_legacy,IV35_hf_fast,LLaVA_legacy,LLaVA_fast}.txt

  Use --profiles-dir to keep results from different thread counts in separate directories.

Usage:
  python full_benchmark.py [--num-threads N] [--model MODEL]
  python full_benchmark.py --torch-compile [--compile-mode MODE]
  MODEL choices: all (default), qwen, internvl25, internvl35, llava
"""
import os

import argparse
from pathlib import Path

import torch

from data import load_images, load_images_w3, load_images_w4
from measurement import time_fn, profile_fn, env_info
from models import (
    load_processors,
    get_internvl_manual_processor,
    get_internvl35_hf_processor,
    MODEL_ID_QWEN,
    MODEL_ID_INTERNVL35,
    MODEL_ID_LLAVA,
)

N_WARMUP = 10
N_TIMED = 100
N_WARMUP_W4 = 2
N_TIMED_W4 = 8
_COMPILE_WARMUP_MIN = 20
_COMPILE_WARMUP_W4_MIN = 4


def _apply_compile(fn, mode):
    return torch.compile(fn, mode=mode, dynamic=True)


def run(name, call, n_images, profile_path, n_warmup, n_timed):
    median_ms, p95_p50 = time_fn(call, n_warmup=n_warmup, n_timed=n_timed, desc=name)
    profile_fn(name, call, profile_path)

    print(f"\n--- {name} ---")
    print(f"  median:        {median_ms:8.2f} ms/batch ({median_ms/n_images:.3f} ms/img)")
    print(f"  p95 - p50:     {p95_p50:8.2f} ms")
    print(f"  profile:       {profile_path}")
    return median_ms


def run_workload(label, images, pdir,
                 q_slow, q_fast,
                 iv_manual,
                 iv35_slow, iv35_fast,
                 llava_slow, llava_fast,
                 n_warmup, n_timed, model_filter="all"):
    n = len(images)
    wdir = pdir / label.upper()

    print(f"\n{'='*60}")
    print(f"workload {label}: {n} images  (warmup={n_warmup}, timed={n_timed})")
    print(f"{'='*60}")

    results = {}

    if model_filter in ("all", "qwen"):
        results["q_slow"] = run(
            f"Qwen2.5-VL legacy ({label})",
            lambda: q_slow(images=images, return_tensors="np"),
            n, wdir / "Q25_legacy.txt",
            n_warmup, n_timed,
        )
        results["q_fast"] = run(
            f"Qwen2.5-VL fast ({label})",
            lambda: q_fast(images=images, return_tensors="pt"),
            n, wdir / "Q25_fast.txt",
            n_warmup, n_timed,
        )

    if model_filter in ("all", "internvl25"):
        results["iv_manual"] = run(
            f"InternVL2.5 Manual Card ({label})",
            lambda: iv_manual(images),
            n, wdir / "IV25_manual.txt",
            n_warmup, n_timed,
        )

    if model_filter in ("all", "internvl35"):
        results["iv35_slow"] = run(
            f"InternVL3.5 HF legacy ({label})",
            lambda: iv35_slow(images),
            n, wdir / "IV35_hf_legacy.txt",
            n_warmup, n_timed,
        )
        results["iv35_fast"] = run(
            f"InternVL3.5 HF fast ({label})",
            lambda: iv35_fast(images),
            n, wdir / "IV35_hf_fast.txt",
            n_warmup, n_timed,
        )

    if model_filter in ("all", "llava"):
        results["llava_slow"] = run(
            f"LLaVA legacy ({label})",
            lambda: llava_slow(images=images, return_tensors="np"),
            n, wdir / "LLaVA_legacy.txt",
            n_warmup, n_timed,
        )
        results["llava_fast"] = run(
            f"LLaVA fast ({label})",
            lambda: llava_fast(images=images, return_tensors="pt"),
            n, wdir / "LLaVA_fast.txt",
            n_warmup, n_timed,
        )

    print(f"\n=== summary {label} ===")
    if "q_slow" in results and "q_fast" in results:
        print(f"  Qwen legacy / fast ratio:          {results['q_slow'] / results['q_fast']:.2f}x")
    if "iv35_slow" in results and "iv35_fast" in results:
        print(f"  IV35 HF legacy / fast ratio:       {results['iv35_slow'] / results['iv35_fast']:.2f}x")
    if "iv_manual" in results and "iv35_fast" in results:
        print(f"  IV25 manual / IV35 HF fast ratio:  {results['iv_manual'] / results['iv35_fast']:.2f}x")
    if "q_fast" in results and "iv_manual" in results:
        print(f"  Qwen fast / IV25 manual ratio:     {results['q_fast'] / results['iv_manual']:.2f}x")
    if "q_fast" in results and "iv35_fast" in results:
        print(f"  Qwen fast / IV35 HF fast ratio:    {results['q_fast'] / results['iv35_fast']:.2f}x")
    if "llava_slow" in results and "llava_fast" in results:
        print(f"  LLaVA legacy / fast ratio:         {results['llava_slow'] / results['llava_fast']:.2f}x")


def main():
    ap = argparse.ArgumentParser(
        description="Full benchmark baseline: Qwen, InternVL2.5, InternVL3.5, LLaVA on W2, W3 and W4.")
    ap.add_argument("--profiles-dir", default="profiles")
    ap.add_argument("--n-warmup", type=int, default=N_WARMUP)
    ap.add_argument("--n-timed", type=int, default=N_TIMED)
    ap.add_argument("--n-warmup-w4", type=int, default=N_WARMUP_W4)
    ap.add_argument("--n-timed-w4", type=int, default=N_TIMED_W4)
    ap.add_argument(
        "--num-threads", type=int, default=1, metavar="N",
        help="OMP/MKL/torch intra-op thread count (default: 1 = single-thread baseline).",
    )
    ap.add_argument(
        "--model", default="all",
        choices=["all", "qwen", "internvl25", "internvl35", "llava"],
        help="Run only a specific model's variants (default: all).",
    )
    ap.add_argument(
        "--torch-compile", action="store_true",
        help="Wrap each processor with torch.compile() before benchmarking.",
    )
    ap.add_argument(
        "--compile-mode", default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode (default: 'default'). "
             "'reduce-overhead' is mainly useful on GPU; 'max-autotune' is slow to compile.",
    )
    args = ap.parse_args()

    n_threads = args.num_threads
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    torch.set_num_threads(n_threads)

    print(env_info())

    pdir = Path(args.profiles_dir)
    pdir.mkdir(exist_ok=True)

    model_filter = args.model

    q_slow = q_fast = iv_manual = iv35_slow = iv35_fast = llava_slow = llava_fast = None

    if model_filter in ("all", "qwen"):
        q_slow, q_fast = load_processors(MODEL_ID_QWEN)
        print(f"Loaded: {MODEL_ID_QWEN} (legacy, fast)")

    if model_filter in ("all", "internvl25"):
        iv_manual = get_internvl_manual_processor(max_num=12)
        print("Loaded: OpenGVLab/InternVL2_5-8B (manual card)")

    if model_filter in ("all", "internvl35"):
        iv35_slow = get_internvl35_hf_processor(use_fast=False)
        iv35_fast = get_internvl35_hf_processor(use_fast=True)
        print(f"Loaded: {MODEL_ID_INTERNVL35} (HF legacy, HF fast, crop_to_patches=True)")

    if model_filter in ("all", "llava"):
        llava_slow, llava_fast = load_processors(MODEL_ID_LLAVA)
        print(f"Loaded: {MODEL_ID_LLAVA} (legacy, fast)")

    if args.torch_compile:
        print(f"\n[torch.compile] mode={args.compile_mode!r}, dynamic=True")
        if args.n_warmup < _COMPILE_WARMUP_MIN:
            print(
                f"[torch.compile] WARNING: --n-warmup={args.n_warmup} may be too low. "
                f"Recommend >= {_COMPILE_WARMUP_MIN} so compilation finishes before timing."
            )
        if args.n_warmup_w4 < _COMPILE_WARMUP_W4_MIN:
            print(
                f"[torch.compile] WARNING: --n-warmup-w4={args.n_warmup_w4} may be too low. "
                f"Recommend >= {_COMPILE_WARMUP_W4_MIN} for W4."
            )
        if q_slow    is not None: q_slow    = _apply_compile(q_slow,    args.compile_mode)
        if q_fast    is not None: q_fast    = _apply_compile(q_fast,    args.compile_mode)
        if iv_manual is not None: iv_manual = _apply_compile(iv_manual, args.compile_mode)
        if iv35_slow is not None: iv35_slow = _apply_compile(iv35_slow, args.compile_mode)
        if iv35_fast is not None: iv35_fast = _apply_compile(iv35_fast, args.compile_mode)
        if llava_slow is not None: llava_slow = _apply_compile(llava_slow, args.compile_mode)
        if llava_fast is not None: llava_fast = _apply_compile(llava_fast, args.compile_mode)

    images_w2 = load_images(n_images=32, img_size=(1024, 1024))
    images_w3 = load_images_w3()
    images_w4 = load_images_w4()

    print(f"\nW2 image size: {images_w2[0].size} x {len(images_w2)} images")
    sizes_w3 = [img.size for img in images_w3]
    print(f"W3 sizes (first 4): {sizes_w3[:4]} ...")
    print(f"W4 image size: {images_w4[0].size} x {len(images_w4)} images")

    run_workload(
        "W2", images_w2, pdir,
        q_slow, q_fast, iv_manual, iv35_slow, iv35_fast, llava_slow, llava_fast,
        n_warmup=args.n_warmup, n_timed=args.n_timed, model_filter=model_filter,
    )
    run_workload(
        "W3", images_w3, pdir,
        q_slow, q_fast, iv_manual, iv35_slow, iv35_fast, llava_slow, llava_fast,
        n_warmup=args.n_warmup, n_timed=args.n_timed, model_filter=model_filter,
    )
    run_workload(
        "W4", images_w4, pdir,
        q_slow, q_fast, iv_manual, iv35_slow, iv35_fast, llava_slow, llava_fast,
        n_warmup=args.n_warmup_w4, n_timed=args.n_timed_w4, model_filter=model_filter,
    )


if __name__ == "__main__":
    main()
