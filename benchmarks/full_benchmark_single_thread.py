"""Full single-thread baseline — Qwen2.5-VL, InternVL2.5, and LLaVA on W3 and W4.

Workloads:
  W3: 32 images with random sizes [512, 2048] and aspect ratio in [0.5, 2.0].
  W4: 8 images at 2480×3508 (A4 @ 300 dpi). Fewer iterations due to size.

Processor variants:
  Qwen2.5-VL:  legacy (use_fast=False) and fast (use_fast=True).
  InternVL2.5: HF legacy, HF fast, and manual model-card pipeline (dynamic tiling).
  LLaVA-NeXT:  legacy (use_fast=False) and fast (use_fast=True).

Protocol:
  - Single thread (OMP/MKL/torch all = 1).
  - W3: 10 warmup + 100 timed iterations.
  - W4: 2 warmup + 8 timed iterations (large images; keep wall time reasonable).
  - time.perf_counter_ns(), gc disabled during timed loop.
  - Pre-decoded PIL images in RAM, no GPU.

Profile outputs (one .txt per variant × workload):
  profiles/Q25_W{3,4}_{legacy,fast}.txt
  profiles/IV25_W{3,4}_{hf_legacy,hf_fast,manual}.txt
  profiles/LLaVA_W{3,4}_{legacy,fast}.txt

Usage:
  python full_benchmark_single_thread.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from pathlib import Path

import torch
torch.set_num_threads(1)

from data import load_images_w3, load_images_w4
from measurement import time_fn, profile_and_measure, env_info
from models import (
    load_processors,
    get_internvl_manual_processor,
    MODEL_ID_QWEN,
    MODEL_ID_INTERNVL,
    MODEL_ID_LLAVA,
)

N_WARMUP = 10
N_TIMED = 100
N_WARMUP_W4 = 2
N_TIMED_W4 = 8


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


def run_workload(label, images, pdir,
                 q_slow, q_fast,
                 iv_hf_slow, iv_hf_fast, iv_manual,
                 llava_slow, llava_fast,
                 n_warmup, n_timed):
    n = len(images)
    tag = label.upper()

    print(f"\n{'='*60}")
    print(f"workload {label}: {n} images  (warmup={n_warmup}, timed={n_timed})")
    print(f"{'='*60}")

    q_slow_ms = run(
        f"Qwen2.5-VL legacy ({label})",
        lambda: q_slow(images=images, return_tensors="np"),
        n, pdir / f"Q25_{tag}_legacy.txt",
        n_warmup, n_timed,
    )
    q_fast_ms = run(
        f"Qwen2.5-VL fast ({label})",
        lambda: q_fast(images=images, return_tensors="pt"),
        n, pdir / f"Q25_{tag}_fast.txt",
        n_warmup, n_timed,
    )
    iv_slow_ms = run(
        f"InternVL2.5 HF Legacy ({label})",
        lambda: iv_hf_slow(images=images, return_tensors="np"),
        n, pdir / f"IV25_{tag}_hf_legacy.txt",
        n_warmup, n_timed,
    )
    iv_fast_ms = run(
        f"InternVL2.5 HF Fast ({label})",
        lambda: iv_hf_fast(images=images, return_tensors="pt"),
        n, pdir / f"IV25_{tag}_hf_fast.txt",
        n_warmup, n_timed,
    )
    iv_manual_ms = run(
        f"InternVL2.5 Manual Card ({label})",
        lambda: iv_manual(images),
        n, pdir / f"IV25_{tag}_manual.txt",
        n_warmup, n_timed,
    )
    llava_slow_ms = run(
        f"LLaVA legacy ({label})",
        lambda: llava_slow(images=images, return_tensors="np"),
        n, pdir / f"LLaVA_{tag}_legacy.txt",
        n_warmup, n_timed,
    )
    llava_fast_ms = run(
        f"LLaVA fast ({label})",
        lambda: llava_fast(images=images, return_tensors="pt"),
        n, pdir / f"LLaVA_{tag}_fast.txt",
        n_warmup, n_timed,
    )

    print(f"\n=== summary {label} ===")
    print(f"  Qwen legacy / fast ratio:        {q_slow_ms / q_fast_ms:.2f}x")
    print(f"  IV25 HF legacy / HF fast ratio:  {iv_slow_ms / iv_fast_ms:.2f}x")
    print(f"  IV25 manual / HF fast ratio:     {iv_manual_ms / iv_fast_ms:.2f}x")
    print(f"  Qwen fast / IV25 manual ratio:   {q_fast_ms / iv_manual_ms:.2f}x")
    print(f"  LLaVA legacy / fast ratio:       {llava_slow_ms / llava_fast_ms:.2f}x")


def main():
    ap = argparse.ArgumentParser(
        description="Full single-thread baseline: Qwen, InternVL, LLaVA on W3 and W4.")
    ap.add_argument("--profiles-dir", default="profiles")
    ap.add_argument("--n-warmup", type=int, default=N_WARMUP)
    ap.add_argument("--n-timed", type=int, default=N_TIMED)
    ap.add_argument("--n-warmup-w4", type=int, default=N_WARMUP_W4)
    ap.add_argument("--n-timed-w4", type=int, default=N_TIMED_W4)
    args = ap.parse_args()

    print(env_info())

    pdir = Path(args.profiles_dir)
    pdir.mkdir(exist_ok=True)

    q_slow, q_fast = load_processors(MODEL_ID_QWEN)
    iv_hf_slow, iv_hf_fast = load_processors(MODEL_ID_INTERNVL, trust_remote_code=True)
    iv_manual = get_internvl_manual_processor(max_num=12)
    llava_slow, llava_fast = load_processors(MODEL_ID_LLAVA)

    print(f"\nLoaded: {MODEL_ID_QWEN} (legacy, fast)")
    print("Loaded: OpenGVLab/InternVL2_5-8B (HF legacy, HF fast, manual card)")
    print(f"Loaded: {MODEL_ID_LLAVA} (legacy, fast)")

    images_w3 = load_images_w3()
    images_w4 = load_images_w4()

    sizes_w3 = [img.size for img in images_w3]
    print(f"\nW3 sizes (first 4): {sizes_w3[:4]} ...")
    print(f"W4 image size: {images_w4[0].size} x {len(images_w4)} images")

    run_workload(
        "W3", images_w3, pdir,
        q_slow, q_fast, iv_hf_slow, iv_hf_fast, iv_manual, llava_slow, llava_fast,
        n_warmup=args.n_warmup, n_timed=args.n_timed,
    )
    run_workload(
        "W4", images_w4, pdir,
        q_slow, q_fast, iv_hf_slow, iv_hf_fast, iv_manual, llava_slow, llava_fast,
        n_warmup=args.n_warmup_w4, n_timed=args.n_timed_w4,
    )


if __name__ == "__main__":
    main()
