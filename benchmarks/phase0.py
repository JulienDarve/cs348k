# phase0.py
# Sanity check: is HF VLM preprocessing slow enough to care about?
# One workload (32 x 1024x1024), two systems (Qwen2.5-VL legacy vs fast).
# Prints median/p95 ms per call and per image. That's it.

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import gc
import time
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoImageProcessor

torch.set_num_interop_threads(1)
torch.set_num_threads(1)

MODEL    = "Qwen/Qwen2.5-VL-7B-Instruct"
N_IMAGES = 32
IMG_SIZE = 1024
N_WARMUP = 10
N_TIMED  = 100
SEED     = 0


def time_call(fn, label=""):
    for _ in tqdm(range(N_WARMUP), desc=f"{label} warmup", leave=False):
        fn()
    gc.collect()
    gc.disable()
    try:
        times_ns = [0] * N_TIMED
        for i in tqdm(range(N_TIMED), desc=f"{label} timed "):
            t0 = time.perf_counter_ns()
            fn()
            times_ns[i] = time.perf_counter_ns() - t0
    finally:
        gc.enable()
    arr_ms = np.array(times_ns) / 1e6
    return float(np.median(arr_ms)), float(np.percentile(arr_ms, 95))


def report(label, fn):
    med_ms, p95_ms = time_call(fn, label=label)
    per_img = med_ms / N_IMAGES
    print(f"{label}: median {med_ms:8.2f} ms/call  p95 {p95_ms:8.2f} ms  |  per-image median {per_img:7.2f} ms")


def main():
    n_alloc = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

    rng = np.random.default_rng(SEED)
    arrs = rng.integers(0, 256, size=(N_IMAGES, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    images = [Image.fromarray(a) for a in arrs]

    legacy = AutoImageProcessor.from_pretrained(MODEL, use_fast=False)
    fast   = AutoImageProcessor.from_pretrained(MODEL, use_fast=True)

    print(f"fast class: {type(fast).__name__} ({fast.__class__.__module__})")
    print(f"Model:    {MODEL}")
    print(f"Workload: {N_IMAGES} x {IMG_SIZE}x{IMG_SIZE} synthetic uint8 images")
    print(f"Allocated CPUs: {n_alloc} (SLURM_CPUS_PER_TASK or os.cpu_count())")
    print(f"Warmup:   {N_WARMUP}, Timed: {N_TIMED}")
    print()

    # Note: return_tensors="pt" not supported on Q25F fast

    # 1-thread block
    torch.set_num_threads(1)
    print(f"--- torch.get_num_threads() = {torch.get_num_threads()} ---")
    report("Q25  legacy np 1t", lambda: legacy(images=images, return_tensors="np"))
    report("Q25F fast   pt 1t", lambda: fast(images=images, return_tensors="pt"))

    # multi-thread block (only if we actually have cores to use)
    if n_alloc > 1:
        torch.set_num_threads(n_alloc)
        print(f"--- torch.get_num_threads() = {torch.get_num_threads()} ---")
        report(f"Q25  legacy np {n_alloc}t", lambda: legacy(images=images, return_tensors="np"))
        report(f"Q25F fast   pt {n_alloc}t", lambda: fast(images=images, return_tensors="pt"))
        torch.set_num_threads(1)
    else:
        print(f"Skipping multi-thread test: only {n_alloc} CPU allocated. "
              f"Re-run with sbatch --cpus-per-task=8 --exclusive.")


if __name__ == "__main__":
    main()
