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
import cProfile
import gc
import io
import platform
import pstats
import time
import tracemalloc
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor

torch.set_num_threads(1)

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
N_IMAGES = 32
IMG_SIZE = (1024, 1024)
N_WARMUP = 10
N_TIMED = 100


def load_images(img_dir):
    """Pre-decode N_IMAGES RGB PIL images at IMG_SIZE into RAM.

    Reads from img_dir if given, otherwise generates synthetic random images
    (preprocessing cost is content-independent, so this is fine for profiling).
    """
    if img_dir:
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        paths = sorted(p for p in Path(img_dir).rglob("*") if p.suffix.lower() in exts)
        if len(paths) < N_IMAGES:
            raise SystemExit(f"need >= {N_IMAGES} images in {img_dir}, found {len(paths)}")
        return [Image.open(p).convert("RGB").resize(IMG_SIZE) for p in paths[:N_IMAGES]]
    rng = np.random.default_rng(0)
    arrs = rng.integers(0, 256, size=(N_IMAGES, IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8)
    return [Image.fromarray(a) for a in arrs]


def time_fn(fn):
    for _ in range(N_WARMUP):
        fn()
    gc.collect()
    gc.disable()
    try:
        ts = []
        for _ in range(N_TIMED):
            t0 = time.perf_counter_ns()
            fn()
            ts.append(time.perf_counter_ns() - t0)
    finally:
        gc.enable()
    a = np.array(ts, dtype=np.float64) / 1e6
    return float(np.median(a)), float(np.percentile(a, 95) - np.percentile(a, 50))


def output_bytes(out):
    pv = out["pixel_values"]
    if isinstance(pv, torch.Tensor):
        return pv.element_size() * pv.nelement()
    if isinstance(pv, np.ndarray):
        return int(pv.nbytes)
    if isinstance(pv, (list, tuple)):
        return sum(output_bytes({"pixel_values": p}) for p in pv)
    raise TypeError(f"unexpected pixel_values type: {type(pv)}")


def output_shape(out):
    pv = out["pixel_values"]
    if isinstance(pv, (torch.Tensor, np.ndarray)):
        return tuple(pv.shape)
    if isinstance(pv, (list, tuple)):
        return [output_shape({"pixel_values": p}) for p in pv]
    return "?"


def profile_and_measure(name, fn, out_path, top_n=20):
    """Run fn under cProfile, then under tracemalloc; write everything to out_path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pr = cProfile.Profile()
    pr.enable()
    fn()
    pr.disable()
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(top_n)
    profile_text = buf.getvalue()

    gc.collect()
    tracemalloc.start()
    try:
        result = fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    out_b = output_bytes(result)
    ratio = peak / out_b if out_b > 0 else float("nan")

    header = (
        f"=== {name} ===\n"
        f"output shape:    {output_shape(result)}\n"
        f"output bytes:    {out_b/1e6:.2f} MB\n"
        f"peak allocated:  {peak/1e6:.2f} MB  (tracemalloc; undercounts torch allocator)\n"
        f"peak / output:   {ratio:.2f}x\n\n"
    )
    out_path.write_text(header + profile_text)
    return peak, out_b, ratio


def env_info():
    import transformers, torchvision, PIL
    return (
        f"python={platform.python_version()} platform={platform.platform()}\n"
        f"cpu={platform.processor() or platform.machine()}\n"
        f"transformers={transformers.__version__} torch={torch.__version__} "
        f"torchvision={torchvision.__version__} pillow={PIL.__version__} "
        f"numpy={np.__version__}\n"
        f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} "
        f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')} "
        f"torch_threads={torch.get_num_threads()}"
    )


def run(name, processor, images, return_tensors, profile_path):
    call = lambda: processor(images=images, return_tensors=return_tensors)

    median_ms, p95_p50 = time_fn(call)
    peak, out_b, ratio = profile_and_measure(name, call, profile_path)

    print(f"\n--- {name} ---")
    print(f"  median:        {median_ms:8.2f} ms/batch ({median_ms/N_IMAGES:.3f} ms/img)")
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
    args = ap.parse_args()

    print(env_info())
    images = load_images(args.img_dir)
    print(f"\nworkload W2: {len(images)} x {IMG_SIZE} "
          f"({'real images from ' + args.img_dir if args.img_dir else 'synthetic'})")

    pdir = Path(args.profiles_dir)
    slow = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=False)
    fast = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=True)

    slow_ms = run("Qwen2.5-VL legacy (W2)", slow, images, "np", pdir / "Q25_W2_legacy.txt")
    fast_ms = run("Qwen2.5-VL fast (W2)",   fast, images, "pt", pdir / "Q25_W2_fast.txt")

    print(f"\n=== summary ===")
    print(f"  legacy / fast ratio: {slow_ms / fast_ms:.2f}x")


if __name__ == "__main__":
    main()