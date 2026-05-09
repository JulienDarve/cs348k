import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
torch.set_num_threads(1)

from data import load_images_w3
from measurement import time_fn, measure_peak_rss, output_bytes, output_shape
from models import get_llava_hf_processor

N_TIMED = 10


def run_quick_test():
    images = load_images_w3(n_images=N_TIMED)
    print(f"workload W3: {len(images)} images, mixed sizes")

    hf_slow = get_llava_hf_processor(use_fast=False)
    hf_fast = get_llava_hf_processor(use_fast=True)

    # Correctness check: both processors must produce the same output shape.
    result_slow = hf_slow(images=images, return_tensors="np")
    result_fast = hf_fast(images=images, return_tensors="pt")
    shape_slow = output_shape(result_slow)
    shape_fast = output_shape(result_fast)
    assert shape_slow == shape_fast, (
        f"shape mismatch: legacy={shape_slow}, fast={shape_fast}"
    )
    print(f"correctness OK — output shape: {shape_slow}")

    for name, proc, rt in [
        ("LLaVA-NeXT HF legacy (W3)", hf_slow, "np"),
        ("LLaVA-NeXT HF fast (W3)",   hf_fast, "pt"),
    ]:
        fn = lambda proc=proc, rt=rt: proc(images=images, return_tensors=rt)
        median_ms, spread = time_fn(fn, n_warmup=0, n_timed=N_TIMED, desc=name)
        result, peak_rss = measure_peak_rss(fn)
        out_b = output_bytes(result)
        ratio = peak_rss / out_b if out_b > 0 else float("nan")

        print(f"\n--- {name} ---")
        print(f"  median:        {median_ms:8.2f} ms/batch ({median_ms/len(images):.3f} ms/img)")
        print(f"  p95 - p50:     {spread:8.2f} ms")
        print(f"  output shape:  {output_shape(result)}")
        print(f"  output bytes:  {out_b/1e6:.2f} MB")
        print(f"  peak RSS:      {peak_rss/1e6:.2f} MB")
        print(f"  peak / output: {ratio:.2f}x")


if __name__ == "__main__":
    run_quick_test()
