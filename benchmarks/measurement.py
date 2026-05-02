import cProfile
import gc
import io
import platform
import pstats
import resource
import time
import tracemalloc

import numpy as np
import torch
from tqdm import tqdm

def time_fn(fn, n_warmup=10, n_timed=100, desc=None):
    prefix = f"{desc} " if desc else ""
    for _ in tqdm(range(n_warmup), desc=f"{prefix}warmup", leave=False):
        fn()
    gc.collect()
    gc.disable()
    try:
        ts = []
        for _ in tqdm(range(n_timed), desc=f"{prefix}timing"):
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
    """Run fn under cProfile, then under tracemalloc; write results to out_path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pr = cProfile.Profile()
    pr.enable()
    fn()
    pr.disable()
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(top_n)
    profile_text = buf.getvalue()

    gc.collect()
    start_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    tracemalloc.start()
    try:
        result = fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    end_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_bytes = (end_rss - start_rss) * 1024
    actual_peak = max(peak_rss_bytes, peak)

    out_b = output_bytes(result)
    ratio = actual_peak / out_b if out_b > 0 else float("nan")

    header = (
        f"=== {name} ===\n"
        f"output shape:    {output_shape(result)}\n"
        f"output bytes:    {out_b/1e6:.2f} MB\n"
        f"peak alloc (OS): {actual_peak/1e6:.2f} MB\n"
        f"peak / output:   {ratio:.2f}x\n\n"
    )
    out_path.write_text(header + profile_text)
    return actual_peak, out_b, ratio


def env_info():
    import os
    import PIL
    import torchvision
    import transformers
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
