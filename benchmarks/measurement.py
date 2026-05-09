import cProfile
import gc
import io
import platform
import pstats
import resource
import threading
import time

import numpy as np
import torch
from tqdm import tqdm

_PAGE = resource.getpagesize()  # type: ignore[attr-defined]


def _current_rss_bytes():
    with open("/proc/self/statm", "rb") as f:
        return int(f.read().split()[1]) * _PAGE


def measure_peak_rss(fn, sample_hz=2000):
    """Run fn while sampling RSS in a background thread.

    Returns (result, peak_delta_bytes). peak_delta is peak RSS observed
    during fn() minus RSS at entry, so it's the additional working set
    fn() caused (modulo allocator pool retention from prior calls).
    """
    interval = 1.0 / sample_hz
    stop = threading.Event()
    baseline = _current_rss_bytes()
    peak = baseline

    def sampler():
        nonlocal peak
        while not stop.is_set():
            r = _current_rss_bytes()
            if r > peak:
                peak = r
            time.sleep(interval)

    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    try:
        result = fn()
    finally:
        r = _current_rss_bytes()
        if r > peak:
            peak = r
        stop.set()
        t.join()
    return result, peak - baseline


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
    """Run fn under cProfile, then measure peak RSS; write results to out_path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pr = cProfile.Profile()
    pr.enable()
    fn()
    pr.disable()
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(top_n)
    profile_text = buf.getvalue()

    gc.collect()
    result, peak_rss = measure_peak_rss(fn)

    out_b = output_bytes(result)
    ratio = peak_rss / out_b if out_b > 0 else float("nan")

    header = (
        f"=== {name} ===\n"
        f"output shape:    {output_shape(result)}\n"
        f"output bytes:    {out_b/1e6:.2f} MB\n"
        f"peak RSS delta:  {peak_rss/1e6:.2f} MB\n"
        f"peak / output:   {ratio:.2f}x\n\n"
    )
    out_path.write_text(header + profile_text)
    return peak_rss, out_b, ratio


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
