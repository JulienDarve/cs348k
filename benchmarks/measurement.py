import cProfile
import gc
import io
import platform
import pstats
import re
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


def measure_peak_rss(fn, sample_hz=4000):
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


def profile_fn(name, fn, out_path, top_n=20):
    """Run fn under cProfile and write profile stats to out_path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pr = cProfile.Profile()
    pr.enable()
    result = fn()
    pr.disable()
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(top_n)

    out_b = output_bytes(result)
    header = (
        f"=== {name} ===\n"
        f"output shape:    {output_shape(result)}\n"
        f"output bytes:    {out_b/1e6:.2f} MB\n\n"
    )
    out_path.write_text(header + buf.getvalue())


def measure_memory(name, call, n_memory_warmup):
    """Run optional warmup calls then measure peak RSS for a single call."""
    for _ in range(n_memory_warmup):
        call()

    gc.collect()
    result, peak_rss = measure_peak_rss(call)
    out_b = output_bytes(result)
    ratio = peak_rss / out_b if out_b > 0 else float("nan")

    print(f"\n--- {name} ---")
    print(f"  output shape:  {output_shape(result)}")
    print(f"  output:        {out_b/1e6:.2f} MB")
    print(f"  peak RSS:      {peak_rss/1e6:.2f} MB")
    print(f"  peak / output: {ratio:.2f}x")
    return ratio


def time_stages(stage_fns, n_warmup=10, n_timed=30):
    """Time a pipeline broken into named stages, called sequentially each pass.

    stage_fns: dict[str, Callable[[], Any]] — ordered; each callable runs one stage.
      Stages share state via closures and must be called in insertion order each pass.
    Returns: dict[str, float] — stage_name → median ms.
    """
    stages = list(stage_fns.items())
    for _ in range(n_warmup):
        for _, fn in stages:
            fn()
    gc.collect()
    gc.disable()
    try:
        ns_per_stage = {name: [] for name, _ in stages}
        for _ in range(n_timed):
            for name, fn in stages:
                t0 = time.perf_counter_ns()
                fn()
                ns_per_stage[name].append(time.perf_counter_ns() - t0)
    finally:
        gc.enable()
    return {
        name: float(np.median(np.array(ns, dtype=np.float64)) / 1e6)
        for name, ns in ns_per_stage.items()
    }


def parse_cprofile_stages(profile_path, stage_keywords):
    """Parse a cProfile file written by profile_fn and extract per-stage cumtime.

    profile_path: Path to file written by profile_fn().
    stage_keywords: dict[str, list[str]] — stage_name → substrings to match in
      the function name column (case-insensitive). First match in cumtime-sorted
      table wins.

    Returns: dict[str, float | None] with one entry per stage_name (None if no
      keyword matched) plus a special "_total_ms" key for the overall wall time
      parsed from the cProfile summary line.

    Raises FileNotFoundError if profile_path does not exist.
    """
    text = profile_path.read_text()

    # Extract overall wall time from "N function calls in T.TTT seconds"
    total_ms = None
    m = re.search(r"(\d+(?:\.\d+)?)\s+seconds", text)
    if m:
        total_ms = float(m.group(1)) * 1000.0

    # Find the column-header line and parse data rows below it
    rows = []  # list of (cumtime_s, function_name_fragment)
    in_table = False
    for line in text.splitlines():
        if re.match(r"\s*ncalls\s+tottime", line):
            in_table = True
            continue
        if not in_table:
            continue
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        # Minimum: ncalls tottime percall cumtime percall filename:lineno(function)
        if len(parts) < 6:
            continue
        try:
            cumtime_s = float(parts[3])
        except ValueError:
            continue
        # Function name is the last whitespace-separated token
        func_fragment = parts[-1]
        rows.append((cumtime_s, func_fragment))

    result = {"_total_ms": total_ms}
    for stage, keywords in stage_keywords.items():
        matched = None
        for cumtime_s, func_fragment in rows:
            if any(kw.lower() in func_fragment.lower() for kw in keywords):
                matched = cumtime_s * 1000.0
                break
        result[stage] = matched
    return result


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
