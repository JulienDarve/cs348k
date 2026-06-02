import cProfile
import gc
import io
import os
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
    """Run fn under cProfile and write profile stats to out_path.

    Also writes a sibling .pstats file so stage extraction can use the full
    call graph instead of only the printed top-N rows.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path = out_path.with_suffix(out_path.suffix + ".pstats")

    pr = cProfile.Profile()
    pr.enable()
    result = fn()
    pr.disable()
    pr.dump_stats(stats_path)

    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(top_n)

    out_b = output_bytes(result)
    header = (
        f"=== {name} ===\n"
        f"output shape:    {output_shape(result)}\n"
        f"output bytes:    {out_b/1e6:.2f} MB\n\n"
        f"pstats file:     {stats_path}\n"
        f"printed rows:    top {top_n} by cumulative time\n\n"
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

    stage_fns: ordered mapping of stage name to callable. Stages share state
    via closures and must be called in insertion order each pass.
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


def time_stage_pipeline(stage_fns, n_warmup=10, n_timed=30):
    """Time the whole ordered stage pipeline end-to-end.

    This sanity-checks time_stages(): the sum of stage medians should be close
    to, but not exactly the same as, this full-call median.
    """
    stages = list(stage_fns.items())
    for _ in range(n_warmup):
        for _, fn in stages:
            fn()
    gc.collect()
    gc.disable()
    try:
        ts = []
        for _ in range(n_timed):
            t0 = time.perf_counter_ns()
            for _, fn in stages:
                fn()
            ts.append(time.perf_counter_ns() - t0)
    finally:
        gc.enable()
    return float(np.median(np.array(ts, dtype=np.float64)) / 1e6)


def _parse_total_ms_from_text(profile_path):
    text = profile_path.read_text()
    m = re.search(r"(\d+(?:\.\d+)?)\s+seconds", text)
    return float(m.group(1)) * 1000.0 if m else None


def _load_pstats_rows(profile_path):
    stats_path = profile_path.with_suffix(profile_path.suffix + ".pstats")
    if not stats_path.exists():
        return None, None

    stats_obj = pstats.Stats(str(stats_path))
    rows = []
    for (filename, lineno, func_name), (cc, nc, tt, ct, _callers) in stats_obj.stats.items():
        fragment = f"{filename}:{lineno}({func_name})"
        rows.append({
            "func": fragment,
            "search": fragment.lower(),
            "ncalls": nc,
            "primitive_calls": cc,
            "tottime_ms": float(tt) * 1000.0,
            "cumtime_ms": float(ct) * 1000.0,
        })
    rows.sort(key=lambda r: r["cumtime_ms"], reverse=True)
    return float(stats_obj.total_tt) * 1000.0, rows


def _load_text_rows(profile_path):
    """Fallback parser for older profile files without a sibling .pstats file."""
    text = profile_path.read_text()
    rows = []
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
        if len(parts) < 6:
            continue
        try:
            tottime_s = float(parts[1])
            cumtime_s = float(parts[3])
        except ValueError:
            continue
        func_fragment = parts[-1]
        rows.append({
            "func": func_fragment,
            "search": func_fragment.lower(),
            "ncalls": parts[0],
            "primitive_calls": None,
            "tottime_ms": tottime_s * 1000.0,
            "cumtime_ms": cumtime_s * 1000.0,
        })
    return rows


def parse_cprofile_stages(profile_path, stage_keywords, metric="cumtime"):
    """Parse a cProfile file written by profile_fn and extract per-stage time.

    metric is "cumtime" or "tottime". The selected metric is copied to the
    top-level stage keys; "_matches" keeps both self and cumulative time.
    """
    if metric not in ("cumtime", "tottime"):
        raise ValueError("metric must be 'cumtime' or 'tottime'")

    total_ms, rows = _load_pstats_rows(profile_path)
    if rows is None:
        rows = _load_text_rows(profile_path)
        total_ms = _parse_total_ms_from_text(profile_path)

    metric_key = f"{metric}_ms"
    result = {"_total_ms": total_ms, "_metric": metric, "_matches": {}}
    for stage, keywords in stage_keywords.items():
        matched = None
        lowered = [kw.lower() for kw in keywords]
        for row in rows:
            if any(kw in row["search"] for kw in lowered):
                matched = row
                break
        if matched is None:
            result[stage] = None
            result["_matches"][stage] = None
        else:
            result[stage] = matched[metric_key]
            result["_matches"][stage] = {
                "func": matched["func"],
                "ncalls": matched["ncalls"],
                "tottime_ms": matched["tottime_ms"],
                "cumtime_ms": matched["cumtime_ms"],
            }
    return result


def env_info():
    import PIL
    import torchvision
    import transformers
    try:
        import numba
        numba_threads = numba.get_num_threads()
    except Exception:
        numba_threads = "?"
    return (
        f"python={platform.python_version()} platform={platform.platform()}\n"
        f"cpu={platform.processor() or platform.machine()}\n"
        f"transformers={transformers.__version__} torch={torch.__version__} "
        f"torchvision={torchvision.__version__} pillow={PIL.__version__} "
        f"numpy={np.__version__}\n"
        f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} "
        f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')} "
        f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')} "
        f"NUMBA_NUM_THREADS={os.environ.get('NUMBA_NUM_THREADS')} "
        f"torch_threads={torch.get_num_threads()} "
        f"numba_threads={numba_threads}"
    )
