"""Profiling file — HF fast cProfile breakdown + per-stage Numba timing.

Two profiling modes:
  1. cProfile: runs profile_fn() on HF variants (and optionally DSL variants).
     Writes one profile file per variant × workload to --profiles-dir.
     Reads the files back and extracts per-stage cumtime by keyword-matching
     the cProfile function-name column.
  2. Stage timing: instruments the Python orchestration of DSL Numba kernels
     with time.perf_counter_ns() between stages. Returns per-stage median ms.

After both modes run for a workload, a side-by-side comparison table is
printed merging cProfile-derived stage times (HF variants) with wall-clock
stage times (Numba/DSL variants).

Comparable display caveat (printed below each table):
  - cProfile cumtime is *inclusive* (sum of function + all callees), so stage
    times from cProfile overlap and their sum can exceed the total. Expected.
  - Stage timing is *exclusive* (each stage timed in isolation). Their sum
    approximates but slightly underestimates end-to-end wall time.

Qwen stage breakdown (v1 = staged, v3 = fused):
  v1: pil_decode | dims | resize | rescale | normalize | patchify | concat
  v3: pil_decode | dims | prealloc | kernel | grid

LLaVA stage breakdown (v1 and v3 same Python setup; different template):
  v1/v3: pil_decode | tile_select | prealloc | kernel

Usage:
  python benchmarks/bench_profile.py
  python benchmarks/bench_profile.py --num-threads 8 --workloads W2 W4
  python benchmarks/bench_profile.py --stage-only --workloads W2
  python benchmarks/bench_profile.py --cprofile-only --variants hf_fast dsl_v3
  python benchmarks/bench_profile.py --model llava --workloads W4
"""
import gc
import os
import sys
import argparse
from collections import OrderedDict
from pathlib import Path

import numba
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.data import load_images, load_images_w3, load_images_w4
from benchmarks.measurement import time_stages, parse_cprofile_stages, profile_fn, env_info
from benchmarks.models import load_processors, get_llava_hf_processor, MODEL_ID_QWEN
from kernels.qwen_v1_naive import (
    qwen_v1, smart_resize_dims, bilinear_resize, rescale, normalize, patchify,
    PATCH_SIZE, TEMPORAL_PATCH_SIZE, MERGE_SIZE, FACTOR,
    QWEN_MEAN, QWEN_STD, MIN_PIXELS, MAX_PIXELS,
)
from kernels.qwen_v3_storage import _v3_kernel, qwen_v3
from dsl.codegen import build, build_llava, select_best_resolution
from dsl.templates import _patch_output_size, make_template_llava_naive, make_template_llava_full
from pipelines.qwen import pipeline as qwen_pipeline, sched_v1, sched_v2, sched_v3
from pipelines.llava import (
    pipeline as llava_pipeline,
    sched_v1 as llava_sched_v1,
    sched_v2 as llava_sched_v2,
    sched_v3 as llava_sched_v3,
    GRID_PINPOINTS, TILE_SIZE, LLAVA_MEAN, LLAVA_STD, LLAVA_SCALE,
)
from transformers.image_utils import PILImageResampling

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_WARMUP_CPROFILE = 3
N_WARMUP_STAGE    = 3
N_TIMED_STAGE     = 5
N_TIMED_STAGE_W4  = 2   # W4 passes take ~1s each; 5 is enough for a stable median
DEFAULT_PROFILES_DIR = Path("profiles")

QWEN_CPROFILE_VARIANTS = ["hf_fast", "hf_bilinear", "dsl_v1", "dsl_v2", "dsl_v3"]
QWEN_STAGE_VARIANTS    = ["dsl_v1", "dsl_v3"]
LLAVA_CPROFILE_VARIANTS = ["hf_fast", "hf_bilinear", "dsl_v1", "dsl_v2", "dsl_v3"]
LLAVA_STAGE_VARIANTS    = ["dsl_v1", "dsl_v3"]

# Canonical row order for the comparison table
STAGE_ROW_ORDER = [
    "pil_decode", "tile_select", "dims",
    "resize", "rescale", "normalize", "patchify", "concat",
    "prealloc", "kernel", "grid",
    "other (cprof)",
]

# ---------------------------------------------------------------------------
# cProfile keyword map
# ---------------------------------------------------------------------------

HF_STAGE_KEYWORDS = {
    "resize": [
        "interpolate", "F_t.resize", "resize_image_torchvision",
        "_resize_image_pil", "bilinear", "bicubic", "antialias",
        "functional_tensor", "_upsample_bilinear2d", "_upsample_bicubic2d",
    ],
    "normalize": [
        "normalize", "rescale_and_normalize", "standardize", "rescale",
    ],
    "patchify": [
        "patchify", "get_image_patches", "_get_image_patches",
        "unfold", "reshape", "permute",
    ],
    "pil_decode": [
        "frombuffer", "tobytes", "_open_core", "ImageFile", "decode", "libjpeg",
    ],
}

# ---------------------------------------------------------------------------
# cProfile call wrappers
# ---------------------------------------------------------------------------

def _wrap_hf(proc, images, resample=None):
    if resample is not None:
        return proc(images=images, return_tensors="pt", resample=resample)
    return proc(images=images, return_tensors="pt")


def _wrap_qwen_dsl(fn, images, proc_fast):
    pv, grid = fn(images, min_pixels=proc_fast.min_pixels, max_pixels=proc_fast.max_pixels)
    return {"pixel_values": pv, "image_grid_thw": grid}


def _wrap_llava_dsl(fn, images):
    pv, n_tiles = fn(images)
    return {"pixel_values": pv}


# ---------------------------------------------------------------------------
# Qwen stage factory functions
# ---------------------------------------------------------------------------

def make_qwen_v1_stage_fns(images, proc_fast):
    """Per-stage timing for Qwen v1 (staged: separate buffer per op)."""
    min_px = proc_fast.min_pixels
    max_px = proc_fast.max_pixels
    state = {}

    def pil_decode():
        state["arrs"] = [np.asarray(img, dtype=np.uint8) for img in images]

    def dims():
        state["dims"] = [
            smart_resize_dims(a.shape[0], a.shape[1], factor=FACTOR,
                              min_pixels=min_px, max_pixels=max_px)
            for a in state["arrs"]
        ]

    def resize():
        state["resized"] = [
            bilinear_resize(a, oh, ow)
            for a, (oh, ow) in zip(state["arrs"], state["dims"])
        ]

    def rescale_stage():
        state["rescaled"] = [rescale(r) for r in state["resized"]]

    def normalize_stage():
        state["normed"] = [normalize(sc, QWEN_MEAN, QWEN_STD) for sc in state["rescaled"]]

    def patchify_stage():
        state["patched"] = [
            patchify(n, PATCH_SIZE, TEMPORAL_PATCH_SIZE, MERGE_SIZE)
            for n in state["normed"]
        ]

    def concat():
        state["pv"] = np.concatenate(state["patched"], axis=0)

    return OrderedDict([
        ("pil_decode", pil_decode),
        ("dims",       dims),
        ("resize",     resize),
        ("rescale",    rescale_stage),
        ("normalize",  normalize_stage),
        ("patchify",   patchify_stage),
        ("concat",     concat),
    ])


def make_qwen_v3_stage_fns(images, proc_fast):
    """Per-stage timing for Qwen v3 (full fusion: bilinear+rescale+norm+patchify in one kernel)."""
    min_px = proc_fast.min_pixels
    max_px = proc_fast.max_pixels
    n = len(images)
    patch_dim = TEMPORAL_PATCH_SIZE * 3 * PATCH_SIZE * PATCH_SIZE
    state = {}

    def pil_decode():
        state["arrs"] = [np.asarray(img, dtype=np.uint8) for img in images]

    def dims():
        out_h_list, out_w_list = [], []
        for arr in state["arrs"]:
            oh, ow = smart_resize_dims(arr.shape[0], arr.shape[1], factor=FACTOR,
                                       min_pixels=min_px, max_pixels=max_px)
            out_h_list.append(oh)
            out_w_list.append(ow)
        state["out_h_arr"] = np.array(out_h_list, dtype=np.int64)
        state["out_w_arr"] = np.array(out_w_list, dtype=np.int64)

    def prealloc():
        out_h_arr = state["out_h_arr"]
        out_w_arr = state["out_w_arr"]
        n_patches_per = (out_h_arr // PATCH_SIZE) * (out_w_arr // PATCH_SIZE)
        patch_offsets = np.zeros(n, dtype=np.int64)
        for i in range(1, n):
            patch_offsets[i] = patch_offsets[i - 1] + n_patches_per[i - 1]
        total_patches = int(patch_offsets[-1] + n_patches_per[-1])
        state["output"]        = np.empty((total_patches, patch_dim), dtype=np.float32)
        state["patch_offsets"] = patch_offsets
        state["typed_imgs"]    = numba.typed.List(state["arrs"])

    def kernel():
        _v3_kernel(
            state["typed_imgs"], state["out_h_arr"], state["out_w_arr"],
            state["patch_offsets"], QWEN_MEAN, QWEN_STD, state["output"],
            PATCH_SIZE, TEMPORAL_PATCH_SIZE, MERGE_SIZE,
        )

    def grid():
        out_h_arr = state["out_h_arr"]
        out_w_arr = state["out_w_arr"]
        state["grid"] = np.stack([
            np.array([1, int(out_h_arr[i]) // PATCH_SIZE, int(out_w_arr[i]) // PATCH_SIZE],
                     dtype=np.int64)
            for i in range(n)
        ], axis=0)

    return OrderedDict([
        ("pil_decode", pil_decode),
        ("dims",       dims),
        ("prealloc",   prealloc),
        ("kernel",     kernel),
        ("grid",       grid),
    ])


# ---------------------------------------------------------------------------
# LLaVA stage factory
# ---------------------------------------------------------------------------

def make_llava_stage_fns(images, fusion_level):
    """Per-stage timing for LLaVA DSL (all schedules share Python orchestration).

    fusion_level: "naive" → v1 template, "full" → v3 template.
    Stages: pil_decode | tile_select | prealloc | kernel
    """
    if fusion_level == "naive":
        template_fn = make_template_llava_naive(parallel_tiles=True)
    elif fusion_level == "full":
        template_fn = make_template_llava_full(parallel_tiles=True)
    else:
        raise ValueError(f"Unsupported fusion level: {fusion_level!r}")

    mean      = LLAVA_MEAN
    std       = LLAVA_STD
    scale     = np.float32(LLAVA_SCALE)
    tile_size = TILE_SIZE
    gp        = GRID_PINPOINTS
    state     = {}

    def pil_decode():
        state["arrs"] = [np.asarray(img, dtype=np.uint8) for img in images]

    def tile_select():
        state["tile_infos"] = [
            select_best_resolution(a.shape[0], a.shape[1], gp)
            for a in state["arrs"]
        ]

    def prealloc():
        arrs       = state["arrs"]
        tile_infos = state["tile_infos"]
        tile_descs_list, img_idx_list, n_tiles_list = [], [], []
        for b, (arr, (best_h, best_w)) in enumerate(zip(arrs, tile_infos)):
            orig_h, orig_w = arr.shape[:2]
            n_rows = best_h // tile_size
            n_cols = best_w // tile_size
            new_h, new_w = _patch_output_size(orig_h, orig_w, best_h, best_w)
            pad_top  = (best_h - new_h) // 2
            pad_left = (best_w - new_w) // 2
            tile_descs_list.append(
                [orig_h, orig_w, tile_size, tile_size, -1, -1, tile_size, tile_size, 0, 0])
            img_idx_list.append(b)
            for r in range(n_rows):
                for c in range(n_cols):
                    tile_descs_list.append(
                        [orig_h, orig_w, best_h, best_w, r, c, new_h, new_w, pad_top, pad_left])
                    img_idx_list.append(b)
            n_tiles_list.append(1 + n_rows * n_cols)
        n_total    = len(img_idx_list)
        state["tile_descs"]  = np.array(tile_descs_list, dtype=np.int64)
        state["img_idx"]     = np.array(img_idx_list, dtype=np.int64)
        state["output"]      = np.empty((n_total, 3, tile_size, tile_size), dtype=np.float32)
        state["typed_imgs"]  = numba.typed.List(arrs)

    def kernel():
        template_fn(
            state["typed_imgs"], state["tile_descs"], state["img_idx"],
            state["output"], mean, std, scale, tile_size,
        )

    stage_fns = OrderedDict([
        ("pil_decode", pil_decode),
        ("tile_select", tile_select),
        ("prealloc",   prealloc),
        ("kernel",     kernel),
    ])

    # JIT-warm the template (cache=True means reuse on-disk if same body was compiled before)
    print(f"  Warming up LLaVA {fusion_level} template for stage factory...")
    for fn in stage_fns.values():
        fn()

    return stage_fns


# ---------------------------------------------------------------------------
# cProfile run
# ---------------------------------------------------------------------------

def run_cprofile_workload(label, images, calls, profiles_dir, n_warmup, model, n_threads):
    """Run profile_fn for each (variant, callable) and return written paths.

    calls: list of (variant_name, callable).
    Returns: dict[str, Path]
    """
    n = len(images)
    print(f"\n{'='*60}")
    print(f"[CPROFILE] workload {label}: {n} images  (warmup={n_warmup})")
    print(f"{'='*60}")

    paths = {}
    for name, call in calls:
        print(f"\n  {name} ({label}) — warming up {n_warmup}x then profiling...")
        for _ in range(n_warmup):
            call()
        path = profiles_dir / f"{model}_{name}_{label}_{n_threads}t.txt"
        profile_fn(f"{model} {name} {label} {n_threads}t", call, path)
        print(f"  -> wrote {path}")
        paths[name] = path

    return paths


# ---------------------------------------------------------------------------
# Stage timing run
# ---------------------------------------------------------------------------

def run_stage_workload(label, stage_variant_fns, n_warmup, n_timed):
    """Run time_stages for each variant. Returns dict[str, dict[str, float]]."""
    print(f"\n{'='*60}")
    print(f"[STAGE] workload {label}  (warmup={n_warmup}, timed={n_timed})")
    print(f"{'='*60}")

    results = {}
    for variant_name, stage_fns in stage_variant_fns.items():
        print(f"\n  {variant_name} ({label})...")
        results[variant_name] = time_stages(stage_fns, n_warmup=n_warmup, n_timed=n_timed)
        total = sum(results[variant_name].values())
        print(f"  total (sum of stages): {total:.2f} ms")
        for sname, ms in results[variant_name].items():
            pct = ms / total * 100 if total > 0 else float("nan")
            print(f"    {sname:<14} {ms:8.3f} ms  ({pct:.1f}%)")

    return results


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def _fmt_ms(val):
    if val is None:
        return "    N/A  "
    return f"{val:8.1f} "


def _fmt_pct(val, total):
    if val is None or total is None or total <= 0:
        return "   N/A  "
    return f"{val / total * 100:6.1f}% "


def print_comparison_table(label, n_images, n_threads, cprofile_data, stage_results):
    """Print a side-by-side stage breakdown table.

    cprofile_data: dict[variant → dict[stage → float|None, "_total_ms" → float|None]]
    stage_results: dict[variant → dict[stage → float]]
    """
    # Determine all variant columns (cprofile first, then staged)
    cprofile_variants = list(cprofile_data.keys())
    stage_variants    = list(stage_results.keys())
    all_variants      = cprofile_variants + [v for v in stage_variants if v not in cprofile_variants]

    if not all_variants:
        return

    # Column headers
    col_w = 26
    header_row = f"{'stage':<16}|"
    for v in cprofile_variants:
        header_row += f" {(v + ' (cprof)'):<{col_w}}|"
    for v in stage_variants:
        label_v = v + " (staged)"
        header_row += f" {label_v:<{col_w}}|"

    sep = "=" * (17 + len(all_variants) * (col_w + 2))
    thin_sep = "-" * len(sep)

    print(f"\n{sep}")
    print(f"Stage breakdown: {label}  ({n_images} images, {n_threads} thread(s))")
    print(f"{sep}")

    sub_header = f"{'':16}|"
    for _ in cprofile_variants:
        sub_header += f" {'ms/batch':>8}  {'% total':>7}  {'':>6}|"
    for _ in stage_variants:
        sub_header += f" {'ms/batch':>8}  {'% total':>7}  {'':>6}|"

    print(header_row)
    print(sub_header)
    print(thin_sep)

    # Collect totals
    totals = {}
    for v in cprofile_variants:
        totals[v] = cprofile_data[v].get("_total_ms")
    for v in stage_variants:
        s = stage_results[v]
        totals[v] = sum(s.values()) if s else None

    # Determine which stages to show (skip rows where every column is N/A and not fused)
    def _get_val(variant, stage):
        if variant in cprofile_data:
            d = cprofile_data[variant]
            if stage == "other (cprof)":
                total = d.get("_total_ms")
                named_sum = sum(v for k, v in d.items() if k not in ("_total_ms", "other (cprof)") and v is not None)
                if total is None:
                    return None
                diff = total - named_sum
                return diff if diff >= 0 else None  # suppress if negative (overlapping cumtimes)
            return d.get(stage)
        if variant in stage_results:
            return stage_results[variant].get(stage)
        return None

    def _is_fused(variant, stage):
        """True if this stage is fused away in the variant's kernel."""
        if variant not in stage_results:
            return False
        known = set(stage_results[variant].keys())
        qwen_v3_fused  = {"resize", "rescale", "normalize", "patchify", "concat"}
        llava_fused    = {"resize", "rescale", "normalize", "concat"}
        v3_stages      = {"pil_decode", "dims", "prealloc", "kernel", "grid"}
        llava_stages   = {"pil_decode", "tile_select", "prealloc", "kernel"}
        if known == v3_stages and stage in qwen_v3_fused:
            return True
        if known == llava_stages and stage in llava_fused:
            return True
        return False

    for stage in STAGE_ROW_ORDER:
        vals = {v: _get_val(v, stage) for v in all_variants}
        fused = {v: _is_fused(v, stage) for v in all_variants}

        all_na = all(vals[v] is None and not fused[v] for v in all_variants)
        if all_na:
            continue

        row = f"{stage:<16}|"
        for v in cprofile_variants:
            val = vals[v]
            row += f" {_fmt_ms(val)}{_fmt_pct(val, totals.get(v))}       |"
        for v in stage_variants:
            val = vals[v]
            if fused[v]:
                row += f" {'(fused)':>8}  {'(fused)':>7}       |"
            else:
                row += f" {_fmt_ms(val)}{_fmt_pct(val, totals.get(v))}       |"
        print(row)

    print(thin_sep)

    # Total row
    total_row = f"{'TOTAL':<16}|"
    for v in cprofile_variants:
        t = totals.get(v)
        total_row += f" {_fmt_ms(t)}{'100.0%':>7}  {'†' if t else '':>6}|"
    for v in stage_variants:
        t = totals.get(v)
        total_row += f" {_fmt_ms(t)}{'~100%':>7}  {'‡' if t else '':>6}|"
    print(total_row)
    print(sep)

    print("  † cprofile total = wall time from 'N calls in T seconds' summary line.")
    print("  ‡ stage total = sum of stage medians (not end-to-end; Python glue not counted).")
    print("  cProfile cumtime is *inclusive* (nested calls overlap) — stage %s may sum > 100%.")
    print("  Stage timing is *exclusive* (each stage timed in isolation).")

    unmatched = []
    for v in cprofile_variants:
        d = cprofile_data.get(v, {})
        for stage, val in d.items():
            if stage.startswith("_"):
                continue
            if val is None and stage != "other (cprof)":
                unmatched.append(f"{v}:{stage}")
    if unmatched:
        print(f"  Unmatched cprofile stages (keyword not found): {', '.join(unmatched)}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="VLM preprocessing profiler: cProfile breakdown + per-stage Numba timing.")
    ap.add_argument("--model",    choices=["qwen", "llava"], default="qwen")
    ap.add_argument("--num-threads", type=int, default=1, metavar="N")
    ap.add_argument("--workloads", nargs="+", choices=["W2", "W3", "W4"], default=["W2"])
    ap.add_argument("--variants", nargs="+",
                    default=None, metavar="V",
                    help="Variants for cProfile mode. Default: all for model.")
    ap.add_argument("--stage-variants", nargs="+",
                    choices=["dsl_v1", "dsl_v3"], default=None,
                    help="Variants for stage timing mode. Default: dsl_v1 dsl_v3.")
    ap.add_argument("--profiles-dir", type=Path, default=DEFAULT_PROFILES_DIR)
    ap.add_argument("--n-warmup",       type=int, default=N_WARMUP_CPROFILE,
                    help="Warmup calls before each cProfile run (default: 3).")
    ap.add_argument("--n-warmup-stage",    type=int, default=N_WARMUP_STAGE)
    ap.add_argument("--n-timed-stage",     type=int, default=N_TIMED_STAGE)
    ap.add_argument("--n-timed-stage-w4",  type=int, default=N_TIMED_STAGE_W4,
                    help="Timed stage iterations for W4 (default: 5).")
    ap.add_argument("--cprofile-only",  action="store_true", help="Skip stage timing.")
    ap.add_argument("--stage-only",     action="store_true", help="Skip cProfile.")
    args = ap.parse_args()

    if args.cprofile_only and args.stage_only:
        raise SystemExit("--cprofile-only and --stage-only are mutually exclusive")

    n_threads = args.num_threads
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    torch.set_num_threads(n_threads)
    numba.set_num_threads(n_threads)

    print(env_info())

    is_qwen  = args.model == "qwen"
    is_llava = args.model == "llava"

    default_cprofile_variants = QWEN_CPROFILE_VARIANTS if is_qwen else LLAVA_CPROFILE_VARIANTS
    default_stage_variants    = QWEN_STAGE_VARIANTS    if is_qwen else LLAVA_STAGE_VARIANTS
    cprofile_variants = args.variants      or default_cprofile_variants
    stage_variants    = args.stage_variants or default_stage_variants

    # ----- Load processors -----
    if is_qwen:
        q_legacy, q_fast = load_processors(MODEL_ID_QWEN)
        proc_fast = q_fast
        print(f"\nLoaded: {MODEL_ID_QWEN} (legacy, fast)")
    else:
        hf_legacy = get_llava_hf_processor(use_fast=False)
        hf_fast   = get_llava_hf_processor(use_fast=True)
        proc_fast = None
        print("\nLoaded: LLaVA-NeXT (legacy, fast)")

    # ----- Load workloads -----
    selected_wl = set(args.workloads)
    images_w2 = load_images(n_images=32,  img_size=(1024, 1024)) if "W2" in selected_wl else None
    images_w3 = load_images_w3()                                  if "W3" in selected_wl else None
    images_w4 = load_images_w4()                                  if "W4" in selected_wl else None

    if images_w2: print(f"W2: {len(images_w2)} images @ {images_w2[0].size}")
    if images_w3: print(f"W3: {len(images_w3)} images, e.g. {[img.size for img in images_w3[:3]]}")
    if images_w4: print(f"W4: {len(images_w4)} images @ {images_w4[0].size}")

    first_images = images_w2 or images_w3 or images_w4

    # ----- Numba JIT warmup -----
    dsl_fns = {}
    needs_dsl = {v for v in cprofile_variants + stage_variants if v.startswith("dsl_")}

    if is_qwen:
        if needs_dsl or "dsl_v1" in stage_variants or "dsl_v3" in stage_variants:
            print("\nWarming up Numba JIT for Qwen kernels and DSL variants...")
            qwen_v1(first_images[:1], min_pixels=proc_fast.min_pixels, max_pixels=proc_fast.max_pixels)
            qwen_v3(first_images[:1], min_pixels=proc_fast.min_pixels, max_pixels=proc_fast.max_pixels)
            sched_map = {"dsl_v1": sched_v1, "dsl_v2": sched_v2, "dsl_v3": sched_v3}
            for v in sorted(needs_dsl):
                if v in sched_map:
                    fn = build(qwen_pipeline, sched_map[v])
                    fn(first_images[:1], min_pixels=proc_fast.min_pixels, max_pixels=proc_fast.max_pixels)
                    dsl_fns[v] = fn
            print("Qwen JIT ready.")
    else:
        if needs_dsl:
            print("\nWarming up Numba JIT for LLaVA DSL variants...")
            sched_map = {"dsl_v1": llava_sched_v1, "dsl_v2": llava_sched_v2, "dsl_v3": llava_sched_v3}
            for v in sorted(needs_dsl):
                if v in sched_map:
                    fn = build_llava(llava_pipeline, sched_map[v])
                    fn(first_images[:1])
                    dsl_fns[v] = fn
            print("LLaVA JIT ready.")

    # ----- Per-workload loop -----
    workloads_iter = [
        ("W2", images_w2), ("W3", images_w3), ("W4", images_w4),
    ]

    for label, images in workloads_iter:
        if images is None:
            continue

        n_images = len(images)
        print(f"\n{'#'*60}")
        print(f"# Workload {label}: {n_images} images")
        print(f"{'#'*60}")

        # Build cProfile callables
        if is_qwen:
            all_calls_map = {
                "hf_fast":    lambda imgs=images: _wrap_hf(q_fast,   imgs),
                "hf_bilinear":lambda imgs=images: _wrap_hf(q_fast,   imgs, resample=PILImageResampling.BILINEAR),
                "hf_legacy":  lambda imgs=images: _wrap_hf(q_legacy, imgs),
                "dsl_v1":     lambda imgs=images: _wrap_qwen_dsl(dsl_fns["dsl_v1"], imgs, proc_fast),
                "dsl_v2":     lambda imgs=images: _wrap_qwen_dsl(dsl_fns["dsl_v2"], imgs, proc_fast),
                "dsl_v3":     lambda imgs=images: _wrap_qwen_dsl(dsl_fns["dsl_v3"], imgs, proc_fast),
            }
        else:
            all_calls_map = {
                "hf_fast":    lambda imgs=images: _wrap_hf(hf_fast,   imgs),
                "hf_bilinear":lambda imgs=images: _wrap_hf(hf_fast,   imgs, resample=PILImageResampling.BILINEAR),
                "hf_legacy":  lambda imgs=images: _wrap_hf(hf_legacy, imgs),
                "dsl_v1":     lambda imgs=images: _wrap_llava_dsl(dsl_fns["dsl_v1"], imgs),
                "dsl_v2":     lambda imgs=images: _wrap_llava_dsl(dsl_fns["dsl_v2"], imgs),
                "dsl_v3":     lambda imgs=images: _wrap_llava_dsl(dsl_fns["dsl_v3"], imgs),
            }
        calls = [(v, all_calls_map[v]) for v in cprofile_variants if v in all_calls_map]

        # Build stage factory fns
        stage_variant_fns = {}
        if not args.cprofile_only:
            if is_qwen:
                for v in stage_variants:
                    if v == "dsl_v1":
                        stage_variant_fns[v] = make_qwen_v1_stage_fns(images, proc_fast)
                    elif v == "dsl_v3":
                        stage_variant_fns[v] = make_qwen_v3_stage_fns(images, proc_fast)
            else:
                for v in stage_variants:
                    if v == "dsl_v1":
                        stage_variant_fns[v] = make_llava_stage_fns(images, "naive")
                    elif v == "dsl_v3":
                        stage_variant_fns[v] = make_llava_stage_fns(images, "full")

        # Phase 1: cProfile
        cprofile_paths = {}
        if not args.stage_only and calls:
            cprofile_paths = run_cprofile_workload(
                label, images, calls, args.profiles_dir,
                args.n_warmup, args.model, n_threads,
            )

        # Phase 2: Stage timing
        stage_results = {}
        if not args.cprofile_only and stage_variant_fns:
            n_timed = args.n_timed_stage_w4 if label == "W4" else args.n_timed_stage
            stage_results = run_stage_workload(
                label, stage_variant_fns,
                n_warmup=args.n_warmup_stage, n_timed=n_timed,
            )

        # Parse cProfile files and print table
        cprofile_data = {}
        for vname, path in cprofile_paths.items():
            try:
                cprofile_data[vname] = parse_cprofile_stages(path, HF_STAGE_KEYWORDS)
            except FileNotFoundError:
                print(f"WARNING: profile file not found: {path}")

        print_comparison_table(label, n_images, n_threads, cprofile_data, stage_results)


if __name__ == "__main__":
    main()
