"""Phase A codegen: classify the schedule's fusion level, bind templates.

`build(pipeline, schedule) → callable` returns a preprocessing function with
signature `(images, **kwargs) → (pixel_values, image_grid_thw)`. The callable
internally:

  1. Validates the Pipeline against the op registry.
  2. Looks up the coord function (from patchify's "coord" param) and the
     dims function (from resize's "dims_fn" param).
  3. Classifies the schedule's fusion level (naive / pointwise / full) via
     classify_fusion.
  4. Instantiates the matching template from dsl/templates.py.
  5. Wraps it in a (PIL list → ndarray, ndarray) callable.

classify_fusion is the function that "makes the ablation come from the
schedule and not from a hardcoded choice" (§4 of the implementation plan).
Keep it small and legible — it's effectively the proof.
"""

from typing import Callable

import numpy as np
import numba

from dsl import ops as ops_mod
from dsl.algorithm import Pipeline
from dsl.schedule import Schedule
from dsl.coords import get as get_coord
from dsl.templates import (
    make_template_naive,
    make_template_pointwise,
    make_template_full,
    make_template_llava_naive,
    make_template_llava_pointwise,
    make_template_llava_full,
    _patch_output_size,
)
from kernels.qwen_v1_naive import smart_resize_dims


# ---------------------------------------------------------------------------
# Dimension functions: (h, w, **params) → (out_h, out_w)
# ---------------------------------------------------------------------------

def _qwen_smart_resize_adapter(h, w, *, factor, min_pixels, max_pixels):
    return smart_resize_dims(h, w, factor=factor,
                             min_pixels=min_pixels, max_pixels=max_pixels)


DIMS_FNS: dict[str, Callable] = {
    "qwen_smart_resize": _qwen_smart_resize_adapter,
}


# ---------------------------------------------------------------------------
# classify_fusion: schedule → "naive" | "pointwise" | "full"
# ---------------------------------------------------------------------------

def classify_fusion(pipeline: Pipeline, schedule: Schedule) -> str:
    """Pick the fusion level implied by the schedule.

    Rules (in priority order):
      "full"      ↔ schedule.preallocate_output AND the output Func's stage
                    has write_via set (output buffer is allocated once and
                    written via inlined address computation).
      "pointwise" ↔ every pointwise stage (rescale, normalize) has
                    compute_at="inline" (their results live in registers).
      "naive"     ↔ otherwise.

    These three rules cleanly map v1 / v2 / v3 of the hand-fused Qwen kernel.
    """
    output_stage = schedule.of(pipeline.output)
    if schedule.preallocate_output and output_stage.write_via is not None:
        return "full"

    pointwise_ops = {"rescale", "normalize"}
    pointwise_funcs = [f for f in pipeline.funcs if f.op in pointwise_ops]
    if pointwise_funcs and all(
        schedule.of(f.name).compute_at == "inline" for f in pointwise_funcs
    ):
        return "pointwise"

    return "naive"


# ---------------------------------------------------------------------------
# Helpers: pull op-specific algorithmic params out of the Pipeline
# ---------------------------------------------------------------------------

def _gather_pointwise(pipeline: Pipeline):
    """Return (mean, std, scale) as float32 arrays / float."""
    rescale_fs = pipeline.by_op("rescale")
    normalize_fs = pipeline.by_op("normalize")
    if not rescale_fs:
        raise ValueError("pipeline missing required 'rescale' op")
    if not normalize_fs:
        raise ValueError("pipeline missing required 'normalize' op")
    scale = float(rescale_fs[0].params["scale"])
    mean = np.asarray(normalize_fs[0].params["mean"], dtype=np.float32)
    std = np.asarray(normalize_fs[0].params["std"], dtype=np.float32)
    return mean, std, scale


def _gather_patchify(pipeline: Pipeline):
    """Return (patch_size, temporal_patch_size, merge_size, coord_name)."""
    patchify_fs = pipeline.by_op("patchify")
    if not patchify_fs:
        raise ValueError("pipeline missing required 'patchify' op")
    p = patchify_fs[0].params
    return (int(p["patch_size"]),
            int(p["temporal_patch_size"]),
            int(p["merge_size"]),
            p["coord"])


def _resize_params(pipeline: Pipeline):
    """Return the dict of params attached to the (single) resize Func."""
    resize_fs = pipeline.by_op("resize")
    if not resize_fs:
        raise ValueError("pipeline missing required 'resize' op")
    return resize_fs[0].params


# ---------------------------------------------------------------------------
# build: Pipeline + Schedule → preprocessing callable
# ---------------------------------------------------------------------------

def build(pipeline: Pipeline, schedule: Schedule) -> Callable:
    """Compile a Pipeline+Schedule to a callable preprocessor.

    Returned callable signature: `(images, *, min_pixels=None, max_pixels=None)
    → (pixel_values: np.ndarray, image_grid_thw: np.ndarray)`.

    `min_pixels`/`max_pixels` override the resize Func's params if given.
    """
    ops_mod.validate(pipeline)

    fusion = classify_fusion(pipeline, schedule)
    patch_size, temporal_patch_size, merge_size, coord_name = _gather_patchify(pipeline)
    mean, std, scale = _gather_pointwise(pipeline)
    coord = get_coord(coord_name)

    resize_p = _resize_params(pipeline)
    dims_fn_name = resize_p["dims_fn"]
    if dims_fn_name not in DIMS_FNS:
        raise KeyError(f"unknown dims_fn {dims_fn_name!r}; registered: {sorted(DIMS_FNS)}")
    dims_fn = DIMS_FNS[dims_fn_name]
    factor = patch_size * merge_size
    default_min_pixels = int(resize_p.get("min_pixels", 0))
    default_max_pixels = int(resize_p["max_pixels"])

    parallel_batch = schedule.parallel_axis() == "batch"

    if fusion == "naive":
        template = make_template_naive(coord)
    elif fusion == "pointwise":
        template = make_template_pointwise(coord)
    elif fusion == "full":
        template = make_template_full(coord, parallel_batch=parallel_batch)
    else:
        raise ValueError(f"unhandled fusion level {fusion!r}")

    patch_dim = temporal_patch_size * 3 * patch_size * patch_size

    def preprocess(images, *, min_pixels=None, max_pixels=None):
        mn = default_min_pixels if min_pixels is None else int(min_pixels)
        mx = default_max_pixels if max_pixels is None else int(max_pixels)

        n = len(images)
        if n == 0:
            return (np.empty((0, patch_dim), dtype=np.float32),
                    np.empty((0, 3), dtype=np.int64))

        arrs = [np.asarray(img, dtype=np.uint8) for img in images]
        out_h_list = []
        out_w_list = []
        for arr in arrs:
            oh, ow = dims_fn(arr.shape[0], arr.shape[1],
                             factor=factor, min_pixels=mn, max_pixels=mx)
            out_h_list.append(oh)
            out_w_list.append(ow)
        out_h_arr = np.asarray(out_h_list, dtype=np.int64)
        out_w_arr = np.asarray(out_w_list, dtype=np.int64)

        if fusion == "full":
            n_patches_per = (out_h_arr // patch_size) * (out_w_arr // patch_size)
            patch_offsets = np.zeros(n, dtype=np.int64)
            for i in range(1, n):
                patch_offsets[i] = patch_offsets[i - 1] + n_patches_per[i - 1]
            total_patches = int(patch_offsets[-1] + n_patches_per[-1])

            output = np.empty((total_patches, patch_dim), dtype=np.float32)
            typed_imgs = numba.typed.List(arrs)
            template(typed_imgs, out_h_arr, out_w_arr, patch_offsets,
                     mean, std, np.float32(scale), output,
                     patch_size, temporal_patch_size, merge_size)
            pv = output
        else:
            pv = template(arrs, out_h_arr, out_w_arr, mean, std, scale,
                          patch_size, temporal_patch_size, merge_size)

        grid_thw = np.stack([
            np.array([1, int(out_h_arr[i]) // patch_size,
                      int(out_w_arr[i]) // patch_size], dtype=np.int64)
            for i in range(n)
        ], axis=0)
        return pv, grid_thw

    preprocess.fusion = fusion
    return preprocess


# ---------------------------------------------------------------------------
# LLaVA-NeXT codegen
# ---------------------------------------------------------------------------

def select_best_resolution(orig_h: int, orig_w: int, grid_pinpoints: list) -> tuple:
    """Pick the best (best_h, best_w) from grid_pinpoints for this image.

    grid_pinpoints: list of [height, width] pairs (HF convention — pairs are
    iterated as `for height, width in possible_resolutions` in HF source).

    Selection matches HF LlavaNextImageProcessor.select_best_resolution:
      primary criterion: maximize effective area (min of downscaled area and
      original area); secondary: minimize wasted area. Returns (best_h, best_w).
    """
    best = None
    max_effective = 0
    min_wasted = float("inf")
    for ph, pw in grid_pinpoints:          # ph = height, pw = width
        scale = min(pw / orig_w, ph / orig_h)
        ds_w = int(orig_w * scale)
        ds_h = int(orig_h * scale)
        effective = min(ds_w * ds_h, orig_w * orig_h)
        wasted = ph * pw - effective
        if effective > max_effective or (effective == max_effective and wasted < min_wasted):
            max_effective = effective
            min_wasted = wasted
            best = (int(ph), int(pw))
    return best  # (best_h, best_w)


def classify_fusion_llava(pipeline: Pipeline, schedule: Schedule) -> str:
    """Pick the fusion level for a LLaVA pipeline from its schedule.

    Identical logic to classify_fusion but applied to the LLaVA DAG (output
    stage is 'normalize', which uses write_via='llava_chw' for full fusion).

    "full"      ↔ schedule.preallocate_output AND output stage has write_via set.
    "pointwise" ↔ all rescale/normalize stages have compute_at="inline".
    "naive"     ↔ otherwise.
    """
    output_stage = schedule.of(pipeline.output)
    if schedule.preallocate_output and output_stage.write_via is not None:
        return "full"

    pointwise_ops = {"rescale", "normalize"}
    pointwise_funcs = [f for f in pipeline.funcs if f.op in pointwise_ops]
    if pointwise_funcs and all(
        schedule.of(f.name).compute_at == "inline" for f in pointwise_funcs
    ):
        return "pointwise"

    return "naive"


def _gather_tile_params(pipeline: Pipeline):
    """Return (tile_size, grid_pinpoints) from the pipeline's tile Func."""
    tile_fs = pipeline.by_op("tile")
    if not tile_fs:
        raise ValueError("LLaVA pipeline missing required 'tile' op")
    p = tile_fs[0].params
    return int(p["tile_size"]), p["image_grid_pinpoints"]


def build_llava(pipeline: Pipeline, schedule: Schedule) -> Callable:
    """Compile a LLaVA Pipeline+Schedule to a callable preprocessor.

    Returned callable signature:
      (images: list[PIL.Image]) → (pixel_values: np.ndarray,
                                   n_tiles_per_image: np.ndarray)

    pixel_values shape: (N_total_tiles, 3, tile_size, tile_size) float32, CHW.
    n_tiles_per_image shape: (B,) int64. Sum equals N_total_tiles.

    Tile ordering: thumbnail first (index 0 per image), then grid tiles in
    row-major order — matches HF LlavaNextImageProcessor.
    """
    ops_mod.validate(pipeline)

    fusion = classify_fusion_llava(pipeline, schedule)
    mean, std, scale = _gather_pointwise(pipeline)
    tile_size, grid_pinpoints = _gather_tile_params(pipeline)

    parallel = schedule.parallel_axis() == "batch"

    if fusion == "naive":
        template = make_template_llava_naive()
    elif fusion == "pointwise":
        template = make_template_llava_pointwise()
    elif fusion == "full":
        template = make_template_llava_full(parallel_tiles=parallel)
    else:
        raise ValueError(f"unhandled fusion level {fusion!r}")

    def preprocess(images):
        n = len(images)
        if n == 0:
            return (np.empty((0, 3, tile_size, tile_size), dtype=np.float32),
                    np.empty((0,), dtype=np.int64))

        arrs = [np.asarray(img, dtype=np.uint8) for img in images]

        # Select best resolution per image (pure Python, cheap)
        tile_infos = []
        for arr in arrs:
            orig_h, orig_w = arr.shape[:2]
            tile_infos.append(select_best_resolution(orig_h, orig_w, grid_pinpoints))

        if fusion == "full":
            # Build flat tile descriptor arrays for the v3 kernel.
            # Each row: [orig_h, orig_w, best_h, best_w, t_row, t_col,
            #            new_h, new_w, pad_top, pad_left]  (10 columns)
            tile_descs_list = []
            img_idx_list = []
            n_tiles_per_image = []
            for b, (arr, (best_h, best_w)) in enumerate(zip(arrs, tile_infos)):
                orig_h, orig_w = arr.shape[:2]
                n_rows = best_h // tile_size
                n_cols = best_w // tile_size
                # Aspect-preserving dims + padding for grid tiles
                new_h, new_w = _patch_output_size(orig_h, orig_w, best_h, best_w)
                pad_top  = (best_h - new_h) // 2
                pad_left = (best_w - new_w) // 2
                # Thumbnail: direct resize → new_h=tile_size, pad_top=0
                tile_descs_list.append(
                    [orig_h, orig_w, tile_size, tile_size, -1, -1,
                     tile_size, tile_size, 0, 0])
                img_idx_list.append(b)
                # Grid tiles in row-major order
                for r in range(n_rows):
                    for c in range(n_cols):
                        tile_descs_list.append(
                            [orig_h, orig_w, best_h, best_w, r, c,
                             new_h, new_w, pad_top, pad_left])
                        img_idx_list.append(b)
                n_tiles_per_image.append(1 + n_rows * n_cols)

            tile_descs = np.array(tile_descs_list, dtype=np.int64)
            img_idx    = np.array(img_idx_list,    dtype=np.int64)
            n_total    = len(img_idx_list)
            output     = np.empty((n_total, 3, tile_size, tile_size), dtype=np.float32)

            typed_imgs = numba.typed.List(arrs)
            template(typed_imgs, tile_descs, img_idx, output,
                     mean, std, np.float32(scale), tile_size)
            pv = output
            n_tiles_arr = np.array(n_tiles_per_image, dtype=np.int64)
        else:
            # naive / pointwise: template returns (pv, n_tiles_per_image)
            pv, n_tiles_arr = template(arrs, tile_infos, mean, std, scale, tile_size)

        return pv, n_tiles_arr

    preprocess.fusion = fusion
    return preprocess
