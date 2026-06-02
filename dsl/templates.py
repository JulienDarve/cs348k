"""Three Numba template factories, one per fusion level.

Each factory closes over the coord function registry entry and returns a
Python-level orchestration callable. The orchestration matches the buffer
structure of the corresponding hand-fused Qwen kernel so memory benchmarks
remain comparable:

  template_naive       reproduces qwen_v1: each stage its own loop + buffer.
                       4 intermediate allocs per image + 1 concat at the end.

  template_pointwise   reproduces qwen_v2: rescale + normalize inlined into
                       the resize pixel loop. 1 intermediate alloc per image
                       (normalized spatial image) + patchify alloc + concat.

  template_full        reproduces qwen_v3: bilinear + rescale + normalize +
                       patchify fused into one prange-over-batch loop that
                       writes directly into a pre-allocated output buffer.

The factories take a coord registry entry (dsl/coords.py) and any model
specifics (dims, constants) flow in as @njit kernel arguments at call time.
"""

import math

import numpy as np
import numba
from numba import njit, prange

from kernels.bilinear import bilinear_filter_sample, bilinear_resize


def _patch_output_size(orig_h, orig_w, target_h, target_w):
    """Aspect-ratio-preserving resize dims, matching HF get_patch_output_size."""
    scale_w = target_w / orig_w
    scale_h = target_h / orig_h
    if scale_w < scale_h:
        new_w = target_w
        new_h = min(math.ceil(orig_h * scale_w), target_h)
    else:
        new_h = target_h
        new_w = min(math.ceil(orig_w * scale_h), target_w)
    return new_h, new_w


# ---------------------------------------------------------------------------
# template_naive — every stage its own loop + buffer (v1 schedule)
# ---------------------------------------------------------------------------

def make_template_naive(coord, parallel_batch: bool = False):
    """Return a (imgs, out_h_arr, out_w_arr, mean, std, scale, P, T, M) → pv callable.

    Mirrors qwen_v1_naive: four separate @njit functions, four buffers per
    image, np.concatenate at the end.
    """
    col_fn = coord["col_fn"]
    patch_index = coord["patch_index"]

    @njit(cache=True)
    def _rescale(img, scale):
        h, w, c = img.shape
        out = np.empty((h, w, c), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                for ch in range(c):
                    out[y, x, ch] = img[y, x, ch] * scale
        return out

    @njit(cache=True)
    def _normalize(img, mean, std):
        h, w, c = img.shape
        out = np.empty((h, w, c), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                for ch in range(c):
                    out[y, x, ch] = (img[y, x, ch] - mean[ch]) / std[ch]
        return out

    @njit(cache=True)
    def _patchify(img, patch_size, temporal_patch_size, merge_size):
        h, w, c = img.shape
        ph = h // patch_size
        pw = w // patch_size
        n_patches = ph * pw
        patch_dim = temporal_patch_size * c * patch_size * patch_size
        out = np.empty((n_patches, patch_dim), dtype=np.float32)
        for i_ph in range(ph):
            for i_pw in range(pw):
                p_idx = patch_index(i_ph, i_pw, pw, merge_size)
                for t in range(temporal_patch_size):
                    for ch in range(c):
                        for py in range(patch_size):
                            for px in range(patch_size):
                                src_y = i_ph * patch_size + py
                                src_x = i_pw * patch_size + px
                                col = col_fn(
                                    t, ch, py, px, c, patch_size, temporal_patch_size,
                                )
                                out[p_idx, col] = img[src_y, src_x, ch]
        return out

    if parallel_batch:
        @njit(parallel=True, cache=True)
        def _batch_kernel(imgs, out_h_arr, out_w_arr, patch_offsets, mean, std, scale,
                          output, patch_size, temporal_patch_size, merge_size):
            n = np.int64(len(imgs))
            for b in prange(n):
                arr = imgs[b]
                resized = bilinear_resize(arr, out_h_arr[b], out_w_arr[b])
                rescaled = _rescale(resized, scale)
                normalized = _normalize(rescaled, mean, std)
                patched = _patchify(normalized, patch_size, temporal_patch_size, merge_size)
                base = patch_offsets[b]
                for row in range(patched.shape[0]):
                    for col in range(patched.shape[1]):
                        output[base + row, col] = patched[row, col]

        def orchestrate(imgs_uint8, out_h_arr, out_w_arr, mean, std, scale,
                        patch_size, temporal_patch_size, merge_size):
            n = len(imgs_uint8)
            if n == 0:
                patch_dim = int(temporal_patch_size) * 3 * int(patch_size) * int(patch_size)
                return np.empty((0, patch_dim), dtype=np.float32)

            n_patches_per = (out_h_arr // int(patch_size)) * (out_w_arr // int(patch_size))
            patch_offsets = np.zeros(n, dtype=np.int64)
            for i in range(1, n):
                patch_offsets[i] = patch_offsets[i - 1] + n_patches_per[i - 1]
            total_patches = int(patch_offsets[-1] + n_patches_per[-1])
            patch_dim = int(temporal_patch_size) * 3 * int(patch_size) * int(patch_size)
            output = np.empty((total_patches, patch_dim), dtype=np.float32)

            typed_imgs = numba.typed.List(imgs_uint8)
            _batch_kernel(
                typed_imgs, out_h_arr, out_w_arr, patch_offsets,
                mean, std, np.float32(scale), output,
                int(patch_size), int(temporal_patch_size), int(merge_size),
            )
            return output

        return orchestrate

    def orchestrate(imgs_uint8, out_h_arr, out_w_arr, mean, std, scale,
                    patch_size, temporal_patch_size, merge_size):
        outputs = []
        for arr, out_h, out_w in zip(imgs_uint8, out_h_arr, out_w_arr):
            resized = bilinear_resize(arr, int(out_h), int(out_w))  # alloc 1
            rescaled = _rescale(resized, np.float32(scale))         # alloc 2
            normalized = _normalize(rescaled, mean, std)            # alloc 3
            patched = _patchify(normalized,                         # alloc 4
                                int(patch_size),
                                int(temporal_patch_size),
                                int(merge_size))
            outputs.append(patched)
        return np.concatenate(outputs, axis=0)                      # alloc 5

    return orchestrate


# ---------------------------------------------------------------------------
# template_pointwise — rescale + normalize inlined into resize (v2 schedule)
# ---------------------------------------------------------------------------

def make_template_pointwise(coord, parallel_batch: bool = False):
    """Return a (imgs, out_h_arr, out_w_arr, mean, std, scale, P, T, M) → pv callable.

    Mirrors qwen_v2_fused: one fused (resize+rescale+normalize) njit kernel
    that produces a single intermediate normalized spatial image, followed by
    a separate patchify pass.
    """
    col_fn = coord["col_fn"]
    patch_index = coord["patch_index"]

    @njit(cache=True)
    def _resize_normalize(img, out_h, out_w, mean, std, scale):
        h, w, c = img.shape
        out = np.empty((out_h, out_w, c), dtype=np.float32)
        sy = h / out_h
        sx = w / out_w
        sup_y = max(1.0, sy)
        sup_x = max(1.0, sx)
        for y in range(out_h):
            for x in range(out_w):
                src_y = (y + 0.5) * sy - 0.5
                src_x = (x + 0.5) * sx - 0.5
                pixel = bilinear_filter_sample(img, src_x, src_y, sup_x, sup_y)
                for ch in range(c):
                    r = pixel[ch] * scale
                    out[y, x, ch] = (r - mean[ch]) / std[ch]
        return out

    @njit(cache=True)
    def _patchify(img, patch_size, temporal_patch_size, merge_size):
        h, w, c = img.shape
        ph = h // patch_size
        pw = w // patch_size
        n_patches = ph * pw
        patch_dim = temporal_patch_size * c * patch_size * patch_size
        out = np.empty((n_patches, patch_dim), dtype=np.float32)
        for i_ph in range(ph):
            for i_pw in range(pw):
                p_idx = patch_index(i_ph, i_pw, pw, merge_size)
                for t in range(temporal_patch_size):
                    for ch in range(c):
                        for py in range(patch_size):
                            for px in range(patch_size):
                                src_y = i_ph * patch_size + py
                                src_x = i_pw * patch_size + px
                                col = col_fn(
                                    t, ch, py, px, c, patch_size, temporal_patch_size,
                                )
                                out[p_idx, col] = img[src_y, src_x, ch]
        return out

    if parallel_batch:
        @njit(parallel=True, cache=True)
        def _batch_kernel(imgs, out_h_arr, out_w_arr, patch_offsets, mean, std, scale,
                          output, patch_size, temporal_patch_size, merge_size):
            n = np.int64(len(imgs))
            for b in prange(n):
                arr = imgs[b]
                normalized = _resize_normalize(arr, out_h_arr[b], out_w_arr[b],
                                               mean, std, scale)
                patched = _patchify(normalized, patch_size, temporal_patch_size, merge_size)
                base = patch_offsets[b]
                for row in range(patched.shape[0]):
                    for col in range(patched.shape[1]):
                        output[base + row, col] = patched[row, col]

        def orchestrate(imgs_uint8, out_h_arr, out_w_arr, mean, std, scale,
                        patch_size, temporal_patch_size, merge_size):
            n = len(imgs_uint8)
            if n == 0:
                patch_dim = int(temporal_patch_size) * 3 * int(patch_size) * int(patch_size)
                return np.empty((0, patch_dim), dtype=np.float32)

            n_patches_per = (out_h_arr // int(patch_size)) * (out_w_arr // int(patch_size))
            patch_offsets = np.zeros(n, dtype=np.int64)
            for i in range(1, n):
                patch_offsets[i] = patch_offsets[i - 1] + n_patches_per[i - 1]
            total_patches = int(patch_offsets[-1] + n_patches_per[-1])
            patch_dim = int(temporal_patch_size) * 3 * int(patch_size) * int(patch_size)
            output = np.empty((total_patches, patch_dim), dtype=np.float32)

            typed_imgs = numba.typed.List(imgs_uint8)
            _batch_kernel(
                typed_imgs, out_h_arr, out_w_arr, patch_offsets,
                mean, std, np.float32(scale), output,
                int(patch_size), int(temporal_patch_size), int(merge_size),
            )
            return output

        return orchestrate

    def orchestrate(imgs_uint8, out_h_arr, out_w_arr, mean, std, scale,
                    patch_size, temporal_patch_size, merge_size):
        outputs = []
        for arr, out_h, out_w in zip(imgs_uint8, out_h_arr, out_w_arr):
            normalized = _resize_normalize(                     # alloc 1
                arr, int(out_h), int(out_w),
                mean, std, np.float32(scale),
            )
            patched = _patchify(normalized,                     # alloc 2
                                int(patch_size),
                                int(temporal_patch_size),
                                int(merge_size))
            outputs.append(patched)
        return np.concatenate(outputs, axis=0)                  # alloc 3

    return orchestrate


# ---------------------------------------------------------------------------
# template_full — full fusion + preallocated output + prange (v3 schedule)
# ---------------------------------------------------------------------------

def make_template_full(coord, parallel_batch: bool):
    """Return a (imgs, out_h_arr, out_w_arr, patch_offsets, mean, std, scale,
                 output, P, T, M) → None callable.

    Mirrors qwen_v3_storage: bilinear + rescale + normalize + patchify fused
    into a single loop that writes directly into a pre-allocated output buffer.
    parallel_batch=True enables prange over the batch.

    The output buffer is allocated by the codegen wrapper (it knows the total
    patch count) — this kernel only fills it in.
    """
    row_fn_per_pixel = coord["row_fn_per_pixel"]
    col_fn = coord["col_fn"]

    if parallel_batch:
        @njit(parallel=True, cache=True)
        def kernel(imgs, out_h_arr, out_w_arr, patch_offsets, mean, std, scale,
                   output, patch_size, temporal_patch_size, merge_size):
            n = np.int64(len(imgs))
            for b in prange(n):
                img = imgs[b]
                h = img.shape[0]
                w = img.shape[1]
                c = img.shape[2]
                out_h = out_h_arr[b]
                out_w = out_w_arr[b]
                sy = h / out_h
                sx = w / out_w
                sup_y = max(1.0, sy)
                sup_x = max(1.0, sx)
                n_patches_w = out_w // patch_size
                base = patch_offsets[b]
                for y in range(out_h):
                    for x in range(out_w):
                        src_y = (y + 0.5) * sy - 0.5
                        src_x = (x + 0.5) * sx - 0.5
                        pixel = bilinear_filter_sample(img, src_x, src_y, sup_x, sup_y)
                        p_idx_local, py_p, px_p = row_fn_per_pixel(
                            y, x, patch_size, n_patches_w, merge_size,
                        )
                        p_idx = base + p_idx_local
                        for ch in range(c):
                            r = pixel[ch] * scale
                            v = (r - mean[ch]) / std[ch]
                            for t in range(temporal_patch_size):
                                col = col_fn(
                                    t, ch, py_p, px_p,
                                    c, patch_size, temporal_patch_size,
                                )
                                output[p_idx, col] = v
    else:
        @njit(cache=True)
        def kernel(imgs, out_h_arr, out_w_arr, patch_offsets, mean, std, scale,
                   output, patch_size, temporal_patch_size, merge_size):
            n = np.int64(len(imgs))
            for b in range(n):
                img = imgs[b]
                h = img.shape[0]
                w = img.shape[1]
                c = img.shape[2]
                out_h = out_h_arr[b]
                out_w = out_w_arr[b]
                sy = h / out_h
                sx = w / out_w
                sup_y = max(1.0, sy)
                sup_x = max(1.0, sx)
                n_patches_w = out_w // patch_size
                base = patch_offsets[b]
                for y in range(out_h):
                    for x in range(out_w):
                        src_y = (y + 0.5) * sy - 0.5
                        src_x = (x + 0.5) * sx - 0.5
                        pixel = bilinear_filter_sample(img, src_x, src_y, sup_x, sup_y)
                        p_idx_local, py_p, px_p = row_fn_per_pixel(
                            y, x, patch_size, n_patches_w, merge_size,
                        )
                        p_idx = base + p_idx_local
                        for ch in range(c):
                            r = pixel[ch] * scale
                            v = (r - mean[ch]) / std[ch]
                            for t in range(temporal_patch_size):
                                col = col_fn(
                                    t, ch, py_p, px_p,
                                    c, patch_size, temporal_patch_size,
                                )
                                output[p_idx, col] = v

    return kernel


# ---------------------------------------------------------------------------
# LLaVA-NeXT AnyRes templates — three fusion levels
#
# Output: (N_total_tiles, 3, tile_size, tile_size) float32 CHW.
# Tile ordering: thumbnail first (tile[0]), then grid tiles in row-major
# order — matches HF LlavaNextImageProcessor `patches.insert(0, resized_full)`.
#
# bilinear_resize output is float32 in [0, 255] (NOT [0, 1]).
# The rescale kernel applies * scale (1/255) to convert to [0, 1].
# ---------------------------------------------------------------------------

def _enumerate_tiles(arr_uint8, best_h, best_w, tile_size):
    """Enumerate AnyRes tiles as float32 HWC arrays in [0, 255].

    Matches HF LlavaNextImageProcessor exactly:
      tiles[0]:    thumbnail — full image direct-resized to tile_size×tile_size.
      tiles[1..N]: grid tiles — full image resized to an aspect-ratio-preserving
                   sub-size of (best_h, best_w), then center zero-padded to
                   (best_h, best_w), then sliced into tile_size×tile_size crops.

    Zero-padding (value 0.0) means padded pixels normalise to (-mean/std),
    matching HF which pads before rescale+normalize.
    """
    tiles = []
    # Tile 0: thumbnail — direct resize (squash), same as HF
    tiles.append(bilinear_resize(arr_uint8, tile_size, tile_size))
    # Tiles 1..N: aspect-preserving resize → center-pad → crop
    n_rows = best_h // tile_size
    n_cols = best_w // tile_size
    if n_rows * n_cols > 0:
        orig_h, orig_w = arr_uint8.shape[:2]
        new_h, new_w = _patch_output_size(orig_h, orig_w, best_h, best_w)
        pad_top  = (best_h - new_h) // 2
        pad_left = (best_w - new_w) // 2
        resized = bilinear_resize(arr_uint8, new_h, new_w)
        c = arr_uint8.shape[2]
        full_res = np.zeros((best_h, best_w, c), dtype=np.float32)
        full_res[pad_top:pad_top + new_h, pad_left:pad_left + new_w, :] = resized
        for r in range(n_rows):
            for ci in range(n_cols):
                y0, x0 = r * tile_size, ci * tile_size
                tiles.append(full_res[y0:y0 + tile_size, x0:x0 + tile_size, :])
    return tiles


# ---------------------------------------------------------------------------
# template_llava_naive — every stage its own loop + buffer (v1 schedule)
# ---------------------------------------------------------------------------

def make_template_llava_naive():
    """Return a (imgs_uint8, tile_infos, mean, std, scale, tile_size)
    → (pixel_values, n_tiles_per_image) callable.

    Mirrors LLaVA v1: four separate @njit kernels per tile, four buffers,
    np.stack at the end. Still materializes the full-res intermediate for
    grid tiles.
    """

    @njit(cache=True)
    def _rescale_tile(tile_hwc, scale):
        h, w, c = tile_hwc.shape
        out = np.empty((h, w, c), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                for ch in range(c):
                    out[y, x, ch] = tile_hwc[y, x, ch] * scale
        return out

    @njit(cache=True)
    def _normalize_tile(tile_hwc, mean, std):
        h, w, c = tile_hwc.shape
        out = np.empty((h, w, c), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                for ch in range(c):
                    out[y, x, ch] = (tile_hwc[y, x, ch] - mean[ch]) / std[ch]
        return out

    @njit(cache=True)
    def _hwc_to_chw(tile_hwc):
        h, w, c = tile_hwc.shape
        out = np.empty((c, h, w), dtype=np.float32)
        for ch in range(c):
            for y in range(h):
                for x in range(w):
                    out[ch, y, x] = tile_hwc[y, x, ch]
        return out

    def orchestrate(imgs_uint8, tile_infos, mean, std, scale, tile_size):
        all_tiles_chw = []
        n_tiles_per_image = []
        for arr, (best_h, best_w) in zip(imgs_uint8, tile_infos):
            tiles_hwc = _enumerate_tiles(arr, int(best_h), int(best_w), tile_size)
            img_tiles = []
            for tile_hwc in tiles_hwc:
                rescaled   = _rescale_tile(tile_hwc, np.float32(scale))
                normalized = _normalize_tile(rescaled, mean, std)
                tile_chw   = _hwc_to_chw(normalized)
                img_tiles.append(tile_chw)
            n_tiles_per_image.append(len(img_tiles))
            all_tiles_chw.extend(img_tiles)
        if all_tiles_chw:
            pv = np.stack(all_tiles_chw, axis=0)
        else:
            pv = np.empty((0, 3, tile_size, tile_size), dtype=np.float32)
        return pv, np.array(n_tiles_per_image, dtype=np.int64)

    return orchestrate


# ---------------------------------------------------------------------------
# template_llava_pointwise — rescale+normalize+CHW fused per tile (v2 schedule)
# ---------------------------------------------------------------------------

def make_template_llava_pointwise():
    """Return a (imgs_uint8, tile_infos, mean, std, scale, tile_size)
    → (pixel_values, n_tiles_per_image) callable.

    Mirrors LLaVA v2: one fused kernel per tile that rescales, normalizes,
    and transposes to CHW in a single loop. Eliminates the three separate
    per-tile HWC intermediate buffers; the full-res resize intermediate is
    still allocated for grid tiles.
    """

    @njit(cache=True)
    def _rescale_normalize_chw(tile_hwc, mean, std, scale):
        # tile_hwc: (H, W, C) float32 in [0, 255] from bilinear_resize
        h, w, c = tile_hwc.shape
        out = np.empty((c, h, w), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                for ch in range(c):
                    r = tile_hwc[y, x, ch] * scale      # [0,255]*1/255 → [0,1]
                    out[ch, y, x] = (r - mean[ch]) / std[ch]
        return out

    def orchestrate(imgs_uint8, tile_infos, mean, std, scale, tile_size):
        all_tiles_chw = []
        n_tiles_per_image = []
        for arr, (best_h, best_w) in zip(imgs_uint8, tile_infos):
            tiles_hwc = _enumerate_tiles(arr, int(best_h), int(best_w), tile_size)
            img_tiles = []
            for tile_hwc in tiles_hwc:
                tile_chw = _rescale_normalize_chw(tile_hwc, mean, std, np.float32(scale))
                img_tiles.append(tile_chw)
            n_tiles_per_image.append(len(img_tiles))
            all_tiles_chw.extend(img_tiles)
        if all_tiles_chw:
            pv = np.stack(all_tiles_chw, axis=0)
        else:
            pv = np.empty((0, 3, tile_size, tile_size), dtype=np.float32)
        return pv, np.array(n_tiles_per_image, dtype=np.int64)

    return orchestrate


# ---------------------------------------------------------------------------
# template_llava_full — full fusion, preallocated output, prange (v3 schedule)
# ---------------------------------------------------------------------------

def make_template_llava_full(parallel_tiles: bool):
    """Return a (imgs, tile_descs, img_idx, output, mean, std, scale, tile_size)
    → None kernel.

    Mirrors LLaVA v3: zero intermediates. For each output tile, directly
    samples from the original uint8 image using composed coordinates with the
    Pillow-compatible BILINEAR filter (scaled triangle, floor/ceil range bounds),
    then writes rescaled+normalized values to the preallocated CHW output.

    tile_descs: (N_total, 10) int64 —
        [orig_h, orig_w, best_h, best_w, t_row, t_col, new_h, new_w, pad_top, pad_left]
        t_row = t_col = -1 for thumbnail tiles (thumbnail: new_h=tile_size,
        new_w=tile_size, pad_top=pad_left=0).
        new_h/new_w are the aspect-preserving resize dims; pad_top/pad_left are
        the center-padding offsets that place the resized content inside best_res.

    For grid tiles, pixels in the zero-padded border write (-mean/std) directly.
    All coordinate and accumulation arithmetic uses float64 to match bilinear_resize.
    """
    if parallel_tiles:
        @njit(parallel=True, cache=True)
        def kernel(imgs, tile_descs, img_idx, output, mean, std, scale, tile_size):
            n_tiles = np.int64(output.shape[0])
            ts = np.int64(tile_size)
            for t in prange(n_tiles):
                img    = imgs[img_idx[t]]
                orig_h = tile_descs[t, 0]
                orig_w = tile_descs[t, 1]
                t_row  = tile_descs[t, 4]
                t_col  = tile_descs[t, 5]
                new_h  = tile_descs[t, 6]
                new_w  = tile_descs[t, 7]
                pad_top  = tile_descs[t, 8]
                pad_left = tile_descs[t, 9]
                sy = np.float64(orig_h) / np.float64(new_h)
                sx = np.float64(orig_w) / np.float64(new_w)
                sup_y = max(1.0, sy)
                sup_x = max(1.0, sx)
                sc64  = np.float64(scale)
                if t_row < np.int64(0):  # thumbnail: direct resize
                    for y in range(ts):
                        cy = (np.float64(y) + 0.5) * sy - 0.5
                        y_lo = max(np.int64(0), np.int64(math.floor(cy - sup_y)))
                        y_hi = min(orig_h,       np.int64(math.ceil(cy + sup_y)))
                        for x in range(ts):
                            cx = (np.float64(x) + 0.5) * sx - 0.5
                            x_lo = max(np.int64(0), np.int64(math.floor(cx - sup_x)))
                            x_hi = min(orig_w,       np.int64(math.ceil(cx + sup_x)))
                            acc0 = acc1 = acc2 = 0.0
                            tw = 0.0
                            for si in range(y_lo, y_hi):
                                wy = max(0.0, 1.0 - abs(np.float64(si) - cy) / sup_y)
                                for sj in range(x_lo, x_hi):
                                    wx = max(0.0, 1.0 - abs(np.float64(sj) - cx) / sup_x)
                                    w_ij = wy * wx
                                    if w_ij > 0.0:
                                        acc0 += w_ij * img[si, sj, 0]
                                        acc1 += w_ij * img[si, sj, 1]
                                        acc2 += w_ij * img[si, sj, 2]
                                        tw += w_ij
                            if tw > 0.0:
                                output[t, 0, y, x] = (acc0 / tw * sc64 - mean[0]) / std[0]
                                output[t, 1, y, x] = (acc1 / tw * sc64 - mean[1]) / std[1]
                                output[t, 2, y, x] = (acc2 / tw * sc64 - mean[2]) / std[2]
                else:  # grid tile: aspect-preserving resize + zero-pad
                    row_off = t_row * ts
                    col_off = t_col * ts
                    for y in range(ts):
                        for x in range(ts):
                            gy = row_off + np.int64(y)
                            gx = col_off + np.int64(x)
                            cyi = gy - pad_top
                            cxi = gx - pad_left
                            if cyi < np.int64(0) or cyi >= new_h or cxi < np.int64(0) or cxi >= new_w:
                                output[t, 0, y, x] = -mean[0] / std[0]
                                output[t, 1, y, x] = -mean[1] / std[1]
                                output[t, 2, y, x] = -mean[2] / std[2]
                            else:
                                cy = (np.float64(cyi) + 0.5) * sy - 0.5
                                cx = (np.float64(cxi) + 0.5) * sx - 0.5
                                y_lo = max(np.int64(0), np.int64(math.floor(cy - sup_y)))
                                y_hi = min(orig_h,       np.int64(math.ceil(cy + sup_y)))
                                x_lo = max(np.int64(0), np.int64(math.floor(cx - sup_x)))
                                x_hi = min(orig_w,       np.int64(math.ceil(cx + sup_x)))
                                acc0 = acc1 = acc2 = 0.0
                                tw = 0.0
                                for si in range(y_lo, y_hi):
                                    wy = max(0.0, 1.0 - abs(np.float64(si) - cy) / sup_y)
                                    for sj in range(x_lo, x_hi):
                                        wx = max(0.0, 1.0 - abs(np.float64(sj) - cx) / sup_x)
                                        w_ij = wy * wx
                                        if w_ij > 0.0:
                                            acc0 += w_ij * img[si, sj, 0]
                                            acc1 += w_ij * img[si, sj, 1]
                                            acc2 += w_ij * img[si, sj, 2]
                                            tw += w_ij
                                if tw > 0.0:
                                    output[t, 0, y, x] = (acc0 / tw * sc64 - mean[0]) / std[0]
                                    output[t, 1, y, x] = (acc1 / tw * sc64 - mean[1]) / std[1]
                                    output[t, 2, y, x] = (acc2 / tw * sc64 - mean[2]) / std[2]
    else:
        @njit(cache=True)
        def kernel(imgs, tile_descs, img_idx, output, mean, std, scale, tile_size):
            n_tiles = np.int64(output.shape[0])
            ts = np.int64(tile_size)
            for t in range(n_tiles):
                img    = imgs[img_idx[t]]
                orig_h = tile_descs[t, 0]
                orig_w = tile_descs[t, 1]
                t_row  = tile_descs[t, 4]
                t_col  = tile_descs[t, 5]
                new_h  = tile_descs[t, 6]
                new_w  = tile_descs[t, 7]
                pad_top  = tile_descs[t, 8]
                pad_left = tile_descs[t, 9]
                sy = np.float64(orig_h) / np.float64(new_h)
                sx = np.float64(orig_w) / np.float64(new_w)
                sup_y = max(1.0, sy)
                sup_x = max(1.0, sx)
                sc64  = np.float64(scale)
                if t_row < np.int64(0):  # thumbnail: direct resize
                    for y in range(ts):
                        cy = (np.float64(y) + 0.5) * sy - 0.5
                        y_lo = max(np.int64(0), np.int64(math.floor(cy - sup_y)))
                        y_hi = min(orig_h,       np.int64(math.ceil(cy + sup_y)))
                        for x in range(ts):
                            cx = (np.float64(x) + 0.5) * sx - 0.5
                            x_lo = max(np.int64(0), np.int64(math.floor(cx - sup_x)))
                            x_hi = min(orig_w,       np.int64(math.ceil(cx + sup_x)))
                            acc0 = acc1 = acc2 = 0.0
                            tw = 0.0
                            for si in range(y_lo, y_hi):
                                wy = max(0.0, 1.0 - abs(np.float64(si) - cy) / sup_y)
                                for sj in range(x_lo, x_hi):
                                    wx = max(0.0, 1.0 - abs(np.float64(sj) - cx) / sup_x)
                                    w_ij = wy * wx
                                    if w_ij > 0.0:
                                        acc0 += w_ij * img[si, sj, 0]
                                        acc1 += w_ij * img[si, sj, 1]
                                        acc2 += w_ij * img[si, sj, 2]
                                        tw += w_ij
                            if tw > 0.0:
                                output[t, 0, y, x] = (acc0 / tw * sc64 - mean[0]) / std[0]
                                output[t, 1, y, x] = (acc1 / tw * sc64 - mean[1]) / std[1]
                                output[t, 2, y, x] = (acc2 / tw * sc64 - mean[2]) / std[2]
                else:  # grid tile: aspect-preserving resize + zero-pad
                    row_off = t_row * ts
                    col_off = t_col * ts
                    for y in range(ts):
                        for x in range(ts):
                            gy = row_off + np.int64(y)
                            gx = col_off + np.int64(x)
                            cyi = gy - pad_top
                            cxi = gx - pad_left
                            if cyi < np.int64(0) or cyi >= new_h or cxi < np.int64(0) or cxi >= new_w:
                                output[t, 0, y, x] = -mean[0] / std[0]
                                output[t, 1, y, x] = -mean[1] / std[1]
                                output[t, 2, y, x] = -mean[2] / std[2]
                            else:
                                cy = (np.float64(cyi) + 0.5) * sy - 0.5
                                cx = (np.float64(cxi) + 0.5) * sx - 0.5
                                y_lo = max(np.int64(0), np.int64(math.floor(cy - sup_y)))
                                y_hi = min(orig_h,       np.int64(math.ceil(cy + sup_y)))
                                x_lo = max(np.int64(0), np.int64(math.floor(cx - sup_x)))
                                x_hi = min(orig_w,       np.int64(math.ceil(cx + sup_x)))
                                acc0 = acc1 = acc2 = 0.0
                                tw = 0.0
                                for si in range(y_lo, y_hi):
                                    wy = max(0.0, 1.0 - abs(np.float64(si) - cy) / sup_y)
                                    for sj in range(x_lo, x_hi):
                                        wx = max(0.0, 1.0 - abs(np.float64(sj) - cx) / sup_x)
                                        w_ij = wy * wx
                                        if w_ij > 0.0:
                                            acc0 += w_ij * img[si, sj, 0]
                                            acc1 += w_ij * img[si, sj, 1]
                                            acc2 += w_ij * img[si, sj, 2]
                                            tw += w_ij
                                if tw > 0.0:
                                    output[t, 0, y, x] = (acc0 / tw * sc64 - mean[0]) / std[0]
                                    output[t, 1, y, x] = (acc1 / tw * sc64 - mean[1]) / std[1]
                                    output[t, 2, y, x] = (acc2 / tw * sc64 - mean[2]) / std[2]

    return kernel
