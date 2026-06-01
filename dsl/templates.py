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

import numpy as np
import numba
from numba import njit, prange

from kernels.bilinear import bilinear_sample, bilinear_resize


# ---------------------------------------------------------------------------
# template_naive — every stage its own loop + buffer (v1 schedule)
# ---------------------------------------------------------------------------

def make_template_naive(coord):
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

def make_template_pointwise(coord):
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
        for y in range(out_h):
            for x in range(out_w):
                src_y = (y + 0.5) * sy - 0.5
                src_x = (x + 0.5) * sx - 0.5
                pixel = bilinear_sample(img, src_x, src_y)
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
                n_patches_w = out_w // patch_size
                base = patch_offsets[b]
                for y in range(out_h):
                    for x in range(out_w):
                        src_y = (y + 0.5) * sy - 0.5
                        src_x = (x + 0.5) * sx - 0.5
                        pixel = bilinear_sample(img, src_x, src_y)
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
                n_patches_w = out_w // patch_size
                base = patch_offsets[b]
                for y in range(out_h):
                    for x in range(out_w):
                        src_y = (y + 0.5) * sy - 0.5
                        src_x = (x + 0.5) * sx - 0.5
                        pixel = bilinear_sample(img, src_x, src_y)
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

    Returns list of (tile_size, tile_size, 3) float32 arrays:
      tiles[0]:    thumbnail — full image bilinear-resized to tile_size×tile_size.
      tiles[1..N]: grid tiles — full image resized to (best_h, best_w), then
                   sliced into non-overlapping tile_size×tile_size crops in
                   row-major order (left-to-right, top-to-bottom).

    Pure Python (not @njit): calls the njit bilinear_resize from Python.
    Shared by make_template_llava_naive and make_template_llava_pointwise.
    """
    tiles = []
    # Tile 0: thumbnail
    tiles.append(bilinear_resize(arr_uint8, tile_size, tile_size))
    # Tiles 1..N: grid crops from full-res resize
    n_rows = best_h // tile_size
    n_cols = best_w // tile_size
    if n_rows * n_cols > 0:
        full_res = bilinear_resize(arr_uint8, best_h, best_w)
        for r in range(n_rows):
            for c in range(n_cols):
                y0, x0 = r * tile_size, c * tile_size
                tiles.append(full_res[y0:y0+tile_size, x0:x0+tile_size, :])
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
    bilinear-samples from the original uint8 image using composed coordinates
    (grid-tile offset + resize scale in one pass), writes rescaled+normalized
    values to the preallocated CHW output buffer.

    tile_descs: (N_total, 6) int64 — [orig_h, orig_w, best_h, best_w, t_row, t_col]
                t_row = t_col = -1 for thumbnail tiles.
    img_idx:   (N_total,) int64 — which image in imgs each tile belongs to.
    output:    (N_total, 3, tile_size, tile_size) float32 — preallocated.

    parallel_tiles=True enables prange over N_total_tiles.
    """
    if parallel_tiles:
        @njit(parallel=True, cache=True)
        def kernel(imgs, tile_descs, img_idx, output, mean, std, scale, tile_size):
            n_tiles = np.int64(output.shape[0])
            ts = np.int64(tile_size)
            for t in prange(n_tiles):
                img = imgs[img_idx[t]]
                orig_h_f = np.float32(tile_descs[t, 0])
                orig_w_f = np.float32(tile_descs[t, 1])
                best_h_f = np.float32(tile_descs[t, 2])
                best_w_f = np.float32(tile_descs[t, 3])
                t_row    = tile_descs[t, 4]
                t_col    = tile_descs[t, 5]
                sy = orig_h_f / best_h_f
                sx = orig_w_f / best_w_f
                if t_row < np.int64(0):   # thumbnail sentinel
                    row_off = np.float32(0.0)
                    col_off = np.float32(0.0)
                else:
                    row_off = np.float32(t_row * ts)
                    col_off = np.float32(t_col * ts)
                for y in range(ts):
                    for x in range(ts):
                        src_y = (row_off + np.float32(y) + np.float32(0.5)) * sy - np.float32(0.5)
                        src_x = (col_off + np.float32(x) + np.float32(0.5)) * sx - np.float32(0.5)
                        pixel = bilinear_sample(img, src_x, src_y)
                        for ch in range(np.int64(3)):
                            r = pixel[ch] * scale
                            output[t, ch, y, x] = (r - mean[ch]) / std[ch]
    else:
        @njit(cache=True)
        def kernel(imgs, tile_descs, img_idx, output, mean, std, scale, tile_size):
            n_tiles = np.int64(output.shape[0])
            ts = np.int64(tile_size)
            for t in range(n_tiles):
                img = imgs[img_idx[t]]
                orig_h_f = np.float32(tile_descs[t, 0])
                orig_w_f = np.float32(tile_descs[t, 1])
                best_h_f = np.float32(tile_descs[t, 2])
                best_w_f = np.float32(tile_descs[t, 3])
                t_row    = tile_descs[t, 4]
                t_col    = tile_descs[t, 5]
                sy = orig_h_f / best_h_f
                sx = orig_w_f / best_w_f
                if t_row < np.int64(0):
                    row_off = np.float32(0.0)
                    col_off = np.float32(0.0)
                else:
                    row_off = np.float32(t_row * ts)
                    col_off = np.float32(t_col * ts)
                for y in range(ts):
                    for x in range(ts):
                        src_y = (row_off + np.float32(y) + np.float32(0.5)) * sy - np.float32(0.5)
                        src_x = (col_off + np.float32(x) + np.float32(0.5)) * sx - np.float32(0.5)
                        pixel = bilinear_sample(img, src_x, src_y)
                        for ch in range(np.int64(3)):
                            r = pixel[ch] * scale
                            output[t, ch, y, x] = (r - mean[ch]) / std[ch]

    return kernel
