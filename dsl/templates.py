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
            for b in prange(len(imgs)):
                img = imgs[int(b)]
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
            for b in range(len(imgs)):
                img = imgs[int(b)]
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
