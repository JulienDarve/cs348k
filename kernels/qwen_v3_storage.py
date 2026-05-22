"""v3 — Full fusion with pre-allocated output.

Schedule changes from v2:
  - Intermediate normalized-image buffer eliminated: the pixel loop writes
    directly to the final output tensor at the patch coordinates computed
    inline (store_at = root).
  - Output tensor pre-allocated once at function entry for the whole batch,
    sized (total_patches, patch_dim).
  - Batch loop parallelised with prange (schedule: parallel(batch)).

Pipeline per image (no intermediate buffers):
  smart_resize_dims  →  _v3_kernel [bilinear + rescale + normalize + patchify
  (pure Python)          all fused, writes directly to pre-allocated output]

Output: pre-allocated ndarray, shape (total_patches, T*C*P*P).

What this defines for the DSL:
  schedule[resize_normalize].store_at = "root"   (write to final output, not own buffer)
  Pipeline.preallocate_output = True             (output sized for whole batch at entry)
  schedule[batch].parallel = True                (prange over batch)
"""

import numpy as np
import numba
from numba import njit, prange

from kernels.bilinear import bilinear_sample
from kernels.patch_coords import patch_linear_index, patch_output_offset
from kernels.qwen_v1_naive import (
    smart_resize_dims,
    PATCH_SIZE, TEMPORAL_PATCH_SIZE, MERGE_SIZE,
    FACTOR, MIN_PIXELS, MAX_PIXELS,
    QWEN_MEAN, QWEN_STD,
)


# ---------------------------------------------------------------------------
# Fused kernel — @njit(parallel=True), writes directly into pre-allocated output
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _v3_kernel(imgs, out_h_arr, out_w_arr, patch_offsets, mean, std, output,
               patch_size, temporal_patch_size, merge_size):
    """Fused bilinear resize + rescale + normalize + patchify, parallel over batch.

    SCHEDULE: parallel(batch) — each image processed by an independent thread.
    SCHEDULE: serial(y, x) — output-pixel-driven inner loop.
    SCHEDULE: store_at(root) — writes go directly to pre-allocated output; no
              intermediate image buffer is allocated.
    """
    # SCHEDULE: parallel(batch), grain=1
    for b in prange(len(imgs)):
        img = imgs[np.int64(b)]
        out_h = out_h_arr[b]
        out_w = out_w_arr[b]

        # ALGORITHM: scale factors map output pixel centers to source pixel centers
        h = img.shape[0]
        w = img.shape[1]
        c = img.shape[2]
        sy = h / out_h
        sx = w / out_w

        # ALGORITHM: patch grid width for this image
        n_patches_w = out_w // patch_size
        # SCHEDULE: store_at(root) — base row index in the shared output tensor
        base = patch_offsets[b]

        # SCHEDULE: serial(y, x), output-pixel-driven
        for y in range(out_h):
            for x in range(out_w):
                # ALGORITHM: pixel-center aligned sampling (align_corners=False)
                src_y = (y + 0.5) * sy - 0.5
                src_x = (x + 0.5) * sx - 0.5
                # ALGORITHM: bilinear sample → float32 pixel vector (length C)
                pixel = bilinear_sample(img, src_x, src_y)

                # ALGORITHM: patch grid position for this output pixel
                i_ph = y // patch_size
                i_pw = x // patch_size
                py_in_patch = y % patch_size
                px_in_patch = x % patch_size
                # SCHEDULE: store_at(root) — compute destination row in final output
                p_idx = base + patch_linear_index(i_ph, i_pw, n_patches_w, merge_size)

                for ch in range(c):
                    # ALGORITHM: rescale [0,255] → [0,1]
                    # SCHEDULE: compute_at(resize, inline) — result in register only
                    r = pixel[ch] * (1.0 / 255.0)
                    # ALGORITHM: per-channel normalize (subtract mean, divide by std)
                    # SCHEDULE: compute_at(resize, inline) — result in register only
                    v = (r - mean[ch]) / std[ch]
                    # ALGORITHM: temporal tile — write same pixel value to all T slots
                    for t in range(temporal_patch_size):
                        col = patch_output_offset(
                            t, ch, py_in_patch, px_in_patch,
                            c, patch_size, temporal_patch_size,
                        )
                        # SCHEDULE: store_at(root) — write directly to final output
                        output[p_idx, col] = v


# ---------------------------------------------------------------------------
# qwen_v3 — Python orchestration: pre-allocate, then dispatch to _v3_kernel
# ---------------------------------------------------------------------------

def qwen_v3(images, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    """Preprocess a batch of PIL images with full-fusion, pre-allocated output.

    Returns:
        pixel_values: np.ndarray of shape (total_patches, T*C*P*P), float32.
        image_grid_thw: np.ndarray of shape (N_images, 3), int64.
    """
    n = len(images)
    patch_dim = TEMPORAL_PATCH_SIZE * 3 * PATCH_SIZE * PATCH_SIZE
    if n == 0:
        return (np.empty((0, patch_dim), dtype=np.float32),
                np.empty((0, 3), dtype=np.int64))

    # ALGORITHM: PIL images → uint8 numpy arrays
    arrs = [np.asarray(img, dtype=np.uint8) for img in images]

    # ALGORITHM: compute target spatial dimensions per image
    out_h_list = []
    out_w_list = []
    for arr in arrs:
        oh, ow = smart_resize_dims(arr.shape[0], arr.shape[1],
                                   factor=FACTOR,
                                   min_pixels=min_pixels,
                                   max_pixels=max_pixels)
        out_h_list.append(oh)
        out_w_list.append(ow)

    out_h_arr = np.array(out_h_list, dtype=np.int64)
    out_w_arr = np.array(out_w_list, dtype=np.int64)

    # ALGORITHM: patch counts and prefix-sum offsets over the batch
    n_patches_per = (out_h_arr // PATCH_SIZE) * (out_w_arr // PATCH_SIZE)
    patch_offsets = np.zeros(n, dtype=np.int64)
    for i in range(1, n):
        patch_offsets[i] = patch_offsets[i - 1] + n_patches_per[i - 1]
    total_patches = int(patch_offsets[-1] + n_patches_per[-1])

    # SCHEDULE: store_at(root) + preallocate_output — single allocation for whole batch
    output = np.empty((total_patches, patch_dim), dtype=np.float32)

    # SCHEDULE: parallel(batch) via prange inside _v3_kernel
    typed_imgs = numba.typed.List(arrs)
    _v3_kernel(typed_imgs, out_h_arr, out_w_arr, patch_offsets,
               QWEN_MEAN, QWEN_STD, output,
               PATCH_SIZE, TEMPORAL_PATCH_SIZE, MERGE_SIZE)

    # ALGORITHM: build image_grid_thw (T, H_patches, W_patches) per image
    grid_thw = np.stack([
        np.array([1, int(out_h_arr[i]) // PATCH_SIZE, int(out_w_arr[i]) // PATCH_SIZE],
                 dtype=np.int64)
        for i in range(n)
    ], axis=0)

    return output, grid_thw
