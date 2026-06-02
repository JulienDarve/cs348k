"""v2 — Pointwise fusion: rescale and normalize inlined into bilinear resize.

Schedule change from v1: rescale and normalize are inlined into the inner pixel
loop of bilinear_resize (compute_at = inline). The intermediate `resized` and
`rescaled` buffers are eliminated. Only one intermediate buffer (the normalized
spatial image) survives before patchify.

Pipeline per image:
  smart_resize_dims  →  _resize_normalize (fused, alloc 1)  →  patchify (alloc 2)
  (pure Python)

Output: np.concatenate over batch (alloc 3), shape (total_patches, T*C*P*P).

What this defines for the DSL:
  schedule[rescale].compute_at  = "inline"   (into resize pixel loop)
  schedule[normalize].compute_at = "inline"  (into resize pixel loop)
"""

import numpy as np
from numba import njit

from kernels.bilinear import bilinear_filter_sample
from kernels.patch_coords import patch_linear_index, patch_output_offset
from kernels.qwen_v1_naive import (
    smart_resize_dims,
    patchify,
    PATCH_SIZE, TEMPORAL_PATCH_SIZE, MERGE_SIZE,
    FACTOR, MIN_PIXELS, MAX_PIXELS,
    QWEN_MEAN, QWEN_STD,
)


# ---------------------------------------------------------------------------
# Fused resize + rescale + normalize — @njit, alloc 1
# ---------------------------------------------------------------------------

@njit(cache=True)
def _resize_normalize(img, out_h, out_w, mean, std):
    """Bilinear resize + rescale [0,255]→[0,1] + per-channel normalize, fused.

    ALGORITHM: for each output pixel, bilinearly sample the source image.
    SCHEDULE: compute_at(rescale, inline) — rescale result lives in a register,
              not an intermediate buffer.
    SCHEDULE: compute_at(normalize, inline) — normalize result lives in a
              register, not an intermediate buffer.
    """
    h, w, c = img.shape
    # ALGORITHM: allocate the single intermediate buffer (normalized spatial image)
    out = np.empty((out_h, out_w, c), dtype=np.float32)
    # ALGORITHM: scale factors map output pixel centers to source pixel centers
    sy = h / out_h
    sx = w / out_w
    sup_y = max(1.0, sy)
    sup_x = max(1.0, sx)
    for y in range(out_h):
        for x in range(out_w):
            # ALGORITHM: pixel-center aligned sampling (align_corners=False)
            src_y = (y + 0.5) * sy - 0.5
            src_x = (x + 0.5) * sx - 0.5
            # ALGORITHM: Pillow-compatible bilinear sample → float32 pixel vector
            px = bilinear_filter_sample(img, src_x, src_y, sup_x, sup_y)
            for ch in range(c):
                # ALGORITHM: rescale [0,255] → [0,1]
                # SCHEDULE: compute_at(resize, inline) — result in register only
                r = px[ch] * (1.0 / 255.0)
                # ALGORITHM: per-channel normalize (subtract mean, divide by std)
                # SCHEDULE: compute_at(resize, inline) — result in register only
                out[y, x, ch] = (r - mean[ch]) / std[ch]
    return out


# ---------------------------------------------------------------------------
# qwen_v2 — Python orchestration loop (same structure as v1, fewer allocs)
# ---------------------------------------------------------------------------

def qwen_v2(images, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    """Preprocess a batch of PIL images using the fused resize+normalize pipeline.

    Returns:
        pixel_values: np.ndarray of shape (total_patches, T*C*P*P), float32.
        image_grid_thw: np.ndarray of shape (N_images, 3), int64.
    """
    # SCHEDULE: serial batch loop (batch parallelism is a separate schedule axis)
    outputs = []
    grid_thw = []

    for img in images:
        # ALGORITHM: PIL image → (H, W, 3) uint8
        arr = np.asarray(img, dtype=np.uint8)
        h, w = arr.shape[:2]

        # ALGORITHM: compute target spatial dimensions
        out_h, out_w = smart_resize_dims(h, w, factor=FACTOR,
                                         min_pixels=min_pixels,
                                         max_pixels=max_pixels)

        # ALGORITHM: fused bilinear resize + rescale + normalize  →  alloc 1
        # SCHEDULE: rescale and normalize computed_at inline (no separate buffers)
        normalized = _resize_normalize(arr, out_h, out_w, QWEN_MEAN, QWEN_STD)

        # ALGORITHM: spatial patchify + temporal tile  →  alloc 2
        patched = patchify(normalized, PATCH_SIZE, TEMPORAL_PATCH_SIZE, MERGE_SIZE)

        outputs.append(patched)
        grid_thw.append((1,
                          out_h // PATCH_SIZE,
                          out_w // PATCH_SIZE))

    # ALGORITHM: concatenate batch  →  alloc 3
    pixel_values = np.concatenate(outputs, axis=0)
    image_grid_thw = np.array(grid_thw, dtype=np.int64)
    return pixel_values, image_grid_thw
