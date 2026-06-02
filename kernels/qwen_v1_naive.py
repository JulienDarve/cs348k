"""v1 — Naive correctness baseline for the Qwen2.5-VL image preprocessor.

Each preprocessing step is its own @njit function with an intermediate ndarray
allocated between steps. Python orchestration loop over the batch.

Pipeline per image:
  smart_resize_dims  →  bilinear_resize  →  rescale  →  normalize  →  patchify
  (pure Python)          (alloc 1)          (alloc 2)   (alloc 3)     (alloc 4)

Output: np.concatenate over batch (alloc 5), shape (total_patches, T*C*P*P).

This is the algorithm IR. Every subsequent version is verified against this
one before being verified against HF fast.
"""

import math

import numpy as np
import numba
from numba import njit, prange

from kernels.bilinear import bilinear_resize
from kernels.patch_coords import patch_linear_index, patch_output_offset

# ---------------------------------------------------------------------------
# Qwen2.5-VL constants
# ---------------------------------------------------------------------------

PATCH_SIZE = 14
TEMPORAL_PATCH_SIZE = 2
MERGE_SIZE = 2
FACTOR = PATCH_SIZE * MERGE_SIZE  # smart_resize factor = patch_size * merge_size = 28
MIN_PIXELS = 3136       # 4 * 28 * 28  (processor default)
MAX_PIXELS = 12845056   # 16384 * 28 * 28

QWEN_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
QWEN_STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

# ---------------------------------------------------------------------------
# smart_resize_dims — pure Python (no array ops; called once per image)
# ---------------------------------------------------------------------------

def _round_by_factor(n, f):
    return round(n / f) * f

def _floor_by_factor(n, f):
    return math.floor(n / f) * f

def _ceil_by_factor(n, f):
    return math.ceil(n / f) * f


def smart_resize_dims(h, w, factor=FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    """Return (out_h, out_w) that are multiples of factor and within pixel budget.

    Mirrors transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize.
    """
    h_bar = max(factor, _round_by_factor(h, factor))
    w_bar = max(factor, _round_by_factor(w, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        h_bar = _floor_by_factor(h / beta, factor)
        w_bar = _floor_by_factor(w / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar = _ceil_by_factor(h * beta, factor)
        w_bar = _ceil_by_factor(w * beta, factor)
    return int(h_bar), int(w_bar)

# ---------------------------------------------------------------------------
# rescale — @njit, alloc 2
# ---------------------------------------------------------------------------

@njit(cache=True)
def rescale(img):
    """Rescale (H, W, C) float32 pixel values from [0, 255] to [0.0, 1.0]."""
    h, w, c = img.shape
    out = np.empty((h, w, c), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            for ch in range(c):
                out[y, x, ch] = img[y, x, ch] * (1.0 / 255.0)
    return out

# ---------------------------------------------------------------------------
# normalize — @njit, alloc 3
# ---------------------------------------------------------------------------

@njit(cache=True)
def normalize(img, mean, std):
    """Normalize (H, W, C) float32 by per-channel mean and std."""
    h, w, c = img.shape
    out = np.empty((h, w, c), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            for ch in range(c):
                out[y, x, ch] = (img[y, x, ch] - mean[ch]) / std[ch]
    return out

# ---------------------------------------------------------------------------
# patchify — @njit, alloc 4
# ---------------------------------------------------------------------------

@njit(cache=True)
def patchify(img, patch_size, temporal_patch_size, merge_size):
    """Convert normalized (H, W, C) float32 image to Qwen patch matrix.

    Output shape: (N_patches, C * temporal_patch_size * patch_size * patch_size)
    where N_patches = (H // patch_size) * (W // patch_size).

    Replicates the Qwen2VL fast patchify: tile image along a synthetic temporal
    axis (temporal_patch_size copies), emit patches in merge-block row order
    (gh//m, gw//m, mh, mw), with column axis ordering (C, T, P_y, P_x).
    """
    h, w, c = img.shape
    ph = h // patch_size
    pw = w // patch_size
    n_patches = ph * pw
    patch_dim = temporal_patch_size * c * patch_size * patch_size
    out = np.empty((n_patches, patch_dim), dtype=np.float32)

    for i_ph in range(ph):
        for i_pw in range(pw):
            p_idx = patch_linear_index(i_ph, i_pw, pw, merge_size)
            for t in range(temporal_patch_size):
                for ch in range(c):
                    for py in range(patch_size):
                        for px in range(patch_size):
                            src_y = i_ph * patch_size + py
                            src_x = i_pw * patch_size + px
                            col = patch_output_offset(
                                t, ch, py, px, c, patch_size, temporal_patch_size
                            )
                            out[p_idx, col] = img[src_y, src_x, ch]
    return out

# ---------------------------------------------------------------------------
# qwen_v1 — Python orchestration loop (Algorithm IR)
# ---------------------------------------------------------------------------

def qwen_v1(images, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    """Preprocess a batch of PIL images using the naive staged pipeline.

    Returns:
        pixel_values: np.ndarray of shape (total_patches, T*C*P*P), float32.
        image_grid_thw: np.ndarray of shape (N_images, 3), int64.
            Each row is (temporal_patch_size, H//patch_size, W//patch_size).
    """
    outputs = []
    grid_thw = []

    for img in images:
        arr = np.asarray(img, dtype=np.uint8)  # PIL → (H, W, 3) uint8
        h, w = arr.shape[:2]

        # ALGORITHM: compute target spatial dimensions
        out_h, out_w = smart_resize_dims(h, w, factor=FACTOR,
                                         min_pixels=min_pixels,
                                         max_pixels=max_pixels)

        # ALGORITHM: bilinear resize  →  alloc 1
        resized = bilinear_resize(arr, out_h, out_w)

        # ALGORITHM: rescale [0,255]→[0,1]  →  alloc 2
        rescaled = rescale(resized)

        # ALGORITHM: per-channel normalize  →  alloc 3
        normalized = normalize(rescaled, QWEN_MEAN, QWEN_STD)

        # ALGORITHM: spatial patchify + temporal tile  →  alloc 4
        patched = patchify(normalized, PATCH_SIZE, TEMPORAL_PATCH_SIZE, MERGE_SIZE)

        outputs.append(patched)
        grid_thw.append((1,                    # T=1 temporal slot per still image
                          out_h // PATCH_SIZE,
                          out_w // PATCH_SIZE))

    # ALGORITHM: stack batch  →  alloc 5
    pixel_values = np.concatenate(outputs, axis=0)
    image_grid_thw = np.array(grid_thw, dtype=np.int64)
    return pixel_values, image_grid_thw


# ---------------------------------------------------------------------------
# qwen_v1_batch — same staged algorithm, parallel over batch
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _v1_batch_kernel(imgs, out_h_arr, out_w_arr, patch_offsets, output,
                     mean, std, patch_size, temporal_patch_size, merge_size):
    n = np.int64(len(imgs))
    for b in prange(n):
        arr = imgs[b]

        resized = bilinear_resize(arr, out_h_arr[b], out_w_arr[b])
        rescaled = rescale(resized)
        normalized = normalize(rescaled, mean, std)
        patched = patchify(normalized, patch_size, temporal_patch_size, merge_size)

        base = patch_offsets[b]
        for row in range(patched.shape[0]):
            for col in range(patched.shape[1]):
                output[base + row, col] = patched[row, col]


def qwen_v1_batch(images, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    """Naive staged Qwen preprocessing with independent images processed in parallel."""
    n = len(images)
    patch_dim = TEMPORAL_PATCH_SIZE * 3 * PATCH_SIZE * PATCH_SIZE
    if n == 0:
        return (np.empty((0, patch_dim), dtype=np.float32),
                np.empty((0, 3), dtype=np.int64))

    arrs = [np.asarray(img, dtype=np.uint8) for img in images]

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
    n_patches_per = (out_h_arr // PATCH_SIZE) * (out_w_arr // PATCH_SIZE)
    patch_offsets = np.zeros(n, dtype=np.int64)
    for i in range(1, n):
        patch_offsets[i] = patch_offsets[i - 1] + n_patches_per[i - 1]
    total_patches = int(patch_offsets[-1] + n_patches_per[-1])

    output = np.empty((total_patches, patch_dim), dtype=np.float32)
    typed_imgs = numba.typed.List(arrs)
    _v1_batch_kernel(typed_imgs, out_h_arr, out_w_arr, patch_offsets, output,
                     QWEN_MEAN, QWEN_STD,
                     PATCH_SIZE, TEMPORAL_PATCH_SIZE, MERGE_SIZE)

    grid_thw = np.stack([
        np.array([1, int(out_h_arr[i]) // PATCH_SIZE, int(out_w_arr[i]) // PATCH_SIZE],
                 dtype=np.int64)
        for i in range(n)
    ], axis=0)

    return output, grid_thw
