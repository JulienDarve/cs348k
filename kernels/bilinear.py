"""Shared bilinear sampling and resize primitives used by all kernel versions."""

import math

import numpy as np
from numba import njit


@njit(cache=True)
def bilinear_sample(img, x_f, y_f):
    """Bilinearly sample (H, W, C) uint8 img at float coords (x_f, y_f).

    Uses pixel-center convention: coordinate 0.0 refers to the center of the
    top-left pixel. Clamps to image borders (border replication).
    Returns float32 pixel vector of length C.
    """
    h, w, c = img.shape

    x0 = int(math.floor(x_f))
    y0 = int(math.floor(y_f))
    x1 = x0 + 1
    y1 = y0 + 1

    dx = x_f - x0
    dy = y_f - y0

    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))

    out = np.empty(c, dtype=np.float32)
    for ch in range(c):
        tl = float(img[y0, x0, ch])
        tr = float(img[y0, x1, ch])
        bl = float(img[y1, x0, ch])
        br = float(img[y1, x1, ch])
        out[ch] = (tl * (1.0 - dx) * (1.0 - dy) +
                   tr * dx          * (1.0 - dy) +
                   bl * (1.0 - dx) * dy +
                   br * dx          * dy)
    return out


@njit(cache=True)
def bilinear_resize(img, out_h, out_w):
    """Resize (H, W, C) uint8 to (out_h, out_w, C) float32 using Pillow-compatible
    BILINEAR filter.

    For downsampling (scale > 1) uses a scaled triangle filter (support = scale)
    to match Pillow's antialias behavior; for upsampling degenerates to standard
    bilinear. Range bounds use floor/ceil to exactly reproduce Pillow's range
    computation.
    """
    h, w, c = img.shape
    out = np.empty((out_h, out_w, c), dtype=np.float32)
    scale_y = h / out_h
    scale_x = w / out_w
    sup_y = max(1.0, scale_y)
    sup_x = max(1.0, scale_x)

    for oi in range(out_h):
        cy = (oi + 0.5) * scale_y - 0.5
        y_min = max(0, int(math.floor(cy - sup_y)))
        y_max = min(h, int(math.ceil(cy + sup_y)))

        for oj in range(out_w):
            cx = (oj + 0.5) * scale_x - 0.5
            x_min = max(0, int(math.floor(cx - sup_x)))
            x_max = min(w, int(math.ceil(cx + sup_x)))

            tv = np.zeros(c, dtype=np.float64)
            tw = 0.0

            for si in range(y_min, y_max):
                wy = max(0.0, 1.0 - abs(si - cy) / sup_y)
                for sj in range(x_min, x_max):
                    wx = max(0.0, 1.0 - abs(sj - cx) / sup_x)
                    w_ij = wy * wx
                    if w_ij > 0.0:
                        for ch in range(c):
                            tv[ch] += w_ij * img[si, sj, ch]
                        tw += w_ij

            if tw > 0.0:
                for ch in range(c):
                    out[oi, oj, ch] = tv[ch] / tw

    return out
