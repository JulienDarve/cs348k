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
    """Resize (H, W, C) uint8 to (out_h, out_w, C) float32 via bilinear interp.

    Pixel-center aligned (align_corners=False): the center of output pixel i
    maps to source position (i + 0.5) * scale - 0.5.
    """
    h, w, c = img.shape
    out = np.empty((out_h, out_w, c), dtype=np.float32)
    sy = h / out_h
    sx = w / out_w
    for y in range(out_h):
        for x in range(out_w):
            src_y = (y + 0.5) * sy - 0.5
            src_x = (x + 0.5) * sx - 0.5
            out[y, x] = bilinear_sample(img, src_x, src_y)
    return out
