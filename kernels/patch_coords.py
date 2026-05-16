"""Patch addressing utilities shared across kernel versions.

patch_linear_index: used by v1/v2 to navigate the output of patchify.
patch_write_coords: used by v3 to compute the destination address directly
  inside the pixel loop, eliminating the intermediate normalized-image buffer.
"""

from numba import njit


@njit(cache=True)
def patch_linear_index(i_ph, i_pw, n_patches_w):
    """Return the flat patch index for spatial patch position (i_ph, i_pw)."""
    return i_ph * n_patches_w + i_pw


@njit(cache=True)
def patch_output_offset(t, ch, py, px, c, patch_size, temporal_patch_size):
    """Return the column index inside a patch row for temporal/channel/spatial coords.

    Layout matches Qwen2VL patchify: axes ordered (T, C, P_y, P_x) contiguous.
    """
    return (t * (c * patch_size * patch_size) +
            ch * (patch_size * patch_size) +
            py * patch_size +
            px)
