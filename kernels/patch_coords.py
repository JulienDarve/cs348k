"""Patch addressing utilities shared across kernel versions.

patch_linear_index: used by v1/v2 to navigate the output of patchify.
patch_write_coords: used by v3 to compute the destination address directly
  inside the pixel loop, eliminating the intermediate normalized-image buffer.
"""

from numba import njit


@njit(cache=True)
def patch_linear_index(i_ph, i_pw, n_patches_w, merge_size):
    """Return the flat patch index for spatial patch position (i_ph, i_pw).

    Matches the HF Qwen2VL row ordering: patches are emitted in merge-block
    order (gh//m, gw//m, mh, mw), not simple row-major (gh, gw).
    """
    gh_outer = i_ph // merge_size
    mh = i_ph % merge_size
    gw_outer = i_pw // merge_size
    mw = i_pw % merge_size
    n_outer_w = n_patches_w // merge_size
    return ((gh_outer * n_outer_w + gw_outer) * (merge_size * merge_size)
            + mh * merge_size + mw)


@njit(cache=True)
def patch_output_offset(t, ch, py, px, c, patch_size, temporal_patch_size):
    """Return the column index inside a patch row for temporal/channel/spatial coords.

    Layout matches Qwen2VL patchify: axes ordered (C, T, P_y, P_x) contiguous.
    """
    return (ch * (temporal_patch_size * patch_size * patch_size) +
            t * (patch_size * patch_size) +
            py * patch_size +
            px)
