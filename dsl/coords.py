"""Coordinate functions for write_via: model-specific output addressing.

Each entry in COORDS exposes the per-pixel and per-patch addressing primitives
the templates compile against. The full template (Phase A) iterates per output
pixel and needs `row_fn_per_pixel(y, x, ...) -> (p_idx, py, px)`. The naive
and pointwise templates iterate per patch and need `patch_index(i_ph, i_pw,
...) -> p_idx`. Both need `col_fn(t, ch, py, px, ...) -> col`.

For D1-D3 the only registered coord is "qwen_patch_coords", which wraps the
existing `kernels/patch_coords.py` primitives. LLaVA / InternVL coords will
register here in D4-D5.
"""

from numba import njit

from kernels.patch_coords import patch_linear_index, patch_output_offset


@njit(cache=True, inline="always")
def _qwen_row_per_pixel(y, x, patch_size, n_patches_w, merge_size):
    """Map an output-image pixel (y, x) to (patch_row_index, py_in_patch, px_in_patch).

    Qwen uses merge-block ordering on the patch axis. This wraps
    patch_linear_index for the per-pixel template.
    """
    i_ph = y // patch_size
    i_pw = x // patch_size
    py = y % patch_size
    px = x % patch_size
    return patch_linear_index(i_ph, i_pw, n_patches_w, merge_size), py, px


# Public registry. Each entry must expose the same three @njit primitives so
# the template factories can compile against them uniformly across models.
COORDS = {
    "qwen_patch_coords": {
        "row_fn_per_pixel": _qwen_row_per_pixel,
        "patch_index": patch_linear_index,
        "col_fn": patch_output_offset,
    },
}


def get(name: str):
    if name not in COORDS:
        raise KeyError(f"unknown coord {name!r}; registered: {sorted(COORDS)}")
    return COORDS[name]
