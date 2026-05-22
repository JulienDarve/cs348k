"""Verify that AutoImageProcessor and AutoProcessor.image_processor return identical
pixel_values for each benchmarked model.

For InternVL3.5, also verifies that crop_to_patches=True produces different output
from the default crop_to_patches=False, confirming that the explicit kwarg in
get_internvl35_hf_processor() is load-bearing.

Usage:
  cd benchmarks
  python test_models.py
"""
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoProcessor

from models import MODEL_ID_QWEN, MODEL_ID_INTERNVL35, MODEL_ID_LLAVA

# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_image(width=448, height=448, seed=42):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _pixel_values(output):
    pv = output["pixel_values"]
    return pv.float() if isinstance(pv, torch.Tensor) else torch.tensor(np.array(pv)).float()


def _assert_equal(a, b, label):
    pa, pb = _pixel_values(a), _pixel_values(b)
    assert pa.shape == pb.shape, (
        f"{label}: shapes differ — AutoImageProcessor {pa.shape} vs AutoProcessor.image_processor {pb.shape}"
    )
    max_diff = (pa - pb).abs().max().item()
    assert torch.allclose(pa, pb, atol=1e-5), (
        f"{label}: pixel_values differ (max abs diff = {max_diff:.2e})"
    )
    print(f"  PASS  {label}  shape={tuple(pa.shape)}  max_diff={max_diff:.2e}")


# ──────────────────────────────────────────────────────────────────────────────
# per-model tests
# ──────────────────────────────────────────────────────────────────────────────

def test_qwen(image):
    print(f"\n[Qwen2.5-VL]  {MODEL_ID_QWEN}")
    auto_img = AutoImageProcessor.from_pretrained(MODEL_ID_QWEN)
    full_img  = AutoProcessor.from_pretrained(MODEL_ID_QWEN).image_processor

    out_auto = auto_img([image], return_tensors="pt")
    out_full = full_img([image], return_tensors="pt")
    _assert_equal(out_auto, out_full, "Qwen2.5-VL: AutoImageProcessor vs AutoProcessor.image_processor")


def test_internvl35(image):
    print(f"\n[InternVL3.5]  {MODEL_ID_INTERNVL35}")
    auto_img = AutoImageProcessor.from_pretrained(MODEL_ID_INTERNVL35)
    full_img  = AutoProcessor.from_pretrained(MODEL_ID_INTERNVL35).image_processor

    # Both should match when crop_to_patches=True is passed explicitly to AutoImageProcessor.
    out_auto = auto_img([image], crop_to_patches=True, return_tensors="pt")
    out_full = full_img([image], crop_to_patches=True, return_tensors="pt")
    _assert_equal(out_auto, out_full,
                  "InternVL3.5: AutoImageProcessor(crop_to_patches=True) vs AutoProcessor.image_processor(crop_to_patches=True)")

    # Verify crop_to_patches=True is not a no-op — use a larger image so it spans multiple
    # patches; a single 448x448 image maps to exactly one patch and the outputs are identical.
    large_image = _make_image(width=896, height=896)
    out_no_crop = auto_img([large_image], crop_to_patches=False, return_tensors="pt")
    out_auto    = auto_img([large_image], crop_to_patches=True,  return_tensors="pt")
    pv_crop   = _pixel_values(out_auto)
    pv_nocrop = _pixel_values(out_no_crop)
    assert pv_crop.shape != pv_nocrop.shape or not torch.allclose(pv_crop, pv_nocrop, atol=1e-5), (
        "InternVL3.5: crop_to_patches=True and False produced identical output — "
        "expected them to differ"
    )
    print(f"  PASS  InternVL3.5: crop_to_patches=True changes output  "
          f"(with={tuple(pv_crop.shape)} vs without={tuple(pv_nocrop.shape)})")


def test_llava(image):
    print(f"\n[LLaVA]  {MODEL_ID_LLAVA}")
    auto_img = AutoImageProcessor.from_pretrained(MODEL_ID_LLAVA)
    full_img  = AutoProcessor.from_pretrained(MODEL_ID_LLAVA).image_processor

    out_auto = auto_img([image], return_tensors="pt")
    out_full = full_img([image], return_tensors="pt")
    _assert_equal(out_auto, out_full, "LLaVA: AutoImageProcessor vs AutoProcessor.image_processor")


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    image = _make_image()
    test_qwen(image)
    test_internvl35(image)
    test_llava(image)
    print("\nAll tests passed.")
