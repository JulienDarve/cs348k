from pathlib import Path

import numpy as np
from PIL import Image

def load_images(img_dir=None, n_images=32, img_size=(1024, 1024)):
    """Return n_images RGB PIL images at img_size, pre-decoded into RAM.

    Reads from img_dir if given, otherwise generates synthetic random images.
    """
    if img_dir:
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        paths = sorted(p for p in Path(img_dir).rglob("*") if p.suffix.lower() in exts)
        if len(paths) < n_images:
            raise SystemExit(f"need >= {n_images} images in {img_dir}, found {len(paths)}")
        return [Image.open(p).convert("RGB").resize(img_size) for p in paths[:n_images]]
    rng = np.random.default_rng(0)
    arrs = rng.integers(0, 256, size=(n_images, img_size[1], img_size[0], 3), dtype=np.uint8)
    return [Image.fromarray(a) for a in arrs]
