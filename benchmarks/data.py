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


def load_images_w3(n_images=32, seed=0):
    """Return n_images RGB PIL images with random sizes, aspect ratio in [0.5, 2.0].

    Width and height are sampled independently from [256, 1024], then the shorter
    side is rescaled so max/min <= 2.0. Seeded for reproducibility.
    """
    rng = np.random.default_rng(seed)
    images = []
    for _ in range(n_images):
        w = int(rng.integers(256, 1024))
        h = int(rng.integers(256, 1024))
        ratio = max(w, h) / min(w, h)
        if ratio > 2.0:
            if w > h:
                h = w // 2
            else:
                w = h // 2
        arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        images.append(Image.fromarray(arr))
    return images


def load_images_smooth(n_images=4, seed=0):
    """Return n_images RGB PIL images with smooth, low-frequency content.

    Each image is a sum of a few low-frequency 2D sinusoids per channel, so
    neighboring pixels are highly correlated. This makes bilinear vs bicubic
    resize agree to within ~1% of the normalized value range, suitable for a
    tight (atol=0.1) correctness check against the HF fast processor.
    Sizes vary per image to exercise smart_resize.
    """
    rng = np.random.default_rng(seed)
    sizes = [(384, 512), (640, 480), (512, 512), (720, 480)]
    images = []
    for i in range(n_images):
        h, w = sizes[i % len(sizes)]
        yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
        arr = np.zeros((h, w, 3), dtype=np.float32)
        for ch in range(3):
            for _ in range(3):
                fy = rng.uniform(0.5, 3.0)
                fx = rng.uniform(0.5, 3.0)
                phase = rng.uniform(0, 2 * np.pi)
                amp = rng.uniform(0.2, 1.0)
                arr[..., ch] += amp * np.sin(2 * np.pi * (fy * yy + fx * xx) + phase)
        arr -= arr.min(axis=(0, 1), keepdims=True)
        arr /= arr.max(axis=(0, 1), keepdims=True) + 1e-8
        arr = (arr * 255.0).astype(np.uint8)
        images.append(Image.fromarray(arr))
    return images


def load_images_w4(n_images=8, seed=0):
    """Return n_images synthetic RGB PIL images at A4 @ 300 dpi (2480x3508)."""
    rng = np.random.default_rng(seed)
    arrs = rng.integers(0, 256, size=(n_images, 3508, 2480, 3), dtype=np.uint8)
    return [Image.fromarray(a) for a in arrs]
