"""LLaVA-NeXT DSL pipeline and the three schedules (v1, v2, v3).

The same `pipeline` definition (algorithm DAG) is reused across all three
schedules — only the StageSchedule annotations and the preallocate_output
flag change.

LLaVA-NeXT AnyRes DAG: tile → rescale → normalize
  - tile: pick best resolution from image_grid_pinpoints, produce thumbnail
          (full image resized to 336×336) plus a 2D grid of 336×336 crops
          from the image resized to best_resolution. No separate resize Func
          — the bilinear resampling is implicit in the tile execution.
  - rescale/normalize: CLIP normalization per tile.
  - Output: (N_total_tiles, 3, 336, 336) CHW float32.

No patchify op — LLaVA outputs full 336×336 tile images. The ViT 14×14
patch embedding happens inside the model, not in preprocessing.

Constants from llava-hf/llava-v1.6-mistral-7b-hf preprocessor_config.json.
Verified against HF processor in tests/test_dsl_llava_correctness.py check 0.
"""

import numpy as np

from dsl.algorithm import Func, Pipeline
from dsl.schedule import Schedule, StageSchedule

# ---------------------------------------------------------------------------
# Constants — llava-hf/llava-v1.6-mistral-7b-hf
# ---------------------------------------------------------------------------

LLAVA_MEAN = np.array([0.48145467042922974, 0.45782750844955444, 0.40821072459220886], dtype=np.float32)
LLAVA_STD  = np.array([0.2686295509338379,  0.2613025903701782,  0.27577710151672363], dtype=np.float32)
LLAVA_SCALE = 1.0 / 255.0
TILE_SIZE = 336

# image_grid_pinpoints: list of [height, width] pairs (HF convention: pairs
# are iterated as `for height, width in possible_resolutions` in HF source).
# For a 1024×1024 image the best resolution is (672, 672) → 4 grid tiles +
# 1 thumbnail = 5 tiles, matching profiling data (32, 5, 3, 336, 336).
GRID_PINPOINTS = [[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]


# ---------------------------------------------------------------------------
# Algorithm DAG: tile → rescale → normalize
# ---------------------------------------------------------------------------

pipeline = (
    Pipeline()
    .add(Func(
        "tile", "tile", [],     # source op: no upstream Func inputs
        {
            "tile_policy": "llava_anyres",
            "tile_size": TILE_SIZE,
            "image_grid_pinpoints": GRID_PINPOINTS,
        },
    ))
    .add(Func(
        "rescale", "rescale", ["tile"],
        {"scale": LLAVA_SCALE},
    ))
    .add(Func(
        "normalize", "normalize", ["rescale"],
        {"mean": LLAVA_MEAN, "std": LLAVA_STD},
    ))
)
pipeline.output = "normalize"

_NAMES = pipeline.names  # ["tile", "rescale", "normalize"]


# ---------------------------------------------------------------------------
# Three schedules over the one algorithm
# ---------------------------------------------------------------------------

# v1 — every stage its own loop + buffer; no parallelism.
# Materializes: full-res intermediate + per-tile rescale/normalize/CHW buffers.
sched_v1 = Schedule(
    stages={n: StageSchedule() for n in _NAMES},
    preallocate_output=False,
)

# v2 — rescale + normalize + CHW-transpose fused into one kernel per tile.
# Materializes: full-res intermediate only; per-tile goes straight to CHW.
sched_v2 = Schedule(
    stages={
        **{n: StageSchedule() for n in _NAMES},
        "rescale":   StageSchedule(compute_at="inline"),
        "normalize": StageSchedule(compute_at="inline"),
    },
    preallocate_output=False,
)

# v3 — full fusion: no intermediates. Directly bilinear-samples from the
# original image using composed grid-tile + resize coordinates, writes
# rescaled+normalized values into a preallocated CHW output buffer.
# parallel="batch" on the tile stage triggers prange over tiles in the kernel.
sched_v3 = Schedule(
    stages={
        **{n: StageSchedule(compute_at="inline") for n in _NAMES},
        "tile":      StageSchedule(parallel="batch"),
        "normalize": StageSchedule(
            compute_at="inline",
            store_at="root",
            write_via="llava_chw",   # signals CHW direct-write to classify_fusion_llava
        ),
    },
    preallocate_output=True,
)

SCHEDULES = {"v1": sched_v1, "v2": sched_v2, "v3": sched_v3}
