"""Qwen2.5-VL DSL pipeline and the three schedules (v1, v2, v3).

The same `pipeline` definition (algorithm DAG) is reused across all three
schedules — only the StageSchedule annotations and the preallocate_output
flag change. This is the headline demonstration of algorithm/schedule
separation called out in §1 of notes/dsl/dsl_implementation_plan.md.

Constants (PATCH_SIZE, MERGE_SIZE, MIN_PIXELS, MAX_PIXELS, QWEN_MEAN,
QWEN_STD) are imported from kernels/qwen_v1_naive.py to keep one source of
truth for Qwen processor configuration.
"""

from dsl.algorithm import Func, Pipeline
from dsl.schedule import Schedule, StageSchedule
from kernels.qwen_v1_naive import (
    PATCH_SIZE,
    TEMPORAL_PATCH_SIZE,
    MERGE_SIZE,
    MIN_PIXELS,
    MAX_PIXELS,
    QWEN_MEAN,
    QWEN_STD,
)


# ---------------------------------------------------------------------------
# Algorithm DAG: resize → rescale → normalize → patchify
# ---------------------------------------------------------------------------

pipeline = (
    Pipeline()
    .add(Func(
        "resize", "resize", [],
        {
            "interp": "bilinear",
            "dims_fn": "qwen_smart_resize",
            "min_pixels": MIN_PIXELS,
            "max_pixels": MAX_PIXELS,
        },
    ))
    .add(Func(
        "rescale", "rescale", ["resize"],
        {"scale": 1.0 / 255.0},
    ))
    .add(Func(
        "normalize", "normalize", ["rescale"],
        {"mean": QWEN_MEAN, "std": QWEN_STD},
    ))
    .add(Func(
        "patchify", "patchify", ["normalize"],
        {
            "patch_size": PATCH_SIZE,
            "merge_size": MERGE_SIZE,
            "temporal_patch_size": TEMPORAL_PATCH_SIZE,
            "coord": "qwen_patch_coords",
        },
    ))
)
pipeline.output = "patchify"


_NAMES = pipeline.names  # ["resize", "rescale", "normalize", "patchify"]


# ---------------------------------------------------------------------------
# Three schedules over the one algorithm
# ---------------------------------------------------------------------------

# v1 — every stage its own loop + buffer; parallel over batch axis.
# prange with 1 thread runs serially, so this schedule is correct at any thread count.
sched_v1 = Schedule(
    stages={
        **{n: StageSchedule() for n in _NAMES},
        "resize": StageSchedule(parallel="batch"),
    },
    preallocate_output=False,
)

# v2 — rescale and normalize inlined into the resize pixel loop; parallel over batch axis.
sched_v2 = Schedule(
    stages={
        **{n: StageSchedule() for n in _NAMES},
        "resize":    StageSchedule(parallel="batch"),
        "rescale":   StageSchedule(compute_at="inline"),
        "normalize": StageSchedule(compute_at="inline"),
    },
    preallocate_output=False,
)

# v3 — full fusion, write directly into pre-allocated output, parallel(batch).
# prange with 1 thread runs serially, so this schedule is correct at any thread count.
sched_v3 = Schedule(
    stages={
        **{n: StageSchedule(compute_at="inline") for n in _NAMES},
        "resize":   StageSchedule(parallel="batch"),
        "patchify": StageSchedule(
            compute_at="inline",
            store_at="root",
            write_via="qwen_patch_coords",
        ),
    },
    preallocate_output=True,
)


SCHEDULES = {
    "v1": sched_v1,
    "v2": sched_v2,
    "v3": sched_v3,
}
