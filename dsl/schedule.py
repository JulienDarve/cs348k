"""Schedule IR: per-stage annotations + pipeline-level flags.

A Schedule maps Func names to StageSchedule annotations. The annotations
control *how* the algorithm runs (loop nesting, storage location, parallelism,
output addressing) without changing *what* is computed.

The vocabulary is intentionally small (§2 of dsl_implementation_plan.md):

    compute_at = "root"   | "inline"
    store_at   = "root"   | "inline" | "consumer"
    parallel   = "batch"  | "image"  | "tile"  | None
    write_via  = <coord-fn name>     | None    (output-stage only)

plus one pipeline-level flag, preallocate_output.

This module deliberately ships only the primitives the three target models
actually need. See §2 ("What you are deliberately NOT building") for the
omitted primitives (tile axis, vectorize, reorder, split, sliding-window
stencil fusion) and why.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StageSchedule:
    """Schedule annotations for a single Func.

    Defaults reproduce the v1 (all-root, no parallelism) behavior.

    compute_at: where the Func's *body* runs.
        "root"   → own loop nest (own iteration space).
        "inline" → fused into its consumer's loop (no own loop).

    store_at: where the Func's *results* live.
        "root"     → own buffer (allocated separately).
        "inline"   → register-only, no buffer.
        "consumer" → in the consumer's buffer.
        For the pipeline's output Func, "root" + preallocate_output = True
        means the output buffer is allocated once at pipeline entry and the
        fused loop writes directly into it.

    parallel: which axis is parallelized.
        "batch" → prange over images.
        "image" / "tile" → reserved for LLaVA / InternVL stretch goals.
        None    → serial.

    write_via: name of a coord function (see dsl/coords.py). Only meaningful
        on the output Func. When set, the fused template computes the
        destination address inline rather than writing row-major to an
        intermediate.
    """

    compute_at: str = "root"
    store_at: str = "root"
    parallel: Optional[str] = None
    write_via: Optional[str] = None


@dataclass
class Schedule:
    """Per-stage annotations plus pipeline-level flags."""

    stages: dict[str, StageSchedule] = field(default_factory=dict)
    preallocate_output: bool = False

    def of(self, name: str) -> StageSchedule:
        """Return the StageSchedule for `name`, or default StageSchedule()."""
        return self.stages.get(name, StageSchedule())

    def parallel_axis(self) -> Optional[str]:
        """Return the first non-None parallel axis across all stages."""
        for s in self.stages.values():
            if s.parallel is not None:
                return s.parallel
        return None
