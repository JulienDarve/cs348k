"""DSL package for VLM image preprocessing.

Separates algorithm (DAG of ops) from schedule (compute_at / store_at /
parallel / write_via) and dispatches to parameterized Numba templates.
See notes/dsl/dsl_implementation_plan.md for the design.
"""

from dsl.algorithm import Func, Pipeline
from dsl.schedule import Schedule, StageSchedule
from dsl.codegen import build, classify_fusion

__all__ = [
    "Func",
    "Pipeline",
    "Schedule",
    "StageSchedule",
    "build",
    "classify_fusion",
]
