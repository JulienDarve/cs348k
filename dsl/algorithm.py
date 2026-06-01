"""Algorithm IR: a Pipeline is a DAG of Funcs.

A Func names one preprocessing op (resize, rescale, normalize, patchify, ...)
together with its inputs (other Func names) and op-specific params. A Pipeline
holds an ordered list of Funcs and the name of the output Func. The Pipeline
captures *what* is computed; dsl/schedule.py captures *how*.

The Func/Pipeline API mirrors the spec in
notes/dsl/dsl_implementation_plan.md §5.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Func:
    """One node in the algorithm DAG.

    Fields:
        name:   unique identifier inside its Pipeline.
        op:     op kind. One of {"resize", "rescale", "normalize", "patchify",
                "tile", "center_crop"} (only the first four are needed for
                D1-D3 Qwen).
        inputs: names of Funcs this op consumes (empty list for source ops).
        params: op-specific algorithmic parameters. The codegen reads these
                to instantiate the templates (mean/std, scale, patch_size,
                merge_size, temporal_patch_size, dims_fn name, coord name).
    """

    name: str
    op: str
    inputs: list[str]
    params: dict[str, Any]


@dataclass
class Pipeline:
    """An ordered DAG of Funcs with a designated output.

    Build with .add(); set .output to the name of the terminal Func.
    """

    funcs: list[Func] = field(default_factory=list)
    output: str = ""

    def add(self, f: Func) -> "Pipeline":
        self.funcs.append(f)
        return self

    def get(self, name: str) -> Func:
        for f in self.funcs:
            if f.name == name:
                return f
        raise KeyError(f"no Func named {name!r} in pipeline (have {self.names})")

    def has(self, name: str) -> bool:
        return any(f.name == name for f in self.funcs)

    @property
    def names(self) -> list[str]:
        return [f.name for f in self.funcs]

    def by_op(self, op: str) -> list[Func]:
        return [f for f in self.funcs if f.op == op]
