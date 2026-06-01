"""Op registry: the small set of op kinds the DSL understands.

Each op kind has a small schema: which params are required (algorithmic) and
which inputs it expects. The codegen reads this registry to validate Pipelines
and to gather op-specific parameters (mean/std/scale/patch sizes/coord name).

Only the ops needed by the three target models (Qwen, LLaVA, InternVL) are
registered. D1-D3 needs only the first four; tile/center_crop are stubs that
will be fleshed out in D4-D5.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class OpSpec:
    name: str
    required_params: tuple[str, ...]
    n_inputs: int  # number of upstream Funcs this op consumes; -1 means "0 or more source"


OPS: dict[str, OpSpec] = {
    # resize: source op. params include "interp" (algorithm choice) and
    # "dims_fn" (name of the dimension-computation function, e.g.
    # "qwen_smart_resize"; resolved by the codegen).
    "resize": OpSpec(
        name="resize",
        required_params=("interp", "dims_fn"),
        n_inputs=0,
    ),

    # rescale: pointwise multiply by a scalar (typically 1/255).
    "rescale": OpSpec(
        name="rescale",
        required_params=("scale",),
        n_inputs=1,
    ),

    # normalize: per-channel (x - mean) / std.
    "normalize": OpSpec(
        name="normalize",
        required_params=("mean", "std"),
        n_inputs=1,
    ),

    # patchify: spatial patchify + temporal tile. params include the layout
    # parameters and the model-specific coord function name.
    "patchify": OpSpec(
        name="patchify",
        required_params=("patch_size", "merge_size", "temporal_patch_size", "coord"),
        n_inputs=1,
    ),

    # tile / center_crop: stubs for D4-D5 (LLaVA, InternVL).
    "tile": OpSpec(
        name="tile",
        required_params=(),
        n_inputs=1,
    ),
    "center_crop": OpSpec(
        name="center_crop",
        required_params=("size",),
        n_inputs=1,
    ),
}


def validate(pipeline) -> None:
    """Raise ValueError if the Pipeline references unknown ops or is missing required params."""
    seen: set[str] = set()
    for f in pipeline.funcs:
        if f.op not in OPS:
            raise ValueError(f"unknown op {f.op!r} on Func {f.name!r}; known: {sorted(OPS)}")
        spec = OPS[f.op]
        missing = [p for p in spec.required_params if p not in f.params]
        if missing:
            raise ValueError(
                f"Func {f.name!r} (op={f.op!r}) missing required params: {missing}"
            )
        for inp in f.inputs:
            if inp not in seen:
                raise ValueError(
                    f"Func {f.name!r} consumes {inp!r} which is not defined upstream"
                )
        seen.add(f.name)
    if pipeline.output and pipeline.output not in seen:
        raise ValueError(f"Pipeline output {pipeline.output!r} is not a defined Func")
