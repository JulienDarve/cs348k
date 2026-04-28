# Mini-Halide for VLM Image Preprocessing

A scheduling DSL for high-performance image preprocessing pipelines, applied to vision-language model inference.

## Motivation

Image preprocessing for machine learning has historically been a flat, uninteresting workload. In the AlexNet / VGG / ResNet era, preprocessing meant: resize to a fixed input size (224×224 or 256×256 then crop), subtract a per-channel mean, optionally flip. One pass over the pixels, fixed output shape, embarrassingly parallel across the batch. The whole pipeline ran in microseconds and was dwarfed by the convolutional backbone, so nobody fused stages or optimized for cache locality — the cost was negligible and the libraries that emerged (PIL, torchvision transforms, the HuggingFace image processors) reflected that assumption.

Modern vision-language model pipelines have a fundamentally different shape:

- **Variable native resolutions.** Models want to see something close to the actual aspect ratio rather than a square crop, so resize ratios are no longer fixed.
- **Anyres / dynamic tiling.** LLaVA-NeXT, InternVL, and Qwen2-VL split a high-resolution image into a grid of sub-images chosen at runtime from a candidate set of aspect ratios. A single image becomes 5–13 tiles plus a thumbnail.
- **Patching into ViT tokens.** Each tile is patched into 14×14 or 16×16 tokens, sometimes with pixel-shuffle for spatial reduction.
- **Multiple normalization conventions.** CLIP stats, ImageNet stats, and SigLIP stats coexist in the same fleet.
- **Padding to grid-aligned sizes** when crops don't divide evenly.

The pipeline is now deep, conditional, and data-movement-heavy. At inference time it's increasingly visible in profiles — especially for high-resolution serving where one request expands into a dozen tile preprocessings on the CPU. The libraries haven't caught up: they were optimized for the AlexNet-era workload and treat each transform as an independent function call that fully materializes its output before the next stage begins.

The specific inefficiencies in existing implementations are structural, not local:

- **No cross-stage fusion.** torchvision transforms compose via function composition, which forecloses fusion. Each stage writes its full output to DRAM; the next stage reads it back.
- **Format-conversion thrash.** PIL ↔ NumPy ↔ torch.Tensor round-trips, HWC ↔ CHW transposes, uint8 ↔ float32 promotions appear as full passes between stages that disagree on representation.
- **Per-op optimization in isolation.** PIL's resize and torchvision's tensor resize are individually well-tuned, but locally optimal kernels compose into globally suboptimal pipelines because no kernel knows what the next one wants.
- **Wrong parallelism granularity.** torchvision's parallelism is "the DataLoader spawns N worker processes." This works for training throughput but is wrong for inference, where intra-image parallelism (across tiles, rows, channels) is what's available.
- **Committed to a stale workload.** The libraries picked one point in the schedule space, reasonable for 2018, and the workload has moved.

None of these are fixable by writing tighter individual kernels. They are *compositional* failures — they live in the relationships between stages — and recovering them requires exposing the schedule as a parameter rather than baking it into code structure.

## Background: Algorithm/Schedule Separation

Halide (Ragan-Kelley et al., 2013) introduced the central abstraction this project builds on: separating the *algorithm* (what value goes at each output coordinate) from the *schedule* (how the loops execute, in what order, with what tiling, parallelism, and fusion). The same algorithm under different schedules produces the same pixels but can differ by an order of magnitude in runtime. Halide's contribution was a language in which the schedule is a first-class, swappable object — letting the programmer (or an autoscheduler) explore the space without rewriting the math.

The classical scheduling primitives are:

- **Loop reordering and splitting/tiling** — break a loop into outer and inner loops at chosen factors.
- **Parallelism** — mark a loop to run across threads.
- **Vectorization** — mark an inner loop to lower to SIMD.
- **`compute_at` / `store_at`** — for each producer stage, decide where in the consumer's loop nest it gets computed and where its buffer is allocated. This is the heart of fusion: `compute_root` materializes fully, `compute_inline` substitutes into the consumer's expression, `compute_at(consumer, tile_outer)` fuses at tile granularity.

These primitives were designed for image processing pipelines (camera ISPs, stencils, bilateral filters) — workloads with regular access patterns, static producer-consumer relationships, and rectangular domains. VLM preprocessing has exactly this shape, which is why the abstractions transfer.

## Central Idea

Build a small Python library that applies Halide-style algorithm/schedule separation to VLM image preprocessing. The library exposes:

1. An **algorithm layer** where users describe operations as pure functions from output coordinate to value (`resize`, `normalize`, `patch`, `tile`, `pad`).
2. A **schedule layer** where users attach scheduling decisions to each stage of a pipeline — tile sizes, fusion granularity (`compute_at`), parallelism axis, vectorization width.
3. A **lowering pass** that walks the (algorithm, schedule) pair and generates an executable function — a Numba-jitted Python function for the CPU backend, with optional CuPy/Triton generation for a single GPU comparison row.

The same algorithm can be lowered under multiple schedules without modification. This is what enables both the engineering goal (pick the right schedule per pipeline shape) and the scientific goal (run a controlled ablation over the schedule space with the algorithm held fixed).

## Hypothesis

**Schedule choice — fusion, tiling, parallelism, vectorization — materially affects runtime on representative VLM preprocessing pipelines, with at least 2× spread across the schedule space.** If true, this domain warrants the same algorithm/schedule separation Halide brought to camera pipelines and stencils. If the spread is smaller, the result is itself interesting — it would suggest preprocessing is too simple for scheduling DSLs to earn their keep, and that a fixed library of fused implementations would suffice.

The hypothesis is decomposable. The expected ordering of effect sizes:

1. **Fusion (compute_at placement)** — large effect, probably 2–4× on memory-bound chains, because preprocessing is bandwidth-limited and fusion eliminates intermediate DRAM round-trips.
2. **Tile size** — moderate effect (1.2–1.5× spread), meaningful only in the presence of fusion, because tiling controls cache residency of the intermediate.
3. **Parallelism axis** — moderate effect on multi-core, depends on workload shape (batch-dominated vs. tile-dominated).
4. **Vectorization** — small to moderate, often partially captured by Numba's auto-vectorization without explicit hints.

The ablation is what tests these predictions. Each schedule axis is varied independently with the algorithm held fixed, producing a Halide-paper Figure 6-style table.

## Key Assumptions

- **Preprocessing is on the critical path for at least one realistic VLM inference workload.** This will be verified by profiling a real pipeline (LLaVA-NeXT or InternVL at high resolution, batch 1) before finalizing the evaluation. If preprocessing is <5% of end-to-end latency on every workload tested, the end-to-end speedup chart becomes a weak result and the project's contribution narrows to the schedule-axis ablation.
- **The workload is memory-bandwidth-bound rather than compute-bound.** This is what makes fusion the dominant scheduling lever. Bilinear resize is ~4 multiply-adds per output pixel; normalize is one. The arithmetic is trivial relative to the data movement.
- **Numba can produce inner kernels of competitive quality.** The project is not about hand-writing AVX2 intrinsics. The scheduling contribution is at the loop-nest level; the inner kernel quality is delegated to Numba. If Numba's vectorization is materially worse than torchvision's underlying ATen kernels, that confound has to be acknowledged in the evaluation.
- **The static, regular structure assumption holds.** No data-dependent control flow, no dynamic shapes that change per call within a pipeline. Anyres tiling chooses a tile grid per image, but within a chosen grid the work is regular.

## Methods

### Implementation Sketch

The library has roughly four components:

**Expression IR.** A small set of node types (5–10): input access, arithmetic, function call, reduction, region access. Each operation is built from these.

**Func / Pipeline abstraction.** A `Func` holds an algorithm — a function from (output coordinate, channel) to value. A `Pipeline` is a topologically-ordered set of Funcs with producer-consumer edges.

**Schedule object.** Attached to each Func, storing tile factors, parallel/vectorize annotations, and the `compute_at` placement that determines fusion granularity. Schedules are independent of algorithms and can be swapped freely.

**Lowering pass.** Walks the pipeline graph in topological order and emits a Python source string for a Numba `@njit(parallel=True)` function. Outer tile loops, parallel `prange` for parallelism axes, explicit inner loops for tiling, and Numba's vectorization for SIMD. The emitted function is JIT-compiled and benchmarked.

The op set is capped at ~5–7 operations, locked in week 1: bilinear resize, bicubic resize, normalize, patch, tile, pad. The library is roughly a few thousand lines of Python — achievable in a quarter — and a real instance of the algorithm/schedule separation rather than a toy.

### Backend Strategy

CPU is the primary target. The motivation matches: real VLM inference deployments run preprocessing on CPU because GPU memory and compute are reserved for the model. The Halide scheduling primitives also map most directly to CPU hardware (cache hierarchy, thread pool, SIMD width). The implementation path is short: Numba `@njit(parallel=True)` with `prange` for parallel axes and explicit tile loops in generated source.

A single GPU comparison row is a stretch goal, implemented via CuPy or Triton, included only to contextualize the CPU result. It is not a primary contribution because (a) NVIDIA's preprocessing stack (DALI, nvJPEG, NPP) is already highly optimized on GPU, (b) GPU preprocessing isn't standard in deployment, and (c) Triton is effectively a Halide-on-GPU and would compete with rather than complement this work.

### Evaluation

Three results, in order of importance:

**1. Schedule-axis ablation (the scientific contribution).** A table in the style of the Halide paper's Figure 6. Algorithm fixed, schedule axes varied independently — tile size {32, 64, 128}, fusion {none, at outer, inline}, parallelism {none, batch, tile}, vectorization {off, on}. Reports runtime spread across the schedule space and the marginal contribution of each axis. This *is* the test of the central hypothesis.

**2. Microbenchmark vs. baselines.** Bar chart of runtime and memory usage across 3–4 representative pipelines, comparing the best schedule found against torchvision transforms, HuggingFace image processors, and a torch.compile baseline. The torch.compile baseline is included to avoid measuring against a strawman — it captures some fusion automatically and is a fairer comparison than raw torchvision.

**3. End-to-end VLM pipeline.** Chart showing preprocessing speedup translated to end-to-end VLM inference latency. Conditional on profiling showing preprocessing is meaningfully on the critical path (>15% of latency on at least one pipeline). Realistic candidate workloads: LLaVA-NeXT at 1344×1344 batch 1, InternVL with anyres tiling at high resolution, batched serving at moderate resolution.

The evaluation is structured to be honest about what's being measured. The microbenchmark mixes kernel-quality wins with scheduling wins; the ablation isolates the scheduling contribution. Both are reported separately.

## Differentiation

The closest existing work and how this project differs:

**Halide.** This project is straightforwardly inspired by Halide and uses its abstractions. The novelty is not in the abstractions themselves but in (a) applying them to a workload Halide didn't target — VLM-era image preprocessing with anyres tiling — and (b) the empirical study of whether scheduling actually matters in this domain, which is not obvious for ops this simple.

**torchvision / HuggingFace image processors.** These are libraries of optimized implementations, each committed to one point in the schedule space. They cannot fuse across stages and cannot adapt scheduling to pipeline shape. This project's contribution is making the schedule a swappable parameter, which is a structural difference, not a quantitative one.

**NVIDIA DALI.** Graph-based preprocessing library with some fusion. Closest in spirit. Differences: DALI is GPU-focused, doesn't expose Halide-style explicit schedule control to the user, and is optimized for training-time throughput rather than inference-time latency. DALI is the right comparison point if the project were primarily an engineering effort, but DALI doesn't enable the schedule-axis ablation that is this project's scientific contribution.

**torch.compile / TorchInductor.** Captures some fusion automatically post-tracing. Will be included as a baseline. It optimizes a different abstraction layer (the PyTorch op graph) and doesn't expose schedule choices to the user. It is also weaker on PIL-backed and irregular ops, which dominate preprocessing.

**Triton.** Effectively Halide-on-GPU. Excellent for GPU kernel scheduling. Not a CPU tool, and not focused on preprocessing pipelines. Mentioned for completeness; it would be the right comparison for a GPU-primary version of this project, which this is not.

**Custom fused Numba functions.** The strongest skeptical alternative: "couldn't you just hand-write a fused Numba function for each pipeline and capture most of the win?" Yes, partly. That gives you one point in the schedule space per pipeline. It does not give you the controlled ablation that tests whether schedule choice matters, and it does not give you the ability to adapt scheduling to pipeline shape without rewriting code. The DSL framing earns its keep when the schedule space needs to be explored empirically and varied per workload — both of which are true here.

## Scope and Risks

**Scope discipline.** The op set is locked at ~5–7 operations in week 1 and does not grow. Language design has a long tail; new operations cost weeks. The DSL is a Python library with a fluent API, not a parser-and-IR DSL. The autoscheduler — searching the schedule space automatically — is explicitly out of scope; the schedule is hand-specified for the ablation.

**Beating torchvision risks.** torchvision's PIL backend is highly optimized for the workloads it was designed for. The speedup target is on pipeline shapes where torchvision is structurally weak — large batches of small images, anyres tiling at high resolution, fused chains where torchvision's per-stage materialization wastes bandwidth. Picking the wrong evaluation pipeline could produce small or no speedup, which would weaken the engineering result (though not the scheduling-ablation result).

**End-to-end speedup risks.** The model forward pass dominates VLM inference latency. Even a 3× preprocessing speedup may translate to a 5–10% end-to-end improvement, not 50%. The end-to-end chart is contingent on profiling first; if preprocessing is too small a fraction of total latency, this result is dropped and the contribution narrows to the ablation and microbenchmark.

**Numba kernel-quality confound.** If Numba's auto-vectorization produces materially worse inner loops than ATen's hand-tuned kernels, some of the measured speedup gap might come from that direction rather than from fusion and scheduling. The ablation partly controls for this (schedule axes are varied with kernel infrastructure held fixed), but it is a known confound for the microbenchmark and will be acknowledged.

## Summary

The contribution is an abstraction (algorithm/schedule separation for VLM image preprocessing), an implementation of that abstraction (a small Python library that lowers to Numba), and an empirical study of whether the abstraction's central claim — that schedule choice matters — holds in this domain. The deliverable is not a faster preprocessor; it is evidence about *which schedules win where*, with faster preprocessing as a corollary. Whether the central hypothesis is confirmed or refuted, the result is informative: either VLM preprocessing joins the list of domains where Halide-style scheduling is the right abstraction, or it doesn't, and the boundary of where the abstraction earns its keep gets sharper.
