# cs348k

Please go to `MILESTONE_2.md` for the write up and results for Milestone 2!



## Project Proposal

Historically, image pre-processing for machine learning models has been simple, with models hardcoded to a set image size, which may only need resizing and normalization. However, modern vision language model (VLM) pipelines work in the native pixel resolution of the image, and may perform more sophisticated pre-processing, such as dynamic tiling, patching, and multiple types of padding or normalization. In most open source implementations for training VLMs such as Huggingface or PyTorch Transforms, these pipelines are not implemented efficiently, and convert back and forth between several image formats such as numpy, tensors, or Pillow/PIL for different transformations. I seek to develop a DSL inspired by Halide for high-performance VLM image preprocessing, separating algorithm from schedule and providing primitives such as tiling, fusion (compute_at), parallelism, and vectorization. The key metrics I will need to verify are that existing implementations are in fact inefficient, and that the image preprocessing is a significant bottleneck in the pipeline, both of which I suspect to be the case, especially for video or high quality image processing. Precisely, I want a bar chart showing runtimes and memory usage for existing implementations showing that they are slow, and that my method is significantly faster (2-5x). Furthermore, I want a separate chart comparing the entire image processing pipeline including the VLM and show that faster pre-processing increases the overall speed significantly.

## Quickstart

```bash
cd cs248k
conda create -n cs348k python=3.12
conda activate cs348k
poetry install --no-root
```

## Repository structure

Note: Some results in Milestone 1 and 2, as well as some figures contain old data that may not be fully accurate. Stick to the final report pdf for ground truth numbers.

### Root files

- `LINKS.md`: Collects reference links for VLM papers, project pages, Hugging Face code, and model cards.

- `MILESTONE_1.md` Report for Milestone 1

- `MILESTONE_2.md` Report for Milestone 2

- `FINAL_RESULTS.md` Final results for the project

- `pyproject.toml` `poetry.lock` Poetry dependancy management

### `benchmarks/`
Benchmarks existing implementations (Milestone 1)

#### Core files
- `bench_kernels.py`: Benchmarks the Milestone 2 Qwen kernel implementations against HF legacy, HF fast, and HF bilinear baselines.
- `bench_dsl.py`: Extends bench_kernels.py with DSL-compiled variants (dsl_v1/v2/v3) built via dsl/codegen.py to benchmark the DSL ablation alongside hand-written kernels and HF baselines.
- `bench_dsl_llava.py`: Benchmarks LLaVA-NeXT DSL variants (dsl_v1/v2/v3) against HF legacy/fast on W2/W3/W4 workloads; no hand-tuned kernels.
- `bench_profile.py`: cProfile breakdown for HF variants and per-stage wall-clock timing for DSL Numba kernels; outputs profiles/ files and a side-by-side comparison table.
- `full_benchmark.py`: Runs the full benchmark for n threads
- `full_memory_benchmark.py`: Runs the full memory benchmark for configurable thread counts in a separate clean memory-measurement process.

#### Infrastructure 

- `data.py`: Contains `load_images` function. All image loading code should be here.

- `measurement.py`: Timing and profiling harness using `cProfile` and `RSS`. All profiling code should be here.

- `models.py`: Loads the relevant model pre-processing code from huggingface. All huggingface model/pre-processing code fetching should be done here.

#### Testing and development
- `phase0.py`: Sanity-check benchmark for Qwen2.5-VL legacy vs fast preprocessing on W2.

- `phase1.py`: Adds cProfile and memory profiling for the Qwen2.5-VL W2 benchmark.

- `phase2.py`: Benchmarks InternVL2.5 HF and manual preprocessing to test whether the bottleneck generalizes.

- `phase3.py`: Benchmarks Qwen2.5-VL and InternVL2.5 preprocessing on mixed-size W3 and large-image W4 workloads.

- `full_benchmark_single_thread.py`: Runs the full benchmark results on a single thread

- `full_memory_benchmark_single_thread.py`: Runs the full memory benchmark results (there is a bug in full_benchmark.py that makes its memory results invalid; this is the file that re-runs memory by itself, correctly).

- `test_llava.py`: Quick LLaVA-NeXT W3 timing, memory, and output-shape test for legacy and fast processors.

- `test_models.py`: Verifies that `AutoImageProcessor` and `AutoProcessor.image_processor` return identical pixel values for each benchmarked model, and that InternVL3.5's `crop_to_patches=True` flag is load-bearing.

- `test_measurement.py`: Quick Qwen2.5-VL W3 test for the timing and RSS measurement helpers.


### `kernels/`
Hand-fused Qwen2.5-VL preprocessor kernels (Milestone 2).

- `bilinear.py`: Shared `@njit` bilinear sampling and resize primitives.

- `patch_coords.py`: Shared `@njit` patch addressing utilities (flat index and column offset).

- `qwen_v1_naive.py`: v1 naive correctness baseline — each stage (`smart_resize_dims`, `bilinear_resize`, `rescale`, `normalize`, `patchify`) is its own function with an intermediate buffer.

- `qwen_v2_fused.py`: v2 pointwise fusion — `rescale` and `normalize` inlined into the `bilinear_resize` pixel loop (`compute_at = inline`), eliminating two intermediate buffers.

- `qwen_v3_storage.py`: v3 full fusion with pre-allocated output — bilinear resize, rescale, normalize, and patchify fused into a single parallel loop that writes directly to a pre-allocated output tensor (`store_at = root`), eliminating all intermediate buffers.

### `visualizations/`
Contains Python scripts for creating visualizations from the output data, as well as the corresponding figures.

### `dsl/`
Core DSL: algorithm IR, schedule IR, op registry, coord registry, template factories, and Phase A codegen (Milestone 2).

- `algorithm.py`: `Func` and `Pipeline` dataclasses — the algorithm DAG IR.

- `schedule.py`: `StageSchedule` and `Schedule` dataclasses — per-stage `compute_at`/`store_at`/`parallel`/`write_via` annotations plus the pipeline-level `preallocate_output` flag.

- `ops.py`: Op registry (`resize`, `rescale`, `normalize`, `patchify`, `tile`, `center_crop`) and `validate(pipeline)` for op + param + input checks.

- `coords.py`: Registered output-addressing coord functions for `write_via`; Qwen ships `qwen_patch_coords` wrapping `kernels/patch_coords.py`.

- `templates.py`: Qwen `@njit` template factories (`make_template_naive/pointwise/full`) plus LLaVA factories (`make_template_llava_naive/pointwise/full`) and the shared `_enumerate_tiles` helper.

- `codegen.py`: Qwen codegen (`classify_fusion`, `build`) and LLaVA codegen (`classify_fusion_llava`, `build_llava`, `select_best_resolution`).

### `pipelines/`
Per-model DSL pipelines (algorithm DAG + named `Schedule`s).

- `qwen.py`: Qwen2.5-VL `pipeline` plus `sched_v1` / `sched_v2` / `sched_v3` — one algorithm, three schedules.

- `llava.py`: LLaVA-NeXT `pipeline` (tile → rescale → normalize) plus `sched_v1` / `sched_v2` / `sched_v3`; output is `(N_total_tiles, 3, 336, 336)` CHW.

### `tests/`

- `test_libs.py` tests that libaries load

- `test_correctness.py`: Verifies v1/v2/v3 output shape, `image_grid_thw`, and pixel values against HF fast.

- `test_dsl_correctness.py`: D3 spine gate — verifies `classify_fusion` matches v1/v2/v3, DSL-v1/v2/v3 reproduce the hand-fused Qwen kernels, DSL-v3 matches hand-v3 within 1e-7, and all three DSL schedules produce equivalent output.

- `test_dsl_llava_correctness.py`: D4 gate — verifies `classify_fusion_llava`, DSL LLaVA output shape, cross-schedule consistency, and pixel values vs HF bilinear within atol=0.05.

- `test_filter_comparison.py`: Informational (no gate) — prints a max_diff table for DSL-v1 (LLaVA) and Qwen v1 kernel against HF slow/fast processors with both BICUBIC and BILINEAR filters.

## Dependency note

The current lockfile uses `transformers==4.57.6` (`pyproject.toml` allows `>=4.49.0,<5.0.0`). This version supports selecting Hugging Face image processor implementations with `use_fast=False` for the legacy Python processor and `use_fast=True` for the fast processor, which is how the benchmark code compares the two preprocessing paths.
