# cs348k

Please go to `MILESTONE_2.md` for the write up and results for Milestone 2!

## Farmshare

`https://docs.farmshare.stanford.edu/`

`ssh jdarve@login.farmshare.stanford.edu`

`srun -c 8 --mem=16G --time=12:00:00 --pty bash`

## Quickstart

`cd cs348k`

`source $(poetry env info --path)/bin/activate`

`conda activate cs348k`

`git pull`

## Repository structure

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
Contains Python scripts for creating visualizations from the output data.

- `milestone_1_benchmark_charts.py`: Creates the Milestone 1 runtime and peak RSS benchmark charts.

- `milestone_1_profiling_charts.py`: Creates the Milestone 1 profiling breakdown and memory movement share charts.

- `milestone_2_ablation_speedups.py`: Creates a normalized Qwen kernel ablation speedup chart for Milestone 2.

- `milestone_2_runtime_memory_pareto.py`: Creates a runtime versus peak/output memory Pareto chart for the Milestone 2 Qwen kernels.

- `data.py`: Parses AWS benchmark result markdown tables and `FINAL_RESULTS.md`, then prepares derived plotting data for visualizations.

- `final_results_charts.py`: Creates final Qwen and LLaVA runtime, peak-memory, and runtime/memory Pareto charts from `FINAL_RESULTS.md`.

- `llava_w4_profiling_charts.py`: Creates focused LLaVA-NeXT W4 profiling charts showing DSL stage timing and the large-image sampling work that drives the slowdown.

- `final_results_gallery.md`: Links the final generated Qwen and LLaVA result figures.

- `final_slide_figures.py`: Creates the data-backed final presentation figures requested for slides 5 through 8 from `FINAL_RESULTS.md`.

- `final_slide_figures.md`: Links the generated data-backed final presentation figures for slides 5 through 8.

#### `visualizations/figures/`

- `final_qwen_multithread_runtime.png`: Shows 8-thread Qwen runtime per image for HF and DSL implementations.

- `final_llava_multithread_runtime.png`: Shows 8-thread LLaVA-NeXT runtime per image for HF and DSL implementations.

- `final_qwen_multithread_memory.png`: Shows 8-thread Qwen peak/output RSS memory for HF and DSL implementations.

- `final_llava_multithread_memory.png`: Shows 8-thread LLaVA-NeXT peak/output RSS memory for HF and DSL implementations.

- `final_qwen_singlethread_runtime.png`: Shows single-thread Qwen runtime per image for HF and DSL implementations.

- `final_llava_singlethread_runtime.png`: Shows single-thread LLaVA-NeXT runtime per image for HF and DSL implementations.

- `final_qwen_singlethread_memory.png`: Shows single-thread Qwen peak/output RSS memory for HF and DSL implementations.

- `final_llava_singlethread_memory.png`: Shows single-thread LLaVA-NeXT peak/output RSS memory for HF and DSL implementations.

- `final_runtime_memory_pareto.png`: Shows runtime versus peak/output RSS memory Pareto frontiers for Qwen and LLaVA-NeXT across thread settings and workloads.

- `final_llava_w4_profile_breakdown.png`: Compares the LLaVA-NeXT W4 HF Bilinear and DSL v1/v2/v3 profiling breakdowns, distinguishing HF resize from grouped DSL tile-processing calls and explaining the schedules' different fusion levels.

- `final_llava_w4_sampling_work.png`: Shows LLaVA-NeXT W4 output pixel composition and estimated thumbnail/grid source-sampling work.

- `slide_5_qwen_memory_floor.png`: Shows Qwen single-thread peak/output RSS memory for HF baselines and DSL v3 with the output-floor reference line.

- `slide_5_llava_singlethread_memory_floor.png`: Shows LLaVA-NeXT single-thread peak/output RSS memory for HF baselines and DSL v3 with the output-floor reference line.

- `slide_5_qwen_llava_memory_floor.png`: Shows Qwen and LLaVA-NeXT single-thread peak/output RSS memory floor charts side by side for the final slide deck.

- `slide_6_qwen_w4_thread_scaling.png`: Shows Qwen W4 runtime scaling for HF bilinear and DSL v3 across 1, 4, and 8 threads.

- `slide_7_qwen_schedule_axes.png`: Shows Qwen W4 fusion-axis memory/runtime behavior across DSL schedule variants.

- `slide_8_llava_schedule_flat_runtime.png`: Shows LLaVA-NeXT 8-thread runtime for HF bilinear and DSL schedules, plus the single-thread memory-floor summary.

### `results/` 
Notes about intermediate results

- `bench_kernels_results.md`: Summarizes the Milestone 2 `bench_kernels.py` kernel timing and memory results.

- `bench_kernels_results_multi_thread.md`: Summarizes the Milestone 2 `bench_kernels.py --num-threads 8` timing and memory results without torch compile rows.

- `full_benchmarks_multi_thread_results.md`: Summarizes Milestone 1 timing and memory results for the multi-thread benchmark runs.

- `full_benchmarks_single_thread_results.md`: Summarizes Milestone 1 timing and memory results for the single-thread benchmark runs.

- `full_profiling_output_multi_thread.md`: Contains the detailed cProfile output from the multi-thread profiling runs.

- `full_profiling_output_single_thread.md`: Contains the detailed cProfile output from the single-thread profiling runs.

#### `results/aws/`

- `bench_dsl_results.md`: Summarizes AWS `c7i.4xlarge` single-thread W2 timing results for Qwen HF baselines, hand-tuned kernels, and DSL kernel ablations.

- `bench_dsl_results_multi_thread.md`: Summarizes AWS `c7i.4xlarge` multi-thread W2/W3/W4 timing and memory results for Qwen HF baselines, hand-tuned kernels, and DSL kernel ablations.

- `full_benchmarks_multi_thread_results.md`: Summarizes AWS `c7i.4xlarge` multi-thread full benchmark timing and memory results for Qwen, InternVL, and LLaVA.

- `full_benchmarks_single_thread_results.md`: Summarizes AWS `c7i.4xlarge` single-thread full benchmark memory results from the clean memory-only benchmark run.

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


## Project Proposal

Historically, image pre-processing for machine learning models has been simple, with models hardcoded to a set image size, which may only need resizing and normalization. However, modern vision language model (VLM) pipelines work in the native pixel resolution of the image, and may perform more sophisticated pre-processing, such as dynamic tiling, patching, and multiple types of padding or normalization. In most open source implementations for training VLMs such as Huggingface or PyTorch Transforms, these pipelines are not implemented efficiently, and convert back and forth between several image formats such as numpy, tensors, or Pillow/PIL for different transformations. I seek to develop a DSL inspired by Halide for high-performance VLM image preprocessing, separating algorithm from schedule and providing primitives such as tiling, fusion (compute_at), parallelism, and vectorization. The key metrics I will need to verify are that existing implementations are in fact inefficient, and that the image preprocessing is a significant bottleneck in the pipeline, both of which I suspect to be the case, especially for video or high quality image processing. Precisely, I want a bar chart showing runtimes and memory usage for existing implementations showing that they are slow, and that my method is significantly faster (2-5x). Furthermore, I want a separate chart comparing the entire image processing pipeline including the VLM and show that faster pre-processing increases the overall speed significantly.

## Dependency notes

The current lockfile uses `transformers==4.57.6` (`pyproject.toml` allows `>=4.49.0,<5.0.0`). This version supports selecting Hugging Face image processor implementations with `use_fast=False` for the legacy Python processor and `use_fast=True` for the fast processor, which is how the benchmark code compares the two preprocessing paths.
