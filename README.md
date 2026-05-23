# cs348k

Please go to `MILESTONE_1.md` for the write up and results for Milestone 1!


## Farmshare

`https://docs.farmshare.stanford.edu/`

`ssh jdarve@login.farmshare.stanford.edu`

`srun -c 8 --mem=16G --time=12:00:00 --pty bash`

## Quickstart

`cd cs348k`

`source $(poetry env info --path)/bin/activate`

`conda activate cs348k`

`git pull`

## Dependency notes

- The current lockfile uses `transformers==4.57.6` (`pyproject.toml` allows `>=4.49.0,<5.0.0`). This version supports selecting Hugging Face image processor implementations with `use_fast=False` for the legacy Python processor and `use_fast=True` for the fast processor, which is how the benchmark code compares the two preprocessing paths.

## Repository structure

### Root files

- `links.md`: Collects reference links for VLM papers, project pages, Hugging Face code, and model cards.

### `benchmarks/`
Benchmarks existing implementations (Milestone 1)

#### Core files
- `bench_kernels.py`: Benchmarks the Milestone 2 Qwen kernel implementations against HF legacy, HF fast, and HF bilinear baselines.

- `full_benchmark.py`: Runs the full benchmark for n threads

- `full_memory_benchmark.py`: Runs the full memory benchmark for configurable thread counts in a separate clean memory-measurement process.

#### Infrastructure 

- `data.py`: Contains `load_images` function. All iamge loading code should be here.

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

### `results/` 
Notes about intermediate results

- `bench_kernels_results.md`: Summarizes the Milestone 2 `bench_kernels.py` kernel timing and memory results.

- `bench_kernels_results_multi_thread.md`: Summarizes the Milestone 2 `bench_kernels.py --num-threads 8` timing and memory results without torch compile rows.

- `temp_bench_kernel_charts.md`: Temporary markdown tables comparing Milestone 2 kernel runtimes and peak/output memory ratios across W2, W3, and W4.

- `temp_torch_compile_charts.md`: Temporary markdown table comparing Qwen HF legacy and fast runtimes with and without `torch.compile`.

- `temp_full_benchmark_single_thread_charts.md`: Temporary markdown tables comparing single-thread full benchmark runtimes and peak/output memory ratios across W2, W3, and W4.

- `full_benchmarks_multi_thread_results.md`: Summarizes Milestone 1 timing and memory results for the multi-thread benchmark runs.

- `full_benchmarks_single_thread_results.md`: Summarizes Milestone 1 timing and memory results for the single-thread benchmark runs.

- `full_profiling_output_multi_thread.md`: Contains the detailed cProfile output from the multi-thread profiling runs.

- `full_profiling_output_single_thread.md`: Contains the detailed cProfile output from the single-thread profiling runs.

### `tests/`

- `test_libs.py` tests that libaries load

- `test_correctness.py`: Verifies v1 (and future v2/v3) output shape, `image_grid_thw`, and pixel values against HF fast.


## Project Proposal

Historically, image pre-processing for machine learning models has been simple, with models hardcoded to a set image size, which may only need resizing and normalization. However, modern vision language model (VLM) pipelines work in the native pixel resolution of the image, and may perform more sophisticated pre-processing, such as dynamic tiling, patching, and multiple types of padding or normalization. In most open source implementations for training VLMs such as Huggingface or PyTorch Transforms, these pipelines are not implemented efficiently, and convert back and forth between several image formats such as numpy, tensors, or Pillow/PIL for different transformations. I seek to develop a DSL inspired by Halide for high-performance VLM image preprocessing, separating algorithm from schedule and providing primitives such as tiling, fusion (compute_at), parallelism, and vectorization. The key metrics I will need to verify are that existing implementations are in fact inefficient, and that the image preprocessing is a significant bottleneck in the pipeline, both of which I suspect to be the case, especially for video or high quality image processing. Precisely, I want a bar chart showing runtimes and memory usage for existing implementations showing that they are slow, and that my method is significantly faster (2-5x). Furthermore, I want a separate chart comparing the entire image processing pipeline including the VLM and show that faster pre-processing increases the overall speed significantly.

