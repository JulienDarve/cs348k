# cs348k

Please go to `MILESTONE_1.md` for the write up and results for Milestone 1!


## Farmshare

`https://docs.farmshare.stanford.edu/`

`ssh jdarve@login.farmshare.stanford.edu`

`srun -c 8 --mem=16G --pty bash`

## Quickstart

`cd cs348k`

`source $(poetry env info --path)/bin/activate`

`conda activate cs348k`

`git pull`

## Repository structure

### `benchmarks/`
Benchmarks existing implementations (Milestone 1)

- `data.py`: Contains `load_images` function. All iamge loading code should be here.

- `measurement.py`: Timing and profiling harness using `cProfile` and `RSS`. All profiling code should be here.

- `models.py`: Loads the relevant model pre-processing code from huggingface. All huggingface model/pre-processing code fetching should be done here.

- `phase[n].py`: Experiment and results from phase `n` (sanity checks, explained in MILESTONE_1.md)

- `full_benchmark_single_thread.py`: Runs the full benchmark results on a single thread

- `full_benchmark.py`: Runs the full benchmark for n threads

- `full_memory_benchmark_single_thread.py`: Runs the full memory benchmark results (there is a bug in full_benchmark.py that makes its memory results invalid; this is the file that re-runs memory by itself, correctly).


### `kernels/`
Hand-fused Qwen2.5-VL preprocessor kernels (Milestone 2).

- `bilinear.py`: Shared `@njit` bilinear sampling and resize primitives.

- `patch_coords.py`: Shared `@njit` patch addressing utilities (flat index and column offset).

- `qwen_v1_naive.py`: v1 naive correctness baseline — each stage (`smart_resize_dims`, `bilinear_resize`, `rescale`, `normalize`, `patchify`) is its own function with an intermediate buffer.

### `visualizations/`
Contains Jupyter notebooks for creating visualizations from the output data.

### `results/` 
Notes about intermediate results

### `tests/`

- `test_libs.py` tests that libaries load

- `test_correctness.py`: Verifies v1 (and future v2/v3) output shape, `image_grid_thw`, and pixel values against HF fast.


## Project Proposal

Historically, image pre-processing for machine learning models has been simple, with models hardcoded to a set image size, which may only need resizing and normalization. However, modern vision language model (VLM) pipelines work in the native pixel resolution of the image, and may perform more sophisticated pre-processing, such as dynamic tiling, patching, and multiple types of padding or normalization. In most open source implementations for training VLMs such as Huggingface or PyTorch Transforms, these pipelines are not implemented efficiently, and convert back and forth between several image formats such as numpy, tensors, or Pillow/PIL for different transformations. I seek to develop a DSL inspired by Halide for high-performance VLM image preprocessing, separating algorithm from schedule and providing primitives such as tiling, fusion (compute_at), parallelism, and vectorization. The key metrics I will need to verify are that existing implementations are in fact inefficient, and that the image preprocessing is a significant bottleneck in the pipeline, both of which I suspect to be the case, especially for video or high quality image processing. Precisely, I want a bar chart showing runtimes and memory usage for existing implementations showing that they are slow, and that my method is significantly faster (2-5x). Furthermore, I want a separate chart comparing the entire image processing pipeline including the VLM and show that faster pre-processing increases the overall speed significantly.

