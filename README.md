# cs348k

# Note for Milestone 1
Please go to `MILESTONE_1.md` for the write up and results for Milestone 1! Everything in the README.md is more notes for me and not fully updated.

## Activate poetry

`source $(poetry env info --path)/bin/activate`

## Farmshare
Links

`https://docs.farmshare.stanford.edu/`

`ssh jdarve@login.farmshare.stanford.edu`

`srun -c 8 --mem=16G --pty bash`

## Repository structure

`benchmarks/` folder:

- `data.py`: Contains `load_images` function. All iamge loading code should be here.

- `measurement.py`: Timing and profiling harness using `cProfile` and `RSS`. All profiling code should be here.

- `models.py`: Loads the relevant model pre-processing code from huggingface. All huggingface model/pre-processing code fetching should be done here.

- `phase[n]_results.md`: Results from phase `n`.

- `phase0.py`: Self-contained sanity check file. Verifies that image pre-processing is slow.

- `phase1.py`: Pulls together `data.py`, `measurement.py`, and `models.py` to profile Qwen2.5-VL-7B

- `phase2.py`: Same as phase1, Profiles InternVL-2.5


## Project Proposal

Historically, image pre-processing for machine learning models has been simple, with models hardcoded to a set image size, which may only need resizing and normalization. However, modern vision language model (VLM) pipelines work in the native pixel resolution of the image, and may perform more sophisticated pre-processing, such as dynamic tiling, patching, and multiple types of padding or normalization. In most open source implementations for training VLMs such as Huggingface or PyTorch Transforms, these pipelines are not implemented efficiently, and convert back and forth between several image formats such as numpy, tensors, or Pillow/PIL for different transformations. I seek to develop a DSL inspired by Halide for high-performance VLM image preprocessing, separating algorithm from schedule and providing primitives such as tiling, fusion (compute_at), parallelism, and vectorization. The key metrics I will need to verify are that existing implementations are in fact inefficient, and that the image preprocessing is a significant bottleneck in the pipeline, both of which I suspect to be the case, especially for video or high quality image processing. Precisely, I want a bar chart showing runtimes and memory usage for existing implementations showing that they are slow, and that my method is significantly faster (2-5x). Furthermore, I want a separate chart comparing the entire image processing pipeline including the VLM and show that faster pre-processing increases the overall speed significantly.

