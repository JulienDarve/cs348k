# Milestone 1

Check README.md for an overview of the repo.

What are the questions or goals your project aims to answer?

What experiments should be done to answer that question, and how will you know from the outcome of the experiment that you have succeeded?

## Problem Overview

Historically, image pre-processing for machine learning models has been simple, with models hardcoded to a set image size, which may only need resizing and normalization. However, modern vision language model (VLM) pipelines work in the native pixel resolution of the image, and may perform more sophisticated pre-processing, such as dynamic tiling, patching, and multiple types of padding or normalization. For this milestone, I looked into three specific popular models with custom preprocessing: Qwen2.5-VL, InternVL-2.5, and LlaVA-Next. 

In their open source implementations, these pipelines are not implemented efficiently: they convert back and forth between several image formats such as numpy, tensors, or Pillow/PIL for different transformations, and dump the entire image to memory between each transformation. Huggingface and PyTorch have worked towards improving these implementations. PyTorch released torch transforms v2 that provided optimized implementations of common transforms, and Huggingface implements it in their fast implementations. However, as the benchmarking results show, there is large room for improvement.

The goal of this project is to implement a DSL for efficient image pre-processing. The DSL will be based on Halide; I plan to seperate algorithm from schedule for the user. The core question of my project is to what level improved scheduling and a unified pipeline can improve image pre-processing speed.

## Experiments

I started with a multi-phase "sanity check" for the project, to "fail fast" and confirm existing pre-processing pipelines are in fact inefficient and that there is room for improvement. These are implemented as: `phase0.py` which just measures the runtime of Qwen pre-processing, `phase1.py` which expands to profile the memory allocations and peak runtime of each function within the code, and `phase2.py` and `phase3.py` that evaluate on a new model (Intern2.5VL) and simulate "real-life" workloads of variable sized and large images. We bring everything together in `full_benchmark_single_thread.py`. 

We first want to establish that the inefficient preprocessing is a widespread to justify creating an entire DSL rather than a set of hardcoded optimized kernels. To address this, I evaluate on three different open source models: Qwen2.5-VL, InternVL-2.5, and LlaVA-Next using their recommended preprocessing pipelines.

Next, we want to establish that the preprocessing stems from inefficent memory movement within the pipelines, not the computations themselves. This is because our scheduling approach will optimize memory movement, like in Halide where functions like compute_at natively support loop fusion. If the raw computation is the biggest slowdown for the implementation, 





## Benchmarking

bar chart: Qwen Intern LlaVa legacy and fast vs runtime
- data: randomized 256,1048 images
- Hopefully also Multi-thread performance

bar chart: same but W4

bar chart: same but memory allocations

## Profiling

Profiling analysis


3 models

Evaluated at different ratios

Legacy vs Fast

Runtimes

Multiple Threads (?)



## Key Questions


- `phase0.py`: Self-contained sanity check file. Verifies that image pre-processing is slow.

- `phase1.py`: Pulls together `data.py`, `measurement.py`, and `models.py` to profile Qwen2.5-VL-7B

- `phase2.py`: Same as phase1, Profiles InternVL-2.5

## TODOS

- make sure we can properly profile the pytorch code
    - Implement profiling, run isolated test, get feedback
- Implement LlaVa AnyRes


- writeup