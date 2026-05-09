# Milestone 1

Check README.md for an overview of the repo.

What are the questions or goals your project aims to answer?

What experiments should be done to answer that question, and how will you know from the outcome of the experiment that you have succeeded?

## Problem Overview

Historically, image pre-processing for machine learning models has been simple, with models hardcoded to a set image size, which may only need resizing and normalization. However, modern vision language model (VLM) pipelines work in the native pixel resolution of the image, and may perform more sophisticated pre-processing, such as dynamic tiling, patching, and multiple types of padding or normalization. For this milestone, I looked into three specific popular models with custom preprocessing: Qwen2.5-VL, InternVL-2.5, and LlaVA-Next. In their open source implementations, these pipelines are not implemented efficiently, and convert back and forth between several image formats such as numpy, tensors, or Pillow/PIL for different transformations. 





I seek to develop a DSL inspired by Halide for high-performance VLM image preprocessing, separating algorithm from schedule and providing primitives such as tiling, fusion (compute_at), parallelism, and vectorization. The key metrics I will need to verify are that existing implementations are in fact inefficient, and that the image preprocessing is a significant bottleneck in the pipeline, both of which I suspect to be the case, especially for video or high quality image processing. Precisely, I want a bar chart showing runtimes and memory usage for existing implementations showing that they are slow, and that my method is significantly faster (2-5x). Furthermore, I want a separate chart comparing the entire image processing pipeline including the VLM and show that faster pre-processing increases the overall speed significantly.


Discussion of existing implementations

Experiments to run:
- 

## Method

## Benchmarking

bar chart: Qwen Intern LlaVa legacy and fast vs runtime
- data: randomized 256,1048 images
- Hopefully also Multi-thread performance

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