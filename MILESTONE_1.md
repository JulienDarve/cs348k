# Milestone 1

What are the questions or goals your project aims to answer?

What experiments should be done to answer that question, and how will you know from the outcome of the experiment that you have succeeded?

## Problem Overview

Historically, image pre-processing for machine learning models has been simple, with models hardcoded to a set image size, which may only need resizing and normalization. However, modern vision language model (VLM) pipelines work in the native pixel resolution of the image, and may perform more sophisticated pre-processing, such as dynamic tiling, patching, and multiple types of padding or normalization. For this milestone, I looked into three specific popular models with custom preprocessing: Qwen2.5-VL, InternVL-2.5, and LlaVA-Next. 

In their open source implementations, these pipelines are not implemented efficiently: they convert back and forth between several image formats such as numpy, tensors, or Pillow/PIL for different transformations, and dump the entire image to memory between each transformation. Huggingface and PyTorch have worked towards improving these implementations. PyTorch released torch transforms v2 that provided optimized implementations of common transforms, and Huggingface implements it in their `fast` implementations. However, as the benchmarking results show, there is large room for improvement.

The goal of this project is to implement a scheduling DSL for efficient image pre-processing. The DSL will be based on Halide; I plan to seperate algorithm from schedule for the user. The core question of my project is to what level improved scheduling and a unified pipeline can improve image pre-processing speed.

## Experiments

###  Sanity Check, "Is this a real problem?"

I started with a multi-phase "sanity check" for the project, to "fail fast" and confirm existing pre-processing pipelines are in fact inefficient and that there is room for improvement. These are implemented as: `phase0.py` which just measures the runtime of Qwen pre-processing, `phase1.py` which expands to profile the memory allocations and peak runtime of each function within the code, and `phase2.py` and `phase3.py` that evaluate on a new model (Intern2.5VL) and simulate "real-life" workloads of variable sized and large images. We bring everything together in `full_benchmark_single_thread.py` and `full_memory_benchmark_single_thread.py`. All files for this milestone are in the `benchmarks/` folder.

### Is the problem widespread enough to justify a DSL?

We first want to establish that the inefficient preprocessing is a widespread to justify creating an entire DSL rather than a set of hardcoded optimized kernels. To address this, I evaluate on three different open source models: Qwen2.5-VL, InternVL-2.5, and LlaVA-Next using their recommended preprocessing pipelines.

### Is a *scheduling* DSL the right approach?

Next, we want to establish that the preprocessing stems from inefficent memory movement within the pipelines, not the computations themselves. This is because our scheduling approach will optimize memory movement, like in Halide where functions like compute_at natively support loop fusion. If the raw computation is the biggest slowdown for the implementation, changing the schedule of computations will make no difference. 

I performed the following experiments in order to answer this question. I analyzed peak memory allocation during the runtime to see if large amounts of wasteful buffers are generated during computation. I profiled the runtime of each function to see which functions dominate the runtime (if the dominating functions are reshape or transpose operations, that is a sign that scheduling is the right move). 

### How do the models perform on "real-life" tasks?

In order to keep my experiments representative of "real-life" pre-processing pipelines, I run a workload that investigated images with randomized dimensions, as real-life images have variable sizes. This is to differentiate from previous computer vision pipelines that had hardcoded image sizes, as modern VLMs take in any sizes. 

As a second point of comparison, I explored very large images in the same dimensions as A4, 2480 x 3508. These also helped to see if the preprocessing pipelines break down at large inputs.

## Benchmarking
We compare two workloads: W3, which is variable sized images, and W4, which are the large A4 images. We compare the three models, Qwen2.5-VL, InternVL-2.5, and LlaVA-Next. We further compare the huggingface fast vs slow implementations. Each run included 10 warmup steps, and 


### Runtime

The two plots below compare runtimes on the W3 and W4 workloads. 

![w3_runtime_legacy_vs_fast](visualizations\figures\w3_runtime_legacy_vs_fast.png)

![w4_runtime_legacy_vs_fast](visualizations\figures\w4_runtime_legacy_vs_fast.png)

### Peak Memory

The two plots below compare peak memory allocation on the W3 and W4 workloads. 

![w3_peak_rss_legacy_fast_manual](visualizations\figures\w3_peak_rss_legacy_fast_manual.png
)

![w4_peak_memory_legacy_fast_manual](visualizations\figures\w4_peak_rss_legacy_fast_manual.png
)

### Profiling
