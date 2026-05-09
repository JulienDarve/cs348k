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

Success for the sanity check is to verify that median runtime for Qwen is above 20 ms/image in pre-processing time, which starts to be a significant slowdown and bottleneck for inference on a fast GPU.

### Is the problem widespread enough to justify a DSL?

We first want to establish that the inefficient preprocessing is a widespread to justify creating an entire DSL rather than a set of hardcoded optimized kernels. To address this, I evaluate on three different open source models: Qwen2.5-VL, InternVL-2.5, and LlaVA-Next using their recommended preprocessing pipelines. These three models have very different pre-processing pipelines. Through their profile times, we observe they are inefficient in different ways, which justifies a DSL that supports many different, diverse pre-processing implementations.

Note: The AutoImage Processor Huggingface implementation of InternVL-2.5 does not seem to be a real implementation of the paper. I included the paper's recommended implementation in python as "manual" in the results. The implementation is hacky and non-optimizated, and it is not very performance. However, it is the implementation given by the authors, and the one countless engineers use out of the box without second thought, so I benchmarked it as well.

Success for this question would be that all three models are above 20 ms/image in pre-processing time, like Qwen from the sanity check. 

### Is a *scheduling* DSL the right approach?

Next, we want to establish that the preprocessing stems from inefficent memory movement within the pipelines, not the computations themselves. This is because our scheduling approach will optimize memory movement, like in Halide where functions like compute_at natively support loop fusion. If the raw computation is the biggest slowdown for the implementation, changing the schedule of computations will make no difference. 

I performed the following experiments in order to answer this question. I analyzed peak memory allocation during the runtime to see if large amounts of wasteful buffers are generated during computation. I profiled the runtime of each function to see which functions dominate the runtime (if the dominating functions are reshape or transpose operations, that is a sign that scheduling is the right move). 

Success for this question would be that >50% of the runtime is dominated by memory movement operations and the peak memory allocation is 3x the output. 

### How do the models perform on "real-life" tasks?

In order to keep my experiments representative of "real-life" pre-processing pipelines, I run a workload that investigated images with randomized dimensions, as real-life images have variable sizes. This is to differentiate from previous computer vision pipelines that had hardcoded image sizes, as modern VLMs take in any sizes. The variable sizes images have axes values that are randomly selected between 256 and 1048; they are re-balanced to make sure the smallest side is no smaller than 1/2 the larger side.

As a second point of comparison, I explored very large images in the same dimensions as A4, 2480 x 3508. These also helped to see if the preprocessing pipelines break down at large inputs.

## Benchmarking
We compare two workloads: W3, which is variable sized images, and W4, which are the large A4 images. The data is implemented in `data.py`. We compare the three models, Qwen2.5-VL, InternVL-2.5, and LlaVA-Next, which are implemented / loaded in `models.py`. We further compare the huggingface fast vs slow implementations. 


### Runtime

The two plots below compare runtimes on the W3 and W4 workloads. Each run included 10 warmup steps, and we computed the median runtime of 100 runs. To start, observe the y-axis measures the median runtime per batch in milliseconds; some of the W3 images can take up to multiple seconds on certain models! Across the board, the fast implementation is helping significantly (except for Qwen on W4). Qwen performed significantly worse on the large images compared to other models. 

![w3_runtime_legacy_vs_fast](visualizations\figures\w3_runtime_legacy_vs_fast.png)

![w4_runtime_legacy_vs_fast](visualizations\figures\w4_runtime_legacy_vs_fast.png)

### Peak Memory

The two plots below compare peak memory allocation on the W3 and W4 workloads. The peak memory allocation during processing was a significant multiple of the size of the image itself, which suggests significant room for improvement in memory allocation due to scheduling. Suprisingly, for a majority of models the fast implementation increased the amount of peak memory. Once again, Qwen performed significantly worse on the large images compared to other models. 

This was implemented using RSS, sampling the current memory allocation at rapid, regular intervals, implemented in `measurement.py`. I could not use a standard python method like `traceMalloc` as the tensor operations are done in C and were not being recorded. 

![w3_peak_rss_legacy_fast_manual](visualizations\figures\w3_peak_rss_legacy_fast_manual.png)

![w4_peak_memory_legacy_fast_manual](visualizations\figures\w4_peak_rss_legacy_fast_manual.png)

### Profiling

Finally, we profiled the time per function for each function in the pre-processing pipeline using `cProfile`, implemented in `measurement.py`. For readability, I grouped the different functions into 6 categories:
- **`Resize`:** Time spent resizing image pixels. This includes PIL resize calls like `PIL.Image.resize` / `ImagingCore.resize`, and `torch/torchvision` resize calls like `torch.nn.functional.interpolate` or `_upsample_bicubic2d_aa`. This is the "real computation" part of the pipeline, which scheduling cannot help much with.
- **`Patch / Tile Logic`**: Time spent deciding or constructing dynamic image regions: Qwen patchification, LLaVA image patch generation, or InternVL dynamic tiling, examples include `dynamic_preprocess` and `_get_image_patches`. This is another example of a "real computation" part of the pipeline.

- `Rescale / Normalize`: Elementwise operations that can be readily fused by an optimized schedule.
- `Format Conversion`: Time spent converting between torch, pil, or numpy, which can be mitigated with a unified library.
- `Stack / Cat / Reshape / Pad`: These operations represent combining intermediate tensors to create a final output, which can be readily optimized away using an optimized scheduler.
- `Other`: Everything else

The absolute time spent in each computation group is shown below (profiled over a single image). As we can see, in W3 a large percentage of the comptuation has room for optimization as they fall into the latter four categories rather than the first two. In W4, Qwen suffers the most from data movement operations, while the others are dominated by the resize operations.

![w3_profile_runtime_breakdown](visualizations\figures\w3_profile_runtime_breakdown.png)


![w4_profile_runtime_breakdown](visualizations\figures\w4_profile_runtime_breakdown.png)

## Conclusion

After the results of the benchmarks, we want to revisit our original questions. As we can see, VLM pre-processing takes up to tens to hundreds of milliseconds for even the fastest models, which poses a serious risk of bottlenecking an inference pipeline for a fast GPU on a small model - this is a real problem. We further see this is true for all three models, so the problem holds generally. 

From the memory peak allocation results, we can see that large intermediate tensors are allocated by the code (much larger than 3x the final output image size), which indicates a scheduling DSL may be a good fit. Furthermore, from the profiling results, for W3 images, the workload is dominated by memory movement and reshape operations that do not contribute directly to the core task of pre-processing; a scheduling DSL would be able to significantly reduce such operations. 

We observe for the large, uniform W4 images, Intern and LlaVA on the large W4 images, in fact spend a large fraction of the runtime spent on pure resizing. For these particular cases, a DSL may not be needed. However, this still supports the need for a scheduling abstraction: the best optimization strategy changes across model families and image regimes. The benchmarking we have completed helps us idenitify where a scheduling DSL would help the most. 