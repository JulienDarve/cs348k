# Milestone 2 Results

For milestone 2, I hand wrote kernels for Qwen2.5-VL preprocessing. I implemented the same algorithm with three different schedules, and found significant improvements in performance based on schedule choice. From these results, we can inform the set of operations that are most important for designing a DSL. In addition, I polished some elements of the benchmarking harness with a new workload and exploring torch compile.

# Hand-Crafted Qwen Preprocessing Numba Kernels

## Different Schedule, Same Algorithm

I tested three seperate kernels, each implementing the Qwen2.5-VL preprocessing algorithm, but with different schedules.

The first version (v1), in `kernels/qwen_v1_naive.py` is a naive implementation of the kernel. We apply each operation in order:   bilinear_resize  →  rescale  →  normalize  →  patchify. In between each operation, we dump the entire output to a buffer in memory, to input to the next operation.

The second version (v2), in `kernels/qwen_v2_fused.py` optimizes the pipeline by fusing pointwise operations. The rescale and normalize operations are pointwise, so we can perform them at the same time as we compute the bilinear resize. The new set of operations are _resize_normalize  →  patchify. The performance of the pointwise fusion explored the effects of a `compute_at` Halide-like primitive.
 
The third version (v3), in `kernels/qwen_v3_storage.py` further optimizes with full fusion of the operations. The output memory is preallocated, and the function will fill each entry directly without intermediate buffers. Full fusion explores a possible `store_at` primitive as well as a `preallocate_output` primitive. v3 was also tested with parallelism over multiple threads.

### Testing and Caveats

After each implementation version, I verified the correctness of the algorithm in the code in `tests/test_correctness.py`. Given the same randomized input image, I computed the difference in the output between algorithms, and verified they were identical up to floating point errors (1e-7). 

A key note about the implementation: Qwen uses a bicubic kernel with anti-aliasing for its resize. I decided that this would be too complicated to implement for this milestone. Instead, I implemented a bilinear kernel without anti-aliasing, which is much simpler but also faster to run. Using a smooth image input without significant variations that is not sigificantly affected by the kernel for resizing, I verified that my code returns the same output up to 1e-7 error. I also benchmarked the Huggingface implementation of Qwen using a bilinear resizing kernel instead of bicubic, which mitigated the difference in computation but was not perfect. Thus, take the performance differences with a grain of salt.

## Results

We benchmarked the runtime and peak memory allocation of the three versions of the Qwen kernel, against the huggingface implementation. We compared the legacy and fast implementation of huggingface, as well as 

### Chart 1: Runtimes per Workload
Median runtime in ms/img. Best model per workload is bolded.

| Workload | hf_legacy | hf_fast | hf_bilinear | v1 | v2 | v3 |
|---|---:|---:|---:|---:|---:|---:|
| W2 | 98.685 | 120.409 | 120.357 | 133.317 | 113.972 | **39.887** |
| W3 | 37.572 | 32.304 | 32.125 | 46.904 | 39.235 | **18.054** |
| W4 | 836.589 | 1013.734 | 1011.957 | 1212.327 | 997.460 | **334.576** |

### Chart 2: Peak Memory Allocation
Peak / Output RSS memory usage. Best model per workload is bolded.

| Workload | hf_legacy | hf_fast | hf_bilinear | v1 | v2 | v3 |
|---|---:|---:|---:|---:|---:|---:|
| W2 | 2.09x | 3.87x | 3.76x | 1.94x | 1.93x | **1.00x** |
| W3 | 1.96x | 2.29x | 2.33x | 1.86x | 1.75x | **1.00x** |
| W4 | 2.02x | 3.86x | 3.84x | 2.19x | 2.06x | **1.06x** |

### Chart 3: Multi-threaded performance

Median runtime in ms/img with `--num-threads 8`. Best model per workload is bolded.

| Workload | hf_legacy | hf_fast | hf_bilinear | v1 | v2 | v3 |
|---|---:|---:|---:|---:|---:|---:|
| W2 | 88.679 | 42.852 | 42.312 | 123.618 | 112.766 | **42.228** |
| W3 | 37.651 | 13.092 | **12.666** | 48.779 | 44.683 | 19.465 |
| W4 | 948.411 | 374.803 | 372.580 | 1205.010 | 985.119 | **353.863** |

Peak / Output RSS memory usage with `--num-threads 8`. Best model per workload is bolded.

| Workload | hf_legacy | hf_fast | hf_bilinear | v1 | v2 | v3 |
|---|---:|---:|---:|---:|---:|---:|
| W2 | 2.09x | 3.87x | 3.76x | 1.95x | 1.93x | **1.00x** |
| W3 | 1.91x | 2.27x | 2.27x | 1.89x | 1.70x | **1.00x** |
| W4 | 2.02x | 3.84x | 3.84x | 2.19x | 2.06x | **1.08x** |

### Chart 4: Torch Compile Runtimes

Median runtime in ms/img. Best model per workload is bolded.

| Workload | hf_legacy | hf_legacy_compile | hf_fast | hf_fast_compile | hf_bilinear |
|---|---:|---:|---:|---:|---:|
| W2 | 98.685 | **86.853** | 120.409 | 277.863 | 120.357 |
| W3 | 37.572 | 34.403 | 32.304 | 100.328 | **32.125** |
| W4 | **836.589** | 866.885 | 1013.734 | 2141.548 | 1011.957 |


## Insights for DSL


TODO:

- torch compile results
- profiling results
- multi-thread results


## Improvements to Benchmarking Harness since Milestone 1

- Added W2 
- Included Intern3.5-VL
- Multi-thread results (?)
- Torch compile

#### Chart 1: Runtimes

Median runtime in ms/img. Best model per workload is bolded.

| Workload | Qwen Fast | Qwen Legacy | Intern2.5 Manual | InternVL3.5 Legacy | InternVL3.5 Fast | LLaVA Fast | LLaVA Legacy |
|---|---:|---:|---:|---:|---:|---:|---:|
| W2 | 114.956 | 81.322 | 95.687 | 135.192 | 106.103 | **24.071** | 50.830 |
| W3 | 42.037 | 53.807 | 103.369 | 75.622 | 55.974 | **25.549** | 51.560 |
| W4 | 1458.948 | 1402.200 | 314.543 | 277.693 | 224.260 | **71.358** | 311.825 |

#### Chart 2: Peak Memory Allocation

Peak / Output RSS memory usage. Best model per workload is bolded.

| Workload | Qwen Fast | Qwen Legacy | Intern2.5 Manual | InternVL3.5 Legacy | InternVL3.5 Fast | LLaVA Fast | LLaVA Legacy |
|---|---:|---:|---:|---:|---:|---:|---:|
| W2 | 3.77x | 2.13x | **1.89x** | 2.03x | 2.77x | 2.53x | 3.00x |
| W3 | 2.28x | 1.91x | 1.85x | **1.83x** | 2.73x | 2.77x | 2.65x |
| W4 | 3.83x | 2.02x | **1.36x** | 1.55x | 4.07x | 6.10x | 5.65x |


## Conclusion
