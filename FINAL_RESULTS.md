# Final Results

These tables compare the attached AWS `c7i.4xlarge` batch DSL results against Hugging Face implementations for Qwen and LLaVA-NeXT.

Sources:
- Hugging Face rows: `results/aws/bench_dsl_results_multi_thread.md`
- DSL rows: attached output from `benchmarks/bench_dsl.py --num-threads 8 --variants dsl_v1 dsl_v2 dsl_v3`
- LLaVA rows: attached output from `benchmarks/bench_dsl_llava.py --num-threads 8` and `benchmarks/bench_dsl_llava.py --num-threads 1`

## Chart 1: Multi-Threaded Runtime

Median runtime in ms/img with `--num-threads 8` on AWS `c7i.4xlarge`. Best model per workload is bolded.

| Workload | hf_legacy | hf_fast | hf_bilinear | dsl_v1 | dsl_v2 | dsl_v3 |
|---|---:|---:|---:|---:|---:|---:|
| W2 | 55.704 | 18.231 | 17.998 | 15.914 | **15.726** | 17.280 |
| W3 | 21.571 | 5.613 | **5.514** | 7.756 | 7.333 | 8.525 |
| W4 | 691.746 | 164.099 | 163.135 | 166.091 | 150.702 | **143.880** |

## Chart 2: Peak Memory Allocation

Peak / Output RSS memory usage with `--num-threads 8` on AWS `c7i.4xlarge`. Best model per workload is bolded.

| Workload | hf_legacy | hf_fast | hf_bilinear | dsl_v1 | dsl_v2 | dsl_v3 |
|---|---:|---:|---:|---:|---:|---:|
| W2 | 2.10x | 3.89x | 3.75x | 1.40x | **1.10x** | **1.10x** |
| W3 | 1.99x | 2.28x | 2.32x | 1.07x | **1.00x** | **1.00x** |
| W4 | 2.02x | 3.86x | 3.84x | 1.88x | 1.83x | **1.13x** |

## Chart 3: LLaVA Multi-Threaded Runtime

Median runtime in ms/img with `--num-threads 8` on AWS `c7i.4xlarge`. Best model per workload is bolded.

| Workload | hf_legacy | hf_fast | dsl_v1 | dsl_v2 | dsl_v3 |
|---|---:|---:|---:|---:|---:|
| W2 | 37.085 | 6.603 | 6.596 | **6.536** | 6.612 |
| W3 | 23.631 | 4.314 | 3.686 | **3.602** | 3.646 |
| W4 | 156.707 | **28.836** | 44.341 | 44.293 | 44.528 |

## Chart 4: LLaVA Peak Memory Allocation

Peak / Output RSS memory usage with `--num-threads 8` on AWS `c7i.4xlarge`. Best model per workload is bolded.

| Workload | hf_legacy | hf_fast | dsl_v1 | dsl_v2 | dsl_v3 |
|---|---:|---:|---:|---:|---:|
| W2 | 3.44x | 4.05x | 1.06x | **1.00x** | **1.00x** |
| W3 | 1.84x | 1.97x | **1.00x** | **1.00x** | **1.00x** |
| W4 | 3.89x | 4.26x | **2.44x** | **2.44x** | **2.44x** |

## Chart 5: LLaVA Single-Threaded Runtime

Median runtime in ms/img with `--num-threads 1` on AWS `c7i.4xlarge`. Best model per workload is bolded.

| Workload | hf_legacy | hf_fast | dsl_v1 | dsl_v2 | dsl_v3 |
|---|---:|---:|---:|---:|---:|
| W2 | 37.941 | **15.019** | 44.001 | 43.165 | 43.971 |
| W3 | 23.254 | **12.006** | 22.405 | 21.771 | 22.033 |
| W4 | 157.016 | **36.490** | 252.443 | 251.881 | 251.332 |

## Chart 6: LLaVA Single-Threaded Peak Memory Allocation

Peak / Output RSS memory usage with `--num-threads 1` on AWS `c7i.4xlarge`. Best model per workload is bolded.

| Workload | hf_legacy | hf_fast | dsl_v1 | dsl_v2 | dsl_v3 |
|---|---:|---:|---:|---:|---:|
| W2 | 3.44x | 4.05x | **1.00x** | **1.00x** | **1.00x** |
| W3 | 1.66x | 1.77x | **1.00x** | **1.00x** | **1.00x** |
| W4 | 3.39x | 3.07x | **1.00x** | **1.00x** | **1.00x** |
