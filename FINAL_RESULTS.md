# Final Results

These tables compare the attached AWS `c7i.4xlarge` batch DSL results against Hugging Face implementations for Qwen and LLaVA-NeXT.

Sources:
- Hugging Face rows: `results/aws/bench_dsl_results_multi_thread.md`
- DSL rows: attached output from `benchmarks/bench_dsl.py --num-threads 8 --variants dsl_v1 dsl_v2 dsl_v3`
- Qwen single-thread rows: attached output from `benchmarks/bench_dsl.py --variants dsl_v1 dsl_v2 dsl_v3 hf_legacy hf_fast hf_bilinear --num-threads 1`
- LLaVA rows: attached output from `benchmarks/bench_dsl_llava.py --num-threads 8`, `benchmarks/bench_dsl_llava.py --variants hf_bilinear --num-threads 8`, `benchmarks/bench_dsl_llava.py --num-threads 1`, and `benchmarks/bench_dsl_llava.py --variants hf_bilinear --num-threads 1`
- Four-thread rows: attached output from `benchmarks/bench_dsl.py --variants dsl_v1 dsl_v2 dsl_v3 hf_legacy hf_fast hf_bilinear --num-threads 4` and `benchmarks/bench_dsl_llava.py --variants dsl_v1 dsl_v2 dsl_v3 hf_legacy hf_fast hf_bilinear --num-threads 4`

## Qwen Results

### Chart 1: Multi-Threaded Runtime

Median runtime in ms/img with `--num-threads 8` on AWS `c7i.4xlarge`. Best model per workload is bolded.

| Workload | hf_legacy | hf_fast | hf_bilinear | dsl_v1 | dsl_v2 | dsl_v3 |
|---|---:|---:|---:|---:|---:|---:|
| W2 | 55.704 | 18.231 | 17.998 | 15.914 | **15.726** | 17.280 |
| W3 | 21.571 | 5.613 | **5.514** | 7.756 | 7.333 | 8.525 |
| W4 | 691.746 | 164.099 | 163.135 | 166.091 | 150.702 | **143.880** |

### Chart 2: Peak Memory Allocation

Peak / Output RSS memory usage with `--num-threads 8` on AWS `c7i.4xlarge`. Best model per workload is bolded.

| Workload | hf_legacy | hf_fast | hf_bilinear | dsl_v1 | dsl_v2 | dsl_v3 |
|---|---:|---:|---:|---:|---:|---:|
| W2 | 2.10x | 3.89x | 3.75x | 1.40x | **1.10x** | **1.10x** |
| W3 | 1.99x | 2.28x | 2.32x | 1.07x | **1.00x** | **1.00x** |
| W4 | 2.02x | 3.86x | 3.84x | 1.88x | 1.83x | **1.13x** |

### Chart 9: Qwen Four-Threaded Runtime

Median runtime in ms/img with `--num-threads 4` on AWS `c7i.4xlarge`. Best model per workload is bolded.

| Workload | hf_legacy | hf_fast | hf_bilinear | dsl_v1 | dsl_v2 | dsl_v3 |
|---|---:|---:|---:|---:|---:|---:|
| W2 | 63.949 | 27.901 | 27.600 | 28.766 | **26.693** | 31.301 |
| W3 | 23.438 | 7.797 | **7.671** | 12.324 | 11.618 | 13.633 |
| W4 | 725.749 | 244.217 | **241.679** | 289.493 | 256.362 | 265.060 |

### Chart 10: Qwen Four-Threaded Peak Memory Allocation

Peak / Output RSS memory usage with `--num-threads 4` on AWS `c7i.4xlarge`. Best model per workload is bolded.

| Workload | hf_legacy | hf_fast | hf_bilinear | dsl_v1 | dsl_v2 | dsl_v3 |
|---|---:|---:|---:|---:|---:|---:|
| W2 | 2.08x | 3.88x | 3.76x | 1.14x | **1.00x** | **1.00x** |
| W3 | 1.91x | 2.29x | 2.30x | **1.00x** | **1.00x** | **1.00x** |
| W4 | 2.00x | 3.84x | 3.84x | 1.50x | 1.46x | **1.08x** |

### Chart 3: Single-Threaded Runtime

Median runtime in ms/img with `--num-threads 1` on AWS `c7i.4xlarge`. Best model per workload is bolded.

| Workload | hf_legacy | hf_fast | hf_bilinear | dsl_v1 | dsl_v2 | dsl_v3 |
|---|---:|---:|---:|---:|---:|---:|
| W2 | **64.076** | 74.576 | 74.271 | 108.374 | 102.418 | 121.225 |
| W3 | 23.522 | **19.815** | 19.871 | 43.902 | 41.215 | 48.748 |
| W4 | 729.148 | 640.630 | **637.793** | 1070.222 | 953.593 | 994.205 |

### Chart 4: Single-Threaded Peak Memory Allocation

Peak / Output RSS memory usage with `--num-threads 1` on AWS `c7i.4xlarge`. Best model per workload is bolded.

| Workload | hf_legacy | hf_fast | hf_bilinear | dsl_v1 | dsl_v2 | dsl_v3 |
|---|---:|---:|---:|---:|---:|---:|
| W2 | 2.08x | 3.88x | 3.76x | **1.00x** | **1.00x** | **1.00x** |
| W3 | 1.91x | 2.30x | 2.31x | **1.00x** | **1.00x** | **1.00x** |
| W4 | 2.02x | 3.84x | 3.84x | 1.20x | 1.20x | **1.08x** |

## LLaVA Results

### Chart 5: LLaVA Multi-Threaded Runtime

Median runtime in ms/img with `--num-threads 8` on AWS `c7i.4xlarge`. Best model per workload is bolded.

| Workload | hf_legacy | hf_fast | hf_bilinear | dsl_v1 | dsl_v2 | dsl_v3 |
|---|---:|---:|---:|---:|---:|---:|
| W2 | 37.085 | 6.603 | **6.356** | 6.596 | 6.536 | 6.612 |
| W3 | 23.631 | 4.314 | 4.195 | 3.686 | **3.602** | 3.646 |
| W4 | 156.707 | 28.836 | **26.220** | 44.341 | 44.293 | 44.528 |

### Chart 6: LLaVA Peak Memory Allocation

Peak / Output RSS memory usage with `--num-threads 8` on AWS `c7i.4xlarge`. Best model per workload is bolded.

| Workload | hf_legacy | hf_fast | hf_bilinear | dsl_v1 | dsl_v2 | dsl_v3 |
|---|---:|---:|---:|---:|---:|---:|
| W2 | 3.44x | 4.05x | 3.98x | 1.06x | **1.00x** | **1.00x** |
| W3 | 1.84x | 1.97x | 2.49x | **1.00x** | **1.00x** | **1.00x** |
| W4 | 3.89x | 4.26x | 3.92x | **2.44x** | **2.44x** | **2.44x** |

### Chart 11: LLaVA Four-Threaded Runtime

Median runtime in ms/img with `--num-threads 4` on AWS `c7i.4xlarge`. Best model per workload is bolded.

| Workload | hf_legacy | hf_fast | hf_bilinear | dsl_v1 | dsl_v2 | dsl_v3 |
|---|---:|---:|---:|---:|---:|---:|
| W2 | 39.626 | 7.826 | **7.362** | 11.908 | 11.805 | 11.896 |
| W3 | 23.132 | 5.408 | **5.214** | 6.640 | 6.531 | 6.555 |
| W4 | 157.478 | 33.903 | **29.540** | 70.745 | 70.626 | 70.488 |

### Chart 12: LLaVA Four-Threaded Peak Memory Allocation

Peak / Output RSS memory usage with `--num-threads 4` on AWS `c7i.4xlarge`. Best model per workload is bolded.

| Workload | hf_legacy | hf_fast | hf_bilinear | dsl_v1 | dsl_v2 | dsl_v3 |
|---|---:|---:|---:|---:|---:|---:|
| W2 | 3.44x | 4.04x | 2.66x | **1.00x** | **1.00x** | **1.00x** |
| W3 | 2.59x | 2.74x | 2.37x | **1.00x** | **1.00x** | **1.00x** |
| W4 | 2.10x | 2.49x | 1.85x | **1.00x** | **1.00x** | **1.00x** |

### Chart 7: LLaVA Single-Threaded Runtime

Median runtime in ms/img with `--num-threads 1` on AWS `c7i.4xlarge`. Best model per workload is bolded.

| Workload | hf_legacy | hf_fast | hf_bilinear | dsl_v1 | dsl_v2 | dsl_v3 |
|---|---:|---:|---:|---:|---:|---:|
| W2 | 37.941 | 15.019 | **14.663** | 44.001 | 43.165 | 43.971 |
| W3 | 23.254 | 12.006 | **11.946** | 22.405 | 21.771 | 22.033 |
| W4 | 157.016 | 36.490 | **33.912** | 252.443 | 251.881 | 251.332 |

### Chart 8: LLaVA Single-Threaded Peak Memory Allocation

Peak / Output RSS memory usage with `--num-threads 1` on AWS `c7i.4xlarge`. Best model per workload is bolded.

| Workload | hf_legacy | hf_fast | hf_bilinear | dsl_v1 | dsl_v2 | dsl_v3 |
|---|---:|---:|---:|---:|---:|---:|
| W2 | 3.44x | 4.05x | 3.98x | **1.00x** | **1.00x** | **1.00x** |
| W3 | 1.66x | 1.77x | 1.75x | **1.00x** | **1.00x** | **1.00x** |
| W4 | 3.39x | 3.07x | 1.83x | **1.00x** | **1.00x** | **1.00x** |
