# Milestone 2 AWS Markdown Charts

These tables recreate Charts 1-3 and A1-A4 from `MILESTONE_2.md` using only the AWS data in `results/aws/`.

Sources:
- Charts 1-2: `results/aws/bench_dsl_results_single_thread.md`
- Chart 3: `results/aws/bench_dsl_results_multi_thread.md`
- Charts A1-A2: `results/aws/full_benchmarks_single_thread_results.md`
- Charts A3-A4: `results/aws/full_benchmarks_multi_thread_results.md`

## Chart 1: Runtimes per Workload

Median runtime in ms/img on AWS `c7i.4xlarge`. Best model per workload is bolded.

| Workload | hf_legacy | hf_fast | hf_bilinear | v1 | v2 | v3 | dsl_v1 | dsl_v2 | dsl_v3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| W2 | **53.460** | 68.077 | 68.331 | 70.333 | 63.339 | 67.593 | 71.615 | 64.323 | 65.269 |
| W3 | 21.412 | **18.613** | 18.694 | 28.524 | 25.150 | 26.479 | 28.625 | 25.618 | 25.581 |
| W4 | 692.839 | 581.309 | 590.936 | 699.150 | 553.890 | 556.225 | 673.589 | 553.496 | **539.227** |

## Chart 2: Peak Memory Allocation

Peak / Output RSS memory usage as a multiple of output bytes on AWS `c7i.4xlarge`. Best model per workload is bolded.

| Workload | hf_legacy | hf_fast | hf_bilinear | v1 | v2 | v3 | dsl_v1 | dsl_v2 | dsl_v3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| W2 | 2.10x | 3.89x | 3.76x | 1.94x | 1.94x | **1.00x** | 2.06x | 2.03x | **1.00x** |
| W3 | 2.24x | 2.58x | 2.16x | 1.88x | 1.75x | **1.00x** | 1.93x | 1.96x | **1.00x** |
| W4 | 2.03x | 3.87x | 3.89x | 2.22x | 2.06x | **1.09x** | 2.31x | 2.19x | 1.12x |

## Chart 3: Multi-Threaded Performance

Median runtime in ms/img with `--num-threads 8` on AWS `c7i.4xlarge`. Best model per workload is bolded.

| Workload | hf_legacy | hf_fast | hf_bilinear | v1 | v2 | v3 | dsl_v1 | dsl_v2 | dsl_v3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| W2 | 55.704 | 18.231 | 17.998 | 70.162 | 63.498 | 10.065 | 70.790 | 63.176 | **9.375** |
| W3 | 21.571 | 5.613 | 5.514 | 27.533 | 24.987 | 4.796 | 28.102 | 25.126 | **4.714** |
| W4 | 691.746 | 164.099 | 163.135 | 665.822 | 552.281 | 83.618 | 665.436 | 548.470 | **82.147** |

# Appendix

## Chart A1: Runtimes, Single-Threaded

Median runtime in ms/img on AWS `c7i.4xlarge`. Best model per workload is bolded.

| Workload | Qwen Fast | Qwen Legacy | InternVL2.5 Manual | InternVL3.5 Legacy | InternVL3.5 Fast | LLaVA Fast | LLaVA Legacy |
|---|---:|---:|---:|---:|---:|---:|---:|
| W2 | 67.771 | 53.244 | 59.161 | 74.557 | 54.783 | **14.354** | 32.521 |
| W3 | 17.268 | 20.445 | 32.853 | 42.289 | 28.731 | **11.014** | 20.858 |
| W4 | 581.849 | 706.358 | 126.102 | 171.054 | 119.103 | **35.303** | 146.301 |

## Chart A2: Peak Memory Allocation, Single-Threaded

Peak / Output RSS memory usage on AWS `c7i.4xlarge`. Best model per workload is bolded.

| Workload | Qwen Fast | Qwen Legacy | InternVL2.5 Manual | InternVL3.5 Legacy | InternVL3.5 Fast | LLaVA Fast | LLaVA Legacy |
|---|---:|---:|---:|---:|---:|---:|---:|
| W2 | 3.77x | 2.13x | 1.93x | **1.29x** | 2.78x | 3.47x | 2.88x |
| W3 | 1.53x | **1.24x** | 1.64x | 1.32x | 2.50x | 2.62x | 2.37x |
| W4 | 3.84x | 2.02x | 1.25x | **1.10x** | 3.93x | 5.01x | 4.24x |

## Chart A3: Runtimes, 8 Threads

Median runtime in ms/img with `--num-threads 8` on AWS `c7i.4xlarge`. Best model per workload is bolded.

| Workload | Qwen Fast | Qwen Legacy | InternVL2.5 Manual | InternVL3.5 Legacy | InternVL3.5 Fast | LLaVA Fast | LLaVA Legacy |
|---|---:|---:|---:|---:|---:|---:|---:|
| W2 | 18.412 | 56.339 | 34.968 | 78.669 | 21.874 | **5.976** | 34.330 |
| W3 | 5.826 | 20.911 | 19.314 | 43.583 | 11.465 | **4.087** | 22.173 |
| W4 | 170.267 | 674.272 | 111.605 | 169.985 | 83.751 | **28.285** | 142.528 |

## Chart A4: Peak Memory Allocation, 8 Threads

Peak / Output RSS memory usage with `--num-threads 8` on AWS `c7i.4xlarge`. Best model per workload is bolded.

| Workload | Qwen Fast | Qwen Legacy | InternVL2.5 Manual | InternVL3.5 Legacy | InternVL3.5 Fast | LLaVA Fast | LLaVA Legacy |
|---|---:|---:|---:|---:|---:|---:|---:|
| W2 | 3.77x | 2.13x | 1.93x | **1.89x** | 2.78x | 3.53x | 2.91x |
| W3 | 1.82x | 1.44x | 1.51x | **1.43x** | 2.50x | 1.74x | 1.69x |
| W4 | 3.75x | 2.02x | **1.00x** | 1.19x | 2.77x | 2.85x | 1.33x |
