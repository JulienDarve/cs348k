# Milestone 2 AWS Batch DSL Chart

This table compares the attached AWS `c7i.4xlarge` batch DSL results against the three multi-thread Hugging Face Qwen implementations.

Sources:
- Hugging Face rows: `results/aws/bench_dsl_results_multi_thread.md`
- DSL rows: attached output from `benchmarks/bench_dsl.py --num-threads 8 --variants dsl_v1_batch dsl_v2_batch dsl_v3`

## Chart 1: Multi-Threaded Runtime

Median runtime in ms/img with `--num-threads 8` on AWS `c7i.4xlarge`. Best model per workload is bolded.

| Workload | hf_legacy | hf_fast | hf_bilinear | dsl_v1 | dsl_v2 | dsl_v3 |
|---|---:|---:|---:|---:|---:|---:|
| W2 | 55.704 | 18.231 | 17.998 | 16.816 | **15.006** | 17.259 |
| W3 | 21.571 | 5.613 | **5.514** | 7.762 | 7.343 | 8.550 |
| W4 | 691.746 | 164.099 | 163.135 | 170.663 | 145.637 | **143.626** |

## Chart 2: Peak Memory Allocation

Peak / Output RSS memory usage with `--num-threads 8` on AWS `c7i.4xlarge`. Best model per workload is bolded.

| Workload | hf_legacy | hf_fast | hf_bilinear | dsl_v1 | dsl_v2 | dsl_v3 |
|---|---:|---:|---:|---:|---:|---:|
| W2 | 2.10x | 3.89x | 3.75x | 1.40x | **1.10x** | **1.10x** |
| W3 | 1.99x | 2.28x | 2.32x | 1.07x | **1.00x** | **1.00x** |
| W4 | 2.02x | 3.86x | 3.84x | 1.78x | 1.89x | **1.13x** |
