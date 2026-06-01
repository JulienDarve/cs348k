# AWS c7i.4xlarge Multi-Thread W2 DSL Kernel Results

Source: `benchmarks/bench_dsl.py --num-threads 8` timing output on AWS `c7i.4xlarge`.

Peak RSS columns are `N/A` because the captured output contains timing results only.

# W2

| Model | Median ms/batch | Median ms/img | p95 - p50 | Output | Peak RSS | Peak / Output |
|---|---:|---:|---:|---:|---:|---:|
| hf_legacy | 1801.69 ms | 56.303 ms | 18.04 ms | 824.29 MB | N/A | N/A |
| hf_fast | 591.37 ms | 18.480 ms | 4.54 ms | 824.29 MB | N/A | N/A |
| hf_bilinear | 585.26 ms | 18.289 ms | 5.28 ms | 824.29 MB | N/A | N/A |
| v1 | 2295.10 ms | 71.722 ms | 13.41 ms | 824.29 MB | N/A | N/A |
| v2 | 2035.14 ms | 63.598 ms | 3.47 ms | 824.29 MB | N/A | N/A |
| v3 | 310.57 ms | 9.705 ms | 6.77 ms | 824.29 MB | N/A | N/A |
| dsl_v1 | 2252.12 ms | 70.379 ms | 16.18 ms | 824.29 MB | N/A | N/A |
| dsl_v2 | 2021.22 ms | 63.163 ms | 28.58 ms | 824.29 MB | N/A | N/A |
| dsl_v3 | 297.76 ms | 9.305 ms | 11.29 ms | 824.29 MB | N/A | N/A |
