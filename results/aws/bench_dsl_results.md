# AWS c7i.4xlarge Single-Thread W2 DSL Kernel Results

Source: `benchmarks/bench_dsl.py` timing output on AWS `c7i.4xlarge`.

Peak RSS columns are `N/A` because the captured output contains timing results only.

# W2

| Model | Median ms/batch | Median ms/img | p95 - p50 | Output | Peak RSS | Peak / Output |
|---|---:|---:|---:|---:|---:|---:|
| hf_legacy | 1710.73 ms | 53.460 ms | 121.64 ms | 824.29 MB | N/A | N/A |
| hf_fast | 2178.46 ms | 68.077 ms | 7.92 ms | 824.29 MB | N/A | N/A |
| hf_bilinear | 2186.58 ms | 68.331 ms | 3.82 ms | 824.29 MB | N/A | N/A |
| v1 | 2250.66 ms | 70.333 ms | 7.81 ms | 824.29 MB | N/A | N/A |
| v2 | 2026.84 ms | 63.339 ms | 3.55 ms | 824.29 MB | N/A | N/A |
| v3 | 2162.96 ms | 67.593 ms | 16.15 ms | 824.29 MB | N/A | N/A |
| dsl_v1 | 2291.67 ms | 71.615 ms | 7.62 ms | 824.29 MB | N/A | N/A |
| dsl_v2 | 2058.33 ms | 64.323 ms | 38.95 ms | 824.29 MB | N/A | N/A |
| dsl_v3 | 2088.60 ms | 65.269 ms | 4.68 ms | 824.29 MB | N/A | N/A |
