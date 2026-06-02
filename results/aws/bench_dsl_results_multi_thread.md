# AWS c7i.4xlarge Multi-Thread DSL Kernel Results

Source: `benchmarks/bench_dsl.py --num-threads 8` on AWS `c7i.4xlarge`.

Memory was measured before timing in the same run, following the benchmark protocol.

# W2

| Model | Median ms/batch | Median ms/img | p95 - p50 | Output | Peak RSS | Peak / Output |
|---|---:|---:|---:|---:|---:|---:|
| hf_legacy | 1782.54 ms | 55.704 ms | 185.23 ms | 824.29 MB | 1730.14 MB | **2.10x** |
| hf_fast | 583.39 ms | 18.231 ms | 3.49 ms | 824.29 MB | 3210.23 MB | **3.89x** |
| hf_bilinear | 575.93 ms | 17.998 ms | 4.54 ms | 824.29 MB | 3095.00 MB | **3.75x** |
| v1 | 3731.65 ms | 116.614 ms | 10.93 ms | 824.29 MB | 1674.38 MB | **2.03x** |
| v2 | 3543.92 ms | 110.748 ms | 6.90 ms | 824.29 MB | 1674.22 MB | **2.03x** |
| v3 | 544.69 ms | 17.022 ms | 7.66 ms | 824.29 MB | 918.55 MB | **1.11x** |
| dsl_v1 | 3747.23 ms | 117.101 ms | 13.05 ms | 824.29 MB | 1794.36 MB | **2.18x** |
| dsl_v2 | 3523.97 ms | 110.124 ms | 11.84 ms | 824.29 MB | 1768.60 MB | **2.15x** |
| dsl_v3 | 525.22 ms | 16.413 ms | 4.81 ms | 824.29 MB | 918.54 MB | **1.11x** |

# W3

| Model | Median ms/batch | Median ms/img | p95 - p50 | Output | Peak RSS | Peak / Output |
|---|---:|---:|---:|---:|---:|---:|
| hf_legacy | 690.28 ms | 21.571 ms | 26.88 ms | 333.42 MB | 661.93 MB | **1.99x** |
| hf_fast | 179.62 ms | 5.613 ms | 3.13 ms | 333.42 MB | 758.71 MB | **2.28x** |
| hf_bilinear | 176.43 ms | 5.514 ms | 4.02 ms | 333.42 MB | 774.48 MB | **2.32x** |
| v1 | 1485.37 ms | 46.418 ms | 3.57 ms | 333.42 MB | 702.28 MB | **2.11x** |
| v2 | 1401.59 ms | 43.800 ms | 2.11 ms | 333.42 MB | 524.45 MB | **1.57x** |
| v3 | 284.22 ms | 8.882 ms | 2.63 ms | 333.42 MB | 333.43 MB | **1.00x** |
| dsl_v1 | 1496.42 ms | 46.763 ms | 8.20 ms | 333.42 MB | 582.97 MB | **1.75x** |
| dsl_v2 | 1401.19 ms | 43.787 ms | 5.25 ms | 333.42 MB | 553.35 MB | **1.66x** |
| dsl_v3 | 273.16 ms | 8.536 ms | 2.85 ms | 333.42 MB | 333.43 MB | **1.00x** |

# W4

| Model | Median ms/batch | Median ms/img | p95 - p50 | Output | Peak RSS | Peak / Output |
|---|---:|---:|---:|---:|---:|---:|
| hf_legacy | 5533.97 ms | 691.746 ms | 71.71 ms | 1674.62 MB | 3381.16 MB | **2.02x** |
| hf_fast | 1312.79 ms | 164.099 ms | 10.82 ms | 1674.62 MB | 6462.42 MB | **3.86x** |
| hf_bilinear | 1305.08 ms | 163.135 ms | 8.03 ms | 1674.62 MB | 6436.32 MB | **3.84x** |
| v1 | 8309.74 ms | 1038.717 ms | 13.95 ms | 1674.62 MB | 3688.66 MB | **2.20x** |
| v2 | 7492.61 ms | 936.576 ms | 11.46 ms | 1674.62 MB | 3453.44 MB | **2.06x** |
| v3 | 1140.02 ms | 142.503 ms | 13.23 ms | 1674.62 MB | 1726.83 MB | **1.03x** |
| dsl_v1 | 8568.81 ms | 1071.101 ms | 5.26 ms | 1674.62 MB | 3741.31 MB | **2.23x** |
| dsl_v2 | 7515.88 ms | 939.485 ms | 10.86 ms | 1674.62 MB | 3531.58 MB | **2.11x** |
| dsl_v3 | 1098.42 ms | 137.302 ms | 2.64 ms | 1674.62 MB | 1752.80 MB | **1.05x** |
