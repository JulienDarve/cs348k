# AWS c7i.4xlarge Multi-Thread DSL Kernel Results

Source: `benchmarks/bench_dsl.py --num-threads 8` on AWS `c7i.4xlarge`.

Memory was measured before timing in the same run, following the benchmark protocol.

# W2

| Model | Median ms/batch | Median ms/img | p95 - p50 | Output | Peak RSS | Peak / Output |
|---|---:|---:|---:|---:|---:|---:|
| hf_legacy | 1782.54 ms | 55.704 ms | 185.23 ms | 824.29 MB | 1730.14 MB | **2.10x** |
| hf_fast | 583.39 ms | 18.231 ms | 3.49 ms | 824.29 MB | 3210.23 MB | **3.89x** |
| hf_bilinear | 575.93 ms | 17.998 ms | 4.54 ms | 824.29 MB | 3095.00 MB | **3.75x** |
| v1 | 2245.19 ms | 70.162 ms | 8.75 ms | 824.29 MB | 1597.22 MB | **1.94x** |
| v2 | 2031.93 ms | 63.498 ms | 1.31 ms | 824.29 MB | 1596.95 MB | **1.94x** |
| v3 | 322.09 ms | 10.065 ms | 7.51 ms | 824.29 MB | 824.30 MB | **1.00x** |
| dsl_v1 | 2265.26 ms | 70.790 ms | 8.08 ms | 824.29 MB | 1697.89 MB | **2.06x** |
| dsl_v2 | 2021.64 ms | 63.176 ms | 7.54 ms | 824.29 MB | 1672.13 MB | **2.03x** |
| dsl_v3 | 300.00 ms | 9.375 ms | 10.65 ms | 824.29 MB | 824.30 MB | **1.00x** |

# W3

| Model | Median ms/batch | Median ms/img | p95 - p50 | Output | Peak RSS | Peak / Output |
|---|---:|---:|---:|---:|---:|---:|
| hf_legacy | 690.28 ms | 21.571 ms | 26.88 ms | 333.42 MB | 661.93 MB | **1.99x** |
| hf_fast | 179.62 ms | 5.613 ms | 3.13 ms | 333.42 MB | 758.71 MB | **2.28x** |
| hf_bilinear | 176.43 ms | 5.514 ms | 4.02 ms | 333.42 MB | 774.48 MB | **2.32x** |
| v1 | 881.04 ms | 27.533 ms | 5.58 ms | 333.42 MB | 627.11 MB | **1.88x** |
| v2 | 799.57 ms | 24.987 ms | 2.58 ms | 333.42 MB | 558.67 MB | **1.68x** |
| v3 | 153.47 ms | 4.796 ms | 1.58 ms | 333.42 MB | 333.43 MB | **1.00x** |
| dsl_v1 | 899.25 ms | 28.102 ms | 4.62 ms | 333.42 MB | 644.37 MB | **1.93x** |
| dsl_v2 | 804.03 ms | 25.126 ms | 2.11 ms | 333.42 MB | 575.74 MB | **1.73x** |
| dsl_v3 | 150.83 ms | 4.714 ms | 1.62 ms | 333.42 MB | 333.43 MB | **1.00x** |

# W4

| Model | Median ms/batch | Median ms/img | p95 - p50 | Output | Peak RSS | Peak / Output |
|---|---:|---:|---:|---:|---:|---:|
| hf_legacy | 5533.97 ms | 691.746 ms | 71.71 ms | 1674.62 MB | 3381.16 MB | **2.02x** |
| hf_fast | 1312.79 ms | 164.099 ms | 10.82 ms | 1674.62 MB | 6462.42 MB | **3.86x** |
| hf_bilinear | 1305.08 ms | 163.135 ms | 8.03 ms | 1674.62 MB | 6436.32 MB | **3.84x** |
| v1 | 5326.58 ms | 665.822 ms | 2.37 ms | 1674.62 MB | 3662.90 MB | **2.19x** |
| v2 | 4418.25 ms | 552.281 ms | 3.37 ms | 1674.62 MB | 3453.80 MB | **2.06x** |
| v3 | 668.94 ms | 83.618 ms | 172.76 ms | 1674.62 MB | 1805.00 MB | **1.08x** |
| dsl_v1 | 5323.49 ms | 665.436 ms | 3.53 ms | 1674.62 MB | 3819.63 MB | **2.28x** |
| dsl_v2 | 4387.76 ms | 548.470 ms | 6.57 ms | 1674.62 MB | 3610.27 MB | **2.16x** |
| dsl_v3 | 657.18 ms | 82.147 ms | 15.64 ms | 1674.62 MB | 1831.10 MB | **1.09x** |
