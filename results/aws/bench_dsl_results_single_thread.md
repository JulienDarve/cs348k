# AWS c7i.4xlarge Single-Thread DSL Kernel Results

Source: `benchmarks/bench_dsl.py` on AWS `c7i.4xlarge`.

Memory was measured before timing, following the benchmark protocol. W2 memory and timing were captured separately.

# W2

| Model | Median ms/batch | Median ms/img | p95 - p50 | Output | Peak RSS | Peak / Output |
|---|---:|---:|---:|---:|---:|---:|
| hf_legacy | 1710.73 ms | 53.460 ms | 121.64 ms | 824.29 MB | 1730.09 MB | **2.10x** |
| hf_fast | 2178.46 ms | 68.077 ms | 7.92 ms | 824.29 MB | 3209.95 MB | **3.89x** |
| hf_bilinear | 2186.58 ms | 68.331 ms | 3.82 ms | 824.29 MB | 3095.56 MB | **3.76x** |
| v1 | 2250.66 ms | 70.333 ms | 7.81 ms | 824.29 MB | 1597.22 MB | **1.94x** |
| v2 | 2026.84 ms | 63.339 ms | 3.55 ms | 824.29 MB | 1596.94 MB | **1.94x** |
| v3 | 2162.96 ms | 67.593 ms | 16.15 ms | 824.29 MB | 824.30 MB | **1.00x** |
| dsl_v1 | 2291.67 ms | 71.615 ms | 7.62 ms | 824.29 MB | 1697.89 MB | **2.06x** |
| dsl_v2 | 2058.33 ms | 64.323 ms | 38.95 ms | 824.29 MB | 1674.22 MB | **2.03x** |
| dsl_v3 | 2088.60 ms | 65.269 ms | 4.68 ms | 824.29 MB | 824.30 MB | **1.00x** |

# W3

| Model | Median ms/batch | Median ms/img | p95 - p50 | Output | Peak RSS | Peak / Output |
|---|---:|---:|---:|---:|---:|---:|
| hf_legacy | 685.19 ms | 21.412 ms | 2.92 ms | 333.42 MB | 747.84 MB | **2.24x** |
| hf_fast | 595.63 ms | 18.613 ms | 12.92 ms | 333.42 MB | 861.46 MB | **2.58x** |
| hf_bilinear | 598.22 ms | 18.694 ms | 14.48 ms | 333.42 MB | 721.18 MB | **2.16x** |
| v1 | 912.78 ms | 28.524 ms | 3.90 ms | 333.42 MB | 626.32 MB | **1.88x** |
| v2 | 804.79 ms | 25.150 ms | 2.31 ms | 333.42 MB | 582.96 MB | **1.75x** |
| v3 | 847.31 ms | 26.479 ms | 2.96 ms | 333.42 MB | 333.43 MB | **1.00x** |
| dsl_v1 | 916.00 ms | 28.625 ms | 4.53 ms | 333.42 MB | 643.90 MB | **1.93x** |
| dsl_v2 | 819.78 ms | 25.618 ms | 2.60 ms | 333.42 MB | 652.43 MB | **1.96x** |
| dsl_v3 | 818.59 ms | 25.581 ms | 4.39 ms | 333.42 MB | 333.43 MB | **1.00x** |

# W4

| Model | Median ms/batch | Median ms/img | p95 - p50 | Output | Peak RSS | Peak / Output |
|---|---:|---:|---:|---:|---:|---:|
| hf_legacy | 5542.71 ms | 692.839 ms | 39.79 ms | 1674.62 MB | 3407.04 MB | **2.03x** |
| hf_fast | 4650.48 ms | 581.309 ms | 37.17 ms | 1674.62 MB | 6488.52 MB | **3.87x** |
| hf_bilinear | 4727.49 ms | 590.936 ms | 5.29 ms | 1674.62 MB | 6514.45 MB | **3.89x** |
| v1 | 5593.20 ms | 699.150 ms | 7.58 ms | 1674.62 MB | 3715.12 MB | **2.22x** |
| v2 | 4431.12 ms | 553.890 ms | 6.81 ms | 1674.62 MB | 3453.84 MB | **2.06x** |
| v3 | 4449.80 ms | 556.225 ms | 59.34 ms | 1674.62 MB | 1831.23 MB | **1.09x** |
| dsl_v1 | 5388.71 ms | 673.589 ms | 30.79 ms | 1674.62 MB | 3871.48 MB | **2.31x** |
| dsl_v2 | 4427.97 ms | 553.496 ms | 95.48 ms | 1674.62 MB | 3662.35 MB | **2.19x** |
| dsl_v3 | 4313.81 ms | 539.227 ms | 3.55 ms | 1674.62 MB | 1883.30 MB | **1.12x** |
