# W2

| Model | Median ms/batch | Median ms/img | p95 - p50 | Output | Peak RSS | Peak / Output |
|---|---:|---:|---:|---:|---:|---:|
| hf_legacy | 1351.53 ms | 42.235 ms | 13.25 ms | 824.29 MB | 1752.06 MB | **2.13x** |
| hf_fast | 2561.22 ms | 80.038 ms | 13.23 ms | 824.29 MB | 3208.56 MB | **3.89x** |
| hf_bilinear | 2541.00 ms | 79.406 ms | 29.90 ms | 824.29 MB | 3197.13 MB | **3.88x** |
| v1 | 2649.42 ms | 82.794 ms | 73.95 ms | 824.29 MB | 1695.79 MB | **2.06x** |
| v2 | 1550.42 ms | 48.450 ms | 28.20 ms | 824.29 MB | 1682.03 MB | **2.04x** |

# W3

| Model | Median ms/batch | Median ms/img | p95 - p50 | Output | Peak RSS | Peak / Output |
|---|---:|---:|---:|---:|---:|---:|
| hf_legacy | 1326.76 ms | 41.461 ms | 26.97 ms | 333.42 MB | 735.30 MB | **2.21x** |
| hf_fast | 534.96 ms | 16.718 ms | 29.23 ms | 333.42 MB | 838.66 MB | **2.52x** |
| hf_bilinear | 540.12 ms | 16.879 ms | 14.92 ms | 333.42 MB | 789.28 MB | **2.37x** |
| v1 | 1093.04 ms | 34.157 ms | 8.59 ms | 333.42 MB | 633.60 MB | **1.90x** |
| v2 | 576.32 ms | 18.010 ms | 4.64 ms | 333.42 MB | 690.05 MB | **2.07x** |

# W4

| Model | Median ms/batch | Median ms/img | p95 - p50 | Output | Peak RSS | Peak / Output |
|---|---:|---:|---:|---:|---:|---:|
| hf_legacy | 17678.56 ms | 2209.820 ms | 70.39 ms | 1674.62 MB | 3340.19 MB | **1.99x** |
| hf_fast | 5359.54 ms | 669.942 ms | 12.51 ms | 1674.62 MB | 6462.42 MB | **3.86x** |
| hf_bilinear | 5378.14 ms | 672.267 ms | 20.79 ms | 1674.62 MB | 6435.04 MB | **3.84x** |
| v1 | 4749.66 ms | 593.707 ms | 20.57 ms | 1674.62 MB | 3662.29 MB | **2.19x** |
| v2 | 4305.82 ms | 538.227 ms | 19.45 ms | 1674.62 MB | 3453.80 MB | **2.06x** |
