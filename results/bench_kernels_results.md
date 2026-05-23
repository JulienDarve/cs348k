# W2

| Model | Median ms/batch | Median ms/img | p95 - p50 | Output | Peak RSS | Peak / Output |
|---|---:|---:|---:|---:|---:|---:|
| hf_legacy | 3157.91 ms | 98.685 ms | 13.32 ms | 824.29 MB | 1719.96 MB | **2.09x** |
| hf_fast | 3853.10 ms | 120.409 ms | 8.71 ms | 824.29 MB | 3193.77 MB | **3.87x** |
| hf_legacy_compile | 2779.29 ms | 86.853 ms | 7.71 ms | 824.29 MB | 1659.44 MB | **2.01x** |
| hf_fast_compile | 8891.61 ms | 277.863 ms | 70.05 ms | 824.29 MB | 1751.07 MB | **2.12x** |
| hf_bilinear | 3851.43 ms | 120.357 ms | 76.96 ms | 824.29 MB | 3095.32 MB | **3.76x** |
| v1 | 4266.15 ms | 133.317 ms | 40.63 ms | 824.29 MB | 1597.23 MB | **1.94x** |
| v2 | 3647.11 ms | 113.972 ms | 32.09 ms | 824.29 MB | 1594.85 MB | **1.93x** |
| v3 | 1276.39 ms | 39.887 ms | 14.33 ms | 824.29 MB | 824.33 MB | **1.00x** |

# W3

| Model | Median ms/batch | Median ms/img | p95 - p50 | Output | Peak RSS | Peak / Output |
|---|---:|---:|---:|---:|---:|---:|
| hf_legacy | 1202.29 ms | 37.572 ms | 6.67 ms | 333.42 MB | 653.15 MB | **1.96x** |
| hf_fast | 1033.74 ms | 32.304 ms | 15.40 ms | 333.42 MB | 763.07 MB | **2.29x** |
| hf_legacy_compile | 1100.90 ms | 34.403 ms | 115.74 ms | 333.42 MB | 657.36 MB | **1.97x** |
| hf_fast_compile | 3210.50 ms | 100.328 ms | 7.91 ms | 333.42 MB | 655.36 MB | **1.97x** |
| hf_bilinear | 1027.99 ms | 32.125 ms | 34.81 ms | 333.42 MB | 775.69 MB | **2.33x** |
| v1 | 1500.92 ms | 46.904 ms | 1.96 ms | 333.42 MB | 620.67 MB | **1.86x** |
| v2 | 1255.54 ms | 39.235 ms | 2.27 ms | 333.42 MB | 584.89 MB | **1.75x** |
| v3 | 577.71 ms | 18.054 ms | 20.03 ms | 333.42 MB | 333.44 MB | **1.00x** |

# W4

| Model | Median ms/batch | Median ms/img | p95 - p50 | Output | Peak RSS | Peak / Output |
|---|---:|---:|---:|---:|---:|---:|
| hf_legacy | 6692.71 ms | 836.589 ms | 23.30 ms | 1674.62 MB | 3378.33 MB | **2.02x** |
| hf_fast | 8109.87 ms | 1013.734 ms | 105.39 ms | 1674.62 MB | 6466.65 MB | **3.86x** |
| hf_legacy_compile | 6935.08 ms | 866.885 ms | 13.32 ms | 1674.62 MB | 3374.14 MB | **2.01x** |
| hf_fast_compile | 17132.39 ms | 2141.548 ms | 20.68 ms | 1674.62 MB | 3791.89 MB | **2.26x** |
| hf_bilinear | 8095.66 ms | 1011.957 ms | 90.67 ms | 1674.62 MB | 6436.09 MB | **3.84x** |
| v1 | 9698.61 ms | 1212.327 ms | 247.99 ms | 1674.62 MB | 3663.17 MB | **2.19x** |
| v2 | 7979.68 ms | 997.460 ms | 363.06 ms | 1674.62 MB | 3453.54 MB | **2.06x** |
| v3 | 2676.61 ms | 334.576 ms | 18.74 ms | 1674.62 MB | 1778.90 MB | **1.06x** |
