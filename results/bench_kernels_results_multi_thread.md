# W2

| Model | Median ms/batch | Median ms/img | p95 - p50 | Output | Peak RSS | Peak / Output |
|---|---:|---:|---:|---:|---:|---:|
| hf_legacy | 2837.74 ms | 88.679 ms | 56.09 ms | 824.29 MB | 1719.98 MB | **2.09x** |
| hf_fast | 1371.26 ms | 42.852 ms | 16.04 ms | 824.29 MB | 3193.96 MB | **3.87x** |
| hf_bilinear | 1354.00 ms | 42.312 ms | 14.30 ms | 824.29 MB | 3095.49 MB | **3.76x** |
| v1 | 3955.77 ms | 123.618 ms | 22.37 ms | 824.29 MB | 1610.10 MB | **1.95x** |
| v2 | 3608.52 ms | 112.766 ms | 23.29 ms | 824.29 MB | 1594.85 MB | **1.93x** |
| v3 | 1351.30 ms | 42.228 ms | 19.12 ms | 824.29 MB | 824.33 MB | **1.00x** |

# W3

| Model | Median ms/batch | Median ms/img | p95 - p50 | Output | Peak RSS | Peak / Output |
|---|---:|---:|---:|---:|---:|---:|
| hf_legacy | 1204.83 ms | 37.651 ms | 21.07 ms | 333.42 MB | 638.33 MB | **1.91x** |
| hf_fast | 418.96 ms | 13.092 ms | 10.60 ms | 333.42 MB | 755.80 MB | **2.27x** |
| hf_bilinear | 405.31 ms | 12.666 ms | 4.01 ms | 333.42 MB | 758.05 MB | **2.27x** |
| v1 | 1560.94 ms | 48.779 ms | 8.38 ms | 333.42 MB | 629.47 MB | **1.89x** |
| v2 | 1429.86 ms | 44.683 ms | 6.15 ms | 333.42 MB | 565.66 MB | **1.70x** |
| v3 | 622.88 ms | 19.465 ms | 34.55 ms | 333.42 MB | 333.44 MB | **1.00x** |

# W4

| Model | Median ms/batch | Median ms/img | p95 - p50 | Output | Peak RSS | Peak / Output |
|---|---:|---:|---:|---:|---:|---:|
| hf_legacy | 7587.29 ms | 948.411 ms | 131.60 ms | 1674.62 MB | 3384.58 MB | **2.02x** |
| hf_fast | 2998.42 ms | 374.803 ms | 41.37 ms | 1674.62 MB | 6436.23 MB | **3.84x** |
| hf_bilinear | 2980.64 ms | 372.580 ms | 25.98 ms | 1674.62 MB | 6436.14 MB | **3.84x** |
| v1 | 9640.08 ms | 1205.010 ms | 87.19 ms | 1674.62 MB | 3663.10 MB | **2.19x** |
| v2 | 7880.95 ms | 985.119 ms | 132.67 ms | 1674.62 MB | 3453.26 MB | **2.06x** |
| v3 | 2830.90 ms | 353.863 ms | 21.53 ms | 1674.62 MB | 1805.01 MB | **1.08x** |
