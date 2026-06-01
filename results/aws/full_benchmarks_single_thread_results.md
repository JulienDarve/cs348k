# AWS c7i.4xlarge Single-Thread Results

Source: `benchmarks/full_memory_benchmark.py --num-threads 1` on AWS `c7i.4xlarge`.

Timing columns are from `benchmarks/full_benchmark.py`; memory columns are from `benchmarks/full_memory_benchmark.py --num-threads 1`.

# W2

| Model | Median ms/batch | Median ms/img | p95 - p50 | Output | Peak RSS | Peak / Output |
|---|---:|---:|---:|---:|---:|---:|
| Qwen Fast | 2168.68 ms | 67.771 ms | 11.65 ms | 824.29 MB | 3103.99 MB | **3.77x** |
| Qwen Legacy | 1703.80 ms | 53.244 ms | 310.55 ms | 824.29 MB | 1758.77 MB | **2.13x** |
| Intern2.5 Manual | 1893.14 ms | 59.161 ms | 36.00 ms | 770.70 MB | 1489.83 MB | **1.93x** |
| InternVL3.5 Legacy | 2385.81 ms | 74.557 ms | 28.12 ms | 770.70 MB | 995.25 MB | **1.29x** |
| InternVL3.5 Fast | 1753.05 ms | 54.783 ms | 4.70 ms | 770.70 MB | 2138.87 MB | **2.78x** |
| LLaVA Fast | 459.33 ms | 14.354 ms | 8.31 ms | 216.76 MB | 752.11 MB | **3.47x** |
| LLaVA Legacy | 1040.66 ms | 32.521 ms | 23.98 ms | 216.76 MB | 624.31 MB | **2.88x** |

# W3

| Model | Median ms/batch | Median ms/img | p95 - p50 | Output | Peak RSS | Peak / Output |
|---|---:|---:|---:|---:|---:|---:|
| Qwen Fast | 552.57 ms | 17.268 ms | 37.49 ms | 333.42 MB | 508.85 MB | **1.53x** |
| Qwen Legacy | 654.24 ms | 20.445 ms | 68.61 ms | 333.42 MB | 414.80 MB | **1.24x** |
| Intern2.5 Manual | 1051.29 ms | 32.853 ms | 50.11 ms | 472.06 MB | 773.66 MB | **1.64x** |
| InternVL3.5 Legacy | 1353.26 ms | 42.289 ms | 4.60 ms | 472.06 MB | 621.39 MB | **1.32x** |
| InternVL3.5 Fast | 919.39 ms | 28.731 ms | 4.80 ms | 472.06 MB | 1180.16 MB | **2.50x** |
| LLaVA Fast | 352.45 ms | 11.014 ms | 4.52 ms | 216.76 MB | 567.65 MB | **2.62x** |
| LLaVA Legacy | 667.44 ms | 20.858 ms | 5.51 ms | 216.76 MB | 514.32 MB | **2.37x** |

# W4

| Model | Median ms/batch | Median ms/img | p95 - p50 | Output | Peak RSS | Peak / Output |
|---|---:|---:|---:|---:|---:|---:|
| Qwen Fast | 4654.80 ms | 581.849 ms | 10.50 ms | 1674.62 MB | 6436.32 MB | **3.84x** |
| Qwen Legacy | 5650.86 ms | 706.358 ms | 166.78 ms | 1674.62 MB | 3383.13 MB | **2.02x** |
| Intern2.5 Manual | 1008.82 ms | 126.102 ms | 8.35 ms | 134.87 MB | 168.47 MB | **1.25x** |
| InternVL3.5 Legacy | 1368.43 ms | 171.054 ms | 2.31 ms | 134.87 MB | 148.98 MB | **1.10x** |
| InternVL3.5 Fast | 952.82 ms | 119.103 ms | 18.56 ms | 134.87 MB | 530.19 MB | **3.93x** |
| LLaVA Fast | 282.42 ms | 35.303 ms | 3.58 ms | 54.19 MB | 271.63 MB | **5.01x** |
| LLaVA Legacy | 1170.41 ms | 146.301 ms | 4.78 ms | 54.19 MB | 229.88 MB | **4.24x** |
