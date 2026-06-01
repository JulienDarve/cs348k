# AWS c7i.4xlarge Multi-Thread Results

Source: AWS `c7i.4xlarge` multi-thread runs.

Timing columns are from `benchmarks/full_benchmark.py --num-threads 8`; memory columns are from `benchmarks/full_memory_benchmark.py --num-threads 8`.

# W2

| Model | Median ms/batch | Median ms/img | p95 - p50 | Output | Peak RSS | Peak / Output |
|---|---:|---:|---:|---:|---:|---:|
| Qwen Fast | 589.17 ms | 18.412 ms | 4.17 ms | 824.29 MB | 3104.17 MB | **3.77x** |
| Qwen Legacy | 1802.84 ms | 56.339 ms | 75.14 ms | 824.29 MB | 1758.75 MB | **2.13x** |
| Intern2.5 Manual | 1118.97 ms | 34.968 ms | 23.52 ms | 770.70 MB | 1489.69 MB | **1.93x** |
| InternVL3.5 Legacy | 2517.42 ms | 78.669 ms | 47.58 ms | 770.70 MB | 1452.86 MB | **1.89x** |
| InternVL3.5 Fast | 699.96 ms | 21.874 ms | 3.78 ms | 770.70 MB | 2138.87 MB | **2.78x** |
| LLaVA Fast | 191.23 ms | 5.976 ms | 14.71 ms | 216.76 MB | 765.66 MB | **3.53x** |
| LLaVA Legacy | 1098.55 ms | 34.330 ms | 26.75 ms | 216.76 MB | 631.09 MB | **2.91x** |

# W3

| Model | Median ms/batch | Median ms/img | p95 - p50 | Output | Peak RSS | Peak / Output |
|---|---:|---:|---:|---:|---:|---:|
| Qwen Fast | 186.44 ms | 5.826 ms | 2.28 ms | 333.42 MB | 607.82 MB | **1.82x** |
| Qwen Legacy | 669.14 ms | 20.911 ms | 49.58 ms | 333.42 MB | 481.66 MB | **1.44x** |
| Intern2.5 Manual | 618.06 ms | 19.314 ms | 16.31 ms | 472.06 MB | 712.51 MB | **1.51x** |
| InternVL3.5 Legacy | 1394.66 ms | 43.583 ms | 7.35 ms | 472.06 MB | 674.24 MB | **1.43x** |
| InternVL3.5 Fast | 366.88 ms | 11.465 ms | 7.36 ms | 472.06 MB | 1180.16 MB | **2.50x** |
| LLaVA Fast | 130.77 ms | 4.087 ms | 6.19 ms | 216.76 MB | 376.64 MB | **1.74x** |
| LLaVA Legacy | 709.53 ms | 22.173 ms | 2.22 ms | 216.76 MB | 367.32 MB | **1.69x** |

# W4

| Model | Median ms/batch | Median ms/img | p95 - p50 | Output | Peak RSS | Peak / Output |
|---|---:|---:|---:|---:|---:|---:|
| Qwen Fast | 1362.14 ms | 170.267 ms | 14.56 ms | 1674.62 MB | 6285.50 MB | **3.75x** |
| Qwen Legacy | 5394.18 ms | 674.272 ms | 80.67 ms | 1674.62 MB | 3383.67 MB | **2.02x** |
| Intern2.5 Manual | 892.84 ms | 111.605 ms | 3.57 ms | 134.87 MB | 134.88 MB | **1.00x** |
| InternVL3.5 Legacy | 1359.88 ms | 169.985 ms | 11.92 ms | 134.87 MB | 160.98 MB | **1.19x** |
| InternVL3.5 Fast | 670.01 ms | 83.751 ms | 3.77 ms | 134.87 MB | 373.73 MB | **2.77x** |
| LLaVA Fast | 226.28 ms | 28.285 ms | 4.90 ms | 54.19 MB | 154.68 MB | **2.85x** |
| LLaVA Legacy | 1140.23 ms | 142.528 ms | 4.96 ms | 54.19 MB | 72.04 MB | **1.33x** |
