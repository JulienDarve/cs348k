# AWS c7i.4xlarge Multi-Thread Results

Source: `benchmarks/full_memory_benchmark.py --num-threads 8` on AWS `c7i.4xlarge`.

Timing columns are `N/A` because the captured run was the clean memory-only benchmark.

# W2

| Model | Median ms/batch | Median ms/img | p95 - p50 | Output | Peak RSS | Peak / Output |
|---|---:|---:|---:|---:|---:|---:|
| Qwen Fast | N/A | N/A | N/A | 824.29 MB | 3104.17 MB | **3.77x** |
| Qwen Legacy | N/A | N/A | N/A | 824.29 MB | 1758.75 MB | **2.13x** |
| Intern2.5 Manual | N/A | N/A | N/A | 770.70 MB | 1489.69 MB | **1.93x** |
| InternVL3.5 Legacy | N/A | N/A | N/A | 770.70 MB | 1452.86 MB | **1.89x** |
| InternVL3.5 Fast | N/A | N/A | N/A | 770.70 MB | 2138.87 MB | **2.78x** |
| LLaVA Fast | N/A | N/A | N/A | 216.76 MB | 765.66 MB | **3.53x** |
| LLaVA Legacy | N/A | N/A | N/A | 216.76 MB | 631.09 MB | **2.91x** |

# W3

| Model | Median ms/batch | Median ms/img | p95 - p50 | Output | Peak RSS | Peak / Output |
|---|---:|---:|---:|---:|---:|---:|
| Qwen Fast | N/A | N/A | N/A | 333.42 MB | 607.82 MB | **1.82x** |
| Qwen Legacy | N/A | N/A | N/A | 333.42 MB | 481.66 MB | **1.44x** |
| Intern2.5 Manual | N/A | N/A | N/A | 472.06 MB | 712.51 MB | **1.51x** |
| InternVL3.5 Legacy | N/A | N/A | N/A | 472.06 MB | 674.24 MB | **1.43x** |
| InternVL3.5 Fast | N/A | N/A | N/A | 472.06 MB | 1180.16 MB | **2.50x** |
| LLaVA Fast | N/A | N/A | N/A | 216.76 MB | 376.64 MB | **1.74x** |
| LLaVA Legacy | N/A | N/A | N/A | 216.76 MB | 367.32 MB | **1.69x** |

# W4

| Model | Median ms/batch | Median ms/img | p95 - p50 | Output | Peak RSS | Peak / Output |
|---|---:|---:|---:|---:|---:|---:|
| Qwen Fast | N/A | N/A | N/A | 1674.62 MB | 6285.50 MB | **3.75x** |
| Qwen Legacy | N/A | N/A | N/A | 1674.62 MB | 3383.67 MB | **2.02x** |
| Intern2.5 Manual | N/A | N/A | N/A | 134.87 MB | 134.88 MB | **1.00x** |
| InternVL3.5 Legacy | N/A | N/A | N/A | 134.87 MB | 160.98 MB | **1.19x** |
| InternVL3.5 Fast | N/A | N/A | N/A | 134.87 MB | 373.73 MB | **2.77x** |
| LLaVA Fast | N/A | N/A | N/A | 54.19 MB | 154.68 MB | **2.85x** |
| LLaVA Legacy | N/A | N/A | N/A | 54.19 MB | 72.04 MB | **1.33x** |
