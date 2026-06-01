# AWS c7i.4xlarge Single-Thread Memory Results

Source: `benchmarks/full_memory_benchmark.py --num-threads 1` on AWS `c7i.4xlarge`.

Timing columns are `N/A` because the source run was the clean memory-only benchmark.

# W2

| Model | Median ms/batch | Median ms/img | p95 - p50 | Output | Peak RSS | Peak / Output |
|---|---:|---:|---:|---:|---:|---:|
| Qwen Fast | N/A | N/A | N/A | 824.29 MB | 3103.99 MB | **3.77x** |
| Qwen Legacy | N/A | N/A | N/A | 824.29 MB | 1758.77 MB | **2.13x** |
| Intern2.5 Manual | N/A | N/A | N/A | 770.70 MB | 1489.83 MB | **1.93x** |
| InternVL3.5 Legacy | N/A | N/A | N/A | 770.70 MB | 995.25 MB | **1.29x** |
| InternVL3.5 Fast | N/A | N/A | N/A | 770.70 MB | 2138.87 MB | **2.78x** |
| LLaVA Fast | N/A | N/A | N/A | 216.76 MB | 752.11 MB | **3.47x** |
| LLaVA Legacy | N/A | N/A | N/A | 216.76 MB | 624.31 MB | **2.88x** |

# W3

| Model | Median ms/batch | Median ms/img | p95 - p50 | Output | Peak RSS | Peak / Output |
|---|---:|---:|---:|---:|---:|---:|
| Qwen Fast | N/A | N/A | N/A | 333.42 MB | 508.85 MB | **1.53x** |
| Qwen Legacy | N/A | N/A | N/A | 333.42 MB | 414.80 MB | **1.24x** |
| Intern2.5 Manual | N/A | N/A | N/A | 472.06 MB | 773.66 MB | **1.64x** |
| InternVL3.5 Legacy | N/A | N/A | N/A | 472.06 MB | 621.39 MB | **1.32x** |
| InternVL3.5 Fast | N/A | N/A | N/A | 472.06 MB | 1180.16 MB | **2.50x** |
| LLaVA Fast | N/A | N/A | N/A | 216.76 MB | 567.65 MB | **2.62x** |
| LLaVA Legacy | N/A | N/A | N/A | 216.76 MB | 514.32 MB | **2.37x** |

# W4

| Model | Median ms/batch | Median ms/img | p95 - p50 | Output | Peak RSS | Peak / Output |
|---|---:|---:|---:|---:|---:|---:|
| Qwen Fast | N/A | N/A | N/A | 1674.62 MB | 6436.32 MB | **3.84x** |
| Qwen Legacy | N/A | N/A | N/A | 1674.62 MB | 3383.13 MB | **2.02x** |
| Intern2.5 Manual | N/A | N/A | N/A | 134.87 MB | 168.47 MB | **1.25x** |
| InternVL3.5 Legacy | N/A | N/A | N/A | 134.87 MB | 148.98 MB | **1.10x** |
| InternVL3.5 Fast | N/A | N/A | N/A | 134.87 MB | 530.19 MB | **3.93x** |
| LLaVA Fast | N/A | N/A | N/A | 54.19 MB | 271.63 MB | **5.01x** |
| LLaVA Legacy | N/A | N/A | N/A | 54.19 MB | 229.88 MB | **4.24x** |
