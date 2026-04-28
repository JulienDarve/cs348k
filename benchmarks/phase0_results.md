# Phase 0 Results — Sanity Check

**Question:** Is HF VLM preprocessing slow enough to be worth attacking at all?

---

## Environment

| Field | Value |
|-------|-------|
| Host | `wheat-04` (Stanford Farmshare) |
| OS | Linux |
| SLURM CPUs allocated | 8 (`SLURM_CPUS_PER_TASK`) |
| Threads used | 1 (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `torch.set_num_threads(1)`) |
| Model | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Workload (W2) | 32 × 1024×1024 synthetic `uint8` images, pre-decoded into RAM |
| Warmup / Timed | 10 / 100 iterations |
| Timing | `time.perf_counter_ns()`, GC disabled around timed loop |

---

## Results

| System | Backend | Threads | Median ms/batch | p95 ms/batch | Median ms/image |
|--------|---------|---------|-----------------|--------------|-----------------|
| Q25 legacy (`Qwen2VLImageProcessor`) | NumPy / PIL | 1 | 2656.35 | 2686.07 | **83.01** |
| Q25F fast (`Qwen2VLImageProcessorFast`) | torchvision / PyTorch | 1 | 3641.83 | 3661.00 | **113.81** |
| Q25 legacy (`Qwen2VLImageProcessor`) | NumPy / PIL | 8 | 2670.59 | 3015.83 | **83.46** |
| Q25F fast (`Qwen2VLImageProcessorFast`) | torchvision / PyTorch | 8 | 1291.20 | 1421.75 | **40.35** |

Legacy 1t / Fast 1t ratio: **0.73×** (legacy is faster single-threaded)  
Fast 1t / Fast 8t speedup: **2.82×** on 8 threads (sublinear scaling)  
Legacy 1t / Legacy 8t: **1.00×** — legacy does not scale with threads at all

---

## Key Numbers

- **Fast path 1t median: 113.81 ms/image** — well above the 20 ms/image threshold.
- **Fast path 8t median: 40.35 ms/image** — still above 20 ms/image even with 8 threads.
- **Legacy path 1t median: 83.01 ms/image** — also well above 20 ms/image.
- **Legacy path 8t median: 83.46 ms/image** — essentially identical to 1t; the NumPy/PIL path does not scale with threads.

---

## Go/No-Go Decision

**Go. Continue to Phase 1.**

Both processors are deep in the "real headroom" region (≥ 20 ms/image). At 113 ms/image on the fast path, a 3× speedup would save ~75 ms per image — meaningful for any serving workload that processes more than a handful of tiles per request. The p95 values are nearly flat (within 30 ms of median), indicating stable, non-spiky behavior, so the median is a reliable target.

The legacy NumPy/PIL path is completely thread-insensitive: 83.01 ms/image at 1 thread vs. 83.46 ms/image at 8 threads — no scaling whatsoever, confirming the parallelism is entirely absent at the library level. The fast torchvision path does scale (113.81 → 40.35 ms/image, 2.82× on 8 threads), but sublinearly and only because PyTorch dispatches work across threads internally. Crucially, even the best observed number (40.35 ms/image, 8 threads) remains above the 20 ms/image threshold, so headroom is real at every operating point. A fused single-pass kernel that eliminates intermediate DRAM round-trips could plausibly beat the 8-thread fast baseline on a single thread.

Neither processor is near the < 5 ms/image cutoff that would terminate the project. Phase 1 will profile where the time actually goes (Python dispatch vs. C kernels) and measure peak/output memory ratio to confirm or deny the fusion pitch.

---

## Raw Terminal Output

```
(cs348k-py3.12) jdarve@wheat-04:~/cs348k$ python benchmarks/phase0.py
fast class: Qwen2VLImageProcessorFast (transformers.models.qwen2_vl.image_processing_qwen2_vl_fast)
Model:    Qwen/Qwen2.5-VL-7B-Instruct
Workload: 32 x 1024x1024 synthetic uint8 images
Allocated CPUs: 8 (SLURM_CPUS_PER_TASK or os.cpu_count())
Warmup:   10, Timed: 100

--- torch.get_num_threads() = 1 ---
Q25  legacy np 1t timed : 100%|████████████████| 100/100 [04:25<00:00,  2.66s/it]
Q25  legacy np 1t: median  2656.35 ms/call  p95  2686.07 ms  |  per-image median   83.01 ms
Q25F fast   pt 1t timed : 100%|████████████████| 100/100 [06:04<00:00,  3.64s/it]
Q25F fast   pt 1t: median  3641.83 ms/call  p95  3661.00 ms  |  per-image median  113.81 ms

--- torch.get_num_threads() = 8 ---
Q25F fast   pt 8t timed : 100%|████████████████| 100/100 [02:11<00:00,  1.31s/it]
Q25F fast   pt 8t: median  1291.20 ms/call  p95  1421.75 ms  |  per-image median   40.35 ms

--- torch.get_num_threads() = 8 ---
Q25  legacy np 8t timed : 100%|████████████████| 100/100 [04:31<00:00,  2.72s/it]
Q25  legacy np 8t: median  2670.59 ms/call  p95  3015.83 ms  |  per-image median   83.46 ms
```
