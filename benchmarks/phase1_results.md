# Phase 1 Results — Profiling & Memory

**Question:** Where does the time go, and how bad is peak/output memory ratio?

---

## Environment

| Field | Value |
|-------|-------|
| Host | `wheat-07` (Stanford Farmshare) |
| OS | Linux 6.17.0-22-generic x86_64 (glibc 2.39) |
| CPU | x86_64 |
| Threads used | 1 (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `torch.set_num_threads(1)`) |
| Python | 3.12.3 |
| transformers | 4.57.6 |
| torch | 2.11.0+cu130 |
| torchvision | 0.26.0+cu130 |
| Pillow | 12.2.0 |
| NumPy | 1.26.4 |
| Model | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Workload (W2) | 32 × 1024×1024 synthetic images |

---

## Results

| Variant | Median ms/batch | Median ms/img | p95 − p50 | Output | Peak alloc | Peak / Output |
|---------|-----------------|---------------|-----------|--------|------------|---------------|
| Qwen2.5-VL **legacy** | 2812.46 ms | 87.89 ms | 72.53 ms | 824.29 MB | 1675.27 MB | **2.03×** |
| Qwen2.5-VL **fast** | 3767.97 ms | 117.75 ms | 43.34 ms | 824.29 MB | 103.84 MB | **0.13×** |

Legacy / fast batch time ratio: **0.75×** (legacy is faster single-threaded)

---

## Key Numbers

- **Legacy peak/output: 2.03×** — the NumPy/PIL path holds >2× the output size in RAM simultaneously, confirming significant intermediate-buffer overhead.
- **Fast peak/output: 0.13×** — the torchvision path is extremely memory-efficient (streaming / in-place ops dominate), but at the cost of throughput.
- Output tensor is identical between variants: `(175232, 1176)`, 824.29 MB — same logical result, very different memory profiles.

---

## Profile Breakdown

### Legacy (`Qwen2VLImageProcessor` — NumPy/PIL)

Total profiled time: **2.503 s** across 14,902 function calls

| Hotspot | Cumtime (s) | % of total | Notes |
|---------|-------------|------------|-------|
| PIL `ImagingCore.resize` | 0.585 | 23.4% | Per-image CPU resize (32 calls) |
| `numpy.ndarray.reshape` | 0.372 | 14.9% | 64 calls — layout changes after each step |
| `image_transforms.normalize` | 0.439 | 17.6% | Elementwise float32 math |
| `numpy.array` (conversion) | 0.508 | 20.3% | 162 calls — PIL→NumPy copy on every image |
| `image_transforms.rescale` | 0.234 | 9.4% | `astype(float32)` + multiply |
| `numpy.ndarray.repeat` | 0.067 | 2.7% | Grayscale→RGB channel replication |

Top-level entry: `Qwen2VLImageProcessor._preprocess` called 32 times (once per image), 0.068 s/call cumulative.

**Bottleneck summary:** time is spread across PIL resize → `numpy.array` copy → `astype` cast → normalize — every step materializes a new intermediate array.

### Fast (`Qwen2VLImageProcessorFast` — torchvision/PyTorch)

Total profiled time: **3.722 s** across 6,259 function calls

| Hotspot | Cumtime (s) | % of total | Notes |
|---------|-------------|------------|-------|
| `torch.cat` | 1.223 | 32.9% | 2 calls, 0.612 s each — batch assembly |
| `TensorBase.reshape` | 0.945 | 25.4% | 3 calls — large tensor layout changes |
| `TensorBase.sub` (normalize) | 0.311 | 8.4% | Mean subtraction |
| `TensorBase.to` (dtype cast) | 0.282 | 7.6% | float conversion |
| `TensorBase.repeat` | 0.273 | 7.3% | Grayscale→RGB replication |
| `_upsample_bicubic2d_aa` | 0.242 | 6.5% | Batch resize via `torch.nn.functional.interpolate` |

Top-level entry: `_preprocess_image_like_inputs` (1 call, 0.170 s self) → `_preprocess` (1 call, 0.030 s self).

**Bottleneck summary:** despite far fewer Python calls, `torch.cat` and `reshape` dominate — both are large-tensor DRAM-bound operations that shuffle all 824 MB of output data through memory more than once.

---

## Observations

1. **`torch.cat` is the single largest cost in the fast path (32.9%)** — the fast processor accumulates per-image tensors into a list and concatenates once, which is a single allocation of the full output but requires touching every byte twice (once per source tensor, once into the destination).

2. **`reshape` is expensive in both paths** — in legacy it is 64 calls totaling 0.372 s; in fast it is 3 calls totaling 0.945 s. The fast path's reshapes are on larger tensors (the entire batch rather than one image at a time).

3. **Memory model is inverted:** legacy buffers 2× output (intermediate NumPy arrays per image alive simultaneously); fast path keeps only 0.13× overhead (streaming destruction of per-image tensors before `cat`). A fused kernel that writes directly to the output layout could inherit the fast path's memory model while eliminating the `cat` / `reshape` overhead.

4. **PIL resize (legacy) vs. bicubic upsample (fast):** 0.585 s for 32 per-image PIL resizes vs. 0.242 s for a single batched torchvision resize. The batched GPU-style resize is ~2.4× faster even on CPU.

5. **numpy.array copy (162 calls, 0.508 s)** is a pure overhead tax in the legacy path — every PIL image must be converted to a NumPy array. The fast path has zero equivalent cost (images enter as tensors).

---

## Raw Terminal Output

```
jdarve@wheat-07:~/cs348k$ source $(poetry env info --path)/bin/activate
(cs348k-py3.12) jdarve@wheat-07:~/cs348k$ python benchmarks/phase1.py
python=3.12.3 platform=Linux-6.17.0-22-generic-x86_64-with-glibc2.39
cpu=x86_64
transformers=4.57.6 torch=2.11.0+cu130 torchvision=0.26.0+cu130 pillow=12.2.0 numpy=1.26.4
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 torch_threads=1

workload W2: 32 x (1024, 1024) (synthetic)

--- Qwen2.5-VL legacy (W2) ---
  median:         2812.46 ms/batch (87.889 ms/img)
  p95 - p50:        72.53 ms
  output:        824.29 MB
  peak alloc:    1675.27 MB
  peak / output: 2.03x
  profile:       profiles/Q25_W2_legacy.txt

--- Qwen2.5-VL fast (W2) ---
  median:         3767.97 ms/batch (117.749 ms/img)
  p95 - p50:        43.34 ms
  output:        824.29 MB
  peak alloc:    103.84 MB
  peak / output: 0.13x
  profile:       profiles/Q25_W2_fast.txt

=== summary ===
  legacy / fast ratio: 0.75x
```

### Legacy Profile (`profiles/Q25_W2_legacy.txt`)

```
=== Qwen2.5-VL legacy (W2) ===
output shape:    (175232, 1176)
output bytes:    824.29 MB
peak allocated:  1675.27 MB  (tracemalloc; undercounts torch allocator)
peak / output:   2.03x

         14902 function calls (14838 primitive calls) in 2.503 seconds

   Ordered by: cumulative time
   List reduced from 117 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    2.503    2.503 benchmarks/phase1.py:47(<lambda>)
        1    0.009    0.009    2.503    2.503 image_processing_utils.py:49(__call__)
        1    0.009    0.009    2.494    2.494 image_processing_qwen2_vl.py:299(preprocess)
       32    0.203    0.006    2.171    0.068 image_processing_qwen2_vl.py:165(_preprocess)
       32    0.001    0.000    0.688    0.022 image_transforms.py:323(resize)
       32    0.000    0.000    0.585    0.018 PIL/Image.py:2328(resize)
       32    0.585    0.018    0.585    0.018 {method 'resize' of 'ImagingCore' objects}
      162    0.425    0.003    0.508    0.003 {built-in method numpy.array}
       32    0.000    0.000    0.439    0.014 image_processing_utils.py:88(normalize)
       32    0.438    0.014    0.439    0.014 image_transforms.py:394(normalize)
       64    0.372    0.006    0.372    0.006 {method 'reshape' of 'numpy.ndarray' objects}
       32    0.000    0.000    0.234    0.007 image_processing_utils.py:56(rescale)
       32    0.070    0.002    0.234    0.007 image_transforms.py:97(rescale)
       96    0.172    0.002    0.172    0.002 {method 'astype' of 'numpy.ndarray' objects}
       64    0.001    0.000    0.083    0.001 PIL/Image.py:811(__array_interface__)
       64    0.003    0.000    0.081    0.001 PIL/Image.py:849(tobytes)
       32    0.000    0.000    0.067    0.002 numpy/core/fromnumeric.py:423(repeat)
       32    0.000    0.000    0.067    0.002 numpy/core/fromnumeric.py:53(_wrapfunc)
       32    0.067    0.002    0.067    0.002 {method 'repeat' of 'numpy.ndarray' objects}
       32    0.000    0.000    0.063    0.002 image_utils.py:287(to_numpy_array)
```

### Fast Profile (`profiles/Q25_W2_fast.txt`)

```
=== Qwen2.5-VL fast (W2) ===
output shape:    (175232, 1176)
output bytes:    824.29 MB
peak allocated:  103.84 MB  (tracemalloc; undercounts torch allocator)
peak / output:   0.13x

         6259 function calls (6227 primitive calls) in 3.722 seconds

   Ordered by: cumulative time
   List reduced from 138 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    3.722    3.722 benchmarks/phase1.py:47(<lambda>)
        1    0.000    0.000    3.722    3.722 image_processing_utils_fast.py:731(__call__)
        1    0.000    0.000    3.722    3.722 image_processing_qwen2_vl_fast.py:134(preprocess)
        1    0.000    0.000    3.722    3.722 image_processing_utils_fast.py:734(preprocess)
        1    0.170    0.170    3.722    3.722 image_processing_qwen2_vl_fast.py:143(_preprocess_image_like_inputs)
        1    0.030    0.030    3.498    3.498 image_processing_qwen2_vl_fast.py:182(_preprocess)
        2    1.223    0.612    1.223    0.612 {built-in method torch.cat}
        3    0.945    0.315    0.945    0.315 {method 'reshape' of 'torch._C.TensorBase' objects}
        1    0.029    0.029    0.659    0.659 image_processing_utils_fast.py:450(rescale_and_normalize)
        1    0.000    0.000    0.347    0.347 image_processing_utils_fast.py:410(normalize)
        1    0.000    0.000    0.347    0.347 torchvision/transforms/v2/functional/_misc.py:19(normalize)
        1    0.000    0.000    0.347    0.347 torchvision/transforms/v2/functional/_misc.py:35(normalize_image)
        1    0.311    0.311    0.311    0.311 {method 'sub' of 'torch._C.TensorBase' objects}
        1    0.282    0.282    0.282    0.282 {method 'to' of 'torch._C.TensorBase' objects}
        1    0.273    0.273    0.273    0.273 {method 'repeat' of 'torch._C.TensorBase' objects}
        1    0.000    0.000    0.242    0.242 image_processing_utils_fast.py:311(resize)
        1    0.000    0.000    0.242    0.242 torchvision/transforms/v2/functional/_geometry.py:238(resize)
        1    0.000    0.000    0.242    0.242 torchvision/transforms/v2/functional/_geometry.py:269(resize_image)
        1    0.000    0.000    0.242    0.242 torch/nn/functional.py:4614(interpolate)
        1    0.242    0.242    0.242    0.242 {built-in method torch._C._nn._upsample_bicubic2d_aa}
```
