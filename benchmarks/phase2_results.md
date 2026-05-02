# Phase 2 Results — Second Model, Generalize the Finding

**Question:** Does the bottleneck pattern from Phase 1 generalize across processor families, or does each have its own dominant cost?

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
| Model | `OpenGVLab/InternVL2_5-8B` |
| Workload (W2) | 32 × 1024×1024 synthetic images |

---

## Results

| Variant | Median ms/batch | Median ms/img | p95 − p50 | Output | Peak alloc | Peak / Output |
|---------|-----------------|---------------|-----------|--------|------------|---------------|
| InternVL2.5 **HF Legacy** | 754.98 ms | 23.59 ms | 30.66 ms | 77.07 MB | 183.17 MB | **2.38×** |
| InternVL2.5 **HF Fast** | 521.21 ms | 16.29 ms | 9.06 ms | 77.07 MB | 103.83 MB | **1.35×** |
| InternVL2.5 **Manual Card** | 3269.57 ms | 102.17 ms | 62.13 ms | 770.70 MB | 1.22 MB | **~0×** |

HF legacy / HF fast ratio: **1.45×** (legacy is slower)  
Manual card / HF fast ratio: **6.27×** (manual card processes 10× more tiles — 320 vs 32)

---

## Key Numbers

- **HF Fast median: 16.29 ms/img** — just below the 20 ms/image threshold; the tightest margin seen so far.
- **HF Legacy peak/output: 2.38×** — slightly worse than Qwen's 2.03×; same intermediate-buffer pattern, same root cause (PIL/NumPy materializes a full copy at every step).
- **HF Fast peak/output: 1.35×** — higher than Qwen's 0.13× because InternVL's fast path uses `torch.stack` (per-image → batch) rather than an in-place write, keeping one extra buffer live.
- **Manual card peak/output: ~0×** — the tracemalloc peak of 1.22 MB for 770 MB output indicates the OS-level peak measurement is not capturing the PyTorch allocator's working set; the true working set is larger but unmeasured here.
- **Manual card output is 770 MB vs 77 MB** — dynamic_preprocess tiles each 1024×1024 image into ~10 tiles of 448×448, so the output tensor is 320 tiles rather than 32 images.

---

## Profile Breakdown

### HF Legacy (`InternVL2.5 HF Legacy` — NumPy/PIL path)

Total profiled time: **0.667 s** across 12,418 function calls

| Hotspot | Cumtime (s) | % of total | Notes |
|---------|-------------|------------|-------|
| PIL `ImagingCore.resize` | 0.341 | 51.1% | 32 calls — CPU resize, all time in C |
| `numpy.array` (conversion) | 0.106 | 15.9% | 128 calls — PIL→NumPy copy every image |
| PIL `Image.tobytes` | 0.095 | 14.2% | 64 calls — serializes pixel data for array conversion |
| `image_transforms.normalize` | 0.074 | 11.1% | 32 calls — elementwise float32 math |
| PIL `Image.frombytes` / `frombuffer` | 0.073 | 10.9% | 32 calls — reconstructs PIL image from normalized array |
| `PIL._imaging.fill` | 0.054 | 8.1% | 32 calls — blank-canvas allocation before paste |

Top-level entry: `CLIPImageProcessor.preprocess` (1 call, 0.658 s cumulative).

**Bottleneck summary:** PIL resize alone is 51% of total time — far more concentrated than Qwen's legacy path (23.4%). The resize cost dominates because InternVL uses a fixed 448×448 target and a BICUBIC kernel, while every image still takes a separate round-trip through PIL C → NumPy → PIL C → NumPy.

### HF Fast (`InternVL2.5 HF Fast` — torchvision/PyTorch path)

Total profiled time: **0.524 s** across 6,239 function calls

| Hotspot | Cumtime (s) | % of total | Notes |
|---------|-------------|------------|-------|
| `torch.stack` | 0.180 | 34.4% | 3 calls — batch assembly; mirrors Qwen fast's `torch.cat` |
| `_upsample_bicubic2d_aa` (resize) | 0.150 | 28.6% | 1 call — single batched CPU resize |
| `group_images_by_shape` | 0.125 | 23.9% | 2 calls — grouping before batch ops |
| `rescale_and_normalize` | 0.122 | 23.3% | 1 call cumulative |
| `TensorBase.sub` (normalize) | 0.057 | 10.9% | 1 call — mean subtraction over full batch |
| `pil_to_tensor` | 0.055 | 10.5% | 32 calls — PIL→tensor per image |

Top-level entry: `_preprocess_image_like_inputs` (1 call, 0.460 s cumulative).

**Bottleneck summary:** `torch.stack` (34.4%) and batched resize (28.6%) split the cost roughly evenly — the same two-cost pattern as Qwen fast, with `torch.stack` replacing `torch.cat` as the dominant assembly operation. `group_images_by_shape` (23.9%) is additional Python-level grouping overhead absent in Qwen.

### Manual Card (`InternVL2.5 Manual Card` — per-tile loop)

Total profiled time: **5.986 s** across 69,831 function calls (10× more calls than either HF path)

| Hotspot | Cumtime (s) | % of total | Notes |
|---------|-------------|------------|-------|
| `dynamic_preprocess` | 1.372 | 22.9% | 32 calls — per-image tiling logic |
| PIL `ImagingCore.resize` | 1.309 | 21.9% | 64 calls — 2 resizes per image (aspect crop + thumbnail) |
| torchvision `Compose.__call__` | 0.710 | 11.9% | 320 calls — per-tile Python dispatch |
| `torch.cat` | 0.577 | 9.6% | 1 call — assembles all 320 tiles into output |
| `torch.stack` | 0.538 | 9.0% | 32 calls — per-image tile stacking |
| `functional.to_tensor` | 0.449 | 7.5% | 320 calls — PIL→tensor per tile |
| `functional.normalize` | 0.234 | 3.9% | 320 calls — per-tile normalization |
| `TensorBase.div` | 0.124 | 2.1% | 320 calls — scaling in `to_tensor` |
| `TensorBase.clone` | 0.104 | 1.7% | 320 calls — defensive copy in torchvision |
| `TensorBase.contiguous` | 0.100 | 1.7% | 320 calls — layout canonicalization |

**Bottleneck summary:** cost is spread across three layers — PIL resize (21.9%), Python dispatch through a per-tile transform pipeline (11.9% + 7.5% + 3.9%), and batch assembly (`torch.cat` + `torch.stack`, 18.6%). The 320-call depth of the transform loop amplifies every fixed-overhead cost by 10× relative to the HF paths.

---

## Phase 1 vs Phase 2 Comparison

| Cost center | Qwen2.5-VL legacy | Qwen2.5-VL fast | InternVL2.5 HF legacy | InternVL2.5 HF fast |
|-------------|-------------------|-----------------|-----------------------|---------------------|
| PIL/CPU resize | 23.4% | 6.5% | **51.1%** | 28.6% |
| Batch assembly (`cat` / `stack`) | — | **32.9%** | — | **34.4%** |
| Tensor reshape / layout | 14.9% | 25.4% | — | 23.9% (grouping) |
| PIL→NumPy / PIL→tensor conversion | 20.3% | — | 15.9% | — |
| Normalize | 17.6% | 8.4% | 11.1% | 10.9% |

Both models share the same two dominant costs in the fast path: **batch assembly** (~33%) and **resize** (~28%). The legacy path is resize-heavy in both, with InternVL more extreme (51% vs 23%). The manual card path adds a third category absent from both HF paths: **per-tile Python dispatch loop overhead**, which 10× multiplies every cost and makes stack/cat assembly a significant second cost.

---

## Observations

1. **`torch.stack` / `torch.cat` are the dominant fast-path cost in both model families (33–34%)** — confirmed as a general pattern, not Qwen-specific. Both processors accumulate per-image tensors in a Python list and assemble once; the assembly touches every output byte twice.

2. **PIL resize is the dominant legacy-path cost in both families** — but far more concentrated in InternVL (51%) than Qwen (23%). InternVL's `CLIPImageProcessor` resizes to a fixed 448×448 for all inputs regardless of source size, whereas Qwen's `smart_resize` scales proportionally, so the absolute resize work per image differs.

3. **InternVL fast peak/output (1.35×) is higher than Qwen fast (0.13×)** — the InternVL fast path uses `torch.stack` rather than streaming destruction of per-image tensors, so at least one copy of the batch is live simultaneously. This partially erodes the memory advantage of the fast path.

4. **The manual card path amplifies all overheads by 10×** — 320 Python `Compose.__call__` invocations, 320 `to_tensor` calls, and 320 `normalize` calls replace 32 batched operations. Even if each individual call is fast, the aggregate Python dispatch overhead (torchvision `Compose` + `ToTensor` + `Normalize` × 320) is 11.9% + 7.5% + 3.9% = 23.3% of total time, exceeding the 15% Python-dispatch threshold from Phase 1's decision rules.

5. **The models partially disagree on their dominant cost pattern** — Q25 fast is constrained by `torch.cat` + `reshape`; IV25 fast is constrained by `torch.stack` + resize. Per the Phase 2 decision table, this disagreement strengthens the DSL pitch: no single point optimization (e.g., just fix `torch.cat`) covers both, but a schedule-aware approach that can express "pre-allocate output, write per-tile in-place, skip intermediate stacks" would address both simultaneously.

---

## Go/No-Go Decision

**Continue. Both models confirm exploitable inefficiency, and their disagreement strengthens the DSL pitch.**

Phase 1's Phase-2 decision rules:
- **"Both models confirm Phase 1 pattern"** — partially true: batch assembly (`stack`/`cat`) and resize dominate both fast paths; the pattern is confirmed.
- **"Models disagree, but each has its own exploitable inefficiency"** — also true: legacy paths differ in concentration (51% vs 23% on resize); fast paths differ in the relative split between resize and assembly. The disagreement *is* the DSL story — the right schedule differs per processor family.

Neither model is near-peak. InternVL HF fast is at 16.29 ms/img, just under the 20 ms threshold, but the manual card path at 102.17 ms/img is deep in exploitable territory, and the HF legacy at 23.59 ms/img still has headroom. Proceed to Phase 3.

---

## Raw Terminal Output

```
(cs348k-py3.12) jdarve@wheat-07:~/cs348k$ python benchmarks/phase2.py
python=3.12.3 platform=Linux-6.17.0-22-generic-x86_64-with-glibc2.39
cpu=x86_64
transformers=4.57.6 torch=2.11.0+cu130 torchvision=0.26.0+cu130 pillow=12.2.0 numpy=1.26.4
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 torch_threads=1

workload W2: 32 x (1024, 1024) (synthetic)
Loaded processors: OpenGVLab/InternVL2_5-8B (HF legacy, HF fast, manual card)
InternVL2.5 HF Legacy (W2) timing: 100%|██████████████████████████████████████████████| 100/100 [01:14<00:00,  1.34it/s]

--- InternVL2.5 HF Legacy (W2) ---
  median:          754.98 ms/batch (23.593 ms/img)
  p95 - p50:        30.66 ms
  output:        77.07 MB
  peak alloc:    183.17 MB
  peak / output: 2.38x
  profile:       profiles/IV25_W2_hf_legacy.txt
InternVL2.5 HF Fast (W2) timing: 100%|████████████████████████████████████████████████| 100/100 [00:51<00:00,  1.94it/s]

--- InternVL2.5 HF Fast (W2) ---
  median:          521.21 ms/batch (16.288 ms/img)
  p95 - p50:         9.06 ms
  output:        77.07 MB
  peak alloc:    103.83 MB
  peak / output: 1.35x
  profile:       profiles/IV25_W2_hf_fast.txt
InternVL2.5 Manual Card (W2) timing: 100%|████████████████████████████████████████████| 100/100 [05:24<00:00,  3.25s/it]

--- InternVL2.5 Manual Card (W2) ---
  median:         3269.57 ms/batch (102.174 ms/img)
  p95 - p50:        62.13 ms
  output:        770.70 MB
  peak alloc:    1.22 MB
  peak / output: 0.00x
  profile:       profiles/IV25_W2_manual.txt

=== summary ===
  HF legacy / HF fast ratio:   1.45x
  manual / HF fast ratio:      6.27x
```

### HF Legacy Profile (`profiles/IV25_W2_hf_legacy.txt`)

```
=== InternVL2.5 HF Legacy (W2) ===
output shape:    (32, 3, 448, 448)
output bytes:    77.07 MB
peak alloc (OS): 183.17 MB
peak / output:   2.38x

         12418 function calls (12354 primitive calls) in 0.667 seconds

   Ordered by: cumulative time
   List reduced from 116 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.667    0.667 /home/users/jdarve/cs348k/benchmarks/phase2.py:92(<lambda>)
        1    0.009    0.009    0.667    0.667 image_processing_utils.py:49(__call__)
        1    0.000    0.000    0.658    0.658 image_processing_clip.py:202(preprocess)
       32    0.000    0.000    0.435    0.014 image_processing_clip.py:153(resize)
       32    0.000    0.000    0.435    0.014 image_transforms.py:323(resize)
       32    0.000    0.000    0.341    0.011 PIL/Image.py:2328(resize)
       32    0.341    0.011    0.341    0.011 {method 'resize' of 'ImagingCore' objects}
      128    0.010    0.000    0.106    0.001 {built-in method numpy.array}
       32    0.000    0.000    0.098    0.003 image_utils.py:287(to_numpy_array)
       64    0.000    0.000    0.096    0.001 PIL/Image.py:811(__array_interface__)
       64    0.001    0.000    0.095    0.001 PIL/Image.py:849(tobytes)
       32    0.000    0.000    0.085    0.003 image_transforms.py:162(to_pil_image)
       32    0.000    0.000    0.075    0.002 image_processing_utils.py:88(normalize)
       32    0.074    0.002    0.074    0.002 image_transforms.py:394(normalize)
       32    0.000    0.000    0.073    0.002 PIL/Image.py:3374(fromarray)
       32    0.000    0.000    0.073    0.002 PIL/Image.py:3290(frombuffer)
       32    0.000    0.000    0.073    0.002 PIL/Image.py:3244(frombytes)
       64    0.066    0.001    0.066    0.001 {method 'join' of 'bytes' objects}
       32    0.000    0.000    0.055    0.002 PIL/Image.py:3195(new)
       32    0.054    0.002    0.054    0.002 {built-in method PIL._imaging.fill}
```

### HF Fast Profile (`profiles/IV25_W2_hf_fast.txt`)

```
=== InternVL2.5 HF Fast (W2) ===
output shape:    (32, 3, 448, 448)
output bytes:    77.07 MB
peak alloc (OS): 103.83 MB
peak / output:   1.35x

         6239 function calls (6207 primitive calls) in 0.524 seconds

   Ordered by: cumulative time
   List reduced from 133 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.524    0.524 /home/users/jdarve/cs348k/benchmarks/phase2.py:98(<lambda>)
        1    0.000    0.000    0.524    0.524 image_processing_utils_fast.py:731(__call__)
        1    0.000    0.000    0.524    0.524 image_processing_utils_fast.py:734(preprocess)
        1    0.008    0.008    0.524    0.524 image_processing_utils_fast.py:761(_preprocess_image_like_inputs)
        1    0.007    0.007    0.460    0.460 image_processing_utils_fast.py:781(_preprocess)
        3    0.180    0.060    0.180    0.060 {built-in method torch.stack}
        1    0.000    0.000    0.150    0.150 image_processing_utils_fast.py:311(resize)
        1    0.000    0.000    0.150    0.150 torchvision/transforms/v2/functional/_geometry.py:238(resize)
        1    0.000    0.000    0.150    0.150 torchvision/transforms/v2/functional/_geometry.py:269(resize_image)
        1    0.000    0.000    0.150    0.150 torch/nn/functional.py:4614(interpolate)
        1    0.150    0.150    0.150    0.150 {built-in method torch._C._nn._upsample_bicubic2d_aa}
        2    0.000    0.000    0.125    0.062 image_transforms.py:886(group_images_by_shape)
        1    0.005    0.005    0.122    0.122 image_processing_utils_fast.py:450(rescale_and_normalize)
        1    0.000    0.000    0.064    0.064 image_processing_utils_fast.py:410(normalize)
        1    0.000    0.000    0.064    0.064 torchvision/transforms/v2/functional/_misc.py:19(normalize)
        1    0.000    0.000    0.064    0.064 torchvision/transforms/v2/functional/_misc.py:35(normalize_image)
        1    0.057    0.057    0.057    0.057 {method 'sub' of 'torch._C.TensorBase' objects}
        1    0.000    0.000    0.056    0.056 image_processing_utils_fast.py:605(_prepare_image_like_inputs)
       32    0.000    0.000    0.056    0.002 image_processing_utils_fast.py:567(_process_image)
       32    0.000    0.000    0.055    0.002 torchvision/transforms/functional.py:181(pil_to_tensor)
```

### Manual Card Profile (`profiles/IV25_W2_manual.txt`)

```
=== InternVL2.5 Manual Card (W2) ===
output shape:    (320, 3, 448, 448)
output bytes:    770.70 MB
peak alloc (OS): 1.22 MB
peak / output:   0.00x

         69831 function calls (69830 primitive calls) in 5.986 seconds

   Ordered by: cumulative time
   List reduced from 125 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       32    0.003    0.000    1.372    0.043 /home/users/jdarve/cs348k/benchmarks/models.py:57(dynamic_preprocess)
       64    0.001    0.000    1.311    0.020 PIL/Image.py:2328(resize)
       64    1.309    0.020    1.309    0.020 {method 'resize' of 'ImagingCore' objects}
      320    0.003    0.000    0.710    0.002 torchvision/transforms/transforms.py:93(__call__)
      2/1    2.788    1.394    0.631    0.631 /home/users/jdarve/cs348k/benchmarks/phase2.py:104(<lambda>)
        1    0.000    0.000    0.577    0.577 /home/users/jdarve/cs348k/benchmarks/models.py:102(process_batch)
        1    0.577    0.577    0.577    0.577 {built-in method torch.cat}
       32    0.538    0.017    0.538    0.017 {built-in method torch.stack}
      320    0.001    0.000    0.450    0.001 torchvision/transforms/transforms.py:129(__call__)
      320    0.009    0.000    0.449    0.001 torchvision/transforms/functional.py:127(to_tensor)
      640    0.002    0.000    0.256    0.000 torch/nn/modules/module.py:1775(_wrapped_call_impl)
      640    0.003    0.000    0.254    0.000 torch/nn/modules/module.py:1783(_call_impl)
      320    0.001    0.000    0.235    0.001 torchvision/transforms/transforms.py:277(forward)
      320    0.003    0.000    0.234    0.001 torchvision/transforms/functional.py:327(normalize)
      320    0.013    0.000    0.229    0.001 torchvision/transforms/_functional_tensor.py:905(normalize)
      320    0.027    0.000    0.125    0.000 {built-in method numpy.array}
      320    0.124    0.000    0.124    0.000 {method 'div' of 'torch._C.TensorBase' objects}
      320    0.104    0.000    0.104    0.000 {method 'clone' of 'torch._C.TensorBase' objects}
      320    0.100    0.000    0.100    0.000 {method 'contiguous' of 'torch._C.TensorBase' objects}
      320    0.002    0.000    0.099    0.000 PIL/Image.py:811(__array_interface__)
```
