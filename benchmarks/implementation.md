# VLM Preprocessing Baseline — Phased Experiment Plan

**Purpose.** Determine, in stages, whether HF image preprocessing for dynamic-resolution VLMs has enough exploitable slack to justify a schedule-aware DSL. Each phase is cheap, ends in a written go/no-go decision, and the work compounds: the harness from Phase 0 is what Phase 1 profiles, what Phase 2 extends, and so on. If the project survives all phases, we land in the same place as a full benchmark suite — but we get there having killed the project early if it deserves to die.

---

## Phase 0 — 30-minute sanity check

**Question.** Is HF preprocessing slow enough to be worth attacking at all?

**Setup.**
- One workload: 32 × 1024×1024 PIL images, pre-decoded into RAM. Use COCO val2017 or any equivalent natural-image set.
- Two systems: `Qwen2VLImageProcessor` (legacy NumPy/PIL backend) and `Qwen2VLImageProcessorFast` (torchvision backend).
- Single thread: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `torch.set_num_threads(1)`.
- 10 warmup iterations, 100 timed iterations, `time.perf_counter_ns()`. Report median ms per batch and median ms per image.

**Skeleton.**
```python
# phase0.py
import os, gc, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import torch; torch.set_num_threads(1)
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor

def time_fn(fn, n_warmup=10, n_timed=100):
    for _ in range(n_warmup): fn()
    gc.collect(); gc.disable()
    try:
        ts = []
        for _ in range(n_timed):
            t0 = time.perf_counter_ns()
            fn()
            ts.append(time.perf_counter_ns() - t0)
    finally:
        gc.enable()
    return float(np.median(ts) / 1e6)

images = [Image.open(p).convert("RGB").resize((1024, 1024)) for p in IMG_PATHS[:32]]

slow = AutoImageProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=False)
fast = AutoImageProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=True)

slow_ms = time_fn(lambda: slow(images=images, return_tensors="np"))
fast_ms = time_fn(lambda: fast(images=images, return_tensors="pt"))
print(f"legacy: {slow_ms:.1f} ms/batch ({slow_ms/32:.2f} ms/img)")
print(f"fast:   {fast_ms:.1f} ms/batch ({fast_ms/32:.2f} ms/img)")
print(f"ratio:  {slow_ms/fast_ms:.2f}x")
```

**Decision rules (write them down before running):**

| Result | Action |
|--------|--------|
| Fast median < ~5 ms/image | Fast path is near-peak. Pivot scope or kill project. |
| Fast median ≥ ~20 ms/image | Real headroom. Continue to Phase 1. |
| 5 ≤ Fast < 20 ms/image | Ambiguous; continue to Phase 1 but with sharpened pitch. |
| Legacy / Fast ratio ≥ 5× | Python overhead is real; first data point for the DSL story. |

**Deliverable.** Two numbers and a one-paragraph go/no-go note. No figures, no writeup. Total time: ~30 minutes.

---

## Phase 1 — One model, real profile (half day)

**Question.** Where does the time actually go? Specifically: is it in Pillow/NumPy C kernels (only fusion helps) or in Python-level dispatch and concatenation (batching alone helps)?

**Setup.** Same workload and systems as Phase 0. Add:
- `cProfile` over a single run of the timed function (one batch, not 100), top 20 by cumulative time.
- `tracemalloc` peak allocated bytes over a single run.
- Compute peak-allocated / output-tensor-bytes ratio. Ratios near 1 mean little to fuse; ratios > 2 mean intermediates dominate.

**What to look for in the profile.**

| Pattern in top 20 | Interpretation | DSL implication |
|-------------------|----------------|-----------------|
| `PIL.Image.resize`, `Image.Resampling.BICUBIC` dominate | Compute-bound on resize itself | Need a faster resize kernel (Pillow-SIMD baseline first) |
| `_preprocess`, `convert_to_rgb`, `to_numpy_array`, `to_channel_dimension_format` collectively > 15% | Python dispatch + layout churn | Batching + canonical layout is most of the win |
| `np.concatenate`, `np.stack`, `torch.stack` in top 10 | Output-assembly overhead | Pre-allocate output buffer; write in place |
| Roughly even spread across resize / normalize / patchify / stack | Classic fusion target | Fused single-pass kernel is the play |

**Decision rules:**

| Result | Action |
|--------|--------|
| > 85% of time in compiled C/C++ kernels | Fusion has little headroom. Reconsider scope (GPU? schedule search?). |
| > 15% of time in Python dispatch / conversions | Batching pitch is real. Continue to Phase 2. |
| Peak / output ratio > 2× | Fusion pitch is real. Continue to Phase 2. |
| Both of the above | Both DSL pillars confirmed. Continue with confidence. |

**Deliverable.** One profile output (saved to `profiles/Q25_W2.txt`), one memory ratio, one paragraph stating which inefficiencies are confirmed and which aren't.

---

## Phase 2 — Second model, generalize the finding (half day)

**Question.** Does the bottleneck pattern from Phase 1 generalize across processor families, or does each have its own dominant cost?

**Setup.** Same workload. Add InternVL2.5 with the inline `dynamic_preprocess` function copied from `huggingface.co/OpenGVLab/InternVL2_5-8B`. Re-run timing + profile + memory.

InternVL is the right second model because it exercises a fundamentally different strategy (dynamic tiles + thumbnail) than Qwen's smart_resize + patchify. If both processors show the same dominant inefficiency, the DSL has a single, focused job. If they differ, the case for a *general* DSL is stronger because no point solution covers both.

**What this distinguishes:**

| Q25 dominant cost | IV25 dominant cost | What it means |
|-------------------|---------------------|---------------|
| Python dispatch | Python dispatch | One DSL handles both via batching alone |
| Resize + materialization | Per-tile loop + `torch.stack` | DSL must handle both fusion and tile-level parallelism — *stronger* generality argument |
| Mostly C kernels | Mostly C kernels | Project pivot needed; both regimes already near peak |
| Mostly C kernels | Python loops | Pitch narrows to dynamic-tile processors specifically |

LLaVA-NeXT is deliberately deferred to Phase 4. Two model families is enough to test generality; the third is for the final writeup.

**Decision rules.**

| Result | Action |
|--------|--------|
| Both models confirm Phase 1 pattern | Continue. Pitch is robust. |
| Models disagree, but each has its own exploitable inefficiency | Continue. The disagreement *is* the DSL pitch. |
| Both models near-peak | Stop. Pivot project. |

**Deliverable.** Two profile files, two memory ratios, table comparing the two systems' top 5 cost centers side by side. One paragraph.

---

## Phase 3 — Vary the workload, find the real go/no-go (half day)

**Question.** Does the bottleneck shift with workload, and if so, does the variation expose specific scheduling decisions the DSL would need to make?

**Setup.** Keep Q25 and IV25 (and the Fast variants where they exist). Add two workloads beyond W2-batch-uniform:

- **W3-batch-mixed** — 32 images with random sizes in [512, 2048] per dim, aspect in [0.5, 2.0]. Stresses dynamic resize paths and prevents cached-shape optimizations.
- **W4-document** — 8 × 2480×3508 images (A4 @ 300 dpi). Triggers max-tile branches in IV25 and produces large output tensors.

Re-run timing + profile + memory on each combination (4 workloads now, including Phase 0's W2 — go ahead and re-time it for cleanliness, and add a single-image W1 as a cheap fourth row).

**What this answers.** If Q25's profile looks identical across W2/W3/W4, then a single static schedule suffices and the "schedule" abstraction is overkill. If the profile shifts substantially — e.g., `smart_resize` Python work explodes on W3, or `np.concatenate` dominates W4 — that *is* the case for exposing scheduling axes. The DSL only earns its complexity if different workloads want different schedules.

**Decision rules — the real go/no-go for the DSL.**

| Result | Action |
|--------|--------|
| Profile composition stable across workloads | DSL is overkill; a single optimized implementation suffices. Reconsider framing. |
| Profile shifts meaningfully across workloads, with clear "this schedule for this workload" patterns | DSL pitch is fully justified. Proceed to Phase 4. |
| Headroom < 1.5× across all workloads | Stop, even if profiles vary. Not enough room to matter. |
| Headroom ≥ 3× on at least two workloads | Strong proceed. Quantify per-workload targets. |

**Deliverable.** A 2 × 4 table (2 systems × 4 workloads) with median ms, peak/output ratio, and top-3 cost centers per cell. The decision: build the DSL with quantified per-workload targets, or stop. Roughly two days of cumulative work to this point.

---

## Phase 4 — Round out the baseline (only if Phase 3 says go)

Everything from the original spec that didn't make it into Phases 0-3:

- **Add LLaVA-NeXT** (legacy + Fast). Third model family confirms the result for the writeup.
- **Add Fast variants for all systems** where not already included.
- **Add W5-video** (64 × 640×360). The video regime is where preprocessing is genuinely a serving bottleneck and worth a dedicated row.
- **Correctness verification.** Output tensors from any future re-implementation must match HF byte-for-byte at FP32 (or within 1e-5 at FP16) on the first 10 images of each workload. A speed number without this is meaningless.
- **Multi-thread row.** Re-run with `num_threads = physical_cores` to characterize how HF processors scale (they often don't).
- **Optional GPU stretch row.** One config with CuPy or torchvision-on-CUDA, single workload, just to contextualize whether CPU is still the right battle.

**Deliverables (this is the writeup-ready baseline).**

1. **Table 1 (main result).** Rows = workloads, cols = systems. Cells = median ms ± (p95 − p50).
2. **Figure 1.** Throughput (images/sec) per workload per system, bar chart.
3. **Figure 2 (the DSL story).** Stacked bar per system on W3-batch-mixed, decomposing time into {resize, rescale/normalize, patchify/tile, stack/concat, Python dispatch, other} from cProfile.
4. **Table 2.** Peak memory / output memory ratio per system per workload. Quantifies how much fusion can recover.
5. **Appendix.** Full profile traces, env info, version pins, correctness check results.

---

## Cross-cutting protocol (applies to all phases)

- Single thread by default: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `torch.set_num_threads(1)`. Multi-thread is a separate row in Phase 4, not the default comparison.
- Pin CPU governor to `performance` if on Linux.
- `gc.collect(); gc.disable()` around timed loops.
- `time.perf_counter_ns()`, never `time.time()`.
- 10 warmup, 100 timed iterations.
- Pre-decode all PIL images into RAM before the timed section. Preprocessing benchmarks should never include disk I/O.
- Pin versions: `transformers`, `torch`, `torchvision`, `Pillow`, `numpy`. Record exact versions in every result file.
- Record hardware: CPU model, core count, RAM, OS.

## Falsification conditions (write down now, do not loosen later)

These bind the project regardless of how much work has already been done. The point of writing them before profiling is that motivated reasoning afterward can't move them.

1. If Phase 0 shows Fast median < 5 ms/image, stop.
2. If Phase 1 profile shows > 85% time in compiled C kernels, stop or pivot.
3. If Phase 3 shows < 1.5× headroom across all workloads, stop.
4. If Phase 3 shows stable profiles across workloads, the DSL framing is wrong even if there's headroom — reframe before continuing.

## Cumulative time budget

| Phase | Effort | Cumulative |
|-------|--------|------------|
| 0 | 0.5 hr | 0.5 hr |
| 1 | 4 hr | 4.5 hr |
| 2 | 4 hr | 8.5 hr |
| 3 | 4 hr | 12.5 hr |
| 4 | 1-2 days | ~3 days |

Phases 0-3 fit in under two days of focused work. The full baseline lands at roughly three days. The original "do everything at once" spec was a 4-day plan with the decision point at the end; this version puts it at hour 0.5, then again at half-day, then again at day 1.5.