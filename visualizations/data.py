from dataclasses import dataclass
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = ("W2", "W3", "W4")


@dataclass(frozen=True)
class BenchmarkRow:
    workload: str
    model: str
    median_ms_batch: float | None
    median_ms_img: float | None
    p95_minus_p50_ms: float | None
    output_mb: float | None
    peak_rss_mb: float | None
    peak_output: float | None


def _parse_numeric_cell(cell: str) -> float | None:
    cleaned = cell.strip().replace("**", "")
    if cleaned.upper() == "N/A":
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned.replace(",", ""))
    if match is None:
        return None
    return float(match.group(0))


def load_markdown_results(relative_path: str) -> dict[str, dict[str, BenchmarkRow]]:
    rows: dict[str, dict[str, BenchmarkRow]] = {}
    current_workload: str | None = None
    path = REPO_ROOT / relative_path

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line in {"# W2", "# W3", "# W4"}:
            current_workload = line.removeprefix("# ")
            rows[current_workload] = {}
            continue
        if current_workload is None or not line.startswith("|"):
            continue
        if line.startswith("|---") or line.startswith("| Model "):
            continue

        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 7:
            continue

        model = cells[0]
        row = BenchmarkRow(
            workload=current_workload,
            model=model,
            median_ms_batch=_parse_numeric_cell(cells[1]),
            median_ms_img=_parse_numeric_cell(cells[2]),
            p95_minus_p50_ms=_parse_numeric_cell(cells[3]),
            output_mb=_parse_numeric_cell(cells[4]),
            peak_rss_mb=_parse_numeric_cell(cells[5]),
            peak_output=_parse_numeric_cell(cells[6]),
        )
        rows[current_workload][model] = row

    return rows


def _must_get(
    results: dict[str, dict[str, BenchmarkRow]],
    workload: str,
    model: str,
) -> BenchmarkRow:
    try:
        return results[workload][model]
    except KeyError as exc:
        raise KeyError(f"Missing {model!r} for {workload!r}") from exc


def _batch_ms(row: BenchmarkRow) -> float:
    if row.median_ms_batch is None:
        raise ValueError(f"Missing batch runtime for {row.model} {row.workload}")
    return row.median_ms_batch


def _img_ms(row: BenchmarkRow) -> float:
    if row.median_ms_img is None:
        raise ValueError(f"Missing image runtime for {row.model} {row.workload}")
    return row.median_ms_img


def _peak_output(row: BenchmarkRow) -> float:
    if row.peak_output is None:
        raise ValueError(f"Missing peak/output ratio for {row.model} {row.workload}")
    return row.peak_output


def _get_qwen_headline(relative_path: str) -> dict[str, object]:
    results = load_markdown_results(relative_path)
    model_order = ["hf_legacy", "hf_fast", "hf_bilinear", "v3", "dsl_v3"]
    labels = {
        "hf_legacy": "HF Legacy",
        "hf_fast": "HF Fast",
        "hf_bilinear": "HF Bilinear",
        "v3": "Hand v3",
        "dsl_v3": "DSL v3",
    }
    values: dict[str, list[float]] = {labels[model]: [] for model in model_order}

    for workload in WORKLOADS:
        baseline = _batch_ms(_must_get(results, workload, "hf_fast"))
        for model in model_order:
            values[labels[model]].append(baseline / _batch_ms(_must_get(results, workload, model)))

    return {"workloads": WORKLOADS, "series": values}


def get_qwen_singlethread_headline() -> dict[str, object]:
    return _get_qwen_headline("results/aws/bench_dsl_results_single_thread.md")


def get_qwen_multithread_headline() -> dict[str, object]:
    return _get_qwen_headline("results/aws/bench_dsl_results_multi_thread.md")


def get_qwen_thread_scaling() -> dict[str, object]:
    single = load_markdown_results("results/aws/bench_dsl_results_single_thread.md")
    multi = load_markdown_results("results/aws/bench_dsl_results_multi_thread.md")
    model_order = ["hf_fast", "v3", "dsl_v3"]
    labels = {"hf_fast": "HF Fast", "v3": "Hand v3", "dsl_v3": "DSL v3"}
    values: dict[str, list[float]] = {labels[model]: [] for model in model_order}

    for workload in WORKLOADS:
        for model in model_order:
            single_ms = _batch_ms(_must_get(single, workload, model))
            multi_ms = _batch_ms(_must_get(multi, workload, model))
            values[labels[model]].append(single_ms / multi_ms)

    return {"workloads": WORKLOADS, "series": values}


def get_qwen_dsl_schedule_ablation() -> dict[str, object]:
    files = {
        "Single thread": "results/aws/bench_dsl_results_single_thread.md",
        "8 threads": "results/aws/bench_dsl_results_multi_thread.md",
    }
    model_order = ["dsl_v1", "dsl_v2", "dsl_v3"]
    labels = {"dsl_v1": "DSL v1", "dsl_v2": "DSL v2", "dsl_v3": "DSL v3"}
    panels: dict[str, dict[str, list[float]]] = {}

    for panel, relative_path in files.items():
        results = load_markdown_results(relative_path)
        panel_values: dict[str, list[float]] = {labels[model]: [] for model in model_order}
        for workload in WORKLOADS:
            baseline = _batch_ms(_must_get(results, workload, "dsl_v1"))
            for model in model_order:
                panel_values[labels[model]].append(baseline / _batch_ms(_must_get(results, workload, model)))
        panels[panel] = panel_values

    return {"workloads": WORKLOADS, "panels": panels}


def get_qwen_dsl_schedule_ablation_single_thread() -> dict[str, object]:
    data = get_qwen_dsl_schedule_ablation()
    return {"workloads": data["workloads"], "series": data["panels"]["Single thread"]}


def get_qwen_dsl_schedule_ablation_multi_thread() -> dict[str, object]:
    data = get_qwen_dsl_schedule_ablation()
    return {"workloads": data["workloads"], "series": data["panels"]["8 threads"]}


def get_qwen_runtime_memory_pareto() -> dict[str, list[dict[str, float | str]]]:
    results = load_markdown_results("results/aws/bench_dsl_results_multi_thread.md")
    model_order = ["hf_legacy", "hf_fast", "hf_bilinear", "dsl_v1", "dsl_v2", "v3", "dsl_v3"]
    labels = {
        "hf_legacy": "HF Legacy",
        "hf_fast": "HF Fast",
        "hf_bilinear": "HF Bilinear",
        "dsl_v1": "DSL v1",
        "dsl_v2": "DSL v2",
        "v3": "Hand v3",
        "dsl_v3": "DSL v3",
    }
    points: dict[str, list[dict[str, float | str]]] = {}

    for workload in WORKLOADS:
        points[workload] = []
        for model in model_order:
            row = _must_get(results, workload, model)
            points[workload].append(
                {
                    "label": labels[model],
                    "runtime": _img_ms(row),
                    "memory": _peak_output(row),
                    "model": model,
                }
            )

    return points


def get_hf_fast_memory_heatmap() -> dict[str, object]:
    results = load_markdown_results("results/aws/full_benchmarks_multi_thread_results.md")
    models = ["Qwen Fast", "InternVL3.5 Fast", "LLaVA Fast"]
    model_labels = ["Qwen2.5-VL", "InternVL3.5", "LLaVA-NeXT"]
    values = [
        [_peak_output(_must_get(results, workload, model)) for workload in WORKLOADS]
        for model in models
    ]
    return {"workloads": WORKLOADS, "models": model_labels, "values": values}


def _full_benchmark_path(thread_label: str) -> str:
    if thread_label == "single":
        return "results/aws/full_benchmarks_single_thread_results.md"
    if thread_label == "multi":
        return "results/aws/full_benchmarks_multi_thread_results.md"
    raise ValueError(f"Unknown thread label: {thread_label}")


def get_hf_fast_pipeline_runtime(thread_label: str = "multi") -> dict[str, object]:
    results = load_markdown_results(_full_benchmark_path(thread_label))
    models = ["Qwen Fast", "InternVL3.5 Fast", "LLaVA Fast"]
    model_labels = ["Qwen2.5-VL Fast", "InternVL3.5 Fast", "LLaVA-NeXT Fast"]
    values = {
        label: [_img_ms(_must_get(results, workload, model)) for workload in WORKLOADS]
        for model, label in zip(models, model_labels)
    }
    return {"workloads": WORKLOADS, "series": values}


def get_hf_fast_pipeline_memory(thread_label: str = "multi") -> dict[str, object]:
    results = load_markdown_results(_full_benchmark_path(thread_label))
    models = ["Qwen Fast", "InternVL3.5 Fast", "LLaVA Fast"]
    model_labels = ["Qwen2.5-VL Fast", "InternVL3.5 Fast", "LLaVA-NeXT Fast"]
    values = {
        label: [_peak_output(_must_get(results, workload, model)) for workload in WORKLOADS]
        for model, label in zip(models, model_labels)
    }
    return {"workloads": WORKLOADS, "series": values}


def get_legacy_fast_thread_comparison(workload: str, metric: str = "runtime") -> dict[str, object]:
    single = load_markdown_results("results/aws/full_benchmarks_single_thread_results.md")
    multi = load_markdown_results("results/aws/full_benchmarks_multi_thread_results.md")
    models = {
        "Qwen2.5-VL": ("Qwen Legacy", "Qwen Fast"),
        "InternVL3.5": ("InternVL3.5 Legacy", "InternVL3.5 Fast"),
        "LLaVA-NeXT": ("LLaVA Legacy", "LLaVA Fast"),
    }
    series = {
        "Legacy 1T": [],
        "Fast 1T": [],
        "Legacy 8T": [],
        "Fast 8T": [],
    }

    for legacy_name, fast_name in models.values():
        rows = {
            "Legacy 1T": _must_get(single, workload, legacy_name),
            "Fast 1T": _must_get(single, workload, fast_name),
            "Legacy 8T": _must_get(multi, workload, legacy_name),
            "Fast 8T": _must_get(multi, workload, fast_name),
        }
        for label, row in rows.items():
            if metric == "runtime":
                series[label].append(_img_ms(row))
            elif metric == "memory":
                series[label].append(_peak_output(row))
            else:
                raise ValueError(f"Unknown metric: {metric}")

    return {"workload": workload, "models": tuple(models.keys()), "series": series}


def get_hf_fast_thread_scaling_heatmap() -> dict[str, object]:
    single = load_markdown_results("results/aws/full_benchmarks_single_thread_results.md")
    multi = load_markdown_results("results/aws/full_benchmarks_multi_thread_results.md")
    models = ["Qwen Fast", "InternVL3.5 Fast", "LLaVA Fast"]
    model_labels = ["Qwen2.5-VL", "InternVL3.5", "LLaVA-NeXT"]
    values = []

    for model in models:
        row_values = []
        for workload in WORKLOADS:
            single_ms = _batch_ms(_must_get(single, workload, model))
            multi_ms = _batch_ms(_must_get(multi, workload, model))
            row_values.append(single_ms / multi_ms)
        values.append(row_values)

    return {"workloads": WORKLOADS, "models": model_labels, "values": values}
