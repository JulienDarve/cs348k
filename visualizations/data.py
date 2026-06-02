from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = ("W2", "W3", "W4")
FINAL_METHODS = ("hf_legacy", "hf_fast", "hf_bilinear", "dsl_v1", "dsl_v2", "dsl_v3")
FINAL_METHOD_LABELS = {
    "hf_legacy": "HF Legacy",
    "hf_fast": "HF Fast",
    "hf_bilinear": "HF Bilinear",
    "dsl_v1": "DSL v1",
    "dsl_v2": "DSL v2",
    "dsl_v3": "DSL v3",
}


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


def _parse_final_results_heading(line: str, current_family: str | None) -> tuple[str, str, str] | None:
    if "Runtime" in line:
        metric = "runtime"
    elif "Memory" in line:
        metric = "memory"
    else:
        return None

    if "Single-Threaded" in line:
        thread = "single"
    elif "Four-Threaded" in line:
        thread = "four"
    else:
        thread = "multi"

    if "LLaVA" in line:
        family = "llava"
    elif "Qwen" in line:
        family = "qwen"
    else:
        family = current_family
    if family is None:
        return None

    return family, thread, metric


@lru_cache(maxsize=1)
def load_final_results() -> dict[tuple[str, str, str], dict[str, dict[str, float]]]:
    """Parse FINAL_RESULTS.md tables keyed by (family, thread, metric)."""
    path = REPO_ROOT / "FINAL_RESULTS.md"
    lines = path.read_text(encoding="utf-8").splitlines()
    tables: dict[tuple[str, str, str], dict[str, dict[str, float]]] = {}
    family: str | None = None
    pending_key: tuple[str, str, str] | None = None
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if line == "## Qwen Results":
            family = "qwen"
            pending_key = None
            i += 1
            continue
        if line == "## LLaVA Results":
            family = "llava"
            pending_key = None
            i += 1
            continue
        if line.startswith("### Chart ") and family is not None:
            pending_key = _parse_final_results_heading(line, family)
            i += 1
            continue

        if pending_key is None or not line.startswith("| Workload |"):
            i += 1
            continue

        i += 2
        rows: dict[str, dict[str, float]] = {}
        while i < len(lines) and lines[i].strip().startswith("|"):
            row_line = lines[i].strip()
            if row_line.startswith("|---"):
                i += 1
                continue
            cells = [cell.strip() for cell in row_line.strip("|").split("|")]
            if len(cells) != len(FINAL_METHODS) + 1:
                break
            workload = cells[0]
            rows[workload] = {}
            for method, cell in zip(FINAL_METHODS, cells[1:]):
                value = _parse_numeric_cell(cell)
                if value is None:
                    raise ValueError(f"Missing value for {pending_key} {workload} {method}")
                rows[workload][method] = value
            i += 1

        tables[pending_key] = rows

    expected = {
        (family_name, thread, metric)
        for family_name in ("qwen", "llava")
        for thread in ("multi", "single", "four")
        for metric in ("runtime", "memory")
    }
    missing = expected.difference(tables)
    if missing:
        raise ValueError(f"Missing FINAL_RESULTS.md tables: {sorted(missing)}")
    return tables


def get_final_grouped_bar_data(family: str, thread: str, metric: str) -> dict[str, object]:
    tables = load_final_results()
    rows = tables[(family, thread, metric)]
    series = {
        FINAL_METHOD_LABELS[method]: [rows[workload][method] for workload in WORKLOADS]
        for method in FINAL_METHODS
    }
    return {
        "family": family,
        "thread": thread,
        "metric": metric,
        "workloads": WORKLOADS,
        "series": series,
    }


def get_final_selected_grouped_data(
    family: str,
    thread: str,
    metric: str,
    methods: tuple[str, ...],
) -> dict[str, object]:
    tables = load_final_results()
    rows = tables[(family, thread, metric)]
    series = {
        FINAL_METHOD_LABELS[method]: [rows[workload][method] for workload in WORKLOADS]
        for method in methods
    }
    return {
        "family": family,
        "thread": thread,
        "metric": metric,
        "workloads": WORKLOADS,
        "series": series,
    }


def get_qwen_w4_thread_scaling() -> dict[str, object]:
    tables = load_final_results()
    thread_keys = ("single", "four", "multi")
    thread_labels = ("1T", "4T", "8T")
    methods = ("hf_bilinear", "dsl_v3")
    series = {}
    speedups = {}

    for method in methods:
        label = FINAL_METHOD_LABELS[method]
        values = [
            tables[("qwen", thread, "runtime")]["W4"][method]
            for thread in thread_keys
        ]
        series[label] = values
        speedups[label] = values[0] / values[-1]

    return {
        "workload": "W4",
        "threads": thread_labels,
        "series": series,
        "speedups": speedups,
    }


def get_qwen_w4_schedule_axes() -> dict[str, object]:
    tables = load_final_results()
    variants = ("dsl_v1", "dsl_v2", "dsl_v3")
    variant_labels = ("v1", "v2", "v3")
    runtime_rows = tables[("qwen", "multi", "runtime")]["W4"]
    memory_rows = tables[("qwen", "multi", "memory")]["W4"]
    fusion_runtime = [runtime_rows[variant] for variant in variants]
    fusion_memory = [memory_rows[variant] for variant in variants]
    parallel_runtime = [
        tables[("qwen", "single", "runtime")]["W4"]["dsl_v3"],
        tables[("qwen", "multi", "runtime")]["W4"]["dsl_v3"],
    ]

    return {
        "workload": "W4",
        "variant_labels": variant_labels,
        "fusion_runtime": fusion_runtime,
        "fusion_memory": fusion_memory,
        "runtime_spread": max(fusion_runtime) / min(fusion_runtime),
        "parallel_threads": ("1T", "8T"),
        "parallel_runtime": parallel_runtime,
        "parallel_speedup": parallel_runtime[0] / parallel_runtime[-1],
    }


def get_llava_singlethread_memory_floor_summary() -> dict[str, float]:
    rows = load_final_results()[("llava", "single", "memory")]
    dsl_methods = ("dsl_v1", "dsl_v2", "dsl_v3")
    hf_methods = ("hf_legacy", "hf_fast", "hf_bilinear")
    dsl_values = [
        rows[workload][method]
        for workload in WORKLOADS
        for method in dsl_methods
    ]
    hf_values = [
        rows[workload][method]
        for workload in WORKLOADS
        for method in hf_methods
    ]
    return {
        "dsl_min": min(dsl_values),
        "dsl_max": max(dsl_values),
        "hf_min": min(hf_values),
        "hf_max": max(hf_values),
    }


def get_final_pareto_points() -> dict[tuple[str, str, str], list[dict[str, float | str]]]:
    tables = load_final_results()
    points: dict[tuple[str, str, str], list[dict[str, float | str]]] = {}

    for family in ("qwen", "llava"):
        for thread in ("multi", "single"):
            runtime_rows = tables[(family, thread, "runtime")]
            memory_rows = tables[(family, thread, "memory")]
            for workload in WORKLOADS:
                points[(family, thread, workload)] = [
                    {
                        "method": method,
                        "label": FINAL_METHOD_LABELS[method],
                        "runtime": runtime_rows[workload][method],
                        "memory": memory_rows[workload][method],
                    }
                    for method in FINAL_METHODS
                ]

    return points


def get_pareto_frontier(points: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    frontier = []
    best_memory = float("inf")
    for point in sorted(points, key=lambda item: (float(item["runtime"]), float(item["memory"]))):
        memory = float(point["memory"])
        if memory <= best_memory:
            frontier.append(point)
            best_memory = memory
    return frontier


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
    model_order = ["hf_legacy", "hf_fast", "v3", "dsl_v3"]
    labels = {"hf_legacy": "HF Legacy", "hf_fast": "HF Fast", "v3": "Hand v3", "dsl_v3": "DSL v3"}
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
