"""Streamlit dashboard helpers for retrieval evaluation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from eu_taxonomy_rag.evaluation.golden_dataset import DEFAULT_OUTPUT_PATH as GOLDEN_RAW_PATH
from eu_taxonomy_rag.evaluation.golden_dataset_validator import DEFAULT_OUTPUT_PATH as GOLDEN_CLEANED_PATH
from eu_taxonomy_rag.evaluation.metrics import EvaluationReport, RetrievalMetrics
from eu_taxonomy_rag.evaluation.natural_dataset import NATURAL_DATASET_748_PATH
from eu_taxonomy_rag.evaluation.runner import (
    ALL_RETRIEVAL_METHODS,
    DEFAULT_RESULTS_DIR,
    EvaluationRunResult,
    run_retrieval_evaluation,
)
from eu_taxonomy_rag.retrieval.retrieval_methods import RetrievalMethod

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


GOLDEN_CLEANED = _resolve(GOLDEN_CLEANED_PATH)
GOLDEN_RAW = _resolve(GOLDEN_RAW_PATH)
NATURAL_748 = _resolve(NATURAL_DATASET_748_PATH)
RESULTS_DIR = _resolve(DEFAULT_RESULTS_DIR)

METHOD_LABELS: dict[str, str] = {
    "bm25": "BM25",
    "dense_minilm": "Dense · MiniLM",
    "dense_mpnet": "Dense · MPNet",
    "hybrid_minilm": "Hybrid · MiniLM",
    "hybrid_mpnet": "Hybrid · MPNet",
}

METRIC_COLUMNS: tuple[tuple[str, str], ...] = (
    ("recall@1", "Recall@1"),
    ("recall@3", "Recall@3"),
    ("recall@5", "Recall@5"),
    ("mrr", "MRR"),
)


@dataclass(frozen=True)
class EvalDatasetSpec:
    key: str
    label: str
    path: Path
    description: str
    has_metadata: bool = False

    @property
    def exists(self) -> bool:
        return self.path.exists()


AVAILABLE_DATASETS: tuple[EvalDatasetSpec, ...] = (
    EvalDatasetSpec(
        key="golden_cleaned",
        label="Golden (cleaned)",
        path=GOLDEN_CLEANED,
        description="700 rule-based questions (500 simple + 200 complex).",
    ),
    EvalDatasetSpec(
        key="golden_raw",
        label="Golden (raw)",
        path=GOLDEN_RAW,
        description="700 questions before complex-query cleanup.",
    ),
    EvalDatasetSpec(
        key="natural_748",
        label="Natural 748",
        path=NATURAL_748,
        description="648 simple + 100 complex, varied personas.",
        has_metadata=True,
    ),
)


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def get_dataset_spec(key: str) -> EvalDatasetSpec:
    for spec in AVAILABLE_DATASETS:
        if spec.key == key:
            return spec
    raise KeyError(f"Unknown dataset key: {key}")


def run_multi_dataset_evaluation(
    dataset_specs: list[EvalDatasetSpec],
    methods: list[RetrievalMethod],
    *,
    k: int = 5,
    candidate_k: int = 20,
    index_dir: str | Path = ".cache/index",
    limit: int | None = None,
) -> dict[str, EvaluationRunResult]:
    """Exécute le benchmark sur plusieurs datasets."""
    results: dict[str, EvaluationRunResult] = {}
    for spec in dataset_specs:
        if not spec.path.exists():
            continue
        results[spec.key] = run_retrieval_evaluation(
            dataset_path=spec.path,
            methods=methods,
            k=k,
            candidate_k=candidate_k,
            index_dir=index_dir,
            limit=limit,
            build_indexes=len(results) == 0,
        )
    return results


def metrics_to_row(
    dataset_key: str,
    dataset_label: str,
    method: str,
    metrics: RetrievalMetrics,
    *,
    segment: str = "overall",
) -> dict[str, Any]:
    return {
        "dataset_key": dataset_key,
        "dataset": dataset_label,
        "method": method,
        "method_label": method_label(method),
        "segment": segment,
        "recall@1": metrics.recall_at_1,
        "recall@3": metrics.recall_at_3,
        "recall@5": metrics.recall_at_5,
        "mrr": metrics.mrr,
        "num_queries": metrics.num_queries,
    }


def build_overall_comparison_df(results: dict[str, EvaluationRunResult]) -> pd.DataFrame:
    """Tableau method × dataset avec métriques globales."""
    rows: list[dict[str, Any]] = []
    spec_by_key = {spec.key: spec for spec in AVAILABLE_DATASETS}

    for dataset_key, run_result in results.items():
        label = spec_by_key.get(dataset_key, EvalDatasetSpec(dataset_key, dataset_key, Path(run_result.dataset_path), "")).label
        for method_name, report in run_result.methods.items():
            rows.append(metrics_to_row(dataset_key, label, method_name, report.overall))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["method_label"] = pd.Categorical(
        df["method_label"],
        categories=[method_label(method.value) for method in ALL_RETRIEVAL_METHODS],
        ordered=True,
    )
    return df.sort_values(["dataset", "method_label"]).reset_index(drop=True)


def collect_segment_values(
    results: dict[str, EvaluationRunResult],
    segment_field: str,
) -> list[str]:
    """Valeurs uniques d'un segment (difficulty, persona, …) dans les résultats chargés."""
    values: set[str] = set()
    attr = f"by_{segment_field}"
    for run_result in results.values():
        for report in run_result.methods.values():
            segments: dict[str, RetrievalMetrics] = getattr(report, attr, {})
            values.update(segments.keys())
    return sorted(values)


def build_segment_filtered_comparison_df(
    results: dict[str, EvaluationRunResult],
    *,
    difficulties: list[str] | None = None,
    personas: list[str] | None = None,
) -> pd.DataFrame:
    """Tableau method × dataset à partir des métriques globales ou ventilées."""
    if personas:
        return _build_segment_subset_df(results, "persona", personas)
    if difficulties:
        return _build_segment_subset_df(results, "difficulty", difficulties)
    return build_overall_comparison_df(results)


def _build_segment_subset_df(
    results: dict[str, EvaluationRunResult],
    segment_field: str,
    segment_values: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    spec_by_key = {spec.key: spec for spec in AVAILABLE_DATASETS}
    attr = f"by_{segment_field}"

    for dataset_key, run_result in results.items():
        label = spec_by_key.get(
            dataset_key,
            EvalDatasetSpec(dataset_key, dataset_key, Path(run_result.dataset_path), ""),
        ).label
        for method_name, report in run_result.methods.items():
            segments: dict[str, RetrievalMetrics] = getattr(report, attr, {})
            for seg_name in segment_values:
                metrics = segments.get(seg_name)
                if metrics is None:
                    continue
                rows.append(
                    {
                        **metrics_to_row(
                            dataset_key,
                            f"{label} · {seg_name}",
                            method_name,
                            metrics,
                            segment=f"{segment_field}:{seg_name}",
                        ),
                        "segment_label": seg_name,
                        "segment_field": segment_field,
                    }
                )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["method_label"] = pd.Categorical(
        df["method_label"],
        categories=[method_label(method.value) for method in ALL_RETRIEVAL_METHODS],
        ordered=True,
    )
    return df.sort_values(["dataset", "method_label", "segment_label"]).reset_index(drop=True)


def build_segment_comparison_df(
    results: dict[str, EvaluationRunResult],
    segment_field: str,
) -> pd.DataFrame:
    """Tableau ventilé (difficulty, persona, query_type)."""
    rows: list[dict[str, Any]] = []
    spec_by_key = {spec.key: spec for spec in AVAILABLE_DATASETS}

    for dataset_key, run_result in results.items():
        label = spec_by_key.get(dataset_key, EvalDatasetSpec(dataset_key, dataset_key, Path(run_result.dataset_path), "")).label
        for method_name, report in run_result.methods.items():
            segments: dict[str, RetrievalMetrics] = getattr(report, f"by_{segment_field}", {})
            for segment_name, metrics in segments.items():
                rows.append(
                    metrics_to_row(
                        dataset_key,
                        label,
                        method_name,
                        metrics,
                        segment=f"{segment_field}:{segment_name}",
                    )
                )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def pivot_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pivot method × dataset pour une métrique."""
    if df.empty:
        return pd.DataFrame()
    pivot = df.pivot_table(
        index="method_label",
        columns="dataset",
        values=metric,
        aggfunc="first",
    )
    return pivot.sort_index()


def list_saved_results(results_dir: str | Path = RESULTS_DIR) -> list[Path]:
    path = _resolve(Path(results_dir))
    if not path.exists():
        return []
    return sorted(path.glob("retrieval_eval_*.json"), reverse=True)


def saved_result_to_df(payload: dict[str, Any]) -> pd.DataFrame:
    """Convertit un JSON exporté en DataFrame comparable."""
    dataset_path = Path(payload["dataset_path"])
    dataset_key = dataset_path.stem
    rows: list[dict[str, Any]] = []

    for method_name, report in payload.get("methods", {}).items():
        overall = report.get("overall", {})
        rows.append(
            {
                "dataset_key": dataset_key,
                "dataset": dataset_key,
                "method": method_name,
                "method_label": method_label(method_name),
                "segment": "overall",
                "recall@1": overall.get("recall@1", 0.0),
                "recall@3": overall.get("recall@3", 0.0),
                "recall@5": overall.get("recall@5", 0.0),
                "mrr": overall.get("mrr", 0.0),
                "num_queries": overall.get("num_queries", payload.get("num_queries", 0)),
                "generated_at": payload.get("generated_at"),
            }
        )

    return pd.DataFrame(rows)


def load_saved_result(path: str | Path) -> dict[str, Any]:
    import json

    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def latest_saved_result_path(
    dataset_stem: str,
    results_dir: str | Path = RESULTS_DIR,
) -> Path | None:
    """Most recent JSON export for a dataset stem."""
    directory = _resolve(Path(results_dir))
    matches = list(directory.glob(f"retrieval_eval_{dataset_stem}_*.json"))
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def _metrics_from_dict(data: dict[str, Any]) -> RetrievalMetrics:
    return RetrievalMetrics(
        recall_at_1=float(data["recall@1"]),
        recall_at_3=float(data["recall@3"]),
        recall_at_5=float(data["recall@5"]),
        mrr=float(data["mrr"]),
        num_queries=int(data.get("num_queries", 0)),
    )


def evaluation_report_from_dict(data: dict[str, Any]) -> EvaluationReport:
    return EvaluationReport(
        overall=_metrics_from_dict(data["overall"]),
        by_difficulty={
            key: _metrics_from_dict(value) for key, value in data.get("by_difficulty", {}).items()
        },
        by_persona={key: _metrics_from_dict(value) for key, value in data.get("by_persona", {}).items()},
        by_query_type={
            key: _metrics_from_dict(value) for key, value in data.get("by_query_type", {}).items()
        },
    )


def run_result_from_saved_payload(payload: dict[str, Any]) -> EvaluationRunResult:
    return EvaluationRunResult(
        dataset_path=payload["dataset_path"],
        num_queries=int(payload["num_queries"]),
        retrieval_k=int(payload["retrieval_k"]),
        candidate_k=int(payload["candidate_k"]),
        index_dir=str(payload.get("index_dir", "")),
        generated_at=str(payload.get("generated_at", "")),
        methods={
            method_name: evaluation_report_from_dict(report)
            for method_name, report in payload.get("methods", {}).items()
        },
    )


def load_latest_benchmark_results(
    specs: list[EvalDatasetSpec],
) -> tuple[dict[str, EvaluationRunResult], dict[str, Path]]:
    """Load the newest saved JSON per dataset."""
    results: dict[str, EvaluationRunResult] = {}
    sources: dict[str, Path] = {}
    for spec in specs:
        path = latest_saved_result_path(spec.path.stem)
        if path is None:
            continue
        payload = load_saved_result(path)
        results[spec.key] = run_result_from_saved_payload(payload)
        sources[spec.key] = path
    return results, sources


def filter_benchmark_results(
    results: dict[str, EvaluationRunResult],
    *,
    dataset_keys: list[str],
    method_values: list[str],
) -> dict[str, EvaluationRunResult]:
    """Keep only selected datasets and retrieval methods."""
    filtered: dict[str, EvaluationRunResult] = {}
    for dataset_key in dataset_keys:
        run_result = results.get(dataset_key)
        if run_result is None:
            continue
        methods = {
            method_name: report
            for method_name, report in run_result.methods.items()
            if method_name in method_values
        }
        if not methods:
            continue
        filtered[dataset_key] = EvaluationRunResult(
            dataset_path=run_result.dataset_path,
            num_queries=run_result.num_queries,
            retrieval_k=run_result.retrieval_k,
            candidate_k=run_result.candidate_k,
            index_dir=run_result.index_dir,
            methods=methods,
            generated_at=run_result.generated_at,
        )
    return filtered


def build_benchmark_view_df(
    results: dict[str, EvaluationRunResult],
    *,
    dataset_keys: list[str],
    method_values: list[str],
) -> pd.DataFrame:
    """Build the overview table for the current dataset/method filters."""
    filtered = filter_benchmark_results(
        results,
        dataset_keys=dataset_keys,
        method_values=method_values,
    )
    return build_overall_comparison_df(filtered)
