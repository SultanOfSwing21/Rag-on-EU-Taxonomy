"""Exécution du benchmark retrieval sur un dataset d'évaluation."""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from eu_taxonomy_rag.cache.chunk_cache import load_or_build_chunks
from eu_taxonomy_rag.core.models import Chunk
from eu_taxonomy_rag.evaluation.golden_dataset import GoldenQuestion, load_golden_dataset
from eu_taxonomy_rag.evaluation.golden_dataset_validator import DEFAULT_OUTPUT_PATH as GOLDEN_CLEANED_PATH
from eu_taxonomy_rag.evaluation.metrics import (
    EvaluationReport,
    compute_evaluation_report,
)
from eu_taxonomy_rag.evaluation.natural_dataset import NATURAL_DATASET_748_PATH, load_natural_dataset
from eu_taxonomy_rag.pipelines.index_manager import (
    DEFAULT_INDEX_DIR,
    build_bm25_index,
    build_dense_vector_index,
    required_dense_keys_and_bm25,
    search_batch,
)
from eu_taxonomy_rag.retrieval.embeddings import release_embedding_models
from eu_taxonomy_rag.retrieval.retrieval_methods import (
    DENSE_MODELS,
    RetrievalMethod,
    dense_key_for_method,
    requires_sentence_transformers,
)

DEFAULT_RESULTS_DIR = Path("data/evaluation/results")
DEFAULT_DATASET_PATH = GOLDEN_CLEANED_PATH
ALL_RETRIEVAL_METHODS: tuple[RetrievalMethod, ...] = tuple(RetrievalMethod)

ProgressCallback = Callable[[str, RetrievalMethod, int, int], None]
IndexBuildProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True)
class MethodEvaluationResult:
    method: RetrievalMethod
    report: EvaluationReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method.value,
            "metrics": self.report.to_dict(),
        }


@dataclass
class EvaluationRunResult:
    dataset_path: str
    num_queries: int
    retrieval_k: int
    candidate_k: int
    index_dir: str
    methods: dict[str, EvaluationReport] = field(default_factory=dict)
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "num_queries": self.num_queries,
            "retrieval_k": self.retrieval_k,
            "candidate_k": self.candidate_k,
            "index_dir": self.index_dir,
            "generated_at": self.generated_at,
            "methods": {
                method_name: report.to_dict()
                for method_name, report in self.methods.items()
            },
        }

    def save(self, output_path: str | Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=2, ensure_ascii=False)
            file.write("\n")
        return output_path


def load_eval_dataset(path: str | Path) -> list[GoldenQuestion | Any]:
    """Charge un dataset JSONL golden ou natural (avec métadonnées)."""
    path = Path(path)
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            break
        else:
            return []

    if "query_type" in sample or "persona" in sample:
        return load_natural_dataset(path)
    return load_golden_dataset(path)


def build_indexes_for_methods(
    chunks: list[Chunk],
    methods: list[RetrievalMethod],
    *,
    base_dir: str | Path = DEFAULT_INDEX_DIR,
    force_rebuild: bool = False,
    progress_callback: IndexBuildProgressCallback | None = None,
) -> None:
    """Construit uniquement les index nécessaires aux méthodes demandées."""
    dense_keys, needs_bm25 = required_dense_keys_and_bm25(methods)
    steps: list[tuple[str, str]] = [(key, f"dense ({key})") for key in dense_keys]
    if needs_bm25:
        steps.append(("bm25", "BM25"))

    for step_index, (step_key, step_label) in enumerate(steps, start=1):
        if progress_callback is not None:
            progress_callback(step_label, step_index, len(steps))

        if step_key == "bm25":
            build_bm25_index(
                chunks,
                base_dir=base_dir,
                force_rebuild=force_rebuild,
            )
        else:
            build_dense_vector_index(
                chunks,
                step_key,
                base_dir=base_dir,
                force_rebuild=force_rebuild,
            )

    release_embedding_models()


def run_method_evaluation(
    questions: list,
    chunks: list[Chunk],
    method: RetrievalMethod,
    *,
    k: int = 5,
    base_dir: str | Path = DEFAULT_INDEX_DIR,
    candidate_k: int = 20,
) -> EvaluationReport:
    """Evaluate one retrieval method on a dataset (batch query encoding)."""
    queries = [question.question for question in questions]
    batch_results = search_batch(
        method,
        chunks,
        queries,
        k=k,
        base_dir=base_dir,
        candidate_k=candidate_k,
    )
    retrieved_ids_list = [[chunk_id for chunk_id, _ in results] for results in batch_results]
    return compute_evaluation_report(questions, retrieved_ids_list)


def run_retrieval_evaluation(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    methods: list[RetrievalMethod] | None = None,
    *,
    k: int = 5,
    candidate_k: int = 20,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    limit: int | None = None,
    chunks: list[Chunk] | None = None,
    build_indexes: bool = True,
    progress_callback: ProgressCallback | None = None,
    dataset_label: str | None = None,
) -> EvaluationRunResult:
    """Exécute le benchmark retrieval et calcule Recall@K + MRR."""
    dataset_path = Path(dataset_path)
    index_dir = Path(index_dir)
    selected_methods = methods or list(ALL_RETRIEVAL_METHODS)

    questions = load_eval_dataset(dataset_path)
    if limit is not None:
        questions = questions[:limit]

    loaded_chunks = chunks or load_or_build_chunks()
    if build_indexes:
        build_indexes_for_methods(
            loaded_chunks,
            selected_methods,
            base_dir=index_dir,
            force_rebuild=False,
        )

    run_result = EvaluationRunResult(
        dataset_path=str(dataset_path),
        num_queries=len(questions),
        retrieval_k=k,
        candidate_k=candidate_k,
        index_dir=str(index_dir),
    )

    for method_index, method in enumerate(selected_methods, start=1):
        if progress_callback is not None:
            progress_callback(
                dataset_label or str(dataset_path),
                method,
                method_index,
                len(selected_methods),
            )
        report = run_method_evaluation(
            questions,
            loaded_chunks,
            method,
            k=k,
            base_dir=index_dir,
            candidate_k=candidate_k,
        )
        run_result.methods[method.value] = report
        if requires_sentence_transformers(method):
            dense_key = dense_key_for_method(method)
            model_name = DENSE_MODELS[dense_key].model_name
            release_embedding_models([model_name])

    return run_result


def default_output_path(
    dataset_path: str | Path,
    *,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> Path:
    """Chemin de sortie par défaut basé sur le nom du dataset."""
    dataset_stem = Path(dataset_path).stem
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(results_dir) / f"retrieval_eval_{dataset_stem}_{timestamp}.json"


def format_metrics_summary(run_result: EvaluationRunResult) -> str:
    """Résumé texte des métriques pour affichage CLI."""
    lines = [
        f"Dataset: {run_result.dataset_path} ({run_result.num_queries} queries)",
        f"Retrieval k={run_result.retrieval_k}, candidate_k={run_result.candidate_k}",
        "",
        f"{'Method':<16} {'Recall@1':>10} {'Recall@3':>10} {'Recall@5':>10} {'MRR':>10}",
        "-" * 58,
    ]

    for method_name, report in run_result.methods.items():
        metrics = report.overall
        lines.append(
            f"{method_name:<16} "
            f"{metrics.recall_at_1:>10.3f} "
            f"{metrics.recall_at_3:>10.3f} "
            f"{metrics.recall_at_5:>10.3f} "
            f"{metrics.mrr:>10.3f}"
        )

    return "\n".join(lines)
