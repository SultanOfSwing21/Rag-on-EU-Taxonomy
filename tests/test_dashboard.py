import json
from pathlib import Path

import pandas as pd
import pytest

from eu_taxonomy_rag.evaluation.dashboard import (
    AVAILABLE_DATASETS,
    build_overall_comparison_df,
    build_segment_comparison_df,
    build_segment_filtered_comparison_df,
    collect_segment_values,
    filter_benchmark_results,
    latest_saved_result_path,
    load_latest_benchmark_results,
    method_label,
    pivot_metric,
    run_result_from_saved_payload,
    saved_result_to_df,
)
from eu_taxonomy_rag.evaluation.metrics import EvaluationReport, RetrievalMetrics
from eu_taxonomy_rag.evaluation.runner import EvaluationRunResult


def test_method_labels() -> None:
    assert method_label("hybrid_minilm") == "Hybrid · MiniLM"
    assert method_label("unknown") == "unknown"


def test_build_overall_comparison_df() -> None:
    metrics = RetrievalMetrics(0.8, 0.9, 0.95, 0.75, num_queries=10)
    report = EvaluationReport(
        overall=metrics,
        by_difficulty={"simple": metrics},
        by_persona={},
        by_query_type={},
    )
    run_result = EvaluationRunResult(
        dataset_path="data/evaluation/retrieval_golden_dataset_cleaned.jsonl",
        num_queries=10,
        retrieval_k=5,
        candidate_k=20,
        index_dir=".cache/index",
        methods={"bm25": report, "hybrid_minilm": report},
    )

    df = build_overall_comparison_df({"golden_cleaned": run_result})

    assert len(df) == 2
    assert set(df["method"]) == {"bm25", "hybrid_minilm"}
    assert df["recall@1"].iloc[0] == pytest.approx(0.8)


def test_build_segment_comparison_df() -> None:
    simple = RetrievalMetrics(1.0, 1.0, 1.0, 1.0, num_queries=8)
    complex_ = RetrievalMetrics(0.5, 0.7, 0.8, 0.4, num_queries=2)
    report = EvaluationReport(
        overall=simple,
        by_difficulty={"simple": simple, "complex": complex_},
        by_persona={"CFO": simple},
        by_query_type={"natural_simple": simple},
    )
    run_result = EvaluationRunResult(
        dataset_path="data/evaluation/natural_user_queries_748.jsonl",
        num_queries=10,
        retrieval_k=5,
        candidate_k=20,
        index_dir=".cache/index",
        methods={"bm25": report},
    )

    difficulty_df = build_segment_comparison_df({"natural_748": run_result}, "difficulty")
    assert "segment" in difficulty_df.columns
    assert difficulty_df["segment"].str.contains("difficulty:").any()

    persona_df = build_segment_comparison_df({"natural_748": run_result}, "persona")
    assert (persona_df["segment"] == "persona:CFO").any()


def test_collect_segment_values() -> None:
    simple = RetrievalMetrics(1.0, 1.0, 1.0, 1.0, num_queries=8)
    complex_ = RetrievalMetrics(0.5, 0.7, 0.8, 0.4, num_queries=2)
    report = EvaluationReport(
        overall=simple,
        by_difficulty={"simple": simple, "complex": complex_},
        by_persona={"CFO": simple, "auditor": complex_},
        by_query_type={},
    )
    run_result = EvaluationRunResult(
        dataset_path="data/evaluation/natural_user_queries_748.jsonl",
        num_queries=10,
        retrieval_k=5,
        candidate_k=20,
        index_dir=".cache/index",
        methods={"bm25": report},
    )

    assert collect_segment_values({"natural_748": run_result}, "difficulty") == ["complex", "simple"]
    assert collect_segment_values({"natural_748": run_result}, "persona") == ["CFO", "auditor"]


def test_build_segment_filtered_comparison_df() -> None:
    simple = RetrievalMetrics(1.0, 1.0, 1.0, 1.0, num_queries=8)
    complex_ = RetrievalMetrics(0.5, 0.7, 0.8, 0.4, num_queries=2)
    report = EvaluationReport(
        overall=simple,
        by_difficulty={"simple": simple, "complex": complex_},
        by_persona={"CFO": simple},
        by_query_type={},
    )
    run_result = EvaluationRunResult(
        dataset_path="data/evaluation/natural_user_queries_748.jsonl",
        num_queries=10,
        retrieval_k=5,
        candidate_k=20,
        index_dir=".cache/index",
        methods={"bm25": report},
    )
    cache = {"natural_748": run_result}

    overall_df = build_segment_filtered_comparison_df(cache)
    assert len(overall_df) == 1
    assert "segment_label" not in overall_df.columns

    difficulty_df = build_segment_filtered_comparison_df(cache, difficulties=["complex"])
    assert len(difficulty_df) == 1
    assert difficulty_df["segment_label"].iloc[0] == "complex"
    assert difficulty_df["recall@1"].iloc[0] == pytest.approx(0.5)

    persona_df = build_segment_filtered_comparison_df(cache, personas=["CFO"])
    assert len(persona_df) == 1
    assert persona_df["segment_label"].iloc[0] == "CFO"


def test_pivot_metric() -> None:
    df = pd.DataFrame(
        [
            {"method_label": "BM25", "dataset": "Golden", "mrr": 0.8},
            {"method_label": "Hybrid · MiniLM", "dataset": "Golden", "mrr": 0.9},
            {"method_label": "BM25", "dataset": "Natural", "mrr": 0.7},
        ]
    )
    pivot = pivot_metric(df, "mrr")

    assert pivot.loc["BM25", "Golden"] == pytest.approx(0.8)
    assert "Natural" in pivot.columns


def test_saved_result_to_df(tmp_path: Path) -> None:
    payload = {
        "dataset_path": "data/evaluation/retrieval_golden_dataset_cleaned.jsonl",
        "num_queries": 5,
        "generated_at": "2026-01-01T00:00:00+00:00",
        "methods": {
            "bm25": {
                "overall": {
                    "recall@1": 0.6,
                    "recall@3": 0.8,
                    "recall@5": 0.9,
                    "mrr": 0.55,
                    "num_queries": 5,
                }
            }
        },
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    df = saved_result_to_df(json.loads(path.read_text(encoding="utf-8")))

    assert len(df) == 1
    assert df["mrr"].iloc[0] == pytest.approx(0.55)


def test_available_datasets_paths_exist_for_core_files() -> None:
    keys = {spec.key for spec in AVAILABLE_DATASETS}
    assert "golden_cleaned" in keys
    assert "natural_748" in keys


def test_run_result_from_saved_payload_roundtrip() -> None:
    payload = {
        "dataset_path": "data/evaluation/retrieval_golden_dataset_cleaned.jsonl",
        "num_queries": 2,
        "retrieval_k": 5,
        "candidate_k": 20,
        "index_dir": ".cache/index",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "methods": {
            "bm25": {
                "overall": {
                    "recall@1": 0.5,
                    "recall@3": 0.5,
                    "recall@5": 1.0,
                    "mrr": 0.5,
                    "num_queries": 2,
                },
                "by_difficulty": {},
                "by_persona": {},
                "by_query_type": {},
            }
        },
    }

    run_result = run_result_from_saved_payload(payload)
    assert run_result.num_queries == 2
    assert run_result.methods["bm25"].overall.mrr == pytest.approx(0.5)


def test_latest_saved_result_path_prefers_newest(tmp_path: Path) -> None:
    older = tmp_path / "retrieval_eval_retrieval_golden_dataset_cleaned_20260101T000000Z.json"
    newer = tmp_path / "retrieval_eval_retrieval_golden_dataset_cleaned_20260201T000000Z.json"
    older.write_text("{}", encoding="utf-8")
    newer.write_text("{}", encoding="utf-8")

    path = latest_saved_result_path("retrieval_golden_dataset_cleaned", results_dir=tmp_path)
    assert path == newer


def test_filter_benchmark_results_by_dataset_and_method() -> None:
    metrics = RetrievalMetrics(0.8, 0.9, 0.95, 0.75, num_queries=10)
    report = EvaluationReport(overall=metrics, by_difficulty={}, by_persona={}, by_query_type={})
    golden = EvaluationRunResult(
        dataset_path="data/evaluation/retrieval_golden_dataset_cleaned.jsonl",
        num_queries=10,
        retrieval_k=5,
        candidate_k=20,
        index_dir=".cache/index",
        methods={"bm25": report, "hybrid_minilm": report},
    )
    natural = EvaluationRunResult(
        dataset_path="data/evaluation/natural_user_queries_748.jsonl",
        num_queries=10,
        retrieval_k=5,
        candidate_k=20,
        index_dir=".cache/index",
        methods={"dense_minilm": report},
    )
    cache = {"golden_cleaned": golden, "natural_748": natural}

    filtered = filter_benchmark_results(
        cache,
        dataset_keys=["golden_cleaned"],
        method_values=["bm25"],
    )

    assert set(filtered) == {"golden_cleaned"}
    assert set(filtered["golden_cleaned"].methods) == {"bm25"}

