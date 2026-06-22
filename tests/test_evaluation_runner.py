import json
from pathlib import Path

import numpy as np
import pytest

from eu_taxonomy_rag.core.models import Chunk
from eu_taxonomy_rag.evaluation.runner import (
    EvaluationRunResult,
    format_metrics_summary,
    load_eval_dataset,
    run_method_evaluation,
    run_retrieval_evaluation,
)
from eu_taxonomy_rag.pipelines.index_manager import build_all_indexes, clear_index_cache
from eu_taxonomy_rag.retrieval.retrieval_methods import RetrievalMethod


@pytest.fixture(autouse=True)
def reset_caches() -> None:
    clear_index_cache()
    yield
    clear_index_cache()


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="faq-0001",
            question="What is the EU Taxonomy?",
            answer="A classification system for sustainable economic activities.",
            metadata={"section": "General", "index": 1},
        ),
        Chunk(
            chunk_id="faq-0002",
            question="Who must report Taxonomy alignment?",
            answer="Large companies under the NFRD must disclose alignment.",
            metadata={"section": "Reporting", "index": 2},
        ),
        Chunk(
            chunk_id="faq-0003",
            question="What are technical screening criteria?",
            answer="Conditions an activity must meet to be Taxonomy-aligned.",
            metadata={"section": "Climate Delegated Act", "index": 3},
        ),
    ]


@pytest.fixture
def mock_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_embeddings = np.array(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]],
        dtype=np.float32,
    )

    class FakeModel:
        def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False):
            if len(texts) == 1:
                return np.array([[1.0, 0.0]], dtype=np.float32)
            return fake_embeddings

    monkeypatch.setattr(
        "eu_taxonomy_rag.retrieval.embeddings.get_embedding_model",
        lambda model_name: FakeModel(),
    )


@pytest.fixture
def golden_dataset_file(tmp_path: Path) -> Path:
    rows = [
        {
            "question": "What is the EU Taxonomy?",
            "expected_chunk_ids": ["faq-0001"],
            "difficulty": "simple",
        },
        {
            "question": "Who must report Taxonomy alignment?",
            "expected_chunk_ids": ["faq-0002"],
            "difficulty": "simple",
        },
        {
            "question": "How do TSC and reporting relate?",
            "expected_chunk_ids": ["faq-0002", "faq-0003"],
            "difficulty": "complex",
        },
    ]
    path = tmp_path / "golden.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def natural_dataset_file(tmp_path: Path) -> Path:
    rows = [
        {
            "question": "Can you explain the EU Taxonomy for our reporting team?",
            "expected_chunk_ids": ["faq-0001"],
            "difficulty": "simple",
            "query_type": "natural_simple",
            "persona": "CFO",
        },
        {
            "question": "What should we disclose about Taxonomy alignment?",
            "expected_chunk_ids": ["faq-0002"],
            "difficulty": "simple",
            "query_type": "natural_simple",
            "persona": "auditor",
        },
    ]
    path = tmp_path / "natural.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_load_eval_dataset_golden(golden_dataset_file: Path) -> None:
    dataset = load_eval_dataset(golden_dataset_file)

    assert len(dataset) == 3
    assert dataset[0].expected_chunk_ids == ["faq-0001"]
    assert not hasattr(dataset[0], "persona") or dataset[0].persona is None


def test_load_eval_dataset_natural(natural_dataset_file: Path) -> None:
    dataset = load_eval_dataset(natural_dataset_file)

    assert len(dataset) == 2
    assert dataset[0].persona == "CFO"
    assert dataset[0].query_type == "natural_simple"


def test_run_method_evaluation(
    tmp_path: Path,
    sample_chunks: list[Chunk],
    golden_dataset_file: Path,
    mock_embeddings: None,
) -> None:
    build_all_indexes(sample_chunks, base_dir=tmp_path, force_rebuild=True)
    questions = load_eval_dataset(golden_dataset_file)

    report = run_method_evaluation(
        questions,
        sample_chunks,
        RetrievalMethod.BM25,
        k=3,
        base_dir=tmp_path,
    )

    assert report.overall.num_queries == 3
    assert report.overall.recall_at_1 >= 0.0
    assert report.overall.mrr >= 0.0
    assert "simple" in report.by_difficulty
    assert report.by_persona == {}


def test_run_retrieval_evaluation_exports_json(
    tmp_path: Path,
    sample_chunks: list[Chunk],
    golden_dataset_file: Path,
    mock_embeddings: None,
) -> None:
    build_all_indexes(sample_chunks, base_dir=tmp_path, force_rebuild=True)
    output_path = tmp_path / "results.json"

    run_result = run_retrieval_evaluation(
        dataset_path=golden_dataset_file,
        methods=[RetrievalMethod.BM25, RetrievalMethod.HYBRID_MINILM],
        k=3,
        index_dir=tmp_path,
        chunks=sample_chunks,
        build_indexes=False,
    )
    saved_path = run_result.save(output_path)

    payload = json.loads(saved_path.read_text(encoding="utf-8"))
    assert payload["num_queries"] == 3
    assert "bm25" in payload["methods"]
    assert "hybrid_minilm" in payload["methods"]
    assert "mrr" in payload["methods"]["bm25"]["overall"]
    assert "simple" in payload["methods"]["bm25"]["by_difficulty"]


def test_run_retrieval_evaluation_natural_metadata_breakdown(
    tmp_path: Path,
    sample_chunks: list[Chunk],
    natural_dataset_file: Path,
    mock_embeddings: None,
) -> None:
    build_all_indexes(sample_chunks, base_dir=tmp_path, force_rebuild=True)

    run_result = run_retrieval_evaluation(
        dataset_path=natural_dataset_file,
        methods=[RetrievalMethod.BM25],
        k=2,
        index_dir=tmp_path,
        chunks=sample_chunks,
        build_indexes=False,
    )

    report = run_result.methods["bm25"]
    assert "CFO" in report.by_persona
    assert "natural_simple" in report.by_query_type


def test_format_metrics_summary() -> None:
    from eu_taxonomy_rag.evaluation.metrics import EvaluationReport, RetrievalMetrics

    run_result = EvaluationRunResult(
        dataset_path="data/test.jsonl",
        num_queries=10,
        retrieval_k=5,
        candidate_k=20,
        index_dir=".cache/index",
        methods={
            "bm25": EvaluationReport(
                overall=RetrievalMetrics(0.8, 0.9, 0.95, 0.75, num_queries=10),
                by_difficulty={},
                by_persona={},
                by_query_type={},
            )
        },
    )

    summary = format_metrics_summary(run_result)

    assert "bm25" in summary
    assert "0.800" in summary
    assert "Dataset: data/test.jsonl" in summary
