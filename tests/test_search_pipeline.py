from pathlib import Path

import numpy as np
import pytest

from eu_taxonomy_rag.pipelines.index_manager import build_all_indexes, clear_index_cache
from eu_taxonomy_rag.pipelines.ingestion import run_ingestion
from eu_taxonomy_rag.core.models import Chunk
from eu_taxonomy_rag.retrieval.retrieval_methods import RetrievalMethod
from eu_taxonomy_rag.pipelines.search_pipeline import SearchPipeline, create_search_pipeline


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
    class FakeModel:
        def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False):
            return np.array(
                [[float(i), float(i + 1)] for i in range(len(texts))],
                dtype=np.float32,
            )

    monkeypatch.setattr(
        "eu_taxonomy_rag.retrieval.embeddings.get_embedding_model",
        lambda model_name: FakeModel(),
    )


@pytest.fixture
def pipeline(tmp_path: Path, sample_chunks: list[Chunk], mock_embeddings: None) -> SearchPipeline:
    build_all_indexes(sample_chunks, base_dir=tmp_path / "index", force_rebuild=True)
    return SearchPipeline(
        chunks=sample_chunks,
        method=RetrievalMethod.BM25,
        base_dir=tmp_path / "index",
    )


def test_search_returns_top_k(pipeline: SearchPipeline) -> None:
    result = pipeline.search("Taxonomy alignment reporting", k=2)

    assert result.query == "Taxonomy alignment reporting"
    assert result.top_k == 2
    assert all(item.rank >= 1 for item in result.chunks)
    assert all(item.chunk.chunk_id.startswith("faq-") for item in result.chunks)


def test_search_rejects_empty_query(pipeline: SearchPipeline) -> None:
    with pytest.raises(ValueError, match="vide"):
        pipeline.search("   ", k=3)


def test_from_ingestion(tmp_path: Path, mock_embeddings: None) -> None:
    ingestion = run_ingestion(
        faq_path="data/taxonomy_faqs_cleaned.md",
        chunks_cache_path=tmp_path / "chunks.jsonl",
        index_dir=tmp_path / "index",
        force_rebuild=True,
    )

    pipeline = SearchPipeline.from_ingestion(ingestion, method=RetrievalMethod.BM25)
    result = pipeline.search("What is DNSH?", k=3)

    assert result.top_k == 3
    assert pipeline.method == RetrievalMethod.BM25


def test_create_search_pipeline(tmp_path: Path, mock_embeddings: None) -> None:
    pipeline = create_search_pipeline(
        method=RetrievalMethod.HYBRID_MINILM,
        faq_path="data/taxonomy_faqs_cleaned.md",
        chunks_cache_path=tmp_path / "chunks.jsonl",
        index_dir=tmp_path / "index",
    )

    result = pipeline.search("EU Taxonomy reporting", k=1)

    assert result.top_k == 1
    assert len(pipeline.chunks) == 324
