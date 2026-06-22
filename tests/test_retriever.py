from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from eu_taxonomy_rag.pipelines.index_manager import build_all_indexes, clear_index_cache
from eu_taxonomy_rag.core.models import Chunk
from eu_taxonomy_rag.retrieval.retrieval_methods import RetrievalMethod
from eu_taxonomy_rag.retrieval.retriever import Retriever, retrieve


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


def test_retriever_returns_retrieval_result(
    tmp_path: Path,
    sample_chunks: list[Chunk],
    mock_embeddings: None,
) -> None:
    build_all_indexes(sample_chunks, base_dir=tmp_path, force_rebuild=True)

    retriever = Retriever(
        chunks=sample_chunks,
        method=RetrievalMethod.BM25,
        base_dir=tmp_path,
    )
    result = retriever.retrieve("Taxonomy alignment reporting", k=2)

    assert result.query == "Taxonomy alignment reporting"
    assert result.top_k == 2
    assert all(item.rank >= 1 for item in result.chunks)
    assert all(item.score > 0 for item in result.chunks)
    assert all(item.chunk.chunk_id.startswith("faq-") for item in result.chunks)


def test_retriever_assigns_ranks_in_order(
    tmp_path: Path,
    sample_chunks: list[Chunk],
    mock_embeddings: None,
) -> None:
    build_all_indexes(sample_chunks, base_dir=tmp_path, force_rebuild=True)

    result = Retriever(
        chunks=sample_chunks,
        method=RetrievalMethod.HYBRID_MINILM,
        base_dir=tmp_path,
    ).retrieve("What is the EU Taxonomy?", k=3)

    ranks = [item.rank for item in result.chunks]
    assert ranks == list(range(1, len(result.chunks) + 1))


def test_retrieve_function_shortcut(
    tmp_path: Path,
    sample_chunks: list[Chunk],
    mock_embeddings: None,
) -> None:
    build_all_indexes(sample_chunks, base_dir=tmp_path, force_rebuild=True)

    result = retrieve(
        "technical screening criteria",
        sample_chunks,
        method=RetrievalMethod.BM25,
        base_dir=tmp_path,
        k=1,
    )

    assert result.top_k == 1
    assert "technical screening criteria" in result.chunks[0].chunk.question.lower()
