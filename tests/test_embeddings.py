from unittest.mock import MagicMock

import numpy as np
import pytest

from eu_taxonomy_rag.retrieval.embeddings import (
    clear_embedding_cache,
    embed_chunks,
    embed_query,
    embed_texts,
    get_embedding_model,
    get_or_build_chunk_embeddings,
)
from eu_taxonomy_rag.core.models import Chunk


@pytest.fixture(autouse=True)
def reset_embedding_cache() -> None:
    clear_embedding_cache()
    yield
    clear_embedding_cache()


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="faq-0001",
            question="What is the EU Taxonomy?",
            answer="A classification system.",
            metadata={"section": "General", "index": 1},
        ),
        Chunk(
            chunk_id="faq-0002",
            question="Who must report?",
            answer="Large companies.",
            metadata={"section": "Reporting", "index": 2},
        ),
    ]


@pytest.fixture
def mock_model() -> MagicMock:
    model = MagicMock()

    def encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False):
        vectors = []
        for text in texts:
            length = float(len(text))
            vectors.append([length, length + 1.0, length + 2.0])
        return np.asarray(vectors, dtype=np.float32)

    model.encode.side_effect = encode
    return model


def test_embed_texts(mock_model: MagicMock) -> None:
    embeddings = embed_texts(mock_model, ["hello", "hello world"])

    assert embeddings.shape == (2, 3)
    assert embeddings.dtype == np.float32
    assert embeddings[1][0] > embeddings[0][0]


def test_embed_chunks_uses_chunk_text(mock_model: MagicMock, sample_chunks: list[Chunk]) -> None:
    embeddings = embed_chunks(mock_model, sample_chunks)

    assert embeddings.shape == (2, 3)
    texts = mock_model.encode.call_args[0][0]
    assert texts[0] == sample_chunks[0].text
    assert "Question:" in texts[0]
    assert "Answer:" in texts[0]


def test_embed_query(mock_model: MagicMock) -> None:
    vector = embed_query(mock_model, "What is DNSH?")

    assert vector.shape == (3,)
    assert vector.dtype == np.float32


def test_get_embedding_model_is_cached(monkeypatch: pytest.MonkeyPatch, mock_model: MagicMock) -> None:
    import sys
    from types import ModuleType

    created: list[MagicMock] = []

    def factory(model_name: str) -> MagicMock:
        created.append(mock_model)
        return mock_model

    fake_module = ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = factory
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    first = get_embedding_model()
    second = get_embedding_model()

    assert first is second
    assert len(created) == 1


def test_get_or_build_chunk_embeddings_uses_memory_cache(
    sample_chunks: list[Chunk], mock_model: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "eu_taxonomy_rag.retrieval.embeddings.get_embedding_model",
        lambda model_name=None: mock_model,
    )

    first, _ = get_or_build_chunk_embeddings(sample_chunks)
    second, _ = get_or_build_chunk_embeddings(sample_chunks)

    assert np.array_equal(first, second)
    mock_model.encode.assert_called_once()


def test_get_or_build_chunk_embeddings_rebuilds_when_forced(
    sample_chunks: list[Chunk], mock_model: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "eu_taxonomy_rag.retrieval.embeddings.get_embedding_model",
        lambda model_name=None: mock_model,
    )

    get_or_build_chunk_embeddings(sample_chunks)
    get_or_build_chunk_embeddings(sample_chunks, force_rebuild=True)

    assert mock_model.encode.call_count == 2
