import json
from pathlib import Path

import numpy as np
import pytest

from eu_taxonomy_rag.retrieval.bm25_index import Bm25Index
from eu_taxonomy_rag.retrieval.dense_index import (
    build_dense_index,
    dense_index_backend,
    is_faiss_available,
    load_dense_index,
)
from eu_taxonomy_rag.retrieval.hybrid import reciprocal_rank_fusion
from eu_taxonomy_rag.pipelines.index_manager import (
    build_all_indexes,
    clear_index_cache,
    search,
)
from eu_taxonomy_rag.core.models import Chunk
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


def _assert_dense_index_files(directory: Path) -> None:
    assert (directory / "chunk_ids.json").exists()
    assert (directory / "index_backend.json").exists()
    backend = json.loads((directory / "index_backend.json").read_text(encoding="utf-8"))["backend"]
    if backend == "faiss":
        assert (directory / "faiss.index").exists()
    else:
        assert (directory / "embeddings.npy").exists()


def test_dense_index_save_and_load(tmp_path: Path, sample_chunks: list[Chunk]) -> None:
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    chunk_ids = [chunk.chunk_id for chunk in sample_chunks]

    index = build_dense_index(embeddings, chunk_ids)
    index.save(tmp_path / "dense_minilm")

    loaded = load_dense_index(tmp_path / "dense_minilm")
    results = loaded.search(embeddings[0], k=2)

    assert results[0][0] == "faq-0001"
    assert results[0][1] == pytest.approx(1.0)
    assert loaded.chunk_ids == chunk_ids
    _assert_dense_index_files(tmp_path / "dense_minilm")


@pytest.mark.skipif(is_faiss_available(), reason="Only relevant when FAISS is unavailable")
def test_dense_index_uses_numpy_backend_when_faiss_missing() -> None:
    assert dense_index_backend() == "numpy"


def test_bm25_index_save_and_load(tmp_path: Path, sample_chunks: list[Chunk]) -> None:
    index = Bm25Index.build(sample_chunks)
    index.save(tmp_path / "bm25")

    loaded = Bm25Index.load(tmp_path / "bm25")
    results = loaded.search("Taxonomy alignment reporting", k=2)

    assert len(results) == 2
    assert all(chunk_id.startswith("faq-") for chunk_id, _ in results)
    assert loaded.chunk_ids == [chunk.chunk_id for chunk in sample_chunks]


def test_reciprocal_rank_fusion() -> None:
    dense = [("faq-0001", 0.9), ("faq-0002", 0.8), ("faq-0003", 0.7)]
    bm25 = [("faq-0002", 5.0), ("faq-0003", 4.0), ("faq-0001", 3.0)]

    fused = reciprocal_rank_fusion([dense, bm25], k=3)

    assert len(fused) == 3
    assert fused[0][0] in {"faq-0001", "faq-0002"}
    assert all(score > 0 for _, score in fused)


def test_build_all_indexes(tmp_path: Path, sample_chunks: list[Chunk], monkeypatch) -> None:
    fake_embeddings = {
        "sentence-transformers/all-MiniLM-L6-v2": np.array(
            [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32
        ),
        "sentence-transformers/all-mpnet-base-v2": np.array(
            [[0.9, 0.1], [0.7, 0.3], [0.1, 0.9]], dtype=np.float32
        ),
    }

    class FakeModel:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False):
            return fake_embeddings[self.model_name][: len(texts)]

    monkeypatch.setattr(
        "eu_taxonomy_rag.retrieval.embeddings.get_embedding_model",
        lambda model_name: FakeModel(model_name),
    )

    indexes = build_all_indexes(sample_chunks, base_dir=tmp_path, force_rebuild=True)

    assert set(indexes.keys()) == {"dense_minilm", "dense_mpnet", "bm25"}
    _assert_dense_index_files(tmp_path / "dense_minilm")
    _assert_dense_index_files(tmp_path / "dense_mpnet")
    assert (tmp_path / "bm25" / "bm25s_index").exists()


def test_search_hybrid(tmp_path: Path, sample_chunks: list[Chunk], monkeypatch) -> None:
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

    build_all_indexes(sample_chunks, base_dir=tmp_path, force_rebuild=True)
    results = search(
        RetrievalMethod.HYBRID_MINILM,
        sample_chunks,
        "What is the EU Taxonomy?",
        k=2,
        base_dir=tmp_path,
    )

    assert len(results) == 2
    assert results[0][0] in {"faq-0001", "faq-0002", "faq-0003"}


def test_search_batch_dense(tmp_path: Path, sample_chunks: list[Chunk], monkeypatch) -> None:
    from eu_taxonomy_rag.pipelines.index_manager import search_batch

    fake_embeddings = np.array(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]],
        dtype=np.float32,
    )

    class FakeModel:
        def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False, batch_size=64):
            if len(texts) == 1:
                return np.array([[1.0, 0.0]], dtype=np.float32)
            return np.stack([fake_embeddings[0] if "Taxonomy" in text else fake_embeddings[1] for text in texts])

    monkeypatch.setattr(
        "eu_taxonomy_rag.retrieval.embeddings.get_embedding_model",
        lambda model_name: FakeModel(),
    )

    build_all_indexes(sample_chunks, base_dir=tmp_path, force_rebuild=True)
    queries = [
        "What is the EU Taxonomy?",
        "Who must report Taxonomy alignment?",
    ]
    batch = search_batch(
        RetrievalMethod.DENSE_MINILM,
        sample_chunks,
        queries,
        k=2,
        base_dir=tmp_path,
    )

    assert len(batch) == 2
    assert all(len(row) == 2 for row in batch)

