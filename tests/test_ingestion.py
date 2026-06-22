from pathlib import Path

import numpy as np
import pytest

from eu_taxonomy_rag.cache.chunk_cache import save_chunks
from eu_taxonomy_rag.retrieval.embeddings import clear_embedding_cache
from eu_taxonomy_rag.pipelines.index_manager import clear_index_cache
from eu_taxonomy_rag.pipelines.ingestion import run_ingestion
from eu_taxonomy_rag.core.models import Chunk


@pytest.fixture(autouse=True)
def reset_caches() -> None:
    clear_embedding_cache()
    clear_index_cache()
    yield
    clear_embedding_cache()
    clear_index_cache()


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="faq-0001",
            question="What is the EU Taxonomy?",
            answer="A classification system for sustainable economic activities.",
            metadata={"section": "General", "index": 1, "source": "test.md"},
        ),
        Chunk(
            chunk_id="faq-0002",
            question="Who must report Taxonomy alignment?",
            answer="Large companies under the NFRD must disclose alignment.",
            metadata={"section": "Reporting", "index": 2, "source": "test.md"},
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


def test_run_ingestion_builds_chunks_and_indexes(
    tmp_path: Path,
    sample_chunks: list[Chunk],
    mock_embeddings: None,
) -> None:
    faq_path = tmp_path / "faqs.md"
    faq_path.write_text("# placeholder\n", encoding="utf-8")

    chunks_cache = tmp_path / "chunks.jsonl"
    index_dir = tmp_path / "index"
    save_chunks(sample_chunks, chunks_cache)

    result = run_ingestion(
        faq_path=faq_path,
        chunks_cache_path=chunks_cache,
        index_dir=index_dir,
    )

    assert result.chunk_count == 2
    assert result.chunks[0].chunk_id == "faq-0001"
    assert set(result.index_names) == {"dense_minilm", "dense_mpnet", "bm25"}
    dense_dir = index_dir / "dense_minilm"
    assert (dense_dir / "chunk_ids.json").exists()
    assert (dense_dir / "faiss.index").exists() or (dense_dir / "embeddings.npy").exists()
    assert (index_dir / "bm25" / "bm25s_index").exists()


def test_run_ingestion_from_real_faq_file(mock_embeddings: None, tmp_path: Path) -> None:
    result = run_ingestion(
        faq_path="data/taxonomy_faqs_cleaned.md",
        chunks_cache_path=tmp_path / "chunks.jsonl",
        index_dir=tmp_path / "index",
        force_rebuild=True,
    )

    assert result.chunk_count == 324
    assert len(result.index_names) == 3
