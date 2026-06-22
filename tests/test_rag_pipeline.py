from unittest.mock import MagicMock

import pytest

from eu_taxonomy_rag.core.models import Chunk, RetrievedChunk, RetrievalResult


def test_generate_rag_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    from eu_taxonomy_rag.pipelines import rag_pipeline

    chunk = Chunk(
        chunk_id="faq-0001",
        question="What is the EU Taxonomy?",
        answer="A classification system.",
        metadata={"section": "General"},
    )
    retrieval = RetrievalResult(
        query="What is the EU Taxonomy?",
        chunks=(RetrievedChunk(chunk=chunk, score=0.9, rank=1),),
    )

    class FakeRetriever:
        def __init__(self, *args, **kwargs):
            pass

        def retrieve(self, query, k=5):
            return retrieval

    monkeypatch.setattr(rag_pipeline, "Retriever", FakeRetriever)
    monkeypatch.setattr(rag_pipeline, "build_indexes_for_methods", lambda *args, **kwargs: None)

    client = MagicMock()
    client.complete.return_value = "The EU Taxonomy is a classification system. [faq-0001]"

    result = rag_pipeline.generate_rag_answer(
        "What is the EU Taxonomy?",
        [chunk],
        client,
        build_indexes=False,
    )

    assert "classification system" in result.answer
    assert result.chunk_ids == ["faq-0001"]
    client.complete.assert_called_once()
