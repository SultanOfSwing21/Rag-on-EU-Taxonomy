from eu_taxonomy_rag.core.models import Chunk, RetrievedChunk, RetrievalResult
from eu_taxonomy_rag.core.prompt import (
    SYSTEM_PROMPT,
    build_rag_messages,
    build_rag_prompt,
    format_context,
)


def _sample_retrieval() -> RetrievalResult:
    chunk = Chunk(
        chunk_id="faq-0001",
        question="What is the EU Taxonomy?",
        answer="A classification system for sustainable economic activities.",
        metadata={"section": "EU Taxonomy - General", "index": 1},
    )
    return RetrievalResult(
        query="What is the EU Taxonomy?",
        chunks=(RetrievedChunk(chunk=chunk, score=0.92, rank=1),),
    )


def test_format_context_includes_chunk_id_and_content() -> None:
    retrieval = _sample_retrieval()

    context = format_context(retrieval.chunks)

    assert "[faq-0001]" in context
    assert "EU Taxonomy - General" in context
    assert "classification system" in context


def test_format_context_empty() -> None:
    assert format_context([]) == "No context retrieved."


def test_build_rag_prompt_injects_context_and_question() -> None:
    retrieval = _sample_retrieval()

    prompt = build_rag_prompt("What is the EU Taxonomy?", retrieval)

    assert "Context:" in prompt
    assert "[faq-0001]" in prompt
    assert "Question:" in prompt
    assert "What is the EU Taxonomy?" in prompt
    assert "ONLY the context" in prompt


def test_build_rag_messages_has_system_and_user_roles() -> None:
    retrieval = _sample_retrieval()

    messages = build_rag_messages("What is DNSH?", retrieval)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert "Answer ONLY using the context" in messages[0]["content"]
    assert "What is DNSH?" in messages[1]["content"]
