from dataclasses import dataclass
from pathlib import Path

from eu_taxonomy_rag.cache.chunk_cache import load_or_build_chunks
from eu_taxonomy_rag.core.models import Chunk, RetrievalResult
from eu_taxonomy_rag.core.prompt import SYSTEM_PROMPT, build_rag_prompt, format_context
from eu_taxonomy_rag.llm.client import ChatClient
from eu_taxonomy_rag.pipelines.index_manager import DEFAULT_INDEX_DIR
from eu_taxonomy_rag.evaluation.runner import build_indexes_for_methods
from eu_taxonomy_rag.retrieval.retrieval_methods import RetrievalMethod
from eu_taxonomy_rag.retrieval.retriever import Retriever


@dataclass(frozen=True)
class RAGAnswer:
    """Single-turn RAG response (no conversation memory)."""

    question: str
    answer: str
    retrieval: RetrievalResult
    context: str

    @property
    def chunk_ids(self) -> list[str]:
        return [item.chunk.chunk_id for item in self.retrieval.chunks]


def generate_rag_answer(
    question: str,
    chunks: list[Chunk],
    client: ChatClient,
    *,
    method: RetrievalMethod = RetrievalMethod.HYBRID_MINILM,
    k: int = 5,
    candidate_k: int = 20,
    base_dir: str | Path = DEFAULT_INDEX_DIR,
    build_indexes: bool = True,
) -> RAGAnswer:
    """Retrieve context chunks then generate an answer with the LLM."""
    query = question.strip()
    if not query:
        raise ValueError("Question cannot be empty.")

    if build_indexes:
        build_indexes_for_methods(chunks, [method], base_dir=base_dir)

    retrieval = Retriever(
        chunks=chunks,
        method=method,
        base_dir=Path(base_dir),
        candidate_k=candidate_k,
    ).retrieve(query, k=k)

    user_prompt = build_rag_prompt(query, retrieval)
    answer = client.complete(SYSTEM_PROMPT, user_prompt)

    return RAGAnswer(
        question=query,
        answer=answer,
        retrieval=retrieval,
        context=format_context(retrieval.chunks),
    )


def ask_rag(
    question: str,
    client: ChatClient,
    *,
    method: RetrievalMethod = RetrievalMethod.HYBRID_MINILM,
    k: int = 5,
    candidate_k: int = 20,
    base_dir: str | Path = DEFAULT_INDEX_DIR,
) -> RAGAnswer:
    """Convenience wrapper: load chunks and run the RAG pipeline."""
    chunks = load_or_build_chunks()
    return generate_rag_answer(
        question,
        chunks,
        client,
        method=method,
        k=k,
        candidate_k=candidate_k,
        base_dir=base_dir,
        build_indexes=True,
    )
