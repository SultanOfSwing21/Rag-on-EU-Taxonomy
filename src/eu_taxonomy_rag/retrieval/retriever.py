from dataclasses import dataclass, field
from pathlib import Path

from eu_taxonomy_rag.pipelines.index_manager import DEFAULT_INDEX_DIR, search
from eu_taxonomy_rag.core.models import Chunk, RetrievalResult, RetrievedChunk
from eu_taxonomy_rag.retrieval.retrieval_methods import RetrievalMethod


@dataclass
class Retriever:
    """Retriever top-k branché sur les index (dense, BM25 ou hybride)."""

    chunks: list[Chunk]
    method: RetrievalMethod = RetrievalMethod.HYBRID_MINILM
    base_dir: Path = field(default_factory=lambda: DEFAULT_INDEX_DIR)
    candidate_k: int = 20

    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        """Recherche les k chunks les plus pertinents pour une question."""
        chunk_map = {chunk.chunk_id: chunk for chunk in self.chunks}
        raw_results = search(
            self.method,
            self.chunks,
            query,
            k=k,
            base_dir=self.base_dir,
            candidate_k=self.candidate_k,
        )

        retrieved = tuple(
            RetrievedChunk(
                chunk=chunk_map[chunk_id],
                score=score,
                rank=rank,
            )
            for rank, (chunk_id, score) in enumerate(raw_results, start=1)
            if chunk_id in chunk_map
        )

        return RetrievalResult(query=query, chunks=retrieved)


def retrieve(
    query: str,
    chunks: list[Chunk],
    method: RetrievalMethod = RetrievalMethod.HYBRID_MINILM,
    k: int = 5,
    **kwargs,
) -> RetrievalResult:
    """Raccourci fonctionnel autour de Retriever.retrieve()."""
    return Retriever(chunks=chunks, method=method, **kwargs).retrieve(query, k=k)
