from dataclasses import dataclass, field
from pathlib import Path

from eu_taxonomy_rag.cache.chunk_cache import DEFAULT_CACHE_PATH, DEFAULT_FAQ_PATH, load_or_build_chunks
from eu_taxonomy_rag.pipelines.index_manager import DEFAULT_INDEX_DIR, build_all_indexes
from eu_taxonomy_rag.pipelines.ingestion import IngestionResult
from eu_taxonomy_rag.core.models import Chunk, RetrievalResult
from eu_taxonomy_rag.retrieval.retrieval_methods import RetrievalMethod
from eu_taxonomy_rag.retrieval.retriever import Retriever


@dataclass
class SearchPipeline:
    """
    Pipeline de recherche :
    1. Reçoit une question utilisateur
    2. Calcule son embedding (dense / hybride) ou tokenise (BM25)
    3. Retourne les top-k chunks les plus pertinents
    """

    chunks: list[Chunk]
    method: RetrievalMethod = RetrievalMethod.HYBRID_MINILM
    base_dir: Path = field(default_factory=lambda: DEFAULT_INDEX_DIR)
    candidate_k: int = 20

    @classmethod
    def from_ingestion(
        cls,
        ingestion: IngestionResult,
        method: RetrievalMethod = RetrievalMethod.HYBRID_MINILM,
        candidate_k: int = 20,
    ) -> "SearchPipeline":
        return cls(
            chunks=list(ingestion.chunks),
            method=method,
            base_dir=ingestion.index_dir,
            candidate_k=candidate_k,
        )

    def search(self, query: str, k: int = 5) -> RetrievalResult:
        """Exécute une recherche top-k pour une question."""
        question = query.strip()
        if not question:
            raise ValueError("La question ne peut pas être vide.")

        return Retriever(
            chunks=self.chunks,
            method=self.method,
            base_dir=self.base_dir,
            candidate_k=self.candidate_k,
        ).retrieve(question, k=k)


def create_search_pipeline(
    method: RetrievalMethod = RetrievalMethod.HYBRID_MINILM,
    faq_path: str | Path = DEFAULT_FAQ_PATH,
    chunks_cache_path: str | Path = DEFAULT_CACHE_PATH,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    *,
    candidate_k: int = 20,
) -> SearchPipeline:
    """
    Prépare un pipeline de recherche prêt à l'emploi :
    charge les chunks et s'assure que les index existent.
    """
    chunks = load_or_build_chunks(faq_path=faq_path, cache_path=chunks_cache_path)
    build_all_indexes(chunks, base_dir=index_dir)

    return SearchPipeline(
        chunks=chunks,
        method=method,
        base_dir=Path(index_dir),
        candidate_k=candidate_k,
    )
