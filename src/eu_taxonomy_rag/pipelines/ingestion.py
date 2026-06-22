from dataclasses import dataclass
from pathlib import Path

from eu_taxonomy_rag.cache.chunk_cache import DEFAULT_CACHE_PATH, DEFAULT_FAQ_PATH, load_or_build_chunks
from eu_taxonomy_rag.retrieval.embeddings import clear_embedding_cache
from eu_taxonomy_rag.pipelines.index_manager import DEFAULT_INDEX_DIR, build_all_indexes, clear_index_cache
from eu_taxonomy_rag.core.models import Chunk


@dataclass(frozen=True)
class IngestionResult:
    """Résultat d'une exécution complète du pipeline d'ingestion."""

    chunks: tuple[Chunk, ...]
    index_names: tuple[str, ...]
    faq_path: Path
    chunks_cache_path: Path
    index_dir: Path

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)


def run_ingestion(
    faq_path: str | Path = DEFAULT_FAQ_PATH,
    chunks_cache_path: str | Path = DEFAULT_CACHE_PATH,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    *,
    force_rebuild: bool = False,
) -> IngestionResult:
    """
    Pipeline d'ingestion complet :
    1. Parser le fichier FAQ
    2. Générer les chunks (cache JSONL)
    3. Calculer les embeddings (cache mémoire)
    4. Construire les index FAISS + BM25 (cache disque)
    """
    faq_path = Path(faq_path)
    chunks_cache_path = Path(chunks_cache_path)
    index_dir = Path(index_dir)

    if force_rebuild:
        clear_embedding_cache()
        clear_index_cache()

    chunks = load_or_build_chunks(
        faq_path=faq_path,
        cache_path=chunks_cache_path,
        force_rebuild=force_rebuild,
    )

    indexes = build_all_indexes(
        chunks,
        base_dir=index_dir,
        force_rebuild=force_rebuild,
    )

    return IngestionResult(
        chunks=tuple(chunks),
        index_names=tuple(indexes.keys()),
        faq_path=faq_path,
        chunks_cache_path=chunks_cache_path,
        index_dir=index_dir,
    )
