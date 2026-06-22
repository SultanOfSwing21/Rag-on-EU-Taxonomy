from dataclasses import dataclass
from pathlib import Path

from eu_taxonomy_rag.core.models import Chunk
from eu_taxonomy_rag.paths import DEFAULT_INDEX_DIR
from eu_taxonomy_rag.retrieval.bm25_index import Bm25Index
from eu_taxonomy_rag.retrieval.dense_index import (
    DenseVectorIndex,
    build_dense_index,
    dense_index_files_exist,
    load_dense_index,
)
from eu_taxonomy_rag.retrieval.embeddings import (
    embed_queries,
    get_embedding_model,
    get_or_build_chunk_embeddings,
    release_embedding_models,
)
from eu_taxonomy_rag.retrieval.hybrid import reciprocal_rank_fusion
from eu_taxonomy_rag.retrieval.retrieval_methods import (
    DENSE_MODELS,
    RetrievalMethod,
    dense_key_for_method,
    is_dense_method,
    is_hybrid_method,
)

_dense_cache: dict[str, DenseVectorIndex] = {}
_bm25_cache: Bm25Index | None = None


@dataclass(frozen=True)
class IndexArtifact:
    """One on-disk retrieval index required by a set of methods."""

    key: str
    label: str
    directory: Path

    @property
    def exists(self) -> bool:
        if self.key == "bm25":
            return _index_files_exist(self.directory)
        return dense_index_files_exist(self.directory)


def required_dense_keys_and_bm25(
    methods: list[RetrievalMethod],
) -> tuple[tuple[str, ...], bool]:
    """Return dense model keys and whether BM25 is needed for the given methods."""
    dense_keys: set[str] = set()
    needs_bm25 = False

    for method in methods:
        if method == RetrievalMethod.BM25:
            needs_bm25 = True
        elif is_dense_method(method) or is_hybrid_method(method):
            dense_keys.add(dense_key_for_method(method))
            if is_hybrid_method(method):
                needs_bm25 = True

    return tuple(sorted(dense_keys)), needs_bm25


def index_artifacts_for_methods(
    methods: list[RetrievalMethod],
    base_dir: str | Path = DEFAULT_INDEX_DIR,
) -> list[IndexArtifact]:
    """List the index directories required for the selected retrieval methods."""
    dense_keys, needs_bm25 = required_dense_keys_and_bm25(methods)
    artifacts: list[IndexArtifact] = []

    for dense_key in dense_keys:
        config = DENSE_MODELS[dense_key]
        artifacts.append(
            IndexArtifact(
                key=f"dense_{dense_key}",
                label=f"Dense ({config.model_name})",
                directory=dense_index_directory(dense_key, base_dir),
            )
        )

    if needs_bm25:
        artifacts.append(
            IndexArtifact(
                key="bm25",
                label="BM25",
                directory=index_directory(RetrievalMethod.BM25, base_dir),
            )
        )

    return artifacts


def indexes_ready_for_methods(
    methods: list[RetrievalMethod],
    base_dir: str | Path = DEFAULT_INDEX_DIR,
) -> bool:
    """Return True when every required index already exists on disk."""
    artifacts = index_artifacts_for_methods(methods, base_dir=base_dir)
    return bool(artifacts) and all(artifact.exists for artifact in artifacts)


def index_directory(method: RetrievalMethod, base_dir: str | Path = DEFAULT_INDEX_DIR) -> Path:
    return Path(base_dir) / method.value


def dense_index_directory(
    dense_key: str,
    base_dir: str | Path = DEFAULT_INDEX_DIR,
) -> Path:
    return Path(base_dir) / f"dense_{dense_key}"


def build_faiss_index(
    chunks: list[Chunk],
    dense_key: str,
    *,
    base_dir: str | Path = DEFAULT_INDEX_DIR,
    force_rebuild: bool = False,
) -> DenseVectorIndex:
    """Build or reload a dense vector index (FAISS when available, otherwise NumPy)."""
    return build_dense_vector_index(
        chunks,
        dense_key,
        base_dir=base_dir,
        force_rebuild=force_rebuild,
    )


def build_dense_vector_index(
    chunks: list[Chunk],
    dense_key: str,
    *,
    base_dir: str | Path = DEFAULT_INDEX_DIR,
    force_rebuild: bool = False,
) -> DenseVectorIndex:
    """Build or reload a dense vector index (FAISS when available, otherwise NumPy)."""
    config = DENSE_MODELS[dense_key]
    directory = dense_index_directory(dense_key, base_dir)

    if not force_rebuild and dense_index_files_exist(directory):
        try:
            dense_index = load_dense_index(directory)
            _dense_cache[dense_key] = dense_index
            return dense_index
        except ImportError:
            pass

    embeddings, _ = get_or_build_chunk_embeddings(
        chunks,
        model_name=config.model_name,
        force_rebuild=force_rebuild,
        show_progress_bar=False,
    )
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    dense_index = build_dense_index(embeddings, chunk_ids)
    dense_index.save(directory)
    _dense_cache[dense_key] = dense_index
    return dense_index


def build_bm25_index(
    chunks: list[Chunk],
    *,
    base_dir: str | Path = DEFAULT_INDEX_DIR,
    force_rebuild: bool = False,
) -> Bm25Index:
    """Build or reload the BM25 index."""
    global _bm25_cache
    directory = index_directory(RetrievalMethod.BM25, base_dir)

    if not force_rebuild and _index_files_exist(directory):
        bm25_index = Bm25Index.load(directory)
        _bm25_cache = bm25_index
        return bm25_index

    bm25_index = Bm25Index.build(chunks)
    bm25_index.save(directory)
    _bm25_cache = bm25_index
    return bm25_index


def build_all_indexes(
    chunks: list[Chunk],
    *,
    base_dir: str | Path = DEFAULT_INDEX_DIR,
    force_rebuild: bool = False,
) -> dict[str, DenseVectorIndex | Bm25Index]:
    """Build all indexes (2 dense + BM25) for comparison."""
    indexes: dict[str, DenseVectorIndex | Bm25Index] = {}

    for dense_key in DENSE_MODELS:
        indexes[f"dense_{dense_key}"] = build_dense_vector_index(
            chunks,
            dense_key,
            base_dir=base_dir,
            force_rebuild=force_rebuild,
        )

    indexes["bm25"] = build_bm25_index(
        chunks,
        base_dir=base_dir,
        force_rebuild=force_rebuild,
    )
    return indexes


def search_batch(
    method: RetrievalMethod,
    chunks: list[Chunk],
    queries: list[str],
    k: int = 5,
    *,
    base_dir: str | Path = DEFAULT_INDEX_DIR,
    candidate_k: int = 20,
) -> list[list[tuple[str, float]]]:
    """Batch top-k search — encodes all queries in one pass (faster for evaluation)."""
    if not queries:
        return []

    k = min(k, len(chunks))
    candidate_k = min(candidate_k, len(chunks))

    if method == RetrievalMethod.BM25:
        bm25_index = _get_bm25_index(chunks, base_dir)
        return [bm25_index.search(query, k) for query in queries]

    if is_dense_method(method):
        dense_key = dense_key_for_method(method)
        dense_index = _get_dense_index(chunks, dense_key, base_dir)
        config = DENSE_MODELS[dense_key]
        model = get_embedding_model(config.model_name)
        query_embeddings = embed_queries(model, queries)
        return dense_index.search_batch(query_embeddings, k)

    if is_hybrid_method(method):
        dense_key = dense_key_for_method(method)
        dense_index = _get_dense_index(chunks, dense_key, base_dir)
        bm25_index = _get_bm25_index(chunks, base_dir)
        config = DENSE_MODELS[dense_key]
        model = get_embedding_model(config.model_name)
        query_embeddings = embed_queries(model, queries)

        dense_batch = dense_index.search_batch(query_embeddings, candidate_k)
        results: list[list[tuple[str, float]]] = []
        for query, dense_results in zip(queries, dense_batch):
            bm25_results = bm25_index.search(query, candidate_k)
            results.append(reciprocal_rank_fusion([dense_results, bm25_results], k))
        return results

    raise ValueError(f"Unknown retrieval method: {method}")


def search_single(
    method: RetrievalMethod,
    chunks: list[Chunk],
    query: str,
    k: int = 5,
    *,
    base_dir: str | Path = DEFAULT_INDEX_DIR,
    candidate_k: int = 20,
) -> list[tuple[str, float]]:
    return search_batch(
        method,
        chunks,
        [query],
        k=k,
        base_dir=base_dir,
        candidate_k=candidate_k,
    )[0]


def search(
    method: RetrievalMethod,
    chunks: list[Chunk],
    query: str,
    k: int = 5,
    *,
    base_dir: str | Path = DEFAULT_INDEX_DIR,
    candidate_k: int = 20,
) -> list[tuple[str, float]]:
    """Top-k search with the selected method (dense, BM25, or hybrid)."""
    return search_single(
        method,
        chunks,
        query,
        k=k,
        base_dir=base_dir,
        candidate_k=candidate_k,
    )


def search_chunks(
    method: RetrievalMethod,
    chunks: list[Chunk],
    query: str,
    k: int = 5,
    **kwargs,
) -> list[tuple[Chunk, float]]:
    """Like search(), but returns Chunk objects."""
    chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
    results = search(method, chunks, query, k, **kwargs)
    return [(chunk_map[chunk_id], score) for chunk_id, score in results]


def clear_index_cache() -> None:
    """Clear in-memory index cache."""
    global _bm25_cache
    _dense_cache.clear()
    _bm25_cache = None


def _get_dense_index(
    chunks: list[Chunk],
    dense_key: str,
    base_dir: str | Path,
) -> DenseVectorIndex:
    if dense_key not in _dense_cache:
        directory = dense_index_directory(dense_key, base_dir)
        if dense_index_files_exist(directory):
            try:
                _dense_cache[dense_key] = load_dense_index(directory)
            except ImportError:
                _dense_cache[dense_key] = build_dense_vector_index(
                    chunks, dense_key, base_dir=base_dir, force_rebuild=True
                )
        else:
            _dense_cache[dense_key] = build_dense_vector_index(chunks, dense_key, base_dir=base_dir)
    return _dense_cache[dense_key]


def _get_bm25_index(chunks: list[Chunk], base_dir: str | Path) -> Bm25Index:
    global _bm25_cache
    if _bm25_cache is None:
        directory = index_directory(RetrievalMethod.BM25, base_dir)
        if _index_files_exist(directory):
            _bm25_cache = Bm25Index.load(directory)
        else:
            _bm25_cache = build_bm25_index(chunks, base_dir=base_dir)
    return _bm25_cache


def _index_files_exist(directory: Path) -> bool:
    return directory.exists() and any(directory.iterdir())
