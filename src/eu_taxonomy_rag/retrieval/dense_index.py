"""Dense vector index with optional FAISS backend and NumPy fallback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np

CHUNK_IDS_FILENAME = "chunk_ids.json"
EMBEDDINGS_FILENAME = "embeddings.npy"
BACKEND_FILENAME = "index_backend.json"
FAISS_FILENAME = "faiss.index"


class DenseVectorIndex(Protocol):
    chunk_ids: list[str]

    def search(self, query_embedding: np.ndarray, k: int) -> list[tuple[str, float]]: ...

    def save(self, directory: str | Path) -> None: ...


def is_faiss_available() -> bool:
    try:
        import faiss  # noqa: F401
        return True
    except ImportError:
        return False


def dense_index_backend() -> str:
    return "faiss" if is_faiss_available() else "numpy"


def write_backend_marker(directory: Path, backend: str) -> None:
    (directory / BACKEND_FILENAME).write_text(
        json.dumps({"backend": backend}, indent=2),
        encoding="utf-8",
    )


def build_dense_index(embeddings: np.ndarray, chunk_ids: list[str]) -> DenseVectorIndex:
    if is_faiss_available():
        from eu_taxonomy_rag.retrieval.faiss_index import FaissIndex

        return FaissIndex.build(embeddings, chunk_ids)

    from eu_taxonomy_rag.retrieval.numpy_index import NumpyDenseIndex

    return NumpyDenseIndex.build(embeddings, chunk_ids)


def load_dense_index(directory: str | Path) -> DenseVectorIndex:
    directory = Path(directory)
    backend = _detect_backend(directory)

    if backend == "faiss":
        if not is_faiss_available():
            raise ImportError("FAISS index on disk but faiss-cpu is not installed.")
        from eu_taxonomy_rag.retrieval.faiss_index import FaissIndex

        return FaissIndex.load(directory)

    from eu_taxonomy_rag.retrieval.numpy_index import NumpyDenseIndex

    return NumpyDenseIndex.load(directory)


def dense_index_files_exist(directory: Path) -> bool:
    if not directory.exists():
        return False
    if (directory / BACKEND_FILENAME).exists():
        return True
    if (directory / EMBEDDINGS_FILENAME).exists() and (directory / CHUNK_IDS_FILENAME).exists():
        return True
    if (directory / FAISS_FILENAME).exists() and (directory / CHUNK_IDS_FILENAME).exists():
        return True
    return False


def _detect_backend(directory: Path) -> str:
    backend_file = directory / BACKEND_FILENAME
    if backend_file.exists():
        payload = json.loads(backend_file.read_text(encoding="utf-8"))
        return payload["backend"]

    if (directory / FAISS_FILENAME).exists():
        return "faiss"
    if (directory / EMBEDDINGS_FILENAME).exists():
        return "numpy"
    raise FileNotFoundError(f"No dense index found in {directory}")
