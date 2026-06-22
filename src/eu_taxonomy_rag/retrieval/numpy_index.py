import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eu_taxonomy_rag.retrieval.dense_index import (
    BACKEND_FILENAME,
    CHUNK_IDS_FILENAME,
    EMBEDDINGS_FILENAME,
    write_backend_marker,
)

BACKEND_NAME = "numpy"


@dataclass
class NumpyDenseIndex:
    """Dense vector index using cosine similarity on normalized embeddings."""

    embeddings: np.ndarray
    chunk_ids: list[str]

    @classmethod
    def build(cls, embeddings: np.ndarray, chunk_ids: list[str]) -> "NumpyDenseIndex":
        if embeddings.shape[0] != len(chunk_ids):
            raise ValueError("The number of embeddings must match the number of chunk_ids.")

        normalized = _normalize_rows(np.ascontiguousarray(embeddings, dtype=np.float32))
        return cls(embeddings=normalized, chunk_ids=list(chunk_ids))

    def search(self, query_embedding: np.ndarray, k: int) -> list[tuple[str, float]]:
        return self.search_batch(query_embedding.reshape(1, -1), k)[0]

    def search_batch(self, query_embeddings: np.ndarray, k: int) -> list[list[tuple[str, float]]]:
        queries = _normalize_rows(np.ascontiguousarray(query_embeddings, dtype=np.float32))
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)

        scores = queries @ self.embeddings.T
        k = min(k, len(self.chunk_ids))
        if k == 0:
            return [[] for _ in range(scores.shape[0])]

        results: list[list[tuple[str, float]]] = []
        for row_scores in scores:
            top_indices = np.argpartition(-row_scores, k - 1)[:k]
            top_indices = top_indices[np.argsort(-row_scores[top_indices])]
            results.append([(self.chunk_ids[index], float(row_scores[index])) for index in top_indices])
        return results

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / EMBEDDINGS_FILENAME, self.embeddings)
        (directory / CHUNK_IDS_FILENAME).write_text(
            json.dumps(self.chunk_ids, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_backend_marker(directory, BACKEND_NAME)

    @classmethod
    def load(cls, directory: str | Path) -> "NumpyDenseIndex":
        directory = Path(directory)
        embeddings = np.load(directory / EMBEDDINGS_FILENAME)
        chunk_ids = json.loads((directory / CHUNK_IDS_FILENAME).read_text(encoding="utf-8"))
        return cls(embeddings=np.ascontiguousarray(embeddings, dtype=np.float32), chunk_ids=chunk_ids)


def _normalize_rows(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return embeddings / norms


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return vector
    return vector / norm
