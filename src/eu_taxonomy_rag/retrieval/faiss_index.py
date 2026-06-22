import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eu_taxonomy_rag.retrieval.dense_index import (
    BACKEND_FILENAME,
    CHUNK_IDS_FILENAME,
    FAISS_FILENAME,
    write_backend_marker,
)

BACKEND_NAME = "faiss"


@dataclass
class FaissIndex:
    """Dense vector index backed by FAISS (cosine similarity via inner product)."""

    index: object
    chunk_ids: list[str]

    @classmethod
    def build(cls, embeddings: np.ndarray, chunk_ids: list[str]) -> "FaissIndex":
        import faiss

        if embeddings.shape[0] != len(chunk_ids):
            raise ValueError("The number of embeddings must match the number of chunk_ids.")

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
        return cls(index=index, chunk_ids=list(chunk_ids))

    def search(self, query_embedding: np.ndarray, k: int) -> list[tuple[str, float]]:
        return self.search_batch(query_embedding.reshape(1, -1), k)[0]

    def search_batch(self, query_embeddings: np.ndarray, k: int) -> list[list[tuple[str, float]]]:
        import faiss

        queries = np.ascontiguousarray(query_embeddings, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)

        scores, indices = self.index.search(queries, k)
        results: list[list[tuple[str, float]]] = []
        for row_scores, row_indices in zip(scores, indices):
            row: list[tuple[str, float]] = []
            for idx, score in zip(row_indices, row_scores):
                if idx < 0:
                    continue
                row.append((self.chunk_ids[idx], float(score)))
            results.append(row)
        return results

    def save(self, directory: str | Path) -> None:
        import faiss

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / FAISS_FILENAME))
        (directory / CHUNK_IDS_FILENAME).write_text(
            json.dumps(self.chunk_ids, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_backend_marker(directory, BACKEND_NAME)

    @classmethod
    def load(cls, directory: str | Path) -> "FaissIndex":
        import faiss

        directory = Path(directory)
        index = faiss.read_index(str(directory / FAISS_FILENAME))
        chunk_ids = json.loads((directory / CHUNK_IDS_FILENAME).read_text(encoding="utf-8"))
        return cls(index=index, chunk_ids=chunk_ids)
