import json
from dataclasses import dataclass
from pathlib import Path

import bm25s

from eu_taxonomy_rag.core.models import Chunk

BM25_DIRNAME = "bm25s_index"
CHUNK_IDS_FILENAME = "chunk_ids.json"


@dataclass
class Bm25Index:
    """Index lexical sparse basé sur BM25S."""

    retriever: bm25s.BM25
    chunk_ids: list[str]

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "Bm25Index":
        corpus = [chunk.text for chunk in chunks]
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        corpus_tokens = bm25s.tokenize(corpus, stopwords="en")

        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)

        return cls(retriever=retriever, chunk_ids=chunk_ids)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        query_tokens = bm25s.tokenize(query, stopwords="en")
        indices, scores = self.retriever.retrieve(query_tokens, k=k)

        results: list[tuple[str, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            chunk_id = self.chunk_ids[int(idx)]
            results.append((chunk_id, float(score)))
        return results

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.retriever.save(str(directory / BM25_DIRNAME))
        (directory / CHUNK_IDS_FILENAME).write_text(
            json.dumps(self.chunk_ids, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> "Bm25Index":
        directory = Path(directory)
        retriever = bm25s.BM25.load(str(directory / BM25_DIRNAME), load_corpus=False)
        chunk_ids = json.loads((directory / CHUNK_IDS_FILENAME).read_text(encoding="utf-8"))
        return cls(retriever=retriever, chunk_ids=chunk_ids)
