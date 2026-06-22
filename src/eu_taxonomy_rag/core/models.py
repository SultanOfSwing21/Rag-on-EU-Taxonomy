from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FAQ:
    """Une entrée FAQ source extraite du fichier markdown."""

    question: str
    answer: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """Unité indexée pour le retrieval (1 FAQ = 1 chunk)."""

    chunk_id: str
    question: str
    answer: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Texte utilisé pour l'embedding et l'injection dans le prompt RAG."""
        return f"Question: {self.question}\nAnswer: {self.answer}"


@dataclass(frozen=True)
class RetrievedChunk:
    """Chunk récupéré avec son score de similarité."""

    chunk: Chunk
    score: float
    rank: int = 0


@dataclass(frozen=True)
class RetrievalResult:
    """Résultat d'une recherche top-k."""

    query: str
    chunks: tuple[RetrievedChunk, ...]

    @property
    def top_k(self) -> int:
        return len(self.chunks)
