"""Chatbot RAG sur les FAQ de l'EU Taxonomy."""

from eu_taxonomy_rag.core.models import Chunk, FAQ, RetrievedChunk, RetrievalResult

__version__ = "0.1.0"

__all__ = ["Chunk", "FAQ", "RetrievedChunk", "RetrievalResult", "__version__"]
