import gc
import os
from typing import TYPE_CHECKING

import numpy as np

from eu_taxonomy_rag.core.models import Chunk
from eu_taxonomy_rag.retrieval.retrieval_methods import DENSE_MODELS

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = DENSE_MODELS["minilm"].model_name
DEFAULT_CHUNK_BATCH_SIZE = 32
DEFAULT_QUERY_BATCH_SIZE = 16

_model_cache: dict[str, "SentenceTransformer"] = {}
_chunk_embeddings_cache: dict[tuple[str, tuple[str, ...]], np.ndarray] = {}


def is_sentence_transformers_available() -> bool:
    """True when sentence-transformers (and its backend) can be imported."""
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


def embedding_device() -> str:
    """Default to CPU to avoid MPS/CUDA OOM on Apple Silicon during benchmarks."""
    return os.environ.get("EU_TAXONOMY_EMBEDDING_DEVICE", "cpu")


def get_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> "SentenceTransformer":
    """Load the model once and keep it in memory."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for dense and hybrid retrieval. "
            "Install project dependencies with Python 3.10–3.12: "
            "pip install -e \".[ui]\""
        ) from exc

    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name, device=embedding_device())
    return _model_cache[model_name]


def release_embedding_models(model_names: list[str] | None = None) -> None:
    """Free embedding models from RAM (keeps chunk embedding arrays)."""
    if model_names is None:
        _model_cache.clear()
    else:
        for model_name in model_names:
            _model_cache.pop(model_name, None)

    gc.collect()
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def embed_texts(
    model: "SentenceTransformer",
    texts: list[str],
    *,
    show_progress_bar: bool | None = None,
    batch_size: int = DEFAULT_CHUNK_BATCH_SIZE,
) -> np.ndarray:
    """Generate embeddings for a list of texts."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    if show_progress_bar is None:
        show_progress_bar = len(texts) > 50
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress_bar,
        batch_size=batch_size,
    )
    return np.asarray(embeddings, dtype=np.float32)


def embed_chunks(
    model: "SentenceTransformer",
    chunks: list[Chunk],
    *,
    show_progress_bar: bool | None = None,
    batch_size: int = DEFAULT_CHUNK_BATCH_SIZE,
) -> np.ndarray:
    """Generate embeddings for FAQ chunks."""
    texts = [chunk.text for chunk in chunks]
    return embed_texts(model, texts, show_progress_bar=show_progress_bar, batch_size=batch_size)


def embed_queries(
    model: "SentenceTransformer",
    queries: list[str],
    *,
    batch_size: int = DEFAULT_QUERY_BATCH_SIZE,
) -> np.ndarray:
    """Batch-encode evaluation queries in small chunks to limit peak memory."""
    if not queries:
        return np.empty((0, 0), dtype=np.float32)

    batches: list[np.ndarray] = []
    show_progress = len(queries) > 100
    for start in range(0, len(queries), batch_size):
        chunk = queries[start : start + batch_size]
        batches.append(
            embed_texts(
                model,
                chunk,
                show_progress_bar=show_progress and start == 0,
                batch_size=batch_size,
            )
        )
    return np.vstack(batches)


def embed_query(model: "SentenceTransformer", query: str) -> np.ndarray:
    """Generate a single query embedding."""
    return embed_texts(model, [query], show_progress_bar=False, batch_size=1)[0]


def get_or_build_chunk_embeddings(
    chunks: list[Chunk],
    model_name: str = DEFAULT_MODEL_NAME,
    *,
    force_rebuild: bool = False,
    show_progress_bar: bool | None = None,
) -> tuple[np.ndarray, "SentenceTransformer"]:
    """Return chunk embeddings from memory cache, or compute them."""
    chunk_ids = tuple(chunk.chunk_id for chunk in chunks)
    cache_key = (model_name, chunk_ids)
    model = get_embedding_model(model_name)

    if not force_rebuild and cache_key in _chunk_embeddings_cache:
        return _chunk_embeddings_cache[cache_key], model

    embeddings = embed_chunks(model, chunks, show_progress_bar=show_progress_bar)
    _chunk_embeddings_cache[cache_key] = embeddings
    return embeddings, model


def clear_embedding_cache() -> None:
    """Clear in-memory model and embedding caches."""
    _chunk_embeddings_cache.clear()
    release_embedding_models()
