from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class DenseModelConfig:
    key: str
    model_name: str


DENSE_MODELS: dict[str, DenseModelConfig] = {
    "minilm": DenseModelConfig(
        key="minilm",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    ),
    "mpnet": DenseModelConfig(
        key="mpnet",
        model_name="sentence-transformers/all-mpnet-base-v2",
    ),
}


class RetrievalMethod(str, Enum):
    DENSE_MINILM = "dense_minilm"
    DENSE_MPNET = "dense_mpnet"
    BM25 = "bm25"
    HYBRID_MINILM = "hybrid_minilm"
    HYBRID_MPNET = "hybrid_mpnet"


HYBRID_METHODS: dict[RetrievalMethod, str] = {
    RetrievalMethod.HYBRID_MINILM: "minilm",
    RetrievalMethod.HYBRID_MPNET: "mpnet",
}


def is_dense_method(method: RetrievalMethod) -> bool:
    return method in {RetrievalMethod.DENSE_MINILM, RetrievalMethod.DENSE_MPNET}


def is_hybrid_method(method: RetrievalMethod) -> bool:
    return method in HYBRID_METHODS


def requires_sentence_transformers(method: RetrievalMethod) -> bool:
    return is_dense_method(method) or is_hybrid_method(method)


def available_retrieval_methods() -> tuple[RetrievalMethod, ...]:
    """Methods that can run in the current environment."""
    from eu_taxonomy_rag.retrieval.embeddings import is_sentence_transformers_available

    if is_sentence_transformers_available():
        return tuple(RetrievalMethod)
    return (RetrievalMethod.BM25,)


def dense_key_for_method(method: RetrievalMethod) -> str:
    if method == RetrievalMethod.DENSE_MINILM:
        return "minilm"
    if method == RetrievalMethod.DENSE_MPNET:
        return "mpnet"
    if method in HYBRID_METHODS:
        return HYBRID_METHODS[method]
    raise ValueError(f"{method.value} n'est pas une méthode dense ou hybride.")
