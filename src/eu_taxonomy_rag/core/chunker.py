from eu_taxonomy_rag.core.models import Chunk, FAQ


def build_chunks(faqs: list[FAQ]) -> list[Chunk]:
    """Transforme une liste de FAQ en chunks (1 FAQ = 1 chunk)."""
    chunks = [faq_to_chunk(faq) for faq in faqs]
    _ensure_unique_chunk_ids(chunks)
    return chunks


def faq_to_chunk(faq: FAQ) -> Chunk:
    """Convertit une FAQ en chunk avec un identifiant unique."""
    index = faq.metadata.get("index")
    if index is None:
        raise ValueError("La FAQ doit contenir un index dans ses métadonnées.")

    metadata = dict(faq.metadata)

    return Chunk(
        chunk_id=make_chunk_id(int(index)),
        question=faq.question,
        answer=faq.answer,
        metadata=metadata,
    )


def make_chunk_id(index: int) -> str:
    """Génère un identifiant stable à partir de la position dans le fichier."""
    return f"faq-{index:04d}"


def _ensure_unique_chunk_ids(chunks: list[Chunk]) -> None:
    ids = [chunk.chunk_id for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Des chunk_id dupliqués ont été détectés.")
