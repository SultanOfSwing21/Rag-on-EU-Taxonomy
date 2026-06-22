import json
from pathlib import Path

from eu_taxonomy_rag.core.chunker import build_chunks
from eu_taxonomy_rag.core.models import Chunk
from eu_taxonomy_rag.core.parser import parse_faq_file

DEFAULT_FAQ_PATH = Path("data/taxonomy_faqs_cleaned.md")
DEFAULT_CACHE_PATH = Path(".cache/chunks.jsonl")


def load_or_build_chunks(
    faq_path: str | Path = DEFAULT_FAQ_PATH,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
    *,
    force_rebuild: bool = False,
) -> list[Chunk]:
    """Charge les chunks depuis le cache, ou les reconstruit depuis le fichier FAQ."""
    cache_path = Path(cache_path)

    if not force_rebuild and cache_path.exists():
        return load_chunks(cache_path)

    chunks = build_chunks_from_file(faq_path)
    save_chunks(chunks, cache_path)
    return chunks


def build_chunks_from_file(faq_path: str | Path) -> list[Chunk]:
    """Parse le fichier FAQ et génère les chunks."""
    return build_chunks(parse_faq_file(faq_path))


def save_chunks(chunks: list[Chunk], cache_path: str | Path) -> None:
    """Écrit les chunks au format JSONL."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with cache_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk_to_dict(chunk), ensure_ascii=False))
            file.write("\n")


def load_chunks(cache_path: str | Path) -> list[Chunk]:
    """Lit les chunks depuis un fichier JSONL."""
    cache_path = Path(cache_path)
    chunks: list[Chunk] = []

    with cache_path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                chunks.append(chunk_from_dict(json.loads(line)))

    return chunks


def chunk_to_dict(chunk: Chunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "question": chunk.question,
        "answer": chunk.answer,
        "metadata": chunk.metadata,
    }


def chunk_from_dict(data: dict) -> Chunk:
    return Chunk(
        chunk_id=data["chunk_id"],
        question=data["question"],
        answer=data["answer"],
        metadata=data.get("metadata", {}),
    )
