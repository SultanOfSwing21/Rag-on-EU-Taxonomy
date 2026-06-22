from eu_taxonomy_rag.core.models import RetrievedChunk, RetrievalResult

SYSTEM_PROMPT = """You are a helpful assistant specialised in the EU Taxonomy FAQ.

Rules:
- Answer ONLY using the context provided below.
- Do not use outside knowledge.
- If the context does not contain enough information, reply exactly:
  "I cannot answer this question from the available context."
- When relevant, cite the chunk IDs used in your answer (e.g. [faq-0001]).
- Be concise and precise."""


def format_context(chunks: tuple[RetrievedChunk, ...] | list[RetrievedChunk]) -> str:
    """Formate les chunks récupérés pour injection dans le prompt."""
    if not chunks:
        return "No context retrieved."

    blocks: list[str] = []
    for item in chunks:
        section = item.chunk.metadata.get("section", "Unknown section")
        blocks.append(
            f"[{item.chunk.chunk_id}] (section: {section})\n"
            f"Question: {item.chunk.question}\n"
            f"Answer: {item.chunk.answer}"
        )
    return "\n\n---\n\n".join(blocks)


def build_rag_prompt(query: str, retrieval: RetrievalResult) -> str:
    """Construit le prompt utilisateur avec contexte + question."""
    context = format_context(retrieval.chunks)
    return (
        "Context:\n"
        f"{context}\n\n"
        "Question:\n"
        f"{query.strip()}\n\n"
        "Answer using ONLY the context above."
    )


def build_rag_messages(query: str, retrieval: RetrievalResult) -> list[dict[str, str]]:
    """Construit les messages au format chat (OpenAI)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_rag_prompt(query, retrieval)},
    ]
