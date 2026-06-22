from pathlib import Path

from eu_taxonomy_rag.core.models import FAQ


def parse_faq_file(path: str | Path) -> list[FAQ]:
    """Parse le fichier markdown des FAQ EU Taxonomy."""
    return parse_faq_text(Path(path).read_text(encoding="utf-8"), source=str(path))


def parse_faq_text(text: str, source: str | None = None) -> list[FAQ]:
    """Parse le contenu markdown et retourne une liste de FAQ."""
    faqs: list[FAQ] = []
    current_section: str | None = None
    current_question: str | None = None
    answer_lines: list[str] = []

    def flush() -> None:
        nonlocal current_question, answer_lines
        if current_question is None:
            return

        answer = _normalize_answer(answer_lines)
        if not answer:
            return

        metadata: dict[str, object] = {
            "section": current_section,
            "index": len(faqs) + 1,
        }
        if source is not None:
            metadata["source"] = source

        faqs.append(
            FAQ(
                question=current_question,
                answer=answer,
                metadata=metadata,
            )
        )
        current_question = None
        answer_lines = []

    for line in text.splitlines():
        if _is_section_header(line):
            flush()
            current_section = line[2:].strip()
            continue

        if line.startswith("### "):
            title = line[3:].strip()
            if _is_faq_question(title):
                flush()
                current_question = title
            elif current_question is not None:
                answer_lines.append(title)
            continue

        if current_question is not None:
            answer_lines.append(line)

    flush()
    return faqs


def _is_section_header(line: str) -> bool:
    return line.startswith("## ") and not line.startswith("### ")


def _is_faq_question(title: str) -> bool:
    """Distingue une vraie question FAQ d'un sous-titre interne à une réponse."""
    if title.endswith("?"):
        return True

    question_starters = (
        "how ",
        "what ",
        "when ",
        "where ",
        "why ",
        "who ",
        "which ",
        "will ",
        "can ",
        "could ",
        "should ",
        "would ",
        "is ",
        "are ",
        "do ",
        "does ",
        "did ",
        "has ",
        "have ",
        "must ",
        "may ",
    )
    return title.lower().startswith(question_starters)


def _normalize_answer(lines: list[str]) -> str:
    """Regroupe les lignes en paragraphes séparés par une ligne vide."""
    paragraphs: list[str] = []
    current: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        current.append(stripped)

    if current:
        paragraphs.append(" ".join(current))

    return "\n\n".join(paragraphs)
