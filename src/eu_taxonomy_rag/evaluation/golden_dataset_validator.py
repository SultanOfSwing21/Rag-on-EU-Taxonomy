import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from eu_taxonomy_rag.cache.chunk_cache import load_or_build_chunks
from eu_taxonomy_rag.core.models import Chunk
from eu_taxonomy_rag.evaluation.golden_dataset import (
    GoldenQuestion,
    compute_chunk_embeddings,
    load_golden_dataset,
    save_golden_dataset,
)

DEFAULT_INPUT_PATH = Path("data/evaluation/retrieval_golden_dataset.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/evaluation/retrieval_golden_dataset_cleaned.jsonl")

MIN_PAIRWISE_SIMILARITY = 0.35
MIN_TRIPLE_AVG_SIMILARITY = 0.30

BROKEN_PATTERNS = (
    r"\bhow do (should|does|are|is|will|can|there|my|about)\b",
    r"\bhow should (should|does|are|is|will|can)\b",
    r"\bwhat is the relationship between (should|does|are|is|will|can|there|my|to what)\b",
    r"\bwhen dealing with (should|does|are|is|will|what|how|to deal|my)\b",
    r"\bi am assessing (should|does|are|is|will|can|my|will companies|comprehensive)\b",
    r"\bwhat should companies know about both (is |about |does |are |should |my )",
    r"\bboth about .+ and about .+\?",
    r"\?\s+(will|can|should|does|are|is)\s+",
    r"\bfor a business dealing with (at what|should|does|are|is|will|to deal)\b",
    r"\bwhat are the combined taxonomy implications of overall\b",
)

INCOMPLETE_ENDINGS = (
    r"\b(for cars that are|that are|period of|what documentary|provide multiple|"
    r"below which undertakings are|for reaching the|when|where|which|that|and|or|of|to)\s*\?$",
    r"\bqualify as\s*\?$",
    r"\btreat exposures to companies that are not\s*\?$",
)

NATURAL_PAIR_TEMPLATES = (
    "How do {a} and {b} interact under EU Taxonomy rules?",
    "What Taxonomy requirements apply to both {a} and {b}?",
    "I am reporting on {a} and {b} — what should I know for Taxonomy compliance?",
    "What is the link between {a} and {b} under the EU Taxonomy?",
)

NATURAL_TRIPLE_TEMPLATES = (
    "What EU Taxonomy rules apply to {a}, {b}, and {c} together?",
    "How should undertakings address {a}, {b}, and {c} under the Taxonomy?",
    "I need guidance on {a}, {b}, and {c} for Taxonomy reporting — what applies?",
)


@dataclass
class ValidationReport:
    total_complex: int = 0
    accepted: int = 0
    rewritten: int = 0
    regenerated: int = 0
    removed: int = 0
    backfilled: int = 0
    target_complex: int = 200
    final_complex: int = 0
    final_total: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    removed_examples: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ValidationOutcome:
    is_valid: bool
    reasons: tuple[str, ...]


def clean_golden_dataset(
    dataset: list[GoldenQuestion],
    chunks: list[Chunk],
    *,
    seed: int = 42,
    target_complex: int = 200,
) -> tuple[list[GoldenQuestion], ValidationReport]:
    chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
    embeddings = compute_chunk_embeddings(chunks)
    id_to_index = {chunk.chunk_id: index for index, chunk in enumerate(chunks)}

    simple = [item for item in dataset if item.difficulty == "simple"]
    complex_items = [item for item in dataset if item.difficulty == "complex"]

    report = ValidationReport(total_complex=len(complex_items), target_complex=target_complex)
    rng = random.Random(seed)
    cleaned_complex: list[GoldenQuestion] = []
    seen_questions: set[str] = set()

    for item in complex_items:
        outcome = validate_complex_question(
            item.question,
            item.expected_chunk_ids,
            chunk_map,
            embeddings,
            id_to_index,
        )

        if outcome.is_valid:
            natural = rewrite_complex_question(item.expected_chunk_ids, chunk_map, rng)
            natural_outcome = validate_complex_question(
                natural.question,
                natural.expected_chunk_ids,
                chunk_map,
                embeddings,
                id_to_index,
            )
            final_item = natural if natural_outcome.is_valid else item
            normalized = _normalize(final_item.question)
            if normalized in seen_questions:
                _record_rejection(report, "duplicate_after_cleaning")
                continue
            seen_questions.add(normalized)
            cleaned_complex.append(final_item)
            if final_item.question != item.question:
                report.rewritten += 1
            else:
                report.accepted += 1
            continue

        regenerated = _try_regenerate(
            item.expected_chunk_ids,
            chunk_map,
            embeddings,
            id_to_index,
            rng,
        )
        if regenerated is not None:
            normalized = _normalize(regenerated.question)
            if normalized not in seen_questions:
                seen_questions.add(normalized)
                cleaned_complex.append(regenerated)
                report.regenerated += 1
                continue

        report.removed += 1
        for reason in outcome.reasons:
            _record_rejection(report, reason)
        if len(report.removed_examples) < 10:
            report.removed_examples.append(
                {
                    "question": item.question,
                    "expected_chunk_ids": item.expected_chunk_ids,
                    "reasons": list(outcome.reasons),
                }
            )

    cleaned_complex = _backfill_complex_questions(
        cleaned_complex,
        chunks,
        chunk_map,
        embeddings,
        id_to_index,
        rng,
        target_complex,
        seen_questions,
        report,
    )

    cleaned = simple + cleaned_complex
    report.final_complex = len(cleaned_complex)
    report.final_total = len(cleaned)
    return cleaned, report


def validate_complex_question(
    question: str,
    expected_chunk_ids: list[str],
    chunk_map: dict[str, Chunk],
    embeddings: np.ndarray,
    id_to_index: dict[str, int],
) -> ValidationOutcome:
    reasons: list[str] = []
    text = question.strip()

    if not text.endswith("?"):
        reasons.append("missing_question_mark")
    if len(text) < 25:
        reasons.append("too_short")
    if len(text) > 220:
        reasons.append("too_long")
    if text.count("?") > 1:
        reasons.append("multiple_questions")

    for pattern in BROKEN_PATTERNS:
        if re.search(pattern, text, flags=re.I):
            reasons.append("grammatically_broken")
            break

    for pattern in INCOMPLETE_ENDINGS:
        if re.search(pattern, text, flags=re.I):
            reasons.append("incomplete_or_truncated")
            break

    labels = [_chunk_label(chunk_map[cid]) for cid in expected_chunk_ids if cid in chunk_map]
    if _has_duplicate_labels(labels):
        reasons.append("duplicate_topic")

    if _looks_like_faq_concatenation(text, chunk_map, expected_chunk_ids):
        reasons.append("faq_title_concatenation")

    if not _is_realistic_query(text):
        reasons.append("unrealistic_query")

    if any(_is_bad_label(label) for label in labels):
        reasons.append("low_quality_topic_labels")

    if not _chunks_are_related(expected_chunk_ids, embeddings, id_to_index, chunk_map):
        reasons.append("semantically_unrelated_chunks")

    return ValidationOutcome(is_valid=not reasons, reasons=tuple(reasons))


def rewrite_complex_question(
    chunk_ids: list[str],
    chunk_map: dict[str, Chunk],
    rng: random.Random,
) -> GoldenQuestion:
    labels = [_format_label(_chunk_label(chunk_map[cid])) for cid in chunk_ids]
    template = _pick_template(chunk_ids, len(labels), rng)
    question = template.format(**_label_kwargs(labels))
    return GoldenQuestion(
        question=question,
        expected_chunk_ids=list(chunk_ids),
        difficulty="complex",
    )


def _try_regenerate(
    chunk_ids: list[str],
    chunk_map: dict[str, Chunk],
    embeddings: np.ndarray,
    id_to_index: dict[str, int],
    rng: random.Random,
) -> GoldenQuestion | None:
    templates = NATURAL_TRIPLE_TEMPLATES if len(chunk_ids) >= 3 else NATURAL_PAIR_TEMPLATES
    labels = [_format_label(_chunk_label(chunk_map[cid])) for cid in chunk_ids]

    for offset in range(len(templates)):
        template = templates[(hash(tuple(chunk_ids)) + offset) % len(templates)]
        question = template.format(**_label_kwargs(labels))
        candidate = GoldenQuestion(
            question=question,
            expected_chunk_ids=list(chunk_ids),
            difficulty="complex",
        )
        outcome = validate_complex_question(
            candidate.question,
            candidate.expected_chunk_ids,
            chunk_map,
            embeddings,
            id_to_index,
        )
        if outcome.is_valid:
            return candidate

    return None


def _backfill_complex_questions(
    cleaned_complex: list[GoldenQuestion],
    chunks: list[Chunk],
    chunk_map: dict[str, Chunk],
    embeddings: np.ndarray,
    id_to_index: dict[str, int],
    rng: random.Random,
    target: int,
    seen_questions: set[str],
    report: ValidationReport,
) -> list[GoldenQuestion]:
    if len(cleaned_complex) >= target:
        return cleaned_complex

    existing_sets = {tuple(sorted(item.expected_chunk_ids)) for item in cleaned_complex}
    candidates: list[list[str]] = []

    for i, left in enumerate(chunks):
        scores = embeddings @ embeddings[i]
        ranked = np.argsort(scores)[::-1]
        for j in ranked:
            if j == i:
                continue
            if scores[j] < MIN_PAIRWISE_SIMILARITY:
                break
            pair = sorted([left.chunk_id, chunks[j].chunk_id])
            if tuple(pair) not in existing_sets:
                candidates.append(pair)

    rng.shuffle(candidates)
    for chunk_ids in candidates:
        if len(cleaned_complex) >= target:
            break
        if tuple(sorted(chunk_ids)) in existing_sets:
            continue

        regenerated = _try_regenerate(
            chunk_ids,
            chunk_map,
            embeddings,
            id_to_index,
            rng,
        )
        if regenerated is None:
            continue

        normalized = _normalize(regenerated.question)
        if normalized in seen_questions:
            continue

        seen_questions.add(normalized)
        existing_sets.add(tuple(sorted(chunk_ids)))
        cleaned_complex.append(regenerated)
        report.backfilled += 1

    return cleaned_complex


def _chunk_label(chunk: Chunk) -> str:
    quoted = re.findall(r'"([^"]+)"', chunk.question) or re.findall(r"“([^”]+)”", chunk.question)
    for candidate in quoted:
        candidate = candidate.strip()
        if candidate and not _is_bad_label(candidate):
            return candidate

    topic = chunk.question.strip().rstrip("?")
    topic = re.sub(
        r"^\s*(how|what|when|where|why|who|which|will|can|could|should|is|are|do|does)\s+",
        "",
        topic,
        flags=re.I,
    )
    lowered = topic.lower()
    for delimiter in (" mean ", " refers to ", " in section", " under the "):
        if delimiter in lowered:
            topic = topic[: lowered.index(delimiter)]
            break

    words = topic.split()
    if len(words) > 8:
        topic = " ".join(words[:8])

    label = topic.strip()
    if _is_bad_label(label):
        section = chunk.metadata.get("section", "EU Taxonomy")
        index = chunk.metadata.get("index", "?")
        return f"{section} topic {index}"

    return label


def _is_bad_label(label: str) -> bool:
    if len(label.strip()) < 8:
        return True
    return bool(
        re.match(
            r"^(should|does|are|is|will|can|to what|my|there|when|where|how|about|for)\b",
            label.strip(),
            flags=re.I,
        )
    )


def _format_label(label: str) -> str:
    label = label.strip()
    if not label:
        return "this topic"
    return label[0].lower() + label[1:] if len(label) > 1 else label.lower()


def _label_kwargs(labels: list[str]) -> dict[str, str]:
    keys = ["a", "b", "c"]
    return {keys[index]: labels[index] for index in range(len(labels))}


def _pick_template(chunk_ids: list[str], count: int, rng: random.Random) -> str:
    templates = NATURAL_TRIPLE_TEMPLATES if count >= 3 else NATURAL_PAIR_TEMPLATES
    digest = int(hashlib.md5("|".join(chunk_ids).encode()).hexdigest(), 16)
    return templates[digest % len(templates)]


def _has_duplicate_labels(labels: list[str]) -> bool:
    normalized = [_normalize(label) for label in labels if label]
    return len(normalized) != len(set(normalized))


def _looks_like_faq_concatenation(
    question: str,
    chunk_map: dict[str, Chunk],
    chunk_ids: list[str],
) -> bool:
    normalized_q = _normalize(question)
    long_overlap = 0
    for chunk_id in chunk_ids:
        chunk = chunk_map.get(chunk_id)
        if chunk is None:
            continue
        chunk_start = _normalize(chunk.question)[:60]
        if chunk_start and chunk_start in normalized_q:
            long_overlap += 1
    if long_overlap >= 2:
        return True

    if re.search(r"\b(should|does|are|is|will|can|my company)\s+\w+.*\b(and|,\s)\s*(should|does|are|is|will|can|my)\b", question, re.I):
        return True

    return False


def _is_realistic_query(question: str) -> bool:
    words = question.split()
    if len(words) > 42:
        return False
    if question.count(",") > 5:
        return False
    if re.search(r"\bwhat does that mean for my\b", question, re.I):
        return False
    return True


def _chunks_are_related(
    chunk_ids: list[str],
    embeddings: np.ndarray,
    id_to_index: dict[str, int],
    chunk_map: dict[str, Chunk],
) -> bool:
    if len(chunk_ids) < 2:
        return False

    sections = {chunk_map[cid].metadata.get("section") for cid in chunk_ids if cid in chunk_map}
    if len(sections) == 1 and None not in sections:
        return True

    vectors = []
    for chunk_id in chunk_ids:
        index = id_to_index.get(chunk_id)
        if index is None:
            return False
        vectors.append(embeddings[index])

    pairwise = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            pairwise.append(float(np.dot(vectors[i], vectors[j])))

    if not pairwise:
        return False

    if len(chunk_ids) == 2:
        return pairwise[0] >= MIN_PAIRWISE_SIMILARITY

    return (sum(pairwise) / len(pairwise)) >= MIN_TRIPLE_AVG_SIMILARITY and max(pairwise) >= MIN_PAIRWISE_SIMILARITY


def _normalize(text: str) -> str:
    normalized = text.lower().strip()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _record_rejection(report: ValidationReport, reason: str) -> None:
    report.rejection_reasons[reason] = report.rejection_reasons.get(reason, 0) + 1


def run_cleaning(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    seed: int = 42,
) -> tuple[list[GoldenQuestion], ValidationReport]:
    chunks = load_or_build_chunks()
    dataset = load_golden_dataset(input_path)
    cleaned, report = clean_golden_dataset(dataset, chunks, seed=seed)
    save_golden_dataset(cleaned, output_path)
    return cleaned, report
