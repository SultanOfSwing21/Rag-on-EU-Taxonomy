import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from eu_taxonomy_rag.cache.chunk_cache import load_or_build_chunks
from eu_taxonomy_rag.core.models import Chunk
from eu_taxonomy_rag.retrieval.embeddings import get_or_build_chunk_embeddings

DEFAULT_OUTPUT_PATH = Path("data/evaluation/retrieval_golden_dataset.jsonl")

SYNONYMS: dict[str, str] = {
    "EU Taxonomy": "EU taxonomy framework",
    "Taxonomy": "Taxonomy Regulation",
    "technical screening criteria": "TSC",
    "TSC": "technical screening criteria",
    "DNSH": "do no significant harm",
    "do no significant harm": "DNSH",
    "CapEx": "capital expenditure",
    "OpEx": "operating expenditure",
    "undertakings": "companies",
    "companies": "undertakings",
    "reporting undertakings": "companies subject to reporting",
    "Climate Delegated Act": "climate delegated act",
    "Taxonomy-aligned": "aligned with the Taxonomy",
    "Taxonomy-eligible": "eligible under the Taxonomy",
}

SIMPLE_PREFIXES = (
    "Can you explain: {question}",
    "I need to understand: {question}",
    "Please clarify the following: {question}",
    "In the context of the EU Taxonomy, {question_lower}",
    "From a compliance perspective, {question_lower}",
    "For a reporting undertaking, {question_lower}",
)

COMPLEX_TEMPLATES_PAIR = (
    "How do {topic_a} and {topic_b} relate under the EU Taxonomy?",
    "What should companies know about both {topic_a} and {topic_b}?",
    "When dealing with {topic_a}, how does this interact with {topic_b}?",
    "I am assessing {topic_a} and {topic_b} — what are the key Taxonomy requirements?",
    "What is the relationship between {topic_a} and {topic_b} for Taxonomy reporting?",
)

COMPLEX_TEMPLATES_TRIPLE = (
    "How should undertakings address {topic_a}, {topic_b}, and {topic_c} together under the EU Taxonomy?",
    "What are the combined Taxonomy implications of {topic_a}, {topic_b}, and {topic_c}?",
    "For a business dealing with {topic_a}, {topic_b}, and {topic_c}, what rules apply?",
)


@dataclass(frozen=True)
class GoldenQuestion:
    question: str
    expected_chunk_ids: list[str]
    difficulty: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GenerationConfig:
    seed: int = 42
    target_simple: int = 500
    target_complex: int = 200
    neighbor_top_k: int = 8
    min_neighbor_similarity: float = 0.45
    dedup_similarity_threshold: float = 0.92
    embedding_model: str | None = None


@dataclass
class DatasetStats:
    total: int = 0
    simple_count: int = 0
    complex_count: int = 0
    unique_questions: int = 0
    unique_chunks_referenced: int = 0
    pair_complex: int = 0
    triple_complex: int = 0
    avg_expected_chunks_complex: float = 0.0
    sections_covered: list[str] = field(default_factory=list)
    generation_parameters: dict = field(default_factory=dict)


def compute_chunk_embeddings(
    chunks: list[Chunk],
    config: GenerationConfig | None = None,
) -> np.ndarray:
    """Calcule les embeddings des chunks (sentence-transformers ou fallback hash)."""
    config = config or GenerationConfig()
    return _compute_embeddings(chunks, config)


def generate_golden_dataset(
    chunks: list[Chunk],
    config: GenerationConfig | None = None,
) -> tuple[list[GoldenQuestion], DatasetStats]:
    """Génère le golden dataset de retrieval à partir des chunks FAQ."""
    config = config or GenerationConfig()
    rng = random.Random(config.seed)
    valid_ids = {chunk.chunk_id for chunk in chunks}
    chunk_map = {chunk.chunk_id: chunk for chunk in chunks}

    embeddings = _compute_embeddings(chunks, config)
    neighbors = _build_neighbor_map(chunks, embeddings, config)

    simple_candidates = _generate_simple_candidates(chunks, rng)
    complex_candidates = _generate_complex_candidates(chunks, chunk_map, neighbors, rng)

    simple = _deduplicate_questions(simple_candidates, config)[: config.target_simple]
    complex_ = _deduplicate_questions(complex_candidates, config)[: config.target_complex]
    dataset = simple + complex_

    _validate_dataset(dataset, valid_ids)
    stats = _build_stats(dataset, chunks, config)
    return dataset, stats


def save_golden_dataset(
    dataset: list[GoldenQuestion],
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for item in dataset:
            file.write(json.dumps(item.to_dict(), ensure_ascii=False))
            file.write("\n")

    return output_path


def load_golden_dataset(path: str | Path = DEFAULT_OUTPUT_PATH) -> list[GoldenQuestion]:
    path = Path(path)
    items: list[GoldenQuestion] = []

    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            items.append(
                GoldenQuestion(
                    question=data["question"],
                    expected_chunk_ids=data["expected_chunk_ids"],
                    difficulty=data["difficulty"],
                )
            )

    return items


def _generate_simple_candidates(
    chunks: list[Chunk],
    rng: random.Random,
) -> list[GoldenQuestion]:
    candidates: list[GoldenQuestion] = []

    for chunk in chunks:
        variants = _simple_variants(chunk, rng)
        for question in variants:
            candidates.append(
                GoldenQuestion(
                    question=question,
                    expected_chunk_ids=[chunk.chunk_id],
                    difficulty="simple",
                )
            )

    rng.shuffle(candidates)
    return candidates


def _simple_variants(chunk: Chunk, rng: random.Random) -> list[str]:
    question = chunk.question.strip()
    variants: list[str] = [question]

    lower = question[0].lower() + question[1:] if question else question
    variants.append(f"What I want to know is: {lower}")

    shortened = _shorten_question(question)
    if shortened and shortened != question:
        variants.append(shortened)

    synonym_version = _apply_synonyms(question)
    if synonym_version != question:
        variants.append(synonym_version)

    prefix = rng.choice(SIMPLE_PREFIXES)
    variants.append(prefix.format(question=question, question_lower=lower))

    if "?" not in question:
        variants.append(f"{question}?")

    section = chunk.metadata.get("section", "EU Taxonomy")
    variants.append(f"[{section}] {question}")

    seen: set[str] = set()
    unique: list[str] = []
    for variant in variants:
        normalized = _normalize_question(variant)
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(variant.strip())

    return unique


def _generate_complex_candidates(
    chunks: list[Chunk],
    chunk_map: dict[str, Chunk],
    neighbors: dict[str, list[tuple[str, float]]],
    rng: random.Random,
) -> list[GoldenQuestion]:
    candidates: list[GoldenQuestion] = []

    for anchor in chunks:
        related = neighbors.get(anchor.chunk_id, [])
        if not related:
            continue

        pair_neighbors = related[:3]
        for neighbor_id, _ in pair_neighbors:
            neighbor = chunk_map[neighbor_id]
            topic_a = _extract_topic(anchor.question)
            topic_b = _extract_topic(neighbor.question)
            template = rng.choice(COMPLEX_TEMPLATES_PAIR)
            question = template.format(topic_a=topic_a, topic_b=topic_b)
            chunk_ids = sorted({anchor.chunk_id, neighbor_id})
            candidates.append(
                GoldenQuestion(
                    question=question,
                    expected_chunk_ids=chunk_ids,
                    difficulty="complex",
                )
            )

        if len(related) >= 2 and rng.random() < 0.35:
            id_b, id_c = related[0][0], related[1][0]
            chunk_b, chunk_c = chunk_map[id_b], chunk_map[id_c]
            template = rng.choice(COMPLEX_TEMPLATES_TRIPLE)
            question = template.format(
                topic_a=_extract_topic(anchor.question),
                topic_b=_extract_topic(chunk_b.question),
                topic_c=_extract_topic(chunk_c.question),
            )
            chunk_ids = sorted({anchor.chunk_id, id_b, id_c})
            candidates.append(
                GoldenQuestion(
                    question=question,
                    expected_chunk_ids=chunk_ids,
                    difficulty="complex",
                )
            )

    rng.shuffle(candidates)
    return candidates


def _compute_embeddings(chunks: list[Chunk], config: GenerationConfig) -> np.ndarray:
    try:
        embeddings, _ = get_or_build_chunk_embeddings(
            chunks,
            model_name=config.embedding_model,
            force_rebuild=False,
        )
        return embeddings
    except (ImportError, ModuleNotFoundError, OSError):
        return _hash_embedding_fallback(chunks, config.seed)


def _hash_embedding_fallback(chunks: list[Chunk], seed: int, dim: int = 128) -> np.ndarray:
    """Fallback déterministe si sentence-transformers n'est pas disponible."""
    vectors = np.zeros((len(chunks), dim), dtype=np.float32)
    for row, chunk in enumerate(chunks):
        words = _normalize_question(chunk.text).split()
        for word in words:
            digest = hashlib.md5(f"{seed}:{word}".encode()).hexdigest()
            index = int(digest, 16) % dim
            vectors[row, index] += 1.0
        norm = np.linalg.norm(vectors[row])
        if norm > 0:
            vectors[row] /= norm
    return vectors


def _build_neighbor_map(
    chunks: list[Chunk],
    embeddings: np.ndarray,
    config: GenerationConfig,
) -> dict[str, list[tuple[str, float]]]:
    neighbor_map: dict[str, list[tuple[str, float]]] = {}

    for index, chunk in enumerate(chunks):
        scores = embeddings @ embeddings[index]
        ranked_indices = np.argsort(scores)[::-1]

        neighbors: list[tuple[str, float]] = []
        for neighbor_index in ranked_indices:
            if neighbor_index == index:
                continue

            neighbor_chunk = chunks[neighbor_index]
            score = float(scores[neighbor_index])
            if score < config.min_neighbor_similarity:
                continue

            same_section = chunk.metadata.get("section") == neighbor_chunk.metadata.get("section")
            if not same_section and score < config.min_neighbor_similarity + 0.15:
                continue

            neighbors.append((neighbor_chunk.chunk_id, score))
            if len(neighbors) >= config.neighbor_top_k:
                break

        neighbor_map[chunk.chunk_id] = neighbors

    return neighbor_map


def _extract_topic(question: str, max_words: int = 12) -> str:
    """Extrait un libellé court et lisible à partir d'une question FAQ."""
    topic = question.strip().rstrip("?")

    quoted = re.findall(r'"([^"]+)"', topic) or re.findall(r"“([^”]+)”", topic)
    if quoted:
        return quoted[0].strip().lower()

    topic = re.sub(
        r"^\s*(how|what|when|where|why|who|which|will|can|could|should|is|are|do|does)\s+",
        "",
        topic,
        flags=re.I,
    )

    lowered = topic.lower()
    for delimiter in (" mean ", " refers to ", " refer to ", " in section", " under the ", " pursuant to "):
        if delimiter in lowered:
            topic = topic[: lowered.index(delimiter)]
            break

    words = topic.split()
    if len(words) > max_words:
        topic = " ".join(words[:max_words])

    return topic.strip().lower() or "this topic"


def _shorten_question(question: str) -> str:
    parts = re.split(r"[;,]", question, maxsplit=1)
    if len(parts) > 1 and len(parts[0]) > 20:
        return parts[0].strip() + "?"
    words = question.split()
    if len(words) > 12:
        return " ".join(words[:12]).rstrip(",") + "?"
    return question


def _apply_synonyms(text: str) -> str:
    updated = text
    for source, target in SYNONYMS.items():
        if source.lower() in updated.lower():
            updated = re.sub(re.escape(source), target, updated, flags=re.IGNORECASE, count=1)
            break
    return updated


def _normalize_question(question: str) -> str:
    normalized = question.lower().strip()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _question_similarity(a: str, b: str) -> float:
    set_a = set(_normalize_question(a).split())
    set_b = set(_normalize_question(b).split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _deduplicate_questions(
    questions: list[GoldenQuestion],
    config: GenerationConfig,
) -> list[GoldenQuestion]:
    kept: list[GoldenQuestion] = []
    normalized_seen: set[str] = set()

    for item in questions:
        normalized = _normalize_question(item.question)
        if normalized in normalized_seen:
            continue

        if any(
            _question_similarity(item.question, existing.question) >= config.dedup_similarity_threshold
            for existing in kept
            if existing.difficulty == item.difficulty
        ):
            continue

        normalized_seen.add(normalized)
        kept.append(item)

    return kept


def _validate_dataset(dataset: list[GoldenQuestion], valid_ids: set[str]) -> None:
    for item in dataset:
        if not item.question.strip():
            raise ValueError("Question vide détectée dans le dataset.")
        if item.difficulty not in {"simple", "complex"}:
            raise ValueError(f"Difficulté invalide: {item.difficulty}")
        if not item.expected_chunk_ids:
            raise ValueError("expected_chunk_ids ne peut pas être vide.")
        for chunk_id in item.expected_chunk_ids:
            if chunk_id not in valid_ids:
                raise ValueError(f"chunk_id inconnu: {chunk_id}")


def _build_stats(
    dataset: list[GoldenQuestion],
    chunks: list[Chunk],
    config: GenerationConfig,
) -> DatasetStats:
    simple = [q for q in dataset if q.difficulty == "simple"]
    complex_ = [q for q in dataset if q.difficulty == "complex"]
    referenced = {chunk_id for q in dataset for chunk_id in q.expected_chunk_ids}
    sections = {
        chunks[[c.chunk_id for c in chunks].index(chunk_id)].metadata.get("section", "Unknown")
        for chunk_id in referenced
    }

    return DatasetStats(
        total=len(dataset),
        simple_count=len(simple),
        complex_count=len(complex_),
        unique_questions=len({_normalize_question(q.question) for q in dataset}),
        unique_chunks_referenced=len(referenced),
        pair_complex=sum(1 for q in complex_ if len(q.expected_chunk_ids) == 2),
        triple_complex=sum(1 for q in complex_ if len(q.expected_chunk_ids) == 3),
        avg_expected_chunks_complex=(
            sum(len(q.expected_chunk_ids) for q in complex_) / len(complex_) if complex_ else 0.0
        ),
        sections_covered=sorted(section for section in sections if section),
        generation_parameters={
            "seed": config.seed,
            "target_simple": config.target_simple,
            "target_complex": config.target_complex,
            "neighbor_top_k": config.neighbor_top_k,
            "min_neighbor_similarity": config.min_neighbor_similarity,
            "dedup_similarity_threshold": config.dedup_similarity_threshold,
            "source_chunks": len(chunks),
        },
    )
