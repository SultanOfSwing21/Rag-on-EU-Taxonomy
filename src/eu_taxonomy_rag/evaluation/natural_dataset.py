import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from eu_taxonomy_rag.cache.chunk_cache import load_or_build_chunks
from eu_taxonomy_rag.core.models import Chunk
from eu_taxonomy_rag.evaluation.golden_dataset import GoldenQuestion, load_golden_dataset
from eu_taxonomy_rag.llm.client import ChatClient, OpenAIChatClient

DEFAULT_SOURCE_PATH = Path("data/evaluation/retrieval_golden_dataset_cleaned.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/evaluation/natural_user_queries.jsonl")
NATURAL_DATASET_748_PATH = Path("data/evaluation/natural_user_queries_748.jsonl")

SYSTEM_PROMPT = """You rewrite EU Taxonomy FAQ benchmark questions into realistic user queries for a RAG chatbot.

Rules:
- Preserve the original intent exactly.
- Keep the same scope (simple = one topic, complex = multiple related topics).
- Write ONE natural question only.
- Use plain business language (sustainability, finance, reporting, consulting, or end-user tone as instructed).
- Do NOT copy regulatory wording verbatim from the source.
- Do NOT mention chunk IDs, FAQ numbers, or that this is a rewrite.
- Do NOT add markdown, quotes, labels, or explanations.
- Output only the final user question."""

PERSONAS: tuple[tuple[str, str], ...] = (
    (
        "sustainability_officer",
        "a sustainability reporting officer at a listed EU company preparing CSRD and Taxonomy disclosures",
    ),
    (
        "finance_team",
        "a finance controller working on EU Taxonomy KPIs (turnover, CapEx, OpEx)",
    ),
    (
        "reporting_team",
        "a regulatory reporting specialist preparing Taxonomy eligibility and alignment tables",
    ),
    (
        "consultant",
        "an ESG consultant advising corporate clients on Taxonomy compliance",
    ),
    (
        "disclosure_preparer",
        "a group reporting manager preparing first-time Taxonomy disclosures",
    ),
    (
        "bank_analyst",
        "a sustainable finance analyst at a bank reviewing portfolio Taxonomy exposure",
    ),
    (
        "direct",
        "a business user asking a direct practical question with minimal context",
    ),
)


@dataclass(frozen=True)
class NaturalQuery:
    question: str
    expected_chunk_ids: list[str]
    difficulty: str
    query_type: str | None = None
    persona: str | None = None
    similarity_score: float | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        return {key: value for key, value in payload.items() if value is not None}


@dataclass
class NaturalDatasetStats:
    source_path: str
    output_path: str
    selected_simple: int = 0
    selected_complex: int = 0
    generated_simple: int = 0
    generated_complex: int = 0
    retries: int = 0
    model: str = "gpt-4o-mini"
    seed: int = 42
    personas_used: list[str] = field(default_factory=list)


def select_source_questions(
    dataset: list[GoldenQuestion],
    *,
    n_simple: int = 200,
    n_complex: int = 100,
    seed: int = 42,
) -> list[GoldenQuestion]:
    """Sélectionne des questions sources diversifiées depuis le golden dataset nettoyé."""
    rng = random.Random(seed)
    simple = [item for item in dataset if item.difficulty == "simple"]
    complex_ = [item for item in dataset if item.difficulty == "complex"]

    selected_simple = _select_diverse(simple, n_simple, rng, key=lambda item: tuple(item.expected_chunk_ids))
    selected_complex = _select_diverse(complex_, n_complex, rng, key=lambda item: tuple(sorted(item.expected_chunk_ids)))

    rng.shuffle(selected_simple)
    rng.shuffle(selected_complex)
    return selected_simple + selected_complex


def generate_natural_query(
    source: GoldenQuestion,
    chunk_map: dict[str, Chunk],
    client: ChatClient,
    persona: tuple[str, str],
) -> str:
    """Réécrit une question source en requête utilisateur naturelle via LLM."""
    persona_id, persona_description = persona
    context = _build_chunk_context(source.expected_chunk_ids, chunk_map)
    user_prompt = _build_user_prompt(source, context, persona_description, persona_id)
    question = client.complete(SYSTEM_PROMPT, user_prompt)
    return _clean_llm_output(question)


def generate_natural_dataset(
    sources: list[GoldenQuestion],
    chunks: list[Chunk],
    client: ChatClient,
    *,
    seed: int = 42,
    max_retries: int = 3,
) -> tuple[list[NaturalQuery], NaturalDatasetStats]:
    chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
    rng = random.Random(seed)
    personas = list(PERSONAS)

    stats = NaturalDatasetStats(
        source_path=str(DEFAULT_SOURCE_PATH),
        output_path=str(DEFAULT_OUTPUT_PATH),
        seed=seed,
    )
    results: list[NaturalQuery] = []

    for source in sources:
        if source.difficulty == "simple":
            stats.selected_simple += 1
        else:
            stats.selected_complex += 1

        rewritten: str | None = None
        chosen_persona: tuple[str, str] | None = None
        for attempt in range(max_retries):
            persona = personas[(hash(source.question) + attempt) % len(personas)]
            if persona[0] not in stats.personas_used:
                stats.personas_used.append(persona[0])

            candidate = generate_natural_query(source, chunk_map, client, persona)
            if _is_acceptable_rewrite(source.question, candidate):
                rewritten = candidate
                chosen_persona = persona
                break
            stats.retries += 1

        if rewritten is None or chosen_persona is None:
            raise ValueError(f"Impossible de générer une requête naturelle valide pour: {source.question[:80]}")

        query_type = "natural_multihop" if source.difficulty == "complex" else "natural_simple"
        results.append(
            NaturalQuery(
                question=rewritten,
                expected_chunk_ids=list(source.expected_chunk_ids),
                difficulty=source.difficulty,
                query_type=query_type,
                persona=chosen_persona[0],
            )
        )
        if source.difficulty == "simple":
            stats.generated_simple += 1
        else:
            stats.generated_complex += 1

    return results, stats


def save_natural_dataset(dataset: list[NaturalQuery], output_path: str | Path = DEFAULT_OUTPUT_PATH) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for item in dataset:
            file.write(json.dumps(item.to_dict(), ensure_ascii=False))
            file.write("\n")

    return output_path


def load_natural_dataset(path: str | Path = DEFAULT_OUTPUT_PATH) -> list[NaturalQuery]:
    path = Path(path)
    items: list[NaturalQuery] = []

    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            items.append(
                NaturalQuery(
                    question=data["question"],
                    expected_chunk_ids=data["expected_chunk_ids"],
                    difficulty=data["difficulty"],
                    query_type=data.get("query_type"),
                    persona=data.get("persona"),
                    similarity_score=data.get("similarity_score"),
                )
            )

    return items


def run_natural_dataset_generation(
    source_path: str | Path = DEFAULT_SOURCE_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    n_simple: int = 200,
    n_complex: int = 100,
    seed: int = 42,
    model: str = "gpt-4o-mini",
    client: ChatClient | None = None,
) -> tuple[list[NaturalQuery], NaturalDatasetStats]:
    source_dataset = load_golden_dataset(source_path)
    chunks = load_or_build_chunks()
    sources = select_source_questions(source_dataset, n_simple=n_simple, n_complex=n_complex, seed=seed)

    llm_client = client or OpenAIChatClient(model=model)
    dataset, stats = generate_natural_dataset(sources, chunks, llm_client, seed=seed)
    stats.source_path = str(source_path)
    stats.output_path = str(output_path)
    stats.model = model
    save_natural_dataset(dataset, output_path)
    return dataset, stats


def _select_diverse(
    items: list[GoldenQuestion],
    target: int,
    rng: random.Random,
    key,
) -> list[GoldenQuestion]:
    grouped: dict[tuple[str, ...], list[GoldenQuestion]] = {}
    for item in items:
        grouped.setdefault(key(item), []).append(item)

    keys = list(grouped.keys())
    rng.shuffle(keys)

    selected: list[GoldenQuestion] = []
    for group_key in keys:
        if len(selected) >= target:
            break
        selected.append(rng.choice(grouped[group_key]))

    if len(selected) < target:
        remaining = [item for item in items if item not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: target - len(selected)])

    return selected[:target]


def _build_chunk_context(chunk_ids: list[str], chunk_map: dict[str, Chunk]) -> str:
    blocks: list[str] = []
    for chunk_id in chunk_ids:
        chunk = chunk_map.get(chunk_id)
        if chunk is None:
            continue
        section = chunk.metadata.get("section", "Unknown section")
        answer_preview = chunk.answer[:220].replace("\n", " ")
        blocks.append(
            f"- [{chunk_id}] Section: {section}\n"
            f"  FAQ topic: {chunk.question}\n"
            f"  Answer preview: {answer_preview}"
        )
    return "\n".join(blocks)


def _build_user_prompt(
    source: GoldenQuestion,
    context: str,
    persona_description: str,
    persona_id: str,
) -> str:
    scope = (
        "This is a multi-topic question. The user needs information that combines several related Taxonomy topics."
        if source.difficulty == "complex"
        else "This is a single-topic question."
    )
    return (
        f"Persona: {persona_description}\n"
        f"Style id: {persona_id}\n"
        f"Difficulty: {source.difficulty}\n"
        f"{scope}\n\n"
        f"Source benchmark question:\n{source.question}\n\n"
        f"Underlying FAQ context (for intent only — do not copy wording):\n{context}\n\n"
        "Rewrite as a realistic chatbot user query."
    )


def _clean_llm_output(text: str) -> str:
    cleaned = text.strip().strip('"').strip("'")
    cleaned = re.sub(r"^(question|user query)\s*:\s*", "", cleaned, flags=re.I)
    return cleaned.strip()


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _word_overlap(a: str, b: str) -> float:
    words_a = set(_normalize(a).split())
    words_b = set(_normalize(b).split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def _is_acceptable_rewrite(source: str, rewritten: str) -> bool:
    if len(rewritten) < 20:
        return False
    if _normalize(source) == _normalize(rewritten):
        return False
    if _word_overlap(source, rewritten) > 0.72:
        return False
    if rewritten.count("?") > 2:
        return False
    return True
