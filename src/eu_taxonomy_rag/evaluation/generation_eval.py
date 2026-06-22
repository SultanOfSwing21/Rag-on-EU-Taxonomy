"""NLI-based groundedness / faithfulness evaluation for generated RAG answers.

This module provides a **diagnostic** faithfulness metric: it splits an answer into
short claims and checks each claim against retrieved context with a lightweight NLI
model. It is useful for monitoring and debugging, not as a perfect automatic judge.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any, Protocol

from eu_taxonomy_rag.core.models import Chunk

logger = logging.getLogger(__name__)

DEFAULT_NLI_MODEL = "typeform/distilbert-base-uncased-mnli"
MIN_CLAIM_LENGTH = 10
ENTAILMENT_THRESHOLD = 0.45
CONTRADICTION_THRESHOLD = 0.55
LEXICAL_OVERLAP_THRESHOLD = 0.80
CITATION_PATTERN = re.compile(r"\[faq-\d+\]", re.IGNORECASE)
CANONICAL_ABSTENTION = "i cannot answer this question from the available context"
ABSTENTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^i can(?:not|'t) answer(?: this question)? from the available context\.?$",
        re.IGNORECASE,
    ),
    re.compile(r"^i do not have enough information in the (?:provided )?context", re.IGNORECASE),
    re.compile(r"^the (?:available )?context does not contain enough information", re.IGNORECASE),
)

LABEL_SUPPORTED = "supported"
LABEL_CONTRADICTED = "contradicted"
LABEL_NOT_ENOUGH_INFO = "not_enough_info"

MNLI_LABEL_MAP = {
    "entailment": LABEL_SUPPORTED,
    "contradiction": LABEL_CONTRADICTED,
    "neutral": LABEL_NOT_ENOUGH_INFO,
    "label_0": LABEL_CONTRADICTED,
    "label_1": LABEL_NOT_ENOUGH_INFO,
    "label_2": LABEL_SUPPORTED,
}


def is_generation_eval_enabled() -> bool:
    """Return True when groundedness evaluation is enabled (``ENABLE_GENERATION_EVAL``)."""
    import os

    return os.environ.get("ENABLE_GENERATION_EVAL", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass(frozen=True)
class ClaimEvaluation:
    """Evaluation of a single atomic claim against retrieved context."""

    claim: str
    label: str
    confidence: float
    claim_score: float
    best_chunk_id: str | None = None
    best_chunk_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GenerationEvaluationResult:
    """Aggregate groundedness metrics for one generated answer."""

    faithfulness_score: float
    contradiction_rate: float
    unsupported_rate: float
    num_claims: int
    supported_claims: int
    contradicted_claims: int
    unsupported_claims: int
    best_claim_score: float
    avg_claim_score: float
    score_range: float
    claims: tuple[ClaimEvaluation, ...] = field(default_factory=tuple)
    warning: str | None = None
    evaluation_failed: bool = False
    abstention_response: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["claims"] = [claim.to_dict() for claim in self.claims]
        return payload


class NLIClassifier(Protocol):
    def predict(self, premise: str, hypothesis: str) -> dict[str, float]:
        """Return label probabilities keyed by canonical label names."""


def _normalize_text(text: str) -> str:
    cleaned = CITATION_PATTERN.sub("", text or "").lower()
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _split_into_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    sentences: list[str] = []
    for part in parts:
        sentence = re.sub(r"\s+", " ", part).strip(" -•")
        if len(sentence) >= MIN_CLAIM_LENGTH:
            sentences.append(sentence)
    return sentences


def _lexical_support_score(claim: str, premise: str) -> float | None:
    """Fast path for near-verbatim overlap between a claim and a context passage."""
    claim_norm = _normalize_text(claim)
    premise_norm = _normalize_text(premise)
    if not claim_norm or not premise_norm:
        return None
    if claim_norm in premise_norm or premise_norm in claim_norm:
        return 1.0

    claim_tokens = claim_norm.split()
    premise_tokens = set(premise_norm.split())
    if not claim_tokens:
        return None
    overlap = sum(1 for token in claim_tokens if token in premise_tokens) / len(claim_tokens)
    if overlap >= LEXICAL_OVERLAP_THRESHOLD:
        return overlap
    return None


def _label_from_probs(probs: dict[str, float]) -> tuple[str, float]:
    """Pick a label from one NLI distribution (per best-matching passage)."""
    supported = probs[LABEL_SUPPORTED]
    contradicted = probs[LABEL_CONTRADICTED]
    neutral = probs[LABEL_NOT_ENOUGH_INFO]

    if supported >= ENTAILMENT_THRESHOLD and supported >= contradicted:
        return LABEL_SUPPORTED, supported
    if contradicted >= CONTRADICTION_THRESHOLD and contradicted > supported:
        return LABEL_CONTRADICTED, contradicted
    return LABEL_NOT_ENOUGH_INFO, max(neutral, supported, contradicted)


def _premises_for_chunk(chunk: Chunk) -> list[str]:
    """Build short premise variants — NLI works better on answer-sized passages."""
    premises: list[str] = []
    answer = chunk.answer.strip()
    question = chunk.question.strip()

    if answer:
        premises.append(answer)
        premises.extend(_split_into_sentences(answer))

    if question and answer:
        premises.append(f"{question} {answer}")

    deduped: list[str] = []
    seen: set[str] = set()
    for premise in premises:
        key = _normalize_text(premise)
        if key and key not in seen:
            seen.add(key)
            deduped.append(premise)
    return deduped or [chunk.text]


def _premises_for_text(chunk_text: str) -> list[str]:
    answer_match = re.search(r"Answer:\s*(.+)", chunk_text, flags=re.IGNORECASE | re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
        premises = [answer, *(_split_into_sentences(answer))]
        question_match = re.search(r"Question:\s*(.+?)(?:\nAnswer:|\Z)", chunk_text, flags=re.IGNORECASE | re.DOTALL)
        if question_match:
            question = question_match.group(1).strip()
            premises.append(f"{question} {answer}")
        deduped: list[str] = []
        seen: set[str] = set()
        for premise in premises:
            key = _normalize_text(premise)
            if key and key not in seen:
                seen.add(key)
                deduped.append(premise)
        return deduped or [chunk_text]
    return _split_into_sentences(chunk_text) or [chunk_text]


def split_answer_into_claims(answer: str) -> list[str]:
    """Split a generated answer into short atomic claims."""
    cleaned = CITATION_PATTERN.sub("", answer or "").strip()
    if not cleaned:
        return []

    return _split_into_sentences(cleaned)


def is_abstention_answer(answer: str) -> bool:
    """Return True when the model declined to answer from retrieved context."""
    cleaned = re.sub(r"\s+", " ", (answer or "").strip())
    if not cleaned:
        return False

    if _normalize_text(cleaned) == CANONICAL_ABSTENTION:
        return True

    claims = split_answer_into_claims(cleaned)
    if len(claims) != 1:
        return False

    return any(pattern.search(claims[0]) for pattern in ABSTENTION_PATTERNS)


def build_abstention_evaluation_result() -> GenerationEvaluationResult:
    """Return a neutral result for refusal / abstention replies."""
    return GenerationEvaluationResult(
        faithfulness_score=0.0,
        contradiction_rate=0.0,
        unsupported_rate=0.0,
        num_claims=0,
        supported_claims=0,
        contradicted_claims=0,
        unsupported_claims=0,
        best_claim_score=0.0,
        avg_claim_score=0.0,
        score_range=0.0,
        claims=tuple(),
        warning=(
            "Abstention response detected — groundedness evaluation skipped "
            "(no factual claims to verify)."
        ),
        abstention_response=True,
    )


def _normalize_label(label: str) -> str:
    key = label.strip().lower().replace(" ", "_")
    return MNLI_LABEL_MAP.get(key, LABEL_NOT_ENOUGH_INFO)


def _canonical_probs(raw: dict[str, float]) -> dict[str, float]:
    canonical = {
        LABEL_SUPPORTED: 0.0,
        LABEL_CONTRADICTED: 0.0,
        LABEL_NOT_ENOUGH_INFO: 0.0,
    }
    for label, score in raw.items():
        canonical[_normalize_label(label)] = max(canonical[_normalize_label(label)], float(score))
    return canonical


def build_generation_evaluation_result(
    claims: list[ClaimEvaluation],
    *,
    warning: str | None = None,
    evaluation_failed: bool = False,
) -> GenerationEvaluationResult:
    """Compute aggregate metrics from per-claim evaluations."""
    if not claims:
        return GenerationEvaluationResult(
            faithfulness_score=0.0,
            contradiction_rate=0.0,
            unsupported_rate=0.0,
            num_claims=0,
            supported_claims=0,
            contradicted_claims=0,
            unsupported_claims=0,
            best_claim_score=0.0,
            avg_claim_score=0.0,
            score_range=0.0,
            claims=tuple(),
            warning=warning or "No valid claims found in the generated answer.",
            evaluation_failed=evaluation_failed,
        )

    supported = sum(1 for claim in claims if claim.label == LABEL_SUPPORTED)
    contradicted = sum(1 for claim in claims if claim.label == LABEL_CONTRADICTED)
    unsupported = sum(1 for claim in claims if claim.label == LABEL_NOT_ENOUGH_INFO)
    total = len(claims)
    claim_scores = [claim.claim_score for claim in claims]
    best_score = max(claim_scores)
    worst_score = min(claim_scores)
    avg_score = sum(claim_scores) / total

    return GenerationEvaluationResult(
        faithfulness_score=supported / total,
        contradiction_rate=contradicted / total,
        unsupported_rate=unsupported / total,
        num_claims=total,
        supported_claims=supported,
        contradicted_claims=contradicted,
        unsupported_claims=unsupported,
        best_claim_score=best_score,
        avg_claim_score=avg_score,
        score_range=best_score - worst_score,
        claims=tuple(claims),
        warning=warning,
        evaluation_failed=evaluation_failed,
    )


def _score_claim_against_premise(
    claim: str,
    premise: str,
    classifier: NLIClassifier,
) -> tuple[dict[str, float], bool]:
    lexical_score = _lexical_support_score(claim, premise)
    if lexical_score is not None:
        return {
            LABEL_SUPPORTED: lexical_score,
            LABEL_CONTRADICTED: 0.0,
            LABEL_NOT_ENOUGH_INFO: max(0.0, 1.0 - lexical_score),
        }, True

    return _canonical_probs(classifier.predict(premise, claim)), False


def _evaluate_claim_against_chunks(
    claim: str,
    chunk_targets: list[tuple[str, list[str]]],
    classifier: NLIClassifier,
) -> ClaimEvaluation:
    """Score one claim against all retrieved chunks and keep the best match."""
    best_probs: dict[str, float] | None = None
    best_chunk_id: str | None = None
    best_chunk_text: str | None = None
    best_entailment = -1.0

    for chunk_id, premises in chunk_targets:
        for premise in premises:
            probs, _lexical = _score_claim_against_premise(claim, premise, classifier)
            entailment = probs[LABEL_SUPPORTED]
            if entailment > best_entailment:
                best_entailment = entailment
                best_probs = probs
                best_chunk_id = chunk_id
                best_chunk_text = premise[:240]

    if best_probs is None:
        best_probs = {
            LABEL_SUPPORTED: 0.0,
            LABEL_CONTRADICTED: 0.0,
            LABEL_NOT_ENOUGH_INFO: 1.0,
        }

    label, confidence = _label_from_probs(best_probs)
    return ClaimEvaluation(
        claim=claim,
        label=label,
        confidence=confidence,
        claim_score=best_probs[LABEL_SUPPORTED],
        best_chunk_id=best_chunk_id,
        best_chunk_text=best_chunk_text,
    )


@lru_cache(maxsize=1)
def get_nli_classifier(model_name: str = DEFAULT_NLI_MODEL) -> NLIClassifier:
    """Load and cache the NLI model (lazy, once per process)."""
    return _TransformersNLIClassifier(model_name=model_name)


class _TransformersNLIClassifier:
    def __init__(self, model_name: str = DEFAULT_NLI_MODEL) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.eval()
        self._id2label = {
            int(index): str(label).lower()
            for index, label in self._model.config.id2label.items()
        }

    def predict(self, premise: str, hypothesis: str) -> dict[str, float]:
        import torch

        inputs = self._tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1).tolist()
        return {
            self._id2label.get(index, f"label_{index}"): float(prob)
            for index, prob in enumerate(probs)
        }


def _chunk_targets(
    context_chunks: list[Chunk] | list[str],
    chunk_ids: list[str] | None,
) -> list[tuple[str, list[str]]]:
    if not context_chunks:
        return []

    if isinstance(context_chunks[0], Chunk):
        chunks = context_chunks  # type: ignore[list-item]
        return [(chunk.chunk_id, _premises_for_chunk(chunk)) for chunk in chunks]

    texts = [str(text) for text in context_chunks]
    ids = chunk_ids or [f"chunk-{index + 1}" for index in range(len(texts))]
    return [
        (chunk_id, _premises_for_text(chunk_text))
        for chunk_id, chunk_text in zip(ids, texts, strict=False)
    ]


def evaluate_generation(
    answer: str,
    context_chunks: list[Chunk] | list[str],
    *,
    chunk_ids: list[str] | None = None,
    classifier: NLIClassifier | None = None,
) -> GenerationEvaluationResult:
    """Evaluate groundedness of ``answer`` against retrieved context chunks."""
    if is_abstention_answer(answer):
        return build_abstention_evaluation_result()

    claims_text = split_answer_into_claims(answer)
    if not claims_text:
        return build_generation_evaluation_result(
            [],
            warning="No valid claims found in the generated answer.",
        )

    chunk_targets = _chunk_targets(context_chunks, chunk_ids)
    if not chunk_targets:
        return build_generation_evaluation_result(
            [],
            warning="No retrieved context available for groundedness evaluation.",
        )

    try:
        nli = classifier or get_nli_classifier()
    except Exception as exc:
        logger.exception("Failed to load NLI model")
        return build_generation_evaluation_result(
            [],
            warning=f"NLI evaluation unavailable: {exc}",
            evaluation_failed=True,
        )

    try:
        claim_evaluations = [
            _evaluate_claim_against_chunks(claim, chunk_targets, nli) for claim in claims_text
        ]
        return build_generation_evaluation_result(claim_evaluations)
    except Exception as exc:
        logger.exception("NLI evaluation failed")
        return build_generation_evaluation_result(
            [],
            warning=f"NLI evaluation failed: {exc}",
            evaluation_failed=True,
        )
