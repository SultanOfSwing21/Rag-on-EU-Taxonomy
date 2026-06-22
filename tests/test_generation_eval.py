import pytest

from eu_taxonomy_rag.core.models import Chunk
from eu_taxonomy_rag.evaluation.generation_eval import (
    ClaimEvaluation,
    GenerationEvaluationResult,
    build_generation_evaluation_result,
    evaluate_generation,
    is_abstention_answer,
    split_answer_into_claims,
)


class FakeNLI:
    def __init__(self, label: str = "entailment", confidence: float = 0.9) -> None:
        self.label = label
        self.confidence = confidence

    def predict(self, premise: str, hypothesis: str) -> dict[str, float]:
        if self.label == "entailment":
            return {"entailment": self.confidence, "neutral": 0.05, "contradiction": 0.05}
        if self.label == "contradiction":
            return {"entailment": 0.05, "neutral": 0.05, "contradiction": self.confidence}
        return {"entailment": 0.1, "neutral": self.confidence, "contradiction": 0.1}


def test_split_answer_into_claims_removes_citations() -> None:
    answer = (
        "The EU Taxonomy is a classification system. "
        "It helps report green activities. [faq-0001]"
    )
    claims = split_answer_into_claims(answer)
    assert len(claims) == 2
    assert all("[faq-" not in claim for claim in claims)


def test_build_generation_evaluation_result_metrics() -> None:
    claims = [
        ClaimEvaluation("Claim A", "supported", 0.9, 0.9),
        ClaimEvaluation("Claim B", "contradicted", 0.8, 0.1),
        ClaimEvaluation("Claim C", "not_enough_info", 0.7, 0.2),
    ]
    result = build_generation_evaluation_result(claims)

    assert result.num_claims == 3
    assert result.supported_claims == 1
    assert result.contradicted_claims == 1
    assert result.unsupported_claims == 1
    assert result.faithfulness_score == pytest.approx(1 / 3)
    assert result.contradiction_rate == pytest.approx(1 / 3)
    assert result.unsupported_rate == pytest.approx(1 / 3)
    assert result.best_claim_score == pytest.approx(0.9)
    assert result.avg_claim_score == pytest.approx((0.9 + 0.1 + 0.2) / 3)
    assert result.score_range == pytest.approx(0.8)


def test_build_generation_evaluation_result_no_claims() -> None:
    result = build_generation_evaluation_result([])

    assert result.num_claims == 0
    assert result.faithfulness_score == 0.0
    assert result.warning is not None


def test_evaluate_generation_supported_claims() -> None:
    chunk = Chunk(
        chunk_id="faq-0001",
        question="What is the EU Taxonomy?",
        answer="A classification system for sustainable activities.",
    )
    answer = "The EU Taxonomy is a classification system for sustainable activities."
    result = evaluate_generation(answer, [chunk], classifier=FakeNLI())

    assert result.num_claims == 1
    assert result.supported_claims == 1
    assert result.faithfulness_score == pytest.approx(1.0)
    assert result.evaluation_failed is False


def test_is_abstention_answer_detects_canonical_refusal() -> None:
    assert is_abstention_answer("I cannot answer this question from the available context.")
    assert is_abstention_answer("I can't answer from the available context.")
    assert not is_abstention_answer(
        "The EU Taxonomy is a classification system for sustainable activities."
    )


def test_evaluate_generation_skips_abstention_response() -> None:
    chunk = Chunk(
        "faq-0304",
        "What is the EU Taxonomy?",
        "The EU Taxonomy is a green classification system.",
    )
    answer = "I cannot answer this question from the available context."
    result = evaluate_generation(answer, [chunk], classifier=FakeNLI())

    assert result.abstention_response is True
    assert result.num_claims == 0
    assert result.supported_claims == 0
    assert "Abstention response" in (result.warning or "")


def test_evaluate_generation_no_valid_claims() -> None:
    chunk = Chunk("faq-0001", "Q", "A")
    result = evaluate_generation("short", [chunk], classifier=FakeNLI())

    assert result.num_claims == 0
    assert result.faithfulness_score == 0.0
    assert "No valid claims" in (result.warning or "")


def test_evaluate_generation_ignores_irrelevant_chunks() -> None:
    class SelectiveNLI:
        def predict(self, premise: str, hypothesis: str) -> dict[str, float]:
            if "classification system" in premise.lower():
                return {"entailment": 0.88, "neutral": 0.07, "contradiction": 0.05}
            return {"entailment": 0.03, "neutral": 0.92, "contradiction": 0.05}

    relevant = Chunk(
        "faq-0001",
        "What is the EU Taxonomy?",
        "A classification system for sustainable activities.",
    )
    irrelevant = Chunk("faq-0999", "Other topic?", "Completely unrelated banking content.")
    answer = "The EU Taxonomy is a classification system for sustainable activities."
    result = evaluate_generation(answer, [irrelevant, relevant], classifier=SelectiveNLI())

    assert result.supported_claims == 1
    assert result.faithfulness_score == pytest.approx(1.0)


def test_evaluate_generation_lexical_overlap_marks_supported() -> None:
    class NeverEntailsNLI:
        def predict(self, premise: str, hypothesis: str) -> dict[str, float]:
            return {"entailment": 0.1, "neutral": 0.8, "contradiction": 0.1}

    chunk = Chunk(
        "faq-0001",
        "Q",
        "Undertakings must report Taxonomy-aligned CapEx in their annual reports.",
    )
    answer = "Undertakings must report Taxonomy-aligned CapEx in their annual reports."
    result = evaluate_generation(answer, [chunk], classifier=NeverEntailsNLI())

    assert result.supported_claims == 1
    assert result.faithfulness_score == pytest.approx(1.0)


def test_evaluate_generation_nli_failure_is_safe() -> None:
    chunk = Chunk("faq-0001", "Q", "A long enough answer for claim splitting.")

    class BrokenNLI:
        def predict(self, premise: str, hypothesis: str) -> dict[str, float]:
            raise RuntimeError("model failed")

    result = evaluate_generation(
        "This is a valid claim sentence for testing.",
        [chunk],
        classifier=BrokenNLI(),
    )

    assert result.evaluation_failed is True
    assert result.num_claims == 0
    assert "failed" in (result.warning or "").lower()


def test_evaluate_generation_model_load_failure_is_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*args, **kwargs):
        raise OSError("cannot load model")

    monkeypatch.setattr(
        "eu_taxonomy_rag.evaluation.generation_eval.get_nli_classifier",
        _raise,
    )
    chunk = Chunk("faq-0001", "Q", "A")
    result = evaluate_generation(
        "This is a valid claim sentence for testing.",
        [chunk],
    )
    assert result.evaluation_failed is True
    assert result.num_claims == 0


def test_generation_evaluation_result_to_dict() -> None:
    result = GenerationEvaluationResult(
        faithfulness_score=1.0,
        contradiction_rate=0.0,
        unsupported_rate=0.0,
        num_claims=1,
        supported_claims=1,
        contradicted_claims=0,
        unsupported_claims=0,
        best_claim_score=0.9,
        avg_claim_score=0.9,
        score_range=0.0,
        claims=(ClaimEvaluation("Claim", "supported", 0.9, 0.9),),
    )
    payload = result.to_dict()
    assert payload["claims"][0]["label"] == "supported"
