from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int,
    *,
    rrf_k: int = 60,
) -> list[tuple[str, float]]:
    """Fusionne plusieurs classements via Reciprocal Rank Fusion (RRF)."""
    scores: dict[str, float] = defaultdict(float)

    for ranked in ranked_lists:
        for rank, (chunk_id, _) in enumerate(ranked, start=1):
            scores[chunk_id] += 1.0 / (rrf_k + rank)

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]
