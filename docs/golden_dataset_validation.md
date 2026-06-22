# Golden Dataset Validation Report

## Purpose

The raw golden dataset (`data/evaluation/retrieval_golden_dataset.jsonl`) contains automatically generated complex questions. Many of these were **template artefacts** — grammatically broken concatenations of FAQ titles rather than realistic user queries.

This validation step filters, rewrites, regenerates, or removes poor complex questions while keeping all **500 simple questions** unchanged.

**Output:** `data/evaluation/retrieval_golden_dataset_cleaned.jsonl`

---

## Validation rules

Each **complex** question is evaluated against the following rules.

### Reject if

| Rule | Description |
|---|---|
| **Duplicate topic** | The same label or phrase appears twice (e.g. *"forest management and forest management"*) |
| **FAQ title concatenation** | The question embeds raw FAQ openings (`should…`, `does…`, `my company's activity…`) |
| **Grammatically broken** | Patterns like *"How do should…"*, *"When dealing with should…"*, *"I am assessing will…"* |
| **Incomplete / truncated** | Ends abruptly (*"for cars that are?"*, *"period of?"*) |
| **Missing question mark** | Does not end with `?` |
| **Multiple questions** | Contains more than one `?` |
| **Too short / too long** | `< 25` or `> 220` characters |
| **Unrealistic query** | Too many words (> 42) or commas (> 5) |
| **Low-quality topic labels** | Extracted chunk labels start with auxiliary verbs |
| **Semantically unrelated chunks** | Expected chunks are not neighbours in embedding space |

### Semantic relatedness criteria

- **Pairs:** cosine similarity ≥ `0.35` **or** same FAQ section
- **Triples:** average pairwise similarity ≥ `0.30` **and** max pair ≥ `0.35` **or** all chunks share the same section

Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (with hash fallback if unavailable).

---

## Processing strategy

For each complex question in the raw dataset:

```
1. Validate original question
   ├─ PASS → rewrite into natural user phrasing (keep chunk IDs)
   │         ├─ rewrite valid → keep rewritten version
   │         └─ else → keep original
   └─ FAIL → try regenerate from same chunk set (multiple templates)
              ├─ success → add regenerated question
              └─ fail → remove question
```

### Rewrite (accepted / rewritten)

Valid questions are rewritten using **natural templates** and **clean chunk labels**:

- Prefer quoted activity names from FAQ (`"Forest management"`)
- Fallback: section + FAQ index for poorly structured source questions

**Example rewrite:**

> *"What Taxonomy requirements apply to both forest management and rehabilitation and restoration of forests?"*

### Regeneration (same chunk set)

If validation fails, the same `expected_chunk_ids` are kept and new questions are generated from alternate templates:

- *"How do {A} and {B} interact under EU Taxonomy rules?"*
- *"I am reporting on {A} and {B} — what should I know for Taxonomy compliance?"*
- *"What EU Taxonomy rules apply to {A}, {B}, and {C} together?"*

Up to all template variants are tried before removal.

### Backfill

After cleaning, if fewer than **200** valid complex questions remain, new **pair questions** are generated from semantically related chunk pairs (embedding neighbours ≥ `0.35`) not already covered.

---

## Validation report (2026-06-21)

| Metric | Count |
|---|---|
| **Input complex questions** | 200 |
| **Accepted as-is** | 0 |
| **Rewritten** | 81 |
| **Regenerated (same chunk set)** | 97 |
| **Removed** | 18 |
| **Backfilled (new pairs)** | 22 |
| **Final complex questions** | **200** |
| **Final total (simple + complex)** | **700** |

### Rejection reasons (removed questions)

| Reason | Count |
|---|---|
| Duplicate topic | 15 |
| Duplicate after cleaning | 4 |
| Too long | 6 |
| Grammatically broken | 1 |
| FAQ title concatenation | 2 |
| Unrealistic query | 1 |

### Examples of removed questions

- *"What should companies know about both about considerations on nuclear waste and about considerations on nuclear waste?"* → duplicate topic
- *"How do my company's activity is not covered… and my company's activity is not taxonomy-aligned…?"* → FAQ concatenation
- *"How do construction of new buildings and construction of new buildings relate…?"* → duplicate topic

---

## Final dataset statistics

| Metric | Raw | Cleaned |
|---|---|---|
| Simple questions | 500 | 500 (unchanged) |
| Complex questions | 200 | 200 |
| Total | 700 | 700 |
| Questions with `...` truncation | 0 | 0 |
| Unique complex chunk sets | ~200 | ~200 |

Simple questions were not modified — they map 1:1 to individual FAQ chunks with paraphrased user formulations.

---

## Schema (unchanged)

```json
{
  "question": "What Taxonomy requirements apply to both forest management and rehabilitation and restoration of forests?",
  "expected_chunk_ids": ["faq-0002", "faq-0003"],
  "difficulty": "complex"
}
```

---

## Known limitations

1. **Label quality varies** — when FAQs lack quoted activity names, labels fall back to *"Section topic N"*.
2. **Complex ground truth remains heuristic** — related chunks are embedding neighbours, not manually annotated.
3. **Template-based rewrites** — natural but still rule-generated, not LLM-authored.
4. **Backfill uses pairs only** — triple questions reduced after cleaning; backfill prioritises pair coverage.
5. **Same-section bias** — semantic filter accepts same-section pairs even at moderate similarity.

---

## How to regenerate

```bash
# 1. Generate raw dataset
python scripts/generate_golden_dataset.py

# 2. Validate and clean
python scripts/validate_golden_dataset.py
```

Or in Python:

```python
from eu_taxonomy_rag.cache.chunk_cache import load_or_build_chunks
from eu_taxonomy_rag.evaluation.golden_dataset import load_golden_dataset
from eu_taxonomy_rag.evaluation.golden_dataset_validator import clean_golden_dataset, save_golden_dataset

chunks = load_or_build_chunks()
raw = load_golden_dataset("data/evaluation/retrieval_golden_dataset.jsonl")
cleaned, report = clean_golden_dataset(raw, chunks, seed=42, target_complex=200)
save_golden_dataset(cleaned, "data/evaluation/retrieval_golden_dataset_cleaned.jsonl")
print(report.to_dict())
```

**Parameters:** `seed=42`, `target_complex=200`, `MIN_PAIRWISE_SIMILARITY=0.35`

---

## Why this cleaned dataset suits BM25, Vector Search and Hybrid evaluation

| Method | What cleaned complex questions test |
|---|---|
| **BM25** | Natural phrasing with regulatory keywords without broken FAQ fragments hurting lexical matching |
| **Vector Search** | Semantically related multi-chunk queries where wording differs from source FAQs |
| **Hybrid** | Combined ability to retrieve **all** expected chunks for multi-topic business questions |

Removing concatenated FAQ titles ensures metrics reflect **real retrieval quality**, not artefacts of poor question generation.

Use **`retrieval_golden_dataset_cleaned.jsonl`** for all evaluation runs (steps 16–18).
