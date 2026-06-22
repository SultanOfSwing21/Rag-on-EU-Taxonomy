# Golden Retrieval Dataset

## Purpose

The golden retrieval dataset is an automatically generated evaluation set used to measure how well different retrieval methods find the correct FAQ chunks.

Each entry contains:

- a **user question** (simple or complex)
- the **expected chunk ID(s)** that should be retrieved
- a **difficulty label**

This dataset supports retrieval evaluation with metrics such as **Recall@K**, **Precision@K**, and **MRR** across:

- **BM25** (lexical search)
- **Vector Search** (dense embeddings)
- **Hybrid Retrieval** (dense + BM25 fusion)

The dataset is stored at:

`data/evaluation/retrieval_golden_dataset.jsonl`

---

## Generation methodology

The dataset is built automatically from the existing FAQ chunks (`324` chunks from `data/taxonomy_faqs_cleaned.md`).

Generation is **reproducible** using a fixed random seed (`42`).

Pipeline:

1. Load FAQ chunks from cache or rebuild from source
2. Compute chunk embeddings (`all-MiniLM-L6-v2`)
3. Generate simple question variants (1 expected chunk each)
4. Find semantic neighbours for complex multi-hop questions
5. Generate complex questions from related chunk pairs/triples
6. Deduplicate highly similar questions
7. Validate all referenced chunk IDs
8. Save to JSONL

Regenerate with:

```bash
source .venv/bin/activate
python scripts/generate_golden_dataset.py
```

---

## Simple questions (~500)

**Target:** 500 questions  
**Expected chunks:** exactly 1 per question

For each FAQ chunk, several realistic user formulations are generated:

| Technique | Example |
|---|---|
| Original question | Direct copy of the FAQ question |
| Conversational prefix | "Can you explain: …" / "I need to understand: …" |
| Shorter formulation | Truncated at comma or first 12 words |
| Synonym replacement | Taxonomy → EU taxonomy framework, TSC ↔ technical screening criteria |
| Section context | "[Climate Delegated Act] …" |
| Lowercase rephrasing | "What I want to know is: …" |

Each simple question preserves the **original intent** of one FAQ and maps to **one ground-truth chunk**.

---

## Complex multi-hop questions (~200)

**Target:** 200 questions  
**Expected chunks:** 2 or 3 per question

Complex questions are built by combining semantically related chunks.

### Semantic neighbour selection

1. Embed all chunks with `all-MiniLM-L6-v2`
2. Compute cosine similarity between chunk vectors
3. For each anchor chunk, keep the top neighbours above similarity threshold `0.45`
4. Prefer neighbours from the **same FAQ section**, unless similarity is very high (`≥ 0.60`)
5. Limit to top `8` neighbours per chunk

This avoids combining unrelated topics (e.g. forestry + financial reporting) unless they are genuinely close in embedding space.

### Question construction

- **Pairs (2 chunks):** templates such as  
  *"How do {topic A} and {topic B} relate under the EU Taxonomy?"*
- **Triples (3 chunks):** templates such as  
  *"What are the combined Taxonomy implications of {topic A}, {topic B}, and {topic C}?"*

Topics are extracted from FAQ questions using this priority:

1. **Quoted activity name** (e.g. `"Forest management"`) when present
2. Otherwise, text **before** delimiters like *" mean "*, *" in section"*
3. Otherwise, the first **12 words** after removing the leading question word

No artificial `"..."` truncation is applied.

---

## Dataset schema

Each line is a JSON object:

### Simple example

```json
{
  "question": "Can you explain: What is the EU Taxonomy?",
  "expected_chunk_ids": ["faq-0001"],
  "difficulty": "simple"
}
```

### Complex example

```json
{
  "question": "How do forest management DNSH criteria and Taxonomy reporting relate under the EU Taxonomy?",
  "expected_chunk_ids": ["faq-0032", "faq-0033"],
  "difficulty": "complex"
}
```

| Field | Type | Description |
|---|---|---|
| `question` | string | User query to send to the retriever |
| `expected_chunk_ids` | list[string] | Ground-truth chunk ID(s) |
| `difficulty` | string | `"simple"` or `"complex"` |

---

## Dataset statistics

Generated on: 2026-06-21

| Metric | Value |
|---|---|
| **Total questions** | 700 |
| **Simple questions** | 500 |
| **Complex questions** | 200 |
| **Unique questions (after dedup)** | 700 |
| **Unique chunks referenced** | 312 / 324 |
| **Complex pairs (2 chunks)** | 180 |
| **Complex triples (3 chunks)** | 20 |
| **Avg expected chunks (complex)** | 2.1 |
| **Sections covered** | 7 |

### Sections covered

- Climate Delegated Act
- Complementary Climate Delegated Act
- Disclosures Delegated Act - General
- EU Taxonomy - General
- Taxonomy-Alignment Reporting
- Taxonomy-Eligibility reporting (part 1)
- Taxonomy-Eligibility reporting (part 2)

### Generation parameters

| Parameter | Value |
|---|---|
| Random seed | `42` |
| Target simple | `500` |
| Target complex | `200` |
| Neighbour top-k | `8` |
| Min neighbour similarity | `0.45` |
| Dedup similarity threshold | `0.92` |
| Source chunks | `324` |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |

---

## Deduplication and validation

### Deduplication

- Exact duplicates removed after text normalisation (lowercase, punctuation stripped)
- Near-duplicates removed when Jaccard word similarity ≥ `0.92` (within the same difficulty level)

### Validation

- Every `expected_chunk_id` must exist in the source chunk set
- Questions must be non-empty
- Difficulty must be `"simple"` or `"complex"`

---

## Known limitations

1. **Template-based generation** — questions are rule-generated, not LLM-authored. Wording can feel repetitive.
2. **Complex ground truth is approximate** — semantic neighbours are a heuristic; some multi-hop questions may have alternative valid chunks.
3. **Simple questions inherit FAQ phrasing** — many remain close to the original official wording.
4. **Not all chunks are covered** — 312 of 324 chunks appear in the dataset; 12 chunks have no generated question.
5. **English only** — matches the source FAQ document language.
6. **Embedding dependency** — neighbour selection quality depends on the embedding model used during generation.

---

## Why this suits BM25, Vector Search and Hybrid evaluation

| Method | What the dataset tests |
|---|---|
| **BM25** | Lexical overlap on simple paraphrases and keyword-rich regulatory terms (TSC, DNSH, CapEx) |
| **Vector Search** | Semantic matching when user wording diverges from official FAQ phrasing |
| **Hybrid** | Combined performance on both lexical and semantic variants, especially complex multi-topic questions |

- **Simple questions** isolate single-chunk retrieval (Recall@1, Recall@3, Recall@5)
- **Complex questions** test whether the retriever surfaces **all required chunks** within top-K
- **Fixed seed + JSONL format** make comparisons across methods reproducible and fair

---

## Python usage

```python
from eu_taxonomy_rag.evaluation.golden_dataset import load_golden_dataset

dataset = load_golden_dataset()
simple = [q for q in dataset if q.difficulty == "simple"]
complex_ = [q for q in dataset if q.difficulty == "complex"]

print(len(simple), len(complex_))
print(simple[0].question, simple[0].expected_chunk_ids)
```
