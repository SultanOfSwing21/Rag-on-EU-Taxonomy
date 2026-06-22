# Natural User Query Dataset

## Purpose

`natural_user_queries.jsonl` is a **second evaluation dataset** for retrieval benchmarking. It contains realistic chatbot-style questions derived from `retrieval_golden_dataset_cleaned.jsonl`, but rewritten to sound like queries from sustainability officers, finance teams, consultants, and other business users.

The original golden dataset is **not modified**.

| Dataset | Role |
|---|---|
| `retrieval_golden_dataset_cleaned.jsonl` | Rule-based benchmark (paraphrases + templates) |
| `natural_user_queries.jsonl` | LLM-rewritten realistic user queries |

Both share the same **`expected_chunk_ids`** ground truth for fair comparison across retrieval methods.

---

## Generation methodology

### 1. Source selection

From `retrieval_golden_dataset_cleaned.jsonl`:

| Type | Selected | Source pool |
|---|---|---|
| Simple | 200 | 500 cleaned simple questions |
| Complex | 100 | 200 cleaned complex questions |

Selection uses **seed `42`** and prioritises **diverse chunk sets** (one source question per unique `expected_chunk_ids` group when possible).

### 2. Context provided to the LLM

For each selected source question, the LLM receives:

- The **source benchmark question** (intent reference)
- **FAQ context** for each expected chunk:
  - section name
  - official FAQ question
  - short answer preview (~220 characters)
- A **persona / style** instruction
- Difficulty (`simple` or `complex`)

The LLM is instructed to preserve intent but **not copy regulatory wording verbatim**.

### 3. Persona rotation

Each rewrite uses one of seven personas:

| Persona | Voice |
|---|---|
| `sustainability_officer` | CSRD / Taxonomy disclosure preparer |
| `finance_team` | Finance controller working on KPIs |
| `reporting_team` | Regulatory reporting specialist |
| `consultant` | ESG advisor |
| `disclosure_preparer` | Group reporting manager |
| `bank_analyst` | Sustainable finance / portfolio analyst |
| `direct` | Direct practical business question |

Personas rotate across questions; on failed validation, a different persona is tried (up to 3 attempts).

### 4. LLM configuration

| Parameter | Value |
|---|---|
| Provider | OpenAI |
| Model | `gpt-4o-mini` |
| Temperature | `0.7` |
| API key | `OPENAI_API_KEY` environment variable |

### 5. Quality checks (post-generation)

A rewrite is **rejected** if:

- Too short (< 20 characters)
- Identical to source after normalisation
- Word overlap with source > 72% (likely verbatim copy)
- Contains more than 2 question marks

Failed rewrites are retried with a different persona.

---

## Schema

```json
{
  "question": "I'm new to sustainability reporting. What exactly is the EU Taxonomy and why does it matter for companies?",
  "expected_chunk_ids": ["faq-0304"],
  "difficulty": "simple"
}
```

Complex example:

```json
{
  "question": "We are aligning both forest management DNSH criteria and rehabilitation activities. How should we treat them together in our Taxonomy reporting?",
  "expected_chunk_ids": ["faq-0002", "faq-0003"],
  "difficulty": "complex"
}
```

---

## How to generate

```bash
export OPENAI_API_KEY="sk-..."
source .venv/bin/activate
python scripts/generate_natural_queries.py
```

Output: `data/evaluation/natural_user_queries.jsonl`

Python API:

```python
from eu_taxonomy_rag.evaluation.natural_dataset import run_natural_dataset_generation

dataset, stats = run_natural_dataset_generation(
    n_simple=200,
    n_complex=100,
    seed=42,
)
print(stats)
```

---

## Dataset composition (target)

| Metric | Target |
|---|---|
| Simple queries | 200 |
| Complex queries | 100 |
| **Total** | **300** |

Ground truth (`expected_chunk_ids`) is **copied unchanged** from the selected source question in the cleaned golden dataset.

---

## Known limitations

1. **Requires OpenAI API** — the dataset is not generated with templates; you need a valid `OPENAI_API_KEY` and network access.
2. **LLM variability** — re-running generation produces different wording (temperature 0.7); ground truth stays stable.
3. **Intent preservation is LLM-judged** — no human annotation loop; occasional drift is possible on very technical FAQs.
4. **Cost & time** — 300 LLM calls; expect a few minutes and small API cost.
5. **English only** — matches the source FAQ document.
6. **Overlap check is heuristic** — 72% word overlap threshold may allow near-paraphrases or reject some valid rewrites.

---

## Why this dataset complements the golden dataset

| Aspect | Golden (cleaned) | Natural |
|---|---|---|
| Wording | Rule-based paraphrase / template | LLM natural language |
| User realism | Moderate | High |
| Reproducibility | Fully deterministic | Seed + model dependent |
| Best for | Controlled regression tests | Realistic chatbot evaluation |

Use **both datasets** to evaluate BM25, dense retrieval, and hybrid search:

- Golden → stable, reproducible benchmark
- Natural → realistic user query distribution

---

## Files

| File | Description |
|---|---|
| `src/eu_taxonomy_rag/evaluation/natural_dataset.py` | Selection, LLM rewrite, validation |
| `src/eu_taxonomy_rag/llm/client.py` | OpenAI client |
| `scripts/generate_natural_queries.py` | CLI generator |
| `data/evaluation/natural_user_queries.jsonl` | Generated dataset (after running script) |
| `tests/test_natural_dataset.py` | Unit tests (fake LLM) |
