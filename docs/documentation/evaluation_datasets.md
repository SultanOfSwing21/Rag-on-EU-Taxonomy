## Evaluation datasets

### Why we built our own datasets

To improve retrieval, we need **ground truth**: for each test question, we must know which
FAQ chunk(s) *should* be retrieved. Without that, we cannot compute Recall@K or MRR, and we
cannot compare BM25 vs dense vs hybrid objectively.

The challenge is that **real user questions do not come with labels**. Creating a representative,
unbiased evaluation set is one of the hardest parts of a RAG project — especially on a
closed regulatory corpus where experts are expensive and questions are niche.

We therefore built **three related datasets**, each with a different purpose and different
trade-offs. None is perfect alone; together they stress-test retrieval from complementary angles.

---

### The three datasets

| Dataset | File | Questions | How it was built |
|---------|------|-----------|------------------|
| **Golden (raw)** | `retrieval_golden_dataset.jsonl` | 700 | Fully programmatic, seed 42 |
| **Golden (cleaned)** | `retrieval_golden_dataset_cleaned.jsonl` | 700 | Raw golden + validation & rewrite pass |
| **Natural 748** | `natural_user_queries_748.jsonl` | 748 | ChatGPT rewrites with simulated user personas |

All three share the same core structure:

```json
{
  "question": "...",
  "expected_chunk_ids": ["faq-0042"],
  "difficulty": "simple"
}
```

The `expected_chunk_ids` are the **ground truth** used to score retrieval.

---

### Dataset 1 — Golden (raw): programmatic generation

**Motivation:** create a **large, reproducible, zero-cost** benchmark that can be regenerated
identically from code (seed `42`). No API calls, no human annotation, no LLM variability.

#### Simple questions (~500)

For each of the 324 FAQ chunks, we generate several **rule-based paraphrases** of the official
question:

| Technique | Example idea |
|-----------|--------------|
| Original wording | Copy of the official FAQ question |
| Conversational prefix | *"Can you explain: …"* / *"I need to understand: …"* |
| Shorter formulation | Truncated at comma or first ~12 words |
| Synonym replacement | Taxonomy → EU taxonomy framework; TSC → technical screening criteria |
| Section context | Prefix with section name in brackets |
| Lowercase rephrasing | *"What I want to know is: …"* |

Each simple question maps to **exactly one** `expected_chunk_id` — the chunk it was derived from.

**What this tests:** can the retriever find the right FAQ when wording varies slightly from
the official text? This is the baseline retrieval task.

#### Complex questions (~200)

Built by combining **semantically related** FAQ chunks:

1. Embed all chunks with MiniLM
2. Find neighbours above cosine similarity threshold (≥ 0.45)
3. Prefer neighbours from the **same FAQ section** (avoid random pairings)
4. Combine pairs/triples into multi-topic questions via templates, e.g.  
   *"How do {topic A} and {topic B} relate under the EU Taxonomy?"*

Each complex question has **2–3 expected chunk IDs**.

**What this tests:** when a user question spans multiple topics, does retrieval surface
**all** required FAQs within top-k?

#### Limitations of the raw golden set

1. **Template artefacts** — many complex questions were grammatically broken concatenations
   of FAQ titles (*"How do should companies…"*) rather than realistic queries.
2. **Repetitive phrasing** — rule-based paraphrases still feel mechanical.
3. **Circular grounding** — questions are derived from the same chunks they label as ground
   truth. The dataset is **aligned by construction**: we know the answer because we built the
   question from it. Real users ask questions *without* that alignment.
4. **Coverage gaps** — 312 of 324 chunks appear; 12 have no generated question.
5. **English only** — matches the source document.

Despite these limits, the raw golden set is invaluable for **regression testing**: same seed,
same questions, every time.

---

### Dataset 2 — Golden (cleaned): validation pass

**Motivation:** keep all 500 simple questions unchanged, but **fix or remove** poor complex
questions from the raw set.

The raw programmatic generation produced complex questions that were technically valid
( correct chunk IDs) but **unrealistic as user queries**. A benchmark should measure retrieval
quality, not robustness to broken grammar.

#### What the validator does

For each **complex** question, apply rules:

| Reject if… | Why |
|------------|-----|
| Duplicate topic in question | Template artefact (*"forest management and forest management"*) |
| Raw FAQ title concatenation | Not a natural user query |
| Grammatically broken patterns | *"How do should…"*, truncated endings |
| Semantically unrelated expected chunks | Neighbours below similarity threshold and different sections |

**Processing strategy:**

- Valid question → rewrite into natural phrasing (keep chunk IDs)
- Invalid → regenerate from same chunk set with alternate templates
- Still invalid → remove; backfill from new semantic neighbour pairs

**Output:** `retrieval_golden_dataset_cleaned.jsonl` — still 700 questions, but complex
questions are more readable.

#### Why keep both raw and cleaned in the Benchmark UI?

Comparing raw vs cleaned shows **how much retrieval scores change** when unrealistic queries
are removed. If a method only wins on raw complex questions, it may be overfitting to noise.

#### Remaining limitations

Even cleaned, golden questions are still **constructed from known chunks**. They remain
partially aligned with their ground truth. They are excellent for controlled experiments but
**not a substitute for real user logs**.

---

### Dataset 3 — Natural 748: LLM rewrites with personas

**Motivation:** approximate how **real business users** might ask questions — sustainability
officers, finance controllers, consultants — without access to production query logs.

#### How it was built

1. **Select** a subset from the cleaned golden dataset (seed `42`, diverse chunk coverage):
   - 648 simple source questions (expanded pool vs initial 200-target design)
   - 100 complex source questions
2. For each selected question, call **ChatGPT (`gpt-4o-mini`)** with:
   - the source benchmark question (intent reference)
   - FAQ context for each expected chunk (section, official question, answer preview)
   - a **persona instruction** (rotating across 7 roles)
   - instruction to preserve intent but **not copy regulatory wording verbatim**
3. **Validate** each rewrite (length, overlap with source, multiple question marks, etc.)
4. **Copy `expected_chunk_ids` unchanged** from the source question

#### The seven personas

| Persona | Simulated user |
|---------|----------------|
| `sustainability_officer` | CSRD / Taxonomy disclosure preparer |
| `finance_team` | Finance controller working on KPIs |
| `reporting_team` | Regulatory reporting specialist |
| `consultant` | ESG advisor |
| `disclosure_preparer` | Group reporting manager |
| `bank_analyst` | Sustainable finance analyst |
| `direct` | Direct practical business question |

**What this tests:** retrieval when queries sound like a chatbot, not like an FAQ index.

#### Limitations of the natural dataset

1. **Requires OpenAI API** to regenerate — not fully offline.
2. **LLM variability** — temperature 0.7 means re-runs produce different wording (ground truth
   stays stable; scores are not bitwise reproducible across regenerations).
3. **Intent preservation is LLM-judged** — no human annotation loop; occasional drift possible.
4. **Same ground-truth bias** — we still *know* the target chunks because they come from the
   golden source questions. The LLM rewrites the surface form, not the underlying label.
   Real production queries may target FAQs we never thought to label.
5. **Persona simulation ≠ real users** — personas are prompt instructions, not demographic data.
6. **English only**.

#### Why natural complements golden

| Aspect | Golden (cleaned) | Natural 748 |
|--------|------------------|-------------|
| Wording | Rule-based / templates | Conversational, role-specific |
| Reproducibility | Fully deterministic | Seed + model dependent |
| Realism | Moderate | Higher |
| Best for | Regression, method comparison | Stress-test paraphrase + persona variance |

**Use both.** If hybrid beats BM25 on golden but not on natural, the method may be exploiting
template patterns rather than true semantic robustness.

---

### Simple vs complex questions (all datasets)

| Type | Expected chunks | What it measures |
|------|-----------------|------------------|
| **Simple** | 1 | *"Did we find the right FAQ?"* — Recall@1 is strict hit/miss |
| **Complex** | 2–3 | *"Did we find all relevant FAQs?"* — Recall@K measures partial coverage |

Complex questions are harder and more realistic for multi-topic regulatory queries, but
**ground truth for multi-chunk questions is inherently debatable**: other valid chunks might
exist that we did not label.

---

### The fundamental difficulty: representative, unbiased evaluation

All three datasets share a structural limitation:

> **Questions are created with knowledge of the answers.**

- Golden: paraphrased directly from target chunks
- Natural: rewritten from golden questions that already encode the target chunks

This is standard in academic benchmarks but **differs from production**, where users ask
questions without aligning to a pre-selected FAQ. No synthetic dataset fully captures:

- out-of-scope questions (*"What is the weather?"*)
- ambiguous questions mapping to multiple valid FAQs
- questions whose answer spans FAQs we did not label
- evolving regulatory interpretation not yet in the FAQ

We mitigated bias by:

- varying phrasing (rules + LLM personas)
- separating simple and complex difficulty
- validating and cleaning unrealistic complex templates
- comparing methods on **two independent dataset styles**

But we remain transparent: **metrics measure performance on our labelled sets**, not on
all possible real-world queries. Human review and production logging would be the next step
in a deployed system.

---

### Further reading

- [`docs/golden_dataset.md`](../golden_dataset.md) — full generation parameters
- [`docs/golden_dataset_validation.md`](../golden_dataset_validation.md) — validation rules detail
- [`docs/natural_dataset.md`](../natural_dataset.md) — LLM rewrite pipeline
