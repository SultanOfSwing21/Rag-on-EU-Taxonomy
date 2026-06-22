## KPI tracking over time

### Why track KPIs at all?

RAG systems are not "build once and done." During experimentation you change retrieval methods,
prompt wording, k parameters, embedding models. Without persisted metrics, every session
starts from zero and you cannot demonstrate **whether changes helped**.

The assignment explicitly asked for metrics to show improvement (or regression) across approaches.
We implemented **two complementary tracking layers** because retrieval and generation fail
for different reasons.

---

### The two evaluation layers

| Layer | Question it answers | When it runs | Storage |
|-------|---------------------|--------------|---------|
| **1. Retrieval** | *Are we finding the right FAQ chunks?* | Batch (Benchmark page) | JSON exports in `data/evaluation/results/` |
| **2. Generation** | *Is the LLM faithful to what we retrieved?* | Online (each Chatbot turn) | SQLite `.cache/generation_eval.db` |

Think of it as a pipeline with two checkpoints:

```
User question
    → [Layer 1: Retrieval KPIs]  Recall@K, MRR
    → LLM generation
    → [Layer 2: Faithfulness KPIs]  supported / contradicted / unsupported claims
```

You need both. High faithfulness with bad retrieval means the model is honest about thin
context. High retrieval with low faithfulness means the model is misusing good context.

---

### Layer 1 — Retrieval KPIs

**What is stored:** each benchmark run exports a JSON file with:

- dataset name (golden cleaned, golden raw, natural 748)
- retrieval method (BM25, hybrid, etc.)
- Recall@1, Recall@3, Recall@5, MRR
- optional segment breakdowns (difficulty, persona)

**Where to view:** **Benchmark** page — run fresh evaluation or load latest saved JSON.

**Why JSON files instead of a database?**

- Benchmark runs are **batch artefacts** — you want immutable snapshots to compare
  *"hybrid before/after parameter change"*
- Easy to version, share, and inspect outside the app
- No schema migration for experiment exports

**Typical use:** run benchmark → save JSON → change one variable → re-run → compare files.

---

### Layer 2 — Generation faithfulness KPIs

**What is stored:** for each Chatbot interaction:

- question and generated answer
- retrieval method, k, chunk IDs retrieved
- per-claim NLI labels and confidence scores
- aggregate faithfulness, contradiction rate, unsupported rate
- timestamp

**Where to view:**

- **Chatbot → History** — individual turns
- **Chatbot → Metrics** — aggregates and trends

**Why SQLite?**

- Chat interactions are **continuous and session-oriented**
- You want append-only history as you iterate on prompts
- Lightweight, local, no server required for a demo/portfolio project

---

### How the layers work together during experimentation

| Experiment | Layer 1 signal | Layer 2 signal |
|------------|----------------|----------------|
| Switch BM25 → hybrid | Recall@K should improve on natural dataset | Faithfulness may improve if better context arrives |
| Tighten abstention prompt | Retrieval unchanged | Contradiction rate should drop; unsupported may rise |
| Increase k from 5 to 10 | Recall@K on complex questions may rise | Faithfulness may drop if noise chunks confuse LLM |
| Change LLM model | Retrieval unchanged | Faithfulness / contradiction patterns may shift |

This separation helps **localise failures** instead of blaming "the RAG system" generically.

---

### What we do not track (yet)

- Centralised experiment dashboard across machines
- Automatic regression alerts on CI
- User feedback / thumbs up-down
- Cost per query (LLM tokens)
- Latency percentiles

These would be expected in production MLOps but were out of scope for a focused demonstration.

---

### Configuration

| Variable | Effect |
|----------|--------|
| `ENABLE_GENERATION_EVAL=true` | Layer 2 active (default) |
| `ENABLE_GENERATION_EVAL=false` | Skip NLI and SQLite writes; Layer 1 unaffected |
| `EU_TAXONOMY_EVAL_DB` | Custom path for SQLite database |
