# Rag-on-EU-Taxonomy

Small training RAG on the EU Taxonomy FAQs. The goal is to design and implement the best candidate for a simple chat bot aiming at retrieving relevant information on the EU Taxonomy.

## Quick start

**Requirements:** Python 3.10–3.12

### Option A (recommended)

```bash
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[ui]"
eu-taxonomy-rag
```

On macOS/Linux, activate the venv with `source .venv/bin/activate` instead of the PowerShell line above.

### Option B — one command

```bash
./scripts/start.sh
```

The launcher will automatically:

1. Build FAQ **chunks** from `data/taxonomy_faqs_cleaned.md` (cached in `.cache/chunks.jsonl`)
2. Initialize the **generation evaluation** SQLite database (`.cache/generation_eval.db`)
3. Open the **Streamlit** dashboard in your browser

On first use, open the **Benchmark** page and click **Build indexes** to create retrieval indexes (BM25 + dense). This step downloads embedding models and may take several minutes; later runs reuse `.cache/index/`.

### LLM credentials (Chatbot tab only, optional)

No `.env` file is required to start the app. Benchmark, interactive test, and data explorer work without any LLM key.

To use the **Chatbot** tab, either:

- copy the template and add a provider key:

```bash
cp .env.example .env
# then edit .env with your API key(s)
```

- or enter credentials in the UI and click **Save credentials to .env** (the app creates `.env` on first save).

### Optional extras

```bash
pip install -e ".[ui,faiss,dev]"   # FAISS dense index + tests
```

## Streamlit pages

| Page | Purpose |
|------|---------|
| **Chatbot** | RAG Q&A + groundedness evaluation |
| **Benchmark** | Multi-dataset retrieval evaluation (Recall@K, MRR) |
| **Interactive test** | Side-by-side retrieval comparison |
| **Data explorer** | Browse chunks and evaluation datasets |
| **Saved results** | Compare exported benchmark JSON files |

## Generation evaluation (groundedness)

The chatbot can optionally evaluate each generated answer for **groundedness / faithfulness** against the retrieved FAQ chunks.

### What it does

After the LLM produces an answer, the app:

1. Splits the answer into short atomic claims.
2. Compares each claim to the retrieved chunks with a lightweight NLI model (`typeform/distilbert-base-uncased-mnli`).
3. Labels each claim as `supported`, `contradicted`, or `not_enough_info`.
4. Stores the interaction and scores in a local SQLite database (`.cache/generation_eval.db`).
5. Displays per-answer metrics in the **Chat** tab and historical aggregates in **History** and **Metrics**.

### Metrics

| Metric | Meaning |
|--------|---------|
| **Faithfulness** | `supported_claims / total_claims` |
| **Contradiction rate** | Share of claims classified as contradicted |
| **Unsupported rate** | Share of claims with not enough information |
| **Best / average claim score** | Max and mean entailment probability across claims |
| **Score range** | Spread between best and worst claim entailment scores |

### Enable / disable

Set in your environment or `.env`:

```bash
ENABLE_GENERATION_EVAL=true   # default
ENABLE_GENERATION_EVAL=false  # skip NLI evaluation and SQLite writes
```

When disabled, the chatbot still retrieves and generates answers normally.

### Limitations

- This is a **diagnostic monitor**, not a perfect automatic judge.
- NLI models can misclassify paraphrases, implicit reasoning, or domain-specific wording.
- Claim splitting is sentence-based and may miss or over-split complex answers.
- Scores depend on the quality of retrieved chunks; poor retrieval lowers faithfulness even when the LLM is cautious.
- The first evaluation downloads the NLI model and may take a few seconds on CPU.

**Scoring notes:** each claim is matched against the **best** retrieved passage (answer text and sentence-level variants), not an aggregate across all chunks. Near-verbatim overlap with a chunk answer is detected lexically before NLI. Abstention replies (`I cannot answer this question from the available context.`) skip groundedness scoring.

## Project layout

- `app/streamlit_app.py` — main Streamlit dashboard
- `app/chatbot_page.py` — RAG chatbot
- `src/eu_taxonomy_rag/cli.py` — bootstrap + launch command
- `src/eu_taxonomy_rag/evaluation/generation_eval.py` — NLI groundedness evaluation
- `src/eu_taxonomy_rag/storage/evaluation_store.py` — SQLite persistence

## Development

```bash
pip install -e ".[ui,dev]"
pytest
eu-taxonomy-rag --bootstrap-only   # rebuild FAQ chunks without opening the UI
```
