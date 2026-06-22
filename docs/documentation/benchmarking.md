## Retrieval benchmarking

### Why benchmark before trusting the chatbot

Retrieval is the **foundation** of this RAG system. The LLM only sees what retrieval returns.

If the wrong FAQ chunks are retrieved:

- a capable LLM may **hallucinate** details not in the context
- a well-prompted LLM may **abstain** even when the corpus contains the answer
- faithfulness scores will **blame the LLM** for what is actually a retrieval failure

We therefore measure retrieval **independently** on labelled datasets before drawing conclusions
about generation quality. This follows the project principle:

> *Measure first, generate second.*

---

### Metrics

| Metric | Formula / intuition | When it matters most |
|--------|---------------------|----------------------|
| **Recall@K** | \|expected ∩ top-k\| / \|expected\| | Complex questions — are *all* required chunks in the top-k window? |
| **MRR** | Mean of 1/rank of first relevant chunk | Simple questions — how fast does the right FAQ appear? |

#### Recall@K in practice

- **Simple question** (1 expected chunk): Recall@1 is binary — hit or miss.
- **Complex question** (2–3 chunks): Recall@5 might be 0.67 if only 2 of 3 expected chunks
  appear in top-5. This directly models whether the LLM receives enough context.

#### Why not precision alone?

On a 324-chunk corpus with k=5, precision is less informative than recall for multi-chunk
questions. We care whether **all needed FAQs** are present, not whether every returned chunk
is relevant (though that matters for LLM noise).

---

### What the Benchmark page does

The **Benchmark** page is the control centre for retrieval experimentation. It was designed
to answer: *"Which retrieval method works best, on which query types, and did my last change
help or hurt?"*

#### 1. Index management

Before any evaluation, you must **build indexes** for the selected methods (BM25, dense, hybrid).

- First build downloads embedding models — can take several minutes.
- Indexes persist in `.cache/index/`.
- **Force rebuild** if chunks or embedding models change.

This step is separate from benchmarking so you can rebuild indexes once and run many evaluations.

#### 2. Dataset selection

Choose one or more of the three evaluation datasets:

| Dataset | Role in benchmarking |
|---------|----------------------|
| **Golden (cleaned)** | Primary regression benchmark — stable, reproducible |
| **Golden (raw)** | Compare impact of complex-question validation |
| **Natural 748** | Realistic phrasing stress-test with persona metadata |

You can run all three in one pass to see whether a method generalises across dataset styles.

#### 3. Configurable evaluation parameters

| Control | Purpose |
|---------|---------|
| **Top-k retrieval** | Match the k used in production chatbot (default 5) |
| **Candidate-k (hybrid)** | Wider fusion pool for hybrid methods (default 20) |
| **Question limit** | Quick smoke test on N questions before full 700+ run |
| **Method multiselect** | Compare BM25, dense MiniLM/MPNet, hybrid variants |

#### 4. Run and persist results

- **Run benchmark** — executes retrieval for every (dataset × method × question) combination
  and computes aggregate metrics.
- **Save JSON results** — exports to `data/evaluation/results/` with timestamp for later comparison.

Saved runs let you compare **before/after** when you change embedding models, chunking, or
fusion parameters — without re-running manually.

#### 5. Results visualisation

After a run, the page offers three views:

| Tab | What it shows |
|-----|----------------|
| **Heatmaps** | Method × dataset performance matrix per metric — quick global comparison |
| **Bar charts** | Grouped bars per method — easier to read absolute differences |
| **Breakdowns** | Segment by **difficulty** (simple/complex), **persona** (natural set), or **query type** — exposes *where* a method wins or fails |

**Why segmentation matters:** a method with high overall Recall@5 might fail on complex
questions or on `consultant`-persona phrasing. Aggregate scores hide these failures.

#### 6. Filtered comparison

You can filter results by difficulty or persona **without re-running** the benchmark — useful
for diagnosing whether hybrid helps specifically on complex multi-chunk queries or on
conversational natural-language phrasing.

---

### How to interpret results

| Observation | Possible interpretation |
|-------------|-------------------------|
| BM25 wins on golden, hybrid wins on natural | Dense/semantic signal helps on paraphrased queries; lexical match enough for template-like golden questions |
| Recall@1 low but Recall@5 high | Right FAQ is in the neighbourhood but not ranked first — reranking might help |
| Complex Recall@K low across methods | Multi-chunk ground truth may be hard, or k is too small, or neighbour-based labels are imperfect |
| Method A beats B on raw golden but not cleaned | Method A may exploit unrealistic complex question artefacts |

Always cross-check surprising results in **Interactive test** with a few manual queries.

---

### What benchmarking does not cover

- **End-to-end answer quality** — use Chatbot + faithfulness eval for that.
- **Out-of-scope questions** — datasets only contain in-scope labelled queries.
- **Latency at scale** — 324 chunks is tiny; production corpora need separate perf testing.
- **Human preference** — metrics are automatic approximations, not user satisfaction.

---

### Typical workflow

1. Build indexes (once)
2. Run benchmark on **Golden (cleaned)** — establish baseline
3. Run on **Natural 748** — check realistic phrasing
4. Compare methods; pick default for Chatbot (typically `hybrid_minilm`)
5. Save JSON; change one variable; re-run; compare exports
6. Spot-check failures in **Interactive test** and **Data explorer**
