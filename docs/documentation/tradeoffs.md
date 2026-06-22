## Design decisions & trade-offs

This page documents **why** we chose specific approaches — not just what was implemented.
Each decision reflects constraints of the project: closed FAQ corpus, single-turn Q&A,
need for measurable experimentation, and deliverability within limited time.

---

### Ingestion & chunking

| Decision | Alternative considered | Why we chose this |
|----------|------------------------|-------------------|
| **1 FAQ = 1 chunk** | Sliding window, semantic splits, sentence-level chunks | Official FAQs are self-contained Q&A units; splitting risks incomplete answers; merging mixes topics |
| Rule-based parser | LLM-based chunking | Deterministic, auditable, no API cost; every chunk traceable to source line |
| Keep sub-headings inside answers | Treat all `###` as questions | Five false positives in source file; parser heuristic preserves content without fake chunks |

---

### Indexing & retrieval

| Decision | Alternative considered | Why we chose this |
|----------|------------------------|-------------------|
| **Cosine similarity for dense retrieval** | Dot product on unnormalised vectors, Euclidean distance | Standard for sentence embeddings; with L2-normalised vectors, cosine = dot product; interpretable as semantic angle |
| Embedding models as encoders only | "MiniLM retrieves" mental model | Models produce vectors; cosine ranks them — separate concerns, clearer debugging |
| Cosine sufficient without reranking | Cross-encoder reranker | 324 whole-FAQ chunks; retrieval is "which FAQ?" not "which paragraph?"; cosine + hybrid covers most cases at this scale |
| **Hybrid RRF** | Weighted sum of BM25 + cosine scores | Scores on incomparable scales; rank fusion is robust and parameter-light |
| NumPy index default | FAISS required | Full cosine scan on 324 vectors is instant; avoids dependency unless corpus grows |
| BM25 + dense (not dense alone) | Dense-only retrieval | Regulatory acronyms and exact terms favour lexical matching; hybrid combines both signal types |
| Default `hybrid_minilm` | `hybrid_mpnet` or dense-only | Best speed/quality trade-off for interactive demo |

---

### Evaluation datasets

| Decision | Alternative considered | Why we chose this |
|----------|------------------------|-------------------|
| Programmatic golden dataset | Manual expert annotation | Reproducible, scalable, zero annotation cost; seed 42 for regression |
| Validation / cleaned pass | Use raw golden only | Raw complex questions had template artefacts; cleaned set measures retrieval not grammar robustness |
| Natural dataset via ChatGPT | More rule-based paraphrases | Personas simulate realistic business users; tests paraphrase beyond templates |
| Keep `expected_chunk_ids` from source | Re-label after LLM rewrite | Ground truth stable across golden → natural; fair method comparison |
| Three datasets in UI | Single combined set | Raw vs cleaned vs natural expose different failure modes and biases |
| Simple + complex questions | Simple only | Multi-FAQ queries reflect real regulatory complexity; tests top-k coverage |

**Acknowledged trade-off:** all datasets are **constructed with knowledge of target chunks** —
good for controlled benchmarks, not fully representative of production query distribution.

---

### Generation & faithfulness

| Decision | Alternative considered | Why we chose this |
|----------|------------------------|-------------------|
| Prompt engineering | Fine-tuning on FAQ pairs | Faster iteration; answers exist in retrieved text; retrieval quality dominates |
| Strict abstention phrase | Open-ended "I don't know" | Canonical phrase detectable in eval; reduces false confidence |
| Chunk ID citations | No citations | Auditability for regulatory context |
| Local NLI faithfulness | RAGAS, LLM-as-judge | Offline, reproducible, no per-eval API cost |
| Sentence-level claims | Atomic proposition extraction model | Simpler pipeline; good enough for diagnostic trends |
| SQLite for chat eval | Same JSON as benchmarks | Online append-only history vs batch snapshot exports serve different workflows |

---

### Application & delivery

| Decision | Alternative considered | Why we chose this |
|----------|------------------------|-------------------|
| Streamlit UI | FastAPI + React, CLI only | Fast to build; sufficient for demo; business logic stays in `src/` package |
| Single-turn chat | Multi-turn memory | Matches use case; simpler eval |
| Indexes built on demand | Pre-committed index artefacts in repo | Repo stays light; embeddings depend on local env |
| No API key for benchmarks | LLM required everywhere | Reviewers can evaluate retrieval without credentials |
| Docker + docker-compose | Local-only setup | Reproducible deployment for reviewers and portfolio |

---

### How to read this table in an interview

Each "alternative considered" is a conversation starter. The message is not that our choices
are universally optimal — it is that they are **appropriate for this corpus, scale, and goal**:
a measurable, demonstrable RAG loop on a closed regulatory FAQ.

Production would revisit: vector database, reranking, human eval loop, observability, and
multi-turn session management.
