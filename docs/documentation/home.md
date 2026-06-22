### Why this project exists

Regulatory knowledge must be **accurate, traceable, and auditable**. Generic LLMs can sound
confident while citing nothing — unacceptable when the topic is EU Taxonomy compliance.

This application combines three ideas we consider non-negotiable for domain RAG:

1. **Retrieval first** — find the right official FAQ before generating anything
2. **Constrained generation** — answer only from retrieved context, with explicit abstention
3. **Measurement** — score both retrieval quality and answer faithfulness, and persist results
   so experimentation is evidence-based, not anecdotal

We did not set out to build "a chatbot". We set out to build a **measurable RAG loop** on a
closed regulatory corpus — where every design choice can be explained, tested, and improved.

### What you can do here

| Area | What to try |
|------|-------------|
| **Ask** | **Chatbot** — single-turn Q&A with citations and per-answer faithfulness scoring |
| **Compare** | **Interactive test** — same question, side-by-side across retrieval methods |
| **Benchmark** | **Benchmark** — Recall@K & MRR on three labelled datasets (1,400+ queries total) |
| **Explore** | **Data explorer** — browse FAQ chunks and evaluation question sets |
| **Understand** | **Documentation** — motivations, trade-offs, and methodology behind every layer |

### Design philosophy

> Measure first, generate second. If retrieval fails, no prompt engineering will save the answer.

Most RAG demos stop at a chat interface. Here, the chatbot is the **last step** of a pipeline
that starts with chunking strategy, index design, cosine-based dense retrieval, hybrid fusion,
programmatic golden datasets, LLM-persona natural queries, and NLI faithfulness tracking.

Every response can be traced to FAQ chunk IDs. Every chat turn can be scored against the
chunks that were actually retrieved.

### Pipeline at a glance

### Under the hood

- **Hybrid search** — BM25 lexical matching + dense **cosine similarity**, fused with RRF
- **1 FAQ = 1 chunk** — whole official answers, not scattered passages; cosine ranks FAQ units
- **Three evaluation datasets** — programmatic golden (raw + cleaned) and ChatGPT natural rewrites
- **Two KPI layers** — retrieval benchmarks (JSON) + generation faithfulness (SQLite)
- **No API key required** for benchmarks, exploration, and documentation

Read the **Documentation** sections for the full reasoning behind each choice.
