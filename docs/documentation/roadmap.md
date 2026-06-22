## Known limits & future improvements

### Current limits (honest assessment)

| Area | Limitation | Practical impact |
|------|------------|------------------|
| **Corpus** | 324 English FAQs only | Out-of-scope questions not well handled |
| **Chunking** | 1 FAQ = 1 chunk, no overlap | Long FAQs may be harder to match semantically; splitting was rejected to preserve answer completeness |
| **Retrieval** | Cosine on whole-chunk embeddings, no reranker | Sufficient at this scale; may miss fine-grained ranking within top-10 |
| **Datasets** | Synthetic ground truth aligned with source chunks | Metrics optimistic vs real user query distribution |
| **Generation** | Single-turn, prompt-only | No conversational follow-up or clarification |
| **Faithfulness** | Lightweight NLI on CPU | Approximate on regulatory language; not a compliance audit |
| **Ops** | No CI regression, no tracing | Manual benchmark comparison only |

We document these openly because a consulting deliverable must show **judgment**, not just features.

---

### If more time were available

#### Retrieval & indexing

1. **Cross-encoder reranking** — re-score top-20 hybrid candidates with a dedicated
   query-chunk relevance model before passing top-k to the LLM.
2. **Query expansion / HyDE** — generate hypothetical answer before embedding search;
   useful if user queries are very short or vague.
3. **Vector database** — pgvector, OpenSearch, or managed service for larger or evolving corpora.
4. **Embedding model evaluation** — systematic comparison including domain-fine-tuned encoders.

#### Evaluation

5. **Human annotation loop** — expert labelling of a held-out query set for unbiased ground truth.
6. **RAGAS or equivalent** — answer relevancy, context precision, end-to-end scoring alongside NLI.
7. **CI benchmark regression** — fail PR if Recall@K drops beyond threshold on golden cleaned.
8. **Adversarial test set** — out-of-scope, ambiguous, and trick questions.

#### Generation & product

9. **Multi-turn sessions** — retrieve fresh context per turn with explicit memory boundaries.
10. **Structured output** — JSON schema enforcing citations and abstention.
11. **Observability** — Langfuse, OpenTelemetry: latency, cost, faithfulness per request.
12. **Incremental FAQ ingestion** — detect source updates, refresh affected indexes only.

---

### Hypotheses & assumptions

These underpin the architecture. If any prove false in a real deployment, the design should change.

| Assumption | If wrong… |
|------------|-----------|
| Users ask in English | Need multilingual embeddings and translated FAQs |
| Official FAQ text is authoritative | Need versioned sources and update pipeline |
| Most answers live in one FAQ chunk | Need splitting or multi-hop retrieval |
| Paraphrase is the main retrieval challenge | Need stronger lexical resources or domain embeddings |
| Single-turn covers primary use case | Need session management and context carry-over |
| Synthetic benchmarks approximate production | Need production query logging and human eval |
| Measuring retrieval + faithfulness is enough for iteration | Need end-to-end user satisfaction metrics |

---

### How this project should be positioned

This is a **demonstration of method**: structured ingestion, justified retrieval choices,
programmatic evaluation datasets with known biases, benchmarking UI, and faithfulness tracking.

It is **not** production-ready compliance software. The value is in the **reasoning chain**
visible in Documentation — each choice motivated, measured, and limited.

That is the expected posture for an AI architect role: ship something that runs, prove something
that improves, and know exactly what you would do next.
