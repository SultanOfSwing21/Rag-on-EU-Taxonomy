## Retrieval

### What retrieval actually does

Given a user question, retrieval answers one question:

> **Which FAQ chunks should the LLM see as context?**

Everything else in the RAG pipeline depends on this step. If retrieval returns the wrong FAQ,
the LLM will either hallucinate, abstain incorrectly, or give an incomplete answer — no amount
of prompt engineering fixes bad context.

### Two families of methods

We expose five retrieval methods, built from two underlying mechanisms:

| Family | Mechanism | Methods |
|--------|-----------|---------|
| **Lexical** | BM25 term matching | `bm25` |
| **Dense** | Embedding + **cosine similarity** | `dense_minilm`, `dense_mpnet` |
| **Hybrid** | BM25 rank + dense cosine rank → RRF fusion | `hybrid_minilm`, `hybrid_mpnet` |

### Dense retrieval: embeddings encode, cosine retrieves

A common misconception is that *"MiniLM retrieves documents"*. It does not.

The pipeline for dense retrieval is:

```
1. Encode each chunk (question + answer)  →  chunk vector   [embedding model]
2. Encode the user query                  →  query vector   [same embedding model]
3. Compute cosine similarity(query, each chunk vector)     [retrieval scoring]
4. Return top-k chunks by score
```

**The embedding model's job** is to map text into a vector space where semantically similar
texts are close together.

**Cosine similarity's job** is to measure that closeness and produce a ranked list.

We compare different **embedding models** (MiniLM vs MPNet) because they produce different
vector spaces — but the **retrieval metric is always cosine similarity** in this project.

#### Why cosine is sufficient here

Cosine similarity works well for our use case because of how we chunked the data:

| Property of our corpus | Implication for retrieval |
|------------------------|---------------------------|
| **1 FAQ = 1 chunk** | Each vector represents a **complete official answer**, not a sentence fragment |
| **324 chunks total** | Full scan is fast; no approximate search required |
| **Answers are not scattered** | We never need to retrieve and reassemble pieces from different parts of a document |
| **Query → FAQ mapping** | The task is *"which FAQ entry matches this question?"* — a whole-document matching problem |

In a long unstructured report, you might need hierarchical retrieval, passage reranking, or
multi-vector representations per document. Here, the FAQ format **already solved the segmentation
problem** at ingestion time. Cosine similarity between whole-chunk embeddings is the right tool.

#### What cosine does not solve

- **Exact acronym matching** when the query uses different words entirely → BM25 helps (hybrid).
- **Multi-FAQ questions** where several chunks are needed → increasing `k` and `candidate_k`
  helps, but ground truth for multi-chunk queries is harder to evaluate (see **Golden dataset**).
- **Domain vocabulary** not well represented in general embedding models → hybrid + BM25 mitigates
  part of this; fine-tuned embeddings would be a future step.

### BM25 retrieval

BM25 scores chunks by lexical overlap between query terms and chunk text.

- **Strength:** exact regulatory terms, activity names, acronyms.
- **Weakness:** fails when users paraphrase heavily or use conversational language.
- **Role in this project:** complementary signal in hybrid fusion, not the primary method alone.

### Hybrid fusion: Reciprocal Rank Fusion (RRF)

We combine BM25 and dense rankings with **RRF**, not by blending raw scores:

```
score(chunk) = Σ  1 / (k + rank_i)
```

summed over each ranked list (BM25 rank, dense cosine rank), with default `k = 60`.

#### Why RRF instead of score normalisation?

BM25 scores and cosine similarities live on **incomparable scales**. Normalising them (min-max,
z-score) is fragile and sensitive to the query and corpus. RRF only uses **ranks**, which makes
it robust: a chunk that is 1st in BM25 and 3rd in dense gets a high fused score regardless
of the absolute score values.

#### Why hybrid is the default (`hybrid_minilm`)

Empirically on FAQ-style corpora, hybrid retrieval captures:

- lexical precision (BM25) for regulatory terminology
- semantic flexibility (cosine/dense) for paraphrased user questions

MiniLM is the default dense backbone because it is **fast enough for interactive use** and
good enough for this corpus size. MPNet is available for comparison when quality matters more
than latency.

### Key parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `k` | 5 | Chunks passed to the LLM as context |
| `candidate_k` | 20 | Pool size before final top-k (especially for hybrid fusion) |

**Why k = 5?** Most simple questions need one FAQ; complex multi-topic questions may need 2–3.
Five chunks leave headroom without flooding the LLM context with irrelevant FAQs.

**Why candidate_k = 20?** Hybrid fusion needs a wider pool so both BM25 and dense candidates
can surface before the final cut. Too small a pool loses hybrid benefit; too large adds noise.

### How to compare methods

- **Interactive test** — same question, side-by-side rankings across methods (qualitative).
- **Benchmark** — Recall@K and MRR on labelled datasets (quantitative).

Default recommendation for this project: **`hybrid_minilm`**.
