## Indexing

### Why indexing is a separate step

Retrieval must be **fast and repeatable**. At query time, we cannot re-embed all 324 chunks
and scan the full corpus from scratch on every user click — especially during benchmarks
that run hundreds of queries × several methods.

Indexing pre-computes the data structures that make search efficient: inverted term frequencies
for BM25, and pre-stored embedding matrices for dense retrieval.

Indexes are built **on demand** from the **Benchmark** page (or automatically before RAG if
they are missing). This keeps the repository lightweight: indexes depend on the local environment
and downloaded models, so they live in `.cache/index/` rather than in git.

### Three index families

| Index | Technology | What it captures |
|-------|------------|------------------|
| **BM25** | `bm25s` lexical index | Keyword overlap — strong on regulatory acronyms and exact terms (TSC, DNSH, CapEx, NFRD) |
| **Dense MiniLM** | `all-MiniLM-L6-v2` | Semantic similarity — catches paraphrases when users do not use official wording |
| **Dense MPNet** | `all-mpnet-base-v2` | Higher-quality embeddings, slower — useful for comparison, not required for the default path |

#### Why both lexical and dense?

FAQ users may ask questions in two very different ways:

- **Lexical:** *"What are the DNSH criteria for forest management?"* — close to official terms → BM25 excels.
- **Semantic:** *"How do we prove our activity does no significant harm to ecosystems?"* — no exact keyword overlap → dense retrieval helps.

Neither method alone covers both patterns well. That is why hybrid retrieval (see **Retrieval** section) combines them.

### Embeddings vs retrieval: an important distinction

It is easy to conflate **embedding models** with **retrieval logic**. They play different roles:

| Role | What it does |
|------|--------------|
| **Embedding model** (MiniLM, MPNet) | Converts text (chunk or query) into a fixed-size vector. This is an *encoding* step. |
| **Retrieval scoring** (cosine similarity) | Compares the query vector to chunk vectors and ranks chunks by similarity. This is the *ranking* step. |

**The embedding model does not retrieve anything by itself.** It only produces vectors.
**Cosine similarity** (dot product of L2-normalised vectors) is what actually scores and
ranks chunks in our dense index.

We use cosine because:

1. **It is the standard metric** for sentence embedding models trained with cosine-style objectives.
2. **Our chunks are whole FAQ units** — each vector represents a complete Q&A, not a fragment.
   We are matching *question intent → FAQ entry*, not finding a specific sentence in a long document.
3. **The corpus is small (324 chunks)** — a full cosine scan is trivially fast; we do not need
   approximate nearest-neighbour heuristics for this project scale.

### Dense backend: NumPy vs FAISS

| Backend | When used | Rationale |
|---------|-----------|-----------|
| **NumPy** (default) | Always available | Brute-force cosine over 324 normalised vectors is sub-millisecond on CPU. No extra dependency. |
| **FAISS** (optional) | `pip install -e ".[faiss]"` | Same cosine logic, optimised for larger corpora. Relevant if the FAQ grows to thousands of chunks or multi-source ingestion. |

Both backends store **the same embeddings**; only the search implementation differs.

Embeddings are cached on disk under `.cache/index/` so rebuilds are skipped unless forced.

### BM25 index

BM25 builds a sparse inverted index over tokenised chunk text (question + answer combined).

- **Motivation:** regulatory FAQs contain precise terminology that dense models sometimes
  smooth over. BM25 preserves exact token matches.
- **Limitation:** brittle on paraphrases and conversational phrasing — which is why we do not
  rely on BM25 alone.

### How to build indexes

1. Open **Benchmark**
2. Select the retrieval methods you want to evaluate
3. Click **Build indexes**

The first build downloads embedding models from Hugging Face and may take several minutes.
Subsequent launches reuse the cache.
