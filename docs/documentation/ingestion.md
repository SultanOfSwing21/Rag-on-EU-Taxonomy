## Ingestion & chunking

### Why this step matters

Before any retrieval or generation can work, the raw FAQ document must become a **structured,
searchable knowledge base**. The quality of every downstream step — retrieval scores, LLM answers,
faithfulness checks — depends on how faithfully we represent the source material.

We deliberately kept this layer **simple and deterministic**: no LLM-based chunking, no sliding
windows, no semantic splitting. The FAQ document already defines its own boundaries.

### Source document

**File:** `data/taxonomy_faqs_cleaned.md`

- **324** official EU Taxonomy FAQ entries
- **7** thematic sections (Climate Delegated Act, Disclosure, Eligibility reporting, etc.)
- Language: English (matching the official source)

### Core decision: one FAQ = one chunk

Each chunk contains:

- the **official question** (as written in the FAQ)
- the **full answer** (including sub-headings and examples inside the answer)
- **metadata** (section name, position in file, source path)

#### Why we chose this

The EU Taxonomy FAQ is not a long unstructured report. It is a **reference document** where
each entry was authored as a standalone unit: one question, one complete answer.

| Approach | What goes wrong |
|----------|-----------------|
| **Split one FAQ into several chunks** | Retrieval may return only part of an answer. Sub-sections, conditions, or examples are lost — and the chatbot answers without the full official context. |
| **Merge several FAQs into one chunk** | Unrelated topics get mixed. A user asking about reporting rules might retrieve content about forest management. Citations become ambiguous. |
| **One FAQ = one chunk** | Each retrieval returns a **complete, traceable** official answer. The chunk ID maps 1:1 to a known FAQ entry. |

This choice also shapes retrieval strategy later: we are not looking for a needle in a haystack
scattered across paragraphs. We are looking for **the right FAQ entry** among 324 self-contained
units. That is a fundamentally different — and easier — retrieval problem than open-domain
document search.

#### Connection to cosine similarity (dense retrieval)

Because each chunk is a **complete answer**, we do not need fine-grained passage retrieval
within long documents. The embedding of a chunk represents an entire Q&A pair. Cosine similarity
between the query embedding and chunk embeddings is enough to rank **which FAQ** is relevant,
not which sentence inside a 50-page PDF.

### Parser edge case: false `###` headings

The markdown file contains **329 lines** starting with `###`, but only **324** are real questions.

Five lines are **sub-headings inside answers** — internal titles that break a long official
response into parts (e.g. *"Mandatory use: disclosure"* inside the answer to
*"How will companies use the EU Taxonomy?"*).

They use the same `###` syntax as real questions, which confuses naive parsers.

**Our rule:** if a `###` line does not read like a standalone user question (no `?`, no
question word like *How / What / Will*), it stays **inside the parent answer** rather than
becoming a separate chunk.

Nothing is deleted — the text is preserved in context. We simply avoid creating five fake
FAQ entries that would pollute retrieval and evaluation.

See [`docs/chunking.md`](../chunking.md) for the full analysis of all five cases.

### Caching

Parsed chunks are written to `.cache/chunks.jsonl` after the first run.

- **Why cache?** Parsing is fast, but downstream steps (embedding, indexing) depend on stable
  chunk IDs. Caching guarantees reproducibility across sessions.
- **Force rebuild:** `eu-taxonomy-rag --force-rebuild` re-parses from source if the FAQ file
  changes.

### What we did not do (and why)

- **No overlap between chunks** — not needed when each chunk is already a complete unit.
- **No metadata enrichment via LLM** — keeps ingestion auditable; all content comes from the
  official file.
- **No multi-language support** — the source is English only; adding translations would require
  a separate quality and alignment strategy.
