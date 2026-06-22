## Generation & prompt engineering

### Why we chose prompt engineering over fine-tuning

The assignment scope is a **closed FAQ corpus** (324 entries) with a strong retrieval layer
already in place. In this setting:

| Approach | Pros | Cons for this project |
|----------|------|------------------------|
| **Fine-tuning** | Can adapt model tone and domain vocabulary | Requires training data, compute, model versioning; hard to iterate quickly; retrieval quality still dominates |
| **Prompt engineering** | Fast to change, easy to audit, no training pipeline | Depends on model following instructions |

We prioritised **measurable iteration**: change retrieval method or prompt rules, re-run benchmarks,
compare faithfulness scores. Fine-tuning would add operational complexity without guaranteed
gain on a corpus where answers already exist verbatim in retrieved chunks.

The LLM's job here is narrow: **summarise and cite retrieved FAQs**, not invent regulatory knowledge.

---

### Design goals for generation

1. **Grounded** — every claim traceable to retrieved context
2. **Scoped** — refuse to answer from outside knowledge
3. **Honest** — abstain when context is insufficient
4. **Traceable** — cite chunk IDs so users can verify sources
5. **Consistent** — low temperature, stable behaviour across sessions

---

### System prompt rules

The system prompt enforces:

- Answer **only** from the context provided below
- **Do not use outside knowledge**
- If context is insufficient, reply exactly:  
  `"I cannot answer this question from the available context."`
- When relevant, cite chunk IDs (e.g. `[faq-0042]`)
- Be concise and precise

#### Why strict abstention?

Regulatory Q&A has a **high cost of confident wrong answers**. A generic helpful tone
(*"I think companies should…"*) is worse than a clear *"I cannot answer from the available
context."* We canonicalise the abstention phrase so faithfulness evaluation can detect it.

#### Why citations?

Chunk IDs connect the generated answer to **auditable sources**. In a compliance context,
*"the model said X"* is insufficient — reviewers need *"the model said X based on FAQ faq-0042"*.

---

### User prompt structure

```
Context:
  [faq-0001] (section: …)
  Question: …
  Answer: …

  ---

  [faq-0042] …

Question:
  <user query>

Answer using ONLY the context above.
```

**Motivation:** separate system rules (behaviour) from user content (data). The context block
mirrors the chunk structure from ingestion — question + full answer — so the LLM sees the same
units retrieval selected.

---

### LLM configuration

| Setting | Default | Rationale |
|---------|---------|-----------|
| Model | `gpt-4o-mini` | Good instruction-following, low cost, fast enough for demo |
| Temperature | `0.2` | Factual consistency; less creative paraphrase of regulatory text |
| Max tokens | `1024` | FAQ answers are long but rarely need more for a focused reply |

#### Supported providers

OpenAI, Azure OpenAI, AWS Bedrock, OpenAI-compatible APIs.

**Why multi-provider?** Consulting deployments rarely standardise on one vendor. The chat
layer is isolated behind a `ChatClient` protocol so the RAG pipeline stays provider-agnostic.

Configure in **Chatbot → Connection** or via `.env`.

---

### Single-turn only

The application handles **one isolated question per request** — no conversation memory.

**Why:**

- Matches the stated use case (regulatory lookup, not a coaching session)
- Avoids context drift where earlier turns introduce information not in retrieved FAQs
- Simplifies evaluation — each turn is independently scored

Multi-turn would require explicit memory management and retrieval refresh per turn.

---

### What we did not optimise (yet)

- **Answer style adaptation per persona** — same prompt for all users
- **Query rewriting before retrieval** — user question goes directly to retriever
- **Citation enforcement via structured output** — citations are prompt-requested, not schema-validated
- **Streaming responses** — full response generated before display

These are natural extensions if the system moves toward production.
