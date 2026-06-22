## Faithfulness / groundedness

### Why measure faithfulness?

Retrieval metrics tell us whether we found the right FAQs. They do **not** tell us whether
the LLM **used them correctly**.

A model can retrieve the perfect chunk and still:

- add facts not present in the context
- contradict the official answer
- over-generalise from partial context

**Faithfulness** (also called **groundedness**) measures whether the **generated answer is
supported by the retrieved chunks** — not by the full corpus, not by the model's training data.

This is the second layer of our evaluation strategy (see **KPI tracking**).

---

### Why NLI instead of LLM-as-judge?

| Approach | Why we chose / rejected |
|----------|-------------------------|
| **NLI model (local)** | Reproducible, offline, no API cost per evaluation, consistent across runs |
| **LLM-as-judge** | Flexible but expensive, variable across models/temperatures, harder to reproduce |
| **RAGAS / frameworks** | Powerful end-to-end metrics — valuable future step, more dependencies |

We use `typeform/distilbert-base-uncased-mnli` — a lightweight Natural Language Inference
model that classifies whether a **premise** (chunk text) entails a **hypothesis** (claim).

Good enough for **relative comparisons** during experimentation; not a perfect regulatory auditor.

---

### Pipeline

1. **Generate** answer with LLM (Chatbot)
2. **Split** answer into short claims (sentence-based splitting)
3. For each claim, score entailment against **each retrieved chunk**
4. **Label** each claim:
   - `supported` — entailed by at least one chunk
   - `contradicted` — contradicted by context
   - `not_enough_info` — neutral / insufficient evidence
5. **Aggregate** into session metrics
6. **Persist** to SQLite for history

#### Special case: canonical abstention

If the model outputs the exact abstention phrase  
*`"I cannot answer this question from the available context."`*  
claims are scored accordingly — abstention is treated as a valid grounded behaviour when
context is genuinely insufficient.

---

### Metrics

| Metric | Meaning | How to use it |
|--------|---------|---------------|
| **Faithfulness** | `supported_claims / total_claims` | Overall groundedness — higher is better |
| **Contradiction rate** | Share of contradicted claims | **Red flag** — model conflicts with source |
| **Unsupported rate** | Share of `not_enough_info` | May indicate vague answer or weak retrieval |
| **Best / avg claim score** | Entailment confidence | Diagnostic — low scores despite `supported` label |

#### How to interpret during experimentation

When you change retrieval method or prompt:

- **Faithfulness up, retrieval Recall@K up** — healthy improvement
- **Faithfulness down, retrieval stable** — prompt or model issue
- **Faithfulness down, retrieval down** — fix retrieval first
- **High contradiction rate** — serious issue; review retrieved chunks and answer manually

---

### Important limitations

We are explicit: this is a **diagnostic tool**, not a compliance certification.

| Limitation | Impact |
|------------|--------|
| NLI trained on general text | Struggles with regulatory paraphrases and domain acronyms |
| Sentence-level claim splitting | Misses nuance across sentences; compound claims may be mis-scored |
| Scores only retrieved chunks | If retrieval missed the right FAQ, faithfulness cannot recover |
| No human review loop | Automatic labels can disagree with expert judgment |
| English only | Matches corpus language |

**Garbage in, garbage out:** faithfulness evaluates against **what was retrieved**, not the
full 324-FAQ corpus. Poor retrieval caps faithfulness regardless of LLM quality.

Disable entirely with `ENABLE_GENERATION_EVAL=false` if you only need retrieval benchmarking.

---

### Where to view results

| Location | Content |
|----------|---------|
| **Chatbot → Chat** | Per-answer claim breakdown immediately after generation |
| **Chatbot → History** | Past interactions with scores |
| **Chatbot → Metrics** | Aggregate trends over session / time |

---

### Why this matters for the project narrative

The assignment asked for **at least one metric** to track improvement during experimentation.

We track **two families**:

1. **Retrieval** — Recall@K, MRR (Benchmark)
2. **Faithfulness** — NLI claim labels (Chatbot)

Together they cover the full RAG loop: *find the right FAQ* → *say something faithful about it*.
