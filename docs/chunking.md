# How We Read the EU Taxonomy FAQ File

This note explains, in plain language, how the project reads the FAQ document (`data/taxonomy_faqs_cleaned.md`) before turning each question-and-answer pair into a searchable unit for the chatbot.

---

## What we are trying to do

The chatbot needs to search through hundreds of official EU Taxonomy questions and answers. Before it can do that, the raw document must be split into clear, separate items:

- one **question**
- one **answer**
- some **context** (for example, which section of the document it belongs to)

Each valid question-and-answer pair will later become **one chunk** — one self-contained piece of knowledge the chatbot can retrieve when a user asks something similar.

---

## Why one chunk = one FAQ

This project treats **each FAQ entry as a single, indivisible unit of knowledge**. That is a deliberate choice, not just a convenient default.

### How official FAQs are designed

The EU Taxonomy FAQ document is written as a reference tool. Each entry follows the same pattern:

- **One question** — a specific point a reader might have
- **One answer** — the precise official response to that point, including any internal structure (sub-headings, examples, cross-references) needed to understand it fully

The authors did not write the document as one long essay to be cut into arbitrary pieces. They wrote it as **324 separate question-and-answer pairs**, each meant to stand on its own.

### What a chunk needs to contain

When someone asks the chatbot a question, the system should retrieve **exactly the FAQ entry that answers it** — not half of one answer, not a mix of two unrelated entries.

That means each chunk must include:

- the **question** (so the system knows what topic the chunk is about)
- the **full answer** (so the chatbot has everything it needs to respond accurately)
- **metadata** such as the section title (so answers can be traced back to their source)

Nothing essential should be missing. Nothing unrelated should be added. In other words: **one chunk mirrors one FAQ, as the document authors intended.**

This is why the five sub-headings described later in this document are **kept inside the parent answer** rather than turned into separate chunks. They are part of the official reply — not standalone questions. Splitting them out would give the chatbot an incomplete answer.

### Why we do not split or merge FAQs

| Approach | Problem for this project |
|---|---|
| **Split one FAQ into several chunks** | Risk of retrieving only part of an answer. Sub-headings, examples, or conditions may be lost — and the chatbot would answer without the full official context. |
| **Merge several FAQs into one chunk** | Mixes unrelated topics. A user asking about reporting rules might get content about forest management. Retrieval becomes less precise, and citations become unclear. |
| **One FAQ = one chunk** | Matches the document structure. Each retrieval returns a complete, traceable official answer. |

### Is this always the right approach?

For **this document and this project**, yes. FAQ-style content is a strong case for one-chunk-per-entry because the authoring format already defines the natural boundaries.

In other situations — for example, very long policy documents or unstructured reports — smaller or overlapping chunks can sometimes work better. Some answers in this FAQ are long, and that can make search slightly harder. But splitting them would trade a small retrieval challenge for a much bigger problem: **incomplete or misleading answers**.

For a homework focused on retrieval quality and faithful answers from official sources, **respecting the FAQ unit is the right call**.

---

## How the document is organised

The FAQ file is written in Markdown, a simple text format. It uses headings to structure the content:

| Marker | Meaning | Example |
|---|---|---|
| `##` | A **section** title (a broad theme) | `## Climate Delegated Act` |
| `###` | Usually a **question** | `### Will the technical screening criteria…?` |
| Plain text below | The **answer** to that question | One or more paragraphs |

In total, the file contains:

- **7 sections**
- **329 lines** starting with `###`
- **324 real questions** once the file is read correctly

---

## What the parser does

The parser is a small program that reads the file automatically. For each real question, it extracts:

1. **The question** — the text after `###`
2. **The answer** — all the text that follows, until the next question or section
3. **Metadata** — extra useful information, such as:
   - which section the item belongs to
   - its position in the file (1, 2, 3…)
   - the source file name

The result is a list of **324 FAQ items**, each ready to become one chunk in the next step.

---

## Why 329 `###` lines become only 324 questions

At first glance, it looks like there are **329 questions** because there are **329 lines** starting with `###`.

In practice, **5 of those lines are not questions**. They are **sub-headings inside an answer** — smaller titles that break a long answer into parts. They use the same `###` format as real questions, which can be confusing when reading the file by eye or by a simple script.

The parser is designed to tell the difference:

- **Real question** → starts a new FAQ item
- **Sub-heading** → stays inside the answer of the question above it

---

## The 5 lines we did not keep as separate questions

### 1. Line 1365 — *Contribution to climate change mitigation (CCM) and climate change adaptation (CCA)*

**Parent question:**  
*How should reporting undertakings address ‘double-counting’ in the context of business activities contributing to multiple environmental objectives?*

**Why not kept separately:**  
This line is a **section title within the answer**, not a new question. It introduces the first part of a long, structured reply about climate change mitigation and adaptation. Keeping it as its own question would split one answer into several unrelated items and lose important context.

---

### 2. Line 1386 — *For activities contributing substantially to CCM, treatment of CapEx…*

**Parent question:**  
Same as above (*double-counting* question).

**Why not kept separately:**  
This is another **internal sub-heading** in the same answer. It discusses how capital expenditure (CapEx) should be treated for certain activities. It does not ask something new on its own — it continues explaining the same topic.

---

### 3. Line 1390 — *Treatment of an activity making substantial contribution to multiple environmental objectives…*

**Parent question:**  
Same as above (*double-counting* question).

**Why not kept separately:**  
Again, this is a **sub-part of the same answer**, covering activities that contribute to more than one environmental objective. Treating it as a standalone question would duplicate and fragment the explanation.

---

### 4. Line 2860 — *Mandatory use: disclosure*

**Parent question:**  
*How will companies use the EU Taxonomy?*

**Why not kept separately:**  
The answer to this question is organised in two parts: mandatory uses and voluntary uses. *Mandatory use: disclosure* is the **title of the first part**, not a separate FAQ entry. The text below it explains disclosure obligations in detail.

---

### 5. Line 2868 — *Voluntary use: guide for investments*

**Parent question:**  
*How will companies use the EU Taxonomy?*

**Why not kept separately:**  
This is the **title of the second part** of the same answer, covering voluntary uses of the Taxonomy (for example, to guide investment decisions). It belongs with the parent question, not as an independent FAQ.

---

## How the parser tells a question from a sub-heading

Real FAQ questions in this document usually:

- **end with a question mark** (`?`), or
- **start with a question word** such as *How*, *What*, *Will*, *Can*, *Is*, *Are*, *Should*, and so on.

The 5 sub-headings above do neither. They read like **topic labels** or **chapter titles** inside a longer answer, not like questions a user would ask on their own.

That is why they are **included inside the answer text** rather than treated as new FAQ items.

---

## What happens next

The next step in the project is **chunking**: each of the **324 parsed FAQ items** will become **one chunk** with a unique ID, keeping the question, answer, and metadata together — following the one-FAQ-per-chunk principle described above.

This keeps retrieval simple, preserves complete official answers, and makes it easy to show the user exactly which FAQ was used to generate a reply.

---

## Summary

| Item | Count |
|---|---|
| Lines starting with `###` in the file | 329 |
| Real questions identified | 324 |
| Sub-headings kept inside answers | 5 |
| FAQ items ready for chunking | 324 |

The parser does not delete any content from those 5 sub-headings — it **keeps their text inside the parent answer** so nothing is lost. It simply avoids treating them as separate questions, which would create incorrect or incomplete search results.
