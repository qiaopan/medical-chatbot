# Agentic Medical RAG — Controlled Healthcare Assistant

> **Experimental starting point:** this repository began from the open-source [AIwithhassan/medical-chatbot](https://github.com/AIwithhassan/medical-chatbot) baseline. The work in this repository extends that baseline into an explainable, controlled Medical RAG prototype with explicit routing, high-risk handling, citations, and an auditable execution trace.

## Project overview

This project explores a practical question: **how can a retrieval-augmented healthcare assistant decide when it may retrieve grounded information, and when it must avoid an autonomous answer and escalate instead?**

It is an interview-oriented research prototype for local demonstration. It is not a clinical decision-support product, a diagnostic system, or a replacement for professional medical care.

## Upstream baseline

| Area | Starting point |
| --- | --- |
| Repository | [AIwithhassan/medical-chatbot](https://github.com/AIwithhassan/medical-chatbot) |
| Core idea | A medical chatbot using retrieval-augmented generation (RAG) |
| Role in this project | Experimental baseline used to reproduce and extend a RAG workflow |

The upstream project provided the initial application direction. The architecture, safety-oriented orchestration, retrieval improvements, source presentation, tests, and project documentation below describe the extensions developed in this repository.

## My contributions

- Added a lightweight, inspectable agent router with an explicit allow-list of controlled actions.
- Added deterministic high-risk checks for severe chest pain, possible stroke symptoms, severe breathing difficulty, and possible anaphylaxis.
- Added a controlled escalation path that skips routine RAG generation and avoids autonomous treatment recommendations for high-risk requests.
- Added a chronological agent trace showing input processing, risk classification, selected action, retrieval, generation, citations, and completion.
- Integrated local regex-based PII masking before external model calls.
- Improved retrieval with classification-enhanced queries, query rewriting, FAISS maximum marginal relevance (MMR), and retrieved-chunk deduplication.
- Added grounded-answer prompting and readable source citations with document, section, and page context.
- Added a Streamlit interface that exposes the answer, sources, routing decision, risk level, and agent trace.
- Added unit, retrieval, orchestration, and Streamlit smoke tests.

## Research question

Can a small RAG application be made more reliable and explainable by separating **routing and safety decisions** from **retrieval and answer generation**, while keeping the workflow lightweight enough to run locally?

## Architecture

```text
User question
    |
    v
Local PII masking
    |
    v
Deterministic high-risk checks ---- high risk ----> Controlled escalation
    |                                                   |
    | normal                                            v
    v                                             No routine RAG answer
Agent router / allowed action                           |
    |                                                   v
    v                                              Agent trace
search_medical_knowledge
    |
    v
Classification-assisted query -> query rewrite -> FAISS MMR retrieval
    |
    v
Grounded answer generation -> citations -> agent trace
```

The router deliberately exposes its decision logic in application code rather than delegating arbitrary tool choice to an LLM. The LLM cannot run shell commands or arbitrary tools.

## Experimental setup

| Component | Configuration |
| --- | --- |
| Interface | Streamlit localhost application |
| Knowledge source | New Zealand Immunisation Handbook content in a persisted FAISS index |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Retrieval | FAISS with MMR (`k=6`, `fetch_k=20`, `lambda_mult=0.5`) |
| Model provider | Groq via LangChain (`openai/gpt-oss-20b` by default; configurable) |
| Safety controls | Local PII masking, deterministic high-risk rules, controlled escalation |
| Trace | In-memory per-request execution trace |

## Results and current evidence

The repository includes automated checks for:

- deterministic routing of chest-pain, stroke, and breathing-emergency prompts to the escalation action;
- high-risk requests bypassing classifier and routine RAG retrieval;
- normal requests selecting the medical-knowledge retrieval action;
- retrieval loading the persisted FAISS index and returning six sourced chunks;
- grounded-answer prompts preserving citation structure; and
- Streamlit rendering of both the standard interface and the high-risk trace.

These checks demonstrate software behaviour, not clinical safety, medical accuracy, retrieval quality, or real-world effectiveness. Live model calls are opt-in and require a valid API key.

## Run locally

Use Python 3.11 and install the project dependencies. Set a Groq key in your environment:

```bash
export GROQ_API_KEY="your-key"
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
streamlit run optimal/medibot_category_env.py \
  --server.address 127.0.0.1 \
  --server.port 8501
```

Open `http://127.0.0.1:8501`.

To run the automated checks:

```bash
python -m unittest discover -s tests -v
python scripts/healthcheck.py
```

## Demo prompts

**Normal retrieval path**

```text
Can a 50-year-old still receive Tdap if they missed the 45-year vaccination?
```

**Controlled high-risk path**

```text
I have severe chest pain. What medicine should I take?
```

## Limitations

- Prototype only; not clinical advice, a validated triage system, or a production healthcare product.
- High-risk rules are intentionally narrow and conservative; they do not cover every emergency.
- PII masking is regex-based and is not a validated privacy control.
- Grounded citations are displayed, but claim-level citation correctness is not automatically scored.
- The standard path depends on external model availability and the configured API key.
- No production identity, access control, clinical workflow integration, or durable audit store is included.

## Repository map

| Path | Purpose |
| --- | --- |
| `optimal/medibot_category_env.py` | Streamlit application and controlled RAG workflow |
| `optimal/agent_orchestrator.py` | Deterministic routing, escalation action, and trace primitives |
| `optimal/connect_memory_with_llm_category_env.py` | Terminal RAG entry point |
| `tests/test_groq_app.py` | Automated routing, retrieval, prompt, and UI checks |
| `DEMO.md` | Concise interview-demo walkthrough |
| `docs/` | Architecture history and implementation notes |

## Responsible-use note

For symptoms that may indicate an emergency, seek urgent help from local emergency services. In New Zealand, call 111. This project intentionally avoids providing autonomous diagnosis or treatment recommendations for the high-risk scenarios it recognizes.
