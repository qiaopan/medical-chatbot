# Architecture History

This document records the repository's architecture as it changed over time, why each change was made, and what remains planned. Dates use the local New Zealand development date.

## Current architecture snapshot — 2026-08-13

The application is a local Streamlit medical RAG demo focused on New Zealand immunisation guidance.

```text
User question (Streamlit or terminal)
        |
        v
Local regex PII masking
        |
        v
Deterministic router / emergency rules
        |
        +-----------------------------+
        |                             |
        v                             v
Groq classification          Controlled escalation
        |                     (normal RAG skipped)
        v
Query enrichment + rewrite
        |
        v
MiniLM -> FAISS MMR -> evidence
        |
        v
Grounded Groq answer + citations
        +-------------+---------------+
                      v
                Agent Trace
```

### Runtime components

| Area | Current implementation |
| --- | --- |
| Browser UI | Streamlit: `optimal/medibot_category_env.py` |
| Terminal RAG | `optimal/connect_memory_with_llm_category_env.py` |
| Classification and PII masking | `optimal/category/classify_text_to_json.py` |
| Ingestion/index builder | `optimal/create_memory_for_llm_optimized_fixed.py` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`, CPU-local |
| Vector store | Committed FAISS index under `vectorstore/db_faiss/` |
| Retrieval | LangChain FAISS MMR with six final chunks |
| LLM provider | Groq through `groq` and `langchain-groq` |
| Default model | `openai/gpt-oss-20b`, configurable with `GROQ_MODEL` |
| Configuration | Ignored root `.env`; `GROQ_API_KEY` is required |
| Runtime | Native Apple Silicon Python 3.11 environment in ignored `.venv/` |
| Automated checks | `unittest`, Streamlit `AppTest`, optional live Groq test, HTTP healthcheck |
| Orchestration | Typed plain-Python router in `optimal/agent_orchestrator.py` |
| Controlled actions | `search_medical_knowledge`, `escalate_high_risk_request` |
| Trace | In-memory JSON-friendly ordered events stored with Streamlit chat state |

### Current request lifecycle

1. The UI receives the user's question.
2. The classifier module masks prototype PII patterns locally.
3. Deterministic rules check for obvious emergencies before any remote model call.
4. A high-risk request selects `escalate_high_risk_request`, skips ordinary RAG, and returns a controlled response.
5. A normal request proceeds to Groq structured classification and selects `search_medical_knowledge`.
6. Classification labels enrich the query; Groq rewrites it for search.
7. MiniLM embeds the query and FAISS performs MMR selection from 20 candidates to six chunks.
8. Groq produces an answer constrained to retrieved evidence.
9. The UI displays answer, sources, risk, selected action, rewritten query, classification, and trace.

The terminal entry point still uses the direct RAG flow. The interview UI is the canonical agentic path.

## Change log

### 2026-08-13 — Apple Silicon local runtime established

Reason:

- The new development machine is Apple Silicon, while the installed global Homebrew, Python, and Anaconda tools were Intel builds running through Rosetta.
- The system Python was 3.14, which is incompatible with the frozen `numpy==2.3.1` marker and risky for the pinned ML/scientific stack.

Changes:

- Created an isolated, native `osx-arm64` Python 3.11 environment at `.venv/`.
- Installed the frozen project dependencies without global installation.
- Confirmed native wheels for FAISS, Torch, NumPy, SciPy, PyArrow, tokenizers, and related compiled packages.
- Added `.venv/` and generated `__pycache__/` directories to `.gitignore`.

Verification:

- Core modules imported successfully.
- The committed FAISS index loaded successfully.
- A real MMR query returned six Handbook chunks.
- Streamlit started and its health endpoint returned HTTP 200.

### 2026-08-13 — Azure OpenAI replaced by Groq

Reason:

- The Azure endpoint/configuration was unavailable or expired.
- Groq was requested as a lower-cost provider for the interview demo.
- The repository contained an older Groq implementation in Git history, confirming Groq was already an established project direction.

Changes:

- Replaced `AzureChatOpenAI` with LangChain `ChatGroq` for query rewriting and grounded answer generation.
- Replaced the Azure OpenAI classifier client with the Groq Python client.
- Consolidated runtime configuration around `GROQ_API_KEY`.
- Added configurable `GROQ_MODEL`, `GROQ_CLASSIFIER_MODEL`, and `GROQ_REQUEST_TIMEOUT` settings.
- Selected `openai/gpt-oss-20b` as the default cost-conscious model.
- Added a 30-second timeout and two retries for remote model requests.
- Restored pinned `groq==0.29.0` and `langchain-groq==0.3.5`; removed Azure-specific direct dependencies from `requirements.txt`.

Retrieval semantics were not changed: the application still uses the same MiniLM embedding model, committed FAISS index, MMR settings, query enrichment, evidence-only prompt, and citation formatting.

### 2026-08-13 — Offline retrieval reliability and automated checks

Reason:

- Cached MiniLM startup still attempted a Hugging Face network metadata request, causing offline tests to fail.
- Repeated manual API-key and frontend checks were error-prone.

Changes:

- Configured runtime query embeddings to load MiniLM from the existing local cache only. The separate index builder retains its normal download behavior.
- Added `tests/test_groq_app.py` covering:
  - clear missing-key errors;
  - Groq model configuration;
  - classifier import and local PII availability without credentials;
  - FAISS/MMR retrieval and source metadata;
  - grounded-answer citation wiring;
  - Streamlit page rendering;
  - an opt-in live Groq key/model test.
- Added `scripts/healthcheck.py` for repeatable Streamlit HTTP health checks.
- Verified a complete live path: Groq classification, query rewrite, six-chunk retrieval, grounded generation, and citations.

### Earlier baseline — before 2026-08-13

The current baseline was already more capable than the repository's first simple RAG version. It included:

- Streamlit and terminal entry points;
- local regex PII masking;
- structured query/category classification;
- classification-enhanced query rewriting;
- MiniLM embeddings and a persisted FAISS index;
- MMR retrieval and document deduplication;
- evidence-only generation prompts;
- Handbook filename, section, and page citations.

An intermediate revision used Azure OpenAI for classification, rewriting, and answer generation. Before that, the original Streamlit prototype used `ChatGroq` with a basic `RetrievalQA` chain and top-k similarity retrieval. The 2026-08-13 Groq migration keeps the newer retrieval and citation improvements instead of reverting to that older architecture.

## Current upgrade integration decision — 2026-08-13

The agentic upgrade uses plain Python orchestration rather than an agent framework. The workflow has only two controlled actions and one deterministic branch, so a framework would add implementation cost without improving interview clarity or reliability.

There is one deliberate deviation from the conceptual target diagram: local PII masking and deterministic emergency detection run before remote Groq intent classification. This ensures an obvious emergency can be escalated even when Groq is unavailable and prevents that request from entering query rewriting, FAISS retrieval, or autonomous answer generation. Normal medical-information requests continue through Groq classification because those labels improve retrieval focus.

## Agentic architecture implementation scope

The requirements in `AGENTS.md` and the implementation plan in `docs/mvp-plan.md` describe a lightweight controlled agent layer:

```text
Observe -> PII process -> Decide risk/action
                           |              |
                           v              v
             search_medical_knowledge   escalate_high_risk_request
                           |              |
                           v              v
                 grounded RAG answer   controlled escalation
                           \              /
                            v            v
                         visible agent trace
```

Implemented work:

- added deterministic emergency/risk routing in a typed shared module;
- exposed exactly two controlled actions;
- prevented classification, retrieval, and autonomous treatment generation for deterministic emergency matches;
- recorded an in-memory JSON-friendly request trace;
- showed selected action, risk, rewritten query, sources, and trace in Streamlit;
- added focused normal-route and high-risk-route tests.

Deferred deliberately: a larger shared-service refactor of the terminal and UI paths. It is not required for the interview vertical slice and would enlarge the diff without improving the demonstrated safety decision.

### 2026-08-13 — Lightweight agentic orchestration added

Reason:

- The interview story needed to demonstrate controlled action selection around—not instead of—the existing RAG pipeline.
- Clear emergencies needed deterministic handling and a visible explanation of why normal generation was skipped.

Changes:

- Added typed intent, risk, action, decision, and trace structures.
- Added deterministic rules for severe chest pain, stroke-like symptoms, severe breathing difficulty, and anaphylaxis-like language.
- Added `search_medical_knowledge` as the controlled wrapper around existing classification, rewrite, FAISS/MMR, generation, and citation behavior.
- Added `escalate_high_risk_request` with a fixed emergency response and no retrieval or treatment recommendation.
- Added visible selected action, risk, rewritten query, and Agent Trace to Streamlit.
- Added deterministic tests proving the escalation path does not call the classifier or RAG tool.

## Known limitations at this revision

- This is a prototype, not a clinical decision system.
- PII masking and classification are simplified and not clinically validated.
- Groq availability, quotas, model lifecycle, and network access are external dependencies.
- The runtime expects MiniLM to have been cached once before offline use.
- The committed FAISS index is loaded with trusted pickle deserialization and must not be replaced with an untrusted index.
- Citation numbers are requested in the prompt but are not yet programmatically validated against every generated claim.
- The JSONL corpus is not included by the current ingestion loader/index.
- There is no production authentication, RBAC, human approval workflow, database, or clinical-system integration.

## Repeatable verification commands

```bash
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover -s tests -v
RUN_LIVE_GROQ_TESTS=1 .venv/bin/python -m unittest tests.test_groq_app.LiveGroqTests -v
.venv/bin/python scripts/healthcheck.py
```

Start the frontend:

```bash
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
.venv/bin/streamlit run optimal/medibot_category_env.py \
  --server.address 127.0.0.1 \
  --server.port 8501
```
