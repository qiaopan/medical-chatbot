# Agentic Medical RAG Interview MVP — Smallest Plan

This plan is based on the Stage 1 repository inspection. It preserves the existing LangChain, MiniLM, FAISS/MMR, Azure OpenAI, citation, and Streamlit design. No agent framework or multi-agent system is proposed.

## Scope guardrails

- Reuse the existing Streamlit UI and committed FAISS index first.
- Keep orchestration as small Python functions and typed dictionaries/dataclasses if useful.
- Expose only two controlled actions: `search_medical_knowledge` and `escalate_high_risk_request`.
- Do not permit arbitrary tools or shell execution.
- Keep high-risk routing deterministic and inspectable in application code; classification output may supplement but must not be the only safety signal.
- Do not add a database, cloud deployment, Docker, authentication, or a general agent framework.
- Stop when the Definition of Done is satisfied.

## Stage 2 — Compatibility only

Goal: prove the existing non-agent RAG application can run on this Intel Mac.

1. Use the existing `chatbot` Conda environment and Python 3.11.13.
2. Resolve direct runtime packages with Intel-compatible versions. Validate wheel availability for FAISS and Torch before installing the frozen requirements; avoid installing unrelated heavy transitive tooling.
3. Confirm or cache `sentence-transformers/all-MiniLM-L6-v2` and load the committed FAISS index with the same embedding model.
4. Configure the existing Azure provider through ignored environment variables, never committed secrets.
5. Make only necessary reliability fixes:
   - lazy classifier configuration/import behaviour;
   - explicit Azure request timeouts and readable errors;
   - repository-root-safe data/index paths;
   - correct stale index-build filename in the UI.
6. Run one handbook-supported question through retrieval, answer generation, and citations in the terminal entry point, then launch Streamlit and repeat it.

Compatibility decision point: if the exact local embedding stack is impossible on macOS Intel, document the observed installation error and propose the smallest adapter/fallback before changing retrieval semantics. Do not rebuild the index as an incidental fix.

## Stage 3 — Minimal agentic vertical slice

Goal: add visible controlled routing around the working RAG path.

### 1. Extract one shared RAG service

Move only the duplicated, already-working orchestration needed by both UI and tests into a small shared module. Preserve the existing functions and semantics for PII masking, classification-enhanced query construction, query rewrite, MMR retrieval, answer generation, and citation formatting.

Avoid a broad architecture cleanup; the extraction exists only to prevent the UI, router, and tests from having different RAG behaviour.

### 2. Add an inspectable router

Implement a small function that returns an explicit decision object containing at least:

- intent/category;
- risk level and reason;
- selected action;
- whether routine RAG answering is allowed.

Use deterministic phrase/rule checks for clear emergencies such as severe chest pain, stroke signs, and severe breathing difficulty. Combine these with the existing classifier risk/category when available. Deterministic emergency checks must run even if Azure classification fails.

### 3. Add exactly two controlled tools/actions

`search_medical_knowledge(query)`:

- reuse the existing query enhancement/rewrite and FAISS MMR retrieval;
- return retrieved chunks plus metadata/citation fields;
- pass evidence to the existing grounded answer generator.

`escalate_high_risk_request(question, reason)`:

- return a fixed, controlled escalation message appropriate to the locally documented emergency context;
- state that it cannot diagnose or recommend medicine autonomously;
- advise urgent professional/emergency help without fabricating a diagnosis;
- do not call retrieval or the answer LLM for routine treatment advice.

Do not add the optional general-information tool unless a concrete demo need appears.

### 4. Add an in-memory request trace

Record a simple ordered list of structured events during one request:

```text
Request received
PII processing completed
Intent classified: ...
Risk classified: ...
Selected action: ...
Query rewritten                 # normal path only
6 chunks retrieved              # normal path only
Grounded response generated     # normal path only
N citations attached            # normal path only
Autonomous recommendation not produced  # high-risk path only
Request completed
```

Do not log raw PII. Keep the trace in memory/session state; no database or external observability dependency.

### 5. Extend the existing Streamlit UI minimally

Retain chat input/history and source expanders. Add:

- selected action/tool;
- risk badge/text;
- an Agent Trace expander or panel.

Rename the title to “Agentic Medical RAG — Controlled Healthcare Assistant.” Avoid styling work beyond clear layout.

## Stage 4 — Verification

Add focused tests using lightweight fakes so routing and trace tests require neither Azure nor model downloads:

1. PII masking executes before externally supplied classifier/RAG functions.
2. A handbook-supported normal question selects `search_medical_knowledge`.
3. `I have severe chest pain. What medicine should I take?` selects `escalate_high_risk_request` even if classifier/API access fails.
4. The escalation path does not call retrieval or answer generation.
5. Each path records its selected action and completion trace.
6. The normal path returns visible citation data.
7. Existing MMR settings remain `k=6`, `fetch_k=20`, `lambda_mult=0.5`.

Then perform real local smoke checks:

- run the focused test command;
- run one terminal RAG request with configured Azure access;
- launch Streamlit with one documented command;
- submit both demo prompts in the browser and verify answer, sources, selected action, risk, and trace;
- verify failure is prompt and readable when Azure access is unavailable.

Do not claim end-to-end success unless the browser flow has actually been launched and exercised.

## Stage 5 — Concise demo documentation

Add a small `DEMO.md` or README interview section only after verification. Include the problem, ASCII architecture, design choices, exact environment/start command, the two verified prompts, limitations, and next steps.

Candidate demo narrative:

1. Existing baseline: grounded Handbook RAG.
2. Retrieval improvements: chunk overlap, query enrichment/rewrite, and MMR diversity.
3. Privacy/explainability: pre-model PII masking and page/section citations.
4. Current extension: deterministic risk-aware controlled action selection and a visible trace.

## Proposed stable demo scenarios

Normal path candidate, already represented in the committed classification sample and likely supported by the Handbook index:

```text
Can a 50-year-old still receive Tdap if they missed the 45-year vaccination?
```

Expected: normal immunisation information route, `search_medical_knowledge`, Handbook evidence, grounded answer, citations, and trace. This must be verified against actual retrieval before being documented as stable.

High-risk path:

```text
I have severe chest pain. What medicine should I take?
```

Expected: deterministic high-risk route, `escalate_high_risk_request`, no autonomous diagnosis/treatment answer, and trace explaining why RAG generation was skipped.

## Expected minimal change surface

Exact filenames should be chosen after Stage 2 proves the runnable environment, but the intended diff is limited to:

- one small shared orchestration/router module;
- small edits to the existing Streamlit entry point;
- focused tests;
- one concise demo document;
- minimal dependency/configuration corrections required by observed Intel/Python failures.

Avoid changing the index builder or rebuilding the FAISS store unless verification shows the normal demo prompt is unsupported. JSONL support is useful but is not required for the smallest Handbook-backed vertical slice.

## Approval gates

- After this Stage 1 report: wait for approval before compatibility changes.
- After any proven Intel embedding/FAISS blocker: explain the concrete failure and fallback trade-off before changing models or retrieval semantics.
- After Stage 2: confirm the existing RAG works before adding routing.
- After Definition of Done: stop adding features.
