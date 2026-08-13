# AGENTS.md — Novamind Interview Demo

## 1. Mission

This repository is being prepared for a short technical interview demo for an Applied AI / AI Engineer role.

The candidate already has a working Medical RAG project. Do NOT replace it with a new project and do NOT rewrite the architecture from scratch.

The goal is to extend the existing Medical RAG application into a small, reliable, explainable **Agentic Medical RAG demo** that can be run locally on an older Intel Mac.

The demo should prove three things:

1. The existing RAG system can retrieve trusted medical knowledge and produce grounded answers with citations.
2. An agent layer can decide which controlled action/tool to use.
3. High-risk requests can be handled differently and every important step can be traced.

This is an interview MVP, not a production rewrite.

## 2. Critical constraints

### Hardware / environment
- Development machine is an older 2017 Intel Mac.
- Avoid heavy new dependencies.
- Avoid dependencies that require recent Apple Silicon, CUDA, GPU, or a new macOS version.
- Do not upgrade the whole dependency tree unless absolutely necessary.
- Prefer the existing working libraries already present in the repository.
- Prefer Python 3.10 or 3.11 compatible packages if the current project permits them.
- Do not introduce Docker unless the existing environment already uses it and it clearly reduces risk.
- Do not require a cloud deployment.
- Target: localhost demo.

### Time
- The MVP must be achievable in a few hours.
- Prefer a complete, stable vertical slice over extra features.
- Do not build a multi-agent system.
- Do not add unnecessary frameworks merely to make the project look sophisticated.

### Integrity
- Do not claim the system is Novamind/Novie or copy any proprietary product.
- This must remain an independent medical-agent demo based on the candidate's own existing RAG work.
- Do not fabricate metrics, test results, or capabilities.
- If a feature is mocked, label it clearly in code and demo documentation.

## 3. First task: inspect before coding

Before changing any code:

1. Inspect the repository structure.
2. Identify:
   - application entry point(s)
   - current RAG pipeline
   - document loading
   - chunking
   - embedding model
   - vector store / FAISS usage
   - MMR retrieval
   - query rewriting
   - PII masking
   - query/category classification
   - LLM provider(s)
   - citation generation
   - any current UI
   - tests
3. Inspect the current Python environment:
   - macOS version
   - CPU architecture
   - Python version
   - package manager
   - dependency files
4. Run the smallest existing command that proves the current project works.
5. Record all failing dependencies and explain whether each failure is:
   - Python-version related
   - macOS related
   - Intel/ARM architecture related
   - package-version conflict
   - missing configuration/API key
6. Propose the smallest compatibility fix.

DO NOT start a broad refactor before reporting the inspection results.

## 4. Product story

The demo is called:

# Agentic Medical RAG — Controlled Healthcare Assistant

The product problem:

A normal RAG chatbot retrieves information and generates an answer, but enterprise AI systems often need to decide **what action is appropriate**, use only controlled tools, treat risky requests differently, and expose a trace of what happened.

The demo should extend the existing medical RAG system with a lightweight agent/orchestration layer.

The interview story is:

- Iteration 1: basic grounded medical RAG.
- Iteration 2: retrieval quality improvements such as MMR/query rewriting/chunking.
- Iteration 3: privacy and traceability with PII masking and citations.
- Current experiment: add a lightweight agent layer for tool selection, risk handling and traceability.

## 5. Required MVP workflow

```text
User Question
      |
      v
PII / Input Processing
      |
      v
Agent Router / Decision
      |
      +-----------------------+
      |                       |
      v                       v
Medical Knowledge Tool    High-Risk Escalation
(RAG retrieval)           (controlled response)
      |
      v
Retrieved Evidence
      |
      v
Grounded LLM Answer
      |
      v
Citations
      |
      v
Agent Trace / Audit View
```

The system must NOT allow the LLM to execute arbitrary shell commands or arbitrary tools.

## 6. Agent design

Keep the agent simple and explainable.

### Tool 1: `search_medical_knowledge`
Purpose:
- Retrieve relevant trusted medical document chunks.
- Reuse the existing FAISS/MMR/RAG implementation wherever possible.

Input:
- rewritten/search-friendly query

Output:
- retrieved chunks
- scores if available
- document/source metadata

### Tool 2: `escalate_high_risk_request`
Purpose:
- Handle requests that should not be answered as routine autonomous medical guidance.

Examples:
- severe chest pain
- signs of stroke
- severe breathing difficulty
- other clearly high-risk scenarios already supported by project rules

Output:
- controlled escalation response
- reason/category
- no fabricated diagnosis

### Optional Tool 3: `answer_general_information`
Only add this if it already fits the architecture cleanly.

Do not add more tools unless necessary.

## 7. Orchestration requirements

The implementation should make these concepts visible and easy to explain:

1. **Observe** — receive user request
2. **Decide** — classify intent/risk and choose allowed tool
3. **Act** — call the selected controlled tool
4. **Observe result** — receive retrieved evidence or escalation result
5. **Respond** — generate grounded response and attach citations

The routing logic must be inspectable in code.

Do not hide all orchestration inside a giant prompt.

Where practical, enforce deterministic safety/routing rules in application code rather than relying only on prompt instructions.

## 8. Trace / audit requirements

For each request, produce a lightweight trace such as:

```text
Request received
PII processing completed
Intent classified: medical_information
Risk classified: medium
Selected tool: search_medical_knowledge
Query rewritten
4 chunks retrieved
Grounded response generated
3 citations attached
Request completed
```

For a high-risk example:

```text
Request received
Intent classified: medical_advice
Risk classified: high
Selected action: escalate_high_risk_request
Autonomous medical recommendation not produced
Request completed
```

Implementation may use in-memory objects or JSON logs.

Do NOT add a database just for this demo.

## 9. UI requirements

Goal: browser-based localhost demo if technically feasible on the current Mac.

Reuse any existing UI first.

If no usable UI exists, implement the lightest option that works reliably with the current environment.

The UI only needs four conceptual areas:

1. **Question** — text input and submit button
2. **Answer** — generated response
3. **Sources** — citations / retrieved document references
4. **Agent Trace** — classification, selected tool/action, major workflow steps

Optional:
- risk level badge
- latency
- model name

Do NOT spend time on animations, authentication, dashboards, user accounts, charts, or elaborate styling.

A clean plain interface is better than an unstable polished one.

## 10. Demo scenarios

Prepare at least two stable demo scenarios.

### Scenario A — Normal RAG / agent tool use
Use a question that is well supported by the existing medical knowledge base.

Desired behaviour:
- classify as normal/medium-risk information request
- choose RAG search tool
- retrieve evidence
- answer using evidence
- show citations
- show trace

### Scenario B — High-risk handling
Use a clearly high-risk question.

Example:
`I have severe chest pain. What medicine should I take?`

Desired behaviour:
- identify high risk
- do not present autonomous treatment as if it were a diagnosis
- trigger escalation/control path
- trace why the normal RAG answer path was not used

## 11. Retrieval behaviour

Preserve and reuse the current RAG implementation.

When touching retrieval code, keep these concepts explicit because they may be discussed in the interview:

- document loading
- chunking and overlap
- embeddings
- FAISS/vector search
- top-k
- MMR
- query rewriting
- metadata
- citations
- retrieval vs generation

Do not silently replace MMR/FAISS with a completely different stack merely because another library is newer.

If a dependency is impossible to run on the current Mac, propose a minimal fallback and explain the trade-off before changing it.

## 12. Dependency strategy

Priority order:

1. Reuse currently installed/working dependencies.
2. Pin a compatible version of an existing dependency.
3. Replace only the failing component with a lightweight equivalent.
4. Add a new dependency only when it materially improves demo reliability.

Avoid:
- GPU frameworks
- local large language models
- unnecessary vector database servers
- Kubernetes
- Docker unless already working
- multiple agent frameworks
- large frontend frameworks if there is already a usable UI

If the local embedding model is the compatibility blocker, investigate whether:
- the existing index can be reused, or
- a lightweight/API-based embedding fallback can be added behind a clearly named adapter.

Do not silently change retrieval semantics.

## 13. LLM / API behaviour

Reuse an existing configured LLM provider if possible.

Requirements:
- API keys come from environment variables.
- Never commit API keys.
- Add reasonable request timeouts.
- Handle API failure gracefully.
- Log which model/provider is used.
- Keep prompts short and readable.
- Separate system policy/orchestration from user content.

If an external LLM fails during the interview, the app should fail clearly rather than hang indefinitely.

## 14. Testing

Minimum tests/checks:

1. Existing RAG path still works.
2. Normal medical question selects the RAG tool.
3. High-risk question selects escalation path.
4. Trace records selected action.
5. Citations remain present for the normal RAG path.
6. App starts with one documented command.

Run the relevant tests after every meaningful change.

Do not claim "done" until the app has been launched end-to-end locally.

## 15. README/demo documentation

Add a short interview-demo section to README or create `DEMO.md`.

It should include:

### Problem
Why a plain RAG pipeline was extended with agent orchestration.

### Architecture
A small ASCII diagram of the workflow.

### Design choices
Explain:
- why controlled tools
- why RAG
- why MMR if already used
- why risk routing
- why traceability

### Run
Exact commands to start the demo.

### Demo cases
Two copy-paste questions.

### Limitations
Be explicit, for example:
- prototype only
- simplified risk classification
- no real clinical workflow integration
- no production identity/RBAC
- evaluation is limited

### Next steps
Possible production improvements:
- systematic RAG evaluation
- stronger policy engine
- RBAC / identity
- human approval workflow
- observability
- model/cost monitoring

Keep this concise.

## 16. Engineering rules

- Prefer small, readable functions.
- Avoid clever abstractions.
- Preserve existing behaviour unless required by the demo.
- Use type hints where consistent with the repo.
- Add comments only where logic is not obvious.
- Do not perform unrelated refactoring.
- Do not rename large parts of the existing repository.
- Do not delete working functionality.
- Keep the Git diff easy for the candidate to understand before the interview.

Every major new file/function must be explainable by the candidate.

## 17. Definition of done

The MVP is complete when all of the following are true:

- [ ] Existing Medical RAG still works.
- [ ] The app can run locally on the current Intel Mac.
- [ ] A browser UI is available if feasible without destabilising the project.
- [ ] A normal question triggers RAG retrieval.
- [ ] A high-risk question triggers the controlled escalation path.
- [ ] Citations are visible for grounded answers.
- [ ] Agent/tool decision is visible.
- [ ] A trace/audit view is visible.
- [ ] There is one documented start command.
- [ ] Two stable demo prompts are documented.
- [ ] Relevant tests/checks pass.
- [ ] No secrets are committed.
- [ ] The code remains simple enough to explain in an interview.

Once these are satisfied, STOP adding features.

## 18. Required working process for Codex

Work in stages.

### Stage 1 — Inspect
Do not modify code.
Report:
- current architecture
- entry points
- current RAG flow
- environment/dependency risks
- smallest implementation plan

### Stage 2 — Compatibility
Fix only what is required to get the existing application running reliably on this Mac.

### Stage 3 — MVP
Implement:
- agent/router
- controlled tools
- high-risk path
- trace
- minimal UI integration

### Stage 4 — Verify
Run tests and launch the complete demo.

### Stage 5 — Explain
Provide:
- changed files
- architecture explanation
- exact start command
- two demo prompts
- likely interview questions about the implementation
- known limitations

Do not proceed into extra features unless explicitly asked.
