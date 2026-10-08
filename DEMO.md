# Agentic Medical RAG — Interview Demo

## Story

1. Iteration 1: basic grounded medical RAG.
2. Iteration 2: chunk overlap, query enrichment/rewrite, FAISS MMR, top-k selection, and metadata.
3. Iteration 3: local PII masking, grounded citations, and source metadata.
4. Current upgrade: a lightweight deterministic router, two controlled actions, and a visible trace.

The orchestration layer wraps the existing RAG pipeline; it does not replace it.

## Architecture

```text
Question -> PII masking -> deterministic router
                              |             |
                              v             v
                 search_medical_knowledge  escalate_high_risk_request
                    |                         |
          classification -> rewrite           | no normal RAG
                    |                         |
             FAISS MMR -> evidence             |
                    |                         |
          grounded answer -> citations         |
                    +------------+------------+
                                 v
                            Agent Trace
```

Deterministic emergency checks run before remote Groq classification. This makes the safety path available when the model API is unavailable and prevents obvious emergency requests from reaching retrieval or autonomous answer generation.

## Run

```bash
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
.venv/bin/streamlit run optimal/medibot_category_env.py \
  --server.address 127.0.0.1 \
  --server.port 8501
```

Open `http://127.0.0.1:8501`.

## Demo cases

Normal RAG:

```text
Can a 50-year-old still receive Tdap if they missed the 45-year vaccination?
```

High risk:

```text
I have severe chest pain. What medicine should I take?
```

## Verification

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/healthcheck.py
```

## Limitations

- Prototype only; not clinical advice or a validated triage system.
- Emergency rules are intentionally narrow and conservative.
- No production identity, RBAC, clinical workflow, or human approval integration.
- Groq availability and quota remain external dependencies for normal RAG.
- Citation attachment is visible, but claim-level citation correctness is not automatically scored.

## Next steps

- Systematic retrieval and grounded-answer evaluation.
- Versioned clinical policy rules and human review.
- Production identity/RBAC and audit storage.
- Observability, latency, token, and cost monitoring.
