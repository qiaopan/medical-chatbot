# Current Architecture — Stage 1 Inspection

Inspection date: 2026-08-11

This document describes the repository as inspected. It does not describe proposed agent features as if they already exist.

## Repository snapshot

The project is a small Python/LangChain medical RAG application focused on New Zealand immunisation guidance. It contains a Streamlit chat UI, a terminal RAG entry point, a separate classifier CLI, an index-building script, one PDF, one JSONL corpus, and a committed FAISS index.

There is no README, package lock file, test suite, agent router, high-risk escalation action, or audit trace implementation in the current repository.

## Actual application entry points

| Entry point | Purpose | Current status |
| --- | --- | --- |
| `optimal/medibot_category_env.py` | Primary browser UI. `main()` renders a Streamlit chat and runs the complete RAG flow. Intended invocation is `streamlit run optimal/medibot_category_env.py`. | Source parses, but cannot start in the available `chatbot` environment because `streamlit` is not installed. |
| `optimal/connect_memory_with_llm_category_env.py` | Terminal equivalent of the RAG flow. `main()` prompts on stdin, prints the answer, citations, and classification labels. | Source parses, but runtime dependencies and Azure configuration are absent. |
| `optimal/create_memory_for_llm_optimized_fixed.py` | Offline ingestion/index build. Loads supported documents, chunks them, embeds them, and replaces the FAISS directory. | Source parses. Not run because dependencies/model cache are absent and rebuilding would modify the committed index. |
| `optimal/category/classify_text_to_json.py` | Standalone Azure OpenAI classification CLI that writes classification JSON. | Source parses. Import/run requires classifier Azure variables immediately and the `openai` package is absent. |

The UI and terminal files duplicate most retrieval, prompt, and citation functions. There is no shared application/service module yet.

## Existing RAG pipeline

The Streamlit path performs these steps synchronously for every submitted question:

```text
raw question
  -> import classifier module
  -> local regex PII masking
  -> Azure OpenAI classification
  -> append classification labels to the search query
  -> Azure OpenAI query rewrite
  -> FAISS MMR retrieval
  -> deduplicate retrieved Documents
  -> construct evidence-only answer prompt
  -> Azure OpenAI grounded answer
  -> format citations and excerpts
  -> render answer, citations, and classification JSON
```

The terminal path is materially the same. The implementation is direct and inspectable, but it always follows the RAG path. `risk_level` and `needs_rag` are generated but do not control execution.

## Document loading

`optimal/create_memory_for_llm_optimized_fixed.py` recursively discovers only `*.pdf` and `*.json` under `data/`.

- PDF: `PyPDFLoader` produces one LangChain `Document` per page. A Unix `SIGALRM` timeout of 300 seconds prevents a slow PDF from blocking the build. Common file metadata and inferred headings are added.
- JSON: the complete JSON object is flattened into key-path lines and represented as one `Document`.
- Metadata includes `source`, `relative_source`, `source_dir`, `file_name`, `file_stem`, and `file_type`.
- The Immunisation Handbook receives a fixed main heading plus heuristically extracted/propagated section headings.

Actual data:

- `data/immunisation-handbook-2026-v2.pdf` (about 11 MB) is supported.
- `data/imac_chunks.jsonl` (about 672 KB) is **not supported by the loader**, because discovery handles `.json` but not `.jsonl`.

Safe inspection of the committed FAISS pickle opcodes shows its source metadata names only `immunisation-handbook-2026-v2.pdf`; the JSONL corpus is therefore not represented in the current index.

## Chunking and overlap

The builder uses `RecursiveCharacterTextSplitter` with:

- `chunk_size=800`
- `chunk_overlap=120`
- separators `\n\n`, `\n`, sentence boundary, ideographic full stop, space, then character fallback

Each chunk receives a deterministic 12-character MD5-derived `chunk_id` and a global `chunk_index`. Page and heading metadata inherited from the source document are retained. The sizes are characters, not model tokens.

## Embedding model

Both ingestion and query-time loading use `sentence-transformers/all-MiniLM-L6-v2` through `langchain_huggingface.HuggingFaceEmbeddings`.

This consistency is required to query the existing index correctly. No local Hugging Face model snapshot was found in the user cache, so a first successful run would need network access to download the model unless it exists elsewhere outside the inspected cache.

## FAISS/vector store usage

- The persisted store is `vectorstore/db_faiss/index.faiss` plus `index.pkl` (about 6.2 MB total).
- Ingestion uses `FAISS.from_documents(...)` followed by `save_local(...)`.
- Both query entry points call `FAISS.load_local(..., allow_dangerous_deserialization=True)`.
- The builder deletes and recreates the entire store when explicitly run.
- Retrieval does not expose similarity scores.

`allow_dangerous_deserialization=True` is needed by this LangChain load path but means the pickle must remain trusted. For this localhost demo, only the repository-owned index should be loaded.

## MMR retrieval

MMR is already implemented and is interview-worthy. Both application paths create a retriever with:

- `search_type="mmr"`
- final `k=6`
- candidate `fetch_k=20`
- `lambda_mult=0.5`

Retrieved documents are then deduplicated by `chunk_id`, falling back to source, page, and a content prefix. This provides a clear relevance/diversity trade-off, although there is no automated retrieval evaluation.

## Query rewriting

Query preparation has two stages:

1. `build_classification_enhanced_query` appends useful classification fields and handbook terms while filtering routing/noise labels and duplicates.
2. `rewrite_query` asks the answer Azure LLM to return one search-oriented immunisation query. On any exception it silently falls back to the enhanced query.

This is a useful, explainable retrieval enhancement. It also adds one network/model call and does not expose whether fallback occurred.

## PII masking

`anonymize_text` in the classifier performs local regex replacement with the fixed mask `**`. It covers common names introduced by phrases, email, phone, address, payment identifiers, credentials, IP/MAC address, and several national identifiers. It returns entity type and character offsets, without retaining the matched secret in its audit data.

Important limitations:

- It is regex-based and explicitly labelled as a prototype, so recall and false-positive behaviour are not clinically validated.
- The classifier module validates all `CLASSIFIER_*` Azure variables at import time. Consequently, when classifier credentials are missing, module import fails before the surrounding classification fallback can run; even local PII masking becomes unavailable.
- The input is redacted before classification, retrieval, and answer generation, which is a sound interview story once the import/configuration issue is fixed.

## Query/category classification

The classifier uses the Azure OpenAI Python client with a large, structured immunisation taxonomy prompt. It requests JSON mode and normalises output against allowed values. Fields include category, scenario, population, caller type, risk, sources, handbook terms, `needs_rag`, confidence, and supporting transcript phrases.

The main app catches errors around client creation/classification and creates a fallback classification, but this does not catch the earlier import-time environment validation noted above. Classification currently informs retrieval and answer prompts only. It is not a deterministic safety router, and a `High` result still proceeds to ordinary RAG answering.

`optimal/category/result_text.json` is a committed sample result, not a test or current runtime proof.

## LLM providers

Only Azure OpenAI is present in current code and requirements:

- Main RAG/rewrite model: LangChain `AzureChatOpenAI`; default deployment `gpt-4o-mini-medical-chatbot`, default API version `2024-10-21`.
- Classifier: OpenAI SDK `AzureOpenAI`; classifier-specific endpoint, key, deployment, and API version.

The Git history and requirements comment indicate Groq was removed in favour of Azure OpenAI. No Groq runtime code remains.

No request timeout is explicitly configured in either Azure client. The two clients use different environment-variable namespaces, which permits separate deployments but increases demo configuration burden. No Azure variables were set in the inspected shell, and ignored `.env` files were not present in the repository working tree.

## Citation generation

Citation support is already substantial:

- Retrieved chunks are numbered and placed in the LLM context with generated citation paths.
- The prompt requires square-bracket citations such as `[1]`.
- Citation paths combine filename, main heading, subheading, and one-based PDF page.
- Heading metadata is preferred; regex inference and readable fallbacks are used when metadata is absent.
- The Streamlit UI shows each retrieved source in an expander with its path and a truncated excerpt; the terminal prints paths.

The application displays all six retrieved chunks as sources whether or not the answer cites every chunk. It does not validate that generated citation numbers exist or that every answer claim is supported.

## Existing UI

The existing UI is Streamlit and is suitable for reuse. It provides:

- chat input and message history;
- answer rendering;
- expandable citation/source excerpts;
- expandable classification JSON including risk and PII information;
- sidebar labels for MMR, top-k, model, and preprocessing.

It does not provide a selected action/tool view or a chronological agent trace. Its title and some sidebar text are generic/stale, and an index-build instruction references a filename that does not exist (`create_memory_for_llm_optimized.py` rather than the committed `_fixed.py` file).

## Existing tests and checks

No test files or test configuration exist. There are no unit, integration, retrieval-quality, safety-routing, citation, or UI smoke tests.

Checks run during Stage 1:

- All four Python source files parsed successfully with `ast.parse` under the shell Python.
- The smallest attempted existing UI command in the Python 3.11 `chatbot` environment failed immediately with `ModuleNotFoundError: No module named 'streamlit'`.
- No end-to-end RAG or API call succeeded, so the application must not currently be described as runnable or verified.

## Environment and dependency risks

Observed machine/environment:

- macOS 13.7.8 (Darwin 22.6.0)
- Intel `x86_64`
- default `python3`: 3.14.3
- project `.python-version`: 3.11
- Conda environment `chatbot`: Python 3.11.13
- package tools found: system `pip`, `uv`, and Conda
- the `chatbot` environment does not contain Streamlit or the core RAG packages

Risk classification:

| Finding | Failure class | Impact |
| --- | --- | --- |
| Default shell Python is 3.14.3 while `numpy==2.3.1` is guarded with `python_version < '3.13'`. | Python-version conflict | Installing/running from the default interpreter cannot satisfy the frozen requirements as written. |
| The intended Python 3.11 Conda environment lacks runtime packages. | Missing environment setup | UI and RAG imports fail before application logic runs. |
| `pyproject.toml` declares no dependencies, references a missing `README.md`, and says only `>=3.11` with no upper bound. | Packaging/configuration defect | `uv`/project installs do not reproduce `requirements.txt`, and Python 3.14 appears acceptable when it is not. |
| `requirements.txt` is a large fully frozen environment including Torch, Transformers, PyArrow, pandas, plotting packages, and other transitive packages. | Package-version/architecture risk | Slow, fragile install for a small demo; many packages are not direct app requirements. |
| `torch==2.7.1` on macOS Intel is a likely wheel-availability blocker and must be validated before installation. | Intel/ARM architecture and package-version risk | Local Hugging Face embeddings may be impossible with this exact pin even though the application only needs CPU inference. |
| `faiss-cpu==1.11.0` wheel availability also needs validation specifically for CPython 3.11/macOS x86_64. | Intel/ARM architecture and package-version risk | Existing index cannot be queried if no compatible wheel exists. |
| MiniLM is not present in the inspected Hugging Face cache. | Missing model/network dependency | First retrieval startup may require a download and can be unreliable during an interview. |
| Root and classifier Azure variables are absent. | Missing configuration/API key | Classification, query rewrite, and answer generation cannot run. |
| No explicit Azure request timeout is set. | Runtime reliability defect | A network/API failure can hang longer than acceptable during a demo. |
| Classifier configuration is required at import time. | Code/configuration defect | Missing classifier configuration prevents even the intended fallback and local redaction. |
| The loader ignores the committed JSONL file. | Data-format defect | IMAC content is unavailable to retrieval. |
| Relative data/index paths assume launch from repository root. | Runtime/path defect | Starting from another working directory breaks resource discovery. |

No dependency was installed or upgraded during inspection. Claims about Intel wheel availability remain risks to validate in Stage 2, not fabricated test results.

## What already works at source/design level

- The Python files are syntactically valid.
- A committed FAISS index and its source PDF exist.
- PDF ingestion, metadata enrichment, chunking, MiniLM embedding, and FAISS persistence form a coherent pipeline.
- Query preprocessing, MMR configuration, document deduplication, evidence-only prompting, and citation formatting are implemented.
- Streamlit already covers question, answer, sources, history, and classification display.
- Local regex PII masking is implemented.

These are source-level findings. Runtime operation is not yet verified.

## Interview-worthy parts

- Explicit retrieval-versus-generation separation.
- MMR with visible `k`, `fetch_k`, and diversity setting.
- Classification-enhanced and LLM-rewritten retrieval queries with fallback.
- PII masking before external model calls.
- Rich source metadata and page/section citation paths.
- Grounding prompt that requires context-only answers and an explicit insufficient-evidence response.
- Existing lightweight Streamlit UI that can be extended without a frontend rewrite.

## Broken, obsolete, or incomplete parts

- Neither available Python environment can currently run the app as configured.
- Default Python 3.14 conflicts with the dependency markers; Python 3.11 is the intended path.
- The environment is not reproducible from `pyproject.toml`.
- Exact frozen Torch/FAISS versions may not support this Intel Mac.
- Required Azure configuration is missing, and classifier import fails eagerly.
- JSONL data is silently omitted from ingestion and the existing index.
- Risk classification does not alter behaviour; high-risk requests still reach normal RAG generation.
- There is no controlled-tool abstraction, trace, deterministic high-risk rule, or test suite.
- There are duplicated RAG implementations and stale filenames/UI labels.
- API timeouts and citation validation are absent.

## Smallest compatibility fixes to validate in Stage 2

1. Standardise documented execution on the existing `chatbot` Conda environment with Python 3.11; do not use system Python 3.14.
2. Determine the smallest set of direct dependencies and select Intel-compatible pins, especially for Torch/sentence-transformers and FAISS, without upgrading the whole stack.
3. Prefer reusing the committed FAISS index. Confirm that the exact MiniLM model and a compatible FAISS build can load it before rebuilding anything.
4. Pre-download/cache MiniLM once compatibility is confirmed, or add a clearly named embedding adapter only if local MiniLM is proven impossible. Do not silently change embedding semantics.
5. Make classifier configuration lazy so missing classifier credentials can produce an explicit fallback without disabling local PII masking.
6. Add explicit request timeouts and clear Azure configuration errors.
7. Defer JSONL ingestion/index rebuild unless the selected normal demo scenario is not supported by the Handbook index; adding JSONL support changes indexed content and should be deliberate.

Stage 2 should stop once the existing RAG path starts reliably. Agent features belong to Stage 3 only.
