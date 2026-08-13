import os
import re
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_REQUEST_TIMEOUT = float(os.environ.get("GROQ_REQUEST_TIMEOUT", "30"))


def _import_classifier_module():
    """
    Import category/classify_text_to_json.py without relying on a fixed working directory.

    Supported layouts:
    1. same folder as this file
    2. ./category under the same folder
    3. ../category when this file is under optimal/
    """
    current_dir = Path(__file__).resolve().parent
    candidate_files = [
        current_dir / "classify_text_to_json.py",
        current_dir / "category" / "classify_text_to_json.py",
        current_dir.parent / "category" / "classify_text_to_json.py",
    ]

    for file_path in candidate_files:
        if file_path.exists():
            # Some classify_text_to_json.py versions do "from config import ...".
            # Add its folder to sys.path so config.py in the same folder can be found.
            sys.path.insert(0, str(file_path.parent))

            spec = importlib.util.spec_from_file_location(
                "classify_text_to_json_runtime",
                str(file_path),
            )
            if spec is None or spec.loader is None:
                continue

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

    raise ImportError(
        "Cannot find classify_text_to_json.py. "
        "Put it in ./category/ or the same folder as this file."
    )


@st.cache_resource
def get_vectorstore():
    db_path = Path(DB_FAISS_PATH)
    if not db_path.exists():
        raise FileNotFoundError(
            f"FAISS database not found: {DB_FAISS_PATH}. "
            "Run create_memory_for_llm_optimized.py first."
        )

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"local_files_only": True},
    )
    return FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True,
    )


@st.cache_resource
def get_llm():
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("Missing GROQ_API_KEY in .env or environment variables")

    return ChatGroq(
        api_key=groq_api_key,
        model=GROQ_MODEL,
        temperature=0.0,
        max_tokens=900,
        timeout=GROQ_REQUEST_TIMEOUT,
        max_retries=2,
    )


def classify_and_anonymize_user_query(user_query: str) -> Tuple[str, Dict[str, Any]]:
    """
    1. Anonymise incremental user input.
    2. Call classify_text_to_json.py methods.
    3. Return redacted query and classification labels for prompt optimization.
    """
    classifier = _import_classifier_module()

    # Prefer the anonymizer from classify_text_to_json.py if it exists.
    if hasattr(classifier, "anonymize_text"):
        redacted_query, redactions = classifier.anonymize_text(user_query)
    else:
        redacted_query, redactions = user_query, []

    try:
        client = classifier.create_groq_client()
        classification = classifier.classify_text_with_mini(redacted_query, client)
    except Exception as exc:
        classification = {
            "extracted_question": redacted_query,
            "main_category": "Other / Unclear",
            "secondary_category": "classification failed",
            "vaccine_or_disease": ["Unknown"],
            "medicine_or_treatment": ["Unknown"],
            "target_population": ["Unknown"],
            "caller_type": "Unknown",
            "clinical_scenario": ["Other / Unclear"],
            "age_or_life_stage": ["Unknown"],
            "risk_level": "Unknown",
            "source_needed": ["Unknown"],
            "handbook_search_terms": [],
            "needs_rag": True,
            "confidence": "Low",
            "evidence_from_transcript": [],
            "classification_error": str(exc),
        }

    classification.setdefault("pii_redaction", {})
    classification["pii_redaction"].update(
        {
            "enabled": True,
            "redactions_detected": classification.get("pii_redaction", {}).get("redactions_detected", redactions),
            "redacted_input_text": redacted_query,
        }
    )

    return redacted_query, classification


def ensure_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return [str(value).strip()]


def _normalise_query_part(text: str) -> str:
    """Normalise text for de-duplication in enhanced retrieval queries."""
    return re.sub(r"\s+", " ", text.lower().replace("**", " ")).strip()


def build_classification_enhanced_query(redacted_query: str, classification: Dict[str, Any]) -> str:
    """
    Use classification labels to improve retrieval without duplicating the same question.

    We intentionally do not add source labels such as Handbook / PHARMAC here,
    because they are routing labels rather than useful semantic search terms.
    """
    parts: List[str] = []
    seen = set()
    skip_values = {
        "",
        "Unknown",
        "Other / Unclear",
        "classification failed",
        "Low",
        "Medium",
        "High",
        "Handbook",
        "IMAC website",
        "Medsafe",
        "PHARMAC",
    }

    def add(value: Any) -> None:
        if value is None:
            return

        if isinstance(value, list):
            for item in value:
                add(item)
            return

        text = str(value).strip().strip('"')
        if not text or text in skip_values:
            return

        key = _normalise_query_part(text)
        if not key or key in seen:
            return

        seen.add(key)
        parts.append(text)

    add(redacted_query)

    extracted_question = str(classification.get("extracted_question", "")).strip()
    if extracted_question and _normalise_query_part(extracted_question) != _normalise_query_part(redacted_query):
        add(extracted_question)

    add(classification.get("main_category"))
    add(classification.get("secondary_category"))
    add(classification.get("vaccine_or_disease"))
    add(classification.get("medicine_or_treatment"))
    add(classification.get("target_population"))
    add(classification.get("clinical_scenario"))
    add(classification.get("age_or_life_stage"))
    add(classification.get("handbook_search_terms"))

    return " ".join(parts)



def format_classification_for_prompt(classification: Dict[str, Any]) -> str:
    return f"""
Classification labels:
- main_category: {classification.get("main_category", "Unknown")}
- secondary_category: {classification.get("secondary_category", "Unknown")}
- vaccine_or_disease: {ensure_list(classification.get("vaccine_or_disease"))}
- medicine_or_treatment: {ensure_list(classification.get("medicine_or_treatment"))}
- target_population: {ensure_list(classification.get("target_population"))}
- caller_type: {classification.get("caller_type", "Unknown")}
- clinical_scenario: {ensure_list(classification.get("clinical_scenario"))}
- age_or_life_stage: {ensure_list(classification.get("age_or_life_stage"))}
- risk_level: {classification.get("risk_level", "Unknown")}
- source_needed: {ensure_list(classification.get("source_needed"))}
- confidence: {classification.get("confidence", "Unknown")}
"""


def rewrite_query(llm, search_query: str, classification: Dict[str, Any]) -> str:
    classification_text = format_classification_for_prompt(classification)
    prompt = f"""
Rewrite the user's immunisation question into a clear search query for retrieving medical documents.
Use the classification labels to keep the query focused.
Keep important medical terms, symptoms, drug names, disease names, vaccines, age/life-stage, and risk issue.
Do not answer the question.
Return only one rewritten query.

{classification_text}

Search query draft:
{search_query}
"""
    try:
        rewritten = llm.invoke(prompt).content.strip()
        return rewritten or search_query
    except Exception:
        return search_query


def deduplicate_documents(docs: List[Document]) -> List[Document]:
    seen = set()
    unique_docs = []

    for doc in docs:
        meta = doc.metadata or {}
        key = meta.get("chunk_id") or (meta.get("source"), meta.get("page"), doc.page_content[:120])
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    return unique_docs


def retrieve_documents(vectorstore, query: str) -> List[Document]:
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 20,
            "lambda_mult": 0.5,
        },
    )
    return deduplicate_documents(retriever.invoke(query))



def clean_citation_text(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    if not text or text in {".", "./", "Unknown", "Unknown source"}:
        return fallback
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" /")
    return text or fallback


def title_from_file_name(file_name: str) -> str:
    stem = Path(file_name).stem
    text = stem.replace("_", " ").replace("-", " ")
    text = re.sub(r"\bpdf\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text.title() if text else "Document"


def infer_section_titles(doc: Document) -> Tuple[str, str]:
    """
    Return (main_heading, sub_heading) for citation.

    Preferred source:
    - metadata generated during vector DB creation, e.g. main_heading/sub_heading.

    Fallback:
    - infer headings from the chunk text.
    - if headings cannot be found, use a readable document title and "Relevant excerpt".

    Desired display format:
    /file.pdf/main heading/sub heading p.19
    """
    meta = doc.metadata or {}

    main_heading = (
        meta.get("main_heading")
        or meta.get("book_title")
        or meta.get("main_title")
        or meta.get("section_title")
        or meta.get("section")
        or meta.get("chapter_title")
        or meta.get("chapter")
        or meta.get("h1")
    )
    sub_heading = (
        meta.get("sub_heading")
        or meta.get("section_heading")
        or meta.get("sub_title")
        or meta.get("subsection_title")
        or meta.get("subsection")
        or meta.get("heading")
        or meta.get("h2")
    )

    content = doc.page_content or ""
    lines = [re.sub(r"\s+", " ", line).strip() for line in content.splitlines()]
    lines = [line for line in lines if line]

    # Numbered headings such as:
    # 1 General immunisation principles
    # 1.2 From personal protection to community (herd) immunity
    if not main_heading:
        for line in lines[:25]:
            if re.match(r"^\d{1,2}\s+[A-Z][A-Za-z].{5,120}$", line):
                main_heading = line
                break

    if not sub_heading:
        for line in lines[:35]:
            if re.match(r"^\d{1,2}\.\d+(?:\.\d+)*\s+[A-Z][A-Za-z].{5,140}$", line):
                sub_heading = line
                break

    # Table headings are useful when normal headings are not present in the chunk.
    if not sub_heading:
        table_match = re.search(r"\b(Table\s+[A-Za-z0-9.]+:\s*[^.\n]{5,120})", content)
        if table_match:
            sub_heading = table_match.group(1).strip()

    # Inline fallback for text extracted from PDFs where headings may lose line breaks.
    if not sub_heading:
        inline_sub = re.search(
            r"\b(\d{1,2}\.\d+(?:\.\d+)*\s+[A-Z][A-Za-z][A-Za-z0-9 ,;:()/-]{8,120})",
            content,
        )
        if inline_sub:
            sub_heading = inline_sub.group(1).strip()

    file_name = meta.get("file_name") or Path(meta.get("relative_source") or meta.get("source", "")).name
    if not main_heading and file_name.lower().startswith("immunisation-handbook"):
        main_heading = "1. Immunisation Handbook 2026 V2"

    main_heading = clean_citation_text(main_heading, title_from_file_name(file_name))
    sub_heading = clean_citation_text(sub_heading, "Relevant excerpt")

    return main_heading, sub_heading


def make_citation_path(doc: Document) -> str:
    meta = doc.metadata or {}
    source = meta.get("relative_source") or meta.get("source", "Unknown source")
    file_name = meta.get("file_name") or Path(source).name or "Unknown file"

    main_heading, sub_heading = infer_section_titles(doc)

    page = meta.get("page")
    page_text = f"p.{page + 1}" if isinstance(page, int) else "no page"

    return f"/{file_name}/{main_heading}/{sub_heading} {page_text}"


def build_directory_labels(docs: List[Document]) -> Dict[str, str]:
    """
    Give every unique directory a stable, readable label, e.g.
    D1 = datasheet
    D2 = handbook/catch-up
    """
    directory_to_label: Dict[str, str] = {}

    for doc in docs:
        meta = doc.metadata or {}
        source = meta.get("relative_source") or meta.get("source", "Unknown source")
        source_dir = meta.get("source_dir") or str(Path(source).parent)

        if source_dir not in directory_to_label:
            directory_to_label[source_dir] = f"D{len(directory_to_label) + 1}"

    return directory_to_label


def format_doc_for_context(doc: Document, index: int, directory_labels: Dict[str, str]) -> str:
    citation_path = make_citation_path(doc)

    return f"""
[{index}] Citation: {citation_path}
Content:
{doc.page_content}
"""



def format_source_documents(source_documents: List[Document]):
    formatted_sources = []

    for index, doc in enumerate(source_documents, start=1):
        citation_path = make_citation_path(doc)
        excerpt = " ".join((doc.page_content or "").split())

        formatted_sources.append(
            {
                "title": f"[{index}] {citation_path}",
                "details": citation_path,
                "excerpt": excerpt[:700],
            }
        )

    return formatted_sources



def show_sources(sources):
    if not sources:
        return

    st.markdown("**Citations**")
    for source in sources:
        with st.expander(source["title"]):
            st.caption(source["details"])
            st.write(source["excerpt"])


def generate_answer(
    llm,
    redacted_user_query: str,
    docs: List[Document],
    classification: Dict[str, Any],
) -> str:
    directory_labels = build_directory_labels(docs)
    context = "\n\n".join(
        format_doc_for_context(doc, i, directory_labels)
        for i, doc in enumerate(docs, start=1)
    )
    classification_text = format_classification_for_prompt(classification)

    answer_prompt = f"""
You are a careful medical RAG assistant.

Privacy:
- The user input may be anonymised. Do not try to infer or reconstruct redacted personal information.

Task:
- Answer the user's question using only the provided context.
- Use the classification labels to focus the answer and to choose the relevant type of guidance.
- If the context does not contain the answer, say: "I don't know based on the provided documents."
- Do not invent medical facts.
- Do not over-generalise beyond the retrieved context. Avoid broad claims such as "no upper age limit" unless the context explicitly says so.
- When you use information from a source, cite it using square brackets like [1] or [2].
- The matching citation path is shown in each context block after "Citation:".
- Keep the answer clear and concise.
- Important: This is not a substitute for professional medical advice.

{classification_text}

Context:
{context}

User question:
{redacted_user_query}

Answer:
"""
    return llm.invoke(answer_prompt).content.strip()


def show_classification(classification: Dict[str, Any]):
    st.markdown("**Classification labels**")
    st.json(
        {
            "main_category": classification.get("main_category"),
            "secondary_category": classification.get("secondary_category"),
            "vaccine_or_disease": classification.get("vaccine_or_disease"),
            "medicine_or_treatment": classification.get("medicine_or_treatment"),
            "target_population": classification.get("target_population"),
            "clinical_scenario": classification.get("clinical_scenario"),
            "risk_level": classification.get("risk_level"),
            "source_needed": classification.get("source_needed"),
            "confidence": classification.get("confidence"),
            "pii_redaction": classification.get("pii_redaction"),
        }
    )


def main():
    st.title("Ask Chatbot!")

    with st.sidebar:
        st.markdown("### RAG Settings")
        st.write("Retriever: MMR")
        st.write("Top-k: 6")
        st.write(f"Model: Groq {GROQ_MODEL}")
        st.write("Pre-processing: PII redaction + classification agent")
        st.caption("Run create_memory_for_llm_optimized.py after changing files under data/.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            show_sources(message.get("sources"))
            if message.get("classification"):
                with st.expander("Classification labels"):
                    show_classification(message["classification"])

    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            vectorstore = get_vectorstore()
            llm = get_llm()

            redacted_query, classification = classify_and_anonymize_user_query(prompt)
            enhanced_query = build_classification_enhanced_query(redacted_query, classification)
            rewritten_query = rewrite_query(llm, enhanced_query, classification)
            docs = retrieve_documents(vectorstore, rewritten_query)
            result = generate_answer(llm, redacted_query, docs, classification)
            sources = format_source_documents(docs)

            with st.chat_message("assistant"):
                st.markdown(result)
                show_sources(sources)
                with st.expander("Classification labels"):
                    show_classification(classification)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result,
                    "classification": classification,
                    "sources": sources,
                }
            )

        except Exception as e:
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
