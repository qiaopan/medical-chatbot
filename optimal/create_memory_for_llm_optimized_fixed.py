import json
import re
import hashlib
import signal
import shutil
from pathlib import Path
from typing import Any, List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "data"
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# If one PDF page/file is too slow, skip it instead of blocking the whole build.
PDF_LOAD_TIMEOUT_SECONDS = 300

# Optional: put known bad/huge file keywords here.
SKIP_FILE_KEYWORDS = [
    "The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND",
]


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("PDF loading timeout")


def stable_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def should_skip_file(file_path: Path) -> bool:
    file_text = str(file_path)
    return any(keyword in file_text for keyword in SKIP_FILE_KEYWORDS)


def add_common_metadata(doc: Document, file_path: Path, data_dir: Path) -> Document:
    """Add directory-level source metadata for better citations."""
    try:
        rel_path = file_path.relative_to(data_dir)
    except ValueError:
        rel_path = file_path

    doc.metadata.update(
        {
            "source": str(file_path),
            "relative_source": str(rel_path),
            "source_dir": str(rel_path.parent),
            "file_name": file_path.name,
            "file_stem": file_path.stem,
            "file_type": file_path.suffix.lower().replace(".", ""),
        }
    )
    return doc


def _normalise_heading_text(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    text = text.strip(" .:/")
    return text


def _title_from_file_name(file_path: Path) -> str:
    stem = file_path.stem.replace("_", " ").replace("-", " ")
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem.title() if stem else "Document"


def _is_immunisation_handbook(file_path: Path) -> bool:
    return "immunisation-handbook" in file_path.name.lower()


def _extract_section_heading_from_text(text: str) -> str:
    """
    Extract a TOC-like heading from PDF page text.

    Examples:
    - 2.1.7 Adult vaccination (aged 18 years and older)
    - 1.2 From personal protection to community (herd) immunity
    - Table 2.4: Funded immunisation for adults

    This is a lightweight prototype method. If a PDF exposes a formal TOC/bookmark,
    a production version should use page-to-TOC mapping instead.
    """
    content = text or ""

    raw_lines = content.splitlines()
    lines = []
    for line in raw_lines:
        line = _normalise_heading_text(line)
        if not line:
            continue
        if "IMMUNISATION HANDBOOK" in line.upper():
            continue
        if len(line) > 180:
            continue
        lines.append(line)

    # Prefer section/subsection headings over table headings when visible.
    section_patterns = [
        r"^\d{1,2}\.\d+(?:\.\d+)*\s+[A-Z][A-Za-z0-9 ,;:()/'’\-–]{5,160}$",
        r"^\d{1,2}\s+[A-Z][A-Za-z0-9 ,;:()/'’\-–]{5,160}$",
    ]

    for pattern in section_patterns:
        for line in lines[:60]:
            if re.match(pattern, line):
                return line

    for line in lines[:80]:
        match = re.match(r"^(Table\s+[A-Za-z0-9.]+[:：]\s*[^.\n]{5,160})$", line)
        if match:
            return _normalise_heading_text(match.group(1))

    # Inline fallback for extracted PDFs where heading line breaks are lost.
    inline_patterns = [
        r"\b(\d{1,2}\.\d+(?:\.\d+)*\s+[A-Z][A-Za-z0-9 ,;:()/'’\-–]{8,160})",
        r"\b(Table\s+[A-Za-z0-9.]+[:：]\s*[A-Z][A-Za-z0-9 ,;:()/'’\-–]{5,160})",
    ]

    for pattern in inline_patterns:
        match = re.search(pattern, content)
        if match:
            return _normalise_heading_text(match.group(1))

    return ""


def add_pdf_heading_metadata(docs: List[Document], file_path: Path) -> List[Document]:
    """
    Add citation headings to every PDF page before chunking so citations can use
    /file/main heading/sub heading p.x.

    For the Immunisation Handbook, the top-level citation title is fixed to the
    user-requested display value:
      1. Immunisation Handbook 2026 V2

    Subheadings are extracted from visible numbered headings/table headings and
    propagated to following pages until a new heading is found.
    """
    if _is_immunisation_handbook(file_path):
        main_heading = "1. Immunisation Handbook 2026 V2"
    else:
        main_heading = _title_from_file_name(file_path)

    last_sub_heading = ""

    for doc in docs:
        current_heading = _extract_section_heading_from_text(doc.page_content)
        if current_heading:
            last_sub_heading = current_heading

        doc.metadata["main_heading"] = main_heading
        doc.metadata["book_title"] = main_heading
        doc.metadata["sub_heading"] = last_sub_heading or "Relevant excerpt"
        doc.metadata["section_heading"] = last_sub_heading or "Relevant excerpt"

    return docs


def flatten_json(data: Any, prefix: str = "") -> List[str]:
    """Convert nested JSON into readable key-path lines."""
    lines = []

    if isinstance(data, dict):
        for key, value in data.items():
            new_prefix = f"{prefix}.{key}" if prefix else str(key)
            lines.extend(flatten_json(value, new_prefix))
    elif isinstance(data, list):
        for index, value in enumerate(data):
            new_prefix = f"{prefix}[{index}]"
            lines.extend(flatten_json(value, new_prefix))
    else:
        lines.append(f"{prefix}: {data}")

    return lines


def load_pdf_with_timeout(file_path: Path, data_dir: Path) -> List[Document]:
    """
    Load a PDF with timeout protection.
    Some medical PDFs can make pypdf.extract_text() hang for a long time.
    This prevents one bad PDF from blocking the whole vector database build.
    """
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(PDF_LOAD_TIMEOUT_SECONDS)

    try:
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
        signal.alarm(0)

        docs = [add_common_metadata(doc, file_path, data_dir) for doc in docs]
        docs = add_pdf_heading_metadata(docs, file_path)
        return docs

    except TimeoutException:
        signal.alarm(0)
        print(f"Skipped slow PDF after {PDF_LOAD_TIMEOUT_SECONDS}s: {file_path}")
        return []

    except KeyboardInterrupt:
        signal.alarm(0)
        raise

    except Exception as e:
        signal.alarm(0)
        print(f"Failed to load PDF {file_path}: {e}")
        return []


def load_json(file_path: Path, data_dir: Path) -> List[Document]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lines = flatten_json(data)
    text = "\n".join(lines)

    doc = Document(
        page_content=text,
        metadata={"json_record_count": len(lines)},
    )
    return [add_common_metadata(doc, file_path, data_dir)]


def load_documents(data_path: str) -> List[Document]:
    documents = []
    data_dir = Path(data_path)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data folder not found: {data_path}")

    supported_files = sorted(list(data_dir.rglob("*.pdf")) + list(data_dir.rglob("*.json")))
    print(f"Found {len(supported_files)} supported files under {data_path}")

    for file_path in supported_files:
        if should_skip_file(file_path):
            print(f"Skipped by keyword rule: {file_path}")
            continue

        try:
            if file_path.suffix.lower() == ".pdf":
                file_docs = load_pdf_with_timeout(file_path, data_dir)
            elif file_path.suffix.lower() == ".json":
                file_docs = load_json(file_path, data_dir)
            else:
                continue

            if not file_docs:
                continue

            documents.extend(file_docs)
            print(f"Loaded {file_path.suffix.upper()}: {file_path} -> {len(file_docs)} document/page(s)")

        except KeyboardInterrupt:
            print("\nInterrupted by user. Stop building vector database.")
            raise

        except Exception as e:
            print(f"Failed to load {file_path}: {e}")

    return documents


def create_chunks(documents: List[Document]) -> List[Document]:
    """
    RAG improvement: larger chunk + overlap for better semantic context.
    800/120 is usually better than 500/50 for medical guideline-style QA.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", "。", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        source = chunk.metadata.get("relative_source", chunk.metadata.get("source", "unknown"))
        chunk.metadata["chunk_id"] = stable_id(f"{source}-{index}-{chunk.page_content[:80]}")
        chunk.metadata["chunk_index"] = index

    return chunks


def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def rebuild_faiss_database(text_chunks: List[Document]):
    db_path = Path(DB_FAISS_PATH)

    if db_path.exists():
        print(f"Removing old FAISS database: {DB_FAISS_PATH}")
        shutil.rmtree(db_path)

    embedding_model = get_embedding_model()
    db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)

    print(f"FAISS vector database saved to: {DB_FAISS_PATH}")


def main():
    documents = load_documents(DATA_PATH)
    print(f"Total loaded documents/pages: {len(documents)}")

    if not documents:
        raise ValueError("No PDF or JSON documents found. Please add files under data/.")

    text_chunks = create_chunks(documents)
    print(f"Total text chunks: {len(text_chunks)}")

    rebuild_faiss_database(text_chunks)


if __name__ == "__main__":
    main()
