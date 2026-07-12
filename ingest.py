"""
Document Ingestion Script
=========================
Run this ONCE (or whenever your documents change) to build the ChromaDB vector store.

Usage:
    1. Place your documents in the `documents/` folder
    2. Run: python ingest.py
    3. Start the API: python chatbot_api.py

Supported formats: PDF, DOCX, TXT, MD, CSV, XLSX, XLS, Parquet, JSON

Strategy: Parent-Child Chunking
    - Child chunks (small) are embedded → precise semantic search
    - Parent chunks (large) are stored in docstore.json → rich context for LLM
"""

import os
import io
import json
import logging
import shutil
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from PyPDF2 import PdfReader
from docx import Document
import pandas as pd

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

DOCUMENTS_DIR  = "./documents"
CHROMA_DB_DIR  = "./chroma_db"
DOCSTORE_PATH  = "./docstore.json"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Parent chunks: large → sent to LLM as context
PARENT_CHUNK_SIZE    = 2000
PARENT_CHUNK_OVERLAP = 200

# Child chunks: small → embedded and searched
CHILD_CHUNK_SIZE    = 400
CHILD_CHUNK_OVERLAP = 50

SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.md', '.docx', '.csv', '.xlsx', '.xls', '.parquet', '.json'}


def _readable_value(obj, indent: int = 0) -> str:
    """Convert a JSON value into clean, readable text (no dot-notation paths)."""
    pad = "  " * indent
    lines = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{pad}{k}:")
                lines.append(_readable_value(v, indent + 1))
            else:
                lines.append(f"{pad}{k}: {v}")
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                lines.append(_readable_value(item, indent))
            else:
                lines.append(f"{pad}- {item}")
    else:
        lines.append(f"{pad}{obj}")
    return "\n".join(filter(None, lines))


def _story_to_text(module_id: str, module_name: str, module_desc: str, story: dict) -> str:
    """Convert one ERP user story into a rich, human-readable text block."""
    parts = [
        f"الوحدة: {module_name} ({module_id})",
        f"وصف الوحدة: {module_desc}",
        f"رقم القصة: {story.get('id', '')}",
        f"عنوان القصة: {story.get('title', '')}",
    ]
    if story.get('module'):
        parts.append(f"الوحدة الفرعية: {story['module']}")
    roles = story.get('roles', [])
    if roles:
        roles_str = ', '.join(roles) if isinstance(roles, list) else str(roles)
        parts.append(f"الأدوار: {roles_str}")
    if story.get('goal'):
        parts.append(f"الهدف: {story['goal']}")
    if story.get('description'):
        parts.append(f"الوصف: {story['description']}")
    if story.get('acceptance_criteria'):
        parts.append("معايير القبول:")
        parts.append(_readable_value(story['acceptance_criteria'], indent=1))
    if story.get('notes'):
        parts.append(f"ملاحظات: {story['notes']}")
    return "\n".join(parts)


def _erp_module_to_texts(data: dict) -> list[str]:
    """
    Convert an ERP module JSON (with module_name + stories) into one text block
    per story. This preserves story boundaries so chunking stays within one story.
    """
    module_id   = data.get('module_id', '')
    module_name = data.get('module_name', '')
    module_desc = data.get('description', '')
    stories     = data.get('stories', [])

    texts = []
    for story in stories:
        texts.append(_story_to_text(module_id, module_name, module_desc, story))
    return texts


def _is_erp_module(data) -> bool:
    return isinstance(data, dict) and 'module_name' in data and 'stories' in data


def extract_text(file_path: Path) -> str:
    """Extract plain text from non-JSON files."""
    suffix = file_path.suffix.lower()
    content = file_path.read_bytes()
    text = ""

    if suffix == '.pdf':
        reader = PdfReader(io.BytesIO(content))
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

    elif suffix in ('.txt', '.md'):
        text = content.decode('utf-8', errors='replace') + "\n"

    elif suffix == '.docx':
        doc = Document(io.BytesIO(content))
        for para in doc.paragraphs:
            text += para.text + "\n"

    elif suffix == '.parquet':
        df = pd.read_parquet(io.BytesIO(content))
        text += f"File: {file_path.name}\n"
        text += df.to_string(index=False) + "\n\n"
        text += f"Summary:\n{df.describe().to_string()}\n"

    elif suffix in ('.xlsx', '.xls'):
        df = pd.read_excel(io.BytesIO(content))
        text += f"File: {file_path.name}\n"
        text += df.to_string(index=False) + "\n\n"
        text += f"Summary:\n{df.describe().to_string()}\n"

    elif suffix == '.csv':
        df = pd.read_csv(io.BytesIO(content))
        text += f"File: {file_path.name}\n"
        text += df.to_string(index=False) + "\n\n"
        text += f"Summary:\n{df.describe().to_string()}\n"

    return text


def ingest():
    docs_path = Path(DOCUMENTS_DIR)

    if not docs_path.exists():
        docs_path.mkdir(parents=True)
        logger.info(f"Created '{DOCUMENTS_DIR}/' folder.")
        logger.info("Place your documents there and run this script again.")
        return

    all_files = [f for f in docs_path.iterdir() if f.is_file()]
    files     = [f for f in all_files if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    skipped   = [f.name for f in all_files if f.suffix.lower() not in SUPPORTED_EXTENSIONS]

    if skipped:
        logger.warning(f"Skipping unsupported files: {skipped}")

    if not files:
        logger.error(f"No supported documents found in '{DOCUMENTS_DIR}/'. Aborting.")
        return

    logger.info(f"Found {len(files)} document(s): {[f.name for f in files]}")

    # --- Step 1: Extract texts ---
    # Each entry is a self-contained text unit (one story, one page, etc.)
    all_units  = []   # list of (source_label, text_str)
    successful = []

    for file_path in files:
        logger.info(f"  Extracting: {file_path.name}")
        try:
            if file_path.suffix.lower() == '.json':
                data = json.loads(file_path.read_bytes().decode('utf-8', errors='replace'))
                if _is_erp_module(data):
                    texts = _erp_module_to_texts(data)
                    for i, t in enumerate(texts):
                        if t.strip():
                            all_units.append((f"{file_path.name}[story {i}]", t))
                    logger.info(f"    OK — {len(texts)} stories extracted")
                else:
                    # Generic JSON: readable key-value (no dot-path notation)
                    text = _readable_value(data)
                    if text.strip():
                        all_units.append((file_path.name, text))
                    logger.info(f"    OK — {len(text):,} characters")
            else:
                text = extract_text(file_path)
                if text.strip():
                    all_units.append((file_path.name, text))
                    logger.info(f"    OK — {len(text):,} characters")
                else:
                    logger.warning(f"    No text extracted from {file_path.name} — skipping")
            successful.append(file_path.name)
        except Exception as e:
            logger.error(f"    Failed: {e}")

    if not all_units:
        logger.error("No text could be extracted from any file. Aborting.")
        return

    logger.info(f"  Total text units: {len(all_units)}")

    # --- Step 2: Parent-Child Chunking ---
    # Each unit becomes its own parent document, preserving story boundaries.
    logger.info("Applying Parent-Child Chunking...")

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        length_function=len,
    )

    parent_store = {}   # { parent_id: parent_text }
    child_docs   = []   # LangChain Documents with parent_id in metadata

    parent_id_counter = 0
    for source_label, unit_text in all_units:
        # Each unit is its own parent (preserves story boundaries)
        parent_id = str(parent_id_counter)
        parent_store[parent_id] = unit_text
        parent_id_counter += 1

        children = child_splitter.create_documents(
            [unit_text],
            metadatas=[{"parent_id": parent_id, "source": source_label}]
        )
        child_docs.extend(children)

    logger.info(f"  Created {len(parent_store)} parent units")
    logger.info(f"  Created {len(child_docs)} child chunks (size={CHILD_CHUNK_SIZE})")

    # --- Step 3: Save parent docstore ---
    with open(DOCSTORE_PATH, "w", encoding="utf-8") as f:
        json.dump(parent_store, f, ensure_ascii=False, indent=2)
    logger.info(f"  Saved parent docstore → {DOCSTORE_PATH}")

    # --- Step 4: Load embedding model ---
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    logger.info("(This may take a minute on first run while downloading the model)")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # --- Step 5: Build ChromaDB with child chunks ---
    if os.path.exists(CHROMA_DB_DIR):
        try:
            shutil.rmtree(CHROMA_DB_DIR)
            logger.info(f"Cleared existing ChromaDB at '{CHROMA_DB_DIR}'.")
        except PermissionError:
            logger.error(
                "Cannot delete ChromaDB — the API server is still running and has the files locked.\n"
                "  Stop the server (Ctrl+C) and then re-run ingest.py."
            )
            return

    logger.info(f"Building ChromaDB (child chunks) at '{CHROMA_DB_DIR}'...")
    Chroma.from_documents(
        documents=child_docs,
        embedding=embedding_model,
        persist_directory=CHROMA_DB_DIR,
    )

    logger.info("=" * 50)
    logger.info("Ingestion complete!")
    logger.info(f"  Files processed  : {len(successful)}")
    logger.info(f"  Parent units     : {len(parent_store)}  (stored in docstore.json)")
    logger.info(f"  Child chunks     : {len(child_docs)}  (embedded in ChromaDB)")
    logger.info(f"  Database path    : {os.path.abspath(CHROMA_DB_DIR)}")
    logger.info("You can now start the API: python chatbot_api.py")
    logger.info("=" * 50)


if __name__ == "__main__":
    ingest()
