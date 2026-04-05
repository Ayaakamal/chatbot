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


def _flatten_json(obj, prefix="") -> str:
    """Recursively flatten a JSON object into readable key: value lines."""
    lines = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            label = f"{prefix}.{k}" if prefix else k
            lines.append(_flatten_json(v, label))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            lines.append(_flatten_json(item, f"{prefix}[{i}]"))
    else:
        lines.append(f"{prefix}: {obj}")
    return "\n".join(lines)


def extract_text(file_path: Path) -> str:
    """Extract plain text from a file based on its extension."""
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

    elif suffix == '.json':
        data = json.loads(content.decode('utf-8', errors='replace'))
        text += _flatten_json(data) + "\n"

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

    # --- Step 1: Extract text ---
    all_text   = ""
    successful = []
    for file_path in files:
        logger.info(f"  Extracting: {file_path.name}")
        try:
            text = extract_text(file_path)
            if text.strip():
                all_text += f"\n\n--- {file_path.name} ---\n\n" + text
                successful.append(file_path.name)
                logger.info(f"    OK — {len(text):,} characters")
            else:
                logger.warning(f"    No text extracted from {file_path.name} — skipping")
        except Exception as e:
            logger.error(f"    Failed: {e}")

    if not all_text.strip():
        logger.error("No text could be extracted from any file. Aborting.")
        return

    # --- Step 2: Parent-Child Chunking ---
    logger.info("Applying Parent-Child Chunking...")

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        length_function=len,
    )

    parent_docs = parent_splitter.create_documents([all_text])
    logger.info(f"  Created {len(parent_docs)} parent chunks (size={PARENT_CHUNK_SIZE})")

    parent_store = {}   # { parent_id: parent_text }
    child_docs   = []   # LangChain Documents with parent_id in metadata

    for i, parent in enumerate(parent_docs):
        parent_id = str(i)
        parent_store[parent_id] = parent.page_content

        children = child_splitter.create_documents(
            [parent.page_content],
            metadatas=[{"parent_id": parent_id}]
        )
        child_docs.extend(children)

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
    logger.info(f"  Parent chunks    : {len(parent_docs)}  (stored in docstore.json)")
    logger.info(f"  Child chunks     : {len(child_docs)}  (embedded in ChromaDB)")
    logger.info(f"  Database path    : {os.path.abspath(CHROMA_DB_DIR)}")
    logger.info("You can now start the API: python chatbot_api.py")
    logger.info("=" * 50)


if __name__ == "__main__":
    ingest()
