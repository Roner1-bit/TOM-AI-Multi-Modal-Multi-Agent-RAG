"""
04_rag_retrieval_only.py  (OFFLINE / CACHED MODEL - Option B)

Reusable helpers:
- build_vector_db(file_path) -> (FAISS db, num_chunks)
- retrieve_context(db, query, k=3) -> context string

This version forces embeddings to load from the local Hugging Face cache
using snapshot_download(..., local_files_only=True) so it does NOT hit the network.
"""

import os
from typing import Tuple

from huggingface_hub import snapshot_download

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


EMBED_REPO_ID = "sentence-transformers/all-MiniLM-L6-v2"


def _resolve_local_model_path(repo_id: str = EMBED_REPO_ID) -> str:
    """
    Returns a local filesystem path to the cached snapshot for repo_id.
    Will NOT download. If not cached, raises with a clear message.
    """
    try:
        return snapshot_download(repo_id=repo_id, local_files_only=True)
    except Exception as e:
        raise RuntimeError(
            f"Embedding model not found in local cache: {repo_id}\n\n"
            "Fix:\n"
            "1) Temporarily allow internet ONCE and run:\n"
            "   python -c \"from sentence_transformers import SentenceTransformer; "
            "SentenceTransformer('all-MiniLM-L6-v2'); print('cached')\"\n"
            "2) Then rerun this script.\n\n"
            f"Original error: {e}"
        ) from e


def build_vector_db(
    file_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    embed_repo_id: str = EMBED_REPO_ID,
) -> Tuple[FAISS, int]:
    ext = os.path.splitext(file_path)[1].lower()
    loader = PyPDFLoader(file_path) if ext == ".pdf" else TextLoader(file_path, encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)

    local_model_path = _resolve_local_model_path(embed_repo_id)

    # Point embeddings directly to the local snapshot folder
    embeddings = HuggingFaceEmbeddings(model_name=local_model_path)

    db = FAISS.from_documents(chunks, embeddings)
    return db, len(chunks)


def retrieve_context(db: FAISS, query: str, k: int = 3) -> str:
    hits = db.similarity_search(query, k=k)
    return "\n\n".join([h.page_content for h in hits]) if hits else ""


def main():
    FILE = "sample.pdf"  # or sample.txt
    if not os.path.exists(FILE):
        print(f"❌ Missing {FILE}. Put sample.pdf or sample.txt in this folder.")
        return

    db, n_chunks = build_vector_db(FILE)
    print("✅ Indexed chunks:", n_chunks)

    q = "What is the main idea?"
    ctx = retrieve_context(db, q, k=3)

    print("\nQUERY:", q)
    print("\nCONTEXT (top hits):\n", ctx[:1200])


if __name__ == "__main__":
    main()
