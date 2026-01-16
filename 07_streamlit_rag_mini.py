import streamlit as st
import os, uuid, importlib.util

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

HERE = os.path.dirname(os.path.abspath(__file__))

def load_mod(name, filename):
    path = os.path.join(HERE, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod

# reuse snippet 05 (Groq answer with context)
rag_answer = load_mod("rag_answer", "05_rag_answer_groq.py")

st.title("Mini RAG App (Retrieve + Answer)")

if "db" not in st.session_state:
    st.session_state.db = None
if "fp" not in st.session_state:
    st.session_state.fp = None

doc = st.file_uploader("Upload PDF/TXT", type=["pdf", "txt"])
if doc:
    fp = f"{doc.name}:{doc.size}"
    if st.session_state.db is None or st.session_state.fp != fp:
        with st.spinner("Indexing..."):
            ext = os.path.splitext(doc.name)[1].lower()
            tmp = f"tmp_{uuid.uuid4().hex}{ext}"
            with open(tmp, "wb") as f:
                f.write(doc.getbuffer())
            try:
                loader = PyPDFLoader(tmp) if ext == ".pdf" else TextLoader(tmp, encoding="utf-8")
                docs = loader.load()
                chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(docs)
                st.session_state.db = FAISS.from_documents(
                    chunks, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                )
                st.session_state.fp = fp
                st.success(f"Indexed {len(chunks)} chunks")
            finally:
                try: os.remove(tmp)
                except OSError: pass

q = st.text_input("Ask a question")
if q and st.session_state.db:
    hits = st.session_state.db.similarity_search(q, k=3)
    context = "\n\n".join([h.page_content for h in hits]) if hits else ""

    with st.expander("Top retrieved context (debug)"):
        st.write(context[:2000] + ("..." if len(context) > 2000 else ""))

    try:
        with st.spinner("Answering..."):
            answer = rag_answer.answer_with_context(context, q)
        st.subheader("Answer")
        st.write(answer)
    except Exception as e:
        st.error(f"Could not answer (Groq key missing?): {e}")
        st.info('Set: export GROQ_API_KEY="..."')
else:
    st.info("Upload a document first.")
