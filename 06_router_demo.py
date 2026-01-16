"""
06_router_demo.py

Self-contained router demo that *uses files before it*:
- loads 04_rag_retrieval_only.py (build_vector_db, retrieve_context)
- loads 05_rag_answer_groq.py (answer_with_context)

Because filenames start with numbers, we load them via importlib.

Run:
  python 06_router_demo.py

Optional:
  Put sample.pdf or sample.txt in the same folder for RAG test.
  export GROQ_API_KEY="..." for Groq answer.
"""

import os
import re
import importlib.util


HERE = os.path.dirname(os.path.abspath(__file__))


def load_module_from_file(module_name: str, filename: str):
    path = os.path.join(HERE, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {filename} in {HERE}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


# Load "04" and "05"
rag_retrieval = load_module_from_file("rag_retrieval", "04_rag_retrieval_only.py")
rag_answer = load_module_from_file("rag_answer", "05_rag_answer_groq.py")


def is_code_related(text: str) -> bool:
    code_keywords = [
        "function", "class", "def", "import", "print", "variable",
        "python", "java", "sql", "code", "bug", "loop", "error", "traceback"
    ]
    pattern = re.compile(r"\b(?:" + "|".join(re.escape(w) for w in code_keywords) + r")\b", re.IGNORECASE)
    return bool(pattern.search(text or ""))


def language_answer_stub(question: str) -> str:
    # Keep it simple (no LLM dependency) for fast workshop runs
    return f"LanguageAgent (stub): You asked: {question}\n(Demo) Hook this to your LLM if you want.\n"


def code_answer_stub(question: str) -> str:
    return (
        "CodeAgent (stub):\n"
        "```python\n"
        "def hello():\n"
        "    return 'replace stub with real code assistant'\n"
        "```\n"
    )


def route(question: str, image_path=None, doc_path=None):
    # Priority: vision > rag > code > language
    if image_path:
        return ("VisionAgent (stub): Vision needs your project vision assistant.\n"
                f"(Demo) Would analyze: {image_path}\n"), "VisionAgent"

    if doc_path:
        # Build DB and retrieve context using snippet 04
        db, n_chunks = rag_retrieval.build_vector_db(doc_path)
        ctx = rag_retrieval.retrieve_context(db, question, k=3)

        # Answer using snippet 05 (Groq)
        try:
            ans = rag_answer.answer_with_context(ctx, question)
        except Exception as e:
            ans = (
                "RagAgent: Retrieved context OK, but could not call Groq.\n"
                f"Reason: {e}\n\n"
                "TIP: export GROQ_API_KEY='...'\n"
            )

        return f"RagAgent:\nChunks indexed: {n_chunks}\n\n{ans}", "RagAgent"

    if is_code_related(question):
        return code_answer_stub(question), "CodeAgent"

    return language_answer_stub(question), "LanguageAgent"


def main():
    # ---- Demo cases ----
    tests = [
        ("Explain RAG in one paragraph.", None, None),
        ("Write a Python function to reverse a list.", None, None),
        ("What is the main claim of the document?", None, "sample.pdf"),
        ("Describe the image.", "test.jpg", None),
    ]

    for i, (q, img, doc) in enumerate(tests, 1):
        doc_path = None
        if doc and os.path.exists(os.path.join(HERE, doc)):
            doc_path = os.path.join(HERE, doc)
        elif doc:
            # Try txt fallback
            if os.path.exists(os.path.join(HERE, "sample.txt")):
                doc_path = os.path.join(HERE, "sample.txt")

        resp, agent = route(q, image_path=(os.path.join(HERE, img) if img else None), doc_path=doc_path)

        print("\n" + "=" * 70)
        print(f"TEST {i} | ROUTED TO: {agent}")
        print("QUESTION:", q)
        if doc:
            print("DOC:", doc_path if doc_path else "(missing sample.pdf/sample.txt)")
        if img:
            print("IMAGE:", img, "(stub)")
        print("-" * 70)
        print(resp)


if __name__ == "__main__":
    main()
