"""
12_router_end_to_end.py (SELF-CONTAINED)

Covers:
- Language agent (Groq text)
- Code agent (Groq text, code-oriented prompt)
- RAG agent (uses 04 + 05 snippets)
- Vision agent (Groq vision) using:
  meta-llama/llama-4-maverick-17b-128e-instruct

Requirements (in same folder):
- 04_rag_retrieval_only.py
- 05_rag_answer_groq.py
Optional (for demo):
- sample.pdf or sample.txt (RAG)
- test.jpg / test.png (Vision)

Run:
  cd snippets
  python 12_router_end_to_end.py
"""

import os
import re
import base64
import importlib.util
from dataclasses import dataclass
from typing import Optional, Tuple

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

HERE = os.path.dirname(os.path.abspath(__file__))


# -----------------------------
# Helpers: load numeric modules
# -----------------------------
def load_module_from_file(module_name: str, filename: str):
    path = os.path.join(HERE, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {filename} in {HERE}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


rag_retrieval = load_module_from_file("rag_retrieval", "04_rag_retrieval_only.py")
rag_answer = load_module_from_file("rag_answer", "05_rag_answer_groq.py")


# -----------------------------
# Groq client + settings
# -----------------------------
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise RuntimeError('Missing GROQ_API_KEY. Put it in .env or export it.')

# Use Maverick for both text + vision (as requested)
TEXT_MODEL = os.getenv("GROQ_TEXT_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct")
VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct")

client = Groq(api_key=API_KEY)


# -----------------------------
# Agents
# -----------------------------
class LanguageAgent:
    def invoke(self, question: str) -> str:
        resp = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Be concise."},
                {"role": "user", "content": question},
            ],
            temperature=0.2,
            max_tokens=450,
        )
        return resp.choices[0].message.content.strip()


class CodeAgent:
    def invoke(self, question: str) -> str:
        resp = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior software engineer. "
                        "Return code first in a single code block, then a short explanation."
                    ),
                },
                {"role": "user", "content": question},
            ],
            temperature=0.1,
            max_tokens=700,
        )
        return resp.choices[0].message.content.strip()


class RagAgent:
    def invoke(self, question: str, doc_path: str) -> str:
        db, n_chunks = rag_retrieval.build_vector_db(doc_path)
        context = rag_retrieval.retrieve_context(db, question, k=3)

        if not context.strip():
            return "RAG: No relevant context retrieved from the document."

        # Call snippet 05. Some versions accept `model=...`; some don't.
        try:
            answer = rag_answer.answer_with_context(context, question, model=TEXT_MODEL)
        except TypeError:
            answer = rag_answer.answer_with_context(context, question)

        return f"(Indexed {n_chunks} chunks)\n\n{answer}"


class VisionAgent:
    def invoke(self, question: str, image_path: str) -> str:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        ext = os.path.splitext(image_path)[1].lower()
        if ext == ".png":
            mime = "image/png"
        elif ext == ".webp":
            mime = "image/webp"
        else:
            mime = "image/jpeg"

        data_url = f"data:{mime};base64,{b64}"

        try:
            resp = client.chat.completions.create(
                model=VISION_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful vision assistant."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
                temperature=0.2,
                max_tokens=500,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return (
                "VisionAgent error calling model.\n"
                f"Model: {VISION_MODEL}\n"
                f"Error: {e}\n"
            )


# -----------------------------
# Router
# -----------------------------
@dataclass
class Router:
    language: LanguageAgent = LanguageAgent()
    code: CodeAgent = CodeAgent()
    rag: RagAgent = RagAgent()
    vision: VisionAgent = VisionAgent()

    def is_code_related(self, text: str) -> bool:
        kws = [
            "def", "class", "import", "python", "sql", "java", "bug", "error",
            "traceback", "function", "loop", "regex", "api"
        ]
        pattern = re.compile(r"\b(?:" + "|".join(re.escape(w) for w in kws) + r")\b", re.IGNORECASE)
        return bool(pattern.search(text or ""))

    def route(
        self,
        question: str,
        image_path: Optional[str] = None,
        doc_path: Optional[str] = None,
    ) -> Tuple[str, str]:
        # Priority: Vision > RAG > Code > Language
        if image_path and os.path.exists(image_path):
            return self.vision.invoke(question, image_path), "VisionAgent"

        if doc_path and os.path.exists(doc_path):
            return self.rag.invoke(question, doc_path), "RagAgent"

        if self.is_code_related(question):
            return self.code.invoke(question), "CodeAgent"

        return self.language.invoke(question), "LanguageAgent"


# -----------------------------
# Demo runner
# -----------------------------
def pick_existing(*paths: str) -> Optional[str]:
    for p in paths:
        full = os.path.join(HERE, p)
        if os.path.exists(full):
            return full
    return None


def main():
    router = Router()

    doc = pick_existing("sample.pdf", "sample.txt")
    img = pick_existing("sample.jpg", "sample.jpeg", "sample.png", "sample.webp")

    tests = [
        ("Explain RAG in 3 bullet points.", None, None),
        ("Write a Python function to merge two sorted lists.", None, None),
        ("What is the purpose of this document?", None, doc),
        ("What is shown in this image? List 5 details.", img, None),
    ]

    for i, (q, img_path, doc_path) in enumerate(tests, 1):
        resp, agent = router.route(q, image_path=img_path, doc_path=doc_path)

        print("\n" + "=" * 80)
        print(f"TEST {i} | {agent}")
        print("TEXT_MODEL:", TEXT_MODEL)
        print("VISION_MODEL:", VISION_MODEL)
        print("Q:", q)
        if doc_path:
            print("DOC:", os.path.basename(doc_path))
        if img_path:
            print("IMG:", os.path.basename(img_path))
        print("-" * 80)
        print(resp)


if __name__ == "__main__":
    main()
