import os
import uuid
import base64
import importlib.util
import json
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

HERE = os.path.dirname(os.path.abspath(__file__))


# -----------------------------
# Load snippets 04 + 05 locally
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
# Groq
# -----------------------------
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise RuntimeError('Missing GROQ_API_KEY. Put it in .env or export it.')

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
            max_tokens=650,
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
            max_tokens=950,
        )
        return resp.choices[0].message.content.strip()


class RagAgent:
    def invoke(self, question: str, doc_path: str) -> str:
        db, n_chunks = rag_retrieval.build_vector_db(doc_path)
        context = rag_retrieval.retrieve_context(db, question, k=3)

        if not context.strip():
            return "RAG: No relevant context retrieved from the document."

        try:
            ans = rag_answer.answer_with_context(context, question, model=TEXT_MODEL)
        except TypeError:
            ans = rag_answer.answer_with_context(context, question)

        return f"(Indexed {n_chunks} chunks)\n\n{ans}"


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
            max_tokens=800,
        )
        return resp.choices[0].message.content.strip()


# -----------------------------
# Smart Router (LLM-based)
# -----------------------------
class SmartRouter:
    """
    Uses the LLM to choose one of: vision, rag, code, language
    No hard-coded keywords. We just enforce feasibility:
      - if it picks vision but no image -> fallback
      - if it picks rag but no doc -> fallback
    """

    ROUTER_SYSTEM = (
        "You are a routing controller for a multi-agent assistant.\n"
        "Pick the best agent for the user's request.\n\n"
        "Agents:\n"
        "- vision: questions requiring image understanding\n"
        "- rag: questions that should be answered from the provided document\n"
        "- code: programming, debugging, algorithms, writing code\n"
        "- language: general Q&A, explanations, writing\n\n"
        "Return ONLY valid JSON (no markdown) with keys:\n"
        '{ "agent": "vision|rag|code|language", "reason": "short reason" }\n'
    )

    def route_decision(self, question: str, has_image: bool, has_doc: bool) -> Dict[str, Any]:
        # Keep router call cheap/fast
        resp = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": self.ROUTER_SYSTEM},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "question": question,
                            "has_image": has_image,
                            "has_doc": has_doc,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=120,
        )

        raw = resp.choices[0].message.content.strip()

        # Robust JSON parse: try raw, then try extracting the first {...}
        try:
            return json.loads(raw)
        except Exception:
            try:
                start = raw.find("{")
                end = raw.rfind("}")
                if start != -1 and end != -1 and end > start:
                    return json.loads(raw[start : end + 1])
            except Exception:
                pass

        # Fallback if router output is malformed
        return {"agent": "language", "reason": "router_parse_failed"}

    @staticmethod
    def normalize_agent(agent: str) -> str:
        agent = (agent or "").strip().lower()
        if agent in {"vision", "rag", "code", "language"}:
            return agent
        return "language"


# -----------------------------
# App Router wrapper
# -----------------------------
@dataclass
class Router:
    language: LanguageAgent = LanguageAgent()
    code: CodeAgent = CodeAgent()
    rag: RagAgent = RagAgent()
    vision: VisionAgent = VisionAgent()
    smart: SmartRouter = SmartRouter()

    def route(self, question: str, image_path: Optional[str], doc_path: Optional[str]) -> Tuple[str, str]:
        has_image = bool(image_path)
        has_doc = bool(doc_path)

        decision = self.smart.route_decision(question, has_image=has_image, has_doc=has_doc)
        agent = self.smart.normalize_agent(decision.get("agent"))
        reason = decision.get("reason", "").strip()

        # Enforce feasibility
        if agent == "vision" and not has_image:
            agent = "language"
            reason = (reason + " | no_image_fallback").strip(" |")
        if agent == "rag" and not has_doc:
            agent = "language"
            reason = (reason + " | no_doc_fallback").strip(" |")

        if agent == "vision":
            return self.vision.invoke(question, image_path), f"VisionAgent ({reason})"
        if agent == "rag":
            return self.rag.invoke(question, doc_path), f"RagAgent ({reason})"
        if agent == "code":
            return self.code.invoke(question), f"CodeAgent ({reason})"
        return self.language.invoke(question), f"LanguageAgent ({reason})"


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Mini TOM (All Agents)", layout="wide")
st.title("TOM Multimodal Multi-Agent AI")
st.caption("Auto routes via Groq: Vision / RAG / Code / Language")

router = Router()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "img_path" not in st.session_state:
    st.session_state.img_path = None
if "doc_path" not in st.session_state:
    st.session_state.doc_path = None

with st.sidebar:
    st.subheader("Inputs")

    img_file = st.file_uploader("🖼️ Image", type=["png", "jpg", "jpeg", "webp"])
    if img_file:
        im = Image.open(img_file)
        tmp_img = os.path.join(HERE, f"tmp_{uuid.uuid4().hex}.png")
        im.save(tmp_img)
        st.session_state.img_path = tmp_img
        st.image(im, use_container_width=True)
    else:
        st.session_state.img_path = None

    doc_file = st.file_uploader("📄 PDF/TXT", type=["pdf", "txt"])
    if doc_file:
        ext = os.path.splitext(doc_file.name)[1].lower()
        tmp_doc = os.path.join(HERE, f"tmp_{uuid.uuid4().hex}{ext}")
        with open(tmp_doc, "wb") as f:
            f.write(doc_file.getbuffer())
        st.session_state.doc_path = tmp_doc
        st.success("Document ready for RAG")
    else:
        st.session_state.doc_path = None

    st.markdown("---")
    st.write("Text model:", TEXT_MODEL)
    st.write("Vision model:", VISION_MODEL)

    mode = st.selectbox(
        "Routing mode",
        ["Auto", "Force Language", "Force Code", "Force RAG", "Force Vision"],
        index=0,
    )

# Render history
for role, content, agent in st.session_state.messages:
    with st.chat_message(role):
        if role == "assistant":
            st.markdown(f"**[{agent}]**\n\n{content}")
        else:
            st.markdown(content)

q = st.chat_input("Ask anything...")
if q:
    st.session_state.messages.append(("user", q, ""))

    with st.spinner("Thinking..."):
        try:
            if mode == "Force Vision":
                if not st.session_state.img_path:
                    resp, agent = "Upload an image first to use Vision.", "VisionAgent"
                else:
                    resp = router.vision.invoke(q, st.session_state.img_path)
                    agent = "VisionAgent (forced)"

            elif mode == "Force RAG":
                if not st.session_state.doc_path:
                    resp, agent = "Upload a PDF/TXT first to use RAG.", "RagAgent"
                else:
                    resp = router.rag.invoke(q, st.session_state.doc_path)
                    agent = "RagAgent (forced)"

            elif mode == "Force Code":
                resp = router.code.invoke(q)
                agent = "CodeAgent (forced)"

            elif mode == "Force Language":
                resp = router.language.invoke(q)
                agent = "LanguageAgent (forced)"

            else:
                resp, agent = router.route(q, st.session_state.img_path, st.session_state.doc_path)

        except Exception as e:
            resp, agent = f"Error: {e}", "Error"

    st.session_state.messages.append(("assistant", resp, agent))
    st.rerun()
