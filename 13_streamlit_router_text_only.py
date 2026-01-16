import os
import re
from dotenv import load_dotenv

import streamlit as st
from groq import Groq

load_dotenv()

# -----------------------------
# Groq client + settings
# -----------------------------
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise RuntimeError('Missing GROQ_API_KEY. Put it in .env or export it.')

TEXT_MODEL = os.getenv("GROQ_TEXT_MODEL", "llama-3.3-70b-versatile")
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
            max_tokens=500,
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


# -----------------------------
# Router (text-only)
# -----------------------------
class Router:
    def __init__(self):
        self.language = LanguageAgent()
        self.code = CodeAgent()

    @staticmethod
    def is_code_related(text: str) -> bool:
        kws = [
            "def", "class", "import", "python", "sql", "java", "javascript",
            "bug", "error", "traceback", "function", "loop", "api", "regex"
        ]
        pattern = re.compile(r"\b(?:" + "|".join(re.escape(w) for w in kws) + r")\b", re.IGNORECASE)
        return bool(pattern.search(text or ""))

    def route(self, question: str):
        if self.is_code_related(question):
            return self.code.invoke(question), "CodeAgent"
        return self.language.invoke(question), "LanguageAgent"


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Mini TOM (Text Router Only)", layout="centered")
st.title("Mini TOM (Text Router Only)")
st.caption("Self-contained: routes between LanguageAgent and CodeAgent using Groq.")

router = Router()

if "msgs" not in st.session_state:
    st.session_state.msgs = []

# render history
for role, content, agent in st.session_state.msgs:
    with st.chat_message(role):
        if role == "assistant":
            st.markdown(f"**[{agent}]**\n\n{content}")
        else:
            st.markdown(content)

q = st.chat_input("Ask anything...")
if q:
    st.session_state.msgs.append(("user", q, ""))

    with st.spinner("Thinking..."):
        resp, agent = router.route(q)

    st.session_state.msgs.append(("assistant", resp, agent))
    st.rerun()
