"""
05_rag_answer_groq.py

Reusable helper:
- answer_with_context(context, question) -> string (Groq)

Also runnable as a script.

Requires:
  export GROQ_API_KEY="..."
"""

import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv


def answer_with_context(
    context: str,
    question: str,
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.2,
    max_tokens: int = 300,
) -> str:
    load_dotenv()
    assert os.getenv("GROQ_API_KEY"), "Set GROQ_API_KEY in your environment."
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set. Run: export GROQ_API_KEY='...'\n")

    llm = ChatGroq(model=model, temperature=temperature, max_tokens=max_tokens)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Use CONTEXT only. If missing, say you don't know."),
            ("human", "CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nAnswer:"),
        ]
    )

    chain = prompt | llm
    resp = chain.invoke({"context": context or "", "question": question})
    return resp.content.strip()


def main():
    context = "Paris is the capital of France. It is known for the Eiffel Tower."
    question = "What is the capital of France and what is it known for?"
    print(answer_with_context(context, question))


if __name__ == "__main__":
    main()
