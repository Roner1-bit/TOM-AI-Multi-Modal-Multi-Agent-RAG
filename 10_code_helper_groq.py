import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()  # loads GROQ_API_KEY from .env if present

def main():
    assert os.getenv("GROQ_API_KEY"), "Set GROQ_API_KEY in your environment."

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, max_tokens=450)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a senior software engineer. Return code first in a single code block, then a short explanation.",
            ),
            ("human", "{task}"),
        ]
    )

    chain = prompt | llm

    task = "Write a Python function that checks if a string is a palindrome, ignoring spaces and punctuation."
    resp = chain.invoke({"task": task})
    print(resp.content)


if __name__ == "__main__":
    main()
