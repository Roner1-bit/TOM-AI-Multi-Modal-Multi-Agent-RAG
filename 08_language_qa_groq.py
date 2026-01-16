import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()  # loads GROQ_API_KEY from .env if present


def main() -> None:
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            'Missing GROQ_API_KEY. Add it to a .env file or export it:\n'
            '  echo \'GROQ_API_KEY="gsk_..."\' > .env\n'
            '  # or\n'
            '  export GROQ_API_KEY="gsk_..."\n'
        )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        max_tokens=400,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Be concise."),
            ("human", "{question}"),
        ]
    )

    chain = prompt | llm

    question = "Explain transformers in 5 bullet points."
    resp = chain.invoke({"question": question})
    print(resp.content)


if __name__ == "__main__":
    main()
