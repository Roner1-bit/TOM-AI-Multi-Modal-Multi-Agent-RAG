import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from dotenv import load_dotenv

load_dotenv()  # loads GROQ_API_KEY from .env if present

def main():
    assert os.getenv("GROQ_API_KEY"), "Set GROQ_API_KEY in your environment."

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2, max_tokens=250)

    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are helpful. Use chat history."), MessagesPlaceholder("history"), ("human", "{input}")]
    )

    chain = prompt | llm
    store = InMemoryChatMessageHistory()

    runnable = RunnableWithMessageHistory(
        chain,
        lambda session_id: store,
        input_messages_key="input",
        history_messages_key="history",
    )

    sid = "demo"

    print(runnable.invoke({"input": "My name is Tom."}, config={"configurable": {"session_id": sid}}).content)
    print(runnable.invoke({"input": "What is my name?"}, config={"configurable": {"session_id": sid}}).content)


if __name__ == "__main__":
    main()
