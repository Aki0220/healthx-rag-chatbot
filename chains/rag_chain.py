from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

def build_rag_chain(llm, history_aware_retriever):
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
            "以下のcontextを使って質問に答えてください。"
            "分からない場合は無理に答えず「分からない」と答えてください。\n\n{context}"
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, qa_chain)