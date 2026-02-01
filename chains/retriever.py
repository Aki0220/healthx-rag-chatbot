from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever

def create_history_retriever(llm, retriever):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "会話履歴と最新の入力から、単独でも理解できる質問を生成してください。"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    return create_history_aware_retriever(llm, retriever, prompt)