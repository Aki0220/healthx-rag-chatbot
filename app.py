from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import HumanMessage, AIMessage

from config import *
from loaders.pdf_loader import load_pdfs
from utils.splitter import split_documents
from vectorstore.chroma_store import get_chroma_db
from chains.retriever import create_history_retriever
from chains.rag_chain import build_rag_chain

# =====================
# 🔴 Session State 初期化（最優先）
# =====================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

# =====================
# 画面タイトル
# =====================
st.title("HealthX RAG Chatbot")

# =====================
# RAG 初期化（1回だけ）
# =====================
if st.session_state.rag_chain is None:
    with st.spinner("資料を読み込み中..."):
        docs = load_pdfs(DATA_DIR)
        splitted_docs = split_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP)

        # 🔍 デバッグ表示
        st.write("📄 読み込んだPDFページ数:", len(docs))
        st.write("✂️ 分割チャンク数:", len(splitted_docs))

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

        db = get_chroma_db(splitted_docs, embeddings, DB_DIR)

        retriever = db.as_retriever(
            search_kwargs={"k": 6}
        )
        st.session_state.retriever = retriever

        llm = ChatOpenAI(
            model_name=MODEL_NAME,
            temperature=TEMPERATURE
        )

        history_retriever = create_history_retriever(llm, retriever)
        st.session_state.rag_chain = build_rag_chain(
            llm,
            history_retriever
        )

# =====================
# チャット履歴表示
# =====================
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.write("👤", msg.content)
    elif isinstance(msg, AIMessage):
        st.write("🤖", msg.content)

# =====================
# 入力フォーム
# =====================
query = st.text_input("質問を入力してください")

if st.button("送信") and query:
    # 🔍 検索結果確認
    docs_found = st.session_state.retriever.get_relevant_documents(query)
    st.write("🔍 検索ヒット数:", len(docs_found))

    if docs_found:
        st.write("📄 最初のヒット内容（抜粋）:")
        st.write(docs_found[0].page_content[:300])

    # 🤖 RAG 実行
    result = st.session_state.rag_chain.invoke({
        "input": query,
        "chat_history": st.session_state.chat_history
    })

    answer = str(result["answer"])

    st.session_state.chat_history.extend([
        HumanMessage(content=query),
        AIMessage(content=answer)
    ])

    st.rerun()