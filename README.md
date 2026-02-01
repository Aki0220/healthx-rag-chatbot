# healthx-rag-chatbot
PDFドキュメントを対象に、LangChainとChromaを用いて会話履歴を考慮したRAG型QAチャットボット


# HealthX RAG Chatbot

PDFドキュメントを対象にした **Retrieval-Augmented Generation (RAG)** チャットボットです。  
ユーザーの質問に対して、PDF内の内容を検索し、根拠に基づいた回答を生成します。

UI には **Streamlit** を使用し、RAGのロジックはUIから分離して設計しています。

---

## Overview

- PDF資料をベクトル化して検索可能にする RAG システム
- Chroma を用いたベクトルデータベースの永続化
- 会話履歴を考慮した history-aware retriever を使用
- Streamlit は UI 層のみに限定

---

## Architecture 


 User
↓
Streamlit UI
↓
RAG Chain
├─ History-aware Retriever
│ └─ Chroma Vector DB
└─ LLM (ChatOpenAI)

healthx-rag-chatbot/
├── app.py # Streamlit UI (entry point)
├── config.py # 各種設定値
├── loaders/
│ └── pdf_loader.py # PDF読み込み処理
├── utils/
│ └── splitter.py # テキスト分割処理
├── vectorstore/
│ └── chroma_store.py # Chroma DB管理
├── chains/
│ ├── retriever.py # Retriever構築
│ └── rag_chain.py # RAG Chain構築
├── data/ # PDF資料
├── db/ # ベクトルDB（Git管理外）
├── .gitignore
└── README.md


## How It Works

1. `data/` フォルダ内の PDF を読み込み
2. テキストを一定サイズに分割
3. OpenAI Embeddings を用いてベクトル化
4. Chroma に保存し、検索可能なベクトルDBを構築
5. ユーザーの質問をもとに関連チャンクを検索
6. 検索結果を context として LLM に渡し回答生成

---


## Vector Database Handling

- `db/` ディレクトリは **ベクトルDBのキャッシュ**です
- GitHub には含めません（`.gitignore` で除外）

## Data Handling 

  data/ ディレクトリには以前スクールで使用した資料を使用しています。（許可取得済み）

## Why This Design

 UIとロジックの分離
 　streamlitは表示と入力のみ担当
 再現性のある検索
 　Embeddingモデルを明示
 実務を想定したDB運用
 　ベクトルはDB永続化
 　必要に応じて再生成可能

# How to Run

  ## 仮想環境作成・有効化
        python -m venv env
        env\Scripts\activate

  ## 依存関係インストール
        pip install -r requirements.txt

  ## アプリ起動
        streamlit run app.py

## Notes
  PDFを更新した場合は db/ を削除すると再生成されます

## Technologies Used

  python
  LangChain
  OpenAI API
  Chroma
  Streamlit