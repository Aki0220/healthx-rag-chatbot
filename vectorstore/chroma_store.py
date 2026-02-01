import os
from langchain.vectorstores import Chroma

def get_chroma_db(docs, embeddings, persist_dir):
    if os.path.isdir(persist_dir):
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
    return Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )
