import os
from langchain_community.document_loaders import PyMuPDFLoader

def load_pdfs(folder_path: str):
    docs = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(folder_path, file))
            docs.extend(loader.load())
    return docs