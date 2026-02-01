from langchain.text_splitter import CharacterTextSplitter

def split_documents(docs, chunk_size, chunk_overlap):
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n",
    )
    return splitter.split_documents(docs)