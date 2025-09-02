# ingest.py
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DOCS_DIR = Path("docs")
PERSIST_DIR = "chroma_db"

def main():
    docs = []
    for p in DOCS_DIR.glob("*.txt"):
        docs.extend(TextLoader(str(p), encoding="utf-8").load())

    if not docs:
        print("⚠️ No .txt files in ./docs folder")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)
    vectordb.persist()
    print("✅ Vector DB created in", PERSIST_DIR)

if __name__ == "__main__":
    main()
