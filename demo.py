from pathlib import Path
import pickle
import requests

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Optional: check HuggingFace connectivity
try:
    requests.get("https://huggingface.co", timeout=5)
    print("✅ Hugging Face reachable")
except:
    print("⚠️ Warning: Hugging Face not reachable")

DOCS_DIR = Path("docs")
VECTOR_DB_PATH = "faiss_db.pkl"

def main():
    print("🚀 Starting ingestion...")

    docs = []
    for p in DOCS_DIR.glob("*.txt"):
        print("📄 Found file:", p)
        try:
            loaded = TextLoader(str(p), encoding="utf-8").load()
            print(f"   -> Loaded {len(loaded)} documents from {p}")
            docs.extend(loaded)
        except Exception as e:
            print(f"❌ Failed to load {p}: {e}")

    if not docs:
        print("⚠️ No .txt files loaded, stopping.")
        return

    # Split into chunks
    print("✂️ Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} chunks")

    # Create embeddings
    try:
        print("🔎 Loading embeddings model...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("✅ Embeddings ready")
    except Exception as e:
        print("❌ Error creating embeddings:", e)
        return

    # Build FAISS index
    try:
        print("📦 Building FAISS index...")
        vectordb = FAISS.from_documents(chunks, embeddings)
        print("✅ FAISS index created")
    except Exception as e:
        print("❌ Error creating FAISS index:", e)
        return

    # Save FAISS DB locally
    try:
        with open(VECTOR_DB_PATH, "wb") as f:
            pickle.dump(vectordb, f)
        print(f"🎉 FAISS Vector DB created and saved as {VECTOR_DB_PATH}")
    except Exception as e:
        print("❌ Error saving FAISS DB:", e)

if __name__ == "__main__":
    main()