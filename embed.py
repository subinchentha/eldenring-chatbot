import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

# Try OpenAI or Fall Back to HF
try:
    from langchain_openai import OpenAIEmbeddings
    openai_available = True
except ImportError:
    openai_available = False

from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embedding_model():
    """Selects the best available embedding model."""
    openai_key = os.getenv("OPENAI_API_KEY", None)
    if openai_available and openai_key:
        try:
            print("⚙️ Using OpenAI embeddings...")
            return OpenAIEmbeddings()
        except Exception as e:
            print(f"⚠️ OpenAI not available ({e}), switching to HuggingFace.")
    print("🧠 Using local HuggingFace embeddings (all-MiniLM-L6-v2).")
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def main():
    # Load your scraped wiki data
    file_name = "eldenring_wiki_full.csv"
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"❌ Could not find {file_name}. Run scraper first.")

    df = pd.read_csv(file_name)
    print(f"📄 Loaded {len(df)} wiki pages.")

    # Create documents
    docs = [Document(page_content=row["content"], metadata={"url": row["url"]}) for _, row in df.iterrows()]

    # Split text into chunks
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"✂️ Split into {len(chunks)} text chunks.")

    # Create embeddings (auto-select)
    embeddings = get_embedding_model()

    # Create FAISS index
    print("📦 Creating FAISS vector index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("eldenring_index")

    print("✅ Embeddings and FAISS index created successfully!")


if __name__ == "__main__":
    main()
