import os
import faiss
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama

# --- Config ---
DATA_PATH = "eldenring_wiki_full.csv"
INDEX_PATH = "eldenring_index/index.faiss"
EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3"

# --- Load Embedding Model ---
print("🧠 Loading sentence transformer model...")
embedder = SentenceTransformer(EMBED_MODEL)

# --- Load Wiki Data ---
print("📂 Loading Elden Ring wiki data...")
df = pd.read_csv(DATA_PATH)
texts = df["content"].tolist()

# --- Load FAISS Index ---
print("⚙️ Loading FAISS index...")
dim = embedder.get_sentence_embedding_dimension()
index = faiss.read_index(INDEX_PATH)

# --- Start Ollama Model ---
print(f"🤖 Starting Ollama model: {OLLAMA_MODEL}")
llm = Ollama(model=OLLAMA_MODEL)

# --- Retrieval Function ---
def retrieve_context(query, k=3):
    query_vec = embedder.encode([query])
    distances, indices = index.search(np.array(query_vec, dtype="float32"), k)
    context = "\n\n".join([texts[i][:800] for i in indices[0]])
    return context

# --- Chat Function ---
def ask_ranni(query):
    context = retrieve_context(query)
    prompt = f"""You are Ranni the Witch, a wise NPC from Elden Ring.
Use the context below to answer questions concisely and accurately.

Context:
{context}

Question: {query}

Answer:"""
    return llm.invoke(prompt)

print("\n🧝 Elden Ring Chatbot (Ollama Offline) Ready! Type 'exit' to quit.\n")

while True:
    query = input("You: ").strip()
    if query.lower() == "exit":
        print("👋 Farewell, Tarnished.")
        break
    if not query:
        continue
    try:
        answer = ask_ranni(query)
        print(f"Ranni: {answer}\n")
    except Exception as e:
        print(f"⚠️ Error: {e}\n")
