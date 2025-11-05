import os
import faiss
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
import gradio as gr

# --- Config ---
DATA_PATH = "eldenring_wiki_full.csv"
INDEX_PATH = "eldenring_index/index.faiss"
EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3"

# --- Load models and data ---
print("🧠 Loading embeddings and FAISS index...")
embedder = SentenceTransformer(EMBED_MODEL)

df = pd.read_csv(DATA_PATH)
texts = df["content"].tolist()

dim = embedder.get_sentence_embedding_dimension()
index = faiss.read_index(INDEX_PATH)

print(f"🤖 Connecting to Ollama model: {OLLAMA_MODEL}")
llm = Ollama(model=OLLAMA_MODEL)

# --- Retrieval function ---
def retrieve_context(query, k=3):
    q_vec = embedder.encode([query])
    distances, indices = index.search(np.array(q_vec, dtype="float32"), k)
    return "\n\n".join([texts[i][:1000] for i in indices[0]])

# --- Chatbot logic ---
def chat_with_ranni(query, history):
    try:
        context = retrieve_context(query)
        prompt = f"""You are Ranni the Witch, a wise NPC from Elden Ring.
Answer concisely using the context below.

Context:
{context}

Question: {query}

Answer:"""
        response = llm.invoke(prompt)
        return response
    except Exception as e:
        return f"⚠️ Error: {e}"

# --- Gradio UI ---
theme = gr.themes.Soft(primary_hue="violet", secondary_hue="slate")

with gr.Blocks(theme=theme, title="Elden Ring Chatbot (Offline)") as demo:
    gr.Markdown("# 🧝 Elden Ring Chatbot (Offline)")
    gr.Markdown("Talk to **Ranni the Witch** — ask about weapons, bosses, or quests ⚔️")
    
    chatbot = gr.ChatInterface(
        fn=chat_with_ranni,
        title="Ranni the Witch (Offline)",
        examples=[
            "Where can I find Rivers of Blood?",
            "Best talisman for bleed build?",
            "Who is Maliketh?",
            "How to reach Nokron?"
        ]
    )

demo.launch()
