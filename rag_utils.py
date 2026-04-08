# rag_utils.py — FAISS-based replacement for chromadb
import faiss
import numpy as np
import pickle, os, hashlib
from sentence_transformers import SentenceTransformer

MODEL = SentenceTransformer("all-MiniLM-L6-v2")
INDEX_PATH = "/tmp/faiss_index.pkl"

def _load():
    if os.path.exists(INDEX_PATH):
        with open(INDEX_PATH, "rb") as f:
            return pickle.load(f)
    return {"embeddings": [], "docs": [], "ids": []}

def _save(store):
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(store, f)

def index_document(doc_id: str, text: str):
    store = _load()
    emb = MODEL.encode([text])[0].tolist()
    store["embeddings"].append(emb)
    store["docs"].append(text)
    store["ids"].append(doc_id)
    _save(store)

def search_documents(query: str, n_results: int = 5):
    store = _load()
    if not store["embeddings"]:
        return []
    q_emb = MODEL.encode([query]).astype("float32")
    matrix = np.array(store["embeddings"], dtype="float32")
    index = faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)
    D, I = index.search(q_emb, min(n_results, len(store["docs"])))
    return [store["docs"][i] for i in I[0] if i < len(store["docs"])]

def reset_collection():
    if os.path.exists(INDEX_PATH):
        os.remove(INDEX_PATH)
