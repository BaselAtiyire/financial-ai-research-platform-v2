# rag_utils.py — pure sklearn/numpy replacement, no chromadb or faiss needed
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INDEX_PATH = "/tmp/rag_store.pkl"

def _load():
    if os.path.exists(INDEX_PATH):
        with open(INDEX_PATH, "rb") as f:
            return pickle.load(f)
    return {"docs": [], "ids": []}

def _save(store):
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(store, f)

def index_document(doc_id: str, text: str):
    store = _load()
    # Avoid duplicates
    if doc_id not in store["ids"]:
        store["docs"].append(text)
        store["ids"].append(doc_id)
        _save(store)

def search_documents(query: str, n_results: int = 5):
    store = _load()
    if not store["docs"]:
        return []
    
    vectorizer = TfidfVectorizer(stop_words="english")
    # Fit on all docs + query together
    all_texts = store["docs"] + [query]
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    doc_vectors = tfidf_matrix[:-1]
    query_vector = tfidf_matrix[-1]
    
    scores = cosine_similarity(query_vector, doc_vectors)[0]
    top_indices = np.argsort(scores)[::-1][:n_results]
    
    return [store["docs"][i] for i in top_indices if scores[i] > 0]

def reset_collection():
    if os.path.exists(INDEX_PATH):
        os.remove(INDEX_PATH)
