import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INDEX_PATH = "/tmp/rag_store.pkl"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def _load():
    if os.path.exists(INDEX_PATH):
        with open(INDEX_PATH, "rb") as f:
            return pickle.load(f)
    return {"docs": [], "ids": [], "sources": []}


def _save(store):
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(store, f)


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def index_document(text: str, document_name: str = "unknown") -> int:
    """
    Chunks and indexes a document. Returns number of chunks indexed.
    Matches app.py call: index_document(doc_text, document_name=safe_doc_id)
    """
    store = _load()
    chunks = _chunk_text(text)
    indexed = 0
    for i, chunk in enumerate(chunks):
        chunk_id = f"{document_name}_chunk_{i}"
        if chunk_id not in store["ids"]:
            store["docs"].append(chunk)
            store["ids"].append(chunk_id)
            store["sources"].append(document_name)
            indexed += 1
    _save(store)
    return indexed


def search_documents(query: str, top_k: int = 6) -> dict:
    """
    Searches indexed chunks. Returns dict with 'documents' and 'sources'.
    Matches app.py call: search_documents(question, top_k=6)
    """
    store = _load()
    if not store["docs"]:
        return {"documents": [], "sources": []}

    vectorizer = TfidfVectorizer(stop_words="english")
    all_texts = store["docs"] + [query]

    try:
        tfidf_matrix = vectorizer.fit_transform(all_texts)
    except ValueError:
        return {"documents": [], "sources": []}

    doc_vectors = tfidf_matrix[:-1]
    query_vector = tfidf_matrix[-1]

    scores = cosine_similarity(query_vector, doc_vectors)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]

    retrieved_docs = []
    retrieved_sources = []
    for i in top_indices:
        if scores[i] > 0:
            retrieved_docs.append(store["docs"][i])
            src = store["sources"][i] if i < len(store["sources"]) else "unknown"
            retrieved_sources.append(src)

    return {"documents": retrieved_docs, "sources": retrieved_sources}


def reset_collection():
    """Clears the RAG index."""
    if os.path.exists(INDEX_PATH):
        os.remove(INDEX_PATH)
