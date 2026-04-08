import numpy as np
import pickle
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INDEX_PATH = "/tmp/rag_store.pkl"
CHUNK_SIZE = 600       # words per chunk
CHUNK_OVERLAP = 80     # word overlap between chunks
MIN_CHUNK_WORDS = 30   # skip tiny chunks


def _load() -> dict:
    if os.path.exists(INDEX_PATH):
        try:
            with open(INDEX_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return {"docs": [], "ids": [], "sources": []}


def _save(store: dict) -> None:
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(store, f)


def _clean_text(text: str) -> str:
    """Remove excessive whitespace and normalize text."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _chunk_text(text: str) -> list[str]:
    """Split text into overlapping word-based chunks."""
    text = _clean_text(text)
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + CHUNK_SIZE
        chunk = " ".join(words[start:end])
        if len(chunk.split()) >= MIN_CHUNK_WORDS:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def index_document(text: str, document_name: str = "unknown") -> int:
    """
    Chunks and indexes a document. Returns number of new chunks indexed.
    Signature: index_document(text, document_name=safe_doc_id)
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

    if indexed > 0:
        _save(store)
    return indexed


def search_documents(query: str, top_k: int = 6) -> dict:
    """
    Searches indexed chunks using TF-IDF cosine similarity.
    Returns: {"documents": [...], "sources": [...]}
    """
    store = _load()
    if not store["docs"]:
        return {"documents": [], "sources": []}

    # Boost query by repeating key terms for better TF-IDF weighting
    expanded_query = f"{query} {query}"

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),   # bigrams improve matching
            max_features=10000,
            sublinear_tf=True,    # log normalization for better scoring
        )
        all_texts = store["docs"] + [expanded_query]
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
        if scores[i] > 0.01:  # filter near-zero matches
            retrieved_docs.append(store["docs"][i])
            src = store["sources"][i] if i < len(store["sources"]) else "unknown"
            retrieved_sources.append(src)

    return {"documents": retrieved_docs, "sources": retrieved_sources}


def reset_collection() -> None:
    """Clears the entire RAG index."""
    if os.path.exists(INDEX_PATH):
        os.remove(INDEX_PATH)
