import chromadb
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "financial_docs"

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
client = chromadb.Client()
collection = client.get_or_create_collection(COLLECTION_NAME)


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    if not text or not text.strip():
        return []

    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)

    chunks = []
    start = 0
    text = text.strip()

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        next_start = start + chunk_size - overlap
        if next_start <= start:
            break
        start = next_start

    return chunks


def index_document(text: str, document_name: str) -> int:
    chunks = chunk_text(text)
    if not chunks:
        return 0

    embeddings = embedding_model.encode(chunks).tolist()
    ids = [f"{document_name}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": document_name, "chunk_index": i} for i in range(len(chunks))]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(chunks)


def search_documents(query: str, top_k: int = 5) -> dict:
    if not query or not query.strip():
        return {
            "documents": [],
            "sources": [],
            "metadatas": [],
        }

    query_embedding = embedding_model.encode([query]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    sources = [meta.get("source", "unknown") for meta in metadatas]

    return {
        "documents": documents,
        "sources": sources,
        "metadatas": metadatas,
    }


def reset_collection() -> None:
    global collection
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(COLLECTION_NAME)