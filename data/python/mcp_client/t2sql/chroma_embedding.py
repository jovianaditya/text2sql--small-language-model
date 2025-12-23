import os
import chromadb

from .config import CHROMA_DIR, COLLECTION_NAME, OLLAMA_BASE_URL
from .llm_utils import OllamaEmbeddingFunction

_CHROMA_COL = None

def build_vector_store():
    global _CHROMA_COL
    if _CHROMA_COL is not None:
        return _CHROMA_COL

    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    emb_fn = OllamaEmbeddingFunction(
        base_url=OLLAMA_BASE_URL,
        model="bge-m3:latest",
    )
    _CHROMA_COL = client.get_or_create_collection(
        COLLECTION_NAME,
        embedding_function=emb_fn,
    )
    return _CHROMA_COL