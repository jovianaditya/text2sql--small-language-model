import os
import chromadb
import requests
from typing import Any, Dict
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from .config import CHROMA_DIR, COLLECTION_NAME, OLLAMA_URL

_CHROMA_COL = None

def build_vector_store():
    global _CHROMA_COL
    if _CHROMA_COL is not None:
        return _CHROMA_COL

    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    emb_fn = OllamaEmbeddingFunction(
        base_url=OLLAMA_URL,
        model="bge-m3:latest",
    )
    _CHROMA_COL = client.get_or_create_collection(
        COLLECTION_NAME,
        embedding_function=emb_fn,
    )
    return _CHROMA_COL

class OllamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self, base_url: str, model: str = "bge-m3:latest", timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = int(timeout)

    def __call__(self, texts: Documents) -> Embeddings:
        out: Embeddings = []
        for t in texts:
            resp = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": t},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            out.append(resp.json()["embedding"])
        return out

    @classmethod
    def name(cls) -> str:
        return "ollama"

    @classmethod
    def build_from_config(cls, config: dict) -> "OllamaEmbeddingFunction":
        return cls(
            base_url=str(config.get("base_url", "http://localhost:11434")),
            model=str(config.get("model", "bge-m3:latest")),
            timeout=int(config.get("timeout", 60)),
        )

    # new: allow Chroma to serialize EF config
    def get_config(self) -> Dict[str, Any]:
        return {
            "name": self.name(),
            "base_url": self.base_url,
            "model": self.model,
            "timeout": self.timeout,
        }