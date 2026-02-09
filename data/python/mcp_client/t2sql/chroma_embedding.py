# ======================================================================
# ========================== chroma_embedding.py =======================
# ======================================================================
# Purpose:
# - Modified build_vector_store to include auto-ingest logic if collection is empty.
# - Added logging to indicate whether auto-ingest was triggered or skipped.
# - Retained existing ingestion logic and Ollama embedding function.

# ======================================================================
# ====================== STANDARD LIBRARY IMPORT =======================
# ======================================================================
import os
import json
import requests
from typing import Any, Dict, List
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
import chromadb
# ======================================================================
# ======================== LOCAL MODULE IMPORT =========================
# ======================================================================
from .config import CHROMA_DIR, COLLECTION_NAME, OLLAMA_URL, EXPORT_JSONL_OUTPUT
_CHROMA_COL = None
# ==============================================================================
#  1. MODIFIED BUILD_VECTOR_STORE (AUTO-INGEST LOGIC)
# ==============================================================================
def build_vector_store():
    """
    Singleton koneksi ChromaDB + Auto-Ingest jika kosong.
    """
    global _CHROMA_COL
    if _CHROMA_COL is not None:
        return _CHROMA_COL

    os.makedirs(CHROMA_DIR, exist_ok=True)
    
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    emb_fn = OllamaEmbeddingFunction(
        base_url=OLLAMA_URL,
        model="snowflake-arctic-embed2:568m",
    )
    
    # 1. Connect / Create Container
    _CHROMA_COL = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"}
    )
    
    # 2. CEK ISI & LOGGING
    current_count = _CHROMA_COL.count()
    
    if current_count == 0:
        print(f"[Chroma] Collection '{COLLECTION_NAME}' is EMPTY.")
        print(f"[Chroma] Triggering AUTO-INGEST from: {EXPORT_JSONL_OUTPUT}")
        
        ingest_dense_data(jsonl_dir=EXPORT_JSONL_OUTPUT)
        
        # Refresh object collection
        _CHROMA_COL = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=emb_fn
        )
    else:
        # --- [INI TAMBAHAN LOGS YANG ANDA MINTA] ---
        print(f"[Chroma] Collection '{COLLECTION_NAME}' ready. Contains {current_count} docs.")
        print(f"[Chroma] No update needed. Skipping auto-ingest.")

    return _CHROMA_COL
# ==============================================================================
# 2. INGESTION LOGIC (SAMA SEPERTI SEBELUMNYA)
# ==============================================================================
def ingest_dense_data(jsonl_dir: str, batch_size: int = 50):
    # Kita panggil client langsung disini agar tidak rekursif loop dengan build_vector_store
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    emb_fn = OllamaEmbeddingFunction(base_url=OLLAMA_URL)
    
    # Gunakan get_or_create_collection
    collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=emb_fn)
    
    print(f"[Chroma] Starting ingestion...")
    
    files = [f for f in os.listdir(jsonl_dir) if f.endswith(".jsonl")]
    total_docs = 0

    for filename in files:
        # FILTER PENTING: Hanya baca _dense.jsonl
        if not filename.endswith("_dense.jsonl"):
            continue
            
        filepath = os.path.join(jsonl_dir, filename)
        print(f"[Chroma] Processing: {filename}...")

        batch_ids = []
        batch_docs = []
        batch_metas = []

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    item = json.loads(line)
                    doc_id = item.get("id")
                    text_content = item.get("text")
                    
                    # --- [FIX START] FLATTEN/SERIALIZE METADATA ---
                    raw_meta = item.get("metadata") or {}
                    clean_meta = {}
                    
                    for k, v in raw_meta.items():
                        # Jika value adalah Dict atau List (seperti 'stats'), ubah jadi JSON String
                        if isinstance(v, (dict, list)):
                            clean_meta[k] = json.dumps(v)
                        # Pastikan value tidak None (Chroma tidak suka None di beberapa versi)
                        elif v is None:
                            clean_meta[k] = "" 
                        else:
                            clean_meta[k] = v
                    # --- [FIX END] ---

                    if doc_id and text_content:
                        batch_ids.append(str(doc_id))
                        batch_docs.append(text_content)
                        batch_metas.append(clean_meta)

                    if len(batch_ids) >= batch_size:
                        collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
                        total_docs += len(batch_ids)
                        batch_ids, batch_docs, batch_metas = [], [], []

                except json.JSONDecodeError: continue

        if batch_ids:
            collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
            total_docs += len(batch_ids)

    print(f"[Chroma] Auto-Ingestion Complete. Indexed {total_docs} docs.")

# ==============================================================================
# EXISTING EMBEDDING CLASS
# ==============================================================================
class OllamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self, base_url: str, model: str = "snowflake-arctic-embed2:568m", timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = int(timeout)

    def __call__(self, texts: Documents) -> Embeddings:
        out: Embeddings = []
        # Batching request ke Ollama bisa lebih cepat jika Ollama support,
        # tapi loop manual per item lebih aman untuk error handling.
        for t in texts:
            try:
                resp = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": t},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                if "embedding" in data:
                    out.append(data["embedding"])
                else:
                    # Fallback jika model gagal/kosong
                    print(f"[Ollama] Warning: No embedding returned for text sample.")
                    out.append([0.0]*1024) # Placeholder (sesuaikan dim model)
            except Exception as e:
                print(f"[Ollama] Error embedding text: {e}")
                out.append([0.0]*1024) # Placeholder pada error
                
        return out

    @classmethod
    def name(cls) -> str:
        return "ollama"

    @classmethod
    def build_from_config(cls, config: dict) -> "OllamaEmbeddingFunction":
        return cls(
            base_url=str(config.get("base_url", "http://localhost:11434")),
            model=str(config.get("model", "snowflake-arctic-embed2:568m")),
            timeout=int(config.get("timeout", 60)),
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": self.name(),
            "base_url": self.base_url,
            "model": self.model,
            "timeout": self.timeout,
        }