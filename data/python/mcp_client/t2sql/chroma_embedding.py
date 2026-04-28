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
import hashlib
import requests
from typing import Any, Dict, List
from pathlib import Path
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
import chromadb
# ======================================================================
# ======================== LOCAL MODULE IMPORT =========================
# ======================================================================
from .config import CHROMA_DIR, COLLECTION_NAME, OLLAMA_URL, DENSE_METADATA_JSONL
_CHROMA_COL = None
_CHROMA_SOURCE_SIGNATURE = ""


def _resolve_ingest_jsonl_paths() -> List[str]:
    """
    Resolve the explicit dense JSONL input for vector ingestion.
    """
    dense_path = str(DENSE_METADATA_JSONL or "").strip()
    if not dense_path:
        return []
    path_obj = Path(dense_path)
    if path_obj.is_file():
        return [str(path_obj)]
    return []


def _stamp_path() -> Path:
    return Path(CHROMA_DIR) / f".{COLLECTION_NAME}.dense_source_stamp.json"


def _compute_source_signature(paths: List[str]) -> str:
    sig_parts: List[str] = []
    for raw_path in paths:
        try:
            path = Path(raw_path).resolve()
            digest = hashlib.sha256()
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    if chunk:
                        digest.update(chunk)
            sig_parts.append(f"{path}:{digest.hexdigest()}")
        except OSError:
            sig_parts.append(str(raw_path))
    return "||".join(sig_parts)


def _load_source_signature() -> str:
    stamp = _stamp_path()
    if not stamp.is_file():
        return ""
    try:
        payload = json.loads(stamp.read_text(encoding="utf-8"))
    except Exception:
        return ""
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("signature") or "").strip()


def _save_source_signature(signature: str, paths: List[str]) -> None:
    payload = {
        "signature": str(signature or "").strip(),
        "paths": [str(p) for p in paths if str(p).strip()],
    }
    _stamp_path().write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _rebuild_collection(client, emb_fn, ingest_paths: List[str]):
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"}
    )
    if ingest_paths:
        print(f"[STATUS - CHROMA] Rebuilding collection '{COLLECTION_NAME}' from dense metadata:")
        for p in ingest_paths:
            print(f"  - {p}")
        ingest_dense_data(jsonl_paths=ingest_paths)
    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=emb_fn,
    )

# ==============================================================================
#  1. MODIFIED BUILD_VECTOR_STORE (AUTO-INGEST LOGIC)
# ==============================================================================
def build_vector_store():
    """
    Singleton koneksi ChromaDB + Auto-Ingest jika kosong.
    """
    global _CHROMA_COL, _CHROMA_SOURCE_SIGNATURE

    os.makedirs(CHROMA_DIR, exist_ok=True)
    ingest_paths = _resolve_ingest_jsonl_paths()
    source_signature = _compute_source_signature(ingest_paths)
    stored_signature = _load_source_signature()

    if (
        _CHROMA_COL is not None
        and _CHROMA_SOURCE_SIGNATURE == source_signature
        and stored_signature == source_signature
    ):
        return _CHROMA_COL

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

    needs_rebuild = bool(ingest_paths) and stored_signature != source_signature

    if current_count == 0:
        print(f"[STATUS - CHROMA] Collection '{COLLECTION_NAME}' is EMPTY.")
        if not ingest_paths:
            print(
                f"[STATUS - CHROMA] No JSONL files found for ingestion. "
                f"Checked dense metadata path: DENSE_METADATA_JSONL='{DENSE_METADATA_JSONL}'"
            )
        else:
            print(f"[STATUS - CHROMA] Triggering AUTO-INGEST from {len(ingest_paths)} file(s):")
            for p in ingest_paths:
                print(f"  - {p}")
            ingest_dense_data(jsonl_paths=ingest_paths)
            _save_source_signature(source_signature, ingest_paths)

        # Refresh object collection
        _CHROMA_COL = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=emb_fn
        )
    elif needs_rebuild:
        print(
            f"[STATUS - CHROMA] Dense metadata source changed for collection '{COLLECTION_NAME}'. "
            f"Refreshing vector index."
        )
        _CHROMA_COL = _rebuild_collection(client, emb_fn, ingest_paths)
        _save_source_signature(source_signature, ingest_paths)
    else:
        print(f"[STATUS - CHROMA] Collection '{COLLECTION_NAME}' ready. Contains {current_count} docs.")
        print(f"[STATUS - CHROMA] Dense source unchanged. Skipping auto-ingest.")

    _CHROMA_SOURCE_SIGNATURE = source_signature
    return _CHROMA_COL
# ==============================================================================
# 2. INGESTION LOGIC (SAMA SEPERTI SEBELUMNYA)
# ==============================================================================
def ingest_dense_data(jsonl_paths: List[str], batch_size: int = 50):
    # Kita panggil client langsung disini agar tidak rekursif loop dengan build_vector_store
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    emb_fn = OllamaEmbeddingFunction(base_url=OLLAMA_URL)
    
    # Gunakan get_or_create_collection
    collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=emb_fn)
    
    print(f"[STATUS - CHROMA] Starting ingestion...")
    total_docs = 0

    for filepath in jsonl_paths:
        filepath = str(filepath or "").strip()
        if not filepath or not os.path.isfile(filepath):
            continue
        if not filepath.lower().endswith(".jsonl"):
            continue

        filename = os.path.basename(filepath)
        print(f"[STATUS - CHROMA] Processing: {filename}...")

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

    print(f"[STATUS - CHROMA] Auto-Ingestion Complete. Indexed {total_docs} docs.")

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
