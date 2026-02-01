import os
from .state import RAGState
from .chroma_embedding import build_vector_store
from .config import CHROMA_DIR, COLLECTION_NAME

def ingest_node(state: RAGState) -> RAGState:
    """
    Node Ingestion (Simplified).
    Delegates to Auto-Ingest logic and reports existing stats IMMEDIATELY.
    """
    # 1. Trigger Auto-Ingest (logs dari chroma_embedding akan muncul disini)
    col = build_vector_store()
    
    # 2. Ambil statistik
    total_existing = col.count()
    state["added"] = 0
    state["updated"] = 0
    state["skipped"] = total_existing

    # 3. PRINT REPORT (Langsung cetak di sini agar urutannya di atas)
    #    Kita gunakan format garis yang sama persis dengan request Anda
    line_sep = "─" * 150
    print(f"\n{line_sep}")
    print(f"{'Collection Ingestion Report':^150}")
    print(f"{line_sep}")
    print(f"Added (first run): {state['added']}")
    print(f"Updated (upserted): {state['updated']}")
    print(f"Skipped (unchanged): {state['skipped']}")
    print(f"Total docs now in collection: {total_existing}")
    # Gunakan os.path.abspath agar path terlihat jelas seperti di logs Anda
    abs_chroma = os.path.abspath(CHROMA_DIR)
    print(f"Chroma path: {abs_chroma} | Collection: {COLLECTION_NAME}")
    print(f"{line_sep}\n")

    return state