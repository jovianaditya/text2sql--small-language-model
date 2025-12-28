import json
from pathlib import Path
from typing import Dict, Any, List

from .config import *
from .state import RAGState
from .chroma_embedding import build_vector_store
from .utils import _meta_table, _meta_column, read_jsonl, _norm_str, _norm_list, _stable_json ,_coerce_json_like, _sanitize_for_compare, _sanitize_metadata, _canon_doc, _canon_meta, _compute_canon_hash

def _print_meta_doc_diff(doc_id: str, old_doc: str, new_doc: str, old_meta: Dict[str, Any], new_meta: Dict[str, Any]):
    if not COMPARE_DEBUG:
        return
    a_doc, b_doc = _canon_doc(old_doc), _canon_doc(new_doc)
    a_meta, b_meta = _canon_meta(old_meta), _canon_meta(new_meta)
    if a_doc == b_doc and a_meta == b_meta:
        return
    print(f"[DIFF] {doc_id}")
    if a_doc != b_doc:
        print("  - documents differ")
    if a_meta != b_meta:
        try:
            om = json.loads(a_meta); nm = json.loads(b_meta)
            keys = sorted(set(om.keys()) | set(nm.keys()))
            for k in keys:
                if om.get(k) != nm.get(k):
                    print(f"  * meta[{k}]: {om.get(k)!r} -> {nm.get(k)!r}")
        except Exception:
            print("  - metadatas differ (unparseable)")

def _ingest_one_json(col, meta: dict, source_name: str, debug_id: str):
    """
    Upsert satu item ke koleksi Chroma.
    Return: (added, updated, skipped)
    """
    # 1) tentukan id
    doc_id = meta.get("field_urn") or meta.get("urn")
    if not doc_id:
        table  = _norm_str(_meta_table(meta))
        column = _norm_str(_meta_column(meta))

        if not table and not column:
            return 0, 0, 1  # skip kalau minim info
        doc_id = f"{Path(source_name).name}:{table}::{column}".lower()

    # 2) normalisasi nilai penting
    aliases = _norm_list(meta.get("tags_alias") or meta.get("aliases") or [])
    roles   = _norm_list(meta.get("tags_role")  or meta.get("roles")   or [])
    table   = _norm_str(_meta_table(meta))
    column  = _norm_str(_meta_column(meta))
    desc    = _norm_str(meta.get("description") or "")

    # 3) dokumen teks STABIL
    new_doc = (
        f"table: {table}\n"
        f"column: {column}\n"
        f"description: {desc}\n"
        f"aliases: {', '.join(aliases)}\n"
        f"roles: {', '.join(roles)}"
    )

    # 4) metadata STABIL (tanpa _source & co), isi kolom kanonik
    meta2 = dict(meta or {})
    meta2["table"]       = table
    meta2["tableName"]   = table or meta2.get("tableName")
    meta2["column"]      = column
    meta2["columnName"]  = column or meta2.get("columnName")
    meta2["description"] = desc
    meta2["tags_alias"]  = aliases
    meta2["aliases"]     = aliases
    meta2["tags_role"]   = roles
    meta2["roles"]       = roles
    # JANGAN menyimpan _source agar tidak flip-flop
    if "_source" in meta2:
        del meta2["_source"]
    new_meta = _sanitize_metadata(meta2)

    # <<< ADD: hitung & tanam _canon_hash supaya run berikutnya bisa early-skip >>>
    canon_hash = _compute_canon_hash(new_doc, new_meta)
    new_meta["_canon_hash"] = canon_hash

    # 5) cek eksistensi
    try:
        got = col.get(ids=[doc_id], include=["documents", "metadatas"])
        exists = bool(got.get("ids"))
    except Exception:
        exists = False

    if not exists:
        col.add(ids=[doc_id], documents=[new_doc], metadatas=[new_meta])
        return 1, 0, 0

    # 6) bandingkan STABIL (tipe-agnostik) + early-skip

    # ambil dok & meta lama dari Chroma
    old_doc  = ((got.get("documents") or [""])[0]) if got else ""
    old_meta = ((got.get("metadatas") or [{}])[0]) if got else {}

    # --- EARLY 1: kalau last_ingested / lastIngested sama persis, langsung skip ---
    new_last = meta.get("last_ingested") or meta.get("lastIngested")
    old_last = old_meta.get("last_ingested") or old_meta.get("lastIngested")
    if new_last and old_last and str(new_last) == str(old_last):
        # ingestion timestamp sama → asumsi snapshot sama → nggak usah update
        return 0, 0, 1

    # --- EARLY 2: jika hash sama, langsung skip ---
    old_hash = str(old_meta.get("_canon_hash") or "").strip()
    if old_hash and old_hash == canon_hash:
        return 0, 0, 1

    # cek diff kanonik (doc + meta) sebagai safety net
    same_doc  = _canon_doc(old_doc)   == _canon_doc(new_doc)
    same_meta = _canon_meta(old_meta) == _canon_meta(new_meta)

    if same_doc and same_meta:
        # kalau hash lama belum ada, update sekali untuk “menanam” hash
        if not old_hash:
            col.update(ids=[doc_id], documents=[new_doc], metadatas=[new_meta])
            return 0, 1, 0
        return 0, 0, 1  # unchanged → skip

    # 7) ada beda → update + (opsional) tampilkan diff
    _print_meta_doc_diff(doc_id, old_doc, new_doc, old_meta, new_meta)
    col.update(ids=[doc_id], documents=[new_doc], metadatas=[new_meta])
    return 0, 1, 0

def ingest_node(state: RAGState) -> RAGState:
    from .retrieval import jsonl_sparse_search
    paths = state["jsonl_paths"]
    col = build_vector_store()

    added = updated = skipped = 0

    for path in paths:
        p = Path(path)
        if not p.exists():
            print(f"JSONL not found: {path}")
            continue

        for line_no, (raw, obj) in enumerate(read_jsonl(path), 1):
            if obj is None:
                try:
                    maybe = json.loads(raw)
                    if isinstance(maybe, list):
                        for i, item in enumerate(maybe):
                            if isinstance(item, dict):
                                _a, _u, _s = _ingest_one_json(
                                    col, item, p.name, f"{path}:{line_no}[{i}]"
                                )
                                added += _a
                                updated += _u
                                skipped += _s
                        continue
                except Exception:
                    pass
                print(f"[INGEST] skip {path}:{line_no} (not dict/list)")
                continue

            _a, _u, _s = _ingest_one_json(col, obj, p.name, f"{path}:{line_no}")
            added += _a
            updated += _u
            skipped += _s

    state["added"] = added
    state["updated"] = updated
    state["skipped"] = skipped
    # NEW: warm up BM25 index sekali berdasarkan jsonl_paths
    try:
        jsonl_sparse_search("warmup", state["jsonl_paths"], max_hits=0)
    except Exception as e:
        if SPARSE_DEBUG:
            print(f"[SPARSE] warmup failed (ignored): {e!r}")
    return state