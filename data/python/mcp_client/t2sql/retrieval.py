# ======================================================================
# ============================ retrieval.py ============================
# ======================================================================
# Purpose:
# - Implementation of retrieval logic for T2SQL RAG system.
# - Includes BM25 sparse retrieval and dense retrieval, scoring, and context building.
# - Functions to build allowed blocks and cards for LLM context.

# ======================================================================
# ====================== STANDARD LIBRARY IMPORT =======================
# ======================================================================
import os
import re
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import defaultdict
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
# ======================================================================
# ======================== LOCAL MODULE IMPORT =========================
# ======================================================================
from .config import (
    TOP_K,
    SPARSE_BONUS,
    NAME_QUERY_BONUS,
    ID_QUERY_BONUS,
    MAX_CTX_DOCS,
    SPARSE_MAX_HITS,
    SPARSE_DEBUG,
    RETRIEVE_DEBUG,
)
from .state import RAGState
from .chroma_embedding import build_vector_store
from .utils import _print_banner, _meta_table, _meta_column, read_jsonl, _parse_str_or_list_lower, _extract_roles_aliases, _extract_aliases_for_map
from .helpers_main import _is_productline_lookup, _is_simple_orders_lookup, _wants_customer_name_only, _get_qtype, build_dynamic_probe
from .schema_analyzer import get_all_tables, get_table_synonyms, get_singular_form, find_table_by_keyword, get_table_type, is_transaction_table
# ======================================================================
# ========================== CONTEXT BUILDER ===========================
# ======================================================================
def _build_allowed_block(filtered_allowed: Dict[str, List[str]]) -> str:
    """
    Build blok ### ALLOWED IDENTIFIERS dari dict table -> list kolom.
    """
    lines = []
    for t, cols in filtered_allowed.items():
        uniq = sorted(set(cols))
        lines.append(f"- {t}: {', '.join(uniq)}")
    body = "\n".join(lines) if lines else "- (none)"
    return "### ALLOWED IDENTIFIERS\n" + body
# ==============================================================================
# 1. HELPER: FORMAT STATISTICS
# ==============================================================================
def _format_stats_for_llm(stats_raw: Any) -> str:
    """
    Format metadata stats menjadi string ringkas untuk LLM.
    Disesuaikan dengan format JSONL Anda (sample_value singular).
    """
    if not stats_raw:
        return ""

    # Normalisasi input (Dict atau JSON String)
    stats = {}
    if isinstance(stats_raw, str):
        try:
            stats = json.loads(stats_raw)
        except:
            return "" 
    elif isinstance(stats_raw, dict):
        stats = stats_raw
    else:
        return ""

    parts = []

    # A. SAMPLE VALUES
    # Di data Anda, key-nya adalah "sample_value" (tunggal)
    # Kita juga cek "all_sample_values" (list) jaga-jaga kalau sparse script update lagi
    val_list = stats.get("all_sample_values") or stats.get("samples")
    val_single = stats.get("sample_value")

    if val_list and isinstance(val_list, list):
        # Ambil max 5 sample agar hemat token
        clean_vals = [str(v) for v in val_list[:5] if v is not None]
        if clean_vals:
            parts.append(f"Samples={str(clean_vals)}")
    elif val_single:
        # Format singular
        parts.append(f"Sample='{val_single}'")

    # B. NUMERIC STATS
    if "min" in stats and "max" in stats:
        try:
            parts.append(f"Range=[{stats['min']}..{stats['max']}]")
        except: pass
    
    if "mean" in stats:
        try:
            parts.append(f"Mean={float(stats['mean']):.2f}")
        except: pass

    # C. DISTINCT COUNT (Cardinality)
    if "distinct_count" in stats:
        parts.append(f"Distinct={stats['distinct_count']}")

    if not parts:
        return ""
    
    return " | STATS: " + ", ".join(parts)
# ==============================================================================
# 2. CORE LOGIC: CARD BUILDER (Update fungsi ini)
# ==============================================================================
def _build_cards(
    filtered_allowed: Dict[str, List[str]],
    reranked: List[Dict[str, Any]],
    max_cards: int = 15,
    max_inject: int = 12,
    desc_max_len: int = 100,
) -> str:
    cards: List[str] = []
    present = set()
    
    # debug_count = 0 
    # print(f"\n[DEBUG CARD PREVIEW] Menampilkan Top 5 Kartu untuk LLM:")
    if RETRIEVE_DEBUG:
        debug_count = 0 
        print(f"\n[DEBUG CARD PREVIEW] Menampilkan Top 5 Kartu untuk LLM:")

    for it in (reranked or [])[:max_cards]:
        md = it.get("meta") or {}
        table = _meta_table(md)
        column = _meta_column(md)
        
        if not table or not column:
            continue
        if table not in filtered_allowed:
            continue

        # --- LOGIKA DESKRIPSI (WATERFALL PRIORITY) ---
        desc = ""

        # 1. PRIORITY 1: Metadata Explicit (Hasil Sparse JSONL Anda)
        # JSONL Sparse Anda sekarang punya key "description" di dalam metadata.
        if md.get("description"):
            desc = md.get("description").strip()

        # 2. PRIORITY 2: Parsing Teks Dokumen (Hasil Dense JSONL Anda)
        # JSONL Dense Anda formatnya: "Table: ... Description: <ISI>.. Primary Key..."
        if not desc:
            # Di retrieve_node, teks dokumen disimpan di key "doc" atau "text"
            doc_text = it.get("doc") or it.get("text") or it.get("document") or ""
            
            if "Description:" in doc_text:
                try:
                    # Ambil teks setelah "Description:"
                    start = doc_text.find("Description:") + len("Description:")
                    
                    # Cari batas akhir. Di data Dense Anda, batasnya adalah ".. " (titik dua kali)
                    end = doc_text.find("..", start)
                    
                    # Backup boundaries jika ".. " tidak ketemu
                    if end == -1: end = doc_text.find("Synonyms:", start)
                    if end == -1: end = doc_text.find("Primary Key", start)
                    
                    # Jika ketemu batas, potong stringnya
                    if end != -1:
                        desc = doc_text[start:end].strip()
                    else:
                        # Ambil max 100 char kalau tidak nemu batas
                        desc = doc_text[start:start+100].strip()
                except:
                    pass

        # 3. PRIORITY 3: Fallback Regex (Manusiakan Nama Kolom)
        # Jaga-jaga jika Priority 1 & 2 gagal total
        if not desc:
            # "productCode" -> "product Code" -> "product code"
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', column)
            s2 = re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1).replace("_", " ")
            desc = s2.lower()

        # Bersihkan & Potong Deskripsi
        desc = desc.replace("\n", " ").strip()
        # Hapus karakter sisa aneh jika parsing kurang bersih
        if desc.endswith(".."): desc = desc[:-2].strip()
        
        if len(desc) > desc_max_len:
            desc = desc[:desc_max_len] + "..."
        
        # --- AMBIL STATISTIK ---
        stats_str = _format_stats_for_llm(md.get("stats"))
        
        # --- BUILD STRING ---
        card_str = f"- table: {table} | column: {column} | desc: {desc}{stats_str}"
        cards.append(card_str)
        present.add((table.lower(), column.lower()))

        # Print debug 5 baris pertama
        # if debug_count < 5:
        if RETRIEVE_DEBUG and debug_count < 5:
            print(f"   [{debug_count+1}] {card_str}")
            debug_count += 1

    # --- BAGIAN B: Injection (Kolom Wajib yang Belum Muncul) ---
    injected = 0
    for tbl, cols in (filtered_allowed or {}).items():
        if injected >= max_inject: break
        for col in cols:
            if injected >= max_inject: break
            
            key = (tbl.lower(), col.lower())
            if key in present: continue
            
            # Injection sederhana
            clean_col = col.replace("_", " ")
            card_str = f"- table: {tbl} | column: {col} | desc: {clean_col}"
            cards.append(card_str)
            present.add(key)
            injected += 1

    return "\n".join(cards)
# ======================================================================
# =========================== SPARSE (BM25) ============================
# ======================================================================
_BM25_CACHE: dict[str, Any] = {}

# Stopwords for BM25 tokenization
_STOPWORDS = (set(ENGLISH_STOP_WORDS) - {"name", "names"}) | {"show", "all", "list", "their", "that", "have", "no",
    "include", "includes", "including", "sort", "sorted", "sorting",
    "result", "results", "record", "records", "row", "rows",
    "display", "return", "get", "give", "find", "fetch", "please", "kindly",
    "alphabetical", "alphabetically", "ascending", "descending", "distinct",
    "yang", "dan", "atau", "ke", "dari", "di", "itu", "ini",
    "tampilkan", "daftar",
}

def _normalize_text(txt: str) -> str:
    txt = txt.lower()
    txt = re.sub(r"(\d)[,\.](\d{3})(\b|$)", r"\1\2", txt)
    txt = re.sub(r"(\w)'s\b", r"\1", txt)
    txt = txt.replace("'", "")
    txt = re.sub(r"[^a-z0-9_]+", " ", txt)
    return txt.strip()

def _tokenize_bm25_query(q: str) -> List[str]:
    s = _normalize_text(q)
    raw_tokens = s.split()
    toks: List[str] = []

    STEM_EXCEPTIONS = {"status", "address", "business", "orders", "sales"} 

    for t in raw_tokens:
        if not t:
            continue

        if t in STEM_EXCEPTIONS:
            if t not in _STOPWORDS:
                toks.append(t)
            continue

        if len(t) > 4 and t.endswith("ies"):
            t = t[:-3] + "y"
        elif len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]

        if t and t not in _STOPWORDS:
            toks.append(t)

    return toks

def build_sparse_query_from_question(q: str) -> str:
    toks = _tokenize_bm25_query(q)
    q_l = (q or "").lower()
    if "list price" in q_l and "msrp" not in toks:
        toks.append("msrp")
    return " ".join(toks)

def _doc_tokens_from_meta(meta: Dict[str, Any]) -> List[str]:
    """
    Mengambil token untuk BM25.
    """
    # --- [MODIFIED 1] Use 'keywords' field if available (from sparse.jsonl) ---
    if "keywords" in meta:
        # Langsung pakai field 'keywords' yang sudah kaya nutrisi (sample values, aliases, dll)
        return _tokenize_bm25_query(str(meta["keywords"]))
    
    # --- Fallback: Legacy Logic (jika jsonl belum di-update) ---
    table = (_meta_table(meta) or "").lower()
    column = (_meta_column(meta) or "").lower()
    desc = str(meta.get("description") or "").lower()

    aliases = _parse_str_or_list_lower(meta.get("tags_alias") or meta.get("aliases") or [])
    roles   = _parse_str_or_list_lower(meta.get("tags_role")  or meta.get("roles")   or [])

    extra: List[str] = []
    has_semantic = bool(aliases or roles)

    # Table-level synonyms from schema metadata (dynamic)
    # Get synonyms from JSONL metadata for all tables
    if not has_semantic:
        table_synonyms = get_table_synonyms(table)
        if table_synonyms:
            extra.extend(table_synonyms)

    # Column-level synonyms (manual fallback)
    if column == "msrp": extra += ["price", "list price", "retail price", "harga jual"]
    elif column == "priceeach": extra += ["unit price", "selling price", "harga satuan"]
    elif column == "quantityinstock": extra += ["stock", "stok", "qty", "quantity", "on hand"]
    elif column == "quantityordered": extra += ["qty", "quantity ordered", "jumlah pesanan"]
    elif column == "paymentdate": extra += ["payment date", "tanggal pembayaran"]
    elif column == "amount": extra += ["amount", "nominal", "jumlah pembayaran"]
    if column == "htmldescription": extra += ["html", "description", "html description"]
    if column == "textdescription": extra += ["text", "description", "text description"]

    bag = " ".join([table, column, desc] + aliases + roles + extra)
    return _tokenize_bm25_query(bag)

def jsonl_sparse_search(q: str, jsonl_paths: List[str], max_hits: int = 20) -> List[str]:
    # cache key by files + mtime
    sig = []
    for p in jsonl_paths:
        try:
            st = os.stat(p)
            sig.append(f"{p}:{int(st.st_mtime)}:{int(st.st_size)}")
        except Exception:
            sig.append(p)
    cache_key = "||".join(sorted(sig))

    if cache_key not in _BM25_CACHE:
        tokenized_corpus: List[List[str]] = []
        ids: List[str] = []

        for path in jsonl_paths:
            p = Path(path)
            if not p.exists():
                continue
            srcname = p.name

            for line_no, (raw, obj) in enumerate(read_jsonl(path), 1):
                if not isinstance(obj, dict):
                    # Handle potential list-of-dict
                    try: maybe = json.loads(raw)
                    except: maybe = None
                    if isinstance(maybe, list):
                        for i, item in enumerate(maybe):
                            if not isinstance(item, dict): continue
                            table  = (_meta_table(item) or "").lower()
                            column = (_meta_column(item) or "").lower()
                            
                            # <<< FIX UTAMA DI SINI: Tambahkan item.get("id") >>>
                            doc_id_raw = (
                                item.get("id") 
                                or item.get("field_urn") 
                                or item.get("urn") 
                                or f"{srcname}:{table}::{column}"
                            )
                            
                            doc_id = str(doc_id_raw)
                            tokens = _doc_tokens_from_meta(item)
                            tokenized_corpus.append(tokens)
                            ids.append(doc_id)
                        continue 
                    continue

                # Case: dict normal
                table  = (_meta_table(obj) or "").lower()
                column = (_meta_column(obj) or "").lower()
                
                # <<< FIX UTAMA DI SINI: Tambahkan obj.get("id") >>>
                doc_id_raw = (
                    obj.get("id") 
                    or obj.get("field_urn") 
                    or obj.get("urn") 
                    or f"{srcname}:{table}::{column}"
                )
                
                doc_id = str(doc_id_raw)
                tokens = _doc_tokens_from_meta(obj)
                tokenized_corpus.append(tokens)
                ids.append(doc_id)

        if not tokenized_corpus:
            _BM25_CACHE[cache_key] = {"bm25": None, "ids": []}
        else:
            _BM25_CACHE[cache_key] = {
                "bm25": BM25Okapi(tokenized_corpus),
                "ids": ids,
            }

    entry = _BM25_CACHE[cache_key]
    if not entry["bm25"]:
        return []

    query_tokens = _tokenize_bm25_query(q)
    scores = entry["bm25"].get_scores(query_tokens)

    if not isinstance(scores, list):
        scores = list(scores)

    if not scores or max(scores) <= 0:
        if SPARSE_DEBUG:
            print("[SPARSE] all scores are 0 → returning no sparse hits")
        return []

    ranked = sorted(zip(entry["ids"], scores), key=lambda x: x[1], reverse=True)
    out = [doc_id for doc_id, _ in ranked[:max_hits]]

    if SPARSE_DEBUG:
        print("[SPARSE] top hits:")
        for doc_id, sc in ranked[:min(max_hits, 10)]:
            print(f"  - {doc_id}  score={sc:.3f}")

    return out

def _doc_matches_sparse(md: Dict[str, Any], sparse_ids: List[str]) -> bool:
    """
    Cek apakah dokumen Dense ini ada di daftar hits Sparse.
    Updated: Support format ID simpel "table.column".
    """
    if not md or not sparse_ids:
        return False

    # Konversi semua candidate ID sparse ke lowercase set biar cepat
    sid_set = {x.lower() for x in sparse_ids}

    # Ambil info dari Metadata Dense
    table  = (_meta_table(md) or "").lower()
    column = (_meta_column(md) or "").lower()

    # --- 1. STRATEGI UTAMA: Simple ID Match (table.column) ---
    # Ini sesuai dengan format export_sparse_final.py kita yang baru
    if table and column:
        simple_id = f"{table}.{column}"
        if simple_id in sid_set:
            return True

    # --- 2. STRATEGI FALLBACK: Legacy URN Match ---
    # Cek jika ID tersimpan di field_urn / urn
    urn = str(md.get("field_urn") or md.get("urn") or "").lower()
    if urn and urn in sid_set:
        return True

    # --- 3. STRATEGI FALLBACK: Suffix Match ---
    # Cek jika sparse ID berakhiran :table::column (Gaya DataHub lama)
    if table and column:
        suffix = f":{table}::{column}"
        if any(x.endswith(suffix) for x in sid_set):
            return True

    return False

# =========================
# ======= SCORING =========
# =========================
def apply_column_priority_boost(q_l: str, table: str, col_name: str, roles: List[str], aliases: List[str]) -> float:
    bonus = 0.0
    
    # Generic name boosting
    if any(k in q_l for k in ["name"]):
        if any(r == "name" or "name" in r for r in roles) or "name" in col_name:
            bonus += NAME_QUERY_BONUS
    
    # Customers table
    if table.lower() == "customers":
        mentions_customer = any(k in q_l for k in ["customer", "customers"])
        asks_id_like = any(k in q_l for k in [" id", "id ", " id,", "identifier", "code", "customer number"])
        asks_for_names = any(k in q_l for k in ["name", "names", "nama"])  # ← NEW
        
        # NEW: Detect explicit contact person request
        asks_contact_explicitly = any(k in q_l for k in [
            "contact name", "contact first", "contact last",
            "first name", "last name", "person name"
        ])
        
        # NEW: Strong disambiguation for generic "names"
        if mentions_customer and asks_for_names and not asks_contact_explicitly:
            if "customername" in col_name or "customer_name" in col_name:
                bonus += 0.9
            if "contactfirstname" in col_name or "contactlastname" in col_name:
                bonus -= 0.8
        
        # Existing logic (keep as-is)
        elif mentions_customer and not asks_id_like:
            if "customername" in col_name or "customer_name" in col_name:
                bonus += 0.6
            if "customernumber" in col_name or col_name.endswith("number"):
                bonus -= 0.35
    
    # Products table (unchanged)
    if table.lower() == "products":
        mentions_product = any(k in q_l for k in ["product", "products"])
        asks_id_like = any(k in q_l for k in [" id", "id ", " id,", "identifier", "code", "product code"])
        if mentions_product and not asks_id_like:
            if "productname" in col_name or "product_name" in col_name:
                bonus += 0.5
            if "productcode" in col_name or col_name.endswith("code"):
                bonus -= 0.3
    
    # Generic ID boosting (unchanged)
    if any(k in q_l for k in [" id ", "id ", " id,", "identifier", "code"]):
        if (any(r == "id" or "id" in r for r in roles) or "number" in col_name or col_name.lower().endswith("id")):
            bonus += ID_QUERY_BONUS
    
    return bonus

def _is_mentioned(q_l: str, table: str, synonyms: List[str] = None) -> bool:
    """
    Dynamic mention check using table name and synonyms from schema metadata.

    Args:
        q_l: Lowercased question text
        table: Table name to check (lowercase)
        synonyms: Optional list of synonyms (fetched from schema if not provided)

    Returns:
        True if table is mentioned in question, False otherwise
    """
    if not q_l or not table:
        return False

    # 1. Direct table name match
    if table in q_l:
        return True

    # 2. Singular form match (e.g., "customer" matches "customers")
    singular = get_singular_form(table)
    if singular and singular in q_l:
        return True

    # 3. Synonyms from schema metadata (dynamic - not hardcoded)
    if synonyms is None:
        synonyms = get_table_synonyms(table)
    for synonym in synonyms:
        if synonym in q_l:
            return True

    return False


def score_tables_from_reranked(reranked: List[Dict[str, Any]], query_tokens: List[str]) -> Dict[str, float]:
    scores: Dict[str, float] = defaultdict(float)
    qt = [t.lower() for t in query_tokens]
    q_l = " ".join(qt)

    # Get all tables from schema (dynamic)
    all_tables = get_all_tables()

    # Build mentions map dynamically using schema metadata
    # Instead of: mentions_customer = any(k in q_l for k in ["customer", "customers", ...])
    # We now use: _is_mentioned(q_l, "customers") which checks table name + synonyms from JSONL
    mentions_map = {tbl: _is_mentioned(q_l, tbl) for tbl in all_tables}

    # Extract commonly used mentions for readability (backward compatible)
    mentions_customer = mentions_map.get("customers", False)
    mentions_product = mentions_map.get("products", False)
    mentions_order = mentions_map.get("orders", False)
    mentions_payment = mentions_map.get("payments", False)
    mentions_office = mentions_map.get("offices", False)
    mentions_productline = mentions_map.get("productlines", False)
    mentions_orderdetails = mentions_map.get("orderdetails", False)
    mentions_employee = mentions_map.get("employees", False)

    # Detect aggregate context - indicates we need detail tables
    math_keywords = ['sum', 'count', 'total', 'avg', 'average', 'number of', 'jumlah', 'rata-rata',
                     'percentage', 'percent', 'ratio', 'proportion', 'persentase', 'rasio']
    has_aggregation = any(kw in q_l for kw in math_keywords)

    # Check if we need numeric columns - based on RETRIEVED columns metadata (not hardcoded keywords)
    # Check if any retrieved column has numeric/aggregate characteristics
    needs_numeric_data = _check_needs_numeric_data_from_retrieved(reranked)

    # Only penalize orderdetails if NOT in aggregate context AND NOT needing numeric data
    penalize_orderdetails_default = mentions_order and not mentions_orderdetails and not has_aggregation and not needs_numeric_data

    for it in reranked:
        md = it.get("meta") or {}
        tbl = _meta_table(md)
        if not tbl: continue
        desc = (md.get("description") or "").lower()
        base = 0.0
        tl = tbl.lower()

        if it.get("sparse"): base += 0.5
        if tl in q_l: base += 1.0
        for t in qt:
            if t and t in desc: base += 0.3

        # Dynamic table mention boost (instead of hardcoded "customers", "products", etc.)
        # Uses mentions_map built from schema metadata
        if tl in mentions_map and mentions_map[tl]:
            # High boost for dimension/master tables (customers, products)
            if get_table_type(tl) in ("dimension", "unknown"):
                base += 0.9
            # Medium boost for transaction tables (orders, payments)
            elif is_transaction_table(tl):
                base += 0.7
            # High boost for bridge/lookup tables (offices, productlines, orderdetails, employees)
            else:
                base += 0.8

        # Special handling for orderdetails (bridge table with numeric columns)
        # Don't penalize orderdetails in aggregate/numeric contexts
        if penalize_orderdetails_default and tl == "orderdetails": base -= 0.4

        # Boost orderdetails when numeric data is needed (even if not explicitly mentioned)
        if needs_numeric_data and tl == "orderdetails": base += 0.6

        scores[tbl] += base
    return scores
# ==============================================================================
# NUMERIC DATA DETECTION (METADATA-DRIVEN)
# ==============================================================================
def _check_needs_numeric_data_from_retrieved(reranked: List[Dict[str, Any]]) -> bool:
    """
    Check if the query needs numeric data based on retrieved columns' metadata.

    Uses a metadata-driven approach:
    1. Checks datatype field for numeric types (INTEGER, DECIMAL, NUMERIC, FLOAT, DOUBLE, etc.)
    2. Falls back to column name patterns for backward compatibility

    Args:
        reranked: List of reranked column results with metadata

    Returns:
        True if any retrieved column indicates numeric data is needed
    """
    # Numeric SQL datatypes
    numeric_datatypes = {
        'int', 'integer', 'tinyint', 'smallint', 'mediumint', 'bigint',
        'decimal', 'numeric', 'float', 'double', 'double precision',
        'real', 'money', 'smallmoney', 'bit'
    }

    for item in reranked[:MAX_CTX_DOCS]:
        md = item.get("meta") or {}

        # Primary: Check datatype field from metadata
        datatype = (md.get("datatype") or md.get("data_type") or "").strip().lower()
        if datatype:
            # Remove any size specification (e.g., "decimal(10,2)" -> "decimal")
            base_type = datatype.split('(')[0].strip()
            if base_type in numeric_datatypes:
                # Check if this numeric column is used for calculations
                # (not just IDs or counts)
                col = _meta_column(md).lower()
                # Exclude primary keys and foreign keys (IDs) as they're not used for calculations
                desc = str(md.get("description") or "").lower()
                if not any(kw in col for kw in ['id', 'code', 'key']):
                    return True
                # Check description for calculation-related terms
                if any(kw in desc for kw in ['quantity', 'price', 'amount', 'total', 'cost', 'value', 'each', 'rate']):
                    return True

        # Fallback: Check column name pattern for numeric-sounding columns
        # (only if datatype check above didn't trigger)
        col = _meta_column(md).lower()
        if any(kw in col for kw in ['quantity', 'price', 'amount', 'total', 'value', 'cost', 'each', 'rate']):
            return True

    return False
# ==============================================================================
# BRIDGE TABLE INFERENCE
# ==============================================================================
def _infer_bridge_tables(
    selected_tables: set,
    allowed: Dict[str, List[str]],
    reranked: List[Dict[str, Any]],
) -> set:
    """
    Infer bridge tables needed to connect selected tables.

    Uses FK information from column metadata to detect bridge tables.
    For example:
    - customers + orderdetails → need orders (bridge table)
    - orders + products → need orderdetails (bridge table)

    Args:
        selected_tables: Set of lowercase table names already selected
        allowed: Dict mapping table names to their allowed columns
        reranked: List of reranked column results with metadata

    Returns:
        Set of lowercase bridge table names to add
    """
    if len(selected_tables) < 2:
        return set()

    # Build FK graph from metadata
    # Format: {source_table: {fk_column: target_table}}
    fk_graph: Dict[str, Dict[str, str]] = {}

    for item in reranked:
        md = item.get("meta") or {}
        desc = str(md.get("description") or "")

        # Parse FK references from description
        # Pattern: "FK to targetTable.targetColumn"
        fk_match = re.search(r'FK to (\w+)\.(\w+)', desc, re.IGNORECASE)
        if fk_match:
            source_table = _meta_table(md)
            fk_column = _meta_column(md)
            target_table = fk_match.group(1)

            if source_table and fk_column:
                st_lower = source_table.lower()
                fc_lower = fk_column.lower()
                tt_lower = target_table.lower()

                if st_lower not in fk_graph:
                    fk_graph[st_lower] = {}
                fk_graph[st_lower][fc_lower] = tt_lower

    # Find bridge tables using BFS
    bridge_tables = set()

    # Convert selected_tables to lowercase for comparison
    selected_lower = {t.lower() for t in selected_tables}

    # NEW: Also consider tables with numeric columns in reranked results
    # Uses metadata-driven approach: check datatype field + column patterns as fallback
    numeric_datatypes = {
        'int', 'integer', 'tinyint', 'smallint', 'mediumint', 'bigint',
        'decimal', 'numeric', 'float', 'double', 'double precision', 'real'
    }

    tables_with_numeric_cols = set()
    for item in reranked[:MAX_CTX_DOCS]:
        md = item.get("meta") or {}
        tbl = _meta_table(md)
        col = _meta_column(md)
        if tbl and col:
            # Primary: Check datatype field for numeric types
            datatype = (md.get("datatype") or md.get("data_type") or "").strip().lower()
            col_lower = col.lower()
            is_numeric = False

            if datatype:
                base_type = datatype.split('(')[0].strip()
                is_numeric = base_type in numeric_datatypes
            else:
                # Fallback: Check column name pattern
                is_numeric = any(kw in col_lower for kw in ['quantity', 'price', 'amount', 'value', 'total', 'cost', 'each', 'rate'])

            # Exclude ID columns from numeric consideration
            if is_numeric and not any(kw in col_lower for kw in ['id', 'code', 'key']):
                tables_with_numeric_cols.add(tbl.lower())

    # Expand selection to include tables with numeric columns
    expanded_tables = set(selected_lower) | tables_with_numeric_cols

    # For each pair of expanded tables, check if we need a bridge
    table_list = list(expanded_tables)
    for i, tbl_a in enumerate(table_list):
        for tbl_b in table_list[i+1:]:
            # Check if tbl_a can reach tbl_b via FK path
            path = _find_fk_path(tbl_a, tbl_b, fk_graph, allowed)
            if path:
                # Add intermediate tables (excluding endpoints)
                for bridge in path[1:-1]:
                    if bridge not in selected_lower:
                        bridge_tables.add(bridge)

    # Also check for common FK relationships
    # If table A has FK to table B, and both have columns in results, no bridge needed
    # But if we need columns from A and C where A→B→C, we need B as bridge

    return bridge_tables

def _find_fk_path(
    start: str,
    end: str,
    fk_graph: Dict[str, Dict[str, str]],
    allowed: Dict[str, List[str]],
    max_depth: int = 3,
) -> Optional[List[str]]:
    """
    Find FK path between two tables using BFS.

    Args:
        start: Starting table name (lowercase)
        end: Target table name (lowercase)
        fk_graph: FK relationship graph
        allowed: Allowed tables and columns
        max_depth: Maximum path length to search

    Returns:
        List of table names forming the path, or None if no path found
    """
    from collections import deque

    if start == end:
        return [start]

    # Build reverse graph (target -> sources) for easier traversal
    reverse_graph: Dict[str, set] = {}
    for source, fk_targets in fk_graph.items():
        for fk_col, target in fk_targets.items():
            if target not in reverse_graph:
                reverse_graph[target] = set()
            reverse_graph[target].add(source)

    # BFS from start to end
    queue = deque([(start, [start])])
    visited = {start}

    while queue and len(queue[0][1]) <= max_depth:
        current, path = queue.popleft()

        # Check forward FKs (current -> target)
        if current in fk_graph:
            for target in fk_graph[current].values():
                if target == end:
                    return path + [target]
                if target not in visited and target in [t.lower() for t in allowed.keys()]:
                    visited.add(target)
                    queue.append((target, path + [target]))

        # Check reverse FKs (target -> current)
        if current in reverse_graph:
            for source in reverse_graph[current]:
                if source == end:
                    return path + [source]
                if source not in visited and source in [t.lower() for t in allowed.keys()]:
                    visited.add(source)
                    queue.append((source, path + [source]))

    return None
# ======================================================================
# =========================== RETRIEVE NODE ============================
# ======================================================================
def _has_join_intent(question: str) -> bool:
    q = (question or "").lower()
    join_signals = [
        "join", "gabung", "combine", "combine with", "along with", "together with",
        "and their", "with their", "beserta", "bersama", "dengan",
        "for each", "per ", "setiap",
    ]
    multi_entity = [
        ("customer", "order"), ("customer", "payment"), ("order", "product"),
        ("order", "order detail"), ("order", "orderdetails"), ("employee", "office"),
        ("employee", "customer"), ("office", "employee"),
    ]
    if any(s in q for s in join_signals): return True
    for a, b in multi_entity:
        if a in q and b in q: return True
    return False

def _detect_primary_entity_table(question: str) -> Optional[str]:
    """
    Detect primary entity table from question using schema metadata.

    Dynamic version - uses synonyms from JSONL metadata instead of hardcoded patterns.
    Searches through all tables and their synonyms to find matches in the question.

    Args:
        question: User question text

    Returns:
        First matching table name (lowercase), or None if no match
    """
    q = (question or "").lower()

    # Get all tables from schema
    all_tables = get_all_tables()

    # Build synonym map: {synonym: table}
    synonym_to_table = {}

    # Add table names themselves (both singular and plural)
    for table in all_tables:
        # Direct table name match
        synonym_to_table[table] = table
        # Singular form
        synonym_to_table[get_singular_form(table)] = table

        # Synonyms from metadata (aggregated from all columns)
        for synonym in get_table_synonyms(table):
            synonym_to_table[synonym] = table

    # Find first matching table (check in order of specificity)
    # Priority: multi-word phrases > single words
    # Sort by length (descending) to prioritize longer, more specific matches
    sorted_synonyms = sorted(synonym_to_table.items(), key=lambda x: len(x[0]), reverse=True)

    for synonym, table in sorted_synonyms:
        # Check for phrase match first (more specific)
        if " " in synonym:
            if synonym in q:
                return table
        # Then single word match
        elif synonym.replace("_", "") in q.split():
            return table

    return None


def _is_table_mentioned(question: str, table: str) -> bool:
    """
    Check if a table is mentioned in the question using schema metadata.

    Dynamic replacement for hardcoded mention checks like:
    - mentions_customer = any(k in q for k in ["customer", "customers", "pelanggan"])
    - mentions_product = any(k in q for k in ["product", "products", "produk"])

    Args:
        question: User question text
        table: Table name to check (lowercase)

    Returns:
        True if table is mentioned, False otherwise
    """
    if not question or not table:
        return False

    q = question.lower()
    table_lower = table.lower()

    # 1. Direct table name match
    if table_lower in q:
        return True

    # 2. Singular form match
    singular = get_singular_form(table_lower)
    if singular and singular in q:
        return True

    # 3. Synonyms from schema metadata
    for synonym in get_table_synonyms(table):
        if synonym in q:
            return True

    # 4. Common phrase patterns (split by space to avoid substring false positives)
    # e.g., "product line" should match "product line" but not "line item"
    q_words = set(q.split())
    for synonym in get_table_synonyms(table):
        if " " in synonym:
            synonym_words = set(synonym.split())
            if synonym_words.issubset(q_words):
                return True

    return False


def _should_lock_single_table_ctx(question: str, qtype: str) -> bool:
    if qtype not in {"lookup", "aggregate", "rank"}: return False
    if _has_join_intent(question): return False
    return _detect_primary_entity_table(question) is not None

def retrieve_node(state: RAGState) -> RAGState:
    # -----------------------------
    # 0) Build vector store (DENSE)
    # -----------------------------
    # Asumsi: build_vector_store() diatur untuk membaca *_dense.jsonl
    col = build_vector_store()

    q = (state["question"] or "").strip()
    q_l = q.lower()
    qtype = _get_qtype(state)

    wants_customer_no_id = _wants_customer_name_only(q)
    productline_lookup = _is_productline_lookup(q)
    orders_simple_lookup = _is_simple_orders_lookup(q)

    # --- [MODIFIED 2] Filter Paths: Hanya ambil file yang namanya mengandung "_sparse" ---
    all_paths = state.get("jsonl_paths", [])
    sparse_target_paths = [p for p in all_paths if "_sparse" in str(p)]
    
    # Fallback: Jika tidak ada file _sparse, pakai semua (backward compat)
    if not sparse_target_paths:
        sparse_target_paths = all_paths

    # -----------------------------
    # 1) Sparse query build + BM25 search
    # -----------------------------
    sparse_query_clean = build_sparse_query_from_question(q)
    sparse_tokens = _tokenize_bm25_query(sparse_query_clean)

    # Panggil search dengan path yang sudah difilter
    sparse_field_urns = jsonl_sparse_search(
        sparse_query_clean,
        sparse_target_paths, # <--- Updated variable
        max_hits=SPARSE_MAX_HITS
    )

    # --- get sparse metas sekali, lalu pakai untuk pretty + flags ---
    sparse_docs_pretty: List[str] = []
    sparse_metas: List[Dict[str, Any]] = []

    if sparse_field_urns:
        try:
            got = col.get(ids=sparse_field_urns, include=["metadatas"])
            ids = got.get("ids") or []
            metas = got.get("metadatas") or []
            if metas and isinstance(metas[0], list): metas = metas[0]
            meta_by_id = {i: (m or {}) for i, m in zip(ids, metas)}

            for doc_id in sparse_field_urns:
                md = meta_by_id.get(doc_id, {}) or {}
                sparse_metas.append(md)
                tbl = _meta_table(md) or "?"
                c = _meta_column(md) or "?"
                sparse_docs_pretty.append(f"table={tbl}  column={c}")
        except Exception as e:
            sparse_docs_pretty = list(sparse_field_urns or [])
            sparse_metas = []
            if RETRIEVE_DEBUG: print(f"[WARN] Sparse meta fetch failed: {e!r}")

    sparse_has_customername = any(
        ((_meta_table(m) or "").lower() == "customers") and ((_meta_column(m) or "").lower() == "customername")
        for m in sparse_metas if isinstance(m, dict)
    )

    base_k = max(12, TOP_K)
    dense_k = base_k
    if wants_customer_no_id and not sparse_has_customername:
        dense_k = max(28, base_k)

    # -----------------------------
    # 2) Dense query
    # -----------------------------
    query_text = build_dynamic_probe(q, qtype=qtype)
    res = col.query(
        query_texts=[query_text],
        n_results=dense_k,
        include=["documents", "metadatas", "distances"],
    )

    # -----------------------------
    # 3) Unpack dense results
    # -----------------------------
    dense_docs = (res.get("documents") or [[]])[0]
    if dense_docs and isinstance(dense_docs[0], list): dense_docs = dense_docs[0]
    dense_metas = (res.get("metadatas") or [[]])[0]
    if dense_metas and isinstance(dense_metas[0], list): dense_metas = dense_metas[0]
    dense_dists = (res.get("distances") or [[]])[0]
    if dense_dists and isinstance(dense_dists[0], list): dense_dists = dense_dists[0]
    dense_ids = (res.get("ids") or [[]])[0]
    if dense_ids and isinstance(dense_ids[0], list): dense_ids = dense_ids[0]

    n = min(len(dense_docs), len(dense_metas), len(dense_dists) if dense_dists else len(dense_docs), len(dense_ids) if dense_ids else len(dense_docs))
    dense_docs = dense_docs[:n]
    dense_metas = dense_metas[:n]
    dense_dists = (dense_dists or [0.0] * n)[:n]
    dense_ids = (dense_ids or [""] * n)[:n]

    dense_has_customername = any(
        ((_meta_table(md) or "").lower() == "customers") and ((_meta_column(md) or "").lower() in ("customername", "customer_name"))
        for md in dense_metas
    )

    # -----------------------------
    # 4) Scoring + rerank (dense + sparse bonus + priority boost)
    # -----------------------------
    scored = []
    for rank, (doc_text, md, dist, _id) in enumerate(zip(dense_docs, dense_metas, dense_dists, dense_ids)):
        base = -float(dist) if dist is not None else 0.0
        sparse_hit = _doc_matches_sparse(md, sparse_field_urns)
        table_name = _meta_table(md) or ""
        col_raw = _meta_column(md) or ""
        col_name = col_raw.lower()
        roles, aliases = _extract_roles_aliases(md)

        boosted = base + (SPARSE_BONUS if sparse_hit else 0.0)
        boosted += apply_column_priority_boost(q_l=q_l, table=table_name, col_name=col_name, roles=roles, aliases=aliases)

        scored.append({
            "rank": rank, "id": _id, "doc": doc_text, "meta": md or {},
            "dist": float(dist) if dist is not None else 0.0, "base": base,
            "sparse": sparse_hit, "boosted": boosted,
        })

    # -----------------------------
    # 5) Auto-inject sangat terbatas (customerName)
    # -----------------------------
    if wants_customer_no_id and sparse_has_customername and not dense_has_customername:
        sparse_customername_urn = None
        for u in sparse_field_urns:
            if u.lower().endswith("::customername"):
                sparse_customername_urn = u
                break
        if sparse_customername_urn:
            present_ids = {item["id"] for item in scored if item.get("id")}
            if sparse_customername_urn not in present_ids:
                got = col.get(ids=[sparse_customername_urn])
                if got and got.get("ids"):
                    md = (got.get("metadatas") or [{}])[0]
                    doc = (got.get("documents") or [""])[0]
                    roles, aliases = _extract_roles_aliases(md)
                    table_name = _meta_table(md) or ""
                    col_raw = _meta_column(md) or ""
                    col_name = col_raw.lower()
                    max_boost = max((x["boosted"] for x in scored), default=0.0)
                    boosted = max_boost + 0.05
                    boosted += apply_column_priority_boost(q_l=q_l, table=table_name, col_name=col_name, roles=roles, aliases=aliases)
                    scored.append({
                        "rank": len(scored), "id": sparse_customername_urn, "doc": doc, "meta": md or {},
                        "dist": 0.0, "base": 0.0, "sparse": True, "boosted": boosted,
                    })

    # -----------------------------
    # 6) Final reranked slice
    # -----------------------------
    reranked = sorted(scored, key=lambda x: x["boosted"], reverse=True)
    reranked = reranked[:MAX_CTX_DOCS]

    state["ranked_cols"] = [
        ((_meta_table(it.get("meta") or {}) or "").strip().lower(), (_meta_column(it.get("meta") or {}) or "").strip())
        for it in (reranked or []) if (_meta_table(it.get("meta") or {}) and _meta_column(it.get("meta") or {}))
    ]
    
    # -----------------------------
    # 7) Debug prints
    # -----------------------------
    if RETRIEVE_DEBUG:
        line = _print_banner("RETRIEVE DEBUG")
        print(f"QUESTION      : {q}")
        print(f"DENSE QUERY   : {query_text}")
        print(f"SPARSE QUERY  : {sparse_query_clean}")
        print(f"TOKENS        : {sparse_tokens}")
        print(f"QTYPE         : {qtype}\n")

        total_sparse = len(sparse_docs_pretty)
        max_show = min(7, total_sparse)
        print(f"SPARSE CANDIDATES (top {max_show} of {total_sparse}):")
        if total_sparse == 0: print("  <none>\n")
        else:
            print(f"{'':7}{'table':<14} {'column':<30}")
            for i, s in enumerate(sparse_docs_pretty[:max_show], start=1):
                try:
                    _, rest = s.split("table=", 1)
                    table_part, col_part = rest.split("column=", 2)
                    table = table_part.strip() or "?"
                    column = col_part.strip() or "?"
                except Exception:
                    table = "?"
                    column = s
                print(f"  [{i:02d}] {table:<14} {column:<30}")
            if total_sparse > max_show: print(f"  ... (+{total_sparse - max_show} more)")
            print()

        total_dense = len(dense_metas)
        max_show = min(7, total_dense)
        print(f"DENSE CANDIDATES (top {max_show} of {total_dense}):")
        if total_dense == 0: print("  <none>\n")
        else:
            print(f"{'':7}{'table':<14} {'column':<30}")
            for i in range(max_show):
                md = dense_metas[i] or {}
                tbl = _meta_table(md) or "?"
                col_ = _meta_column(md) or "?"
                print(f"  [{i+1:02d}] {tbl:<14} {col_:<30}")
            if total_dense > max_show: print(f"  ... (+{total_dense - max_show} more)")
            print()

        total_rerank = len(reranked)
        max_show = min(7, total_rerank)
        print(f"RERANKED RESULT (top {max_show} of {total_rerank}):")
        if total_rerank == 0: print("  <no reranked candidates>")
        else:
            print(f"{'':7}{'table':<14} {'column':<30} {'sparse':>8}")
            for i, item in enumerate(reranked[:max_show], start=1):
                md = item.get("meta") or {}
                tbl = _meta_table(md) or "?"
                col_ = _meta_column(md) or "?"
                sparse_flag = "✓" if item.get("sparse") else "✗"
                print(f"  [{i:02d}] {tbl:<14} {col_:<30} {sparse_flag:>8}")
            if total_rerank > max_show: print(f"  ... (+{total_rerank - max_show} more)")
        print(f"{line}\n")

    # -----------------------------
    # 8) Build allowed + alias_map
    # -----------------------------
    allowed = defaultdict(list)
    alias_map = defaultdict(dict)

    for item in reranked:
        md = item["meta"] or {}
        t = _meta_table(md)
        c = _meta_column(md)
        if not (t and c): continue
        t_key = t.strip().lower()
        c_key = c.strip()
        allowed[t_key].append(c_key)
        aliases2 = _extract_aliases_for_map(md)
        for al in aliases2:
            al = al.strip()
            if al: alias_map[t_key][al] = c_key

    # -----------------------------
    # 9) Table scoring + filtered_allowed
    # -----------------------------
    token_for_score = _tokenize_bm25_query(sparse_query_clean)
    table_scores = score_tables_from_reranked(reranked, token_for_score)

    if qtype == "lookup" and wants_customer_no_id:
        filtered_allowed = {t: cols for t, cols in allowed.items() if t.lower() == "customers"}
    elif qtype == "lookup" and productline_lookup:
        only_pl = {t: cols for t, cols in allowed.items() if t.lower() == "productlines"}
        filtered_allowed = only_pl if only_pl else dict(allowed)
    elif qtype == "lookup" and orders_simple_lookup:
        only_orders = {t: cols for t, cols in allowed.items() if t.lower() == "orders"}
        filtered_allowed = only_orders if only_orders else dict(allowed)
    else:
        if _should_lock_single_table_ctx(question=q, qtype=qtype):
            primary = _detect_primary_entity_table(q)
            only_primary = {t: cols for t, cols in allowed.items() if t.lower() == (primary or "")}
            if only_primary: filtered_allowed = only_primary
            else:
                if table_scores:
                    sorted_tbls = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
                    top1 = sorted_tbls[0][0]
                    filtered_allowed = {t: cols for t, cols in allowed.items() if t == top1}
                else: filtered_allowed = dict(allowed)
        else:
            # -----------------------------
            # ENHANCED: Column-driven table selection
            # -----------------------------
            # Instead of just top 2 tables, include tables that:
            # 1. Have high scores (mentioned in question)
            # 2. Have columns that appeared in reranked results
            # 3. Are bridge tables needed to connect selected tables

            if table_scores:
                sorted_tbls = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)

                # DEBUG: Print table scores
                if RETRIEVE_DEBUG:
                    print(f"\n[TABLE SCORES]")
                    for tbl, score in sorted_tbls:
                        print(f"  {tbl}: {score:.2f}")

                # Base: Top 3 tables by score (increased from 2)
                max_top_tables = 3
                top_tables = {name for name, score in sorted_tbls[:max_top_tables] if score > 0.0}

                # Addition: Include tables with any column in reranked results
                # This captures tables that may not be mentioned but have needed columns
                tables_with_reranked_cols = set()
                for item in reranked[:MAX_CTX_DOCS]:
                    md = item.get("meta") or {}
                    tbl = _meta_table(md)
                    if tbl:
                        tables_with_reranked_cols.add(tbl.lower())

                # Merge: top scoring + tables with reranked columns
                # Use all available table keys (from original allowed, not filtered)
                all_table_keys_lower = {tbl.lower(): tbl for tbl in allowed.keys()}

                # DEBUG: Print what we're adding
                if RETRIEVE_DEBUG:
                    print(f"\n[COLUMN-DRIVEN SELECTION]")
                    print(f"  tables_with_reranked_cols: {tables_with_reranked_cols}")
                    print(f"  all_table_keys_lower: {list(all_table_keys_lower.keys())}")
                    to_add = [t for t in tables_with_reranked_cols if t in all_table_keys_lower]
                    print(f"  adding to top_tables: {to_add}")

                top_tables.update(t for t in tables_with_reranked_cols if t in all_table_keys_lower)

                # Addition: Bridge table inference
                # If we have multiple tables, check if we need bridge tables
                if len(top_tables) >= 2:
                    bridge_tables = _infer_bridge_tables(top_tables, allowed, reranked)
                    top_tables.update(bridge_tables)

                if top_tables:
                    filtered_allowed = {t: cols for t, cols in allowed.items() if t.lower() in top_tables}
                else:
                    filtered_allowed = dict(allowed)
            else:
                filtered_allowed = dict(allowed)

    if wants_customer_no_id:
        for t_key in list(filtered_allowed.keys()):
            if t_key.lower() == "customers":
                if "customerName" not in filtered_allowed[t_key]:
                    filtered_allowed[t_key].append("customerName")

    # -----------------------------
    # 10) Context build
    # -----------------------------
    allowed_block = _build_allowed_block(filtered_allowed)
    cards_text = _build_cards(
        filtered_allowed,
        reranked,
        max_cards=15,
        max_inject=12,
        desc_max_len=100,
    )

    context = (allowed_block + "\n\n### CARDS\n" + cards_text).strip()
    context += f"\n\n### _INTERNAL_HINT\n{json.dumps({'alias_map': alias_map}, ensure_ascii=False)}"

    state["context"] = context
    return state