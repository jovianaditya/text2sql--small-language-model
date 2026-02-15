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
    COL_BOOST_DIM_NAME_EXPLICIT,
    COL_BOOST_DIM_CONTACT_PENALTY,
    COL_BOOST_DIM_NAME_DEFAULT,
    COL_BOOST_DIM_ID_PENALTY,
    COL_BOOST_OTHER_NAME_DEFAULT,
    COL_BOOST_OTHER_ID_PENALTY,
    TABLE_SCORE_SPARSE_HIT,
    TABLE_SCORE_TABLE_MENTION,
    TABLE_SCORE_DESC_TOKEN,
    TABLE_SCORE_MENTION_DIM,
    TABLE_SCORE_MENTION_TXN,
    TABLE_SCORE_MENTION_OTHER,
    TABLE_SCORE_BRIDGE_PENALTY,
    TABLE_SCORE_BRIDGE_NUMERIC_BOOST,
    MAX_TOP_TABLES,
    MAX_CTX_DOCS,
    SPARSE_MAX_HITS,
    SPARSE_DEBUG,
    RETRIEVE_DEBUG,
    RETRIEVAL_STATS_MAX_SAMPLES,
    RETRIEVAL_DESC_KEYWORD_HEAD_WORDS,
    RETRIEVAL_CARDS_MAX_DOCS,
    RETRIEVAL_CARDS_MAX_INJECT,
    RETRIEVAL_CARDS_DESC_MAX_LEN,
    RETRIEVAL_CARDS_DEBUG_PREVIEW_COUNT,
    RETRIEVAL_DENSE_DESC_FALLBACK_LEN,
    RETRIEVAL_EXTRA_STOPWORDS,
    RETRIEVAL_GENERIC_MENTION_TERMS,
    RETRIEVAL_JOIN_SIGNALS,
    RETRIEVAL_NUMERIC_SQL_TYPES,
    RETRIEVAL_BM25_STEM_EXCEPTIONS_BASE,
    RETRIEVAL_BM25_STOPWORD_EXCEPTIONS,
    RETRIEVAL_STEM_SUFFIX_IES,
    RETRIEVAL_STEM_SUFFIX_IES_MIN_LEN,
    RETRIEVAL_STEM_SUFFIX_IES_TO,
    RETRIEVAL_STEM_SUFFIX_S,
    RETRIEVAL_STEM_SUFFIX_S_MIN_LEN,
    RETRIEVAL_STEM_SUFFIX_S_EXCLUDE,
    RETRIEVAL_TOKENIZE_ALNUM_PATTERN,
    RETRIEVAL_NORMALIZE_THOUSANDS_PATTERN,
    RETRIEVAL_NORMALIZE_POSSESSIVE_PATTERN,
    RETRIEVAL_NORMALIZE_NON_ALNUM_PATTERN,
    RETRIEVAL_DESC_CAMEL_SPLIT_PATTERN_1,
    RETRIEVAL_DESC_CAMEL_SPLIT_PATTERN_2,
    RETRIEVAL_NAME_QUERY_TERMS,
    RETRIEVAL_ID_LIKE_QUERY_TERMS,
    RETRIEVAL_NAME_REQUEST_TERMS,
    RETRIEVAL_CONTACT_NAME_TERMS,
    RETRIEVAL_NAME_ROLE_HINT_TERMS,
    RETRIEVAL_NAME_COLUMN_HINT_TERMS,
    RETRIEVAL_ID_ROLE_HINT_TERMS,
    RETRIEVAL_ID_COLUMN_SUFFIX_TERMS,
    RETRIEVAL_ID_COLUMN_SUBSTRING_TERMS,
    RETRIEVAL_ID_BOOST_COLUMN_SUFFIX_TERMS,
    RETRIEVAL_ID_BOOST_COLUMN_SUBSTRING_TERMS,
    RETRIEVAL_CONTACT_COLUMN_HINT_TERMS,
    RETRIEVAL_DIMENSION_LIKE_TABLE_TYPES,
    RETRIEVAL_ID_BOOST_QUERY_TERMS,
    RETRIEVAL_NUMERIC_SIGNAL_KEYWORDS,
    RETRIEVAL_NUMERIC_PATTERN_KEYWORDS,
    RETRIEVAL_NUMERIC_EXCLUDE_COLUMN_KEYWORDS,
    RETRIEVAL_DESC_GENERIC_TERMS,
    RETRIEVAL_AGGREGATE_QUERY_TERMS,
    RETRIEVAL_BASE_K_FLOOR,
    RETRIEVAL_DENSE_K_FLOOR,
    RETRIEVAL_SPARSE_FILENAME_MARKER,
    RETRIEVAL_FK_PATH_MAX_DEPTH,
    RETRIEVAL_ID_ROLE_TAGS,
    RETRIEVAL_METRIC_ROLE_TAGS,
    RETRIEVAL_NUMERIC_TYPE_HINTS,
    RETRIEVAL_TABLE_SCORING_MATH_KEYWORDS,
    RETRIEVAL_NAME_ONLY_ID_PATTERN,
    RETRIEVAL_NAME_ONLY_NAME_PATTERN,
    RETRIEVAL_NAME_ONLY_NUMBER_OF_PATTERN,
    RETRIEVAL_DEBUG_FOCUS_COLUMNS,
    RETRIEVAL_TABLE_NAME_PREFIX_STRIP_CHARS,
    RETRIEVAL_TABLE_VARIANT_MIN_TAIL_LEN,
    RETRIEVAL_COLUMN_VARIANT_REPLACEMENTS,
    RETRIEVAL_FK_REFERENCE_PATTERN,
    AGENT_ROUTER_QTYPE_LOOKUP,
    AGENT_ROUTER_QTYPE_AGGREGATE,
    AGENT_ROUTER_QTYPE_RANK,
)
from .state import RAGState
from .chroma_embedding import build_vector_store
from .utils import _print_banner, _meta_table, _meta_column, read_jsonl, _parse_str_or_list_lower, _extract_roles_aliases, _extract_aliases_for_map
from .helpers_main import _get_qtype, build_dynamic_probe
from .schema_analyzer import (
    get_schema_analyzer,
    get_all_tables,
    get_table_synonyms,
    get_singular_form,
    get_table_type,
    is_transaction_table,
    get_column_synonyms,
    get_name_column,
    get_id_column,
)
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
        clean_vals = [str(v) for v in val_list[:RETRIEVAL_STATS_MAX_SAMPLES] if v is not None]
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
# 2. CORE LOGIC: COLUMN RELEVANCE FILTERING (Metadata-Driven)
# ==============================================================================
def _is_column_relevant_to_query(
    column: str,
    table: str,
    question: str,
    qtype: str,
    metadata: Dict[str, Any]
) -> bool:
    """
    Determine if a column is relevant to the query using metadata-driven logic.

    Prevents hallucination by excluding columns that:
    - Are not mentioned in the question
    - Have metric/measure roles but query doesn't ask for that specific metric
    - Are numeric measure columns in count queries

    This is ROBUST because it uses:
    - Column metadata (role, description) from JSONL
    - Query type detection
    - Pattern matching (not hardcoded lists)

    Args:
        column: Column name
        table: Table name
        question: User's question
        qtype: Query type (lookup, aggregate, rank)
        metadata: Column metadata from retrieval

    Returns:
        True if column should be included in context, False otherwise
    """
    col_lower = column.lower()
    q_l = question.lower()
    table_lower = table.lower()

    # === RULE 1: Always include primary keys and identifying columns ===
    roles = _parse_str_or_list_lower(metadata.get("tags_role") or metadata.get("roles") or [])
    if any(r in RETRIEVAL_ID_ROLE_TAGS for r in roles):
        return True

    # === RULE 2: Always include if explicitly mentioned in question ===
    # Check column name (exact word match, not substring)
    # Use word boundary to avoid false positives like "products" matching "quantityInStock"
    q_words = set(q_l.split())
    if col_lower in q_words:
        return True

    # Also check with underscores removed (quantity_in_stock -> quantityinstock)
    col_no_underscore = col_lower.replace("_", "")
    if col_no_underscore in q_words:
        return True

    # Check column description (with word boundary matching)
    desc = str(metadata.get("description") or "").lower()
    # Extract key terms from description (first few words)
    desc_words = set(desc.split()[:RETRIEVAL_DESC_KEYWORD_HEAD_WORDS])
    # Only match if description words actually appear as whole words in question
    if desc_words.intersection(q_words):
        # But exclude generic terms that could cause false positives
        generic_terms = RETRIEVAL_DESC_GENERIC_TERMS
        # Only match if there's a specific term match (not just generic terms)
        specific_matches = desc_words.intersection(q_words) - generic_terms
        if specific_matches:
            return True

    # === RULE 3: For COUNT queries, exclude metric columns not mentioned ===
    if qtype == AGENT_ROUTER_QTYPE_AGGREGATE and "count" in q_l:
        # Count queries only need: PK, name columns, and explicitly mentioned columns
        # Exclude: quantityInStock, buyPrice, etc. unless mentioned
        is_metric = any(r in RETRIEVAL_METRIC_ROLE_TAGS for r in roles)

        # Check if column is a numeric measure
        datatype = (metadata.get("datatype") or "").lower()
        is_numeric = any(t in datatype for t in RETRIEVAL_NUMERIC_TYPE_HINTS)

        # Numeric column patterns (generic, not table-specific)
        numeric_patterns = RETRIEVAL_NUMERIC_PATTERN_KEYWORDS

        # Check if column name contains any numeric pattern
        has_numeric_pattern = any(p in col_lower for p in numeric_patterns)

        if is_metric or (is_numeric and has_numeric_pattern):
            # Only include if COLUMN NAME is explicitly mentioned in question
            # (not just generic numeric keywords like "total")
            # Check column name variants
            normalized_compound = col_lower
            for src, dst in RETRIEVAL_COLUMN_VARIANT_REPLACEMENTS.items():
                normalized_compound = normalized_compound.replace(src, dst)

            col_variants = [
                col_lower,                    # quantityinstock
                col_no_underscore,             # same for no underscore
                normalized_compound,           # configurable compound normalization
            ]
            mentioned_in_question = any(v in q_l for v in col_variants if v)

            if not mentioned_in_question:
                # DEBUG: Show why column is being excluded
                if RETRIEVE_DEBUG and col_lower in RETRIEVAL_DEBUG_FOCUS_COLUMNS:
                    print(f"   [FILTER-RULE3] Excluding {table}.{column}: is_metric={is_metric}, is_numeric={is_numeric}, has_pattern={has_numeric_pattern}")
                return False
            else:
                # DEBUG: Show why column is being kept
                if RETRIEVE_DEBUG and col_lower in RETRIEVAL_DEBUG_FOCUS_COLUMNS:
                    print(f"   [FILTER-RULE3] KEPT {table}.{column}: column name explicitly mentioned")

    # === RULE 4: For aggregate queries (SUM/AVG), include relevant metric columns ===
    if qtype == AGENT_ROUTER_QTYPE_AGGREGATE and any(kw in q_l for kw in RETRIEVAL_AGGREGATE_QUERY_TERMS):
        # These queries need numeric columns, but only relevant ones
        # Default: include (will be filtered by relevance scoring)
        return True

    # === RULE 5: For lookup queries, include filterable and display columns ===
    if qtype == AGENT_ROUTER_QTYPE_LOOKUP:
        # Name columns are always relevant
        if any(term in col_lower for term in RETRIEVAL_NAME_COLUMN_HINT_TERMS):
            return True
        # Include if table is mentioned
        if table_lower in q_l or get_singular_form(table_lower) in q_l:
            return True

    # Default: include (backward compatible)
    return True


# ==============================================================================
# 3. CORE LOGIC: CARD BUILDER
# ==============================================================================
def _build_cards(
    filtered_allowed: Dict[str, List[str]],
    reranked: List[Dict[str, Any]],
    max_cards: int = RETRIEVAL_CARDS_MAX_DOCS,
    max_inject: int = RETRIEVAL_CARDS_MAX_INJECT,
    desc_max_len: int = RETRIEVAL_CARDS_DESC_MAX_LEN,
    question: str = "",  # Add question parameter
    qtype: str = "",     # Add qtype parameter
) -> str:
    cards: List[str] = []
    present = set()
    
    # debug_count = 0
    if RETRIEVE_DEBUG:
        debug_count = 0 
        print(f"\n[DEBUG CARD PREVIEW] Menampilkan Top {RETRIEVAL_CARDS_DEBUG_PREVIEW_COUNT} Kartu untuk LLM:")

    for it in (reranked or [])[:max_cards]:
        md = it.get("meta") or {}
        table = _meta_table(md)
        column = _meta_column(md)

        if not table or not column:
            continue
        if table not in filtered_allowed:
            continue

        # === FILTER: Check if column is relevant to query (metadata-driven) ===
        # This prevents hallucination by excluding irrelevant columns
        # Only applies when question and qtype are provided
        if question and qtype:
            if not _is_column_relevant_to_query(column, table, question, qtype, md):
                # Skip this column - not relevant to the query
                if RETRIEVE_DEBUG:
                    print(f"   [FILTER] Excluding {table}.{column} (not relevant to query)")
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
                        # Fallback truncation length is configurable.
                        desc = doc_text[start:start + RETRIEVAL_DENSE_DESC_FALLBACK_LEN].strip()
                except:
                    pass

        # 3. PRIORITY 3: Fallback Regex (Manusiakan Nama Kolom)
        # Jaga-jaga jika Priority 1 & 2 gagal total
        if not desc:
            # "productCode" -> "product Code" -> "product code"
            s1 = re.sub(RETRIEVAL_DESC_CAMEL_SPLIT_PATTERN_1, r'\1 \2', column)
            s2 = re.sub(RETRIEVAL_DESC_CAMEL_SPLIT_PATTERN_2, r'\1 \2', s1).replace("_", " ")
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

        # Print debug preview rows.
        if RETRIEVE_DEBUG and debug_count < RETRIEVAL_CARDS_DEBUG_PREVIEW_COUNT:
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
_PHRASE_TO_COLUMN_CACHE: Optional[Dict[str, str]] = None

_NUMERIC_SQL_TYPES = set(RETRIEVAL_NUMERIC_SQL_TYPES)

# Stopwords for BM25 tokenization
_STOPWORDS = (set(ENGLISH_STOP_WORDS) - set(RETRIEVAL_BM25_STOPWORD_EXCEPTIONS)) | set(RETRIEVAL_EXTRA_STOPWORDS)

# Generic alias/synonym words that are too broad for table mention detection.
_GENERIC_MENTION_TERMS = set(RETRIEVAL_GENERIC_MENTION_TERMS)

try:
    _TABLE_SINGULAR_ROOTS = {
        get_singular_form(t.lower().strip())
        for t in get_all_tables()
        if t
    }
except Exception:
    _TABLE_SINGULAR_ROOTS = set()

def _table_mention_candidates(table: str, include_synonyms: bool = True) -> List[str]:
    t = (table or "").strip().lower()
    if not t:
        return []

    candidates = {t}
    singular = get_singular_form(t)
    if singular:
        candidates.add(singular.lower())
    if "_" in t:
        candidates.add(t.replace("_", " "))
    if singular and "_" in singular:
        candidates.add(singular.replace("_", " "))

    # Split concatenated table names (productlines -> product lines).
    for base in list(candidates):
        compact = base.replace(" ", "")
        if not compact.isalpha():
            continue
        for root in _TABLE_SINGULAR_ROOTS:
            if not root or root == compact:
                continue
            if compact.startswith(root):
                tail = compact[len(root):]
                if len(tail) >= RETRIEVAL_TABLE_VARIANT_MIN_TAIL_LEN and tail.isalpha():
                    candidates.add(f"{root} {tail}")

    if include_synonyms:
        for syn in get_table_synonyms(t):
            s = str(syn).strip().lower()
            if not s:
                continue
            tokens = [tok for tok in re.findall(RETRIEVAL_TOKENIZE_ALNUM_PATTERN, s) if tok]
            if not tokens:
                continue
            # Skip overly generic synonyms.
            if all(tok in _GENERIC_MENTION_TERMS for tok in tokens):
                continue
            candidates.add(" ".join(tokens))

    # Prioritize longer/more specific terms first.
    return sorted(candidates, key=len, reverse=True)

def _text_mentions_table(text: str, table: str, include_synonyms: bool = True) -> bool:
    q = (text or "").lower()
    if not q:
        return False
    for cand in _table_mention_candidates(table, include_synonyms=include_synonyms):
        pattern = re.escape(cand).replace(r"\ ", r"\s+")
        if re.search(rf"\b{pattern}\b", q):
            return True
    return False

def _normalize_text(txt: str) -> str:
    txt = txt.lower()
    txt = re.sub(RETRIEVAL_NORMALIZE_THOUSANDS_PATTERN, r"\1\2", txt)
    txt = re.sub(RETRIEVAL_NORMALIZE_POSSESSIVE_PATTERN, r"\1", txt)
    txt = txt.replace("'", "")
    txt = re.sub(RETRIEVAL_NORMALIZE_NON_ALNUM_PATTERN, " ", txt)
    return txt.strip()

def _tokenize_bm25_query(q: str) -> List[str]:
    s = _normalize_text(q)
    raw_tokens = s.split()
    toks: List[str] = []

    schema_tokens = set()
    try:
        for tbl in get_all_tables():
            t = str(tbl or "").strip().lower()
            if not t:
                continue
            schema_tokens.add(t)
            singular = get_singular_form(t)
            if singular:
                schema_tokens.add(str(singular).lower())
    except Exception:
        pass

    # Prevent over-stemming for generic nouns and schema entity words.
    STEM_EXCEPTIONS = set(RETRIEVAL_BM25_STEM_EXCEPTIONS_BASE) | schema_tokens

    for t in raw_tokens:
        if not t:
            continue

        if t in STEM_EXCEPTIONS:
            if t not in _STOPWORDS:
                toks.append(t)
            continue

        if RETRIEVAL_STEM_SUFFIX_IES and len(t) >= RETRIEVAL_STEM_SUFFIX_IES_MIN_LEN and t.endswith(RETRIEVAL_STEM_SUFFIX_IES):
            t = t[:-len(RETRIEVAL_STEM_SUFFIX_IES)] + RETRIEVAL_STEM_SUFFIX_IES_TO
        elif (
            RETRIEVAL_STEM_SUFFIX_S
            and len(t) >= RETRIEVAL_STEM_SUFFIX_S_MIN_LEN
            and t.endswith(RETRIEVAL_STEM_SUFFIX_S)
            and (not RETRIEVAL_STEM_SUFFIX_S_EXCLUDE or not t.endswith(RETRIEVAL_STEM_SUFFIX_S_EXCLUDE))
        ):
            t = t[:-1]

        if t and t not in _STOPWORDS:
            toks.append(t)

    return toks

def _phrase_to_column_map() -> Dict[str, str]:
    """
    Build phrase->column token map from schema metadata synonyms.

    Example:
      "list price" -> "msrp"

    This minimizes schema-specific hardcode by deriving aliases dynamically
    from JSONL metadata.
    """
    global _PHRASE_TO_COLUMN_CACHE
    if _PHRASE_TO_COLUMN_CACHE is not None:
        return _PHRASE_TO_COLUMN_CACHE

    mapping: Dict[str, str] = {}
    try:
        analyzer = get_schema_analyzer()
        for table in get_all_tables():
            for col in analyzer.get_table_columns(table):
                col_lower = str(col or "").strip().lower()
                if not col_lower:
                    continue
                for syn in get_column_synonyms(table, col):
                    s = str(syn or "").strip().lower()
                    if not s:
                        continue
                    tokens = [tok for tok in re.findall(RETRIEVAL_TOKENIZE_ALNUM_PATTERN, s) if tok]
                    # Keep only meaningful multi-word phrases to avoid noisy generic matches.
                    if len(tokens) < 2:
                        continue
                    if all(tok in _GENERIC_MENTION_TERMS for tok in tokens):
                        continue
                    phrase = " ".join(tokens)
                    mapping.setdefault(phrase, col_lower)
    except Exception:
        mapping = {}

    _PHRASE_TO_COLUMN_CACHE = mapping
    return mapping

def build_sparse_query_from_question(q: str) -> str:
    toks = _tokenize_bm25_query(q)
    q_l = (q or "").lower()
    # Dynamic phrase alias expansion from schema metadata (e.g., "list price" -> "msrp")
    for phrase, column_token in _phrase_to_column_map().items():
        pattern = re.escape(phrase).replace(r"\ ", r"\s+")
        if re.search(rf"\b{pattern}\b", q_l) and column_token not in toks:
            toks.append(column_token)
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

    # Column-level synonyms from schema metadata (dynamic, schema-agnostic)
    # Get synonyms from JSONL metadata for specific columns
    col_synonyms = get_column_synonyms(table, column)
    if col_synonyms:
        extra.extend(col_synonyms)

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
    """
    Apply column-specific priority boost based on query and column metadata.

    Dynamic version - uses table type and column patterns instead of hardcoded table names.
    """
    bonus = 0.0
    tl = table.lower()
    col_lower = col_name.lower()

    # Generic name boosting (table-agnostic)
    if any(k in q_l for k in RETRIEVAL_NAME_QUERY_TERMS):
        if (
            any(any(term in r for term in RETRIEVAL_NAME_ROLE_HINT_TERMS) for r in roles)
            or any(term in col_lower for term in RETRIEVAL_NAME_COLUMN_HINT_TERMS)
        ):
            bonus += NAME_QUERY_BONUS

    # Check if table is mentioned (dynamic - uses schema metadata)
    table_mentioned = _is_mentioned(q_l, tl)

    if table_mentioned:
        # Detect query intent: ID vs Name
        asks_id_like = any(k in q_l for k in RETRIEVAL_ID_LIKE_QUERY_TERMS)
        asks_for_names = any(k in q_l for k in RETRIEVAL_NAME_REQUEST_TERMS)

        # Detect if asking for "contact" specifically (for tables with contact columns)
        asks_contact_explicitly = any(k in q_l for k in RETRIEVAL_CONTACT_NAME_TERMS)

        # Dynamic column name pattern: {table}name (e.g., customername, productname)
        # Handle both concatenated and underscored forms
        table_prefix = (
            tl.rstrip(RETRIEVAL_TABLE_NAME_PREFIX_STRIP_CHARS)
            if RETRIEVAL_TABLE_NAME_PREFIX_STRIP_CHARS
            else tl
        )
        expected_name_cols = [
            f"{table_prefix}name",      # customername, productname
            f"{table_prefix}_name",     # customer_name, product_name
            f"{tl}name",                # customersname (less common)
            f"{tl}_name",               # customers_name (less common)
        ]

        # Dynamic ID column patterns
        is_id_column = (
            any(any(term in r for term in RETRIEVAL_ID_ROLE_HINT_TERMS) for r in roles) or
            any(col_lower.endswith(suffix) for suffix in RETRIEVAL_ID_COLUMN_SUFFIX_TERMS) or
            any(term in col_lower for term in RETRIEVAL_ID_COLUMN_SUBSTRING_TERMS)
        )

        # Dynamic contact column detection (for tables with contact info)
        is_contact_column = any(term in col_lower for term in RETRIEVAL_CONTACT_COLUMN_HINT_TERMS)

        # Apply table-type-based boosts (dynamic based on table type, not hardcoded names)
        table_type = get_table_type(tl)

        # High boost for dimension tables (customers, products, employees)
        if table_type in RETRIEVAL_DIMENSION_LIKE_TABLE_TYPES:
            if asks_for_names and not asks_contact_explicitly:
                # Boost {table}name columns
                if any(expected in col_lower for expected in expected_name_cols):
                    bonus += COL_BOOST_DIM_NAME_EXPLICIT
                # Penalize contact columns when asking for generic names
                elif is_contact_column:
                    bonus += COL_BOOST_DIM_CONTACT_PENALTY
            elif not asks_id_like:
                # Boost name columns when not asking for ID
                if any(expected in col_lower for expected in expected_name_cols):
                    bonus += COL_BOOST_DIM_NAME_DEFAULT
                # Penalize ID columns when asking for names
                if is_id_column:
                    bonus += COL_BOOST_DIM_ID_PENALTY

        # Medium boost for other table types (transaction, bridge, reference)
        else:
            if not asks_id_like:
                # Boost name columns
                if any(expected in col_lower for expected in expected_name_cols):
                    bonus += COL_BOOST_OTHER_NAME_DEFAULT
                # Penalize ID columns when asking for names
                if is_id_column:
                    bonus += COL_BOOST_OTHER_ID_PENALTY

    # Generic ID boosting (table-agnostic, already dynamic)
    if any(k in q_l for k in RETRIEVAL_ID_BOOST_QUERY_TERMS):
        if (
            any(any(term in r for term in RETRIEVAL_ID_ROLE_HINT_TERMS) for r in roles)
            or any(term in col_lower for term in RETRIEVAL_ID_BOOST_COLUMN_SUBSTRING_TERMS)
            or any(col_lower.endswith(suffix) for suffix in RETRIEVAL_ID_BOOST_COLUMN_SUFFIX_TERMS)
        ):
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

    if synonyms is not None:
        # Preserve compatibility for explicit synonym calls, but keep boundary-safe matching.
        for synonym in synonyms:
            s = str(synonym or "").strip().lower()
            if not s:
                continue
            pattern = re.escape(s).replace(r"\ ", r"\s+")
            if re.search(rf"\b{pattern}\b", q_l):
                return True

    # Default robust detection: strict table mentions + filtered synonyms.
    return _text_mentions_table(q_l, table, include_synonyms=True)


def score_tables_from_reranked(reranked: List[Dict[str, Any]], query_tokens: List[str]) -> Dict[str, float]:
    scores: Dict[str, float] = defaultdict(float)
    qt = [t.lower() for t in query_tokens]
    q_l = " ".join(qt)

    # Get all tables from schema (dynamic)
    all_tables = get_all_tables()
    bridge_tables = {
        str(t).strip().lower()
        for t in all_tables
        if str(t).strip() and get_table_type(str(t).strip().lower()) == "bridge"
    }

    # Build mentions map dynamically using schema metadata
    # Uses _is_mentioned() which checks table name + synonyms from JSONL (schema-agnostic)
    mentions_map = {tbl: _is_mentioned(q_l, tbl) for tbl in all_tables}

    # Detect aggregate context - indicates we need detail tables
    math_keywords = RETRIEVAL_TABLE_SCORING_MATH_KEYWORDS
    has_aggregation = any(kw in q_l for kw in math_keywords)

    # Check if we need numeric columns - based on RETRIEVED columns metadata (not hardcoded keywords)
    # Check if any retrieved column has numeric/aggregate characteristics
    needs_numeric_data = _check_needs_numeric_data_from_retrieved(reranked)

    # Penalize bridge tables by default in non-aggregate/no-numeric contexts,
    # unless bridge tables are explicitly mentioned.
    transaction_table_mentioned = any(
        mentions_map.get(tbl, False) for tbl in all_tables if is_transaction_table(tbl)
    )
    bridge_table_mentioned = any(
        mentions_map.get(tbl, False) for tbl in bridge_tables
    )
    penalize_bridge_default = (
        transaction_table_mentioned and
        not bridge_table_mentioned and
        not has_aggregation and
        not needs_numeric_data
    )

    for it in reranked:
        md = it.get("meta") or {}
        tbl = _meta_table(md)
        if not tbl: continue
        desc = (md.get("description") or "").lower()
        base = 0.0
        tl = tbl.lower()

        if it.get("sparse"): base += TABLE_SCORE_SPARSE_HIT
        if tl in q_l: base += TABLE_SCORE_TABLE_MENTION
        for t in qt:
            if t and t in desc: base += TABLE_SCORE_DESC_TOKEN

        # Dynamic table mention boost (instead of hardcoded "customers", "products", etc.)
        # Uses mentions_map built from schema metadata
        if tl in mentions_map and mentions_map[tl]:
            # High boost for dimension/master tables (customers, products)
            if get_table_type(tl) in ("dimension", "unknown"):
                base += TABLE_SCORE_MENTION_DIM
            # Medium boost for transaction tables (orders, payments)
            elif is_transaction_table(tl):
                base += TABLE_SCORE_MENTION_TXN
            # High boost for bridge/lookup tables (offices, productlines, orderdetails, employees)
            else:
                base += TABLE_SCORE_MENTION_OTHER

        # Generic bridge-table handling (schema-agnostic).
        if penalize_bridge_default and tl in bridge_tables:
            base += TABLE_SCORE_BRIDGE_PENALTY

        if needs_numeric_data and tl in bridge_tables:
            base += TABLE_SCORE_BRIDGE_NUMERIC_BOOST

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
    for item in reranked[:MAX_CTX_DOCS]:
        md = item.get("meta") or {}

        # Primary: Check datatype field from metadata
        datatype = (md.get("datatype") or md.get("data_type") or "").strip().lower()
        if datatype:
            # Remove any size specification (e.g., "decimal(10,2)" -> "decimal")
            base_type = datatype.split('(')[0].strip()
            if base_type in _NUMERIC_SQL_TYPES:
                # Check if this numeric column is used for calculations
                # (not just IDs or counts)
                col = _meta_column(md).lower()
                # Exclude primary keys and foreign keys (IDs) as they're not used for calculations
                desc = str(md.get("description") or "").lower()
                if not any(kw in col for kw in RETRIEVAL_NUMERIC_EXCLUDE_COLUMN_KEYWORDS):
                    return True
                # Check description for calculation-related terms
                if any(kw in desc for kw in RETRIEVAL_NUMERIC_SIGNAL_KEYWORDS):
                    return True

        # Fallback: Check column name pattern for numeric-sounding columns
        # (only if datatype check above didn't trigger)
        col = _meta_column(md).lower()
        if any(kw in col for kw in RETRIEVAL_NUMERIC_SIGNAL_KEYWORDS):
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
        fk_match = re.search(RETRIEVAL_FK_REFERENCE_PATTERN, desc, re.IGNORECASE)
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
                is_numeric = base_type in _NUMERIC_SQL_TYPES
            else:
                # Fallback: Check column name pattern
                is_numeric = any(kw in col_lower for kw in RETRIEVAL_NUMERIC_SIGNAL_KEYWORDS)

            # Exclude ID columns from numeric consideration
            if is_numeric and not any(kw in col_lower for kw in RETRIEVAL_NUMERIC_EXCLUDE_COLUMN_KEYWORDS):
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
    max_depth: int = RETRIEVAL_FK_PATH_MAX_DEPTH,
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
    if any(s in q for s in RETRIEVAL_JOIN_SIGNALS):
        return True

    # Generic multi-entity signal: 2+ explicit table mentions.
    mentioned_tables = [
        t for t in get_all_tables()
        if _text_mentions_table(q, t, include_synonyms=False)
    ]
    if len(mentioned_tables) >= 2:
        return True

    return False


def _wants_name_only_for_table(question: str, table: str) -> bool:
    """
    Generic replacement for table-specific name-only heuristics.
    """
    q = (question or "").lower()
    t = (table or "").strip().lower()
    if not q or not t:
        return False

    if not _text_mentions_table(q, t, include_synonyms=True):
        return False

    # Avoid treating "number of" as ID intent.
    q_clean = re.sub(RETRIEVAL_NAME_ONLY_NUMBER_OF_PATTERN, "", q)
    asks_id_like = bool(re.search(RETRIEVAL_NAME_ONLY_ID_PATTERN, q_clean))
    asks_name_like = bool(re.search(RETRIEVAL_NAME_ONLY_NAME_PATTERN, q_clean))
    return (asks_name_like or not asks_id_like) and not asks_id_like

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
    if not q:
        return None

    # Prefer explicit table-name mentions (no broad synonyms) for single-table lock.
    all_tables = sorted(get_all_tables(), key=len, reverse=True)
    for table in all_tables:
        if _text_mentions_table(q, table, include_synonyms=False):
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
    return _text_mentions_table(question.lower(), table, include_synonyms=False)


def _should_lock_single_table_ctx(question: str, qtype: str) -> bool:
    if qtype not in {AGENT_ROUTER_QTYPE_LOOKUP, AGENT_ROUTER_QTYPE_AGGREGATE, AGENT_ROUTER_QTYPE_RANK}: return False
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

    primary_lookup_table = _detect_primary_entity_table(q) if qtype == AGENT_ROUTER_QTYPE_LOOKUP else None
    prefer_name_table = (
        primary_lookup_table
        if _wants_name_only_for_table(q, primary_lookup_table or "")
        else ""
    )

    # --- [MODIFIED 2] Filter Paths: Hanya ambil file yang namanya mengandung "_sparse" ---
    all_paths = state.get("jsonl_paths", [])
    sparse_target_paths = [p for p in all_paths if RETRIEVAL_SPARSE_FILENAME_MARKER in str(p)]
    
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

    # Get preferred display-name column for lookup contexts (schema-agnostic).
    preferred_name_col = get_name_column(prefer_name_table) if prefer_name_table else None
    preferred_name_col_lower = (preferred_name_col or "").lower()

    sparse_has_preferred_name = any(
        ((_meta_table(m) or "").lower() == prefer_name_table) and ((_meta_column(m) or "").lower() == preferred_name_col_lower)
        for m in sparse_metas if isinstance(m, dict)
    ) if preferred_name_col else False

    base_k = max(RETRIEVAL_BASE_K_FLOOR, TOP_K)
    dense_k = base_k
    if prefer_name_table and preferred_name_col and not sparse_has_preferred_name:
        dense_k = max(RETRIEVAL_DENSE_K_FLOOR, base_k)

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

    dense_has_preferred_name = any(
        ((_meta_table(md) or "").lower() == prefer_name_table) and ((_meta_column(md) or "").lower() in (preferred_name_col_lower, preferred_name_col_lower.replace("n", "_n")))
        for md in dense_metas
    ) if preferred_name_col else False

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
    # 5) Auto-inject sangat terbatas (preferred display-name column)
    # -----------------------------
    if prefer_name_table and preferred_name_col and sparse_has_preferred_name and not dense_has_preferred_name:
        sparse_preferred_name_urn = None
        urn_suffix = f"::{preferred_name_col_lower}"
        for u in sparse_field_urns:
            if u.lower().endswith(urn_suffix):
                sparse_preferred_name_urn = u
                break
        if sparse_preferred_name_urn:
            present_ids = {item["id"] for item in scored if item.get("id")}
            if sparse_preferred_name_urn not in present_ids:
                got = col.get(ids=[sparse_preferred_name_urn])
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
                        "rank": len(scored), "id": sparse_preferred_name_urn, "doc": doc, "meta": md or {},
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
    # 8) Build allowed + alias_map (with column filtering)
    # -----------------------------
    allowed = defaultdict(list)
    alias_map = defaultdict(dict)

    for item in reranked:
        md = item["meta"] or {}
        t = _meta_table(md)
        c = _meta_column(md)
        if not (t and c): continue

        # === FILTER: Check if column is relevant to query (metadata-driven) ===
        # Apply the same filtering logic here to prevent irrelevant columns
        # from appearing in ALLOWED IDENTIFIERS block
        if q and qtype:
            is_relevant = _is_column_relevant_to_query(c, t, q, qtype, md)
            if not is_relevant:
                # Skip this column - not relevant to the query
                if RETRIEVE_DEBUG:
                    print(f"   [FILTER-ALLOWED] Excluding {t}.{c} from allowed identifiers")
                continue
            elif RETRIEVE_DEBUG and c.lower() in RETRIEVAL_DEBUG_FOCUS_COLUMNS:
                # Debug: show why numeric columns passed the filter
                print(f"   [FILTER-ALLOWED] KEPT {t}.{c} - passed relevance check")

        t_key = t.strip().lower()
        c_key = c.strip()
        allowed[t_key].append(c_key)
        aliases2 = _extract_aliases_for_map(md)
        for al in aliases2:
            al = al.strip()
            if al: alias_map[t_key][al] = c_key

    # Robustness guard:
    # for single-table queries, ensure primary table columns from sparse hits are retained
    # so dense noise cannot drop obvious relevant columns (e.g., quantityInStock).
    lock_single = _should_lock_single_table_ctx(question=q, qtype=qtype)
    primary_for_lock = _detect_primary_entity_table(q) if lock_single else None
    if primary_for_lock:
        sparse_primary_cols = {
            (_meta_column(md) or "").strip()
            for md in sparse_metas
            if (_meta_table(md) or "").strip().lower() == primary_for_lock and (_meta_column(md) or "").strip()
        }
        if sparse_primary_cols:
            for c in sorted(sparse_primary_cols):
                if c not in allowed[primary_for_lock]:
                    allowed[primary_for_lock].append(c)

        # For lookup-style prompts, keep the entity's ID/name columns if available.
        if qtype == AGENT_ROUTER_QTYPE_LOOKUP:
            for c in [get_id_column(primary_for_lock), get_name_column(primary_for_lock)]:
                if c and c not in allowed[primary_for_lock]:
                    allowed[primary_for_lock].append(c)

    # -----------------------------
    # 9) Table scoring + filtered_allowed
    # -----------------------------
    token_for_score = _tokenize_bm25_query(sparse_query_clean)
    table_scores = score_tables_from_reranked(reranked, token_for_score)

    if qtype == AGENT_ROUTER_QTYPE_LOOKUP and prefer_name_table:
        only_preferred = {t: cols for t, cols in allowed.items() if t.lower() == prefer_name_table}
        filtered_allowed = only_preferred if only_preferred else dict(allowed)
    else:
        if lock_single:
            primary = primary_for_lock
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
                max_top_tables = MAX_TOP_TABLES
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

    if prefer_name_table and preferred_name_col:
        for t_key in list(filtered_allowed.keys()):
            if t_key.lower() == prefer_name_table:
                if preferred_name_col not in filtered_allowed[t_key]:
                    filtered_allowed[t_key].append(preferred_name_col)

    # -----------------------------
    # 10) Context build
    # -----------------------------
    allowed_block = _build_allowed_block(filtered_allowed)
    cards_text = _build_cards(
        filtered_allowed,
        reranked,
        max_cards=RETRIEVAL_CARDS_MAX_DOCS,
        max_inject=RETRIEVAL_CARDS_MAX_INJECT,
        desc_max_len=RETRIEVAL_CARDS_DESC_MAX_LEN,
        question=q,      # Pass question for column filtering
        qtype=qtype,     # Pass qtype for column filtering
    )

    context = (allowed_block + "\n\n### CARDS\n" + cards_text).strip()
    context += f"\n\n### _INTERNAL_HINT\n{json.dumps({'alias_map': alias_map}, ensure_ascii=False)}"

    # DEBUG: Show key info about context being built
    if RETRIEVE_DEBUG:
        print(f"\n[DEBUG] Building context with {len(filtered_allowed)} tables")
        for tbl, cols in filtered_allowed.items():
            print(f"[DEBUG] {tbl}: {len(cols)} columns")

    state["context"] = context
    return state
