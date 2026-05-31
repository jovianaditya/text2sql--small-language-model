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
import re
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Set
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
    RETRIEVAL_STATS_KEY_MIN,
    RETRIEVAL_STATS_KEY_MAX,
    RETRIEVAL_STATS_KEY_MEAN,
    RETRIEVAL_STATS_KEY_DISTINCT_COUNT,
    RETRIEVAL_META_DESCRIPTION_KEY,
    RETRIEVAL_META_KEYWORDS_KEY,
    RETRIEVAL_DESC_KEYWORD_HEAD_WORDS,
    RETRIEVAL_CARDS_MAX_DOCS,
    RETRIEVAL_CARDS_MAX_INJECT,
    RETRIEVAL_CARDS_DESC_MAX_LEN,
    RETRIEVAL_CARDS_DEBUG_PREVIEW_COUNT,
    RETRIEVAL_SPARSE_DEBUG_PREVIEW_LIMIT,
    RETRIEVAL_RETRIEVE_DEBUG_MAX_SHOW,
    RETRIEVAL_AUTO_INJECT_BONUS_DELTA,
    RETRIEVAL_TOP_TABLE_MIN_SCORE,
    RETRIEVAL_BRIDGE_MIN_TABLE_COUNT,
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
    RETRIEVAL_COUNT_QUERY_TERMS,
    RETRIEVAL_MULTI_ENTITY_MIN_TABLES,
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
    RETRIEVAL_FK_REFERENCE_PATTERN,
    RETRIEVAL_SINGLE_LOCK_MIN_SCORE_GAP,
    RETRIEVAL_SINGLE_LOCK_MAX_NON_PRIMARY_SPARSE_HITS,
    RETRIEVAL_SINGLE_LOCK_FALLBACK_TOPK,
    RETRIEVAL_EXPAND_MAX_TABLES,
    RETRIEVAL_TELEMETRY_MAX_ITEMS,
    AGENT_ROUTER_QTYPE_LOOKUP,
    AGENT_ROUTER_QTYPE_AGGREGATE,
    SPARSE_METADATA_JSONL,
)
from .state import RAGState
from .chroma_embedding import build_vector_store
from .utils import _print_banner, _meta_table, _meta_column, read_jsonl, _parse_str_or_list_lower, _extract_aliases_for_map
from .helpers_main import _get_qtype, build_dynamic_probe
from .rulebook import question_has_hierarchy_role_intent
from .schema_analyzer import (
    get_schema_analyzer,
    get_all_tables,
    get_table_synonyms,
    get_singular_form,
    get_table_type,
    is_transaction_table,
    get_column_synonyms,
    get_column_entity_variants,
    get_column_description,
    get_column_datatype,
    get_column_primary_role,
    get_column_role_tags,
    get_column_alias_tags,
    get_column_stats,
    get_name_column,
    get_id_column,
    get_fk_relationships,
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


def _parse_allowed_block(ctx: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    text = str(ctx or "")
    m = re.search(r"### ALLOWED IDENTIFIERS(.*?)(?:\n### |\Z)", text, flags=re.S)
    if not m:
        return out
    for raw in (m.group(1) or "").splitlines():
        line = raw.strip()
        if not line.startswith("- "):
            continue
        body = line[2:]
        parts = body.split(":", 1)
        table = str(parts[0] if parts else "").strip().strip("`").lower()
        if not table:
            continue
        cols: List[str] = []
        if len(parts) == 2:
            cols = [
                str(c or "").strip().strip("`")
                for c in parts[1].split(",")
                if str(c or "").strip()
            ]
        out[table] = cols
    return out


def _extract_cards_and_hint(ctx: str) -> tuple[str, str]:
    text = str(ctx or "")
    cards_text = ""
    hint_text = ""

    cards_marker = "\n\n### CARDS\n"
    hint_marker = "\n\n### _INTERNAL_HINT\n"

    cards_idx = text.find(cards_marker)
    hint_idx = text.find(hint_marker)

    if cards_idx >= 0:
        cards_start = cards_idx + len(cards_marker)
        if hint_idx > cards_start:
            cards_text = text[cards_start:hint_idx].strip()
        else:
            cards_text = text[cards_start:].strip()

    if hint_idx >= 0:
        hint_start = hint_idx + len(hint_marker)
        hint_text = text[hint_start:].strip()

    return cards_text, hint_text


def _normalize_stats_dict(stats_raw: Any) -> Dict[str, Any]:
    if isinstance(stats_raw, dict):
        return dict(stats_raw)
    if isinstance(stats_raw, str):
        try:
            parsed = json.loads(stats_raw)
        except Exception:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _merge_unique_lower(values: List[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for value in values or []:
        s = str(value or "").strip().lower()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _column_structured_metadata(table: str, column: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    md = metadata if isinstance(metadata, dict) else {}
    nested_meta = md.get("metadata") if isinstance(md.get("metadata"), dict) else {}
    table_l = str(table or "").strip().lower()
    col_s = str(column or "").strip()

    raw_roles = _parse_str_or_list_lower(
        md.get("tags_role")
        or md.get("roles")
        or nested_meta.get("tags_role")
        or nested_meta.get("roles")
        or []
    )
    raw_aliases = _parse_str_or_list_lower(
        md.get("tags_alias")
        or md.get("aliases")
        or md.get("synonyms")
        or nested_meta.get("tags_alias")
        or nested_meta.get("aliases")
        or nested_meta.get("synonyms")
        or []
    )

    raw_primary_role = str(
        nested_meta.get("primary_role")
        or md.get("primary_role")
        or ""
    ).strip().lower()
    primary_role = raw_primary_role
    role_tags: List[str] = []
    alias_tags: List[str] = []
    stats = _normalize_stats_dict(nested_meta.get("stats") or md.get("stats"))
    description = str(
        nested_meta.get("description")
        or md.get(RETRIEVAL_META_DESCRIPTION_KEY)
        or md.get("description")
        or ""
    ).strip()
    datatype = str(
        nested_meta.get("datatype")
        or md.get("datatype")
        or md.get("data_type")
        or ""
    ).strip()

    if table_l and col_s:
        try:
            schema_primary_role = str(get_column_primary_role(table_l, col_s) or "").strip().lower()
        except Exception:
            schema_primary_role = ""
        if schema_primary_role:
            primary_role = schema_primary_role
        try:
            role_tags = _parse_str_or_list_lower(get_column_role_tags(table_l, col_s) or [])
        except Exception:
            role_tags = []
        try:
            alias_tags = _parse_str_or_list_lower(get_column_alias_tags(table_l, col_s) or [])
        except Exception:
            alias_tags = []
        try:
            schema_stats = _normalize_stats_dict(get_column_stats(table_l, col_s) or {})
        except Exception:
            schema_stats = {}
        if schema_stats:
            stats.update({k: v for k, v in schema_stats.items() if v not in (None, "")})
        try:
            schema_desc = str(get_column_description(table_l, col_s) or "").strip()
        except Exception:
            schema_desc = ""
        if schema_desc:
            description = schema_desc
        try:
            schema_datatype = str(get_column_datatype(table_l, col_s) or "").strip()
        except Exception:
            schema_datatype = ""
        if schema_datatype:
            datatype = schema_datatype

    roles = _merge_unique_lower(([primary_role] if primary_role else []) + role_tags + raw_roles)
    aliases = _merge_unique_lower(alias_tags + raw_aliases)
    return {
        "primary_role": primary_role,
        "roles": roles,
        "aliases": aliases,
        "stats": stats,
        "description": description,
        "datatype": datatype,
    }


def _compute_retrieval_evidence_strength(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute lightweight retrieval evidence strength from telemetry counters.
    Higher score indicates stronger retrieval support for schema linking.
    """
    sparse = int(snapshot.get("sparse_candidates", 0) or 0)
    dense = int(snapshot.get("dense_candidates", 0) or 0)
    reranked = int(snapshot.get("reranked_candidates", 0) or 0)
    allowed_tables = int(snapshot.get("allowed_tables_after_filter", 0) or 0)
    allowed_cols = int(snapshot.get("allowed_columns_after_filter", 0) or 0)
    cols_before = int(snapshot.get("allowed_columns_before_filter", 0) or 0)

    sparse_signal = min(1.0, float(sparse) / 3.0)
    dense_signal = min(1.0, float(dense) / 12.0)
    rerank_signal = min(1.0, float(reranked) / 6.0)
    table_signal = min(1.0, float(allowed_tables) / 2.0)
    col_signal = min(1.0, float(allowed_cols) / 8.0)
    retention_signal = (
        min(1.0, float(allowed_cols) / float(cols_before))
        if cols_before > 0
        else 1.0
    )
    dual_channel_signal = 1.0 if sparse > 0 and dense > 0 else 0.0

    score = (
        0.20 * sparse_signal
        + 0.15 * dense_signal
        + 0.25 * rerank_signal
        + 0.10 * table_signal
        + 0.15 * col_signal
        + 0.10 * retention_signal
        + 0.05 * dual_channel_signal
    )
    score = max(0.0, min(1.0, float(score)))

    if score >= 0.75:
        tier = "strong"
    elif score >= 0.45:
        tier = "moderate"
    else:
        tier = "weak"

    return {
        "score": round(score, 3),
        "tier": tier,
        "signals": {
            "sparse": round(sparse_signal, 3),
            "dense": round(dense_signal, 3),
            "reranked": round(rerank_signal, 3),
            "allowed_tables": round(table_signal, 3),
            "allowed_columns": round(col_signal, 3),
            "column_retention": round(retention_signal, 3),
            "dual_channel": round(dual_channel_signal, 3),
        },
    }


def _build_expanded_cards_from_schema(
    expanded_allowed: Dict[str, List[str]],
    *,
    state: RAGState,
    analyzer: Any,
    max_cards: int,
    desc_max_len: int,
) -> str:
    """
    Regenerate cards for expanded context from schema metadata + ranked column hints.
    """
    if not expanded_allowed:
        return ""

    ranked_cols = state.get("ranked_cols") or []
    ranked_by_table: Dict[str, List[str]] = defaultdict(list)
    for tbl, col in ranked_cols:
        t = str(tbl or "").strip().lower()
        c = str(col or "").strip()
        if not t or not c:
            continue
        bucket = ranked_by_table[t]
        if c not in bucket:
            bucket.append(c)

    cards: List[str] = []
    for table in expanded_allowed.keys():
        allowed_cols = [str(c).strip() for c in (expanded_allowed.get(table) or []) if str(c).strip()]
        if not allowed_cols:
            continue
        ranked_first = [c for c in ranked_by_table.get(table, []) if c in allowed_cols]
        candidate_cols = ranked_first + [c for c in allowed_cols if c not in set(ranked_first)]
        for col in candidate_cols:
            desc = ""
            if analyzer is not None:
                try:
                    desc = str(analyzer.get_column_description(table, col) or "").strip()
                except Exception:
                    desc = ""
            if desc and len(desc) > desc_max_len:
                desc = desc[:desc_max_len].rstrip() + "..."
            cards.append(f"- table: {table} | column: {col} | desc: {desc or '(no description)'}")
            if len(cards) >= max_cards:
                return "\n".join(cards)

    return "\n".join(cards)


def _build_expanded_fk_hints(selected_tables: List[str]) -> str:
    """
    Build compact FK connector hints for expanded context.
    """
    table_scope = {str(t or "").strip().lower() for t in (selected_tables or []) if str(t or "").strip()}
    if len(table_scope) < 2:
        return ""

    try:
        fk_map = get_fk_relationships() or {}
    except Exception:
        fk_map = {}

    edges: List[str] = []
    seen: Set[str] = set()
    for src, rels in (fk_map or {}).items():
        src_l = str(src or "").strip().lower()
        if not src_l or src_l not in table_scope:
            continue
        for fk_col, tgt in (rels or {}).items():
            tgt_l = str(tgt or "").strip().lower()
            fk_l = str(fk_col or "").strip()
            if not tgt_l or not fk_l or tgt_l not in table_scope:
                continue
            tgt_id = (get_id_column(tgt_l) or "").strip("`")
            edge = f"- {src_l}.{fk_l} -> {tgt_l}.{tgt_id}" if tgt_id else f"- {src_l}.{fk_l} -> {tgt_l}"
            if edge in seen:
                continue
            seen.add(edge)
            edges.append(edge)
            if len(edges) >= 12:
                break
        if len(edges) >= 12:
            break

    if not edges:
        return ""
    return "JOIN KEY HINTS:\n" + "\n".join(edges)


def build_expanded_context_from_state(state: RAGState) -> Dict[str, Any]:
    """
    Build a broader ALLOWED IDENTIFIERS context from current retrieval state.
    Used as an optional fallback when acceptance fails due to tight context.
    """
    current_ctx = str(state.get("context") or "")
    allowed_current = _parse_allowed_block(current_ctx)
    if not allowed_current:
        return {
            "used": False,
            "reason": "no_allowed_identifiers",
            "context": current_ctx,
            "expanded_table_count": 0,
        }

    table_order: List[str] = []
    seen: set[str] = set()

    for t in allowed_current.keys():
        tl = str(t or "").strip().lower()
        if tl and tl not in seen:
            seen.add(tl)
            table_order.append(tl)

    for tbl, _ in (state.get("ranked_cols") or []):
        tl = str(tbl or "").strip().lower()
        if tl and tl not in seen:
            seen.add(tl)
            table_order.append(tl)

    max_tables = max(1, int(RETRIEVAL_EXPAND_MAX_TABLES))
    selected_tables = table_order[:max_tables]

    # Add FK connector tables so expanded context can preserve joinability.
    if len(selected_tables) >= 2 and len(selected_tables) < max_tables:
        fk_graph: Dict[str, Dict[str, str]] = {}
        try:
            fk_graph = get_fk_relationships() or {}
        except Exception:
            fk_graph = {}
        try:
            all_tables = get_all_tables() or []
        except Exception:
            all_tables = []
        allowed_universe = {
            str(t or "").strip().lower(): []
            for t in all_tables
            if str(t or "").strip()
        }
        if fk_graph and allowed_universe:
            connector_tables: List[str] = []
            for i, left in enumerate(selected_tables):
                for right in selected_tables[i + 1:]:
                    path = _find_fk_path(left, right, fk_graph, allowed_universe)
                    if not path:
                        continue
                    for bridge in path[1:-1]:
                        bridge_l = str(bridge or "").strip().lower()
                        if bridge_l and bridge_l not in seen and bridge_l not in connector_tables:
                            connector_tables.append(bridge_l)
                if len(selected_tables) >= max_tables:
                    break
            for bridge in connector_tables:
                if len(selected_tables) >= max_tables:
                    break
                if bridge not in seen:
                    seen.add(bridge)
                    selected_tables.append(bridge)

    analyzer = None
    try:
        analyzer = get_schema_analyzer()
    except Exception:
        analyzer = None

    expanded_allowed: Dict[str, List[str]] = {}
    for table in selected_tables:
        cols: List[str] = []
        if analyzer is not None:
            try:
                cols = [str(c).strip() for c in (analyzer.get_table_columns(table) or []) if str(c).strip()]
            except Exception:
                cols = []
        if not cols:
            cols = [str(c).strip() for c in (allowed_current.get(table) or []) if str(c).strip()]
        if cols:
            expanded_allowed[table] = sorted(set(cols))

    if not expanded_allowed:
        return {
            "used": False,
            "reason": "no_expandable_tables",
            "context": current_ctx,
            "expanded_table_count": 0,
        }

    _, hint_text = _extract_cards_and_hint(current_ctx)
    allowed_block = _build_allowed_block(expanded_allowed)
    cards_block = _build_expanded_cards_from_schema(
        expanded_allowed,
        state=state,
        analyzer=analyzer,
        max_cards=RETRIEVAL_CARDS_MAX_DOCS,
        desc_max_len=RETRIEVAL_CARDS_DESC_MAX_LEN,
    )
    fk_hints = _build_expanded_fk_hints(list(expanded_allowed.keys()))
    if fk_hints:
        cards_block = (cards_block + "\n\n" + fk_hints).strip() if cards_block else fk_hints
    if not cards_block:
        cards_block = "(expanded cards unavailable)"
    new_ctx = (allowed_block + "\n\n### CARDS\n" + cards_block).strip()
    if hint_text:
        new_ctx += f"\n\n### _INTERNAL_HINT\n{hint_text}"

    return {
        "used": True,
        "reason": "expanded_allowed_identifiers",
        "context": new_ctx,
        "expanded_table_count": len(expanded_allowed),
        "tables": sorted(expanded_allowed.keys()),
    }
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
    if RETRIEVAL_STATS_KEY_MIN in stats and RETRIEVAL_STATS_KEY_MAX in stats:
        try:
            parts.append(f"Range=[{stats[RETRIEVAL_STATS_KEY_MIN]}..{stats[RETRIEVAL_STATS_KEY_MAX]}]")
        except: pass
    
    if RETRIEVAL_STATS_KEY_MEAN in stats:
        try:
            parts.append(f"Mean={float(stats[RETRIEVAL_STATS_KEY_MEAN]):.2f}")
        except: pass

    # C. DISTINCT COUNT (Cardinality)
    if RETRIEVAL_STATS_KEY_DISTINCT_COUNT in stats:
        parts.append(f"Distinct={stats[RETRIEVAL_STATS_KEY_DISTINCT_COUNT]}")

    if not parts:
        return ""
    
    return " | STATS: " + ", ".join(parts)
# ==============================================================================
# 2. CORE LOGIC: COLUMN RELEVANCE FILTERING (Metadata-Driven)
# ==============================================================================
def _is_likely_filtered_requested_metric(
    column: str,
    table: str,
    question: str,
    qtype: str,
    metadata: Dict[str, Any],
) -> bool:
    """
    Log-only diagnostic helper.
    Returns True when a filtered column looks like a requested metric in a mixed
    count + metric aggregate question.
    """
    if qtype != AGENT_ROUTER_QTYPE_AGGREGATE:
        return False

    q_l = str(question or "").lower()
    q_words = set(_normalize_text(question).split())
    if not q_l or not q_words:
        return False

    has_count_intent = any(k in q_l for k in RETRIEVAL_COUNT_QUERY_TERMS)
    has_non_count_agg_intent = bool(
        re.search(r"\b(sum|avg|average|mean|min|max|minimum|maximum)\b", q_l, flags=re.I)
    )
    has_total_metric_intent = "total" in q_words and bool(q_words.intersection(RETRIEVAL_NUMERIC_SIGNAL_KEYWORDS))
    if not (has_count_intent and (has_non_count_agg_intent or has_total_metric_intent)):
        return False

    col_meta = _column_structured_metadata(table, column, metadata)
    roles = list(col_meta.get("roles") or [])
    is_metric_role = any(r in RETRIEVAL_METRIC_ROLE_TAGS for r in roles)
    datatype = str(col_meta.get("datatype") or "").lower()
    is_numeric = any(t in datatype for t in RETRIEVAL_NUMERIC_TYPE_HINTS)
    col_lower = str(column or "").strip().lower()
    has_numeric_pattern = any(p in col_lower for p in RETRIEVAL_NUMERIC_PATTERN_KEYWORDS)
    if not (is_metric_role or (is_numeric and has_numeric_pattern)):
        return False

    mention_tokens: Set[str] = set()
    mention_tokens.update(tok for tok in re.findall(RETRIEVAL_TOKENIZE_ALNUM_PATTERN, col_lower) if tok)

    try:
        for v in get_column_entity_variants(table, column):
            mention_tokens.update(tok for tok in re.findall(RETRIEVAL_TOKENIZE_ALNUM_PATTERN, str(v).lower()) if tok)
    except Exception:
        pass
    try:
        for syn in get_column_synonyms(table, column) or []:
            mention_tokens.update(tok for tok in re.findall(RETRIEVAL_TOKENIZE_ALNUM_PATTERN, str(syn).lower()) if tok)
    except Exception:
        pass

    alias_variants = list(col_meta.get("aliases") or [])
    for alias in alias_variants:
        mention_tokens.update(tok for tok in re.findall(RETRIEVAL_TOKENIZE_ALNUM_PATTERN, str(alias).lower()) if tok)

    cleaned_tokens = {
        t for t in mention_tokens
        if t and t not in _GENERIC_MENTION_TERMS and t not in RETRIEVAL_DESC_GENERIC_TERMS
    }
    return bool(cleaned_tokens.intersection(q_words))


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
    q_norm = _normalize_text(question)
    q_words = set(q_norm.split())
    table_lower = table.lower()
    col_meta = _column_structured_metadata(table, column, metadata)
    has_count_intent = any(k in q_l for k in RETRIEVAL_COUNT_QUERY_TERMS)
    # Mixed aggregate intent should keep metric columns (e.g., "count and average amount").
    has_non_count_agg_intent = bool(
        re.search(r"\b(sum|avg|average|mean|min|max|minimum|maximum)\b", q_l, flags=re.I)
    )
    has_total_metric_intent = "total" in q_words and bool(q_words.intersection(RETRIEVAL_NUMERIC_SIGNAL_KEYWORDS))
    mixed_count_metric_intent = has_count_intent and (has_non_count_agg_intent or has_total_metric_intent)

    def _normalized_phrase(text: str) -> str:
        return " ".join(_normalize_text(text).split())

    def _question_mentions_variant(variant: str) -> bool:
        norm = _normalized_phrase(variant)
        if not norm:
            return False
        parts = norm.split()
        if not parts:
            return False
        if len(parts) == 1:
            return parts[0] in q_words
        pattern = re.escape(norm).replace(r"\ ", r"\s+")
        return bool(re.search(rf"\b{pattern}\b", q_norm))

    def _column_variants() -> Set[str]:
        variants: Set[str] = set()
        try:
            variants.update(get_column_entity_variants(table, column))
        except Exception:
            pass

        for alias in list(col_meta.get("aliases") or []):
            alias_s = str(alias or "").strip()
            if alias_s:
                variants.add(alias_s)

        normalized_variants: Set[str] = set()
        for variant in variants:
            norm = _normalized_phrase(variant)
            if not norm:
                continue
            toks = norm.split()
            if len(toks) > 1 or (toks and toks[0] not in _GENERIC_MENTION_TERMS):
                normalized_variants.add(norm)
        return normalized_variants

    explicit_column_match = any(_question_mentions_variant(v) for v in _column_variants())

    # === RULE 1: Always include primary keys and identifying columns ===
    roles = list(col_meta.get("roles") or [])
    if any(r in RETRIEVAL_ID_ROLE_TAGS for r in roles):
        return True

    # === RULE 2: Always include if explicitly mentioned in question ===
    # This now uses schema-driven column variants + metadata aliases so unseen
    # paraphrases can survive the late relevance filter too.
    if explicit_column_match:
        return True

    # Check column description (with word boundary matching)
    desc = str(col_meta.get("description") or "").lower()
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
    if qtype == AGENT_ROUTER_QTYPE_AGGREGATE and has_count_intent and not mixed_count_metric_intent:
        # Count queries only need: PK, name columns, and explicitly mentioned columns
        # Exclude: quantityInStock, buyPrice, etc. unless mentioned
        is_metric = any(r in RETRIEVAL_METRIC_ROLE_TAGS for r in roles)

        # Check if column is a numeric measure
        datatype = str(col_meta.get("datatype") or "").lower()
        is_numeric = any(t in datatype for t in RETRIEVAL_NUMERIC_TYPE_HINTS)

        # Numeric column patterns (generic, not table-specific)
        numeric_patterns = RETRIEVAL_NUMERIC_PATTERN_KEYWORDS

        # Check if column name contains any numeric pattern
        has_numeric_pattern = any(p in col_lower for p in numeric_patterns)

        if is_metric or (is_numeric and has_numeric_pattern):
            # Only include if the metric column itself is explicitly referenced
            # via schema-driven name/synonym variants, not just generic words.
            if not explicit_column_match:
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
    drop_counters: Optional[Dict[str, int]] = None,
) -> str:
    cards: List[str] = []
    present = set()
    
    # debug_count = 0
    if RETRIEVE_DEBUG:
        debug_count = 0 
        print(f"\n[RETRIEVAL - CARD PREVIEW] Menampilkan Top {RETRIEVAL_CARDS_DEBUG_PREVIEW_COUNT} Kartu untuk LLM:")

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
                if drop_counters is not None:
                    drop_counters["cards_irrelevant_column"] = int(
                        drop_counters.get("cards_irrelevant_column", 0)
                    ) + 1
                if RETRIEVE_DEBUG:
                    print(f"   [FILTER] Excluding {table}.{column} (not relevant to query)")
                    if _is_likely_filtered_requested_metric(column, table, question, qtype, md):
                        print(
                            f"   [FILTER-ALERT] {table}.{column} filtered despite mixed count+metric intent; "
                            "check metric mention coverage."
                        )
                continue

        col_meta = _column_structured_metadata(table, column, md)

        # --- LOGIKA DESKRIPSI (WATERFALL PRIORITY) ---
        desc = ""

        # 1. PRIORITY 1: structured/schema-aware description
        if col_meta.get("description"):
            desc = str(col_meta.get("description") or "").strip()

        # 2. PRIORITY 2: Fallback Regex (Manusiakan Nama Kolom)
        # Jaga-jaga jika deskripsi terstruktur tidak tersedia
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
        stats_str = _format_stats_for_llm(col_meta.get("stats") or md.get("stats"))
        
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
    if RETRIEVAL_META_KEYWORDS_KEY in meta:
        # Langsung pakai field 'keywords' yang sudah kaya nutrisi (sample values, aliases, dll)
        return _tokenize_bm25_query(str(meta[RETRIEVAL_META_KEYWORDS_KEY]))

    # Dense exports commonly store the searchable text separately and keep
    # table/column details under a nested metadata object.
    if meta.get("text"):
        return _tokenize_bm25_query(str(meta.get("text") or ""))

    nested_meta = meta.get("metadata") if isinstance(meta.get("metadata"), dict) else {}
    
    # --- Fallback: Legacy Logic (jika jsonl belum di-update) ---
    table = (_meta_table(meta) or _meta_table(nested_meta) or "").lower()
    column = (_meta_column(meta) or _meta_column(nested_meta) or "").lower()
    col_meta = _column_structured_metadata(table, column, meta)
    desc = str(col_meta.get("description") or "").lower()
    aliases = list(col_meta.get("aliases") or [])
    roles = list(col_meta.get("roles") or [])

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

def jsonl_sparse_search(q: str, jsonl_paths: List[str], max_hits: int = SPARSE_MAX_HITS) -> List[str]:
    # Cache key by sparse file content hash so any metadata edit invalidates BM25.
    sig = []
    for p in jsonl_paths:
        try:
            path = Path(p).resolve()
            digest = hashlib.sha256()
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    if chunk:
                        digest.update(chunk)
            sig.append(f"{path}:{digest.hexdigest()}")
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
                            if not tokens:
                                continue
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
                if not tokens:
                    continue
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
        for doc_id, sc in ranked[:min(max_hits, RETRIEVAL_SPARSE_DEBUG_PREVIEW_LIMIT)]:
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
        col_name = _meta_column(md) or ""
        col_meta = _column_structured_metadata(tbl, col_name, md)
        desc = str(col_meta.get("description") or "").lower()
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
        table = _meta_table(md) or ""
        column = _meta_column(md) or ""
        col_meta = _column_structured_metadata(table, column, md)
        roles = {str(role or "").strip().lower() for role in (col_meta.get("roles") or []) if str(role or "").strip()}

        # Primary: Check datatype field from metadata
        datatype = str(col_meta.get("datatype") or "").strip().lower()
        if datatype:
            # Remove any size specification (e.g., "decimal(10,2)" -> "decimal")
            base_type = datatype.split('(')[0].strip()
            if base_type in _NUMERIC_SQL_TYPES:
                # Check if this numeric column is used for calculations
                # (not just IDs or counts)
                col = column.lower()
                if roles.intersection({"pk", "fk", "id", "identifier", "primary_key", "foreign_key", "reference"}):
                    continue
                # Exclude primary keys and foreign keys (IDs) as they're not used for calculations
                desc = str(col_meta.get("description") or "").lower()
                if not any(kw in col for kw in RETRIEVAL_NUMERIC_EXCLUDE_COLUMN_KEYWORDS):
                    return True
                # Check description for calculation-related terms
                if any(kw in desc for kw in RETRIEVAL_NUMERIC_SIGNAL_KEYWORDS):
                    return True
                stats = col_meta.get("stats") or {}
                if any(isinstance(stats.get(k), (int, float)) and not isinstance(stats.get(k), bool) for k in ("min", "max", "mean", "stddev")):
                    return True

        # Fallback: Check column name pattern for numeric-sounding columns
        # (only if datatype check above didn't trigger)
        col = column.lower()
        if any(kw in col for kw in RETRIEVAL_NUMERIC_SIGNAL_KEYWORDS):
            return True

    return False
# ==============================================================================
# BRIDGE TABLE INFERENCE
# ==============================================================================
def _merge_fk_graph(base: Dict[str, Dict[str, str]], extra: Dict[str, Dict[str, str]]) -> None:
    for src, fk_map in (extra or {}).items():
        src_l = str(src or "").strip().lower()
        if not src_l:
            continue
        dst = base.setdefault(src_l, {})
        for fk_col, target in (fk_map or {}).items():
            fk_l = str(fk_col or "").strip().lower()
            tgt_l = str(target or "").strip().lower()
            if fk_l and tgt_l and fk_l not in dst:
                dst[fk_l] = tgt_l


def _build_schema_fk_graph() -> Dict[str, Dict[str, str]]:
    graph: Dict[str, Dict[str, str]] = {}
    try:
        fk_rels = get_fk_relationships() or {}
    except Exception:
        fk_rels = {}
    for src_table, fk_map in fk_rels.items():
        st_lower = str(src_table or "").strip().lower()
        if not st_lower:
            continue
        for fk_col, target_table in (fk_map or {}).items():
            fc_lower = str(fk_col or "").strip().lower()
            tt_lower = str(target_table or "").strip().lower()
            if not fc_lower or not tt_lower:
                continue
            graph.setdefault(st_lower, {})[fc_lower] = tt_lower
    return graph


def _build_effective_fk_graph(reranked: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    """
    Build FK graph from reranked metadata, then backfill from schema metadata so
    connector discovery can still traverse unseen bridge tables.
    """
    fk_graph: Dict[str, Dict[str, str]] = {}
    for item in reranked:
        md = item.get("meta") or {}
        source_table = _meta_table(md)
        fk_column = _meta_column(md)
        col_meta = _column_structured_metadata(source_table or "", fk_column or "", md)
        desc = str(col_meta.get("description") or "")

        fk_match = re.search(RETRIEVAL_FK_REFERENCE_PATTERN, desc, re.IGNORECASE)
        if not fk_match:
            continue
        target_table = fk_match.group(1)
        if not (source_table and fk_column and target_table):
            continue

        st_lower = str(source_table).strip().lower()
        fc_lower = str(fk_column).strip().lower()
        tt_lower = str(target_table).strip().lower()
        if st_lower and fc_lower and tt_lower:
            fk_graph.setdefault(st_lower, {})[fc_lower] = tt_lower

    _merge_fk_graph(fk_graph, _build_schema_fk_graph())
    return fk_graph


def _canonicalize_table_column(table: str, column: str) -> str:
    t = str(table or "").strip().lower()
    c = str(column or "").strip()
    if not t or not c:
        return c
    try:
        analyzer = get_schema_analyzer()
        for candidate in analyzer.get_table_columns(t) or []:
            cand_s = str(candidate or "").strip()
            if cand_s and cand_s.lower() == c.lower():
                return cand_s
    except Exception:
        pass
    return c


def _infer_bridge_paths(
    selected_tables: set,
    allowed: Dict[str, List[str]],
    reranked: List[Dict[str, Any]],
) -> Tuple[List[List[str]], Dict[str, Dict[str, str]]]:
    """
    Discover FK connector paths between selected tables using the full schema
    graph as the traversal universe, while keeping endpoints anchored to the
    currently selected table set.
    """
    if len(selected_tables) < 2:
        return [], _build_effective_fk_graph(reranked)

    fk_graph = _build_effective_fk_graph(reranked)

    try:
        table_universe = {
            str(t or "").strip().lower()
            for t in (get_all_tables() or [])
            if str(t or "").strip()
        }
    except Exception:
        table_universe = set()
    if not table_universe:
        table_universe = {str(t).strip().lower() for t in allowed.keys() if str(t).strip()}

    selected_lower = {str(t).strip().lower() for t in selected_tables if str(t).strip()}
    paths: List[List[str]] = []
    seen_paths: Set[Tuple[str, ...]] = set()
    table_list = sorted(selected_lower)
    for i, tbl_a in enumerate(table_list):
        for tbl_b in table_list[i + 1:]:
            path = _find_fk_path(tbl_a, tbl_b, fk_graph, allowed, table_universe=table_universe)
            if not path or len(path) < 2:
                continue
            path_key = tuple(path)
            if path_key in seen_paths:
                continue
            seen_paths.add(path_key)
            paths.append(path)
    return paths, fk_graph


def _augment_allowed_with_fk_path_columns(
    allowed: Dict[str, List[str]],
    paths: List[List[str]],
    fk_graph: Dict[str, Dict[str, str]],
) -> None:
    """
    Seed connector-path tables with the join-key columns needed for downstream
    SQL generation: FK columns on traversed edges plus each path table's ID key.
    """
    for path in paths or []:
        normalized_path = [str(t or "").strip().lower() for t in path if str(t or "").strip()]
        if not normalized_path:
            continue

        for table_name in normalized_path:
            allowed.setdefault(table_name, [])
            id_col = str(get_id_column(table_name) or "").strip()
            if id_col:
                id_col = _canonicalize_table_column(table_name, id_col)
                if id_col and id_col not in allowed[table_name]:
                    allowed[table_name].append(id_col)

        for left, right in zip(normalized_path, normalized_path[1:]):
            for fk_col, target in (fk_graph.get(left) or {}).items():
                if str(target or "").strip().lower() != right:
                    continue
                fk_name = _canonicalize_table_column(left, fk_col)
                if fk_name and fk_name not in allowed[left]:
                    allowed[left].append(fk_name)
            for fk_col, target in (fk_graph.get(right) or {}).items():
                if str(target or "").strip().lower() != left:
                    continue
                fk_name = _canonicalize_table_column(right, fk_col)
                if fk_name and fk_name not in allowed[right]:
                    allowed[right].append(fk_name)


def _find_fk_path(
    start: str,
    end: str,
    fk_graph: Dict[str, Dict[str, str]],
    allowed: Dict[str, List[str]],
    table_universe: Optional[Set[str]] = None,
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

    allowed_tables = {
        str(t or "").strip().lower()
        for t in (table_universe or set(allowed.keys()))
        if str(t or "").strip()
    }
    if start not in allowed_tables or end not in allowed_tables:
        return None

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
                if target not in visited and target in allowed_tables:
                    visited.add(target)
                    queue.append((target, path + [target]))

        # Check reverse FKs (target -> current)
        if current in reverse_graph:
            for source in reverse_graph[current]:
                if source == end:
                    return path + [source]
                if source not in visited and source in allowed_tables:
                    visited.add(source)
                    queue.append((source, path + [source]))

    return None
# ======================================================================
# =========================== RETRIEVE NODE ============================
# ======================================================================
def _has_join_intent(question: str, primary_table: str = "") -> bool:
    q = (question or "").lower()
    weak_join_signals = {"for each", "per", "setiap"}
    matched_join_signals = [
        s for s in RETRIEVAL_JOIN_SIGNALS
        if s and s in q
    ]
    if any(str(s).strip().lower() not in weak_join_signals for s in matched_join_signals):
        return True

    # Generic multi-entity signal: 2+ explicit table mentions.
    mentioned_tables = [
        t for t in get_all_tables()
        if _text_mentions_table(q, t, include_synonyms=False)
    ]
    if len(mentioned_tables) >= RETRIEVAL_MULTI_ENTITY_MIN_TABLES:
        return True

    # Relation-like cue from inflected entity terms (e.g., "ordered", "shipping")
    # should be treated as join intent when it points to a non-primary table.
    primary_l = str(primary_table or "").strip().lower()
    for tbl in get_all_tables():
        tbl_l = str(tbl or "").strip().lower()
        if not tbl_l or (primary_l and tbl_l == primary_l):
            continue
        singular = str(get_singular_form(tbl_l) or tbl_l).strip().lower().replace("_", "")
        if singular.isalpha() and len(singular) >= 4:
            if re.search(rf"\b{re.escape(singular)}(?:ed|ing)\b", q):
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


def _should_lock_single_table_ctx(question: str, qtype: str) -> bool:
    primary = _detect_primary_entity_table(question)
    if qtype not in {AGENT_ROUTER_QTYPE_LOOKUP, AGENT_ROUTER_QTYPE_AGGREGATE}:
        return False
    if _has_join_intent(question, primary_table=primary):
        return False
    return primary is not None


def _has_self_hierarchy_lookup_intent(question: str, primary_table: str, qtype: str) -> bool:
    primary = str(primary_table or "").strip().lower()
    if qtype != AGENT_ROUTER_QTYPE_LOOKUP or not primary:
        return False
    if not question_has_hierarchy_role_intent(question):
        return False
    try:
        fk_map = get_fk_relationships() or {}
    except Exception:
        fk_map = {}
    rels = fk_map.get(primary, {}) or {}
    return any(str(target or "").strip().lower() == primary for target in rels.values())


def _filter_allowed_for_self_hierarchy_lookup(
    question: str,
    primary_table: str,
    qtype: str,
    allowed: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    allowed_map = dict(allowed or {})
    if not _has_self_hierarchy_lookup_intent(question, primary_table, qtype):
        return allowed_map

    primary = str(primary_table or "").strip().lower()
    if not primary:
        return allowed_map

    keep_tables: Set[str] = {primary}
    for table_name in allowed_map.keys():
        table_l = str(table_name or "").strip().lower()
        if not table_l or table_l == primary:
            continue
        if _text_mentions_table(question, table_l, include_synonyms=False):
            keep_tables.add(table_l)

    filtered = {
        table_name: cols
        for table_name, cols in allowed_map.items()
        if str(table_name or "").strip().lower() in keep_tables
    }
    return filtered or allowed_map


def _single_table_lock_is_high_confidence(
    primary_table: str,
    table_scores: Dict[str, float],
    reranked: List[Dict[str, Any]],
    evidence_strength: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Decide whether single-table lock should be strict.
    High confidence when:
    - primary table score clearly leads table #2 by configured gap, and
    - sparse evidence from non-primary tables is limited.
    """
    primary = str(primary_table or "").strip().lower()
    if not primary:
        return False

    # Weak retrieval evidence should not force strict single-table lock.
    evidence_tier = str((evidence_strength or {}).get("tier") or "").strip().lower()
    if evidence_tier == "weak":
        return False

    # No score signal available: preserve previous behavior (strict lock).
    if not table_scores:
        return True

    scores_norm = {
        str(t or "").strip().lower(): float(s)
        for t, s in (table_scores or {}).items()
        if str(t or "").strip()
    }
    if primary not in scores_norm:
        return False

    primary_score = scores_norm.get(primary, 0.0)
    second_best = max((s for t, s in scores_norm.items() if t != primary), default=None)
    score_gap = primary_score if second_best is None else (primary_score - second_best)

    non_primary_sparse_hits = 0
    for item in (reranked or [])[:MAX_CTX_DOCS]:
        md = item.get("meta") or {}
        tbl = (_meta_table(md) or "").strip().lower()
        if not tbl or tbl == primary:
            continue
        if bool(item.get("sparse", False)):
            non_primary_sparse_hits += 1

    return bool(
        score_gap >= float(RETRIEVAL_SINGLE_LOCK_MIN_SCORE_GAP)
        and non_primary_sparse_hits <= int(RETRIEVAL_SINGLE_LOCK_MAX_NON_PRIMARY_SPARSE_HITS)
    )

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
    if not all_paths and str(SPARSE_METADATA_JSONL or "").strip():
        all_paths = [SPARSE_METADATA_JSONL]
    sparse_target_paths = [p for p in all_paths if RETRIEVAL_SPARSE_FILENAME_MARKER in str(p)]
    
    # Fallback: if caller passed only the explicit sparse path without marker filtering,
    # or when running with a custom sparse filename, keep using the configured inputs.
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
        col_meta = _column_structured_metadata(table_name, col_raw, md)
        roles = list(col_meta.get("roles") or [])
        aliases = list(col_meta.get("aliases") or [])

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
                    table_name = _meta_table(md) or ""
                    col_raw = _meta_column(md) or ""
                    col_name = col_raw.lower()
                    col_meta = _column_structured_metadata(table_name, col_raw, md)
                    roles = list(col_meta.get("roles") or [])
                    aliases = list(col_meta.get("aliases") or [])
                    max_boost = max((x["boosted"] for x in scored), default=0.0)
                    boosted = max_boost + RETRIEVAL_AUTO_INJECT_BONUS_DELTA
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
        max_show = min(RETRIEVAL_RETRIEVE_DEBUG_MAX_SHOW, total_sparse)
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
        max_show = min(RETRIEVAL_RETRIEVE_DEBUG_MAX_SHOW, total_dense)
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
        max_show = min(RETRIEVAL_RETRIEVE_DEBUG_MAX_SHOW, total_rerank)
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
    retrieval_drop_reasons: Dict[str, int] = {}

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
                retrieval_drop_reasons["allowed_irrelevant_column"] = int(
                    retrieval_drop_reasons.get("allowed_irrelevant_column", 0)
                ) + 1
                if RETRIEVE_DEBUG:
                    print(f"   [FILTER-ALLOWED] Excluding {t}.{c} from allowed identifiers")
                    if _is_likely_filtered_requested_metric(c, t, q, qtype, md):
                        print(
                            f"   [FILTER-ALERT] {t}.{c} removed from allowed identifiers despite mixed count+metric intent; "
                            "check metric mention coverage."
                        )
                continue
            elif RETRIEVE_DEBUG and c.lower() in RETRIEVAL_DEBUG_FOCUS_COLUMNS:
                # Debug: show why numeric columns passed the filter
                print(f"   [FILTER-ALLOWED] KEPT {t}.{c} - passed relevance check")

        t_key = t.strip().lower()
        c_key = c.strip()
        allowed[t_key].append(c_key)
        col_meta = _column_structured_metadata(t, c, md)
        aliases2 = _merge_unique_lower(
            list(col_meta.get("aliases") or []) + _extract_aliases_for_map(md)
        )
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
    allowed_col_count_pre = sum(len(set(cols or [])) for cols in (allowed or {}).values())
    lock_evidence_snapshot = {
        "sparse_candidates": int(len(sparse_field_urns or [])),
        "dense_candidates": int(len(dense_metas or [])),
        "reranked_candidates": int(len(reranked or [])),
        "allowed_tables_after_filter": int(len(allowed or {})),
        "allowed_columns_after_filter": int(allowed_col_count_pre),
        "allowed_columns_before_filter": int(allowed_col_count_pre),
    }
    lock_evidence_strength = _compute_retrieval_evidence_strength(lock_evidence_snapshot)

    # Aggregate override:
    # if join-intent heuristic is noisy but one primary table dominates strongly,
    # force single-table lock to keep IR scope stable for simple grouped aggregates.
    force_single_lock = False
    if not lock_single and qtype == AGENT_ROUTER_QTYPE_AGGREGATE:
        primary_aggregate = _detect_primary_entity_table(q)
        has_join_intent = _has_join_intent(q, primary_table=primary_aggregate or "")
        if primary_aggregate and not has_join_intent:
            lock_high_conf = _single_table_lock_is_high_confidence(
                primary_table=primary_aggregate,
                table_scores=table_scores,
                reranked=reranked,
                evidence_strength=lock_evidence_strength,
            )
            if lock_high_conf:
                lock_single = True
                primary_for_lock = primary_aggregate
                force_single_lock = True

    if qtype == AGENT_ROUTER_QTYPE_LOOKUP and prefer_name_table:
        only_preferred = {t: cols for t, cols in allowed.items() if t.lower() == prefer_name_table}
        filtered_allowed = only_preferred if only_preferred else dict(allowed)
    else:
        if lock_single:
            primary = primary_for_lock
            lock_high_conf = _single_table_lock_is_high_confidence(
                primary_table=primary,
                table_scores=table_scores,
                reranked=reranked,
                evidence_strength=lock_evidence_strength,
            )
            if lock_high_conf:
                only_primary = {t: cols for t, cols in allowed.items() if t.lower() == (primary or "")}
                if only_primary:
                    filtered_allowed = only_primary
                else:
                    if table_scores:
                        sorted_tbls = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
                        top1 = sorted_tbls[0][0]
                        filtered_allowed = {t: cols for t, cols in allowed.items() if t == top1}
                    else:
                        filtered_allowed = dict(allowed)
            else:
                fallback_k = max(1, int(RETRIEVAL_SINGLE_LOCK_FALLBACK_TOPK))
                sorted_tbls = sorted(table_scores.items(), key=lambda x: x[1], reverse=True) if table_scores else []
                fallback_tables = {str(t).strip().lower() for t, _ in sorted_tbls[:fallback_k] if str(t).strip()}
                if primary:
                    fallback_tables.add(primary.lower())

                filtered_allowed = {
                    t: cols for t, cols in allowed.items()
                    if t.lower() in fallback_tables
                }
                if not filtered_allowed:
                    filtered_allowed = dict(allowed)

                if RETRIEVE_DEBUG:
                    print(
                        "[RETRIEVAL - LOCK] Low-confidence single-table lock -> fallback "
                        f"to top-{fallback_k} tables: {sorted(fallback_tables)}"
                    )
            if RETRIEVE_DEBUG and force_single_lock:
                print(
                    "[RETRIEVAL - LOCK] Aggregate override -> forced single-table lock "
                    f"to primary table: {primary_for_lock}"
                )
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
                table_score_map = {str(t).strip().lower(): float(s) for t, s in sorted_tbls if str(t).strip()}

                # DEBUG: Print table scores
                if RETRIEVE_DEBUG:
                    print(f"\n[RETRIEVAL - TABLE SCORES]")
                    for tbl, score in sorted_tbls:
                        print(f"  {tbl}: {score:.2f}")

                # Base: Top 3 tables by score (increased from 2)
                max_top_tables = MAX_TOP_TABLES
                score_floor = float(RETRIEVAL_TOP_TABLE_MIN_SCORE)
                # Aggregate-specific floor tuning:
                # - count-only intents stay conservative (avoid noisy extra tables),
                # - metric intents (quantity/amount/sum/avg/...) allow detail tables.
                if qtype == AGENT_ROUTER_QTYPE_AGGREGATE:
                    is_count_like = bool(
                        re.search(r"\b(how\s+many|number\s+of|count|total\s+number)\b", q_l, flags=re.I)
                    )
                    metric_intent_terms = {
                        "quantity", "amount", "price", "cost", "value",
                        "total", "metric", "score",
                        "multiply", "multiplied", "times",
                    }
                    metric_role_terms = {
                        str(role or "").strip().lower()
                        for role in RETRIEVAL_METRIC_ROLE_TAGS
                        if str(role or "").strip()
                    }
                    for item in reranked[:MAX_CTX_DOCS]:
                        md = item.get("meta") or {}
                        table_name = _meta_table(md) or ""
                        col_name = _meta_column(md) or ""
                        if not table_name or not col_name:
                            continue
                        col_meta = _column_structured_metadata(table_name, col_name, md)
                        roles = list(col_meta.get("roles") or [])
                        aliases = list(col_meta.get("aliases") or [])
                        role_hits = {
                            str(role or "").strip().lower()
                            for role in roles
                            if str(role or "").strip()
                        }
                        if not (role_hits & metric_role_terms):
                            continue
                        metric_intent_terms.update(
                            tok for tok in re.findall(r"[a-z0-9]+", str(col_name or "").lower()) if tok
                        )
                        for alias in aliases:
                            metric_intent_terms.update(
                                tok for tok in re.findall(r"[a-z0-9]+", str(alias or "").lower()) if tok
                            )
                        try:
                            for syn in get_column_synonyms(table_name, col_name) or []:
                                metric_intent_terms.update(
                                    tok for tok in re.findall(r"[a-z0-9]+", str(syn or "").lower()) if tok
                                )
                        except Exception:
                            pass
                    q_terms = {tok for tok in re.findall(r"[a-z0-9]+", q_l) if tok}
                    has_metric_intent = bool(
                        re.search(r"\b(sum|avg|average|mean|min|max|minimum|maximum)\b", q_l, flags=re.I)
                        or (q_terms & metric_intent_terms)
                    )
                    if is_count_like and not has_metric_intent:
                        score_floor = max(score_floor, 1.0)
                    elif has_metric_intent:
                        score_floor = max(score_floor, 0.5)
                top_tables = {
                    name
                    for name, score in sorted_tbls[:max_top_tables]
                    if score > score_floor
                }

                # Addition: Include tables with any column in reranked results
                # This captures tables that may not be mentioned but have needed columns
                tables_with_reranked_cols = set()
                tables_with_temporal_cols = set()
                table_first_rank: Dict[str, int] = {}
                for item in reranked[:MAX_CTX_DOCS]:
                    md = item.get("meta") or {}
                    tbl = _meta_table(md)
                    if tbl:
                        tbl_l = tbl.lower()
                        tables_with_reranked_cols.add(tbl_l)
                        table_first_rank.setdefault(tbl_l, len(table_first_rank))

                        # Temporal-capable table detection from metadata and column name.
                        col = (_meta_column(md) or "").strip().lower()
                        col_meta = _column_structured_metadata(tbl_l, col, md)
                        datatype = str(col_meta.get("datatype") or "").strip().lower()
                        base_type = datatype.split("(")[0].strip() if datatype else ""
                        is_temporal = base_type in {"date", "datetime", "timestamp", "time", "year"}
                        if not is_temporal and col:
                            is_temporal = any(k in col for k in ("date", "time", "timestamp", "year", "month", "day"))
                        if is_temporal:
                            tables_with_temporal_cols.add(tbl_l)

                # Temporal-intent safeguard:
                # if question asks a year/date condition but selected tables contain no
                # temporal-capable table, force-add the best temporal table candidate.
                has_temporal_intent = bool(
                    re.search(
                        r"\b((?:19|20)\d{2}|year|month|day|date|time|latest|earliest|newest|oldest|recent)\b",
                        q_l,
                        flags=re.I,
                    )
                )
                if (
                    qtype == AGENT_ROUTER_QTYPE_AGGREGATE
                    and has_temporal_intent
                    and tables_with_temporal_cols
                    and not top_tables.intersection(tables_with_temporal_cols)
                ):
                    best_temporal = sorted(
                        tables_with_temporal_cols,
                        key=lambda t: (
                            -float(table_score_map.get(t, 0.0)),
                            int(table_first_rank.get(t, 10**9)),
                            t,
                        ),
                    )[0]
                    top_tables.add(best_temporal)

                # Merge: top scoring + tables with reranked columns
                # Use all available table keys (from original allowed, not filtered)
                all_table_keys_lower = {tbl.lower(): tbl for tbl in allowed.keys()}

                # DEBUG: Print what we're adding
                if RETRIEVE_DEBUG:
                    print(f"\n[RETRIEVAL - COLUMN-DRIVEN SELECTION]")
                    print(f"  tables_with_reranked_cols: {tables_with_reranked_cols}")
                    print(f"  all_table_keys_lower: {list(all_table_keys_lower.keys())}")
                    to_add = [
                        t
                        for t in tables_with_reranked_cols
                        if t in all_table_keys_lower and table_score_map.get(t, 0.0) > score_floor
                    ]
                    print(f"  adding to top_tables: {to_add}")
                    if (
                        qtype == AGENT_ROUTER_QTYPE_AGGREGATE
                        and has_temporal_intent
                        and tables_with_temporal_cols
                    ):
                        print(f"  temporal_tables: {sorted(tables_with_temporal_cols)}")

                top_tables.update(
                    t
                    for t in tables_with_reranked_cols
                    if t in all_table_keys_lower and table_score_map.get(t, 0.0) > score_floor
                )

                # Addition: Bridge table inference
                # If we have multiple tables, check if we need bridge tables
                if len(top_tables) >= RETRIEVAL_BRIDGE_MIN_TABLE_COUNT:
                    connector_paths, fk_graph = _infer_bridge_paths(top_tables, allowed, reranked)
                    top_tables_lower = {
                        str(t).strip().lower()
                        for t in top_tables
                        if str(t).strip()
                    }
                    bridge_paths = [
                        path
                        for path in connector_paths
                        if any(
                            bridge not in top_tables_lower
                            for bridge in path[1:-1]
                        )
                    ]
                    if bridge_paths:
                        _augment_allowed_with_fk_path_columns(allowed, bridge_paths, fk_graph)
                    bridge_tables = {
                        bridge
                        for path in bridge_paths
                        for bridge in path[1:-1]
                        if bridge not in top_tables_lower
                    }
                    top_tables.update(bridge_tables)

                if top_tables:
                    filtered_allowed = {t: cols for t, cols in allowed.items() if t.lower() in top_tables}
                else:
                    filtered_allowed = dict(allowed)
            else:
                filtered_allowed = dict(allowed)

    filtered_allowed = _filter_allowed_for_self_hierarchy_lookup(
        q,
        primary_lookup_table or "",
        qtype,
        filtered_allowed,
    )

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
        drop_counters=retrieval_drop_reasons,
    )

    context = (allowed_block + "\n\n### CARDS\n" + cards_text).strip()
    context += f"\n\n### _INTERNAL_HINT\n{json.dumps({'alias_map': alias_map}, ensure_ascii=False)}"

    # DEBUG: Show key info about context being built
    if RETRIEVE_DEBUG:
        print(f"\n[DEBUG] Building context with {len(filtered_allowed)} tables")
        for tbl, cols in filtered_allowed.items():
            print(f"[DEBUG] {tbl}: {len(cols)} columns")

    telem_limit = max(1, int(RETRIEVAL_TELEMETRY_MAX_ITEMS))
    sparse_preview = [str(s) for s in (sparse_docs_pretty or [])[:telem_limit]]
    dense_preview = [
        f"{(_meta_table(md) or '?')}.{(_meta_column(md) or '?')}"
        for md in (dense_metas or [])[:telem_limit]
    ]
    rerank_preview = [
        f"{(_meta_table((it.get('meta') or {})) or '?')}.{(_meta_column((it.get('meta') or {})) or '?')}"
        for it in (reranked or [])[:telem_limit]
    ]
    filtered_allowed_col_count = sum(len(set(cols or [])) for cols in (filtered_allowed or {}).values())
    allowed_col_count = allowed_col_count_pre
    retrieval_snapshot = {
        "qtype": str(qtype or ""),
        "sparse_candidates": int(len(sparse_field_urns or [])),
        "dense_candidates": int(len(dense_metas or [])),
        "reranked_candidates": int(len(reranked or [])),
        "allowed_tables_before_filter": int(len(allowed or {})),
        "allowed_tables_after_filter": int(len(filtered_allowed or {})),
        "allowed_columns_before_filter": int(allowed_col_count),
        "allowed_columns_after_filter": int(filtered_allowed_col_count),
        "drop_reasons": {str(k): int(v) for k, v in sorted((retrieval_drop_reasons or {}).items())},
        "sparse_preview": sparse_preview,
        "dense_preview": dense_preview,
        "rerank_preview": rerank_preview,
    }
    evidence_strength = _compute_retrieval_evidence_strength(retrieval_snapshot)
    retrieval_snapshot["evidence_strength"] = evidence_strength
    retrieval_snapshot["evidence_score"] = float(evidence_strength.get("score", 0.0))
    retrieval_snapshot["evidence_tier"] = str(evidence_strength.get("tier") or "weak")

    print(
        "[RETRIEVE] "
        f"qtype={retrieval_snapshot['qtype'] or 'unknown'} "
        f"sparse={retrieval_snapshot['sparse_candidates']} "
        f"dense={retrieval_snapshot['dense_candidates']} "
        f"reranked={retrieval_snapshot['reranked_candidates']} "
        f"allowed_tables={retrieval_snapshot['allowed_tables_before_filter']}->"
        f"{retrieval_snapshot['allowed_tables_after_filter']} "
        f"allowed_cols={retrieval_snapshot['allowed_columns_before_filter']}->"
        f"{retrieval_snapshot['allowed_columns_after_filter']} "
        f"evidence={retrieval_snapshot['evidence_tier']}("
        f"{retrieval_snapshot['evidence_score']:.2f})"
    )
    print(
        "[RETRIEVE - PREVIEW] "
        f"sparse={retrieval_snapshot['sparse_preview'] or ['none']} "
        f"dense={retrieval_snapshot['dense_preview'] or ['none']} "
        f"rerank={retrieval_snapshot['rerank_preview'] or ['none']}"
    )

    state["context"] = context
    state["retrieval_context_mode"] = "primary"
    state["retrieval_retry_used"] = False
    state["retrieval_retry_reason"] = ""
    state["retrieval_expanded_table_count"] = 0
    state["retrieval_snapshot"] = retrieval_snapshot
    return state
