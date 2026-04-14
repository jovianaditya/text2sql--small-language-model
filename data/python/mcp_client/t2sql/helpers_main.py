# ======================================================================
# ========================== helpers_main.py ===========================
# ======================================================================
# Purpose:
# - Helper Cross Module
# - SQL whitelist parsing/validation (ALLOWED IDENTIFIERS)
# - Postprocess Heuristics
# - Generation prompt builder (LLM instruction)
# - Main Pipeline : Postprocess SQL and Normalize Pipeline

# ======================================================================
# ====================== STANDARD LIBRARY IMPORT =======================
# ======================================================================
import re
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
# ======================================================================
# ======================== LOCAL MODULE IMPORT =========================
# ======================================================================
if TYPE_CHECKING:
    from .state import RAGState
from .utils import _dbg_print_two
from .config import (
    AGENT_AST_GUARDS_ENABLED,
    AGENT_AST_GUARDS_DEBUG,
    AGENT_AST_SEMANTIC_FIRST,
    AGENT_NORMALIZE_REWRITE_BUDGET,
    AGENT_NORMALIZE_EARLY_STOP,
    DOMAIN_HEURISTICS_ENABLED,
    HELPERS_EXISTENCE_SUPPORT_VERBS_GENERIC,
    HELPERS_EXISTENCE_COMPAT_VERBS,
    HELPERS_PROBE_GENERIC_METRIC_TERMS,
    HELPERS_PROBE_LOCALIZED_METRIC_TERMS,
    HELPERS_SEMANTIC_ALIAS_LISTPRICE_INCLUDE_TERMS,
    HELPERS_SEMANTIC_ALIAS_LISTPRICE_EXCLUDE_TERMS,
    HELPERS_SEMANTIC_ALIAS_PRICE_INCLUDE_TERMS,
    HELPERS_SEMANTIC_ALIAS_PRICE_EXCLUDE_TERMS,
    HELPERS_WINNER_ROW_GENERIC_OBJECT_TERMS,
)
from .schema_analyzer import (
    get_transaction_tables,
    get_anchor_columns,
    get_all_tables,
    get_aggregatable_columns,
    get_filterable_date_columns,
    get_fk_relationships,
    get_id_column,
    get_name_column,
    get_table_synonyms,
    get_column_synonyms,
    get_column_entity_variants,
    get_column_role,
    get_schema_analyzer,
    get_singular_form,
)
from .helpers_core import (
    _parse_allowed_identifiers, _question_requires_word_boundary_regex, _question_requires_structural_regex, _question_requires_regex, has_join,
    _question_requires_inner_join,
    _question_wants_include_zero,
    _question_wants_dual_fact_rollup,
    _SQL_KEYWORDS,  # SQL keywords constant
    # phase 0
    _fix_bad_backticks, _canonicalize_table_names, _escape_reserved_column_aliases,
    # phase 1
    _move_rhs_filter_from_where_to_on,  # Only triggers for "ALL entities" questions
    _rewrite_left_join_to_inner_if_not_needed, _rewrite_left_join_to_inner_if_rhs_filtered,
    _rewrite_inner_join_filter_to_exists, _rewrite_inner_join_existence_to_exists,
    _rewrite_invalid_grouped_existence_subquery_to_join,
    _remove_redundant_and_ornamental_joins, _rewrite_invalid_fk_join_conditions, has_non_inner_join, join_count,
    # phase 2
    _drop_redundant_column_aliases, _drop_unused_from_table_aliases, _strip_alias_if_single_table, _repair_invalid_qualified_columns, _repair_simple_single_table_column_names,
    # phase 3
    _prefer_friendly_names, _ensure_entity_id_and_name_if_requested, _ensure_explicit_simple_projection_columns, _prune_non_primary_dimensions_for_primary_aggregate, _drop_redundant_raw_name_columns_when_concat_present, _wants_name_only,
    _expand_group_by_aliases_regex, _is_potential_winner_row, _enforce_explicit_group_by,
    _remove_invalid_select_columns_with_group_by, _fix_order_by_group_by_consistency,
    _normalize_alias_lengths,
    # phase 4
    _winner_row_sanity_date_in_groupby, _strip_having_max_self_subquery, _fix_mixed_max_groupby_rownumber, _fix_redundant_groupby_with_rn_filter, _fix_inconsistent_subquery_alias, _fix_inner_rn_filter_scope, _looks_like_max_date_tiebreaker,
    _question_wants_winner_row, _rewrite_winner_row_guardrail, _rewrite_having_alias_to_expr,
    _force_restore_groupby_if_aggregate_without_groupby,
    _preexecute_sql_safety_pass,
    _rewrite_winner_row_aggregate_misuse, _rewrite_single_table_tuple_in_winner_row, _rewrite_explicit_rank_aggregate_window, _fix_winner_row_window_order_priority, _fix_winner_row_metric_order_priority, _rewrite_latest_year_same_year_aggregate, _rewrite_latest_year_correlated_maxdate_misuse, _rewrite_latest_year_derived_subquery_misuse, _rewrite_bridge_latest_year_tuple_filter, _rewrite_latest_year_tuple_in_filter, _move_latest_year_filter_where_to_inner_on, _winner_row_final_sanity,
    _canonicalize_latest_year_same_year_pattern,
    _rewrite_latest_year_projection_to_year,
    _rewrite_dual_fact_rollup_fanout,
    _rewrite_grouped_winner_row_ranked_aggregate,
    _sanitize_order_by_alias_assignment,
    _rewrite_invalid_metric_bridge_rollup,
    _rewrite_missing_metric_join_in_rollup,
    _rewrite_missing_joined_agg_threshold_filter,
    _rewrite_missing_overall_avg_having,
    _rewrite_employee_latest_payment_customer_identity,
    _drop_tuple_in_filter_from_include_zero_ranked_winner,
    _rewrite_metric_aggregate_column_mismatch,
    _rewrite_row_level_nonnull_to_group_existence,
    _rewrite_role_label_count_semantics,
    _drop_unrequested_exists_filters,
    _drop_unrequested_group_key_filters,
    _drop_unrequested_temporal_filters,
    _rewrite_missing_temporal_bridge_exclusion,
    _rewrite_temporal_year_exclusion_antijoin,
    _prefer_not_exists_for_simple_exclusion,
    _repair_temporal_not_exists_scope,
    _question_wants_latest_year_scope_with_aggregate,
    _split_top_level_and,
    _split_top_level_commas,
    _has_top_level_or,
    # phase 4b - Aggregate in WHERE fix
    _fix_aggregate_in_where,
    _fix_raw_column_where_to_having,
    _add_missing_count_for_group_by,
    # phase 5 — Pattern, REGEXP, and Predicate Normalization
    _guardrail_fix_like_regex_anchors, _rewrite_simple_regexp_to_like, _canonicalize_structural_regex, _canonicalize_boundary_regex, _guardrail_fix_mysql_regexp_literals,
    # phase 6
    _expand_missing_null_to_null_or_empty, _ensure_reasonable_distinct, _question_wants_count_distinct, _ensure_reasonable_count_distinct, _drop_redundant_group_by_for_distinct_lookup, _rewrite_case_split_to_union, _normalize_simple_set_operation_branches, _rewrite_union_case_label_to_literal_branches, _rewrite_concat_label_union, _strip_unneeded_count_distinct, _parse_from_join_aliases,
    _ast_guardrails,
    # phase 7
    _simplify_default_window_frame, _rewrite_correlated_offset_subquery_to_window, _rewrite_same_table_group_count_self_join_to_window, _rewrite_running_self_join_aggregate_to_window, _rewrite_single_table_case_band_derived_subquery, _enforce_window_sequence_direction, _repair_offset_window_projection_shape, _preserve_requested_base_columns_in_running_self_join, _enforce_requested_order_direction, _remove_top_level_order_by_if_not_requested, _maybe_add_limit, _collapse_spaces_outside_quotes
)
# ======================================================================
# ===================== SQL ALIAS & KEYWORD CONSTANTS ==================
# ======================================================================
# 🆕 Dynamic schema-based configuration (agnostic)
TRANSACTION_TABLES = get_transaction_tables()  # Auto-detected from schema metadata

# 🆕 Dynamic anchor signals (Metric columns from metadata)
# Build from schema: for each table, get columns with Role: Metric
_ANCHOR_SIGNALS = {}
for table in get_all_tables():
    anchors = get_anchor_columns(table)
    if anchors:
        _ANCHOR_SIGNALS[table] = set(anchors)

_ALNUM = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
try:
    _TABLE_SINGULAR_ROOTS = {
        get_singular_form(t.lower().strip())
        for t in get_all_tables()
        if t
    }
except Exception:
    _TABLE_SINGULAR_ROOTS = set()
# ======================================================================
# ============================ WHITELIST ===============================
# ======================================================================
def get_single_allowed_table(ctx: str) -> Optional[str]:
    """
    Jika ALLOWED IDENTIFIERS hanya berisi 1 tabel, return nama tabel itu.
    Kalau >1 atau 0 tabel, return None.
    """
    allowed = _parse_allowed_identifiers(ctx)
    if len(allowed) == 1:
        return next(iter(allowed.keys()))
    return None
# ======================================================================
# ========================= HELPER CROSS MODULE ========================
# ======================================================================
# -----------------------------------------------------------
# -------------------- HELPER CROSS MODULE ------------------
# ------------------------- agents.py -----------------------
# -----------------------------------------------------------
def _identify_used_identifiers(sql: str) -> Dict[Optional[str], Set[str]]:
    """
    Return mapping:
      - qualifier -> set(columns)  e.g. {"p": {"amount","paymentDate"}}
      - None      -> set(unqualified tokens) e.g. {None: {"amount"}}
    """
    if not sql:
        return {}

    out: Dict[Optional[str], Set[str]] = {}
    
    # 1) Hapus string literals
    code_only = re.sub(r"'(?:''|[^'])*'", "''", sql)
    code_only = re.sub(r'"(?:""|[^"])*"', '""', code_only)
    keywords = _SQL_KEYWORDS

    # 2) Ambil table names yang muncul di FROM/JOIN (untuk exclude)
    table_names: Set[str] = set()
    for m in re.finditer(
        r"\b(?:from|join)\s+(`?[A-Za-z_][A-Za-z0-9_]*`?(?:\.`?[A-Za-z_][A-Za-z0-9_]*`?)?)",
        code_only,
        flags=re.I,
    ):
        raw = (m.group(1) or "").replace("`", "")
        tbl = raw.split(".")[-1]
        if tbl:
            table_names.add(tbl.lower())

    # Exclude aliases (table + derived + output)
    exclude: Set[str] = set()

    # 3) Table aliases from FROM/JOIN (Simple Table)
    #    Contoh: FROM payments p -> p is alias
    a2t = _parse_from_join_aliases(sql) 
    exclude.update({a.lower() for a in a2t.keys() if a})

    # =======================================================================
    # PERBAIKAN: DETEKSI DERIVED TABLE ALIASES (Subquery Alias)
    # =======================================================================
    # Strategi: Cari pola tutup kurung ')' diikuti oleh sebuah kata (alias).
    # Regex ini menangkap:
    #   ) alias
    #   ) AS alias
    # Kita abaikan keyword SQL (WHERE, GROUP, ORDER, ON, LEFT, JOIN, dll)
    
    potential_aliases = re.finditer(r"\)\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*)\b", code_only, flags=re.I)
    
    forbidden_alias_keywords = {
        "where", "group", "order", "limit", "having", 
        "join", "left", "right", "inner", "outer", "cross",
        "on", "and", "or", "union", "except", "intersect", "select"
    }
    
    for m in potential_aliases:
        candidate = m.group(1).lower()
        if candidate not in forbidden_alias_keywords and candidate not in keywords:
             exclude.add(candidate)
    # =======================================================================

    # 5) Output aliases: SELECT ... AS alias
    m_sel = re.search(r"\bselect\b(.*?)\bfrom\b", code_only, flags=re.I | re.S)
    if m_sel:
        select_part = m_sel.group(1)
        for m in re.finditer(r"\bas\s+(?:`([^`]+)`|([A-Za-z_]\w*))", select_part, flags=re.I):
            alias = (m.group(1) or m.group(2) or "").strip().lower()
            if alias:
                exclude.add(alias)

    # 6) Collect qualified usages: alias.col / table.col
    for m in re.finditer(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)\b",
        code_only,
        flags=re.I,
    ):
        qual = m.group(1)
        col = m.group(2)
        ql = (qual or "").lower()

        if not qual or not col:
            continue
        if ql in keywords:
            continue
        if ql in exclude:
            continue

        out.setdefault(qual, set()).add(col)

    # Clean up qualified occurrences
    code_no_qual = re.sub(
        r"\b[A-Za-z_][A-Za-z0-9_]*\s*\.\s*[A-Za-z_][A-Za-z0-9_]*\b",
        " ",
        code_only,
        flags=re.I,
    )

    # 7) Unqualified token scan
    tok_pat = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")
    for m in tok_pat.finditer(code_no_qual):
        tok = m.group(1)
        low = tok.lower()

        if low in keywords:
            continue
        if low in table_names:
            continue
        if low in exclude:
            continue
        
        # skip function call
        j = m.end()
        while j < len(code_no_qual) and code_no_qual[j].isspace():
            j += 1
        if j < len(code_no_qual) and code_no_qual[j] == "(":
            continue

        out.setdefault(None, set()).add(tok)

    return out

def validate_identifiers(sql: str, ctx: str) -> List[str]:
    allowed = _parse_allowed_identifiers(ctx)
    used = _identify_used_identifiers(sql)
    errs: List[str] = []

    schema_analyzer = None
    try:
        schema_analyzer = get_schema_analyzer()
    except Exception:
        schema_analyzer = None

    # --------------------------------------------------
    # Alias -> table mapping dari FROM / JOIN
    # --------------------------------------------------
    alias2table = _parse_from_join_aliases(sql)  # alias -> table (canonical)
    allowed_lower = {t.lower(): t for t in allowed.keys()}
    schema_cols_cache: Dict[str, Set[str]] = {}

    def _allowed_cols_for_table(tbl_name: str) -> Set[str]:
        """
        Build effective allowed columns for a table.
        Uses request-scoped ALLOWED IDENTIFIERS, and augments with full schema
        columns only for tables that already exist in ALLOWED.
        This prevents false "invalid column" from sparse retrieval context
        while still blocking unknown tables.
        """
        canon = (tbl_name or "").strip("`")
        if not canon:
            return set()
        key = canon.lower()
        if key in schema_cols_cache:
            return schema_cols_cache[key]

        # Start from ALLOWED block (request-scoped).
        cols = {str(c).strip("`").lower() for c in allowed.get(canon, set())}

        # Augment with real schema cols only if table itself is allowed.
        if canon in allowed and schema_analyzer is not None:
            try:
                cols.update(
                    str(c).strip("`").lower()
                    for c in schema_analyzer.get_table_columns(canon)
                )
            except Exception:
                pass

        schema_cols_cache[key] = cols
        return cols

    # --------------------------------------------------
    # Detect derived-table aliases: FROM ( ... ) AS t
    # --------------------------------------------------
    derived_aliases: Set[str] = set()
    for m in re.finditer(
        r"\b(?:from|join)\s*\([^)]*\)\s*(?:as\s+)?([A-Za-z_]\w*)\b",
        sql,
        flags=re.I | re.S,
    ):
        derived_aliases.add(m.group(1))

    derived_output_aliases: Set[str] = set()
    try:
        import sqlglot
        from sqlglot import exp

        ast = sqlglot.parse_one(sql, read="mysql")
        for subq in ast.find_all(exp.Subquery):
            inner_sel = subq.this if isinstance(getattr(subq, "this", None), exp.Select) else None
            if not isinstance(inner_sel, exp.Select):
                continue
            for expr in inner_sel.expressions or []:
                alias_name = str(getattr(expr, "alias_or_name", "") or "").strip("`")
                if alias_name:
                    derived_output_aliases.add(alias_name.lower())
    except Exception:
        derived_output_aliases = set()

    # --------------------------------------------------
    # Remap used qualifiers (alias -> real table)
    # --------------------------------------------------
    remapped: Dict[Optional[str], Set[str]] = {}
    for q, cols in used.items():
        if q is None:
            remapped.setdefault(None, set()).update(cols)
            continue

        q_norm = q.strip("`")

        # Jika qualifier adalah derived-table alias → BIARKAN
        if q_norm in derived_aliases:
            remapped.setdefault(q_norm, set()).update(cols)
            continue

        # Jika qualifier adalah alias tabel → map ke tabel asli
        t = alias2table.get(q_norm) or alias2table.get(q_norm.lower())
        if t:
            canon = allowed_lower.get(t.lower(), t)
            remapped.setdefault(canon, set()).update(cols)
        else:
            # Bisa jadi qualifier sudah nama tabel
            canon = allowed_lower.get(q_norm.lower(), q_norm)
            remapped.setdefault(canon, set()).update(cols)

    used = remapped

    # --------------------------------------------------
    # Tentukan apakah ini single-table query
    # --------------------------------------------------
    explicit_tables = {
        t for t in used.keys()
        if t is not None and t not in derived_aliases
    }

    single_table: Optional[str] = None
    if len(explicit_tables) == 1:
        single_table = next(iter(explicit_tables))
    elif len(explicit_tables) == 0 and len(allowed) == 1:
        single_table = next(iter(allowed.keys()))

    # --------------------------------------------------
    # Validate qualified identifiers
    # --------------------------------------------------
    for tbl, cols in used.items():
        if tbl is None:
            continue

        # SKIP derived tables (subquery aliases)
        if tbl in derived_aliases:
            continue

        if tbl not in allowed:
            errs.append(f"table `{tbl}` not in ALLOWED")
            continue

        eff_allowed = _allowed_cols_for_table(tbl)
        bad = [c for c in cols if c.lower() not in eff_allowed]
        if bad:
            errs.append(f"{tbl} -> invalid columns: {', '.join(bad)}")

    # --------------------------------------------------
    # Validate unqualified columns (single-table only)
    # --------------------------------------------------
    if single_table:
        allowed_cols = _allowed_cols_for_table(single_table)
        for col in used.get(None, set()):
            if col.lower() in derived_output_aliases:
                continue
            if col.lower() not in allowed_cols:
                errs.append(
                    f"{single_table} -> invalid column (unqualified): {col}"
                )

    return errs

def _question_asks_for_winner_row_object(q_text: str) -> bool:
    q = (q_text or "").lower().strip()
    if not q:
        return False

    generic_object_alt = "|".join(
        sorted((re.escape(t) for t in HELPERS_WINNER_ROW_GENERIC_OBJECT_TERMS if t), key=len, reverse=True)
    )
    if generic_object_alt and re.search(rf"\b(?:{generic_object_alt})\b", q, flags=re.I):
        return True

    superlative_pattern = re.compile(r"\b(?:largest|highest|latest|best|most|max|min|top)\b", flags=re.I)
    windows = [
        q[m.end(): m.end() + 96]
        for m in superlative_pattern.finditer(q)
    ] or [q]

    try:
        schema_tables = list(get_all_tables() or [])
    except Exception:
        schema_tables = []

    for table in schema_tables:
        candidates = _table_mention_candidates(table, include_synonyms=True)
        candidates.update(_table_inflected_mention_terms(table))
        patterns = [
            re.escape(cand).replace(r"\ ", r"\s+")
            for cand in candidates
            if cand
        ]
        if not patterns:
            continue
        entity_alt = "|".join(sorted(patterns, key=len, reverse=True))
        for window in windows:
            if re.search(rf"\b(?:{entity_alt})\b", window, flags=re.I):
                return True

    return False


def classify_query(question: str) -> str:
    q = " " + (question or "").strip().lower() + " "
    # ---------------------------------------------------------------------------
    # --- 0. DETEKSI ORDINAL RANKING (always "rank") ---
    # "second highest", "third largest", etc. ALWAYS need ROW_NUMBER subquery
    # ---------------------------------------------------------------------------
    ordinal_pattern = r"\b(second|third|fourth|fifth|2nd|3rd|4th|5th|kedua|ketiga|keempat|kelima)(?:\s+|-)+(highest|lowest|largest|smallest|biggest|top|bottom|best|worst|most|least|terbesar|terkecil|tertinggi|terendah)\b"
    ordinal_row_pattern = r"\b(second|third|fourth|fifth|2nd|3rd|4th|5th|kedua|ketiga|keempat|kelima)\b(?:\s+\w+){0,8}\s+(row|record|entry)\b"
    if re.search(ordinal_pattern, q) or (
        re.search(ordinal_row_pattern, q) and re.search(r"\b(sorted|ordered)\s+by\b", q)
    ):
        return "rank"
    # ---------------------------------------------------------------------------
    # --- 1. DETEKSI RANK (The "Winner-Row" Pattern) ---
    # Syarat: Mencari 'baris data' tunggal dalam sebuah kelompok.
    # Pattern: [Group Context] + [Superlative] + [Object, not count]
    # ---------------------------------------------------------------------------
    has_group = re.search(r"\b(per|each|every|within|partition)\b", q)
    has_superlative = re.search(r"\b(largest|highest|latest|best|most|max|min|top)\b", q)
    # Mencari objek row/entity secara generic, atau entity mention yang
    # muncul dekat superlative, bukan noun bisnis yang di-hardcode.
    asks_for_object = _question_asks_for_winner_row_object(q)
    is_math_agg = re.search(r"\b(count|sum|total|average|avg|number of)\b", q)

    if has_group and has_superlative and asks_for_object and not is_math_agg:
        return "rank"
    # ---------------------------------------------------------------------------
    # --- 2. DETEKSI AGGREGATE (The "Math/Summary" Pattern) ---
    # Ciri: Melakukan operasi matematika pada sekumpulan baris.
    # ---------------------------------------------------------------------------
    if is_math_agg or re.search(r"\b(classify|bands?|grouping|distribution)\b", q):
        return "aggregate"
    # ---------------------------------------------------------------------------
    # --- 3. DEFAULT (The "Data Retrieval" Pattern) ---
    # Jika hanya filter (WHERE) dan sorting (ORDER BY), masuk ke Lookup.
    # ---------------------------------------------------------------------------
    return "lookup"

def _get_qtype(state: "RAGState") -> str:
    q = (state.get("question") or "").strip()
    if not q:
        return "lookup"

    qtype = state.get("qtype") or ""
    if qtype:
        return qtype

    qtype = classify_query(q)
    state["qtype"] = qtype
    return qtype

def _get_requires_regex(state: "RAGState") -> bool:
    q = state.get("question", "") or ""
    cached_q = state.get("_requires_regex_q")

    # cache valid hanya jika question sama dan value bool
    if cached_q == q and isinstance(state.get("requires_regex"), bool):
        return state["requires_regex"]

    needs = _question_requires_regex(q)
    state["requires_regex"] = needs
    state["_requires_regex_q"] = q
    return needs

def _regex_requirement_ok(
    sql: str,
    requires_regex: bool,
    requires_boundary: bool,
    requires_structural: bool = False,
) -> bool:
    if not (requires_regex or requires_boundary or requires_structural):
        return True

    raw = sql or ""
    low = raw.lower()
    has_regexp = ("regexp" in low) or ("rlike" in low)
    if not has_regexp:
        return False

    if requires_boundary:
        if "(^|[^[:alnum:]_])" in raw:
            return True
        return ("[[:<:]]" in raw) and ("[[:>:]]" in raw)

    if requires_structural:
        if not (("^" in raw) and ("$" in raw)):
            return False
        structural_ok_patterns = [
            r"\[\^ \]\+(?:\s+\[\^ \]\+)+",
            r"\[:space:\]",
            r"\[\[:alnum:\]_\]\+(?:\s+\[\[:alnum:\]_\]\+)+",
        ]
        return any(re.search(pat, raw) for pat in structural_ok_patterns)

    return ("^" in raw) and ("$" in raw)
# -----------------------------------------------------------
# -------------------- HELPER CROSS MODULE ------------------
# ------------------------- retrieval.py --------------------
# -----------------------------------------------------------
def _mentions_id_like_for_probe(text: str) -> bool:
    """
    Versi khusus untuk probe:
    - Abaikan frasa 'number of' supaya tidak dianggap ID-like.
      Contoh: 'total number of orders' → bukan minta id.
    """
    t = (text or "").lower()
    # buang pola 'number of', 'total number of', dll
    t = re.sub(r"\b(total\s+)?number\s+of\b", "", t)
    return bool(re.search(r"\b(id|identifier|nomor|no|code|kode)\b", t))

def build_dynamic_probe(question: str, qtype: Optional[str] = None, max_tokens: int = 16) -> str:
    """
    Build a compact probe tail for embedding retrieval.

    Design goals:
    - Keep the tail short.
    - Add only schema-grounded tokens that are relevant to the question.
    - Avoid ClassicModels-specific table branching.
    """
    q_raw = question or ""
    q = q_raw.lower()
    toks_q = q.split()
    q_tokens = re.findall(r"[a-z0-9]+", q)

    try:
        analyzer = get_schema_analyzer()
    except Exception:
        analyzer = None

    transaction_tables = {
        str(t).strip().lower()
        for t in (get_transaction_tables() or [])
        if str(t).strip()
    }

    # Keep mention detection conservative for probe expansion.
    def _strict_table_mentioned(q_text: str, table: str) -> bool:
        t = (table or "").strip().lower()
        if not t:
            return False

        candidates = {t}
        singular = get_singular_form(t)
        if singular:
            candidates.add(singular.lower())
        if "_" in t:
            candidates.add(t.replace("_", " "))
        if singular and "_" in singular:
            candidates.add(singular.replace("_", " "))

        # Split concatenated table names using schema roots:
        # productlines -> product lines, orderdetails -> order details
        for base in list(candidates):
            compact = base.replace(" ", "")
            if not compact.isalpha():
                continue
            for root in _TABLE_SINGULAR_ROOTS:
                if not root or root == compact:
                    continue
                if compact.startswith(root):
                    tail = compact[len(root):]
                    if len(tail) >= 3 and tail.isalpha():
                        candidates.add(f"{root} {tail}")

        for cand in candidates:
            pattern = re.escape(cand).replace(r"\ ", r"\s+")
            m = re.search(rf"\b{pattern}\b", q_text)
            if not m:
                continue
            if cand == "order":
                tail = q_text[m.end(): m.end() + 12]
                if re.match(r"\s+by\b", tail):
                    continue
            return True
        if _iter_inflected_entity_mentions(q_text, t):
            return True
        return False

    mentioned_tables_dynamic = {
        t for t in get_all_tables()
        if _strict_table_mentioned(q, t)
    }

    semantic_stopwords = {
        "a", "an", "the", "of", "for", "to", "in", "on", "at", "by", "with",
        "and", "or", "from", "per", "each", "every", "all", "any", "this",
        "that", "these", "those", "show", "list", "return", "display", "give",
        "find", "fetch", "get", "table", "column", "data", "value", "values",
    }
    metric_roles = {"metric", "measure", "amount", "quantity"}

    def _semantic_tokens(text: str) -> Set[str]:
        tokens = {
            tok
            for tok in re.findall(r"[a-z0-9]+", str(text or "").lower())
            if tok and len(tok) >= 2 and tok not in semantic_stopwords and not tok.isdigit()
        }
        return tokens

    def _column_semantic_terms(table: str, column: str) -> Set[str]:
        terms = set()
        col = str(column or "").strip()
        if not col:
            return terms
        terms.update(_semantic_tokens(col))
        try:
            for syn in get_column_synonyms(table, col) or []:
                terms.update(_semantic_tokens(syn))
        except Exception:
            pass
        try:
            for variant in get_column_entity_variants(table, col) or []:
                terms.update(_semantic_tokens(variant))
        except Exception:
            pass
        return terms

    def _is_metric_id_like(column: str) -> bool:
        col_lower = str(column or "").strip().lower()
        if not col_lower:
            return False
        return bool(
            col_lower.endswith("id")
            or col_lower.endswith("number")
            or col_lower.endswith("code")
            or col_lower.endswith("key")
        )

    def _ordered_unique(items: List[str]) -> List[str]:
        seen: Set[str] = set()
        ordered: List[str] = []
        for item in items:
            tok = str(item or "").strip().lower()
            if not tok or tok in seen:
                continue
            seen.add(tok)
            ordered.append(tok)
        return ordered

    schema_metric_terms: List[str] = []
    metric_scope_tables = sorted(mentioned_tables_dynamic) or sorted(get_all_tables())
    for table in metric_scope_tables:
        anchor_cols = {str(c or "").strip().lower() for c in (get_anchor_columns(table) or [])}
        for col in get_aggregatable_columns(table):
            col_l = str(col or "").strip().lower()
            role = get_column_role(table, col)
            if role not in metric_roles and col_l not in anchor_cols and _is_metric_id_like(col):
                continue
            terms = _column_semantic_terms(table, col)
            if not terms:
                continue
            schema_metric_terms.extend(sorted(terms))
    schema_metric_terms = _ordered_unique(schema_metric_terms)
    schema_metric_term_set = set(schema_metric_terms)
    question_metric_fallback_terms = {
        tok for tok in q_tokens
        if tok in (HELPERS_PROBE_GENERIC_METRIC_TERMS | HELPERS_PROBE_LOCALIZED_METRIC_TERMS)
    }
    metric_query_terms = set(schema_metric_term_set)
    metric_query_terms.update(question_metric_fallback_terms)
    if not metric_query_terms:
        metric_query_terms.update(HELPERS_PROBE_GENERIC_METRIC_TERMS)
        metric_query_terms.update(HELPERS_PROBE_LOCALIZED_METRIC_TERMS)
    metric_terms_in_question = _ordered_unique([tok for tok in q_tokens if tok in metric_query_terms])

    def _display_name_column_for_probe(table: str) -> Optional[str]:
        name_col = get_name_column(table)
        if not name_col:
            return None
        col_lower = name_col.lower()
        role = get_column_role(table, name_col)
        if role == "name":
            return name_col
        if any(part in col_lower for part in ("firstname", "lastname", "surname", "givenname")):
            return None
        return name_col

    def _column_mentioned_in_question(table: str, column: str) -> bool:
        for variant in get_column_entity_variants(table, column):
            pattern = re.escape(str(variant).strip().lower()).replace(r"\ ", r"\s+")
            if pattern and re.search(rf"\b{pattern}\b", q):
                return True
        return False

    def _rank_columns_for_probe(
        table: str,
        columns: List[str],
        *,
        prefer_metric: bool = False,
        prefer_temporal: bool = False,
    ) -> List[str]:
        ranked: List[Tuple[int, int, str]] = []
        for idx, column in enumerate(columns):
            col = str(column or "").strip()
            if not col:
                continue
            role = get_column_role(table, col)
            anchor_cols = {str(c or "").strip().lower() for c in (get_anchor_columns(table) or [])}
            semantic_terms = _column_semantic_terms(table, col)
            overlap_terms = semantic_terms.intersection(metric_terms_in_question)
            score = 0
            if _column_mentioned_in_question(table, col):
                score += 5
            if prefer_metric:
                if role in metric_roles or str(col).strip().lower() in anchor_cols:
                    score += 3
                if overlap_terms:
                    score += min(len(overlap_terms), 2) * 2
                elif semantic_terms.intersection(metric_query_terms):
                    score += 1
            if prefer_temporal:
                if role in {"date", "datetime", "timestamp", "time"}:
                    score += 3
                col_lower = col.lower()
                if ("date" in col_lower) or ("time" in col_lower) or col_lower.endswith("at"):
                    score += 2
            ranked.append((score, idx, col))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [col for _, _, col in ranked]

    def _is_semantic_metric_column(table: str, column: str) -> bool:
        col = str(column or "").strip()
        if not col:
            return False
        role = get_column_role(table, col)
        if role in metric_roles:
            return True
        col_l = col.lower()
        anchor_cols = {str(c or "").strip().lower() for c in (get_anchor_columns(table) or [])}
        if col_l in anchor_cols:
            return True
        if _is_metric_id_like(col):
            return False
        return bool(_column_semantic_terms(table, col).intersection(metric_query_terms))

    def _table_probe_label(table: str) -> Optional[str]:
        singular = get_singular_form(table)
        if singular:
            label = singular.replace("_", " ").strip()
            if label:
                return label
        return str(table or "").replace("_", " ").strip() or None

    def _table_has_geo_variant(table: str, targets: Set[str]) -> bool:
        if not analyzer:
            return False
        table_lower = str(table or "").strip().lower()
        if not table_lower:
            return False
        try:
            columns = analyzer.get_table_columns(table_lower)
        except Exception:
            columns = []
        wanted = {str(t).strip().lower() for t in targets if str(t).strip()}
        for column in columns:
            variants = get_column_entity_variants(table_lower, column)
            for variant in variants:
                tokens = re.split(r"[\s_]+", str(variant or "").strip().lower())
                if any(tok in wanted for tok in tokens):
                    return True
        return False

    def _append_table_probe_tokens(table: str, tokens: List[str]) -> None:
        table_lower = str(table or "").strip().lower()
        if not table_lower:
            return

        label = _table_probe_label(table_lower)
        if label:
            tokens.append(label)

        id_col = get_id_column(table_lower)
        name_col = _display_name_column_for_probe(table_lower)
        if id_like:
            if id_col:
                tokens.append(id_col)
        else:
            if name_col:
                tokens.append(name_col)
            elif id_col:
                tokens.append(id_col)

        metric_candidates = [
            col
            for col in get_aggregatable_columns(table_lower)
            if _is_semantic_metric_column(table_lower, col)
        ]
        explicit_metric_match = any(
            _column_mentioned_in_question(table_lower, col)
            for col in metric_candidates
        )
        if explicit_metric_match:
            metric_cols = _rank_columns_for_probe(
                table_lower,
                metric_candidates,
                prefer_metric=True,
            )
            for col in metric_cols[:2]:
                col_lower = col.lower()
                role = get_column_role(table_lower, col)
                if col in {id_col, name_col}:
                    continue
                if role == "id":
                    continue
                if col_lower.endswith("id") or col_lower.endswith("number") or col_lower.endswith("code"):
                    continue
                tokens.append(col)

        date_candidates = get_filterable_date_columns(table_lower)
        explicit_temporal_match = any(
            _column_mentioned_in_question(table_lower, col)
            for col in date_candidates
        )
        wants_temporal_columns = explicit_temporal_match or (
            table_lower in transaction_tables and qtype != "lookup"
        )
        if wants_temporal_columns:
            date_cols = _rank_columns_for_probe(
                table_lower,
                date_candidates,
                prefer_temporal=True,
            )
            for col in date_cols[:1]:
                if col not in {id_col, name_col}:
                    tokens.append(col)

    # Base lexical signals.
    has_metric_word = bool(metric_terms_in_question)
    has_count_word = any(k in q for k in [
        "count", "berapa banyak", "how many"
    ])
    has_time_word = any(k in q for k in [
        "today", "yesterday", "last", "this", "bulan", "minggu", "tahun",
        "quarter", "month", "week", "year", "tanggal", "hari",
        "mingguan", "bulanan", "202", "201"
    ])

    id_like = _mentions_id_like_for_probe(q)

    # ---------------------------------------------------------------------------
    # 1) Build seed tokens
    # ---------------------------------------------------------------------------
    base: List[str] = []
    if qtype == "lookup":
        base = []
    elif qtype in {"aggregate_by", "aggregate_all"}:
        if has_metric_word:
            base += metric_terms_in_question[:3] or schema_metric_terms[:3] or ["amount", "price"]
        if has_count_word:
            base += ["count"]
        if has_time_word:
            base += ["date", "year", "month"]
    else:
        if has_time_word:
            base += ["date"]

    # ---------------------------------------------------------------------------
    # 2) Schema-driven table and column hints
    # ---------------------------------------------------------------------------
    for table in sorted(mentioned_tables_dynamic):
        _append_table_probe_tokens(table, base)

    # ---------------------------------------------------------------------------
    # 2b) Geography hints
    # ---------------------------------------------------------------------------
    mentions_country_word = any(k in q for k in ["country", "negara"])
    mentions_city_word = any(k in q for k in ["city", "kota"])
    mentions_region_word = any(k in q for k in ["region", "wilayah", "territory", "territories"])
    has_country_column = any(
        _table_has_geo_variant(table, {"country", "nation"})
        for table in mentioned_tables_dynamic
    )
    has_city_column = any(
        _table_has_geo_variant(table, {"city", "municipality", "town"})
        for table in mentioned_tables_dynamic
    )
    geo_tokens: List[str] = []

    if mentions_country_word and not mentions_city_word:
        geo_tokens.append("country")
    elif mentions_city_word and not mentions_country_word:
        geo_tokens.append("city")
    elif mentions_country_word and mentions_city_word:
        geo_tokens += ["country", "city"]
    elif mentions_region_word:
        if has_city_column:
            geo_tokens.append("city")
        if has_country_column:
            geo_tokens.append("country")

    if re.search(r"\bfrom\s+\w+\b", q_raw, flags=re.I) and has_country_column:
        if "country" not in geo_tokens:
            geo_tokens.append("country")

    base += geo_tokens

    # Generic name hint.
    if any(k in q for k in ["name", "nama"]):
        base += ["name"]

    # Very short questions still need a tiny semantic tail.
    if len(toks_q) <= 2 and not base:
        base += ["amount", "date"]

    # ---------------------------------------------------------------------------
    # 3) Deduplicate and truncate
    # ---------------------------------------------------------------------------
    seen = set()
    tokens: List[str] = []
    for t in base:
        t_norm = t.strip()
        if not t_norm:
            continue
        if t_norm not in seen:
            seen.add(t_norm)
            tokens.append(t_norm)
        if len(tokens) >= max_tokens:
            break
    # Kalau akhirnya tidak ada token tambahan, pakai question apa adanya
    if not tokens:
        return q_raw.strip()
    probe_tail = " ".join(tokens).strip()
    return f"{q_raw.strip()} {probe_tail}".strip()

def _table_mention_candidates(table: str, include_synonyms: bool = True) -> Set[str]:
    """
    Build mention candidates from schema metadata without table-specific hardcoding.
    """
    t = (table or "").strip().lower()
    if not t:
        return set()

    candidates = {t}
    singular = get_singular_form(t)
    if singular:
        candidates.add(singular.lower())

    if "_" in t:
        candidates.add(t.replace("_", " "))
    if singular and "_" in singular:
        candidates.add(singular.replace("_", " "))

    # Split concatenated names using other schema roots:
    # productlines -> product lines, orderdetails -> order details
    for base in list(candidates):
        compact = base.replace(" ", "")
        if not compact.isalpha():
            continue
        for root in _TABLE_SINGULAR_ROOTS:
            if not root or root == compact:
                continue
            if compact.startswith(root):
                tail = compact[len(root):]
                if len(tail) >= 3 and tail.isalpha():
                    candidates.add(f"{root} {tail}")

    if include_synonyms:
        for syn in get_table_synonyms(t):
            syn_low = str(syn).strip().lower()
            if syn_low:
                candidates.add(syn_low)

    return {c for c in candidates if c}

def _table_mentioned(q_text: str, table: str, include_synonyms: bool = True) -> bool:
    q = (q_text or "").lower()
    if not q:
        return False
    for cand in _table_mention_candidates(table, include_synonyms=include_synonyms):
        pattern = re.escape(cand).replace(r"\ ", r"\s+")
        if re.search(rf"\b{pattern}\b", q):
            return True
    if _iter_inflected_entity_mentions(q, table):
        return True
    return False


def _table_inflected_mention_terms(table: str) -> Set[str]:
    """
    Build verb-like inflected terms for entity mentions.
    Example: orders -> ordered/ordering.
    """
    t = (table or "").strip().lower()
    if not t:
        return set()

    stems: Set[str] = set()
    singular = get_singular_form(t)
    if singular and singular.isalpha():
        stems.add(singular.lower())

    # Include compact single-token candidates/synonyms as optional stems.
    for cand in _table_mention_candidates(t, include_synonyms=True):
        token = str(cand or "").strip().lower().replace(" ", "")
        if not token or not token.isalpha():
            continue
        if len(token) >= 3:
            stems.add(token)
        if token.endswith("s") and len(token) >= 4:
            stems.add(token[:-1])
        if token.endswith("ment") and len(token) >= 6:
            stems.add(token[:-4])

    terms: Set[str] = set()
    for stem in list(stems):
        s = stem.strip().lower()
        if not s or len(s) < 3:
            continue
        if s.endswith("ed") or s.endswith("ing"):
            terms.add(s)
        if s.endswith("e"):
            terms.add(f"{s}d")
            if len(s) > 1:
                terms.add(f"{s[:-1]}ing")
        else:
            terms.add(f"{s}ed")
            terms.add(f"{s}ing")
        if s.endswith("y") and len(s) > 3:
            terms.add(f"{s[:-1]}ied")

    # Common irregulars from derived verb stems.
    if "pay" in stems:
        terms.add("paid")

    return {term for term in terms if term and len(term) >= 4}


def _base_form_from_inflected(term: str) -> str:
    t = str(term or "").strip().lower()
    if not t:
        return ""
    if t == "paid":
        return "pay"
    if t.endswith("ied") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith("ing") and len(t) > 5:
        base = t[:-3]
        if base.endswith(base[-1:] * 2):
            base = base[:-1]
        return base
    if t.endswith("ed") and len(t) > 4:
        base = t[:-2]
        if base.endswith("i") and len(base) > 1:
            return base[:-1] + "y"
        return base
    return t


def _question_compat_existence_verbs(question: str) -> Set[str]:
    if not DOMAIN_HEURISTICS_ENABLED:
        return set()
    q = (question or "").lower().strip()
    if not q:
        return set()

    q_tokens = set(re.findall(r"[a-z0-9]+", q))
    matched: Set[str] = set()
    for verb in HELPERS_EXISTENCE_COMPAT_VERBS:
        v = str(verb or "").strip().lower()
        if not v:
            continue
        if " " in v:
            pattern = re.escape(v).replace(r"\ ", r"\s+")
            if re.search(rf"\b{pattern}\b", q, flags=re.I):
                matched.add(v)
            continue
        if v in q_tokens:
            matched.add(v)
    return matched


def _entity_support_verb_terms(table: str, question: str = "") -> Set[str]:
    verbs: Set[str] = set(HELPERS_EXISTENCE_SUPPORT_VERBS_GENERIC)
    inflected_terms = _table_inflected_mention_terms(table)
    verbs.update(inflected_terms)
    verbs.update(
        base
        for base in (_base_form_from_inflected(term) for term in inflected_terms)
        if base and len(base) >= 3
    )
    # Backward-compatible fallback only when table-derived support verbs are not
    # already grounded by the question wording.
    q = (question or "").lower().strip()
    has_table_grounded_match = False
    if q:
        q_tokens = set(re.findall(r"[a-z0-9]+", q))
        for verb in verbs:
            vv = str(verb or "").strip().lower()
            if not vv:
                continue
            if " " in vv:
                pattern = re.escape(vv).replace(r"\ ", r"\s+")
                if re.search(rf"\b{pattern}\b", q, flags=re.I):
                    has_table_grounded_match = True
                    break
            elif vv in q_tokens:
                has_table_grounded_match = True
                break
    if not has_table_grounded_match:
        compat_verbs = _question_compat_existence_verbs(question)
        if DOMAIN_HEURISTICS_ENABLED:
            verbs.update(compat_verbs if question else HELPERS_EXISTENCE_COMPAT_VERBS)
    return {verb for verb in verbs if verb}


def _is_locally_negated(q_text: str, start_idx: int, window_chars: int = 56) -> bool:
    left = (q_text or "")[max(0, start_idx - window_chars):start_idx]
    if not left:
        return False
    return bool(
        re.search(
            r"\b(?:no|without|never|not)\b(?:[\s,;:()'\"-]+\w+){0,4}\s*$",
            left,
            flags=re.I,
        )
    )


def _iter_inflected_entity_mentions(q_text: str, table: str) -> List[Tuple[str, bool]]:
    q = (q_text or "").lower()
    if not q:
        return []

    mentions: List[Tuple[str, bool]] = []
    terms = sorted(_table_inflected_mention_terms(table), key=len, reverse=True)
    for term in terms:
        pat = rf"\b{re.escape(term)}\b"
        for m in re.finditer(pat, q):
            # Avoid treating ORDER BY grammar as orders-entity mention.
            if term == "ordered":
                tail = q[m.end(): m.end() + 12]
                if re.match(r"\s+by\b", tail):
                    continue
            negated = _is_locally_negated(q, m.start())
            mentions.append((term, negated))
    return mentions


def _has_non_negated_pattern_match(q_text: str, pattern: str) -> bool:
    q = (q_text or "").lower()
    if not q or not pattern:
        return False
    for m in re.finditer(pattern, q, flags=re.I):
        frag = m.group(0) or ""
        if re.search(r"\b(?:no|without|never|not)\b", frag, flags=re.I):
            continue
        if _is_locally_negated(q, m.start()):
            continue
        return True
    return False


def _question_targets_table_exclusion(question: str, table: str) -> bool:
    """
    Detect when a question expresses exclusion/absence intent for a specific table.
    Examples:
    - "with no orders"
    - "without any payments"
    - "have never placed orders"
    """
    q = (question or "").lower()
    if not q:
        return False

    candidates = _table_mention_candidates(table, include_synonyms=True)
    if not candidates:
        return False

    entity_alt = "|".join(
        sorted((re.escape(c).replace(r"\ ", r"\s+") for c in candidates), key=len, reverse=True)
    )

    # Negation followed by entity mention.
    if re.search(
        rf"\b(?:no|without(?:\s+any)?|never|not)\b(?:[\s,;:()'\"-]+\w+){{0,10}}[\s,;:()'\"-]+(?:{entity_alt})\b",
        q,
        flags=re.I,
    ):
        return True

    # Narrow reverse form to avoid cross-entity false positives like:
    # "payment ... never ... orders" (where "never" targets orders, not payments).
    if re.search(
        rf"\b(?:{entity_alt})\b(?:[\s,;:()'\"-]+\w+){{0,3}}[\s,;:()'\"-]+\b(?:with\s+no|without(?:\s+any)?)\b",
        q,
        flags=re.I,
    ):
        return True

    # Verb-like negated mention: "never ordered", "not paid", etc.
    for _, is_negated in _iter_inflected_entity_mentions(q, table):
        if is_negated:
            return True

    return False


def _question_targets_table_existence(question: str, table: str) -> bool:
    """
    Detect explicit existence intent for a specific table/entity:
    - "at least one payment"
    - "with payments"
    - "have made payments"
    """
    q = (question or "").lower()
    if not q:
        return False
    # Ignore quoted label text (e.g., "'Paid'/'Unpaid'") to avoid false
    # existence triggers from output aliases/status literals.
    q_sem = re.sub(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"", " ", q)

    has_exclusion = _question_targets_table_exclusion(q_sem, table)
    inflected_mentions = _iter_inflected_entity_mentions(q_sem, table)
    has_non_negated_inflected = any(not is_negated for _, is_negated in inflected_mentions)

    candidates = _table_mention_candidates(table, include_synonyms=True)
    if not candidates:
        return has_non_negated_inflected

    entity_alt = "|".join(
        sorted((re.escape(c).replace(r"\ ", r"\s+") for c in candidates), key=len, reverse=True)
    )

    # Quantified existence: "at least one <entity>"
    quantified_pattern = (
        rf"\b(?:at\s+least\s+one|one\s+or\s+more|any)\b"
        rf"(?:[\s,;:()'\"-]+\w+){{0,8}}[\s,;:()'\"-]+(?:{entity_alt})\b"
    )
    has_quantified_existence = _has_non_negated_pattern_match(q_sem, quantified_pattern)

    # Avoid false positives for metric-request phrasing:
    # "with total payment amount", "having average orders", etc.
    metric_connector_match = re.search(
        rf"\b(?:with|having)\b(?:[\s,;:()'\"-]+\w+){{0,3}}[\s,;:()'\"-]+"
        rf"\b(?:total|sum|count|average|avg|amount|number)\b"
        rf"(?:[\s,;:()'\"-]+\w+){{0,6}}[\s,;:()'\"-]+(?:{entity_alt})\b",
        q_sem,
        flags=re.I,
    )

    # Connector pattern: "with/having <entity>"
    connector_pattern = (
        rf"\b(?:with|having)\b"
        rf"(?:[\s,;:()'\"-]+\w+){{0,6}}[\s,;:()'\"-]+(?:{entity_alt})\b"
    )
    has_connector_existence = bool(
        not metric_connector_match and _has_non_negated_pattern_match(q_sem, connector_pattern)
    )

    # Verb-based existence mention.
    support_verbs = sorted(_entity_support_verb_terms(table, q_sem), key=len, reverse=True)
    support_verb_alt = "|".join(re.escape(v) for v in support_verbs if v)
    verb_pattern = (
        rf"\b(?:{support_verb_alt})\b"
        rf"(?:[\s,;:()'\"-]+\w+){{0,8}}[\s,;:()'\"-]+(?:{entity_alt})\b"
    )
    has_verb_existence = _has_non_negated_pattern_match(q_sem, verb_pattern)

    has_positive_existence = bool(
        has_quantified_existence
        or has_connector_existence
        or has_verb_existence
        or has_non_negated_inflected
    )

    # Pure exclusion query (no positive evidence) should not force INNER JOIN.
    if has_exclusion and not has_positive_existence:
        return False

    return has_positive_existence


def _question_has_direct_entity_existence_signal(question: str, table: str) -> bool:
    """
    Narrower existence detector used only for join-hint guidance.
    It captures explicit row-presence intent ("with payments", "at least one order")
    but intentionally ignores metric/status-only mentions such as
    "total payment amount" or "payment_status".
    """
    q = (question or "").lower()
    if not q:
        return False
    q_sem = re.sub(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"", " ", q)

    candidates = _table_mention_candidates(table, include_synonyms=True)
    if not candidates:
        return False

    entity_alt = "|".join(
        sorted((re.escape(c).replace(r"\ ", r"\s+") for c in candidates), key=len, reverse=True)
    )

    quantified_pattern = (
        rf"\b(?:at\s+least\s+one|one\s+or\s+more|any)\b"
        rf"(?:[\s,;:()'\"-]+\w+){{0,8}}[\s,;:()'\"-]+(?:{entity_alt})\b"
    )
    if _has_non_negated_pattern_match(q_sem, quantified_pattern):
        return True

    metric_connector_match = re.search(
        rf"\b(?:with|having)\b(?:[\s,;:()'\"-]+\w+){{0,3}}[\s,;:()'\"-]+"
        rf"\b(?:total|sum|count|average|avg|amount|number|status)\b"
        rf"(?:[\s,;:()'\"-]+\w+){{0,6}}[\s,;:()'\"-]+(?:{entity_alt})\b",
        q_sem,
        flags=re.I,
    )
    connector_pattern = (
        rf"\b(?:with|having)\b"
        rf"(?:[\s,;:()'\"-]+\w+){{0,6}}[\s,;:()'\"-]+(?:{entity_alt})\b"
    )
    if not metric_connector_match and _has_non_negated_pattern_match(q_sem, connector_pattern):
        return True

    inflected_mentions = _iter_inflected_entity_mentions(q_sem, table)
    return any(not is_negated for _, is_negated in inflected_mentions)


def _is_simple_single_table_lookup(question: str, table: str) -> bool:
    """
    Generic non-aggregate lookup detector for a single subject table.
    """
    q = (question or "").lower()
    t = (table or "").strip().lower()
    if not q or not t:
        return False

    if not _table_mentioned(q, t, include_synonyms=True):
        return False

    aggregate_terms = ["total", "sum", "avg", "average", "mean", "count", "berapa banyak", "how many"]
    if any(term in q for term in aggregate_terms):
        return False

    return True


def _related_tables_for(table: str) -> Set[str]:
    t = (table or "").strip().lower()
    if not t:
        return set()

    related: Set[str] = set()
    try:
        fk_map = get_fk_relationships() or {}
    except Exception:
        fk_map = {}

    for src_table, src_targets in fk_map.items():
        src = (src_table or "").strip().lower()
        if not src:
            continue
        for target_table in (src_targets or {}).values():
            tgt = (target_table or "").strip().lower()
            if not tgt:
                continue
            if src == t and tgt != t:
                related.add(tgt)
            if tgt == t and src != t:
                related.add(src)

    return related


def _is_simple_table_identifier_lookup(question: str, table: str) -> bool:
    """
    Generic identifier-list lookup detector for a single table.
    """
    q = (question or "").lower()
    t = (table or "").strip().lower()
    if not _is_simple_single_table_lookup(question, t):
        return False

    id_col = get_id_column(t)
    id_col_lower = (id_col or "").lower()
    id_col_spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", id_col or "").replace("_", " ").lower()
    entity_candidates = _table_mention_candidates(t, include_synonyms=True)
    entity_alt = "|".join(
        sorted((re.escape(c).replace(r"\ ", r"\s+") for c in entity_candidates), key=len, reverse=True)
    )
    mentions_identifier = bool(
        (id_col_lower and re.search(rf"\b{re.escape(id_col_lower)}\b", q))
        or (id_col_spaced and id_col_spaced in q)
        or (
            entity_alt
            and re.search(
                rf"\b(?:{entity_alt})\b(?:[\s,;:()'\"-]+\w+){{0,2}}[\s,;:()'\"-]+(?:id|ids|number|numbers|code|codes|nomor|kode|no)\b",
                q,
                flags=re.I,
            )
        )
    )
    if not mentions_identifier:
        return False

    for related_table in _related_tables_for(t):
        if _table_mentioned(q, related_table, include_synonyms=True):
            return False

    return True


# ======================================================================
# ===================== POSTPROCESS HEURISTIC ==========================
# ======================================================================
def pick_preferred_anchor_from_ranked_cols(
    ranked_cols: List[tuple[str, str]],
    top_n: int = 12,
) -> Optional[str]:
    if not ranked_cols:
        return None

    head = ranked_cols[:max(1, top_n)]

    # 1) signal-based (lebih kuat) - use dynamic transaction tables
    for tbl in TRANSACTION_TABLES:
        sig = _ANCHOR_SIGNALS.get(tbl, set())
        for t, c in head:
            if (t or "").lower() == tbl and c in sig:
                return tbl

    # 2) fallback: first transaction table in head
    for t, _ in head:
        tl = (t or "").lower()
        if tl in TRANSACTION_TABLES:
            return tl

    return None

def normalize_anchor_for_simple_inner_join(sql: str, preferred_anchor: Optional[str]) -> str:
    if not sql or not preferred_anchor:
        return sql

    if not has_join(sql):
        return sql
    if has_non_inner_join(sql):
        return sql
    if join_count(sql) != 1:
        return sql

    pat = re.compile(
        r"""
        ^(?P<prefix>.*?)
        \bFROM\s+(?P<t1>`?[A-Za-z_][A-Za-z0-9_]*`?)\s+(?:AS\s+)?(?P<a1>[A-Za-z_][A-Za-z0-9_]*)
        \s+\b(?:INNER\s+)?JOIN\s+(?P<t2>`?[A-Za-z_][A-Za-z0-9_]*`?)\s+(?:AS\s+)?(?P<a2>[A-Za-z_][A-Za-z0-9_]*)
        \s+\bON\s+(?P<on>.*?)
        (?P<tail>\s+(?:WHERE|GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT)\b.*?|;?\s*)$
        """,
        flags=re.IGNORECASE | re.DOTALL | re.VERBOSE,
    )
    m = pat.match(sql.strip())
    if not m:
        return sql

    def _strip_ticks(x: str) -> str:
        return (x or "").strip().strip("`")

    t1 = _strip_ticks(m.group("t1")).lower()
    t2 = _strip_ticks(m.group("t2")).lower()
    pref = preferred_anchor.lower()

    if pref == t1:
        return sql
    if pref != t2:
        return sql

    prefix = m.group("prefix")
    t1_raw, a1 = m.group("t1"), m.group("a1")
    t2_raw, a2 = m.group("t2"), m.group("a2")
    on_clause = m.group("on").strip()
    tail = m.group("tail") or ""

    return f"{prefix}FROM {t2_raw} {a2} JOIN {t1_raw} {a1} ON {on_clause}{tail}"

def sanitize_notes(sql: str, notes: str) -> str:
    if not notes:
        return notes
    if not has_join(sql or ""):
        return "No joins used."
    return notes.strip()

def _fix_undefined_alias_typos(sql: str) -> str:
    """
    Fix undefined alias typos where LLM hallucinates similar aliases.
    Example: p2.amount when only 'p' is defined -> p.amount
    
    Strategy:
    1. Find all defined table/subquery aliases
    2. Find all alias.column references
    3. If alias is undefined but similar to a defined alias, replace it
    """
    sql_up = sql.upper()
    
    # Collect defined aliases from:
    # 1. "... FROM table AS alias" or "... FROM table alias"
    # 2. "... JOIN table AS alias" or "... JOIN table alias"  
    # 3. "... FROM (...) AS alias" or "... FROM (...) alias"
    
    defined_aliases = set()
    
    # Pattern 1: FROM/JOIN table AS alias
    for m in re.finditer(r'\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\b', sql, re.IGNORECASE):
        tbl_name = m.group(1).lower()
        alias = m.group(2).lower()
        # Skip if "alias" is a SQL keyword
        if alias not in ('where', 'group', 'order', 'having', 'limit', 'on', 'and', 'or', 'as', 'join', 'inner', 'left', 'right', 'select'):
            defined_aliases.add(alias)
    
    # Pattern 2: FROM (...) AS alias or FROM (...) alias  
    for m in re.finditer(r'\)\s*(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\b', sql, re.IGNORECASE):
        alias = m.group(1).lower()
        if alias not in ('where', 'group', 'order', 'having', 'limit', 'on', 'and', 'or', 'as'):
            defined_aliases.add(alias)
    
    if not defined_aliases:
        return sql
    
    # Find all alias.column patterns used
    used_aliases = set()
    # Get all valid table names dynamically from schema
    all_tables_lower = {t.lower() for t in get_all_tables()}
    for m in re.finditer(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\b', sql, re.IGNORECASE):
        used_alias = m.group(1).lower()
        # Skip aliases that match actual table names (dynamic, schema-agnostic)
        if used_alias not in all_tables_lower:
            used_aliases.add(used_alias)
    
    # Find undefined aliases and try to match to defined ones
    result = sql
    for used in used_aliases:
        if used not in defined_aliases:
            # Try to find a similar defined alias
            # Strategy: If used is like "p2" and defined has "p", map p2 -> p
            for defined in defined_aliases:
                # Check if used is defined + digits (e.g., p2 when p exists)
                if used.startswith(defined) and used[len(defined):].isdigit():
                    print(f"[FIX_ALIAS] Correcting typo: {used} -> {defined}")
                    # Replace all occurrences of the undefined alias
                    result = re.sub(
                        rf'\b{re.escape(used)}\.',
                        f'{defined}.',
                        result,
                        flags=re.IGNORECASE
                    )
                    break
                # Check if used differs by just one char at end (e.g., pmt2 vs pmt)
                elif len(used) == len(defined) + 1 and used[:-1] == defined and used[-1].isdigit():
                    print(f"[FIX_ALIAS] Correcting typo: {used} -> {defined}")
                    result = re.sub(
                        rf'\b{re.escape(used)}\.',
                        f'{defined}.',
                        result,
                        flags=re.IGNORECASE
                    )
                    break
    
    return result

def _normalize_subquery_alias_to_standard(sql: str) -> str:
    """
    Normalize subquery aliases - DEPRECATED: Now just returns SQL unchanged.

    Previously normalized aliases to hardcoded standards (pmt, emp, cust, etc.)
    but this caused issues when LLM used different aliases in different places.

    The fix_inconsistent_subquery_alias function now handles the case where
    LLM hallucinates inconsistent aliases within a single query.
    """
    # Simply return the SQL unchanged - let LLM choose its own aliases
    return sql

def _fix_missing_space_before_clauses(sql: str) -> str:
    # Normalize whitespace before SQL clauses
    sql = re.sub(r"\s+(?=(ORDER\s+BY|GROUP\s+BY|HAVING|LIMIT|WHERE)\b)", " ", sql, flags=re.I)
    # Add space after punctuation before clauses
    sql = re.sub(
        r"([)\],;`])(?=(ORDER\s+BY|GROUP\s+BY|HAVING|LIMIT|WHERE)\b)",
        r"\1 ",
        sql,
        flags=re.I,
    )
    # FIX: Add space between identifier and uppercase SQL keyword (NO re.I flag)
    # This splits "customerNameHAVING" but NOT "creditLimit" (because 'imit' != 'LIMIT')
    sql = re.sub(r"(\b[A-Za-z_][A-Za-z0-9_]*)(LIMIT|WHERE|HAVING)\b", r"\1 \2", sql)
    sql = re.sub(r"(\b[A-Za-z_][A-Za-z0-9_]*)(ORDER\s+BY)\b", r"\1 \2", sql)
    sql = re.sub(r"(\b[A-Za-z_][A-Za-z0-9_]*)(GROUP\s+BY)\b", r"\1 \2", sql)
    # Collapse multiple spaces to single space
    sql = re.sub(r"  +", " ", sql)
    return sql
# -----------------------------------------------------------
# ------------------ POSTPROCESS HEURISTIC ------------------
# ----------------------- Repair Alias ----------------------
# -----------------------------------------------------------
def _extract_internal_hint(ctx: str) -> Dict[str, Dict[str, str]]:
    m = re.search(r"### _INTERNAL_HINT\s+({.*})\s*$", ctx, flags=re.S)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(1))
        return obj.get("alias_map") or {}
    except Exception:
        return {}

def _repair_with_aliases(sql: str, ctx: str) -> str:
    """
    Repair column names using alias_map plus schema-metadata fallbacks.
    Do not touch text inside quotes ('...', "...", `...`) so explicit
    labels and aliases produced by the LLM stay intact.
    """
    alias_map_per_table = _extract_internal_hint(ctx)

    # global map (lower → canonical)
    global_map: Dict[str, Optional[str]] = {}
    for _t, amap in (alias_map_per_table or {}).items():
        for al, canon in (amap or {}).items():
            al = str(al).lower()
            if al not in global_map:
                global_map[al] = canon
            elif global_map[al] != canon:
                global_map[al] = None

    # Build dynamic fallback from schema metadata
    dynamic_fallback: Dict[str, str] = {}
    allowed_cols_case: Dict[str, str] = {}
    allowed_identifiers = _parse_allowed_identifiers(ctx)
    for _t, _cols in allowed_identifiers.items():
        for _c in _cols:
            lc = str(_c).lower()
            if lc not in allowed_cols_case:
                allowed_cols_case[lc] = _c

    def _canon_col(col: str) -> str:
        return allowed_cols_case.get(str(col).lower(), col)

    def _put_fallback(alias: str, canon: str):
        a = str(alias or "").strip().lower()
        if not a:
            return
        c = _canon_col(canon)
        prev = dynamic_fallback.get(a)
        if prev is None and a not in dynamic_fallback:
            dynamic_fallback[a] = c
        elif prev != c:
            # ambiguous alias -> disable automatic replacement for this alias
            dynamic_fallback[a] = None

    # Always map canonical columns to themselves in normalized form.
    # This provides case-style normalization like firstname -> firstName.
    for canon_col in allowed_cols_case.values():
        cc = str(canon_col)
        _put_fallback(cc.lower(), cc)
        _put_fallback(cc.lower().replace("_", ""), cc)

    try:
        analyzer = get_schema_analyzer()
    except Exception:
        analyzer = None

    try:
        schema_tables = list(get_all_tables())
    except Exception:
        schema_tables = list(allowed_identifiers.keys())

    allowed_tables_case = {
        str(table).strip("`").lower(): str(table).strip("`")
        for table in allowed_identifiers.keys()
        if str(table).strip("`")
    }

    def _fallback_scope_tables(sql_text: str) -> List[str]:
        """
        Scope schema-driven alias fallback to tables already active in the
        current SQL when possible, then fall back to request-scoped ALLOWED
        tables. This keeps generic aliases like bare `name` table-aware.
        """
        scoped: List[str] = []
        seen: Set[str] = set()

        def _push(table_name: str) -> None:
            raw = str(table_name or "").strip().strip("`")
            if not raw:
                return
            canon = allowed_tables_case.get(raw.lower(), raw)
            low = canon.lower()
            if low in seen:
                return
            seen.add(low)
            scoped.append(canon)

        code_parts = re.split(r"('(?:''|[^'])*'|\"(?:\"\"|[^\"])*\")", sql_text or "")
        code_only = "".join(part if idx % 2 == 0 else " " for idx, part in enumerate(code_parts))
        table_pat = re.compile(
            r"\b(?:FROM|JOIN)\s+((?:`?[A-Za-z_][A-Za-z0-9_]*`?\s*\.\s*)?`?[A-Za-z_][A-Za-z0-9_]*`?)",
            flags=re.I,
        )
        for match in table_pat.finditer(code_only):
            raw_ref = str(match.group(1) or "").strip()
            if not raw_ref:
                continue
            parts = [p.strip().strip("`") for p in raw_ref.split(".") if p.strip()]
            if not parts:
                continue
            _push(parts[-1])

        if scoped:
            return scoped

        for table_name in allowed_identifiers.keys():
            _push(table_name)
        return scoped or list(schema_tables)

    fallback_tables = _fallback_scope_tables(sql)

    def _semantic_alias_keys(text: str) -> Set[str]:
        raw = str(text or "").strip().lower()
        if not raw:
            return set()
        return {
            raw,
            raw.replace(" ", ""),
            raw.replace(" ", "_"),
            raw.replace("-", ""),
            raw.replace("-", "_"),
            raw.replace("_", ""),
        }

    def _add_semantic_column_alias(
        alias: str,
        *,
        include_terms: Set[str],
        exclude_terms: Optional[Set[str]] = None,
    ) -> None:
        if alias in dynamic_fallback:
            return
        alias_keys = _semantic_alias_keys(alias)
        include_keys = set()
        for term in include_terms:
            include_keys.update(_semantic_alias_keys(term))
        exclude_keys = set()
        for term in (exclude_terms or set()):
            exclude_keys.update(_semantic_alias_keys(term))

        def _expand_keys_from_schema(seed_keys: Set[str]) -> Set[str]:
            if not seed_keys:
                return set()
            expanded: Set[str] = set()
            for table in fallback_tables:
                try:
                    table_columns = list(analyzer.get_table_columns(table)) if analyzer is not None else list(allowed_identifiers.get(table, ()))
                except Exception:
                    table_columns = list(allowed_identifiers.get(table, ()))
                for col in table_columns:
                    semantic_keys = _semantic_alias_keys(col)
                    try:
                        for syn in get_column_synonyms(table, col) or []:
                            semantic_keys.update(_semantic_alias_keys(syn))
                    except Exception:
                        pass
                    if semantic_keys.intersection(seed_keys):
                        expanded.update(semantic_keys)
            return expanded

        include_keys.update(_expand_keys_from_schema(include_keys | alias_keys))
        exclude_keys.update(_expand_keys_from_schema(exclude_keys))

        candidates: List[Tuple[int, int, int, str]] = []
        for table in fallback_tables:
            try:
                table_columns = list(analyzer.get_table_columns(table)) if analyzer is not None else list(allowed_identifiers.get(table, ()))
            except Exception:
                table_columns = list(allowed_identifiers.get(table, ()))
            for col_idx, col in enumerate(table_columns):
                semantic_keys = _semantic_alias_keys(col)
                try:
                    for syn in get_column_synonyms(table, col) or []:
                        semantic_keys.update(_semantic_alias_keys(syn))
                except Exception:
                    pass
                overlap_keys = semantic_keys.intersection(include_keys)
                if not overlap_keys:
                    continue
                if exclude_keys and semantic_keys.intersection(exclude_keys):
                    continue
                score = len(overlap_keys) * 10
                score += len(semantic_keys.intersection(alias_keys)) * 20
                role = get_column_role(table, col)
                if role in {"metric", "measure", "amount", "quantity"}:
                    score += 5
                if str(col or "").strip().lower() == str(alias or "").strip().lower():
                    score += 20
                candidates.append((score, len(candidates), col_idx, col))

        if not candidates:
            return

        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        best_score, _, _, best_col = candidates[0]
        best_canon = _canon_col(best_col)
        tied_best = {
            _canon_col(col)
            for score, _, _, col in candidates
            if score == best_score
        }
        if len(tied_best) > 1:
            return
        _put_fallback(alias, best_canon)

    for table in fallback_tables:
        # Add name column mapping
        name_col = get_name_column(table)
        if name_col:
            # "name" -> actual name column (only if not already mapped)
            if "name" not in dynamic_fallback:
                _put_fallback("name", name_col)
            # Self-reference for case normalization (productname -> productName)
            _put_fallback(name_col.lower(), name_col)

        # Add ID column mapping
        id_col = get_id_column(table)
        if id_col:
            # Common ID aliases
            id_aliases = ["id", f"{table[:-1]}id", f"{table[:-1]}_id", f"id_{table[:-1]}", id_col.lower()]
            for alias in id_aliases:
                _put_fallback(alias, id_col)

        # Add metadata synonym mappings (schema-driven, no table hardcode).
        if analyzer is not None:
            try:
                for col in analyzer.get_table_columns(table):
                    canon = _canon_col(col)
                    for syn in get_column_synonyms(table, col):
                        syn_low = str(syn).strip().lower()
                        if not syn_low:
                            continue
                        candidates = {
                            syn_low,
                            syn_low.replace(" ", ""),
                            syn_low.replace(" ", "_"),
                            syn_low.replace("-", ""),
                            syn_low.replace("-", "_"),
                        }
                        for cand in candidates:
                            if re.match(r"^[a-z_][a-z0-9_]*$", cand):
                                _put_fallback(cand, canon)
            except Exception:
                pass

    # Keep semantic alias seeds centralized so they can be externalized or
    # replaced with richer schema metadata later without changing call sites.
    semantic_alias_specs = [
        (
            "listprice",
            set(HELPERS_SEMANTIC_ALIAS_LISTPRICE_INCLUDE_TERMS),
            set(HELPERS_SEMANTIC_ALIAS_LISTPRICE_EXCLUDE_TERMS),
        ),
        (
            "price",
            set(HELPERS_SEMANTIC_ALIAS_PRICE_INCLUDE_TERMS),
            set(HELPERS_SEMANTIC_ALIAS_PRICE_EXCLUDE_TERMS),
        ),
    ]
    for alias, include_terms, exclude_terms in semantic_alias_specs:
        _add_semantic_column_alias(
            alias,
            include_terms=include_terms,
            exclude_terms=exclude_terms,
        )

    # Merge: dynamic_fallback only if not already in global_map
    for al, canon in dynamic_fallback.items():
        if canon is None:
            continue
        al = al.lower()
        if al not in global_map:
            global_map[al] = canon
        elif global_map[al] is None:
            pass

    keywords = _SQL_KEYWORDS
    token_pat = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")
    # Preserve explicit aliases (e.g., AS customer_id) and their references.
    # Without this, alias repair can rewrite output alias tokens into schema columns.
    protected_aliases = {
        (m.group(1) or "").lower()
        for m in re.finditer(r"\bAS\s+([A-Za-z_][A-Za-z0-9_]*)\b", sql, flags=re.I)
        if m.group(1)
    }

    def replace_in_code_segment(seg: str) -> str:
        def smart_repl(m):
            w = m.group(0)
            lw = w.lower()
            if lw in keywords:
                return w
            if lw in protected_aliases:
                return w
            canon = global_map.get(lw)
            if not canon:
                return w
            if not _ALNUM.match(canon):
                return w
            return canon
        return token_pat.sub(smart_repl, seg)

    parts = re.split(r"('(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`[^`]*`)", sql)
    out_parts = []
    for idx, part in enumerate(parts):
        if idx % 2 == 1:
            out_parts.append(part)
        else:
            out_parts.append(replace_in_code_segment(part))
    return "".join(out_parts)


def _repair_backticked_identifier_phrases(sql: str, ctx: str) -> str:
    """
    Repair hallucinated backticked identifier phrases (e.g. `check number`)
    into canonical schema columns (e.g. `checkNumber`) for single-table/simple
    drafts. Keep explicit output aliases (AS `...`) untouched.
    """
    if not sql or "`" not in sql:
        return sql

    allowed = _parse_allowed_identifiers(ctx)
    if not allowed:
        return sql

    allowed_tables_case: Dict[str, str] = {}
    allowed_cols_case: Dict[str, str] = {}
    for table, cols in allowed.items():
        t_case = str(table).strip("`")
        if t_case:
            allowed_tables_case[t_case.lower()] = t_case
        for col in cols:
            c_case = str(col).strip("`")
            if c_case:
                allowed_cols_case.setdefault(c_case.lower(), c_case)

    if not allowed_cols_case:
        return sql

    stop_tokens = {
        "the", "a", "an", "of", "for", "to", "with", "any", "all", "each",
        "every", "single", "exactly", "display", "show", "list",
    }

    def _phrase_tokens(text: str, drop_stopwords: bool = False) -> List[str]:
        split_camel = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text or "")
        tokens = re.findall(r"[a-z0-9]+", split_camel.lower())
        if drop_stopwords:
            tokens = [t for t in tokens if t not in stop_tokens]
        return tokens

    def _phrase_key(text: str, drop_stopwords: bool = False) -> str:
        return "".join(_phrase_tokens(text, drop_stopwords=drop_stopwords))

    key_to_col: Dict[str, Optional[str]] = {}

    def _put_key(key: str, canon_col: str) -> None:
        if not key:
            return
        prev = key_to_col.get(key)
        if prev is None and key not in key_to_col:
            key_to_col[key] = canon_col
        elif prev != canon_col:
            key_to_col[key] = None

    # Canonical column keys.
    for canon_col in allowed_cols_case.values():
        _put_key(_phrase_key(canon_col, drop_stopwords=False), canon_col)
        _put_key(_phrase_key(canon_col, drop_stopwords=True), canon_col)

    # Synonym-based keys from metadata.
    for table, cols in allowed.items():
        for col in cols:
            canon_col = allowed_cols_case.get(str(col).lower(), str(col))
            for syn in get_column_synonyms(table, col) or []:
                syn_txt = str(syn or "").strip()
                if not syn_txt:
                    continue
                _put_key(_phrase_key(syn_txt, drop_stopwords=False), canon_col)
                _put_key(_phrase_key(syn_txt, drop_stopwords=True), canon_col)

    quoted_ident = re.compile(r"`([^`]+)`")

    def _best_overlap_match(raw_text: str) -> Optional[str]:
        raw_tokens = _phrase_tokens(raw_text, drop_stopwords=True)
        if not raw_tokens:
            return None
        raw_set = set(raw_tokens)

        best_col: Optional[str] = None
        best_score = 0.0
        for canon_col in allowed_cols_case.values():
            col_tokens = _phrase_tokens(canon_col, drop_stopwords=True)
            if not col_tokens:
                continue
            col_set = set(col_tokens)
            overlap = raw_set.intersection(col_set)
            if not overlap:
                continue
            score = len(overlap) / max(1, len(raw_set))
            if "amount" in raw_set and "amount" in col_set:
                score += 0.40
            if "number" in raw_set and "number" in col_set:
                score += 0.20
            if overlap == raw_set:
                score += 0.15
            if score > best_score:
                best_score = score
                best_col = canon_col

        if best_score >= 0.65:
            return best_col
        return None

    def _replace_identifier(m: re.Match) -> str:
        raw = (m.group(1) or "").strip()
        if not raw:
            return m.group(0)

        # Preserve explicit output aliases.
        prefix = sql[max(0, m.start() - 16):m.start()]
        if re.search(r"\bAS\s+$", prefix, flags=re.I):
            return m.group(0)

        low = raw.lower()
        if low in allowed_tables_case:
            return f"`{allowed_tables_case[low]}`"
        if low in allowed_cols_case:
            return f"`{allowed_cols_case[low]}`"

        candidate = None
        key_full = _phrase_key(raw, drop_stopwords=False)
        key_drop = _phrase_key(raw, drop_stopwords=True)
        if key_full:
            candidate = key_to_col.get(key_full)
        if not candidate and key_drop:
            candidate = key_to_col.get(key_drop)
        if not candidate:
            candidate = _best_overlap_match(raw)

        if candidate:
            return f"`{candidate}`"
        return m.group(0)

    return quoted_ident.sub(_replace_identifier, sql)

# ======================================================================
# ==================== DYNAMIC JOIN CONTEXT FROM RETRIEVAL ==============
# ======================================================================

def _extract_fk_from_context(ctx: str) -> Dict[str, Dict[str, str]]:
    """
    Dynamically extract FK relationships from retrieval context.

    Parses context for "FK to X.Y" patterns (from JSONL metadata).

    Returns:
        {
            "source_table": {"fk_column": "target_table"}
        }
    """
    fk_map = defaultdict(dict)

    # Pattern: "FK to targetTable.targetColumn"
    fk_pattern = re.compile(r'FK to (\w+)\.(\w+)', re.IGNORECASE)

    # Also extract source table and column from context lines
    # Format: "table: X | column: Y | ... FK to Z.W"
    for line in ctx.split('\n'):
        match = fk_pattern.search(line)
        if match:
            target_table = match.group(1).lower()
            target_col = match.group(2).lower()

            # Try to find source table and column in the same line
            table_match = re.search(r'table:\s*(\w+)', line, re.IGNORECASE)
            col_match = re.search(r'column:\s*(\w+)', line, re.IGNORECASE)

            if table_match and col_match:
                source_table = table_match.group(1).lower()
                source_col = col_match.group(1).lower()
                fk_map[source_table][source_col] = target_table

    return dict(fk_map)


def _infer_tables_from_question_columns(
    question: str,
    ranked_cols: list,
    router_tables: list
) -> Dict[str, List[str]]:
    """
    Infer which tables are needed based on question mentions of columns.

    When question doesn't explicitly mention table names (e.g., "total amount", "quantity ordered"),
    use retrieval results to find which tables have matching columns.

    Returns:
        {
            "table_name": ["matched_column1", "matched_column2", ...]
        }
    """
    q_lower = question.lower()

    def _question_mentions_column(q_text: str, column: str) -> bool:
        """
        Robust column mention check:
        - direct column name
        - camelCase/snake_case spaced form
        - informative token-level match (e.g., quantityOrdered -> "ordered")
        """
        if not column:
            return False

        col_lower = column.lower()
        spaced = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', column).replace("_", " ").lower()

        # 1) Direct full-name match
        if col_lower in q_text:
            return True

        # 2) Spaced variant for camelCase/snake_case
        if spaced != col_lower and spaced in q_text:
            return True

        # 3) Informative token boundary match (avoid weak generic tokens)
        stop_tokens = {
            "id", "ids", "code", "codes", "number", "numbers",
            "name", "names", "date", "dates", "line", "lines",
            "type", "types", "status", "each", "table", "order",
            "quantity", "price", "amount", "total", "average", "value"
        }
        for token in re.findall(r"[a-z0-9]+", spaced):
            if len(token) < 4 or token in stop_tokens:
                continue
            # "ordered by ..." usually indicates sorting intent, not orderdetails semantics
            if token == "ordered" and re.search(r"\bordered\s+by\b", q_text):
                continue
            if re.search(rf"\b{re.escape(token)}\b", q_text):
                return True

        return False

    table_matches = defaultdict(list)

    for tbl, col in ranked_cols:
        if _question_mentions_column(q_lower, col):
            table_matches[tbl.lower()].append(col)

    # Always include tables explicitly mentioned in router_tables
    for tbl in router_tables:
        tbl_lower = tbl.lower()
        if tbl_lower not in table_matches:
            table_matches[tbl_lower] = []

    return dict(table_matches)


def _build_fk_join_chain(
    question: str,
    ranked_cols: list,
    router_tables: list,
    fk_map: Dict[str, Dict[str, str]],
    primary_table: str = None
) -> List[Dict[str, str]]:
    """
    Build FK join chain for multi-table queries.

    For questions that mention columns from multiple tables but don't explicitly
    specify the join path, use FK relationships to construct the join chain.

    Uses primary_table to determine the correct FROM/JOIN order.

    Returns:
        [
            {"from": "table1", "join": "table2", "on": "table1.fk_col = table2.pk_col"},
            ...
        ]
    """
    if len(router_tables) < 2:
        return []

    # Infer which tables are needed based on question
    table_matches = _infer_tables_from_question_columns(question, ranked_cols, router_tables)

    if len(table_matches) < 2:
        return []

    # Build FK join chain
    join_chain = []

    # If primary_table is specified, start with it and find joins to other tables
    if primary_table:
        primary_lower = primary_table.lower()
        router_tables_lower = [t.lower() for t in router_tables]

        # Get other tables (excluding primary)
        other_tables = [t for t in router_tables_lower if t != primary_lower]

        # For each other table, find the FK relationship
        for other_tbl in other_tables:
            # Check if primary_table has FK to other_tbl
            if primary_lower in fk_map:
                for fk_col, target in fk_map[primary_lower].items():
                    if target.lower() == other_tbl:
                        join_chain.append({
                            "from": primary_lower,
                            "join": other_tbl,
                            "on": f"{primary_lower}.{fk_col} = {other_tbl}.{_infer_target_column(fk_col, other_tbl, ranked_cols)}"
                        })
                        break

            # Check if other_tbl has FK to primary_table (reverse FK)
            if other_tbl in fk_map:
                for fk_col, target in fk_map[other_tbl].items():
                    if target.lower() == primary_lower:
                        join_chain.append({
                            "from": primary_lower,
                            "join": other_tbl,
                            "on": f"{other_tbl}.{fk_col} = {primary_lower}.{_infer_target_column(fk_col, primary_lower, ranked_cols)}"
                        })
                        break
    else:
        # Fallback: For each pair of tables, check if there's a FK relationship
        router_tables_lower = [t.lower() for t in router_tables]

        for i, source_tbl in enumerate(router_tables_lower):
            if source_tbl not in fk_map:
                continue

            for fk_col, target_tbl in fk_map[source_tbl].items():
                if target_tbl in router_tables_lower:
                    join_chain.append({
                        "from": source_tbl,
                        "join": target_tbl,
                        "on": f"{source_tbl}.{fk_col} = {target_tbl}.{_infer_target_column(fk_col, target_tbl, ranked_cols)}"
                    })

    return join_chain


def _infer_target_column(
    fk_col: str,
    target_table: str,
    ranked_cols: list
) -> str:
    """
    Infer the target column name for a FK relationship.

    Example: If fk_col is an identifier-like reference for the target entity,
    the target column is often the same identifier on the referenced table.
    """
    # Check if we have this column in ranked_cols
    for tbl, col in ranked_cols:
        if tbl.lower() == target_table.lower():
            col_lower = col.lower()
            # Common pattern: fk_col and target column share the same identifier name.
            if fk_col.lower() == col_lower:
                return col

    # Fallback: use fk_col as target_col (common pattern)
    return fk_col


def _build_dynamic_join_context(
    state: "RAGState",
    router_decision: dict = None
) -> str:
    """
    Build dynamic join context from retrieval results and FK relationships.

    Uses:
    1. ranked_cols from state (what retrieval found)
    2. FK relationships from context
    3. router_tables from state
    4. Question column mentions (implicit table detection)

    Returns:
        Context block with FK relationships and detected tables
    """
    ranked_cols = state.get("ranked_cols", [])
    router_tables = state.get("router_tables", [])
    question = state.get("question", "")
    ctx = state.get("context", "")

    if not ranked_cols or not router_tables:
        return ""

    fk_map = _extract_fk_from_context(ctx)

    lines = []
    lines.append("\n### JOIN RELATIONSHIPS")

    # Build FK join chain (pass primary_table to determine correct FROM/JOIN order)
    primary_table = state.get("primary_table", "")
    join_chain = _build_fk_join_chain(question, ranked_cols, router_tables, fk_map, primary_table)

    if join_chain:
        lines.append("# Detected FK join chain:")
        for join in join_chain:
            lines.append(f"#   FROM {join['from']} JOIN {join['join']} ON {join['on']}")
    else:
        lines.append("# FK relationships detected from schema:")
        # Build FK relationships between detected tables
        router_tables_lower = {t.lower(): t for t in router_tables}

        for source, fks in fk_map.items():
            if source in router_tables_lower:
                for fk_col, target in fks.items():
                    if target in router_tables_lower:
                        lines.append(f"#   {source}.{fk_col} REFERENCES {target}")

    # Show which columns matched from question
    table_matches = _infer_tables_from_question_columns(question, ranked_cols, router_tables)

    if table_matches:
        lines.append("\n# Columns matched from question:")
        for tbl, cols in table_matches.items():
            if cols:
                cols_str = ", ".join(cols)
                lines.append(f"#   {tbl}: {cols_str}")

    # Show what columns are available for each detected table
    lines.append("\n# Available columns per table:")
    table_cols = defaultdict(list)
    for tbl, col in ranked_cols:
        table_cols[tbl.lower()].append(col)

    for tbl in router_tables:
        tbl_lower = tbl.lower()
        if tbl_lower in table_cols:
            cols_str = ", ".join(table_cols[tbl_lower][:8])  # Show first 8 columns
            if len(table_cols[tbl_lower]) > 8:
                cols_str += f" ... ({len(table_cols[tbl_lower])} total)"
            lines.append(f"#   {tbl}: {cols_str}")

    return "\n".join(lines)


def _build_join_key_hints_for_prompt(
    state: "RAGState",
    *,
    max_edges: int = 8,
) -> List[str]:
    """
    Build FK-consistent JOIN key hints for first-pass prompt generation.

    This is intentionally conservative:
    - Scope to routed/allowed tables only.
    - Prefer edges inferred from retrieval-aware join chain.
    - Fall back to schema FK graph when needed.
    """
    if not state or max_edges <= 0:
        return []

    question = state.get("question", "") or ""
    ctx = state.get("context", "") or ""
    ranked_cols = state.get("ranked_cols", []) or []
    router_tables = [str(t).strip().lower() for t in (state.get("router_tables", []) or []) if str(t).strip()]
    primary_table = str(state.get("primary_table", "") or "").strip().lower()

    if not router_tables:
        return []

    allowed = _parse_allowed_identifiers(ctx)
    allowed_tables = {str(t).strip().lower() for t in (allowed or {}).keys() if str(t).strip()}
    table_scope = set(router_tables)
    if allowed_tables:
        table_scope = table_scope.intersection(allowed_tables) or table_scope
    if not table_scope:
        return []

    fk_map_ctx = _extract_fk_from_context(ctx) or {}
    join_chain = _build_fk_join_chain(
        question=question,
        ranked_cols=ranked_cols,
        router_tables=[t for t in router_tables if t in table_scope],
        fk_map=fk_map_ctx,
        primary_table=primary_table,
    )

    edges: List[str] = []
    seen: Set[str] = set()

    for join in join_chain:
        on_expr = str(join.get("on", "")).strip()
        if not on_expr:
            continue
        if on_expr not in seen:
            seen.add(on_expr)
            edges.append(on_expr)
        if len(edges) >= max_edges:
            return edges

    try:
        schema_fk = get_fk_relationships() or {}
    except Exception:
        schema_fk = {}

    ordered_scope = []
    if primary_table and primary_table in table_scope:
        ordered_scope.append(primary_table)
    ordered_scope.extend([t for t in router_tables if t in table_scope and t not in ordered_scope])

    def _append_edge(src: str, fk_col: str, tgt: str) -> None:
        tgt_id = (get_id_column(tgt) or "").strip("`")
        edge = f"{src}.{fk_col} = {tgt}.{tgt_id}" if tgt_id else f"{src}.{fk_col} = {tgt}.{fk_col}"
        if edge not in seen:
            seen.add(edge)
            edges.append(edge)

    for src in ordered_scope:
        for fk_col, tgt in (schema_fk.get(src, {}) or {}).items():
            src_l = str(src).strip().lower()
            fk_l = str(fk_col).strip()
            tgt_l = str(tgt).strip().lower()
            if not src_l or not fk_l or not tgt_l:
                continue
            if tgt_l not in table_scope:
                continue
            _append_edge(src_l, fk_l, tgt_l)
            if len(edges) >= max_edges:
                return edges

    for src, rels in (fk_map_ctx or {}).items():
        src_l = str(src).strip().lower()
        if not src_l or src_l not in table_scope:
            continue
        for fk_col, tgt in (rels or {}).items():
            fk_l = str(fk_col).strip()
            tgt_l = str(tgt).strip().lower()
            if not fk_l or not tgt_l or tgt_l not in table_scope:
                continue
            _append_edge(src_l, fk_l, tgt_l)
            if len(edges) >= max_edges:
                return edges

    return edges


def _order_tables_by_fk_distance(primary_table: str, tables: List[str]) -> List[str]:
    """
    Order tables by shortest FK-graph distance from primary_table.
    Falls back to original order when FK metadata is unavailable.
    """
    if not primary_table or not tables:
        return tables

    primary = (primary_table or "").strip().lower()
    table_list = [t for t in tables if t]
    if not table_list:
        return table_list

    try:
        schema_fk = get_fk_relationships() or {}
    except Exception:
        return table_list

    # Build undirected adjacency from FK metadata: source -> target and target -> source.
    graph: Dict[str, Set[str]] = defaultdict(set)
    for source_table, fk_map in schema_fk.items():
        src = (source_table or "").strip().lower()
        if not src:
            continue
        for _fk_col, target_table in (fk_map or {}).items():
            tgt = (target_table or "").strip().lower()
            if not tgt:
                continue
            graph[src].add(tgt)
            graph[tgt].add(src)

    if primary not in graph:
        return table_list

    # BFS shortest distance from primary.
    distances: Dict[str, int] = {primary: 0}
    queue: List[str] = [primary]
    q_idx = 0
    while q_idx < len(queue):
        cur = queue[q_idx]
        q_idx += 1
        cur_dist = distances[cur]
        for nxt in graph.get(cur, set()):
            if nxt in distances:
                continue
            distances[nxt] = cur_dist + 1
            queue.append(nxt)

    original_idx = {t.lower(): i for i, t in enumerate(table_list)}
    sorted_tables = sorted(
        table_list,
        key=lambda t: (
            distances.get((t or "").strip().lower(), 10**9),
            original_idx.get((t or "").strip().lower(), 10**9),
        ),
    )
    return sorted_tables


def _router_has_escalation_reason(router_decision: Dict[str, Any], reason: str) -> bool:
    reasons = router_decision.get("escalation_reasons", []) or []
    reason_l = str(reason or "").strip().lower()
    return any(str(r or "").strip().lower() == reason_l for r in reasons)


def _should_force_required_inner_join_chain(
    question: str,
    router_decision: Dict[str, Any],
    router_tables: List[str],
    ctx: str = "",
) -> bool:
    table_count = len([t for t in (router_tables or []) if str(t).strip()])
    if table_count <= 1 or _question_wants_include_zero(question, ctx):
        return False
    if _router_has_escalation_reason(router_decision, "multi_table_distinct_threshold"):
        return True
    if _question_wants_latest_year_scope_with_aggregate(question):
        return True
    return False


def _shortest_fk_table_path(primary_table: str, target_table: str) -> List[str]:
    primary = str(primary_table or "").strip().lower()
    target = str(target_table or "").strip().lower()
    if not primary or not target:
        return []
    if primary == target:
        return [primary]

    try:
        schema_fk = get_fk_relationships() or {}
    except Exception:
        schema_fk = {}

    graph: Dict[str, Set[str]] = defaultdict(set)
    for src, rels in (schema_fk or {}).items():
        src_l = str(src or "").strip().lower()
        if not src_l:
            continue
        for _, tgt in (rels or {}).items():
            tgt_l = str(tgt or "").strip().lower()
            if not tgt_l:
                continue
            graph[src_l].add(tgt_l)
            graph[tgt_l].add(src_l)

    if primary not in graph or target not in graph:
        return []

    queue: List[List[str]] = [[primary]]
    seen: Set[str] = {primary}
    q_idx = 0
    while q_idx < len(queue):
        path = queue[q_idx]
        q_idx += 1
        node = path[-1]
        for nxt in sorted(graph.get(node, set())):
            if nxt in seen:
                continue
            next_path = path + [nxt]
            if nxt == target:
                return next_path
            seen.add(nxt)
            queue.append(next_path)
    return []


def _build_connector_path_rules(primary_table: str, ordered_tables: List[str]) -> List[str]:
    primary = str(primary_table or "").strip().lower()
    if not primary:
        return []

    rules: List[str] = []
    seen_rules: Set[str] = set()
    for target in ordered_tables:
        target_l = str(target or "").strip().lower()
        if not target_l or target_l == primary:
            continue
        path = _shortest_fk_table_path(primary, target_l)
        if len(path) <= 2:
            continue
        connector = path[-2]
        rule = (
            "CONNECTOR PATH RULE (CRITICAL): "
            f"Reach {target_l} via {connector}. "
            f"Do NOT skip {connector} and do NOT join {target_l} directly to {primary}."
        )
        if rule not in seen_rules:
            seen_rules.add(rule)
            rules.append(rule)
    return rules


def _find_tables_with_implicit_columns(
    question: str,
    ranked_cols: list,
    relevant_tables: list,
    ctx: str
) -> Set[str]:
    """
    Find tables whose columns are implicitly mentioned in the question.

    AGNOSTIC approach - works for ANY column type (numeric, string, date, etc.).
    Uses retrieval results (ranked_cols) to find columns mentioned in the question,
    then filters to only include tables that have FK relationships with relevant tables.

    Example:
        Question: "show customer names and order dates"
        -> Finds: customers (customerName), orders (orderDate)
        -> Only adds tables that have FK relationships with already-selected tables

    Args:
        question: User's question
        ranked_cols: List of (table, column) tuples from retrieval
        relevant_tables: Tables already selected by router
        ctx: Retrieval context (for FK extraction)

    Returns:
        Set of table names to add (lowercase)
    """
    if not ranked_cols or not relevant_tables:
        return set()

    q_lower = question.lower()
    relevant_lower = {t.lower() for t in relevant_tables}
    fk_map = _extract_fk_from_context(ctx)

    def _is_key_or_fk_like_column(tbl_lower: str, col_name: str) -> bool:
        """
        Conservative detector for bridge/key-like columns in implicit-table discovery.
        We avoid adding new tables solely from these columns unless the table itself
        is explicitly mentioned in the question.
        """
        c = (col_name or "").lower().strip()
        if not c:
            return False

        # Explicit FK metadata from context.
        if c in {str(k).lower() for k in fk_map.get(tbl_lower, {}).keys()}:
            return True

        # Table primary identifier is also key-like.
        id_col = (get_id_column(tbl_lower) or "").lower().strip()
        if id_col and c == id_col:
            return True

        # Generic identifier suffixes (schema-agnostic).
        if re.search(r"(id|code|number)$", c, flags=re.I):
            return True

        return False

    def _column_match_type(q_text: str, column: str) -> str:
        """
        Return match confidence for implicit table discovery:
        - "exact": direct full/spaced column phrase found
        - "token": token-level semantic fallback match
        - "none" : no match
        """
        if not column:
            return "none"

        col_lower = column.lower()
        spaced = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', column).replace("_", " ").lower()

        # Direct / phrase match
        if col_lower in q_text:
            return "exact"
        if spaced != col_lower and spaced in q_text:
            return "exact"

        # Guardrail: avoid false positives like customerName matching only "customer id".
        semantic_suffix_keywords = {
            "name": {"name", "names", "nama"},
            "date": {"date", "dates", "tanggal", "tgl", "time", "waktu"},
            "number": {"number", "numbers", "id", "ids", "code", "codes", "nomor", "kode", "no"},
            "code": {"code", "codes", "id", "ids", "number", "numbers", "nomor", "kode", "no"},
            "id": {"id", "ids", "number", "numbers", "code", "codes", "nomor", "kode", "no"},
        }
        col_tokens = re.findall(r"[a-z0-9]+", spaced)
        if len(col_tokens) >= 2:
            suffix = col_tokens[-1]
            suffix_terms = semantic_suffix_keywords.get(suffix)
            if suffix_terms:
                has_suffix_intent = any(
                    re.search(rf"\b{re.escape(term)}\b", q_text)
                    for term in suffix_terms
                )
                if not has_suffix_intent:
                    return "none"

        # Token boundary match for informative tokens only
        stop_tokens = {
            "id", "ids", "code", "codes", "number", "numbers",
            "name", "names", "date", "dates", "line", "lines",
            "type", "types", "status", "each", "table", "order",
            "quantity", "price", "amount", "total", "average", "value"
        }
        for token in re.findall(r"[a-z0-9]+", spaced):
            if len(token) < 4 or token in stop_tokens:
                continue
            # "ordered by ..." usually indicates sorting intent, not orderdetails semantics
            if token == "ordered" and re.search(r"\bordered\s+by\b", q_text):
                continue
            if re.search(rf"\b{re.escape(token)}\b", q_text):
                return "token"

        return "none"

    # Find tables whose columns are mentioned in question
    mentioned_tables = defaultdict(set)  # {table: {matched_columns}}

    for tbl, col in ranked_cols:
        tbl_lower = tbl.lower()

        # Skip if already in relevant tables
        if tbl_lower in relevant_lower:
            continue

        match_type = _column_match_type(q_lower, col)
        if match_type != "none":
            # Guardrail: do not expand tables from implicit FK/key-like columns only
            # unless table entity is explicitly mentioned in the question.
            if _is_key_or_fk_like_column(tbl_lower, col):
                table_explicitly_mentioned = _table_mentioned(q_lower, tbl_lower, include_synonyms=True)
                # For key-like columns, token-level fuzzy matches are too noisy
                # unless the table itself is explicitly mentioned.
                if (match_type != "exact") and (not table_explicitly_mentioned):
                    continue
            mentioned_tables[tbl_lower].add(col)

    if not mentioned_tables:
        return set()

    # [FIX] Filter out tables whose columns are already available in existing relevant tables
    # Example: If payments.customerNumber exists, don't add customers just for customerNumber
    # This prevents single-table queries from being detected as multi-table
    available_cols_in_existing = set()
    allowed_identifiers = _parse_allowed_identifiers(ctx)

    # Prefer full schema columns from ALLOWED IDENTIFIERS for selected tables
    # (ranked_cols can be sparse and miss shared columns like productLine).
    for tbl_lower in relevant_lower:
        matched_table = None
        for original_tbl in allowed_identifiers.keys():
            if original_tbl.lower() == tbl_lower:
                matched_table = original_tbl
                break

        if matched_table:
            for col in allowed_identifiers.get(matched_table, set()):
                available_cols_in_existing.add(str(col).lower())
        else:
            # Fallback when ALLOWED block is incomplete
            for rel_tbl, col in ranked_cols:
                if rel_tbl.lower() == tbl_lower:
                    available_cols_in_existing.add(col.lower())

    # Filter out tables whose columns are already covered
    filtered_mentioned_tables = {}
    for tbl, cols in mentioned_tables.items():
        cols_not_available = set()
        for col in cols:
            col_lower = col.lower()
            # Check if this column or its word components are already available
            already_covered = False

            # Direct match
            if col_lower in available_cols_in_existing:
                already_covered = True
            else:
                # Check word component overlap (e.g., "customerNumber" covers "customer")
                for avail_col in available_cols_in_existing:
                    if col_lower in avail_col or avail_col in col_lower:
                        already_covered = True
                        break
                    # Check word component overlap
                    col_words = set(col_lower.split('_'))
                    avail_words = set(avail_col.split('_'))
                    if col_words & avail_words:  # intersection non-empty
                        already_covered = True
                        break

            if not already_covered:
                cols_not_available.add(col)

        if cols_not_available:
            filtered_mentioned_tables[tbl] = cols_not_available

    mentioned_tables = filtered_mentioned_tables
    if not mentioned_tables:
        return set()

    # Only include tables that have FK relationship with relevant tables
    tables_to_add = set()

    for tbl in mentioned_tables:
        # Check if this table has FK to/from any relevant table
        has_fk_connection = False

        # Check forward FK: tbl -> relevant_table
        if tbl in fk_map:
            for target_tbl in fk_map[tbl].values():
                if target_tbl in relevant_lower:
                    has_fk_connection = True
                    break

        # Check reverse FK: relevant_table -> tbl
        if not has_fk_connection:
            for rel_tbl in relevant_lower:
                if rel_tbl in fk_map and tbl in fk_map[rel_tbl].values():
                    has_fk_connection = True
                    break

        if has_fk_connection:
            tables_to_add.add(tbl)

    return tables_to_add


# ======================================================================
# ================= ROUTING & GENERATION PROMPT BUILDER ================
# ======================================================================
def make_router_prompt(question: str, schema_summary: str) -> str:
    """
    Prompt Router yang dioptimalkan untuk Model SLM (1.5B/7B).
    Output JSON: complexity & qtype.
    """
    return f"""You are a SQL router. Analyze the question and schema to decide complexity.

Schema Tables: [{schema_summary}]

DEFINITIONS:
1. COMPLEXITY: "simple"
   - Single table operations (Filter, Sort, Limit).
   - Attribute Check (IS NULL/Empty).
   - FK Filtering (Filtering by ID present in the main table).
   - Global Rank (ORDER BY ... LIMIT).

2. COMPLEXITY: "complex"
   - Multi-table Joins.
   - Relationship Check (e.g. "customers with no orders").
   - Advanced Logic: "Top N per Group", HAVING, Subqueries.

3. QTYPE:
   - "lookup": Retrieve rows, Global Sorting.
   - "aggregate": Math (SUM, AVG, COUNT) with GROUP BY.
   - "rank": Group-wise ranking (e.g., "Top product PER category").

CRITICAL RULES:
1. "Missing" or "No" [X]:
   - If [X] is a COLUMN -> SIMPLE.
   - If [X] is a separate TABLE -> COMPLEX.
2. Filtering:
   - By ID/Code -> SIMPLE.
   - By Name/Desc -> COMPLEX.
3. Ranking:
   - "Highest/Lowest" (Global) -> SIMPLE ("lookup").
   - "Highest/Lowest PER [Group]" -> COMPLEX ("rank").

Question: {question}

Return JSON object with keys: complexity (simple/complex), qtype (lookup/aggregate/rank).
"""

# ======================================================================
# ================ INTENT IR SOFT HINT (IR-INFORMED LLM) ===============
# ======================================================================
def _build_intent_ir_hint(state: "RAGState") -> str:
    """
    Build a soft intent IR hint for the LLM when IR was built but not used
    as the planner path. This allows the LLM to benefit from structured
    intent analysis even when the deterministic planner cannot be used.

    Safety guards:
    - Skip if confidence < 0.50 (too unreliable)
    - Skip if IR payload is empty or planner-incompatible
    - Require at least one structural field (dimensions/metrics/filters)
      beyond just subject_entity + qtype
    """
    if not state:
        return ""

    intent_ir = state.get("intent_ir") or {}
    ir_used = state.get("ir_used", False)

    # Only add hint when IR was built but not used as planner path
    if ir_used or not intent_ir:
        return ""

    confidence = float(intent_ir.get("confidence", 0.0) or 0.0)

    # Guard: skip hint if confidence is too low to be useful
    if confidence < 0.50:
        return ""

    # Guard: skip if IR explicitly flagged as incompatible or empty
    base_gate_reasons = set(state.get("ir_base_gate_reasons") or [])
    if "empty_ir_payload" in base_gate_reasons or "planner_incompatible" in base_gate_reasons:
        return ""

    # Guard: skip if IR couldn't resolve a HAVING-like threshold —
    # the filter (and often the metric) will be wrong, misleading the LLM.
    gate_reasons = set(state.get("ir_gate_reasons") or [])
    if "having_like_threshold_unresolved" in gate_reasons:
        return ""

    subject = str(intent_ir.get("subject_entity") or "").strip()
    qtype = str(intent_ir.get("qtype") or "").strip()

    parts = []
    if subject:
        parts.append(f"Subject entity: {subject}")
    if qtype:
        parts.append(f"Query type: {qtype}")

    # Dimensions
    has_structural = False
    dims = intent_ir.get("dimensions") or []
    if dims:
        dim_strs = []
        for d in dims:
            if not isinstance(d, dict):
                continue
            t = str(d.get("table") or "").strip()
            c = str(d.get("column") or "").strip()
            if t and c:
                dim_strs.append(f"{t}.{c}")
        if dim_strs:
            parts.append(f"Dimensions: {', '.join(dim_strs)}")
            has_structural = True

    # Metrics
    metrics = intent_ir.get("metrics") or []
    if metrics:
        met_strs = []
        for m in metrics:
            if not isinstance(m, dict):
                continue
            agg = str(m.get("agg") or "").strip().upper()
            t = str(m.get("table") or "").strip()
            c = str(m.get("column") or "*").strip()
            # Skip bogus metrics like SUM(*), AVG(*) — only COUNT(*) is valid
            if c == "*" and agg != "COUNT":
                continue
            if agg:
                met_strs.append(f"{agg}({t}.{c})")
        if met_strs:
            parts.append(f"Metrics: {', '.join(met_strs)}")
            has_structural = True

    # Filters
    filters = intent_ir.get("filters") or []
    if filters:
        filt_strs = []
        for f in filters:
            if not isinstance(f, dict):
                continue
            t = str(f.get("table") or "").strip()
            c = str(f.get("column") or "").strip()
            op = str(f.get("op") or "").strip()
            val = f.get("value")
            if t and c and op:
                filt_strs.append(f"{t}.{c} {op} {val}")
        if filt_strs:
            parts.append(f"Filters: {', '.join(filt_strs)}")
            has_structural = True

    # Limit
    limit = intent_ir.get("limit")
    if isinstance(limit, int) and limit > 0:
        parts.append(f"Limit: {limit}")
        has_structural = True

    # Guard: require at least one structural field beyond subject+qtype
    if not has_structural:
        return ""

    # Guard: aggregate qtype with no metrics is likely a misclassification
    # (e.g. window/cumulative query mislabeled as aggregate) - dims-only
    # hint biases LLM to treat columns as grouping-only, not SELECT targets.
    has_valid_metrics = any(f"Metrics:" in p for p in parts)
    if qtype.lower() == "aggregate" and not has_valid_metrics:
        return ""

    hint = (
        f"\n[CANDIDATE INTENT SUMMARY] (confidence={confidence:.2f}):\n"
        + "\n".join(f"  - {p}" for p in parts)
        + "\nUse this intent only as a reference. "
        "Prioritize schema context and the question if any conflict appears."
    )
    return hint


def make_generation_prompt(ctx: str, question: str, router_decision: dict = None, state: "RAGState" = None) -> str:
    """
    Dynamic Prompt Builder.
    Versi Benchmark Standard (ANSI SQL Strict):
    1. Memaksa pengulangan ekspresi di GROUP BY (ANSI Compliant).
    2. Melarang UNION untuk binning.
    3. Memaksa Single Line SQL untuk mencegah JSON Crash pada query panjang.
    4. Menangani Error 1055 dengan strict column selection.
    5. Dynamic join context from retrieval results.
    """
    router_decision = router_decision or {}
    complexity = router_decision.get("complexity", "simple")
    qtype = router_decision.get("qtype", "lookup")
    has_grouping = router_decision.get("has_grouping", False)
    has_exclusion = router_decision.get("has_exclusion", False)
    is_ordinal_ranking = router_decision.get("is_ordinal_ranking", False)
    is_window_aggregate = router_decision.get("is_window_aggregate", False)
    partition_hint = router_decision.get("partition_hint", None)
    is_explicit_rank_function = router_decision.get("is_explicit_rank_function", False)
    is_winner_row = router_decision.get("is_winner_row", False)
    is_set_operation_intent = router_decision.get("is_set_operation_intent", False)
    is_simple_set_operation = router_decision.get("is_simple_set_operation", False)
    is_simple_scalar_subquery_single_table = router_decision.get("is_simple_scalar_subquery_single_table", False)
    router_tables = state.get("router_tables", []) if state else []
    
    # Detect tiebreaker pattern: "if multiple...choose the one with latest/earliest"
    has_tiebreaker = bool(re.search(
        r"\bif\s+(multiple|several|there\s+are\s+ties?|same)\b.*\b(choose|pick|select|take)\s+(the\s+)?(one|latest|earliest|first|last)\b",
        question.lower()
    ))
    
    # --- 1. BANGUN CONSTRAINT DINAMIS ---
    constraints = []

    # === [PRIORITY 1] TOP N GLOBAL RANKING RULE (FIRST - before all other rules) ===
    # Detect "Top N" or "N highest/lowest" pattern that should use ORDER BY + LIMIT
    # This is DIFFERENT from ordinal ranking ("second highest per group")
    q_lower = question.lower()
    # Pattern: "10 highest", "top 10", "10 largest", "10 products with the highest"
    top_n_pattern = re.search(r'\b(top\s+\d+|\d+\s+(highest|lowest|largest|smallest|most|least|tertinggi|terendah|terbesar|terkecil))\b', q_lower)
    # Also check for generic noun-phrase variants like:
    # "10 products with the highest ...", "5 territories with the most ...",
    # "3 result rows with the lowest ..."
    top_n_context = re.search(
        r"\b\d+\s+(?:[a-z_][a-z0-9_]*\s+){0,4}(?:with\s+(?:the\s+)?)?(highest|lowest|largest|smallest|most|least)\b",
        q_lower,
        re.IGNORECASE,
    )

    is_top_n_global_ranking = bool(top_n_pattern or top_n_context) and not is_ordinal_ranking

    if is_top_n_global_ranking:
        constraints.append(
            "TOP N QUERY (HIGHEST PRIORITY): This question asks for 'Top N' or 'N highest/lowest'. "
            "YOU MUST USE: ORDER BY column DESC LIMIT N. "
            "EXAMPLE SHAPE: SELECT <requested columns> FROM <main table> ORDER BY <requested metric> DESC LIMIT N. "
            "FORBIDDEN: Do NOT use MAX() in WHERE clause. Do NOT use subquery with LIMIT. "
            "FORBIDDEN: WHERE col = (SELECT MAX(col) ...). "
            "CORRECT: Just ORDER BY and LIMIT directly on the main table."
        )

    # === [PRIORITY 2] GENERAL ANTI-HALLUCINATION RULE ===
    constraints.append(
        "CRITICAL RULE: Copy ALL column names EXACTLY as shown in the Schema cards. "
        "Do NOT change the casing style or spelling of schema columns."
    )
    constraints.append(
        "CRITICAL RULE: Do NOT add SQL clauses (WHERE, HAVING, LIMIT, ORDER BY) "
        "unless the user EXPLICITLY requests them in the question. "
        "NEVER add filters based on STATS or sample values - only use what the user asks for."
    )
    if _question_wants_count_distinct(question):
        constraints.append(
            "COUNT DISTINCT RULE (CRITICAL): When the question asks how many distinct, unique, or different values appear, "
            "use COUNT(DISTINCT target_column). "
            "Do NOT use plain COUNT(*) or COUNT(id) for that metric. "
            "Do NOT GROUP BY the distinct target column unless the question explicitly asks to display that column."
        )
    requires_boundary_regex = _question_requires_word_boundary_regex(question)
    requires_regex = _question_requires_regex(question)
    requires_structural_regex = _question_requires_structural_regex(question)
    if requires_boundary_regex:
        constraints.append(
            "REGEX RULE (CRITICAL): This question needs true regex boundary semantics. "
            "Use REGEXP (or RLIKE), not LIKE. "
            "Do NOT weaken whole-word or word-boundary intent into a simple substring match."
        )
    elif requires_regex:
        constraints.append(
            "REGEX RULE (CRITICAL): This question requires a structural pattern match. "
            "Use REGEXP (or RLIKE), not LIKE, when the shape must be exact."
        )
        constraints.append(
            "REGEX RULE (CRITICAL): When the question describes an exact structure, prefer anchored regex with ^ and $. "
            "Do NOT use a loose LIKE pattern for exact structural matching."
        )
        if requires_structural_regex:
            constraints.append(
                "REGEX RULE (CRITICAL): For exact word-count or space-separated structure (for example 'exactly two words'), "
                "use a structural REGEXP pattern. "
                "Do NOT reduce it to `LIKE '% %'`, because that only checks for any space and is semantically too weak."
            )
    if is_set_operation_intent:
        union_keyword = "UNION" if re.search(
            r"\b(?:remove\s+duplicate\s+rows|without\s+duplicates|no\s+duplicate|distinct)\b",
            question.lower(),
        ) else "UNION ALL"
        constraints.append(
            f"SET OPERATION RULE (CRITICAL): This question requires a set operation. "
            f"Use {union_keyword} to combine branch queries into one result. "
            "Each branch MUST return the same number of columns in the same order and compatible types."
        )
        constraints.append(
            "SET OPERATION RULE (CRITICAL): If the question asks for an identifier plus a label, "
            "return them as TWO SEPARATE columns. "
            "Do NOT pack them into one CONCAT(...) column. "
            "Example pattern: `SELECT entity_id, 'label_a' AS label_name ... UNION SELECT entity_id, 'label_b' AS label_name ...`"
        )
        constraints.append(
            "SET OPERATION RULE (CRITICAL): Do NOT emulate UNION by using multiple CASE WHEN columns in one SELECT. "
            "Wrong: `SELECT CASE WHEN ... THEN entity_id END AS label_a, CASE WHEN ... THEN entity_id END AS label_b FROM ...`. "
            "Correct: one SELECT branch per label, then combine branches with UNION or UNION ALL."
        )
        constraints.append(
            "SET OPERATION RULE (CRITICAL): If each UNION branch represents one specific label or year, "
            "assign that branch label as a constant literal in the SELECT list. "
            "Do NOT reuse the same CASE WHEN label expression in every branch. "
            "Wrong: both branches use the same CASE-based label expression. "
            "Correct: first branch selects `'label_a' AS label_name`, second branch selects `'label_b' AS label_name`."
        )
        if is_simple_set_operation:
            constraints.append(
                "SIMPLE SET OPERATION RULE: Keep each branch as a simple single-table SELECT with its own WHERE filter. "
                "Do NOT use subqueries, window functions, GROUP BY, or extra joins unless the question explicitly requires them."
            )
    # === [NEW] FOR EACH = ALL VALUES, NO WHERE FILTER ===
    for_each_triggers = ["for each", "each ", "per ", "every "]
    if any(t in question.lower() for t in for_each_triggers):
        constraints.append(
            "FOR EACH RULE (CRITICAL): When the question says 'For each X' or 'Per X', you MUST return ALL values of X. "
            "Do NOT add WHERE filters on the grouping column unless the user explicitly filters it. "
            "Do NOT add subqueries to filter grouping columns. "
            "TABLE SELECTION RULE: When counting 'number of Y' per each X, use FROM the Y table (the entity being counted), "
            "NOT the dimension table for X. "
            "Example: 'for each grouping label, count entities' -> FROM entity_table GROUP BY grouping_label (NOT FROM grouping_dimension_table). "
            "The COUNT(*) counts rows in the FROM table, so use the table containing the entities to count. "
            "CORRECT: SELECT groupingCol, COUNT(*) FROM entity_table GROUP BY groupingCol "
            "WRONG: SELECT groupingCol, COUNT(*) FROM dimension_table GROUP BY groupingCol (this counts dimension rows, not entities)"
        )

    # === [NEW] EXPLICIT RANK() FUNCTION RULE ===
    # Detect "a rank of X" / "rank employees" -> use RANK() OVER window function
    if is_explicit_rank_function:
        partition_rank_hint = ""
        if partition_hint:
            partition_rank_hint = (
                f" The grouping phrase is '{partition_hint}', so partition by the column representing {partition_hint}. "
                f"Do NOT partition by an unrelated descriptive column."
            )
        constraints.append(
            "RANK FUNCTION RULE (CRITICAL): The question asks for a 'rank' of items. "
            "You MUST use the RANK() or DENSE_RANK() window function. "
            "CORRECT PATTERN: `SELECT col1, col2, RANK() OVER (PARTITION BY groupCol ORDER BY orderCol DESC) AS employee_rank FROM table` "
            "WRONG: Using COUNT(*) - that gives a total count, NOT a rank. "
            "WRONG: Using ROW_NUMBER() - use RANK() for tied values. "
            "IMPORTANT: RANK() assigns the same rank to equal values and skips ranks for ties."
            f"{partition_rank_hint}"
        )

    # === [NEW] WINDOW AGGREGATE RULE: "within the same X" pattern ===
    if is_window_aggregate:
        partition_col_hint = ""
        if partition_hint:
            # Try to infer column name from hint (e.g., "office" -> "officeCode")
            partition_col_hint = (
                f" The grouping phrase is '{partition_hint}'. Look for the schema column that represents {partition_hint}, "
                f"such as `{partition_hint}Code`, `{partition_hint}Number`, `{partition_hint}Id`, or the exact descriptive column. "
                f"Do NOT partition by another selected attribute just because it also appears in the output."
            )
        constraints.append(
            f"WINDOW AGGREGATE RULE (CRITICAL): The phrase 'within the same' or 'in the same' means you MUST use a WINDOW FUNCTION with PARTITION BY. "
            f"CORRECT: `SELECT col1, col2, COUNT(*) OVER (PARTITION BY groupColumn) AS total FROM table` "
            f"WRONG: `COUNT(*) OVER ()` without PARTITION BY. "
            f"WRONG: Using GROUP BY with window functions.{partition_col_hint} "
            f"IMPORTANT: Do NOT add GROUP BY clause when using window functions for this pattern."
        )

    has_offset_window_intent = bool(
        re.search(r"\b(?:previous|prior|next|following)\b", question.lower())
        and (
            re.search(r"\bordered\s+by\b", question.lower())
            or re.search(r"\bbased\s+on\b[^.]{0,80}\border\b", question.lower())
            or re.search(r"\b(?:date|time|timestamp)[-\s]+order\b", question.lower())
        )
    )
    if has_offset_window_intent:
        constraints.append(
            "OFFSET WINDOW RULE (CRITICAL): When the question asks for a previous/prior value, use LAG(...). "
            "When it asks for a next/following value, use LEAD(...). "
            "Use a window function with PARTITION BY the entity group (for example the same customer) and ORDER BY the sequence column named in the question. "
            "If the question says 'ordered by X' without saying descending, use ascending order for the window sequence."
        )

    # [A] STATS UTILIZATION (profiling statistics)
    constraints.append(
        "DATA INTELLIGENCE: Use the provided 'STATS' in the Context responsibly:\n"
        "   - SAMPLES: Use exact string values found in 'Sample=...'. (e.g. if Sample='USA', do not query 'United States').\n"
        "   - RANGE/DISTRIBUTION: `Min`/`Max` values are provided ONLY as a reference to describe data distribution. "
        "     NEVER hardcode these values into the query. Use SQL logic (e.g. `ORDER BY`, `MAX()`, comparison operators) instead."
    )
    # [B] SIMPLE QUERY RULES
    if complexity == "simple":
        constraints.append("SCOPE: This query requires ONE TABLE ONLY. DO NOT JOIN.")
        constraints.append("FILTERING: Filter by ID/Code directly in the main table.")

        if qtype == "lookup" and not is_top_n_global_ranking:
            constraints.append(
                "SIMPLE LOOKUP RULE: Do NOT use aggregation functions like MAX(), MIN(), or SUM(). "
                "Just SELECT the column directly. Use 'ORDER BY column DESC' for 'latest/newest/highest'."
            )
        elif qtype == "aggregate":
            # Simple aggregate queries must use GROUP BY
            constraints.append(
                "AGGREGATE RULE: When computing aggregates (COUNT, SUM, AVG, MIN, MAX) 'per each X', "
                "you MUST use GROUP BY on the grouping column. Use HAVING for filters on aggregates, NOT WHERE."
            )

    if is_simple_scalar_subquery_single_table:
        constraints.append(
            "SIMPLE SCALAR SUBQUERY RULE (CRITICAL): This question compares rows to a single-table scalar baseline such as "
            "the overall average, MAX(date), or MIN(date). "
            "You MUST keep the base table as the main FROM table and use a scalar subquery in WHERE. "
            "Example: `WHERE date_col = (SELECT MAX(date_col) FROM base_table)` or "
            "`WHERE metric_col > (SELECT AVG(metric_col) FROM base_table)`."
        )
        constraints.append(
            "SIMPLE SCALAR SUBQUERY RULE (CRITICAL): Do NOT replace the scalar subquery with only ORDER BY/LIMIT on the outer query. "
            "If the question asks for all rows on the latest or earliest date, filter to that date first, then apply any requested final ORDER BY."
        )

    # [C] RANKING RULES
    # Check winner-row pattern FIRST (before qtype check)
    # This ensures winner-row constraints are applied even when qtype="aggregate" due to grouping
    if is_winner_row or qtype == "rank":
        constraints.append(
            "WINNER-ROW STRATEGY: You MUST use the standard 'Select-Filter-Rank' pattern."
        )
        if len(router_tables) == 1 and router_tables[0]:
            constraints.append(
                f"SINGLE-TABLE RANKING RULE (CRITICAL): This ranking query should use ONLY the `{router_tables[0]}` table. "
                "Do NOT add joins to lookup tables unless the question explicitly asks for columns that only exist there."
            )

        # ORDINAL RANKING: Detect the specific rank number from question
        if is_ordinal_ranking:
            # Extract the ordinal number from question
            q_lower = question.lower()
            ordinal_map = {
                "second": 2, "2nd": 2, "kedua": 2,
                "third": 3, "3rd": 3, "ketiga": 3,
                "fourth": 4, "4th": 4, "keempat": 4,
                "fifth": 5, "5th": 5, "kelima": 5,
            }
            rank_n = 1  # default
            for word, num in ordinal_map.items():
                if word in q_lower:
                    rank_n = num
                    break
            
            constraints.append(
                f"ORDINAL RANKING RULE (CRITICAL): The question asks for the {rank_n}th highest/lowest item. "
                f"Use SIMPLE pattern: `WHERE rn = {rank_n}` in the outer query. "
                f"This AUTOMATICALLY excludes entities with fewer than {rank_n} records - NO NEED for HAVING or extra COUNT checks."
            )
            constraints.append(
                f"STRICT SYNTAX (FOLLOW EXACTLY): "
                f"`SELECT t.col1, t.col2 FROM (SELECT col1, col2, ROW_NUMBER() OVER (PARTITION BY groupCol ORDER BY valueCol DESC) AS rn FROM table) t WHERE t.rn = {rank_n}` "
                f"Do NOT use GROUP BY, HAVING, or MAX(CASE WHEN...) - just WHERE rn = {rank_n} is enough!"
            )
            constraints.append(
                "ORDINAL RANKING RULE (CRITICAL): Do NOT use MySQL session variables such as `@rn`, `@cn`, or manual running counters. "
                "Use the standard ANSI window function `ROW_NUMBER()` only."
            )
            constraints.append(
                f"REDUNDANT PATTERN WARNING: When you use `WHERE rn = {rank_n}`, each group already has exactly ONE row. "
                f"THEREFORE: (1) NO GROUP BY needed - it's redundant! (2) NO MAX(CASE WHEN rn = {rank_n}...) needed - just select the column directly! "
                f"WRONG: `SELECT t.customer, MAX(CASE WHEN rn = {rank_n} THEN amount END) FROM (...) t WHERE t.rn = {rank_n} GROUP BY t.customer` "
                f"CORRECT: `SELECT t.customer, t.amount FROM (...) t WHERE t.rn = {rank_n}`"
            )
            constraints.append(
                f"ALIAS CONSISTENCY (CRITICAL): Use the SAME alias everywhere! "
                f"WRONG: `SELECT p.col FROM (...) t WHERE t.rn = {rank_n}` - 'p' is undefined! "
                f"CORRECT: `SELECT t.col FROM (...) t WHERE t.rn = {rank_n}` - same alias 't' everywhere!"
            )
        else:
            constraints.append(
                "WINNER-ROW RULE (CRITICAL): You MUST use ROW_NUMBER() subquery pattern ONLY. "
                "CORRECT: `SELECT t.col1, t.col2 FROM (SELECT col1, col2, ROW_NUMBER() OVER (PARTITION BY groupCol ORDER BY valueCol DESC) AS rn FROM table) t WHERE t.rn = 1` "
                "WRONG: Using MAX() with GROUP BY - this does NOT give you the correct row with all its columns! "
            )

            # Add tiebreaker constraint if detected
            if has_tiebreaker:
                constraints.append(
                    "TIEBREAKER RULE (CRITICAL): The question says 'if multiple, choose the one with latest/earliest'. "
                    "You MUST add a SECOND column to ORDER BY for tiebreaking: "
                    "`ROW_NUMBER() OVER (PARTITION BY groupCol ORDER BY primaryCol [direction], tiebreakCol [direction]) AS rn` "
                    "Use DESC for 'latest/highest/largest', ASC for 'earliest/lowest/smallest'. "
                    "DO NOT use MAX(), GROUP BY, LEFT JOIN, or correlated subquery - just the simple subquery pattern!"
                )

            # ANTI-PATTERN PROHIBITION: Prevent mixing MAX() GROUP BY with ROW_NUMBER()
            constraints.append(
                "STRICT PROHIBITION (SYNTAX ERROR): DO NOT mix MAX()/MIN()/GROUP BY with ROW_NUMBER(). "
                "These are MUTUALLY EXCLUSIVE patterns. Choose ONE:\n"
                "• If you need ROW_NUMBER(): Do NOT use MAX(), MIN(), SUM(), COUNT(), GROUP BY, or HAVING\n"
                "• If you need GROUP BY aggregation: Do NOT use ROW_NUMBER(), RANK(), or window functions\n"
                "For this query, use ONLY the ROW_NUMBER() pattern - no GROUP BY, no MAX(), no aggregate functions."
            )
        
        constraints.append(
            "CRITICAL PROHIBITION: Do NOT put `ROW_NUMBER()` inside a `GROUP BY` clause. "
            "Do NOT use `HAVING` for ranking. Use the subquery pattern above."
        )

    # [D] PRIMARY TABLE GUIDANCE (CRITICAL FOR ALL MULTI-TABLE QUERIES)
    # Apply to ALL non-simple queries (lookup, aggregate, ranking) to prevent wrong FROM clause order
    # This ensures the LLM uses the correct primary table as the first table in FROM clause
    if complexity != "simple" and state:
        primary_table = state.get("primary_table")
        primary_reason = state.get("primary_table_reason")
        join_key_hints = _build_join_key_hints_for_prompt(state, max_edges=8)

        print(f"[PROMPT DEBUG] complexity={complexity}, state provided, primary_table={primary_table}")

        if primary_table:
            other_tables = state.get("router_tables", [])
            other_tables = [t for t in other_tables if t != primary_table]
            other_tables = _order_tables_by_fk_distance(primary_table, other_tables)
            requires_inner_join = _question_requires_inner_join(question, ctx)
            wants_include_zero = _question_wants_include_zero(question, ctx)
            exclusion_tables = {t for t in other_tables if _question_targets_table_exclusion(question, t)}
            existence_tables = {t for t in other_tables if _question_targets_table_existence(question, t)}
            if _question_wants_dual_fact_rollup(question) and len(other_tables) >= 2:
                direct_existence_tables = {
                    t for t in other_tables
                    if _question_has_direct_entity_existence_signal(question, t)
                }
                if direct_existence_tables:
                    existence_tables = {t for t in existence_tables if t in direct_existence_tables}
            force_required_inner_chain = _should_force_required_inner_join_chain(
                question=question,
                router_decision=router_decision,
                router_tables=state.get("router_tables", []) or [],
                ctx=ctx,
            )
            connector_path_rules = _build_connector_path_rules(primary_table, other_tables)
            if wants_include_zero and exclusion_tables:
                # Preserve include-zero semantics:
                # if a table is explicitly in exclusion/no-data intent, do not also
                # force existence for the same table at table-hint level.
                existence_tables = {t for t in existence_tables if t not in exclusion_tables}
            mixed_intent_tables = exclusion_tables.intersection(existence_tables)

            # Join type must be decided per table.
            # Global "requires_inner_join" is only a fallback when no table-level
            # existence/exclusion target can be identified.
            force_inner_all = bool(
                requires_inner_join
                and not wants_include_zero
                and not exclusion_tables
                and not existence_tables
            )
            qtype_l = str(qtype or "").strip().lower()
            exclusion_aggregate_prefers_inner_non_target = bool(
                has_exclusion
                and (qtype_l == "aggregate")
                and not wants_include_zero
            )

            # More explicit guidance - tell LLM exactly which table should be first
            hint_msg = f"TABLE ORDER (CRITICAL): Start your query with FROM {primary_table}"
            if other_tables:
                join_steps = []
                for t in other_tables:
                    join_target = str(t).strip()
                    if wants_include_zero and t in exclusion_tables:
                        join_steps.append(f"LEFT JOIN {join_target}")
                    elif force_required_inner_chain:
                        join_steps.append(f"INNER JOIN {join_target}")
                    elif t in mixed_intent_tables:
                        join_steps.append(f"INNER JOIN {join_target}")
                    elif t in exclusion_tables:
                        join_steps.append(f"LEFT JOIN {join_target}")
                    elif t in existence_tables:
                        join_steps.append(f"INNER JOIN {join_target}")
                    elif force_inner_all:
                        join_steps.append(f"INNER JOIN {join_target}")
                    elif exclusion_aggregate_prefers_inner_non_target:
                        # In exclusion+aggregate questions, non-target supporting tables
                        # are usually part of required positive slice (e.g., metric fact tables).
                        join_steps.append(f"INNER JOIN {join_target}")
                    else:
                        join_steps.append(f"LEFT JOIN {join_target}")
                hint_msg += ", then " + ", ".join(join_steps)
            if primary_reason:
                hint_msg += f". Reason: {primary_reason}"
            hint_msg += f". Follow this exact table order."

            print(f"[PROMPT DEBUG] Adding hint: {hint_msg}")
            constraints.append(hint_msg)

            # JOIN TYPE GUIDANCE (CRITICAL)
            if wants_include_zero and exclusion_tables:
                constraints.append(
                    "JOIN TYPE (CRITICAL): Include-zero intent detected (e.g., 'including ... with no/without'). "
                    "Preserve primary rows with LEFT JOIN for optional related tables; do NOT force INNER JOIN on exclusion-target tables."
                )
            elif force_required_inner_chain:
                constraints.append(
                    "JOIN TYPE (CRITICAL): This query needs a required INNER JOIN chain for the metric path. "
                    "Use INNER JOIN for each required supporting table in that path; do NOT weaken it to LEFT JOIN "
                    "unless the question explicitly asks to preserve unmatched primary rows."
                )
            elif mixed_intent_tables:
                constraints.append(
                    "JOIN TYPE (CRITICAL): Same table has mixed intent (both required and excluded in different slices). "
                    "Use INNER JOIN for the positive slice and enforce exclusion with NOT EXISTS/anti-join logic; "
                    "do NOT rely on a single LEFT JOIN for both intents."
                )
            elif exclusion_tables and (existence_tables or force_inner_all):
                constraints.append(
                    "JOIN TYPE (CRITICAL): Mixed intent detected. "
                    "Use INNER JOIN for required-existence tables, and LEFT JOIN + IS NULL/NOT EXISTS for exclusion tables (e.g., 'no/never/without')."
                )
                constraints.append(
                    "ANTI-JOIN (CRITICAL): For exclusion intent, prefer NOT EXISTS over NOT IN. "
                    "Correlate the excluded table by key."
                )
            elif existence_tables:
                constraints.append(
                    "JOIN TYPE (CRITICAL): Use INNER JOIN only for tables explicitly required by existence intent "
                    "(e.g., 'at least one', 'with/having/placed/paid ...'). "
                    "Use LEFT JOIN for non-target related tables to avoid dropping primary rows."
                )
            elif force_inner_all:
                # Filtering pattern: use INNER JOIN to exclude non-matching rows
                constraints.append(
                    "JOIN TYPE (CRITICAL): This query has explicit filtering intent (e.g., 'only ... that have/with ...'). "
                    "Use **INNER JOIN** to exclude entities without matching data. "
                    "This takes precedence over 'for each' phrasing."
                )
            elif exclusion_aggregate_prefers_inner_non_target:
                constraints.append(
                    "JOIN TYPE (CRITICAL): Exclusion + aggregate intent detected. "
                    "Use INNER JOIN for positive-slice supporting tables (non-exclusion targets), "
                    "and keep exclusion logic with NOT EXISTS / anti-join pattern."
                )
            else:
                # Default: use LEFT JOIN to preserve all entities from primary table
                constraints.append(
                    "JOIN TYPE (CRITICAL): Use **LEFT JOIN** to preserve all entities from the primary table."
                )
                constraints.append(
                    "FILTER RULE (CRITICAL): If using LEFT JOIN, filters for the JOINED table MUST be in the **ON** clause, NOT WHERE."
                )

            for connector_rule in connector_path_rules:
                constraints.append(connector_rule)

        if join_key_hints:
            constraints.append(
                "JOIN KEY HINTS (CRITICAL): Use only FK-consistent JOIN predicates from this list: "
                + "; ".join(join_key_hints)
                + "."
            )
            constraints.append(
                "JOIN PATH RULE (CRITICAL): Use the listed FK edges as an actual connected path. "
                "Do NOT invent a direct join between two tables unless that direct FK edge is listed."
            )

    # [E] REPORTING/AGGREGATION RULES
    # Skip aggregation rules for winner-row queries (they use ROW_NUMBER, not GROUP BY)
    is_reporting = (qtype == "aggregate") or has_grouping or has_exclusion
    if is_reporting and not is_winner_row:
        constraints.append(
            "GROUP BY RULE: Every column in the SELECT list that is not aggregated MUST be in the GROUP BY clause."
        )

    # [E.1] PERCENTAGE/RATIO CALCULATION RULES
    percentage_triggers = ["percentage", "percent", "ratio", "proportion"]
    if any(t in question.lower() for t in percentage_triggers):
        constraints.append(
            "PERCENTAGE CALCULATION RULE: When asked for percentage, you MUST compute it as a formula. "
            "Do NOT just return counts; return the actual percentage value."
        )

    # [F] DATE RULES
    constraints.append(
        "DATE LABEL STRICT RULE: If asking for Month Name, GROUP BY both MONTH(date) and DATE_FORMAT(date, '%M')."
        "DATE FILTERING RULES:\n"
        "   - For year filtering: Use `YEAR(date_column) = YYYY` (NOT SUBSTR).\n"
        "   - For month filtering: Use `MONTH(date_column) = M`.\n"
        "   - For range filtering: Use `date_column BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD'`."
    )
    constraints.append(
        "JOIN ALIAS RULE (CRITICAL): In each JOIN ... ON clause, only reference aliases that are already introduced "
        "(current table alias + previously joined aliases). NEVER reference a future alias."
    )
    constraints.append(
        "DATE COLUMN VALIDATION RULE (CRITICAL): Apply YEAR()/DATE filters only on columns that exist in the schema cards. "
        "Do NOT invent date columns on non-date tables."
    )

    # [F.1] LATEST-YEAR + SAME-YEAR SUM RULE
    if _question_wants_latest_year_scope_with_aggregate(question):
        constraints.append(
            "LATEST-YEAR AGGREGATION RULE (CRITICAL): If the question asks for latest/most-recent year and "
            "sum in that same year, do NOT use `MAX(date)` + `SUM(amount)` on all rows in one GROUP BY. "
            "Use 2-step pattern: (1) subquery to get latest year per entity, "
            "(2) join/filter rows to that year, then compute `MAX(date)` and `SUM(amount)`."
        )

    # [G] COLUMN INTEGRITY RULES (BENCHMARK MODE)
    # Extract explicitly mentioned columns from question for stricter enforcement
    import re as _re
    q_lower = question.lower()

    # PRIORITY: Check for "Include X, Y, and Z" pattern (highest priority)
    include_pattern = r'\binclude\s+(?:the\s+)?(.+?)(?:\.|$)'
    include_match = _re.search(include_pattern, q_lower)

    requested_cols = []
    explicit_include = False

    if include_match:
        explicit_include = True
        include_text = include_match.group(1)
        # Split by comma and "and"
        parts = _re.split(r',\s*|\s+and\s+', include_text)
        for p in parts:
            p = p.strip()
            # Remove articles and stop words at the end
            p = _re.sub(r'\s+(in\s+(?:the\s+)?result|respectively)?$', '', p)
            if p and len(p) > 2:
                requested_cols.append(p)

    # If not "Include" pattern, try other patterns
    if not explicit_include:
        col_pattern = r'\b(return|show|display|get)\s+(?:the\s+)?(.+?)(?:\.|$)'
        col_match = _re.search(col_pattern, q_lower)
        if col_match:
            col_text = col_match.group(2)
            # Split by comma and "and"
            parts = _re.split(r',\s*|\s+and\s+', col_text)
            requested_cols = [p.strip() for p in parts if p.strip() and len(p.strip()) > 2]

    if requested_cols:
        cols_list = ", ".join([f"'{c}'" for c in requested_cols[:5]])  # Limit to 5
        if explicit_include:
            # Strongest constraint for "Include X, Y, Z" pattern
            constraints.append(
                f"CRITICAL COLUMN REQUIREMENT: The question explicitly says 'Include {cols_list}'. "
                f"Select ONLY these columns (plus the 'For each' entity identifier or display label when required). "
                f"Do NOT add any other unrequested identifiers, labels, or auxiliary columns."
            )
        else:
            constraints.append(
                f"EXPLICIT COLUMN REQUIREMENT: The user explicitly requested these items: {cols_list}. "
                f"You MUST include ALL of them in your SELECT clause. Do not skip any."
            )
            # Stronger constraint: prohibit extra columns
            constraints.append(
                f"STRICT COLUMN LIMIT: Select ONLY these columns: {cols_list}. "
                f"Do NOT add any other columns (IDs, codes, names) that were not explicitly requested."
            )
    else:
        constraints.append(
            "STRICT COLUMN SELECTION: Select EXACTLY the columns requested by the user. "
            "Do NOT add extra identifiers, codes, labels, or auxiliary columns unless explicitly mentioned."
        )
    constraints.append(
        "IMPLICIT SELECTION RULE: If the user says 'For each [Entity]' or 'List [Entity]', "
        "you MUST include that entity's identifier or display label in the SELECT list, even if not explicitly listed in 'Include ...'. "
        "Example: 'For each grouping entity, show the aggregated metric' -> "
        "You MUST select the grouping entity identifier or display label AND the aggregated metric."
    )
    constraints.append(
        "NO EXTRA COLUMNS: Apart from the required grouping entity identifier or display label, "
        "do NOT add extra identifiers, codes, or labels unless asked."
    )
    constraints.append(
        "ALIAS CONSISTENCY: If you use a table alias (e.g. `t`, `p`, `cust`), "
        "you MUST prefix ALL columns in the SELECT clause with that alias. "
        "Bad: `SELECT p.id, name FROM ...` | Good: `SELECT p.id, p.name FROM ...`"
    )
    constraints.append(
        "ALIAS STYLE: Let aliases be concise and readable. Prefer short snake_case names (roughly <= 30-36 chars)."
    )
    # [G] ANTI-HALLUCINATION & LOGIC EFFICIENCY
    # 1. Month Sorting
    if "month" in question.lower() and ("order" in question.lower() or "sort" in question.lower()):
        constraints.append(
            "CHRONOLOGICAL SORTING RULE: If grouping by Month Name, you MUST also ORDER BY the month number `MONTH(dateColumn)`."
        )
        constraints.append(
            "GROUP BY CONSISTENCY: To avoid SQL Error 1055, if you ORDER BY `MONTH(dateColumn)`, "
            "you MUST also include `MONTH(dateColumn)` in the GROUP BY clause."
        )

    # 3. Categorization Strategy (ANSI SQL STANDARD FIX)
    # Memaksa ANSI Standard (Repeat Expression) tapi melarang UNION.
    if "case" in question.lower() or "band" in question.lower() or "range" in question.lower():
        constraints.append(
            "BINNING STRATEGY: Use a `CASE WHEN ... END` statement in a **SINGLE** SELECT query. "
            "ANSI SQL RULE: You MUST repeat the exact `CASE WHEN` expression in the `GROUP BY` clause. "
            "Do NOT group by the alias. Do NOT use UNION."
        )

    # 4. Aggregation Safety (Anti Error 1055)
    if has_grouping or "group" in question.lower():
        constraints.append(
            "STRICT GROUP BY RULE (Error 1055): If you GROUP BY a derived column (e.g. `CASE WHEN`), "
            "you MUST NOT select the original raw column (e.g. `MSRP`) unless it is wrapped in an aggregate function (COUNT, SUM). "
            "ONLY select the derived expression and Aggregates."
        )
    # 5. Pattern Matching Rule (Balanced & Robust)
    # Trigger dipersempit hanya untuk kata yang benar-benar butuh presisi
    pattern_triggers = ["exactly", "digit", "alphanumeric", "alnum", "followed by"]
    
    if any(t in question.lower() for t in pattern_triggers):
        constraints.append(
            "PATTERN LOGIC: Use `LIKE` for simple prefix/suffix. "
            "Use `REGEXP` ONLY for strict requirements: "
            "1. Exact digit/character counts (Template: `REGEXP '^[0-9]{N}$'`). "
            "2. Mixed types like Letters followed by Digits (Template: `REGEXP '^[a-z]+[0-9]+$'`). "
            "Always use `^` and `$` anchors in REGEXP to ensure a full-string match."
        )

    # 6. Sorting Direction
    desc_triggers = [
    "latest", "newest", "most recent", "descending", 
    "highest", "greatest", "largest", "biggest"
    ]

    asc_triggers = [
        "ascending", "alphabetical", "alphabetically",
        "smallest", "lowest", "oldest first", "earliest first"
    ]

    # Default untuk "sorted" tanpa qualifier
    sort_requested = any(k in question.lower() for k in ["sort", "sorted", "order by"])

    question_lower = question.lower()

    if any(t in question_lower for t in desc_triggers):
        constraints.append(
            "SORT DIRECTION: The user implies a descending order. You MUST append `DESC` to the ORDER BY clause."
        )
    elif any(t in question_lower for t in asc_triggers):
        constraints.append(
            "SORT DIRECTION: The user requests ascending order. You MUST append `ASC` to the ORDER BY clause."
        )
    elif sort_requested:
        # "sorted by X" without qualifier → default ascending (natural order)
        constraints.append(
            "SORT DIRECTION: The user requests sorting without specifying direction. "
            "Use ascending order (ASC) as default, which means A-Z for text, 0-9 for numbers, oldest-to-newest for dates. "
            "You can omit ASC keyword (ascending is SQL default)."
        )
    else:
        constraints.append(
            "NO UNREQUESTED SORTING: Do NOT add 'ORDER BY' clause unless the user explicitly asks for ranking or sorting."
        )

    # 6. JSON FORMATTING (CRITICAL FOR 3B STABILITY)
    # Ini tidak melanggar standar SQL (karena SQL whitespace insensitive), 
    # tapi vital agar 3B tidak crash saat generate query panjang (CASE WHEN).
    constraints.append(
        "JSON FORMATTING: The 'sql' field must be a SINGLE LINE string. Remove all newlines (\\n) from the SQL."
    )

    # --- 2. BUILD DYNAMIC JOIN CONTEXT (from retrieval) ---
    join_context = ""
    if state:
        join_context = _build_dynamic_join_context(state, router_decision)

    # --- 3. RAKIT CONSTRAINT BLOCK ---
    constraint_block = ""
    if constraints:
        constraint_block = "\n[STRICT CONSTRAINTS & HEURISTICS]:\n" + "\n".join([f"- {c}" for c in constraints])

    # --- 3.5. INTENT IR SOFT HINT (IR-informed LLM) ---
    intent_hint = ""
    if state:
        intent_hint = _build_intent_ir_hint(state)
        if intent_hint:
            state["ir_hint_passed"] = True
            ir_conf = float((state.get("intent_ir") or {}).get("confidence", 0.0) or 0.0)
            ir_reasons = state.get("ir_gate_reasons") or state.get("ir_base_gate_reasons") or []
            print(
                f"[IR-HINT] Intent IR passed as soft hint to LLM "
                f"(confidence={ir_conf:.2f}, "
                f"gate_reasons={ir_reasons or ['planner_fallback']})"
            )
            print(f"[IR-HINT] Content:{intent_hint}")
        else:
            ir_data = state.get("intent_ir") or {}
            ir_conf = float(ir_data.get("confidence", 0.0) or 0.0)
            ir_used = state.get("ir_used", False)
            base_reasons = set(state.get("ir_base_gate_reasons") or [])
            gate_reasons = set(state.get("ir_gate_reasons") or [])
            skip_reason = (
                "ir_used_as_planner" if ir_used
                else "no_intent_ir" if not ir_data
                else f"low_confidence({ir_conf:.2f}<0.50)" if ir_conf < 0.50
                else "empty_ir_payload" if "empty_ir_payload" in base_reasons
                else "planner_incompatible" if "planner_incompatible" in base_reasons
                else "having_like_threshold_unresolved" if "having_like_threshold_unresolved" in gate_reasons
                else "aggregate_without_valid_metrics" if str(ir_data.get("qtype") or "").lower() == "aggregate"
                else "no_structural_fields"
            )
            print(f"[IR-HINT] Skipped: {skip_reason}")

    # --- 4. RETURN PROMPT LENGKAP ---
    return fr"""
You are a senior SQL assistant for MySQL 8.0.

Use ONLY tables/columns listed under ALLOWED IDENTIFIERS in Context.
If Context lists aliases, use the canonical column name.

Output:
Return EXACTLY one JSON object:
{{"sql":"<final MySQL query>","notes":"<brief rationale, max 12 words>"}}
No Markdown. End SQL with a semicolon.
If NO JOINs, notes MUST be exactly: "No joins used."

Context:
{ctx}
{join_context}

Question:
{question}
{constraint_block}
{intent_hint}
""".strip()
# ======================================================================
# =========================== MAIN PIPELINE ============================
# ================ Postprocess SQL & Normalize Pipeline ================
# ======================================================================  
def _postprocess_sql(sql: str, question: str, ctx: str = "") -> str:
    DEBUG_WR = True
    if not sql:
        return sql
    s = sql.strip()
    
    # -------------------------------------------------------------------------
    # Phase -1: Fix undefined alias typos (e.g., p2.amount -> p.amount)
    # LLM sometimes hallucinates aliases like p2 when only p is defined
    # -------------------------------------------------------------------------
    s = _fix_undefined_alias_typos(s)
    
    # -------------------------------------------------------------------------
    # Phase -0.5: Normalize subquery aliases to standard (e.g., p -> pmt)
    # -------------------------------------------------------------------------
    s = _normalize_subquery_alias_to_standard(s)
    
    # CRITICAL FIX: Normalize ALL major clause boundaries FIRST
    # Pattern: ensure space before WHERE/GROUP BY/ORDER BY/HAVING/LIMIT
    # This must handle: "col1,col2HAVING" and "col1 HAVING" cases
    # NO re.I flag: creditLimit won't match (lowercase 'imit'), but HAVING will match (uppercase)
    s = re.sub(
    r"(\b[A-Za-z_][A-Za-z0-9_]*)(WHERE|GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT)\b",
    r"\1 \2",
    s)   
    s = re.sub(r"(?<=\))(?=\b(?:WHERE|GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT)\b)", " ", s, flags=re.I)
    
    # Extra safety: collapse multiple spaces (but preserve inside quotes)
    parts = re.split(r"('(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`[^`]*`)", s)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r"\s{2,}", " ", parts[i])
    s = "".join(parts)
    # Debug checkpoint - verify spacing fixed
    if DEBUG_WR and "HAVING" in s.upper():
        # Check if HAVING still stuck to previous token
        if re.search(r"[A-Za-z0-9_]HAVING", s, flags=re.I):
            print(f"[REFINER - PH4] WARNING: HAVING still stuck after normalization: {s[:200]}")

    def _norm_kw(text: str) -> str:
        CHUNKS = re.compile(
            r"(`[^`]*`|\"([^\"\\]|\\.)*\"|'([^'\\]|\\.)*')",
            re.DOTALL,
        )
        KEYWORDS = [
            "select","from","where","group by","order by","having","limit",
            "inner join","left join","right join","full join","join","on",
        ]

        def normalize_code_segment(seg: str) -> str:
            for kw in KEYWORDS:
                pattern = rf"(?<![A-Za-z0-9_]){kw}(?![A-Za-z0-9_])"
                seg = re.sub(pattern, lambda m: m.group(0).upper(), seg, flags=re.I)
            seg = re.sub(r"[ \t]+", " ", seg)
            return seg

        out, i = [], 0
        for m in CHUNKS.finditer(text):
            out.append(normalize_code_segment(text[i:m.start()]))
            out.append(m.group(0))
            i = m.end()
        out.append(normalize_code_segment(text[i:]))

        return "".join(out)

    def _rewrite_pipes_concat_mysql(text: str) -> str:
        """
        Convert common string-concat style `a || b` into MySQL-safe CONCAT(a, b).
        Applied conservatively only when `||` appears with quoted string literals.
        """
        if not text or "||" not in text:
            return text
        if not re.search(r"(?is)(\|\|\s*['\"]|['\"]\s*\|\|)", text):
            return text

        term = (
            r"(?:"
            r"`?[A-Za-z_][A-Za-z0-9_]*`?(?:\s*\.\s*`?[A-Za-z_][A-Za-z0-9_]*`?)?"
            r"|[A-Za-z_][A-Za-z0-9_]*\s*\([^()]*\)"
            r"|'(?:''|[^'])*'"
            r'|"(?:\"\"|[^"])*"'
            r"|[0-9]+(?:\.[0-9]+)?"
            r")"
        )
        pipe_pat = re.compile(rf"(?is)(?P<l>{term})\s*\|\|\s*(?P<r>{term})")

        out = text
        for _ in range(8):
            new_out = pipe_pat.sub(lambda m: f"CONCAT({m.group('l').strip()}, {m.group('r').strip()})", out)
            if new_out == out:
                break
            out = new_out
        return out

    s = _norm_kw(s)
    s = _rewrite_pipes_concat_mysql(s)
    s = _fix_missing_space_before_clauses(s)
    if not s.endswith(";"):
        s += ";"
    # ---------------------------------------------------------------------------
    # Phase 0 — Syntax & Canonicalization
    # Normalize tokens and table naming early
    # ---------------------------------------------------------------------------
    s = _fix_bad_backticks(s)
    s = _canonicalize_table_names(s, ctx)
    s = _repair_invalid_qualified_columns(s, ctx)
    s = _repair_simple_single_table_column_names(s, ctx)
    # s = ensure_join_aliases(s)  # DISABLED: Let LLM handle its own alias consistency
    s = _escape_reserved_column_aliases(s)  # Escape reserved words used as column aliases (e.g., AS rank -> AS `rank`)
    # ---------------------------------------------------------------------------
    # Phase 1 — JOIN Semantics & Structural Rewrites (FINAL)
    # Decide JOIN necessity and semantics before touching aliases
    # ---------------------------------------------------------------------------
    # For "ALL entities" questions: preserve LEFT JOIN by moving RHS filter to ON
    s = _move_rhs_filter_from_where_to_on(s, question, ctx)
    # Otherwise: convert LEFT JOIN to INNER if RHS is filtered
    s = _rewrite_left_join_to_inner_if_not_needed(s, question, ctx)
    s = _rewrite_left_join_to_inner_if_rhs_filtered(s, question, ctx)
    s = _rewrite_invalid_fk_join_conditions(s, ctx)
    s = _rewrite_invalid_grouped_existence_subquery_to_join(s, question, ctx)
    s = _rewrite_inner_join_filter_to_exists(s)
    s = _rewrite_inner_join_existence_to_exists(s, question, ctx)
    s = _remove_redundant_and_ornamental_joins(s)
    # ---------------------------------------------------------------------------
    # Phase 2 — Alias Hygiene & FROM Cleanup
    # Safe only after JOIN structure is frozen
    # ---------------------------------------------------------------------------
    s = _drop_redundant_column_aliases(s)
    s = _drop_unused_from_table_aliases(s)
    s = _strip_alias_if_single_table(s)
    # ---------------------------------------------------------------------------
    # Phase 2.5 — Force Remove AS from Plain Columns (keep for computed expressions)
    # Plain columns: columnName AS alias → columnName (remove ALL aliases)
    # Computed expressions: COUNT(*) AS alias → KEEP (functions need aliases)
    # ---------------------------------------------------------------------------
    # Plain-column alias stripping is only safe for simple single-table shapes.
    # For JOIN/subquery queries, aliases are often semantically meaningful
    # (selected output names and ORDER BY alias references), so keep them.
    if not has_join(s) and not re.search(r"\bFROM\s*\(", s, flags=re.I):
        # List of SQL functions that indicate computed expressions.
        computed_func_patterns = [
            r'COUNT\s*\(', r'SUM\s*\(', r'AVG\s*\(', r'MIN\s*\(', r'MAX\s*\(',
            r'CASE\s+WHEN', r'IF\s*\(', r'COALESCE\s*\(', r'NULLIF\s*\(',
            r'CAST\s*\(', r'CONVERT\s*\(', r'DATE_FORMAT\s*\(',
            r'ROW_NUMBER\s*\(', r'RANK\s*\(', r'DENSE_RANK\s*\(',
            r'LAG\s*\(', r'LEAD\s*\(', r'FIRST_VALUE\s*\(', r'LAST_VALUE\s*\(',
            r'CONCAT\s*\(', r'GROUP_CONCAT\s*\(', r'SUBSTRING\s*\(', r'SUBSTR\s*\(',
            r'UPPER\s*\(', r'LOWER\s*\(', r'TRIM\s*\(', r'LENGTH\s*\('
        ]

        def _find_top_level_from_idx(text: str) -> int:
            depth = 0
            in_sq = False
            in_dq = False
            in_bt = False
            i = 0
            while i < len(text):
                ch = text[i]
                if ch == "'" and not in_dq and not in_bt:
                    if in_sq and i + 1 < len(text) and text[i + 1] == "'":
                        i += 2
                        continue
                    in_sq = not in_sq
                    i += 1
                    continue
                if ch == '"' and not in_sq and not in_bt:
                    if in_dq and i + 1 < len(text) and text[i + 1] == '"':
                        i += 2
                        continue
                    in_dq = not in_dq
                    i += 1
                    continue
                if ch == "`" and not in_sq and not in_dq:
                    in_bt = not in_bt
                    i += 1
                    continue
                if in_sq or in_dq or in_bt:
                    i += 1
                    continue
                if ch == "(":
                    depth += 1
                    i += 1
                    continue
                if ch == ")":
                    depth = max(0, depth - 1)
                    i += 1
                    continue
                if (
                    depth == 0
                    and i + 4 <= len(text)
                    and text[i:i + 4].lower() == "from"
                    and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_"))
                    and (i + 4 == len(text) or not (text[i + 4].isalnum() or text[i + 4] == "_"))
                ):
                    return i
                i += 1
            return -1

        # Apply this rule ONLY to top-level SELECT list to avoid stripping table aliases
        # from FROM/JOIN clauses (e.g., "FROM customers AS cust").
        from_idx = _find_top_level_from_idx(s)
        select_segment = s if from_idx < 0 else s[:from_idx]
        remainder_segment = "" if from_idx < 0 else s[from_idx:]

        # Pattern to match: plain_column AS alias or table.column AS alias.
        # Only match if NOT a computed expression.
        def remove_plain_column_alias(match):
            column_ref = match.group(1)
            alias_name = match.group(2)
            alias_clean = alias_name.strip('\'"`')
            combined = column_ref + ' ' + alias_clean
            for func_pattern in computed_func_patterns:
                if re.search(func_pattern, combined, re.IGNORECASE):
                    return match.group(0)
            return column_ref

        def remove_plain_column_alias_quoted(match):
            column_ref = match.group(1)
            alias_name = match.group(3)
            alias_clean = alias_name.strip('\'"`')
            combined = column_ref + ' ' + alias_clean
            for func_pattern in computed_func_patterns:
                if re.search(func_pattern, combined, re.IGNORECASE):
                    return match.group(0)
            return column_ref

        # Protect CASE ... END AS alias in SELECT segment.
        case_aliases = {}
        def protect_case_aliases(match):
            alias = f"__CASE_ALIAS_{len(case_aliases)}__"
            case_aliases[alias] = match.group(0)
            return alias

        select_segment = re.sub(
            r'\bCASE\s+WHEN\b.+?\bEND\b\s+AS\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            protect_case_aliases,
            select_segment,
            flags=re.IGNORECASE | re.DOTALL
        )

        # Pattern 1: quoted aliases in SELECT list.
        select_segment = re.sub(
            r"([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)\s+AS\s+(['\"])([^'\"]+)\2",
            remove_plain_column_alias_quoted,
            select_segment,
            flags=re.IGNORECASE
        )

        # Pattern 2: unquoted aliases in SELECT list.
        select_segment = re.sub(
            r"([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)\s+AS\s+([a-zA-Z_][a-zA-Z0-9_]*)",
            remove_plain_column_alias,
            select_segment,
            flags=re.IGNORECASE
        )

        # Restore protected CASE aliases and recombine SQL.
        for alias, original in case_aliases.items():
            select_segment = select_segment.replace(alias, original)
        s = select_segment + remainder_segment

    # NEW: Normalize overly long aliases by removing filler words
    s = _normalize_alias_lengths(s)
    # ---------------------------------------------------------------------------
    # Phase 3 — Column Naming & GROUP BY Consistency
    # Operate on final column identities
    # ---------------------------------------------------------------------------
    s = _prefer_friendly_names(s, question)
    s = _ensure_entity_id_and_name_if_requested(s, question)
    s = _ensure_explicit_simple_projection_columns(s, question, ctx)
    s = _prune_non_primary_dimensions_for_primary_aggregate(s, question, ctx)
    s = _drop_redundant_raw_name_columns_when_concat_present(s, question)
    s = _expand_group_by_aliases_regex(s)
    latest_year_sum_intent = _question_wants_latest_year_scope_with_aggregate(question)
    has_correlated_max_subquery = bool(
        re.search(
            r"\bWHERE\b[\s\S]*?\b(?:IN|=)\s*\(\s*SELECT\s+MAX\s*\(",
            s,
            flags=re.IGNORECASE,
        )
    )
    skip_phase3_groupby_cleanup = latest_year_sum_intent and has_correlated_max_subquery
    if not _is_potential_winner_row(s, question, force=True):
        # REMOVE invalid columns FIRST, then enforce GROUP BY consistency
        # This prevents columns from being added to GROUP BY unnecessarily
        if not skip_phase3_groupby_cleanup:
            s = _remove_invalid_select_columns_with_group_by(s)
            s = _enforce_explicit_group_by(s)
            # Fix ORDER BY to use GROUP BY expressions instead of raw columns
            # e.g., ORDER BY o.orderDate → ORDER BY MONTH(o.orderDate)
            s = _fix_order_by_group_by_consistency(s)
    # ---------------------------------------------------------------------------
    # Phase 3b — Fix invalid GROUP BY syntax (strip DESC/ASC)
    # DESC/ASC is only valid in ORDER BY, not GROUP BY
    # ---------------------------------------------------------------------------
    s = re.sub(r'\bGROUP\s+BY\s+(.*?)(?=\s*(?:HAVING|ORDER\s+BY|LIMIT|;|$))', 
               lambda m: 'GROUP BY ' + re.sub(r'\s+(DESC|ASC)\b', '', m.group(1), flags=re.IGNORECASE),
               s, flags=re.IGNORECASE | re.DOTALL)
    # ---------------------------------------------------------------------------
    # Phase 3c — Reorder GROUP BY: Keys First, Descriptive Columns Later
    # ---------------------------------------------------------------------------
    group_match = re.search(
        r'\bGROUP\s+BY\s+(.*?)(?=\s*(?:HAVING|ORDER\s+BY|LIMIT|;|$))',
        s,
        re.IGNORECASE | re.DOTALL
    )
    
    if group_match:
        group_clause = group_match.group(1).strip()
        group_items = [item.strip() for item in group_clause.split(',')]
        
        if len(group_items) > 1:
            key_patterns = [r'Number$', r'Code$', r'ID$', r'Id$', r'_id$']
            
            key_cols = []
            desc_cols = []
            
            for item in group_items:
                col_name = item.split('.')[-1].strip('`').strip()
                is_key = any(re.search(pattern, col_name, re.IGNORECASE) for pattern in key_patterns)
                
                if is_key:
                    key_cols.append(item)
                else:
                    desc_cols.append(item)
            
            if key_cols:
                reordered = key_cols + desc_cols
                new_group_clause = ', '.join(reordered)
                s = s[:group_match.start(1)] + new_group_clause + s[group_match.end(1):]
    # ---------------------------------------------------------------------------
    # Phase 3.5 — Fix Window Function + GROUP BY Anti-Pattern
    # LLM 7B sometimes incorrectly uses MIN()/MAX()/COUNT() OVER (PARTITION BY ...)
    # when simple GROUP BY aggregate would suffice. This causes Error 1055.
    # Fix: Remove OVER (PARTITION BY ...) from aggregates when GROUP BY is present.
    # Example: MIN(orderDate) OVER (PARTITION BY status) -> MIN(orderDate)
    #
    # CRITICAL: DO NOT strip OVER from window functions (ROW_NUMBER, RANK, etc.)
    # These REQUIRE the OVER clause - removing it makes invalid SQL!
    # ---------------------------------------------------------------------------
    s_upper = s.upper()

    def _has_top_level_group_by(text: str) -> bool:
        up = text.upper()
        depth = 0
        in_sq = False
        in_dq = False
        i = 0
        while i < len(text):
            ch = text[i]
            if ch == "'" and not in_dq:
                if in_sq and i + 1 < len(text) and text[i + 1] == "'":
                    i += 2
                    continue
                in_sq = not in_sq
                i += 1
                continue
            if ch == '"' and not in_sq:
                in_dq = not in_dq
                i += 1
                continue
            if in_sq or in_dq:
                i += 1
                continue
            if ch == "(":
                depth += 1
                i += 1
                continue
            if ch == ")":
                depth = max(0, depth - 1)
                i += 1
                continue
            if depth == 0 and up[i:i + 8] == "GROUP BY":
                left_ok = (i == 0 or not text[i - 1].isalnum())
                right_ok = (i + 8 >= len(text) or not text[i + 8].isalnum())
                if left_ok and right_ok:
                    return True
            i += 1
        return False

    # Skip if query uses window functions that REQUIRE OVER clause
    window_funcs_requiring_over = ["ROW_NUMBER", "RANK", "DENSE_RANK", "LAG", "LEAD",
                                    "FIRST_VALUE", "LAST_VALUE", "NTH_VALUE",
                                    "NTILE", "PERCENT_RANK", "CUME_DIST"]
    has_window_func = any(f"{wf}()" in s_upper or f"{wf} (" in s_upper for wf in window_funcs_requiring_over)

    if _has_top_level_group_by(s) and "OVER" in s_upper and "PARTITION" in s_upper and not has_window_func:
        # Only fix if OVER is used with aggregate functions (MIN, MAX, COUNT, SUM, AVG)
        # Pattern: AGG(...) OVER (PARTITION BY ...) [ORDER BY ...]
        # We want: AGG(...)
        # Use negated character class instead of .*? to avoid issues with nested parens
        window_agg_pattern = re.compile(
            r'\b(MIN|MAX|COUNT|SUM|AVG)\s*\(\s*([^()]*(?:\([^()]*\)[^()]*)*)\s*\)\s+OVER\s*\(\s*PARTITION\s+BY\s+[^)]+\)',
            re.IGNORECASE | re.DOTALL
        )

        def remove_over_clause(match):
            func_name = match.group(1)  # MIN, MAX, etc.
            func_args = match.group(2)  # column name(s)
            return f"{func_name}({func_args})"

        # Apply fix to the entire SQL
        fixed_sql = window_agg_pattern.sub(remove_over_clause, s)
        if fixed_sql != s:
            if DEBUG_WR:
                print("[PH3.5] Fixed window function + GROUP BY anti-pattern")
                print(f"[PH3.5] Before: {s[:200]}")
                print(f"[PH3.5] After:  {fixed_sql[:200]}")
            s = fixed_sql
    # ---------------------------------------------------------------------------
    # Phase 4 — Winner-Row & Aggregate Guardrails
    # Ensure semantic correctness for ranking-style queries
    # ---------------------------------------------------------------------------
    before_phase4 = s

    if DEBUG_WR: print("\n[REFINER - PH4] start ----")
    s = _fix_missing_space_before_clauses(s)
    # FIX: NO re.I flag - creditLimit won't match (lowercase 'imit'), but HAVING will match (uppercase)
    s = re.sub(
    r"(\b[A-Za-z_][A-Za-z0-9_]*)(WHERE|GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT)\b",
    r"\1 \2",
    s)
    s = _fix_missing_space_before_clauses(s)
    if DEBUG_WR: print(f"[REFINER - PH4] after re-spacing: {s[:200]}")

    # pre-sanity only (no rewrite)
    s = _winner_row_sanity_date_in_groupby(s, question)
    if DEBUG_WR: print("[REFINER - PH4] after sanity:", s)

    # remove MAX(..) HAVING self-subquery hacks
    s = _strip_having_max_self_subquery(s)
    if DEBUG_WR: print("[REFINER - PH4] after strip having max self subquery:", s)

    # NEW: Detect and fix malformed MAX/GROUP BY + ROW_NUMBER mixed pattern
    # This catches cases where 7B model incorrectly combines aggregate and window function patterns
    fixed_mixed = _fix_mixed_max_groupby_rownumber(s)
    if fixed_mixed:
        if DEBUG_WR: print("[REFINER - PH4] Fixed malformed MAX/GROUP BY + ROW_NUMBER pattern")
        s = fixed_mixed

    # NEW: Fix redundant GROUP BY + MAX(CASE WHEN rn = N...) when WHERE rn = N is present
    # When WHERE rn = N exists, each group has exactly ONE row, so GROUP BY is redundant
    fixed_redundant = _fix_redundant_groupby_with_rn_filter(s)
    if fixed_redundant:
        if DEBUG_WR: print("[REFINER - PH4] Fixed redundant GROUP BY with WHERE rn = N pattern")
        if DEBUG_WR: print(f"[REFINER - PH4] Before: {s[:200]}")
        if DEBUG_WR: print(f"[REFINER - PH4] After:  {fixed_redundant[:200]}")
        s = fixed_redundant

    # NEW: Fix malformed ordinal-ranking pattern where WHERE rn = N is placed
    # inside the same SELECT block that defines ROW_NUMBER()
    fixed_inner_rn = _fix_inner_rn_filter_scope(s)
    if fixed_inner_rn:
        if DEBUG_WR: print("[REFINER - PH4] Fixed inner WHERE rn = N scope")
        if DEBUG_WR: print(f"[REFINER - PH4] Before: {s[:200]}")
        if DEBUG_WR: print(f"[REFINER - PH4] After:  {fixed_inner_rn[:200]}")
        s = fixed_inner_rn

    # NEW: Fix inconsistent subquery alias hallucination
    # LLM sometimes uses different alias in outer SELECT than the subquery alias
    fixed_alias = _fix_inconsistent_subquery_alias(s)
    if fixed_alias:
        if DEBUG_WR: print("[REFINER - PH4] Fixed inconsistent subquery alias")
        if DEBUG_WR: print(f"[REFINER - PH4] Before: {s[:200]}")
        if DEBUG_WR: print(f"[REFINER - PH4] After:  {fixed_alias[:200]}")
        s = fixed_alias

    # -------------------------------------------------------------
    # CASE 1 — pattern clearly indicates MAX + date tie-breaker
    # → enforce ROW_NUMBER rewrite (FORCE MODE)
    # -------------------------------------------------------------
    if _looks_like_max_date_tiebreaker(s):
        # Jangan force rewrite kalau question JELAS bukan winner-row
        if _question_wants_winner_row(question):
            if DEBUG_WR: print("[REFINER - PH4] detected max-date tie-breaker + winner-row intent -> forcing guardrail")
            # CRITICAL: Check if query is already corrupted
            s_forced = _rewrite_winner_row_guardrail(s, question, force=True)
            forced_rewrite_ok = False
            if (
                s_forced
                and "ROW_NUMBER" in s_forced.upper()
                and not ("GROUP BY" in s_forced.upper() and "ROW_NUMBER" in s_forced.upper())
            ):
                try:
                    import sqlglot
                    sqlglot.parse_one(s_forced, read="mysql")
                    forced_rewrite_ok = True
                except Exception:
                    forced_rewrite_ok = False

            # SANITY: Only accept forced rewrite when structurally parseable.
            # Otherwise fall back to normal winner-row heuristic flow.
            if not forced_rewrite_ok:
                if DEBUG_WR: print("[REFINER - PH4] ABORT: forced rewrite malformed/unparseable, fallback to normal winner-row flow")
                s = before_phase4
                goto_phase5 = False
            else:
                s = s_forced
                if DEBUG_WR: print("[REFINER - PH4] FINAL after forced guardrail:", s)
                # Skip normal heuristic flow - forced rewrite is valid.
                s = _rewrite_having_alias_to_expr(s)
                s = _force_restore_groupby_if_aggregate_without_groupby(before_phase4, s)
                if DEBUG_WR: print("[REFINER - PH4] end ----", s, "\n")
                # DON'T return early - continue to Phase 5-7
                goto_phase5 = True
        else:
            if DEBUG_WR: print("[REFINER - PH4] max-date pattern detected BUT question wants true aggregation -> skip force")
            goto_phase5 = False
    else:
        goto_phase5 = False
    # ------------------------------------------------------------
    #  CASE 2 — normal heuristic winner-row handling
    # ------------------------------------------------------------
    if not goto_phase5:
        s = _rewrite_winner_row_guardrail(s, question)
        if DEBUG_WR: print("[REFINER - PH4] after guardrail:", s)

        s = _rewrite_winner_row_aggregate_misuse(s, question, ctx)
        if DEBUG_WR: print("[REFINER - PH4] after agg-misuse:", s)

        s = _rewrite_single_table_tuple_in_winner_row(s, question)
        if DEBUG_WR: print("[REFINER - PH4] after tuple-IN winner-row fix:", s)

        s = _fix_winner_row_window_order_priority(s, question)
        if DEBUG_WR: print("[REFINER - PH4] after winner-row window order fix:", s)

        s = _fix_winner_row_metric_order_priority(s, question)
        if DEBUG_WR: print("[REFINER - PH4] after winner-row metric priority fix:", s)

        s = _rewrite_explicit_rank_aggregate_window(s, question)
        if DEBUG_WR: print("[REFINER - PH4] after explicit-rank aggregate fix:", s)

        s = _rewrite_metric_aggregate_column_mismatch(s, question, ctx)
        if DEBUG_WR: print("[REFINER - PH4] after metric-agg-column fix:", s)

        s = _rewrite_missing_joined_agg_threshold_filter(s, question, ctx)
        if DEBUG_WR: print("[REFINER - PH4] after joined-agg threshold fix:", s)

        s = _rewrite_missing_overall_avg_having(s, question, ctx)
        if DEBUG_WR: print("[REFINER - PH4] after overall-avg baseline fix:", s)

        s = _rewrite_employee_latest_payment_customer_identity(s, question, ctx)
        if DEBUG_WR: print("[REFINER - PH4] after include-zero winner-row id-source fix:", s)

        s = _drop_tuple_in_filter_from_include_zero_ranked_winner(s, question, ctx)
        if DEBUG_WR: print("[REFINER - PH4] after include-zero tuple-IN drop fix:", s)

        s = _rewrite_invalid_metric_bridge_rollup(s, question, ctx)
        if DEBUG_WR: print("[REFINER - PH4] after metric-bridge path fix:", s)

        s = _rewrite_missing_metric_join_in_rollup(s, question, ctx)
        if DEBUG_WR: print("[REFINER - PH4] after hierarchical metric rollup fix:", s)

        s = _rewrite_dual_fact_rollup_fanout(s, question, ctx)
        if DEBUG_WR: print("[REFINER - PH4] after dual-fact fanout fix:", s)
        s = _sanitize_order_by_alias_assignment(s)
        if DEBUG_WR: print("[REFINER - PH4] after ORDER BY alias sanitize:", s)

        s = _rewrite_latest_year_same_year_aggregate(s, question, ctx)
        if DEBUG_WR: print("[REFINER - PH4] after latest-year-sum fix:", s)

        s = _rewrite_latest_year_correlated_maxdate_misuse(s, question, ctx)
        if DEBUG_WR: print("[REFINER - PH4] after latest-year correlated-max fix:", s)

        s = _rewrite_latest_year_derived_subquery_misuse(s, question, ctx)
        if DEBUG_WR: print("[REFINER - PH4] after latest-year derived-subquery fix:", s)

        s = _canonicalize_latest_year_same_year_pattern(s, question, ctx)
        if DEBUG_WR: print("[REFINER - PH4] after latest-year canonical pattern fix:", s)

        s = _rewrite_latest_year_bridge_tuple_runtime(s, question)
        if DEBUG_WR: print("[REFINER - PH4] after latest-year runtime tuple fix:", s)

        s = _rewrite_bridge_latest_year_tuple_filter(s, question, ctx)
        if DEBUG_WR: print("[REFINER - PH4] after latest-year bridge tuple fix:", s)

        s = _rewrite_latest_year_tuple_in_filter(s, question, ctx)
        if DEBUG_WR: print("[REFINER - PH4] after latest-year tuple-IN fix:", s)

        s = _move_latest_year_filter_where_to_inner_on(s)
        if DEBUG_WR: print("[REFINER - PH4] after latest-year WHERE->ON style fix:", s)

        s = _rewrite_latest_year_projection_to_year(s, question)
        if DEBUG_WR: print("[REFINER - PH4] after latest-year projection->year fix:", s)
        
        if not AGENT_AST_SEMANTIC_FIRST:
            s = _fix_aggregate_in_where(s)
            if DEBUG_WR: print(f"[REFINER - PH4] after agg-in-where fix: {s}")

            # Fix raw column WHERE -> HAVING for aggregate questions
            s = _fix_raw_column_where_to_having(s, question)
            if DEBUG_WR: print(f"[REFINER - PH4] after raw-where-to-having fix: {s}")
        elif DEBUG_WR:
            print("[REFINER - PH4] AST semantic-first enabled: skip regex agg/having surgery")

        # Add missing COUNT(*) for GROUP BY aggregates
        s = _add_missing_count_for_group_by(s, question)
        if DEBUG_WR: print(f"[REFINER - PH4] after add-count fix: {s}")

        s = _rewrite_row_level_nonnull_to_group_existence(s, question)
        if DEBUG_WR: print(f"[REFINER - PH4] after nonnull row-filter->group existence fix: {s}")

        s = _rewrite_role_label_count_semantics(s, question)
        if DEBUG_WR: print(f"[REFINER - PH4] after role-label count fix: {s}")

        s = _drop_unrequested_exists_filters(s, question, ctx)
        if DEBUG_WR: print(f"[REFINER - PH4] after unrequested EXISTS-drop fix: {s}")

        s = _drop_unrequested_temporal_filters(s, question, ctx)
        if DEBUG_WR: print(f"[REFINER - PH4] after unrequested temporal-filter drop fix: {s}")

        s = _rewrite_missing_temporal_bridge_exclusion(s, question, ctx)
        if DEBUG_WR: print(f"[REFINER - PH4] after temporal missing-bridge exclusion fix: {s}")

        s = _rewrite_temporal_year_exclusion_antijoin(s, question, ctx)
        if DEBUG_WR: print(f"[REFINER - PH4] after temporal exclusion anti-join fix: {s}")

        s = _prefer_not_exists_for_simple_exclusion(s, question, ctx)
        if DEBUG_WR: print(f"[REFINER - PH4] after NOT EXISTS anti-join preference: {s}")
        s = _repair_temporal_not_exists_scope(s, ctx)
        if DEBUG_WR: print(f"[REFINER - PH4] after temporal NOT EXISTS scope repair: {s}")

        s = _rewrite_grouped_winner_row_ranked_aggregate(s, question)
        if DEBUG_WR: print("[REFINER - PH4] after grouped winner-row fallback:", s)
        s = _fix_winner_row_window_order_priority(s, question)
        if DEBUG_WR: print("[REFINER - PH4] after post-fallback window order fix:", s)
        s = _fix_winner_row_metric_order_priority(s, question)
        if DEBUG_WR: print("[REFINER - PH4] after post-fallback metric priority fix:", s)
        s = _rewrite_having_alias_to_expr(s)
        s = _winner_row_final_sanity(s, question)

        # fail-safe restore group by if pipeline broke it
        s = _force_restore_groupby_if_aggregate_without_groupby(before_phase4, s)

        if DEBUG_WR: print("[REFINER - PH4] end ----", s, "\n")
    else:
            # Phase 4 was skipped due to goto_phase5, but still need aggregate fix
            if not AGENT_AST_SEMANTIC_FIRST:
                s = _fix_aggregate_in_where(s)
                if DEBUG_WR: print(f"[REFINER - PH4-FORCE] after agg-in-where fix: {s}")
                s = _fix_raw_column_where_to_having(s, question)
                if DEBUG_WR: print(f"[PH4-FORCE] after raw-where-to-having fix: {s}")
            elif DEBUG_WR:
                print("[PH4-FORCE] AST semantic-first enabled: skip regex agg/having surgery")
            s = _add_missing_count_for_group_by(s, question)
            if DEBUG_WR: print(f"[PH4-FORCE] after add-count fix: {s}")

            # Keep generic winner-row misuse repair active in forced path.
            # This avoids model-specific aggregate leakage before Agent3 checks.
            s = _rewrite_winner_row_aggregate_misuse(s, question, ctx)
            if DEBUG_WR: print(f"[PH4-FORCE] after agg-misuse fix: {s}")
            s = _rewrite_single_table_tuple_in_winner_row(s, question)
            if DEBUG_WR: print(f"[PH4-FORCE] after tuple-IN winner-row fix: {s}")
            s = _fix_winner_row_window_order_priority(s, question)
            if DEBUG_WR: print(f"[PH4-FORCE] after winner-row window order fix: {s}")

            s = _rewrite_missing_joined_agg_threshold_filter(s, question, ctx)
            if DEBUG_WR: print(f"[PH4-FORCE] after joined-agg threshold fix: {s}")

            s = _rewrite_missing_overall_avg_having(s, question, ctx)
            if DEBUG_WR: print(f"[PH4-FORCE] after overall-avg baseline fix: {s}")

            s = _rewrite_employee_latest_payment_customer_identity(s, question, ctx)
            if DEBUG_WR: print(f"[PH4-FORCE] after include-zero winner-row id-source fix: {s}")

            s = _rewrite_grouped_winner_row_ranked_aggregate(s, question)
            if DEBUG_WR: print(f"[PH4-FORCE] after grouped winner-row fallback: {s}")
            s = _fix_winner_row_window_order_priority(s, question)
            if DEBUG_WR: print(f"[PH4-FORCE] after post-fallback window order fix: {s}")
            s = _fix_winner_row_metric_order_priority(s, question)
            if DEBUG_WR: print(f"[PH4-FORCE] after post-fallback metric priority fix: {s}")
            s = _rewrite_having_alias_to_expr(s)
            s = _winner_row_final_sanity(s, question)
            s = _force_restore_groupby_if_aggregate_without_groupby(before_phase4, s)
    # ---------------------------------------------------------------------------
    # Phase 5 — Pattern, REGEXP, and Predicate Normalization
    # ---------------------------------------------------------------------------
    if (
        re.search(r"\b(?:regexp_like|regexp|rlike)\b", s, flags=re.I)
        or _question_requires_structural_regex(question)
        or _question_requires_word_boundary_regex(question)
    ):
        s = _guardrail_fix_like_regex_anchors(s, question)
        s = _canonicalize_boundary_regex(s, question)
        s = _canonicalize_structural_regex(s, question)
        s = _guardrail_fix_mysql_regexp_literals(s)
    # ---------------------------------------------------------------------------
    # Phase 6 — NULL / DISTINCT / Result Cardinality Policies
    # ---------------------------------------------------------------------------
    s = _expand_missing_null_to_null_or_empty(s, question)
    s = _ensure_reasonable_distinct(s, question)
    s = _ensure_reasonable_count_distinct(s, question)
    s = _drop_redundant_group_by_for_distinct_lookup(s)
    s = _rewrite_case_split_to_union(s, question)
    s = _normalize_simple_set_operation_branches(s, question)
    s = _rewrite_union_case_label_to_literal_branches(s, question)
    s = _rewrite_concat_label_union(s, question)
    s = _strip_unneeded_count_distinct(s, question)
    s = _drop_unrequested_group_key_filters(s, question, ctx)
    if DEBUG_WR: print(f"[REFINER - PH6] after unrequested group-key filter drop fix: {s}")
    # ---------------------------------------------------------------------------
    # Phase 6.5 — Strip Hallucinated WHERE on GROUP BY Column for "For Each" Queries
    # ---------------------------------------------------------------------------
    for_each_triggers = ["for each", "each ", "per ", "every "]
    if any(t in question.lower() for t in for_each_triggers):
        # Check if there's a GROUP BY clause
        group_match = re.search(r'\bGROUP\s+BY\s+([a-zA-Z0-9_`.]+)', s, re.IGNORECASE)
        if group_match:
            group_col = group_match.group(1).strip().strip('`').lower()
            # Pattern 1: WHERE col IN (...) - ONLY simple value lists, NOT subqueries
            # Use balanced parenthesis matching to detect subqueries
            where_in_match = None
            in_pattern = re.search(
                r'\bWHERE\s+' + re.escape(group_col) + r'\s+IN\s*\(',
                s, re.IGNORECASE
            )
            if in_pattern:
                # Check if the content inside parens is a simple list (no SELECT)
                start = in_pattern.end() - 1  # position of opening (
                depth = 0
                end = start
                for i, ch in enumerate(s[start:], start):
                    if ch == '(':
                        depth += 1
                    elif ch == ')':
                        depth -= 1
                        if depth == 0:
                            end = i
                            break
                inner_content = s[start+1:end]
                # Only treat as hallucinated filter if it's NOT a subquery
                if 'SELECT' not in inner_content.upper():
                    where_in_match = in_pattern
                    # Extend match to include the full IN clause
                    class _Match:
                        def __init__(self, start, end, string):
                            self._start, self._end, self._string = start, end, string
                        def start(self): return self._start
                        def end(self): return self._end
                    where_in_match = _Match(in_pattern.start(), end + 1, s[in_pattern.start():end+1])

            # Pattern 2: WHERE col IS NOT NULL - unnecessary null check on grouping col
            where_not_null_match = re.search(
                r'\bWHERE\s+' + re.escape(group_col) + r'\s+IS\s+NOT\s+NULL\b',
                s, re.IGNORECASE
            )
            where_match = where_in_match or where_not_null_match
            if where_match:
                # Remove the WHERE clause entirely - "for each" means ALL values
                s = s[:where_match.start()] + s[where_match.end():]
                s = re.sub(r'\s{2,}', ' ', s)  # collapse extra spaces
                if DEBUG_WR:
                    print(f"[PH6.5] Removed hallucinated WHERE on GROUP BY column '{group_col}' for 'for each' query")

    # ---------------------------------------------------------------------------
    # Phase 6.6 — Fix Redundant Subquery Anti-Pattern for GROUP BY + HAVING
    # Pattern: WHERE col IN (SELECT col FROM same_table GROUP BY col HAVING AGG(...) > N)
    # Fix: Remove the WHERE subquery, move HAVING to outer query
    # ---------------------------------------------------------------------------
    # Detect the anti-pattern:
    # SELECT col, AGG(...) FROM table WHERE col IN (SELECT col FROM table GROUP BY col HAVING AGG(...) op N) GROUP BY col
    having_subquery_pattern = re.compile(
        r'\bWHERE\s+(\w+)\s+IN\s*\(\s*SELECT\s+(?:DISTINCT\s+)?\1\s+FROM\s+(\w+)\s+GROUP\s+BY\s+\1\s+HAVING\s+(.+?)\s*\)\s+GROUP\s+BY\s+\1\b',
        re.IGNORECASE
    )
    having_match = having_subquery_pattern.search(s)
    if having_match:
        outer_col = having_match.group(1)
        inner_table = having_match.group(2)
        having_clause = having_match.group(3).strip()

        # Check if rewrite is safe:
        # 1) Original same-table anti-pattern
        # 2) Inner subquery references a column that does NOT exist in inner table
        #    (hallucinated cross-table subquery, e.g. orderdetails.productLine)
        outer_from_match = re.search(r'\bFROM\s+(\w+)\b', s, re.IGNORECASE)
        allowed_identifiers = _parse_allowed_identifiers(ctx)
        inner_cols = {
            str(c).strip("`").lower()
            for c in allowed_identifiers.get(inner_table, set())
        }

        should_rewrite = False
        if outer_from_match:
            outer_table = outer_from_match.group(1).lower()
            if outer_table == inner_table.lower():
                should_rewrite = True
            elif outer_col.lower() not in inner_cols:
                should_rewrite = True

        if should_rewrite:
            # This is the anti-pattern - convert to simple GROUP BY + HAVING
            # Remove the WHERE ... IN (...) subquery and add HAVING to outer
            s = s[:having_match.start()] + 'GROUP BY ' + outer_col + ' HAVING ' + having_clause + s[having_match.end():]
            s = re.sub(r'\s{2,}', ' ', s)
            if DEBUG_WR:
                print(f"[PH6.6] Fixed redundant subquery anti-pattern: moved HAVING to outer query")

    # ---------------------------------------------------------------------------
    # Phase 7 — Ordering, Limits, and Output Formatting
    # Cosmetic and execution-safety adjustments only
    # ---------------------------------------------------------------------------
    s = _simplify_default_window_frame(s, question)
    s = _rewrite_single_table_case_band_derived_subquery(s, question)
    s = _rewrite_correlated_offset_subquery_to_window(s, question)
    s = _rewrite_same_table_group_count_self_join_to_window(s, question)
    s = _rewrite_running_self_join_aggregate_to_window(s, question)
    s = _enforce_window_sequence_direction(s, question)
    s = _repair_offset_window_projection_shape(s, question)
    s = _preserve_requested_base_columns_in_running_self_join(s, question)
    s = _enforce_requested_order_direction(s, question)
    s = _remove_top_level_order_by_if_not_requested(s, question)
    s = _maybe_add_limit(question, s)    
    # s = _collapse_spaces_outside_quotes(s)
    # ---------------------------------------------------------------------------
    # Phase 8 — Remove Cosmetic Aliases (FIX UTAMA)
    # ---------------------------------------------------------------------------
    # Masalah: LLM 7B sering menambahkan alias kosmetik: extension AS 'internal phone'.
    # Solusi: Hapus alias JIKA kolomnya hanya terdiri dari huruf/angka/titik/underscore.
    # Logic: 
    # - Match: "extension AS 'phone'" atau "t1.col AS alias"
    # - Ignore: "SUM(amount) AS total" (Karena regex [a-z0-9_.] tidak mencakup tanda kurung ')' )
    
    # Regex:
    # 1. \b([a-zA-Z0-9_.]+)  -> Group 1: Nama kolom (tanpa kurung/aritmatika)
    # 2. \s+AS\s+            -> Keyword AS
    # 3. (['"][^'"]+['"]|...) -> Group 2: Alias (string quote atau identifier biasa)
    
    select_match = re.search(r'\bSELECT\b(.*?)\bFROM\b', s, re.IGNORECASE | re.DOTALL)
    if select_match:
        select_clause = select_match.group(1)
        before_select = s[:select_match.start(1)]
        after_select = s[select_match.end(1):]
        
        # Pattern: Hanya match string aliases yang quoted (bukan identifier aliases)
        # Match: col AS 'alias' atau table.col AS "alias"
        # Tidak match: customers AS c (identifier alias untuk table)
        pattern_alias = re.compile(
            r"\b([a-zA-Z0-9_`.]+)\s+AS\s+(['\"`][^'\"`]+['\"`])",
            re.IGNORECASE
        )
        
        # Apply replacement hanya di SELECT clause (max 3 iterations untuk safety)
        for _ in range(3):
            def replacer(match):
                col_name = match.group(1)
                # Keep CASE...END structure intact
                if col_name.upper() in ('END', 'CASE', 'WHEN', 'THEN', 'ELSE'):
                    return match.group(0)  # Pertahankan as-is
                # Remove cosmetic string alias
                return col_name
            
            new_select = pattern_alias.sub(replacer, select_clause)
            if new_select == select_clause:
                break
            select_clause = new_select
        
        # Reconstruct query
        s = before_select + select_clause + after_select
    # ---------------------------------------------------------------------------
    # Phase 8a — Strip plain identifier aliases for simple columns in ROW_NUMBER queries
    # Example: pmt.amount AS secondHighestPayment -> pmt.amount
    # Only for queries with ROW_NUMBER subquery pattern (ordinal ranking)
    # ---------------------------------------------------------------------------
    if 'ROW_NUMBER' in s.upper() and re.search(r'\)\s*\w+\s+WHERE\s+\w+\.rn\s*=', s, re.IGNORECASE):
        select_match = re.search(r'\bSELECT\b(.*?)\bFROM\s*\(', s, re.IGNORECASE | re.DOTALL)
        if select_match:
            select_clause = select_match.group(1)
            before_select = s[:select_match.start(1)]
            after_select = s[select_match.end(1):]
            
            # Pattern: alias.column AS identifier (plain column with alias)
            # Match: pmt.amount AS secondHighestPayment
            # Skip: COUNT(*) AS total (has parentheses)
            pattern_plain_alias = re.compile(
                r'\b([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*)\s+AS\s+([a-zA-Z_][a-zA-Z0-9_]*)\b',
                re.IGNORECASE
            )
            
            def strip_plain_alias(match):
                col_ref = match.group(1)  # e.g., pmt.amount
                return col_ref  # Remove the AS alias part
            
            new_select = pattern_plain_alias.sub(strip_plain_alias, select_clause)
            if new_select != select_clause:
                if DEBUG_WR:
                    print(f"[PH8a] Stripped plain alias in ROW_NUMBER query")
                s = before_select + new_select + after_select
    # ---------------------------------------------------------------------------
    select_match = re.search(r'\bSELECT\b(.*?)\bFROM\b', s, re.IGNORECASE | re.DOTALL)
    if select_match:
        # ... existing Phase 8 code ...
        pass
    
    # Phase 8b is DISABLED for thesis benchmark

    # ---------------------------------------------------------------------------
    # Phase 8c — REMOVED (let LLM decide aliases)
    # ---------------------------------------------------------------------------
    # Previously standardized table aliases to match _ALIAS_OVERRIDES.
    # Now we let the LLM decide its own aliases without hardcoded overrides.

    # ---------------------------------------------------------------------------
    # Phase 9 — Pre-Execution Safety Pass (Final Guardrail)
    # Catch and repair remaining high-risk anti-patterns before execution.
    # ---------------------------------------------------------------------------
    s = _preexecute_sql_safety_pass(s, question)
    if DEBUG_WR: print("[REFINER - PH9] after pre-exec safety:", s)

    # ---------------------------------------------------------------------------
    # Final Polish
    # ---------------------------------------------------------------------------
    s = _collapse_spaces_outside_quotes(s)
    return s


def _rewrite_latest_year_bridge_tuple_runtime(sql: str, question: str) -> str:
    """
    Narrow runtime fallback for bridge-backed latest-year tuple filters.
    Designed to touch only flat grouped latest-year aggregates.
    """
    if not sql or not _question_wants_latest_year_scope_with_aggregate(question):
        return sql

    s0 = (sql or "").strip().rstrip(";").strip()
    if " in (select max(year(" not in s0.lower():
        return sql

    def _find_kw(text: str, keyword: str, start: int = 0) -> int:
        w = keyword.lower()
        n = len(w)
        depth = 0
        in_sq = in_dq = in_bt = False
        i = 0
        while i < len(text):
            ch = text[i]
            if ch == "'" and not in_dq and not in_bt:
                if in_sq and i + 1 < len(text) and text[i + 1] == "'":
                    i += 2
                    continue
                in_sq = not in_sq
                i += 1
                continue
            if ch == '"' and not in_sq and not in_bt:
                in_dq = not in_dq
                i += 1
                continue
            if ch == "`" and not in_sq and not in_dq:
                in_bt = not in_bt
                i += 1
                continue
            if in_sq or in_dq or in_bt:
                i += 1
                continue
            if ch == "(":
                depth += 1
                i += 1
                continue
            if ch == ")":
                depth = max(0, depth - 1)
                i += 1
                continue
            if depth == 0 and i >= start and i + n <= len(text):
                left_ok = i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")
                right_ok = (i + n) >= len(text) or not (text[i + n].isalnum() or text[i + n] == "_")
                if left_ok and right_ok and text[i:i + n].lower() == w:
                    return i
            i += 1
        return -1

    sel = _find_kw(s0, "select", 0)
    frm = _find_kw(s0, "from", sel + 6)
    grp = _find_kw(s0, "group by", frm + 4)
    if sel < 0 or frm < 0 or grp < 0:
        return sql
    ordp = _find_kw(s0, "order by", grp + 8)
    limitp = _find_kw(s0, "limit", grp + 8)
    tail_pos = min([p for p in [ordp, limitp] if p >= 0], default=len(s0))

    select_part = s0[sel + 6:frm].strip()
    from_part = s0[frm + 4:grp].strip()
    group_part = s0[grp + 8:tail_pos].strip()
    tail_part = s0[tail_pos:].strip() if tail_pos < len(s0) else ""

    where_pos = _find_kw(from_part, "where", 0)
    if where_pos < 0:
        return sql
    from_core = from_part[:where_pos].strip()
    where_body = re.sub(r"^\s*where\s+", "", from_part[where_pos:].strip(), flags=re.I).strip()
    if not from_core or not where_body or _has_top_level_or(where_body):
        return sql

    alias_map = _parse_from_join_aliases(f"SELECT 1 FROM {from_core}")
    if not alias_map:
        return sql
    conjuncts = _split_top_level_and(where_body)
    if not conjuncts:
        return sql

    eq_pairs = list(
        re.finditer(
            r"""`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\.\s*`?([A-Za-z_][A-Za-z0-9_]*)`?\s*=\s*`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\.\s*`?([A-Za-z_][A-Za-z0-9_]*)`?""",
            from_core,
            flags=re.I,
        )
    )

    tuple_pat = re.compile(
        r"""(?is)^\s*\(\s*YEAR\s*\(\s*`?(?P<outer_date_alias>[A-Za-z_][A-Za-z0-9_]*)`?\s*\.\s*`?(?P<outer_date_col>[A-Za-z_][A-Za-z0-9_]*)`?\s*\)\s*,\s*`?(?P<outer_entity_alias>[A-Za-z_][A-Za-z0-9_]*)`?\s*\.\s*`?(?P<outer_entity_col>[A-Za-z_][A-Za-z0-9_]*)`?\s*\)\s+IN\s*\(\s*SELECT\s+(?:MAX\s*\(\s*YEAR\s*\(\s*(?:`?[A-Za-z_][A-Za-z0-9_]*`?\s*\.\s*)?`?(?P<sub_date_col_1>[A-Za-z_][A-Za-z0-9_]*)`?\s*\)\s*\)|YEAR\s*\(\s*MAX\s*\(\s*(?:`?[A-Za-z_][A-Za-z0-9_]*`?\s*\.\s*)?`?(?P<sub_date_col_2>[A-Za-z_][A-Za-z0-9_]*)`?\s*\)\s*\))\s*,\s*(?:`?[A-Za-z_][A-Za-z0-9_]*`?\s*\.\s*)?`?(?P<sub_key_col>[A-Za-z_][A-Za-z0-9_]*)`?\s+FROM\s+`?(?P<sub_date_table>[A-Za-z_][A-Za-z0-9_]*)`?(?:\s+(?:AS\s+)?`?[A-Za-z_][A-Za-z0-9_]*`?)?\s+GROUP\s+BY\s+(?:`?[A-Za-z_][A-Za-z0-9_]*`?\s*\.\s*)?`?(?P<group_col>[A-Za-z_][A-Za-z0-9_]*)`?\s*\)\s*$""",
    )

    for idx, cond in enumerate(conjuncts):
        m = tuple_pat.match((cond or "").strip())
        if not m:
            continue

        outer_date_alias = m.group("outer_date_alias")
        outer_date_col = m.group("outer_date_col")
        outer_entity_alias = m.group("outer_entity_alias")
        outer_entity_col = m.group("outer_entity_col")

        bridge_matches: List[Tuple[str, str, str]] = []
        for m_eq_date in eq_pairs:
            a1, c1, a2, c2 = m_eq_date.group(1), m_eq_date.group(2), m_eq_date.group(3), m_eq_date.group(4)
            a1l, a2l = a1.lower(), a2.lower()
            if a1l == outer_date_alias.lower() and a2l in alias_map and a2l != outer_date_alias.lower():
                bridge_alias, date_join_col, bridge_fk_col = a2, c1, c2
            elif a2l == outer_date_alias.lower() and a1l in alias_map and a1l != outer_date_alias.lower():
                bridge_alias, date_join_col, bridge_fk_col = a1, c2, c1
            else:
                continue

            entity_join_ok = False
            for m_eq_entity in eq_pairs:
                b1, d1, b2, d2 = m_eq_entity.group(1), m_eq_entity.group(2), m_eq_entity.group(3), m_eq_entity.group(4)
                b1l, b2l = b1.lower(), b2.lower()
                if b1l == bridge_alias.lower() and b2l == outer_entity_alias.lower() and d1.lower() == outer_entity_col.lower() and d2.lower() == outer_entity_col.lower():
                    entity_join_ok = True
                    break
                if b2l == bridge_alias.lower() and b1l == outer_entity_alias.lower() and d2.lower() == outer_entity_col.lower() and d1.lower() == outer_entity_col.lower():
                    entity_join_ok = True
                    break
            if entity_join_ok:
                bridge_matches.append((bridge_alias, date_join_col, bridge_fk_col))

        if len(bridge_matches) != 1:
            continue

        bridge_alias, date_join_col, bridge_fk_col = bridge_matches[0]
        bridge_table = alias_map.get(bridge_alias) or alias_map.get(bridge_alias.lower()) or bridge_alias
        outer_date_table = alias_map.get(outer_date_alias) or alias_map.get(outer_date_alias.lower()) or outer_date_alias

        used_aliases = {a.lower() for a in alias_map.keys() if a}
        used_aliases.update(str(t or "").replace("`", "").split(".")[-1].strip().lower() for t in alias_map.values())

        def _pick_alias(base: str) -> str:
            seed = re.sub(r"[^A-Za-z0-9_]+", "_", (base or "")).strip("_") or "t"
            if not re.match(r"[A-Za-z_]", seed[0]):
                seed = f"t_{seed}"
            cand = seed
            n = 2
            while cand.lower() in used_aliases:
                cand = f"{seed}_{n}"
                n += 1
            used_aliases.add(cand.lower())
            return cand

        ly_alias = _pick_alias("latest_years")
        src_alias = _pick_alias("ly_src")
        dt_alias = _pick_alias("ly_dt")

        def _replace_year(text: str) -> str:
            return re.sub(
                rf"(?is)YEAR\s*\(\s*`?{re.escape(outer_date_alias)}`?\s*\.\s*`?{re.escape(outer_date_col)}`?\s*\)",
                f"{ly_alias}.latest_year",
                text,
            )

        select_rebuilt = _replace_year(select_part)
        group_items: List[str] = []
        seen_group: Set[str] = set()
        for item in _split_top_level_commas(group_part):
            replaced = _replace_year((item or "").strip()).strip()
            if not replaced:
                continue
            key = replaced.lower()
            if key in seen_group:
                continue
            seen_group.add(key)
            group_items.append(replaced)
        if not group_items:
            continue

        ly_join = (
            "JOIN (\n"
            "  SELECT\n"
            f"    {src_alias}.{outer_entity_col} AS latest_key,\n"
            f"    MAX(YEAR({dt_alias}.{outer_date_col})) AS latest_year\n"
            f"  FROM {bridge_table} AS {src_alias}\n"
            f"  JOIN {outer_date_table} AS {dt_alias}\n"
            f"    ON {dt_alias}.{date_join_col} = {src_alias}.{bridge_fk_col}\n"
            f"  GROUP BY {src_alias}.{outer_entity_col}\n"
            f") AS {ly_alias}\n"
            f"  ON {ly_alias}.latest_key = {outer_entity_alias}.{outer_entity_col}"
        )

        remaining = [c.strip() for j, c in enumerate(conjuncts) if j != idx and c.strip()]
        remaining.append(f"YEAR({outer_date_alias}.{outer_date_col}) = {ly_alias}.latest_year")
        tail_clause = f"\n{tail_part}" if tail_part else ""
        return (
            f"SELECT {select_rebuilt}\n"
            f"FROM {from_core}\n{ly_join}\n"
            f"WHERE {' AND '.join(remaining)}\n"
            f"GROUP BY {', '.join(group_items)}"
            f"{tail_clause};"
        )

    return sql


def normalize_sql_pipeline(raw_sql: str, question: str, ctx: str, ranked_cols=None) -> str:
    """
    Pipeline standar untuk normalisasi SQL:
    - postprocess (normalize keyword, limit, distinct, dsb)
    - repair alias berdasarkan alias_map
    - postprocess lagi setelah alias diganti
    """
    _dbg_print_two("[REFINER - NORMALIZE] input sql:", raw_sql)
    if not raw_sql:
        return raw_sql

    rewrite_budget = max(1, int(AGENT_NORMALIZE_REWRITE_BUDGET))
    rewrite_count = 0

    def _semantic_fingerprint(text: str) -> str:
        t = (text or "").strip().rstrip(";")
        t = _collapse_spaces_outside_quotes(t)
        t = re.sub(r"\s+", " ", t).strip().lower()
        return t

    def _collect_hard_identifier_errors(errors: List[str]) -> List[str]:
        return [
            e for e in (errors or [])
            if "not in ALLOWED" in e or "invalid column" in e or "invalid columns" in e
        ]

    def _apply_postprocess(sql_in: str) -> str:
        nonlocal rewrite_count
        before_fp = _semantic_fingerprint(sql_in)
        out = _postprocess_sql(sql_in, question, ctx)
        after_fp = _semantic_fingerprint(out)
        if before_fp != after_fp:
            rewrite_count += 1
        return out

    def _ast_stabilize(sql_in: str) -> tuple[str, Dict[str, Any]]:
        if not AGENT_AST_GUARDS_ENABLED:
            return sql_in, {}
        ast_sql, ast_meta = _ast_guardrails(sql_in, ctx=ctx, question=question)
        if ast_meta.get("parse_ok", False) and ast_meta.get("fixes"):
            sql_in = ast_sql
        if AGENT_AST_GUARDS_DEBUG and ast_meta.get("warnings"):
            _dbg_print_two("[REFINER - AST] warnings", f"{ast_meta.get('warnings')}")
        return sql_in, ast_meta

    s = _apply_postprocess(raw_sql)

    if ranked_cols:
        pref = pick_preferred_anchor_from_ranked_cols(ranked_cols, top_n=12)
        s = normalize_anchor_for_simple_inner_join(s, pref)

    s = _repair_with_aliases(s, ctx)
    s = _repair_backticked_identifier_phrases(s, ctx)
    s, ast_meta = _ast_stabilize(s)

    parse_ok = bool(ast_meta.get("parse_ok", False)) if AGENT_AST_GUARDS_ENABLED else True
    hard_identifier_errors = _collect_hard_identifier_errors(validate_identifiers(s, ctx))

    should_early_stop = (
        AGENT_NORMALIZE_EARLY_STOP
        and parse_ok
        and not hard_identifier_errors
        and (rewrite_count >= rewrite_budget or rewrite_budget <= 1)
    )

    if should_early_stop:
        s = _collapse_spaces_outside_quotes(s)
        _dbg_print_two("[REFINER - NORMALIZE] output sql:", s)
        return s

    needs_second_pass = (
        rewrite_count < rewrite_budget
        or not parse_ok
        or bool(hard_identifier_errors)
    )
    if needs_second_pass:
        s = _apply_postprocess(s)
        s, _ = _ast_stabilize(s)

    _dbg_print_two("[REFINER - NORMALIZE] output sql:", s)
    return s
