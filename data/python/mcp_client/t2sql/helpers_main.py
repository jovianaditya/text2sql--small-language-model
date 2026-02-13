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
from typing import TYPE_CHECKING, Dict, List, Optional, Set
from collections import defaultdict
# ======================================================================
# ======================== LOCAL MODULE IMPORT =========================
# ======================================================================
if TYPE_CHECKING:
    from .state import RAGState
from .utils import _dbg_print_two
from .schema_analyzer import (
    get_transaction_tables,
    get_anchor_columns,
    get_all_tables,
    get_fk_relationships,
    get_id_column,
    get_name_column,
    get_all_mentioned_tables,
    get_singular_form,
)
from .helpers_core import (
    _parse_allowed_identifiers, _question_requires_word_boundary_regex, _question_requires_regex, has_join,
    _question_requires_inner_join,
    _SQL_KEYWORDS,  # SQL keywords constant
    # phase 0
    _fix_bad_backticks, _canonicalize_table_names, _escape_reserved_column_aliases,
    # phase 1
    _move_rhs_filter_from_where_to_on,  # Only triggers for "ALL entities" questions
    _rewrite_left_join_to_inner_if_not_needed, _rewrite_left_join_to_inner_if_rhs_filtered,
    _rewrite_inner_join_filter_to_exists, _rewrite_inner_join_existence_to_exists,
    _remove_redundant_and_ornamental_joins, has_non_inner_join, join_count,
    # phase 2
    _drop_redundant_column_aliases, _drop_unused_from_table_aliases, _strip_alias_if_single_table,
    # phase 3
    _wants_customer_name_only, _wants_product_name_only, _prefer_friendly_names, _wants_name_only,
    _expand_group_by_aliases_regex, _is_potential_winner_row, _enforce_explicit_group_by,
    _remove_invalid_select_columns_with_group_by, _fix_order_by_group_by_consistency,
    _normalize_alias_lengths,
    # phase 4
    _winner_row_sanity_date_in_groupby, _strip_having_max_self_subquery, _fix_mixed_max_groupby_rownumber, _fix_redundant_groupby_with_rn_filter, _fix_inconsistent_subquery_alias, _looks_like_max_date_tiebreaker,
    _question_wants_winner_row, _rewrite_winner_row_guardrail, _rewrite_having_alias_to_expr,
    _force_restore_groupby_if_aggregate_without_groupby,
    _rewrite_winner_row_aggregate_misuse, _winner_row_final_sanity,
    # phase 4b - Aggregate in WHERE fix
    _fix_aggregate_in_where,
    _fix_raw_column_where_to_having,
    _add_missing_count_for_group_by,
    # phase 5 — Pattern, REGEXP, and Predicate Normalization
    _guardrail_fix_like_regex_anchors, _rewrite_simple_regexp_to_like, _guardrail_fix_mysql_regexp_literals,
    # phase 6
    _expand_missing_null_to_null_or_empty, _ensure_reasonable_distinct, _strip_unneeded_count_distinct, _parse_from_join_aliases,
    # phase 7
    _remove_top_level_order_by_if_not_requested, _maybe_add_limit, _collapse_spaces_outside_quotes
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
        for m in re.finditer(r"\bas\s+([A-Za-z_]\w*)\b", select_part, flags=re.I):
            exclude.add(m.group(1).lower())

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

    # --------------------------------------------------
    # Alias -> table mapping dari FROM / JOIN
    # --------------------------------------------------
    alias2table = _parse_from_join_aliases(sql)  # alias -> table (canonical)
    allowed_lower = {t.lower(): t for t in allowed.keys()}

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

        bad = [c for c in cols if c not in allowed[tbl]]
        if bad:
            errs.append(f"{tbl} -> invalid columns: {', '.join(bad)}")

    # --------------------------------------------------
    # Validate unqualified columns (single-table only)
    # --------------------------------------------------
    if single_table:
        allowed_cols = allowed.get(single_table, set())
        for col in used.get(None, set()):
            if col not in allowed_cols:
                errs.append(
                    f"{single_table} -> invalid column (unqualified): {col}"
                )

    return errs

def classify_query(question: str) -> str:
    q = " " + (question or "").strip().lower() + " "
    # ---------------------------------------------------------------------------
    # --- 0. DETEKSI ORDINAL RANKING (always "rank") ---
    # "second highest", "third largest", etc. ALWAYS need ROW_NUMBER subquery
    # ---------------------------------------------------------------------------
    ordinal_pattern = r"\b(second|third|fourth|fifth|2nd|3rd|4th|5th|kedua|ketiga|keempat|kelima)\s+(highest|lowest|largest|smallest|biggest|top|bottom|best|worst|most|least|terbesar|terkecil|tertinggi|terendah)\b"
    if re.search(ordinal_pattern, q):
        return "rank"
    # ---------------------------------------------------------------------------
    # --- 1. DETEKSI RANK (The "Winner-Row" Pattern) ---
    # Syarat: Mencari 'baris data' tunggal dalam sebuah kelompok.
    # Pattern: [Group Context] + [Superlative] + [Object, not count]
    # ---------------------------------------------------------------------------
    has_group = re.search(r"\b(per|each|every|within|partition)\b", q)
    has_superlative = re.search(r"\b(largest|highest|latest|best|most|max|min|top)\b", q)
    # Mencari objek (record/payment), bukan mencari jumlah angka (total/count)
    asks_for_object = re.search(r"\b(record|entry|row|payment|order|details|info)\b", q)
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

def _regex_requirement_ok(sql: str, requires_regex: bool, requires_boundary: bool) -> bool:
    if not (requires_regex or requires_boundary):
        return True

    low = (sql or "").lower()
    has_regexp = ("regexp" in low) or ("rlike" in low)
    if not has_regexp:
        return False

    if requires_boundary:
        return "(^|[^[:alnum:]_])" in (sql or "")

    return ("^" in (sql or "")) and ("$" in (sql or ""))
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
    Bangun dense probe untuk embedding query, tapi:
    - Tail dibuat SESINGKAT mungkin.
    - Token ekstra hanya yang benar-benar relevan dengan pertanyaan.
    - Tidak lagi hardcode list panjang 'id date status sum total amount revenue ...'.
    """
    q_raw = question or ""
    q = q_raw.lower()
    toks_q = q.split()
    mentioned_tables_dynamic = get_all_mentioned_tables(q_raw)

    # fitur dasar
    has_metric_word = any(k in q for k in [
    "amount", "revenue", "sales", "omset", "price", "harga",
    "gross", "net"
    ])
    has_count_word = any(k in q for k in [
        "count", "berapa banyak", "how many"
    ])
    has_time_word = any(k in q for k in [
        "today", "yesterday", "last", "this", "bulan", "minggu", "tahun",
        "quarter", "month", "week", "year", "tanggal", "hari",
        "mingguan", "bulanan", "202", "201"
    ])

    # entity utama classicmodels
    mentions_customer = any(k in q for k in ["customer", "customers", "pelanggan"]) or ("customers" in mentioned_tables_dynamic)
    mentions_product  = any(k in q for k in ["product", "products", "produk"]) or ("products" in mentioned_tables_dynamic)
    mentions_order    = any(k in q for k in ["order", "orders", "pesanan"]) or ("orders" in mentioned_tables_dynamic)
    mentions_employee = any(k in q for k in ["employee", "employees", "karyawan"]) or ("employees" in mentioned_tables_dynamic)

    # entity tambahan (tabel lain)
    mentions_productline = any(
        k in q
        for k in [
            "product line", "product lines",
            "lini produk", "line produk",
            "kategori", "category"
        ]
    ) or ("productlines" in mentioned_tables_dynamic)
    mentions_office = any(
        k in q
        for k in [
            "office", "offices", "kantor",
            "territory", "territories", "region", "regional"
        ]
    ) or ("offices" in mentioned_tables_dynamic)
    mentions_payment = any(
        k in q
        for k in [
            "payment", "payments", "pembayaran",
            "check ", "cheque", "cheques"
        ]
    ) or ("payments" in mentioned_tables_dynamic)
    mentions_orderdetails = any(
        k in q
        for k in [
            "order detail", "order details",
            "line item", "line items",
            "item pesanan", "detail pesanan"
        ]
    ) or ("orderdetails" in mentioned_tables_dynamic)

    id_like = _mentions_id_like_for_probe(q)
    # ---------------------------------------------------------------------------
    # 1) Tentukan base tokens
    # ---------------------------------------------------------------------------
    base: List[str] = []
    # a) Untuk lookup murni → jangan banyak tambahan
    if qtype == "lookup":
        # Biasanya cukup entity hints saja (di bawah)
        base = []
    # b) Untuk agregasi → tambahkan sedikit metric/time hints
    elif qtype in {"aggregate_by", "aggregate_all"}:
        if has_metric_word:
            base += ["amount", "revenue", "price"]
        if has_count_word:
            base += ["count"]
        if has_time_word:
            base += ["date", "year", "month"]
    else:
        # qtype other: minimal hints
        if has_time_word:
            base += ["date"]
    # ---------------------------------------------------------------------------
    # 2) Entity-specific hints (now using dynamic column inference)
    # ---------------------------------------------------------------------------
    if mentions_customer:
        base += ["customer"]
        if not id_like:
            name_col = get_name_column("customers")
            if name_col:
                base.append(name_col)
        else:
            id_col = get_id_column("customers")
            if id_col:
                base.append(id_col)

    if mentions_product:
        base += ["product"]
        if not id_like:
            name_col = get_name_column("products")
            if name_col:
                base.append(name_col)
        else:
            id_col = get_id_column("products")
            if id_col:
                base.append(id_col)

    if mentions_order:
        base += ["order"]
        id_col = get_id_column("orders")
        if id_col:
            base.append(id_col)

    if mentions_employee:
        base += ["employee"]
        id_col = get_id_column("employees")
        if id_col:
            base.append(id_col)

    if mentions_productline:
        id_col = get_id_column("productlines")
        if id_col:
            base.append(id_col)

    if mentions_office:
        id_col = get_id_column("offices")
        if id_col:
            base.append(id_col)
        base += ["city", "country", "territory"]

    if mentions_payment:
        base += ["payments", "amount", "paymentDate"]

    if mentions_orderdetails:
        base += ["orderdetails", "quantityOrdered", "priceEach"]
    # Generic table coverage for non-classicmodels schemas:
    # add lightweight hints for any other mentioned table.
    static_tables = {
        "customers", "products", "orders", "employees",
        "productlines", "offices", "payments", "orderdetails"
    }
    for table in sorted(mentioned_tables_dynamic):
        if table in static_tables:
            continue
        singular = get_singular_form(table)
        if singular:
            base.append(singular)
        id_col = get_id_column(table)
        if id_col:
            base.append(id_col)
        if not id_like:
            name_col = get_name_column(table)
            if name_col:
                base.append(name_col)
    # ---------------------------------------------------------------------------
    # 2b) Geo hints (lebih dinamis)
    # ---------------------------------------------------------------------------
    mentions_country_word = any(k in q for k in ["country", "negara"])
    mentions_city_word    = any(k in q for k in ["city", "kota"])
    mentions_region_word  = any(k in q for k in ["region", "wilayah", "territory", "territories"])
    geo_tokens: List[str] = []

    # CASE A: query sebut "country" tapi tidak sebut "city" → fokus ke country saja
    if mentions_country_word and not mentions_city_word:
        geo_tokens.append("country")

    # CASE B: query sebut "city" tapi tidak sebut "country" → fokus ke city saja
    elif mentions_city_word and not mentions_country_word:
        geo_tokens.append("city")

    # CASE C: sebut dua-duanya → boleh tambahin dua-duanya
    elif mentions_country_word and mentions_city_word:
        geo_tokens += ["country", "city"]

    # CASE D: nggak sebut city/country, tapi konteks-nya offices / region
    elif mentions_office or mentions_region_word:
        # offices biasanya lebih nyambung ke city, tapi country juga masih relevan
        geo_tokens += ["city", "country"]

    # CASE E: query tentang customers + ada pola "from <negara>"
    # misal: "customers from Japan", "pelanggan dari USA"
    if mentions_customer:
        if re.search(r"\bfrom\s+\w+\b", q_raw, flags=re.I):
            if "country" not in geo_tokens:
                geo_tokens.append("country")

    # Tambahkan ke base, tapi tetap hormati max_tokens di bawah
    base += geo_tokens

    # Name-specific hints (generic)
    if any(k in q for k in ["name", "nama"]):
        base += ["name"]

    # Super pendek → beri sedikit “bumbu” supaya embedding nggak terlalu kosong
    if len(toks_q) <= 2 and not base:
        base += ["amount", "date"]
    # ---------------------------------------------------------------------------
    # 3) Dedup + truncate
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

def _is_productline_lookup(question: str) -> bool:
    """
    True kalau user lagi minta daftar product lines (bukan agregat).
    Contoh:
      - "show all product lines and their descriptions"
      - "list all product lines"
    Dipakai bareng qtype == 'lookup' supaya nggak mengganggu query agregasi.
    """
    q = (question or "").lower()

    # fokus ke frasa product line(s)
    if "product line" not in q and "product lines" not in q and "lini produk" not in q and "line produk" not in q:
        return False

    # kalau ada kata-kata agregasi → biar classifier saja yang urus, jangan paksa single table
    if any(kw in q for kw in ["total", "sum", "avg", "average", "mean", "count", "berapa banyak", "how many"]):
        return False

    return True

def _is_simple_orders_lookup(question: str) -> bool:
    """
    True kalau user cuma minta daftar order (misal orderNumber) 
    tanpa ngomong 'order details' / 'line items' dan tanpa agregasi.
    Contoh:
      - "show all order numbers between ..."
      - "list order numbers for 2004"
    """
    q = (question or "").lower()

    # harus ada frasa 'order number(s)' biar ini spesifik
    if "order number" not in q and "order numbers" not in q:
        return False

    # kalau ngomong detail/line items → biar classifier & join yang urus
    if any(kw in q for kw in ["order detail", "order details", "line item", "line items"]):
        return False

    # kalau ada kata agregasi → jangan paksa single-table
    if any(
        kw in q
        for kw in ["total", "sum", "avg", "average", "mean", "count",
                   "berapa banyak", "how many", "jumlah "]
    ):
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
    Perbaiki nama kolom berdasar alias_map dan fallback classicmodels,
    TAPI jangan menyentuh teks dalam kutip ('...', "...", `...`)
    supaya alias/label buatan LLM tetap utuh.
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

    for table in get_all_tables():
        # Add name column mapping
        name_col = get_name_column(table)
        if name_col:
            # "name" -> actual name column (only if not already mapped)
            if "name" not in dynamic_fallback:
                dynamic_fallback["name"] = name_col
            # Self-reference for case normalization (productname -> productName)
            dynamic_fallback[name_col.lower()] = name_col

        # Add ID column mapping
        id_col = get_id_column(table)
        if id_col:
            # Common ID aliases
            id_aliases = ["id", f"{table[:-1]}id", f"{table[:-1]}_id", f"id_{table[:-1]}", id_col.lower()]
            for alias in id_aliases:
                if alias not in dynamic_fallback:
                    dynamic_fallback[alias] = id_col

    # Schema-specific aliases (kept as-is for safety)
    # These are patterns that can't be inferred from metadata
    specific_aliases = {
        "listprice": "MSRP",          # products-specific
        "price": "priceEach",          # orderdetails-specific
        "contactfirstname": "contactFirstName",  # customers
        "contactlastname": "contactLastName",    # customers
        "phoneextension": "extension",           # employees
        "firstname": "firstName",     # employees
        "lastname": "lastName",       # employees
        "extension": "extension",     # employees
    }
    for alias, canon in specific_aliases.items():
        if alias not in dynamic_fallback:
            dynamic_fallback[alias] = canon

    # Merge: dynamic_fallback only if not already in global_map
    for al, canon in dynamic_fallback.items():
        al = al.lower()
        if al not in global_map:
            global_map[al] = canon
        elif global_map[al] is None:
            pass

    keywords = _SQL_KEYWORDS
    token_pat = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")

    def replace_in_code_segment(seg: str) -> str:
        def smart_repl(m):
            w = m.group(0)
            lw = w.lower()
            if lw in keywords:
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

    Example: If fk_col is "customerNumber" and target_table is "customers",
    the target column is likely "customerNumber".
    """
    # Check if we have this column in ranked_cols
    for tbl, col in ranked_cols:
        if tbl.lower() == target_table.lower():
            col_lower = col.lower()
            # Common pattern: fk_col = target_col (e.g., customerNumber -> customerNumber)
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

    def _question_mentions_column(q_text: str, column: str) -> bool:
        """
        Robust mention detector for implicit table discovery.
        Handles camelCase/snake_case + token-level semantic matches.
        """
        if not column:
            return False

        col_lower = column.lower()
        spaced = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', column).replace("_", " ").lower()

        # Direct / phrase match
        if col_lower in q_text:
            return True
        if spaced != col_lower and spaced in q_text:
            return True

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
                    return False

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
                return True

        return False

    # Find tables whose columns are mentioned in question
    mentioned_tables = defaultdict(set)  # {table: {matched_columns}}

    for tbl, col in ranked_cols:
        tbl_lower = tbl.lower()

        # Skip if already in relevant tables
        if tbl_lower in relevant_lower:
            continue

        if _question_mentions_column(q_lower, col):
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

    # Extract FK relationships from context
    fk_map = _extract_fk_from_context(ctx)

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
    # Also check for "N products/highest/largest" followed by "stock/amount/etc"
    top_n_context = re.search(r'\b\d+\s+(products|customers|items?|entities?).+?(highest|largest|most)\b', q_lower, re.IGNORECASE)

    is_top_n_global_ranking = bool(top_n_pattern or top_n_context) and not is_ordinal_ranking

    if is_top_n_global_ranking:
        constraints.append(
            "TOP N QUERY (HIGHEST PRIORITY): This question asks for 'Top N' or 'N highest/lowest'. "
            "YOU MUST USE: ORDER BY column DESC LIMIT N. "
            "EXAMPLE: 'Show 10 products with highest stock' → SELECT productCode, productName, quantityInStock FROM products ORDER BY quantityInStock DESC LIMIT 10. "
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
    # === [NEW] FOR EACH = ALL VALUES, NO WHERE FILTER ===
    for_each_triggers = ["for each", "each ", "per ", "every "]
    if any(t in question.lower() for t in for_each_triggers):
        constraints.append(
            "FOR EACH RULE (CRITICAL): When the question says 'For each X' or 'Per X', you MUST return ALL values of X. "
            "Do NOT add WHERE filters on the grouping column unless the user explicitly filters it. "
            "Do NOT add subqueries to filter grouping columns. "
            "TABLE SELECTION RULE: When counting 'number of Y' per each X, use FROM the Y table (the entity being counted), "
            "NOT the dimension table for X. "
            "Example: 'for each product line, count products' -> FROM products GROUP BY productLine (NOT FROM productlines). "
            "The COUNT(*) counts rows in the FROM table, so use the table containing the entities to count. "
            "CORRECT: SELECT groupingCol, COUNT(*) FROM entity_table GROUP BY groupingCol "
            "WRONG: SELECT groupingCol, COUNT(*) FROM dimension_table GROUP BY groupingCol (this counts dimension rows, not entities)"
        )

    # === [NEW] EXPLICIT RANK() FUNCTION RULE ===
    # Detect "a rank of X" / "rank employees" -> use RANK() OVER window function
    if is_explicit_rank_function:
        constraints.append(
            "RANK FUNCTION RULE (CRITICAL): The question asks for a 'rank' of items. "
            "You MUST use the RANK() or DENSE_RANK() window function. "
            "CORRECT PATTERN: `SELECT col1, col2, RANK() OVER (PARTITION BY groupCol ORDER BY orderCol DESC) AS employee_rank FROM table` "
            "WRONG: Using COUNT(*) - that gives a total count, NOT a rank. "
            "WRONG: Using ROW_NUMBER() - use RANK() for tied values. "
            "IMPORTANT: RANK() assigns the same rank to equal values and skips ranks for ties."
        )

    # === [NEW] WINDOW AGGREGATE RULE: "within the same X" pattern ===
    if is_window_aggregate:
        partition_col_hint = ""
        if partition_hint:
            # Try to infer column name from hint (e.g., "office" -> "officeCode")
            partition_col_hint = f" Look for a column like `{partition_hint}Code` or `{partition_hint}Number` or `{partition_hint}Id` to use as the PARTITION BY column."
        constraints.append(
            f"WINDOW AGGREGATE RULE (CRITICAL): The phrase 'within the same' or 'in the same' means you MUST use a WINDOW FUNCTION with PARTITION BY. "
            f"CORRECT: `SELECT col1, col2, COUNT(*) OVER (PARTITION BY groupColumn) AS total FROM table` "
            f"WRONG: `COUNT(*) OVER ()` without PARTITION BY. "
            f"WRONG: Using GROUP BY with window functions.{partition_col_hint} "
            f"IMPORTANT: Do NOT add GROUP BY clause when using window functions for this pattern."
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

    # [C] RANKING RULES
    # Check winner-row pattern FIRST (before qtype check)
    # This ensures winner-row constraints are applied even when qtype="aggregate" due to grouping
    if is_winner_row or qtype == "rank":
        constraints.append(
            "WINNER-ROW STRATEGY: You MUST use the standard 'Select-Filter-Rank' pattern."
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

        print(f"[PROMPT DEBUG] complexity={complexity}, state provided, primary_table={primary_table}")

        if primary_table:
            other_tables = state.get("router_tables", [])
            other_tables = [t for t in other_tables if t != primary_table]
            other_tables_str = ", ".join(other_tables) if other_tables else "none"
            requires_inner_join = _question_requires_inner_join(question, ctx)
            join_keyword = "INNER JOIN" if requires_inner_join else "LEFT JOIN"

            # More explicit guidance - tell LLM exactly which table should be first
            hint_msg = f"TABLE ORDER (CRITICAL): Start your query with FROM {primary_table}"
            if other_tables_str != "none":
                hint_msg += f", then {join_keyword} {other_tables_str}"
            if primary_reason:
                hint_msg += f". Reason: {primary_reason}"
            hint_msg += f". Follow this exact table order."

            print(f"[PROMPT DEBUG] Adding hint: {hint_msg}")
            constraints.append(hint_msg)

            # JOIN TYPE GUIDANCE (CRITICAL)
            if requires_inner_join:
                # Filtering pattern: use INNER JOIN to exclude non-matching rows
                constraints.append(
                    "JOIN TYPE (CRITICAL): This query has explicit filtering intent (e.g., 'only ... that have/with ...'). "
                    "Use **INNER JOIN** to exclude entities without matching data. "
                    "This takes precedence over 'for each' phrasing."
                )
            else:
                # Default: use LEFT JOIN to preserve all entities from primary table
                constraints.append(
                    "JOIN TYPE (CRITICAL): Use **LEFT JOIN** to preserve all entities from the primary table."
                )
                constraints.append(
                    "FILTER RULE (CRITICAL): If using LEFT JOIN, filters for the JOINED table MUST be in the **ON** clause, NOT WHERE."
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
                f"Select ONLY these columns (plus the 'For each' entity identifier). "
                f"Do NOT add any other columns like checkNumber, orderNumber, etc."
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
            "Do NOT add extra columns like IDs, codes, or check numbers unless explicitly mentioned."
        )
    constraints.append(
        "IMPLICIT SELECTION RULE: If the user says 'For each [Entity]' or 'List [Entity]', "
        "you MUST include that Entity's Identifier/Name in the SELECT list, even if not explicitly listed in 'Include ...'. "
        "Example: 'For each Customer, show total amount' -> You MUST Select `customerName` AND `totalAmount`."
    )
    constraints.append(
        "NO EXTRA COLUMNS: Apart from the 'For each' entity, do NOT add other IDs/Codes unless asked."
    )
    constraints.append(
        "ALIAS CONSISTENCY: If you use a table alias (e.g. `t`, `p`, `cust`), "
        "you MUST prefix ALL columns in the SELECT clause with that alias. "
        "Bad: `SELECT p.id, name FROM ...` | Good: `SELECT p.id, p.name FROM ...`"
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

    # --- 4. RETURN PROMPT LENGKAP ---
    return fr"""
You are a senior SQL assistant for MySQL 8.0.

Use ONLY tables/columns listed under ALLOWED IDENTIFIERS in Context.
If Context lists aliases, use the canonical column name.

Output:
Return EXACTLY one JSON object:
{{"sql":"<final MySQL query>","notes":"<brief rationale>"}}
No Markdown. End SQL with a semicolon.
If NO JOINs, notes MUST be exactly: "No joins used."

Context:
{ctx}
{join_context}

Question:
{question}
{constraint_block}
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
            print(f"[PH4] WARNING: HAVING still stuck after normalization: {s[:200]}")

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

    s = _norm_kw(s)
    s = _fix_missing_space_before_clauses(s)
    if not s.endswith(";"):
        s += ";"
    # ---------------------------------------------------------------------------
    # Phase 0 — Syntax & Canonicalization
    # Normalize tokens and table naming early
    # ---------------------------------------------------------------------------
    s = _fix_bad_backticks(s)
    s = _canonicalize_table_names(s, ctx)
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
    # List of SQL functions that indicate computed expressions
    computed_func_patterns = [
        r'COUNT\s*\(', r'SUM\s*\(', r'AVG\s*\(', r'MIN\s*\(', r'MAX\s*\(',
        r'CASE\s+WHEN', r'IF\s*\(', r'COALESCE\s*\(', r'NULLIF\s*\(',
        r'CAST\s*\(', r'CONVERT\s*\(', r'DATE_FORMAT\s*\(',
        r'ROW_NUMBER\s*\(', r'RANK\s*\(', r'DENSE_RANK\s*\(',
        r'LAG\s*\(', r'LEAD\s*\(', r'FIRST_VALUE\s*\(', r'LAST_VALUE\s*\(',
        r'CONCAT\s*\(', r'GROUP_CONCAT\s*\(', r'SUBSTRING\s*\(', r'SUBSTR\s*\(',
        r'UPPER\s*\(', r'LOWER\s*\(', r'TRIM\s*\(', r'LENGTH\s*\('
    ]
    # Pattern to match: plain_column AS alias or table.column AS alias
    # Only match if NOT preceded by a function call
    def remove_plain_column_alias(match):
        column_ref = match.group(1)  # e.g., 'status' or 't.status'
        alias_name = match.group(2) if len(match.groups()) > 1 else match.group(0).split(' AS ')[-1]
        # Strip quotes from alias name for checking
        alias_clean = alias_name.strip('\'"`')
        # Check if this is a computed expression by looking for function patterns
        combined = column_ref + ' ' + alias_clean
        for func_pattern in computed_func_patterns:
            if re.search(func_pattern, combined, re.IGNORECASE):
                # This is a computed expression, keep the AS alias
                return match.group(0)
        # This is a plain column, remove the AS alias
        return column_ref

    # FIX: First, protect CASE WHEN...END AS alias patterns by temporarily replacing them
    # This prevents the regex from incorrectly matching "END AS alias"
    case_aliases = {}
    def protect_case_aliases(match):
        alias = f"__CASE_ALIAS_{len(case_aliases)}__"
        case_aliases[alias] = match.group(0)
        return alias
    s = re.sub(
        r'\bCASE\s+WHEN\b.+?\bEND\b\s+AS\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        protect_case_aliases,
        s,
        flags=re.IGNORECASE | re.DOTALL
    )

    # Pattern 1: Quoted aliases (single or double quotes)
    # Matches: status AS 'total orders', status AS "total", t.status AS "alias"
    s = re.sub(
        r"([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)\s+AS\s+(['\"])([^'\"]+)\2",
        lambda m: remove_plain_column_alias(type('obj', (object,), {'group': lambda self, i: m.group(1) if i == 1 else m.group(3), 'groups': lambda self: (m.group(1), m.group(3))})()),
        s,
        flags=re.IGNORECASE
    )
    # Pattern 2: Unquoted aliases
    # Matches: status AS total_orders, t.status AS total
    s = re.sub(
        r"([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)\s+AS\s+([a-zA-Z_][a-zA-Z0-9_]*)",
        remove_plain_column_alias,
        s,
        flags=re.IGNORECASE
    )

    # FIX: Restore protected CASE WHEN aliases
    for alias, original in case_aliases.items():
        s = s.replace(alias, original)

    # NEW: Normalize overly long aliases by removing filler words
    s = _normalize_alias_lengths(s)
    # ---------------------------------------------------------------------------
    # Phase 3 — Column Naming & GROUP BY Consistency
    # Operate on final column identities
    # ---------------------------------------------------------------------------
    s = _prefer_friendly_names(s, question)
    s = _expand_group_by_aliases_regex(s)
    if not _is_potential_winner_row(s, question, force=True):
        # REMOVE invalid columns FIRST, then enforce GROUP BY consistency
        # This prevents columns from being added to GROUP BY unnecessarily
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

    # Skip if query uses window functions that REQUIRE OVER clause
    window_funcs_requiring_over = ["ROW_NUMBER", "RANK", "DENSE_RANK", "LAG", "LEAD",
                                    "FIRST_VALUE", "LAST_VALUE", "NTH_VALUE",
                                    "NTILE", "PERCENT_RANK", "CUME_DIST"]
    has_window_func = any(f"{wf}()" in s_upper or f"{wf} (" in s_upper for wf in window_funcs_requiring_over)

    if "GROUP BY" in s_upper and "OVER" in s_upper and "PARTITION" in s_upper and not has_window_func:
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

    if DEBUG_WR: print("\n[PH4] start ----")
    # FIX: NO re.I flag - creditLimit won't match (lowercase 'imit'), but HAVING will match (uppercase)
    s = re.sub(
    r"(\b[A-Za-z_][A-Za-z0-9_]*)(WHERE|GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT)\b",
    r"\1 \2",
    s)
    if DEBUG_WR: print(f"[PH4] after re-spacing: {s[:200]}")

    # pre-sanity only (no rewrite)
    s = _winner_row_sanity_date_in_groupby(s, question)
    if DEBUG_WR: print("[PH4] after sanity:", s)

    # remove MAX(..) HAVING self-subquery hacks
    s = _strip_having_max_self_subquery(s)
    if DEBUG_WR: print("[PH4] after strip having max self subquery:", s)

    # NEW: Detect and fix malformed MAX/GROUP BY + ROW_NUMBER mixed pattern
    # This catches cases where 7B model incorrectly combines aggregate and window function patterns
    fixed_mixed = _fix_mixed_max_groupby_rownumber(s)
    if fixed_mixed:
        if DEBUG_WR: print("[PH4] Fixed malformed MAX/GROUP BY + ROW_NUMBER pattern")
        s = fixed_mixed

    # NEW: Fix redundant GROUP BY + MAX(CASE WHEN rn = N...) when WHERE rn = N is present
    # When WHERE rn = N exists, each group has exactly ONE row, so GROUP BY is redundant
    fixed_redundant = _fix_redundant_groupby_with_rn_filter(s)
    if fixed_redundant:
        if DEBUG_WR: print("[PH4] Fixed redundant GROUP BY with WHERE rn = N pattern")
        if DEBUG_WR: print(f"[PH4] Before: {s[:200]}")
        if DEBUG_WR: print(f"[PH4] After:  {fixed_redundant[:200]}")
        s = fixed_redundant

    # NEW: Fix inconsistent subquery alias hallucination
    # LLM sometimes uses different alias in outer SELECT than the subquery alias
    fixed_alias = _fix_inconsistent_subquery_alias(s)
    if fixed_alias:
        if DEBUG_WR: print("[PH4] Fixed inconsistent subquery alias")
        if DEBUG_WR: print(f"[PH4] Before: {s[:200]}")
        if DEBUG_WR: print(f"[PH4] After:  {fixed_alias[:200]}")
        s = fixed_alias

    # -------------------------------------------------------------
    # CASE 1 — pattern clearly indicates MAX + date tie-breaker
    # → enforce ROW_NUMBER rewrite (FORCE MODE)
    # -------------------------------------------------------------
    if _looks_like_max_date_tiebreaker(s):
        # Jangan force rewrite kalau question JELAS bukan winner-row
        if _question_wants_winner_row(question):
            if DEBUG_WR: print("[PH4] detected max-date tie-breaker + winner-row intent -> forcing guardrail")
            # CRITICAL: Check if query is already corrupted
            s_forced = _rewrite_winner_row_guardrail(s, question, force=True)
            # SANITY: If forced rewrite produced garbage, rollback to original
            if not s_forced or ("GROUP BY" in s_forced.upper() and "ROW_NUMBER" in s_forced.upper()):
                if DEBUG_WR: print("[PH4] ABORT: forced rewrite produced malformed SQL, keeping original")
                s = before_phase4
            else:
                s = s_forced
            if DEBUG_WR: print("[PH4] FINAL after forced guardrail:", s)            
            #  Skip normal heuristic flow - we've made our choice
            s = _rewrite_having_alias_to_expr(s)
            s = _force_restore_groupby_if_aggregate_without_groupby(before_phase4, s)
            if DEBUG_WR: print("[PH4] end ----", s, "\n")           
            #  DON'T return early - continue to Phase 5-7
            goto_phase5 = True
        else:
            if DEBUG_WR: print("[PH4] max-date pattern detected BUT question wants true aggregation -> skip force")
            goto_phase5 = False
    else:
        goto_phase5 = False
    # ------------------------------------------------------------
    #  CASE 2 — normal heuristic winner-row handling
    # ------------------------------------------------------------
    if not goto_phase5:
        s = _rewrite_winner_row_guardrail(s, question)
        if DEBUG_WR: print("[PH4] after guardrail:", s)

        s = _rewrite_winner_row_aggregate_misuse(s, question)
        if DEBUG_WR: print("[PH4] after agg-misuse:", s)
        
        s = _fix_aggregate_in_where(s)
        if DEBUG_WR: print(f"[PH4] after agg-in-where fix: {s}")

        # Fix raw column WHERE -> HAVING for aggregate questions
        s = _fix_raw_column_where_to_having(s, question)
        if DEBUG_WR: print(f"[PH4] after raw-where-to-having fix: {s}")

        # Add missing COUNT(*) for GROUP BY aggregates
        s = _add_missing_count_for_group_by(s, question)
        if DEBUG_WR: print(f"[PH4] after add-count fix: {s}")

        s = _rewrite_having_alias_to_expr(s)
        s = _winner_row_final_sanity(s, question)

        # fail-safe restore group by if pipeline broke it
        s = _force_restore_groupby_if_aggregate_without_groupby(before_phase4, s)

        if DEBUG_WR: print("[PH4] end ----", s, "\n")
    else:
            # Phase 4 was skipped due to goto_phase5, but still need aggregate fix
            s = _fix_aggregate_in_where(s)
            if DEBUG_WR: print(f"[PH4-FORCE] after agg-in-where fix: {s}")
            s = _fix_raw_column_where_to_having(s, question)
            if DEBUG_WR: print(f"[PH4-FORCE] after raw-where-to-having fix: {s}")
            s = _add_missing_count_for_group_by(s, question)
            if DEBUG_WR: print(f"[PH4-FORCE] after add-count fix: {s}")        
    # ---------------------------------------------------------------------------
    # Phase 5 — Pattern, REGEXP, and Predicate Normalization
    # ---------------------------------------------------------------------------
    # print("\n[PH5] start ----")  # ADD THIS
    # s = _guardrail_fix_like_regex_anchors(s, question)
    # print(f"[PH5] after fix_anchors: {s[:200]}")  # ADD THIS
    # s = _rewrite_simple_regexp_to_like(s, question)
    # s = _guardrail_fix_mysql_regexp_literals(s)
    # print(f"[PH5] end ---- {s}\n")  # ADD THIS
    # ---------------------------------------------------------------------------
    # Phase 6 — NULL / DISTINCT / Result Cardinality Policies
    # ---------------------------------------------------------------------------
    s = _expand_missing_null_to_null_or_empty(s, question)
    s = _ensure_reasonable_distinct(s, question)
    s = _strip_unneeded_count_distinct(s)
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
    # Final Polish
    # ---------------------------------------------------------------------------
    s = _collapse_spaces_outside_quotes(s)
    return s

def normalize_sql_pipeline(raw_sql: str, question: str, ctx: str, ranked_cols=None) -> str:
    """
    Pipeline standar untuk normalisasi SQL:
    - postprocess (normalize keyword, limit, distinct, dsb)
    - repair alias berdasarkan alias_map
    - postprocess lagi setelah alias diganti
    """
    _dbg_print_two("[normalize] input sql:", raw_sql)
    if not raw_sql:
        return raw_sql

    s = _postprocess_sql(raw_sql, question, ctx)

    if ranked_cols:
        pref = pick_preferred_anchor_from_ranked_cols(ranked_cols, top_n=12)
        s = normalize_anchor_for_simple_inner_join(s, pref)

    s = _repair_with_aliases(s, ctx)
    s = _postprocess_sql(s, question, ctx)
    _dbg_print_two("[normalize] output sql:", s)
    return s
