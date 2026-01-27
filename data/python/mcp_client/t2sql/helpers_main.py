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
from .helpers_core import (
    _parse_allowed_identifiers, _question_requires_word_boundary_regex, _question_requires_regex, has_join,
    # phase 0
    _fix_bad_backticks, _canonicalize_table_names, ensure_join_aliases,
    # phase 1
    _rewrite_left_join_to_inner_if_not_needed, _rewrite_left_join_to_inner_if_rhs_filtered,
    _rewrite_inner_join_filter_to_exists, _rewrite_inner_join_existence_to_exists,
    _drop_redundant_inner_join_same_key, _prune_ornamental_joins_safe, has_non_inner_join, join_count,
    # phase 2
    _drop_redundant_column_aliases, _drop_unused_from_table_aliases, _strip_alias_if_single_table,
    # phase 3
    _wants_customer_name_only, _wants_product_name_only, _prefer_friendly_names, _wants_name_only,
    _expand_group_by_aliases_regex, _is_potential_winner_row, _enforce_explicit_group_by,
    # phase 4
    _winner_row_sanity_date_in_groupby, _strip_having_max_self_subquery, _looks_like_max_date_tiebreaker,
    _question_wants_winner_row, _rewrite_winner_row_guardrail, _rewrite_having_alias_to_expr,
    _force_restore_groupby_if_aggregate_without_groupby,
    _rewrite_winner_row_aggregate_misuse, _winner_row_final_sanity,
    # phase 4b - Aggregate in WHERE fix
    _fix_aggregate_in_where,
    # phase 5 — Pattern, REGEXP, and Predicate Normalization
    _guardrail_fix_like_regex_anchors, _rewrite_simple_regexp_to_like, _guardrail_fix_mysql_regexp_literals,
    # phase 6 
    _expand_missing_null_to_null_or_empty, _ensure_reasonable_distinct, _strip_unneeded_count_distinct, _parse_from_join_aliases,
    # phase 7
    _remove_top_level_order_by_if_not_requested, _maybe_add_limit, _collapse_spaces_outside_quotes, _ALIAS_OVERRIDES
)
# ======================================================================
# ===================== SQL ALIAS & KEYWORD CONSTANTS ==================
# ======================================================================
_SQL_KEYWORDS: Set[str] = {
    "select","from","where","group","by","order","limit","join","left","right","inner",
    "on","and","or","not","as","distinct","count","sum","avg","min","max","having",
    "asc","desc","like","in","between","is","null","case","when","then","else","end",
    "regexp","rlike","over","partition","row_number"
}
TRANSACTION_TABLES = {"payments", "orderdetails", "orders"}
_ANCHOR_SIGNALS = {
    "payments": {"amount", "paymentDate", "customerNumber"},
    "orderdetails": {"quantityOrdered", "priceEach", "orderLineNumber", "orderNumber", "productCode"},
    "orders": {"orderDate", "status", "shippedDate", "requiredDate", "customerNumber"},
}
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
# def _identify_used_identifiers(sql: str) -> Dict[Optional[str], Set[str]]:
#     """
#     Return mapping:
#       - qualifier -> set(columns)  e.g. {"p": {"amount","paymentDate"}}
#       - None      -> set(unqualified tokens) e.g. {None: {"amount"}}

#     Konservatif:
#     - Strip string literals dari scanning.
#     - Exclude keywords, table names, aliases (FROM/JOIN/derived/output).
#     - Skip token yang terlihat seperti function call: token(...).
#     """
#     if not sql:
#         return {}

#     out: Dict[Optional[str], Set[str]] = {}
#     # ---------------------------------------------------------------------------
#     # 1) Hapus string literals/double quotes supaya token scan tidak ke-distract
#     # ---------------------------------------------------------------------------
#     code_only = re.sub(r"'(?:''|[^'])*'", "''", sql)
#     code_only = re.sub(r'"(?:""|[^"])*"', '""', code_only)
#     keywords = _SQL_KEYWORDS
#     # ---------------------------------------------------------------------------
#     # 2) Ambil table names yang muncul di FROM/JOIN (untuk exclude)
#     #    Support schema.table dan backticks secara sederhana.
#     # ---------------------------------------------------------------------------
#     table_names: Set[str] = set()
#     for m in re.finditer(
#         r"\b(?:from|join)\s+(`?[A-Za-z_][A-Za-z0-9_]*`?(?:\.`?[A-Za-z_][A-Za-z0-9_]*`?)?)",
#         code_only,
#         flags=re.I,
#     ):
#         raw = (m.group(1) or "").replace("`", "")
#         tbl = raw.split(".")[-1]
#         if tbl:
#             table_names.add(tbl.lower())
#     # Exclude aliases (table + derived + output)
#     exclude: Set[str] = set()
#     # ---------------------------------------------------------------------------
#     # 3) table aliases from FROM/JOIN
#     # ---------------------------------------------------------------------------
#     a2t = _parse_from_join_aliases(sql)  # alias -> table
#     exclude.update({a.lower() for a in a2t.keys() if a})
#     # ---------------------------------------------------------------------------
#     # 4) derived-table aliases: FROM ( ... ) AS t / JOIN ( ... ) AS t
#     # ---------------------------------------------------------------------------
#     for m in re.finditer(
#         r"\b(?:from|join)\s*\([^)]*\)\s*(?:as\s+)?([A-Za-z_]\w*)\b",
#         code_only,
#         flags=re.I | re.S,
#     ):
#         exclude.add(m.group(1).lower())
#     # ---------------------------------------------------------------------------
#     # 5) output aliases: SELECT ... AS alias (hanya area SELECT ... FROM)
#     # ---------------------------------------------------------------------------
#     m_sel = re.search(r"\bselect\b(.*?)\bfrom\b", code_only, flags=re.I | re.S)
#     if m_sel:
#         select_part = m_sel.group(1)
#         for m in re.finditer(r"\bas\s+([A-Za-z_]\w*)\b", select_part, flags=re.I):
#             exclude.add(m.group(1).lower())
#     # ---------------------------------------------------------------------------
#     # 6) Collect qualified usages: alias.col / table.col
#     # ---------------------------------------------------------------------------
#     # Simpan dulu semua pattern qualifier.column
#     for m in re.finditer(
#         r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)\b",
#         code_only,
#         flags=re.I,
#     ):
#         qual = m.group(1)
#         col = m.group(2)
#         ql = (qual or "").lower()

#         if not qual or not col:
#             continue
#         if ql in keywords:
#             continue
#         if ql in exclude:
#             continue

#         out.setdefault(qual, set()).add(col)
#     # Remove qualified occurrences so they don't pollute unqualified scan
#     code_no_qual = re.sub(
#         r"\b[A-Za-z_][A-Za-z0-9_]*\s*\.\s*[A-Za-z_][A-Za-z0-9_]*\b",
#         " ",
#         code_only,
#         flags=re.I,
#     )
#     # ---------------------------------------------------------------------------
#     # 7) Unqualified token scan (konservatif)
#     # ---------------------------------------------------------------------------
#     tok_pat = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")
#     for m in tok_pat.finditer(code_no_qual):
#         tok = m.group(1)
#         low = tok.lower()

#         if low in keywords:
#             continue
#         if low in table_names:
#             continue
#         if low in exclude:
#             continue
#         # skip jika terlihat seperti function call: tok(...)
#         j = m.end()
#         while j < len(code_no_qual) and code_no_qual[j].isspace():
#             j += 1
#         if j < len(code_no_qual) and code_no_qual[j] == "(":
#             continue

#         out.setdefault(None, set()).add(tok)

#     return out
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
    mentions_customer = any(k in q for k in ["customer", "customers", "pelanggan"])
    mentions_product  = any(k in q for k in ["product", "products", "produk"])
    mentions_order    = any(k in q for k in ["order", "orders", "pesanan"])
    mentions_employee = any(k in q for k in ["employee", "employees", "karyawan"])

    # entity tambahan (tabel lain)
    mentions_productline = any(
        k in q
        for k in [
            "product line", "product lines",
            "lini produk", "line produk",
            "kategori", "category"
        ]
    )
    mentions_office = any(
        k in q
        for k in [
            "office", "offices", "kantor",
            "territory", "territories", "region", "regional"
        ]
    )
    mentions_payment = any(
        k in q
        for k in [
            "payment", "payments", "pembayaran",
            "check ", "cheque", "cheques"
        ]
    )
    mentions_orderdetails = any(
        k in q
        for k in [
            "order detail", "order details",
            "line item", "line items",
            "item pesanan", "detail pesanan"
        ]
    )

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
    # 2) Entity-specific hints
    # ---------------------------------------------------------------------------
    if mentions_customer:
        base += ["customer"]
        if not id_like:
            base += ["customerName"]
        else:
            base += ["customerNumber"]

    if mentions_product:
        base += ["product"]
        if not id_like:
            base += ["productName"]
        else:
            base += ["productCode"]

    if mentions_order:
        base += ["order", "orderNumber"]

    if mentions_employee:
        base += ["employee", "employeeNumber"]

    if mentions_productline:
        base += ["productLine"]

    if mentions_office:
        base += ["officeCode", "city", "country", "territory"]

    if mentions_payment:
        base += ["payments", "amount", "paymentDate"]

    if mentions_orderdetails:
        base += ["orderdetails", "quantityOrdered", "priceEach"]
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

    # 1) signal-based (lebih kuat)
    for tbl in ("payments", "orderdetails", "orders"):
        sig = _ANCHOR_SIGNALS[tbl]
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

    # fallback untuk classicmodels
    classic_fallback = {
        "products": {
            "name": "productName",
            "productname": "productName",
            "listprice": "MSRP",
        },
        "orderdetails": {
            "listprice": "priceEach",
            "price": "priceEach",
        },
        "customers": {
            "name": "customerName",
            "customername": "customerName",
            "id": "customerNumber",
            "customerid": "customerNumber",
            "customer_id": "customerNumber",
            "id_customer": "customerNumber",
            "contactfirstname": "contactFirstName",
            "contactlastname": "contactLastName", 
        },
        "employees": {
            "contactfirstname": "firstName",
            "contactlastname": "lastName",
            "phoneextension": "extension",
            "firstname": "firstName",
            "lastname": "lastName",
            "extension": "extension",
            "employeenumber": "employeeNumber",
        },
    }
    for _t, amap in classic_fallback.items():
        for al, canon in amap.items():
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

def make_generation_prompt(ctx: str, question: str, router_decision: dict = None) -> str:
    """
    Dynamic Prompt Builder.
    Versi Benchmark Standard (ANSI SQL Strict):
    1. Memaksa pengulangan ekspresi di GROUP BY (ANSI Compliant).
    2. Melarang UNION untuk binning.
    3. Memaksa Single Line SQL untuk mencegah JSON Crash pada query panjang.
    4. Menangani Error 1055 dengan strict column selection.
    """
    router_decision = router_decision or {}
    complexity = router_decision.get("complexity", "simple")
    qtype = router_decision.get("qtype", "lookup")
    has_grouping = router_decision.get("has_grouping", False)
    has_exclusion = router_decision.get("has_exclusion", False)
    
    # --- 1. BANGUN CONSTRAINT DINAMIS ---
    constraints = []
    # === [NEW] GENERAL ANTI-HALLUCINATION RULE ===
    constraints.append(
        "CRITICAL RULE: Do NOT add SQL clauses (WHERE, HAVING, LIMIT, ORDER BY) "
        "unless the user EXPLICITLY requests them. "
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
        
        if qtype == "lookup":
            constraints.append(
                "SIMPLE LOOKUP RULE: Do NOT use aggregation functions like MAX(), MIN(), or SUM(). "
                "Just SELECT the column directly. Use 'ORDER BY column DESC' for 'latest/newest/highest'."
            )
            constraints.append(
                "STRICT ALIAS PROHIBITION: Do NOT rename table columns with AS unless it's a computed expression.\n"
                "Never rename plain table columns with AS."
            )

    # [C] RANKING RULES
    if qtype == "rank":
        constraints.append(
            "WINNER-ROW STRATEGY: You MUST use the standard 'Select-Filter-Rank' pattern."
        )
        constraints.append(
            "STRICT SYNTAX TEMPLATE: "
            "`SELECT col1, col2 FROM (SELECT col1, col2, ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ... DESC) AS rn FROM table) WHERE rn = 1`"
        )
        constraints.append(
            "CRITICAL PROHIBITION: Do NOT put `ROW_NUMBER()` inside a `GROUP BY` clause. "
            "Do NOT use `HAVING` for ranking. Use the subquery pattern above."
        )

    # [D] REPORTING/AGGREGATION RULES
    is_reporting = (qtype == "aggregate") or has_grouping or has_exclusion
    if is_reporting and complexity == "complex":
        constraints.append(
            "REPORTING LOGIC: When aggregating a Master Table with a Transaction Table, use **LEFT JOIN**."
        )
        constraints.append(
            "CRITICAL FILTER RULE: If using LEFT JOIN, filters for the RIGHT table MUST be in the **ON** clause."
        )
        constraints.append(
            "GROUP BY RULE: Every column in the SELECT list that is not aggregated MUST be in the GROUP BY clause."
        )

    # [E] DATE RULES
    constraints.append(
        "DATE LABEL STRICT RULE: If asking for Month Name, GROUP BY both MONTH(date) and DATE_FORMAT(date, '%M')."
        "DATE FILTERING RULES:\n"
        "   - For year filtering: Use `YEAR(date_column) = YYYY` (NOT SUBSTR).\n"
        "   - For month filtering: Use `MONTH(date_column) = M`.\n"
        "   - For range filtering: Use `date_column BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD'`."
    )

    # [F] COLUMN INTEGRITY RULES (BENCHMARK MODE)
    constraints.append(
        "STRICT COLUMN SELECTION: Select EXACTLY the columns requested by the user."
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
    # 1. Casing Strictness
    constraints.append(
        "CASE SENSITIVITY RULE: The database uses specific casing (often camelCase like `orderDate`). "
        "You MUST use the exact column names from the Schema. NEVER use snake_case."
    )

    # 2. Month Sorting
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

    # --- 2. RAKIT CONSTRAINT BLOCK ---
    constraint_block = ""
    if constraints:
        constraint_block = "\n[STRICT CONSTRAINTS & HEURISTICS]:\n" + "\n".join([f"- {c}" for c in constraints])

    # --- 3. RETURN PROMPT LENGKAP ---
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
    s = ensure_join_aliases(s)
    # ---------------------------------------------------------------------------
    # Phase 1 — JOIN Semantics & Structural Rewrites (FINAL)
    # Decide JOIN necessity and semantics before touching aliases
    # ---------------------------------------------------------------------------
    s = _rewrite_left_join_to_inner_if_not_needed(s, question, ctx)
    s = _rewrite_left_join_to_inner_if_rhs_filtered(s, question, ctx)
    s = _rewrite_inner_join_filter_to_exists(s)
    s = _rewrite_inner_join_existence_to_exists(s, question)
    s = _drop_redundant_inner_join_same_key(s)
    s = _prune_ornamental_joins_safe(s, question)
    # ---------------------------------------------------------------------------
    # Phase 2 — Alias Hygiene & FROM Cleanup
    # Safe only after JOIN structure is frozen
    # ---------------------------------------------------------------------------
    s = _drop_redundant_column_aliases(s)
    s = _drop_unused_from_table_aliases(s)
    s = _strip_alias_if_single_table(s)
    # ---------------------------------------------------------------------------
    # Phase 3 — Column Naming & GROUP BY Consistency
    # Operate on final column identities
    # ---------------------------------------------------------------------------
    s = _prefer_friendly_names(s, question)
    s = _expand_group_by_aliases_regex(s)
    if not _is_potential_winner_row(s, question, force=True):
        s = _enforce_explicit_group_by(s)
    # ---------------------------------------------------------------------------
    # Phase 3b — Reorder GROUP BY: Keys First, Descriptive Columns Later
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
        s = _rewrite_having_alias_to_expr(s)
        s = _winner_row_final_sanity(s, question)

        # fail-safe restore group by if pipeline broke it
        s = _force_restore_groupby_if_aggregate_without_groupby(before_phase4, s)

        if DEBUG_WR: print("[PH4] end ----", s, "\n")
    else:
            # Phase 4 was skipped due to goto_phase5, but still need aggregate fix
            s = _fix_aggregate_in_where(s)
            if DEBUG_WR: print(f"[PH4-FORCE] after agg-in-where fix: {s}")        
    # ---------------------------------------------------------------------------
    # Phase 5 — Pattern, REGEXP, and Predicate Normalization
    # ---------------------------------------------------------------------------
    # print("\n[PH5] start ----")  # ✅ ADD THIS
    # s = _guardrail_fix_like_regex_anchors(s, question)
    # print(f"[PH5] after fix_anchors: {s[:200]}")  # ✅ ADD THIS
    # s = _rewrite_simple_regexp_to_like(s, question)
    # s = _guardrail_fix_mysql_regexp_literals(s)
    # print(f"[PH5] end ---- {s}\n")  # ✅ ADD THIS
    # ---------------------------------------------------------------------------
    # Phase 6 — NULL / DISTINCT / Result Cardinality Policies
    # ---------------------------------------------------------------------------
    s = _expand_missing_null_to_null_or_empty(s, question)
    s = _ensure_reasonable_distinct(s, question)
    s = _strip_unneeded_count_distinct(s)
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
    select_match = re.search(r'\bSELECT\b(.*?)\bFROM\b', s, re.IGNORECASE | re.DOTALL)
    if select_match:
        # ... existing Phase 8 code ...
        pass
    
    # Phase 8b is DISABLED for thesis benchmark
    
    # ---------------------------------------------------------------------------
    # Phase 8c — Standardize Table Aliases to Match _ALIAS_OVERRIDES
    # ---------------------------------------------------------------------------
    alias_standards = {tbl.lower(): alias for tbl, alias in _ALIAS_OVERRIDES.items()}
    def find_subquery_table(sql_fragment):
        """Extract first table name after FROM in a SQL fragment."""
        match = re.search(r'\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)\b', sql_fragment, re.IGNORECASE)
        return match.group(1).lower() if match else None

    replacements = {}

    # Find all subquery patterns: FROM (...) AS alias
    subquery_start_pattern = re.compile(r'\b(FROM|JOIN)\s+\(', re.IGNORECASE)

    for match in subquery_start_pattern.finditer(s):
        start_pos = match.end() - 1
        
        paren_count = 1
        i = start_pos + 1
        while i < len(s) and paren_count > 0:
            if s[i] == '(':
                paren_count += 1
            elif s[i] == ')':
                paren_count -= 1
            i += 1
        
        if paren_count == 0:
            subquery_content = s[start_pos+1:i-1]
            alias_match = re.match(r'\s+(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\b', s[i:], re.IGNORECASE)
            
            if alias_match:
                current_alias = alias_match.group(1)
                
                # Skip if alias is a SQL keyword
                if current_alias.lower() in _SQL_KEYWORDS:
                    continue
                
                inner_table = find_subquery_table(subquery_content)
                
                if inner_table and inner_table in alias_standards:
                    standard_alias = alias_standards[inner_table]
                    if current_alias.lower() != standard_alias.lower():
                        replacements[current_alias] = standard_alias

    # Handle regular table aliases (non-subquery)
    table_alias_pattern = re.compile(
        r'\b(FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\b',
        re.IGNORECASE
    )

    for match in table_alias_pattern.finditer(s):
        table = match.group(2).lower()
        current_alias = match.group(3)
        
        # Skip if alias is a SQL keyword  ← TAMBAHKAN INI
        if current_alias.lower() in _SQL_KEYWORDS:
            continue
        
        if table in alias_standards:
            standard_alias = alias_standards[table]
            if current_alias.lower() != standard_alias.lower():
                replacements[current_alias] = standard_alias

    if replacements:
        for old_alias, new_alias in replacements.items():
            # Replace qualified references: old.col → new.col
            s = re.sub(
                rf'\b{re.escape(old_alias)}\.',
                f'{new_alias}.',
                s,
                flags=re.IGNORECASE
            )
            
            # Replace subquery alias: ) AS old → ) AS new
            s = re.sub(
                rf'\)\s+(?:AS\s+)?{re.escape(old_alias)}\b',
                f') AS {new_alias}',
                s,
                flags=re.IGNORECASE
            )
            
            # Replace regular table alias: FROM/JOIN table old → FROM/JOIN table new
            s = re.sub(
                rf'\b(FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:AS\s+)?{re.escape(old_alias)}\b',
                rf'\1 \2 {new_alias}',
                s,
                flags=re.IGNORECASE
            )
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