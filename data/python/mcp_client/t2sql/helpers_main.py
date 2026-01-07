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
_parse_allowed_identifiers, _question_requires_regex, _question_requires_word_boundary_regex, has_join,
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
_force_restore_groupby_if_aggregate_without_groupby, _rewrite_winner_row_guardrail,
_rewrite_winner_row_aggregate_misuse, _rewrite_having_alias_to_expr, _winner_row_final_sanity,
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
def _identify_used_identifiers(sql: str) -> Dict[Optional[str], Set[str]]:
    """
    Return mapping:
      - qualifier -> set(columns)  e.g. {"p": {"amount","paymentDate"}}
      - None      -> set(unqualified tokens) e.g. {None: {"amount"}}

    Konservatif:
    - Strip string literals dari scanning.
    - Exclude keywords, table names, aliases (FROM/JOIN/derived/output).
    - Skip token yang terlihat seperti function call: token(...).
    """
    if not sql:
        return {}

    out: Dict[Optional[str], Set[str]] = {}
    # ---------------------------------------------------------------------------
    # 1) Hapus string literals/double quotes supaya token scan tidak ke-distract
    # ---------------------------------------------------------------------------
    code_only = re.sub(r"'(?:''|[^'])*'", "''", sql)
    code_only = re.sub(r'"(?:""|[^"])*"', '""', code_only)
    keywords = _SQL_KEYWORDS
    # ---------------------------------------------------------------------------
    # 2) Ambil table names yang muncul di FROM/JOIN (untuk exclude)
    #    Support schema.table dan backticks secara sederhana.
    # ---------------------------------------------------------------------------
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
    # ---------------------------------------------------------------------------
    # 3) table aliases from FROM/JOIN
    # ---------------------------------------------------------------------------
    a2t = _parse_from_join_aliases(sql)  # alias -> table
    exclude.update({a.lower() for a in a2t.keys() if a})
    # ---------------------------------------------------------------------------
    # 4) derived-table aliases: FROM ( ... ) AS t / JOIN ( ... ) AS t
    # ---------------------------------------------------------------------------
    for m in re.finditer(
        r"\b(?:from|join)\s*\([^)]*\)\s*(?:as\s+)?([A-Za-z_]\w*)\b",
        code_only,
        flags=re.I | re.S,
    ):
        exclude.add(m.group(1).lower())
    # ---------------------------------------------------------------------------
    # 5) output aliases: SELECT ... AS alias (hanya area SELECT ... FROM)
    # ---------------------------------------------------------------------------
    m_sel = re.search(r"\bselect\b(.*?)\bfrom\b", code_only, flags=re.I | re.S)
    if m_sel:
        select_part = m_sel.group(1)
        for m in re.finditer(r"\bas\s+([A-Za-z_]\w*)\b", select_part, flags=re.I):
            exclude.add(m.group(1).lower())
    # ---------------------------------------------------------------------------
    # 6) Collect qualified usages: alias.col / table.col
    # ---------------------------------------------------------------------------
    # Simpan dulu semua pattern qualifier.column
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
    # Remove qualified occurrences so they don't pollute unqualified scan
    code_no_qual = re.sub(
        r"\b[A-Za-z_][A-Za-z0-9_]*\s*\.\s*[A-Za-z_][A-Za-z0-9_]*\b",
        " ",
        code_only,
        flags=re.I,
    )
    # ---------------------------------------------------------------------------
    # 7) Unqualified token scan (konservatif)
    # ---------------------------------------------------------------------------
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
        # skip jika terlihat seperti function call: tok(...)
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
    sql = re.sub(r"\s+(?=(ORDER\s+BY|GROUP\s+BY|HAVING|LIMIT|WHERE)\b)", " ", sql, flags=re.I)
    sql = re.sub(
        r"([)\],;`])(?=(ORDER\s+BY|GROUP\s+BY|HAVING|LIMIT|WHERE)\b)",
        r"\1 ",
        sql,
        flags=re.I,
    )
    sql = re.sub(r"([A-Za-z0-9_\.])(?=(LIMIT|WHERE|HAVING)\b)", r"\1 ", sql)
    sql = re.sub(r"([A-Za-z0-9_\.])(?=ORDER\s+BY\b)", r"\1 ", sql)
    sql = re.sub(r"([A-Za-z0-9_\.])(?=GROUP\s+BY\b)", r"\1 ", sql)
    sql = re.sub(r"\s+(ORDER\s+BY|GROUP\s+BY|HAVING|LIMIT|WHERE)\b", r" \1", sql, flags=re.I)
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
            "kodecustomer": "customerNumber",
            "kode_pelanggan": "customerNumber",
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
# ====================== GENERATION PROMPT BUILDER =====================
# ======================================================================
def make_generation_prompt(ctx: str, question: str) -> str:
    return fr"""
You are a senior SQL assistant for MySQL 8.0.

Use ONLY tables/columns listed under ALLOWED IDENTIFIERS in Context.
If Context lists aliases, use the canonical column name.

Customer naming preference (OUTPUT ONLY):
When selecting customer attributes, prefer customerName over customerNumber unless the user explicitly asks for an id/number/code.
This preference does NOT determine the anchor (driving) table.

ANCHOR (driving table) rule:
1) Row-preservation override:
   If the user asks to include entities with zero related rows (e.g., "including customers with no payments", "never ordered"),
   anchor the entity table (e.g., customers/products) and use LEFT JOIN or NOT EXISTS as appropriate.
2) Otherwise choose anchor by primary measure/filter (use reranked Context evidence when possible):
   - payments if the main filter/measure is payments.paymentDate or payments.amount
   - orderdetails if computing sales/revenue or using orderdetails.quantityOrdered/priceEach
   - orders if filtering by orders.orderDate/status/shippedDate/requiredDate
3) "For each <entity>" implies GROUP BY that entity, not necessarily anchoring that entity.

JOIN rule:
- If all needed columns exist in ONE allowed table: do NOT JOIN.
- JOIN only when needed to fetch columns from multiple tables.

JOIN TYPE DEFAULT (IMPORTANT):
- Use INNER JOIN by default.
- Use LEFT JOIN ONLY when the question explicitly requires keeping entities with zero related rows
  (e.g., "including customers with no payments", "all customers even if no payments", "never ordered/paid", "termasuk yang tidak punya").
- Selecting customerName / grouping by customerName does NOT mean you should keep all customers.
  Anchor is determined by the primary filter/measure (see ANCHOR rule).

LIKE vs REGEXP:
- Use LIKE only for simple wildcard intent: starts with -> 'X%', ends with -> '%X', contains -> '%X%'. Do NOT put ^ or $ inside LIKE.
- Use REGEXP only if: (a) word-boundary intent (word starts with X), (b) strict structure/format (exact length/digits), or (c) user explicitly requests regex.
  - Word-boundary REGEXP must use: (^|[^[:alnum:]_]) as left boundary and [[:alnum:]_]* after prefix. Do NOT use \\b or \\w.
  - Strict-structure REGEXP must be anchored with ^ and $.

DATE LABEL RULE:
- If user asks month name: SELECT DATE_FORMAT(<date_col>, '%M') AS month_name.
  GROUP BY MONTH(<date_col>), DATE_FORMAT(<date_col>, '%M').
  ORDER BY MONTH(<date_col>) (or DESC if user asks latest->earliest).
  Do NOT output helper month number unless asked.
- If user asks day name (weekday): SELECT DATE_FORMAT(<date_col>, '%W') AS day_name.
  GROUP BY DAYOFWEEK(<date_col>), DATE_FORMAT(<date_col>, '%W').
  ORDER BY DAYOFWEEK(<date_col>) (or DESC if asked).
- If user asks each day/per day: SELECT DATE(<date_col>) and GROUP/ORDER BY DATE(<date_col>).

WINNER-ROW RULE (rank per entity):
- If the question asks for largest/smallest per entity AND also asks another column from the same row
  (e.g., paymentDate of the largest payment), DO NOT use MAX/MIN alone.
- Use ROW_NUMBER() OVER (PARTITION BY <entity_id> ORDER BY <metric> DESC, <tie_breaker> ASC) and filter rn = 1.

Output:
Return EXACTLY one JSON object:
{{"sql":"<final MySQL query>","notes":"<brief rationale; for EACH JOIN, name the column it contributes>"}}
No Markdown. End SQL with a semicolon.
If NO JOINs, notes MUST be exactly: "No joins used."

Context:
{ctx}

Question:
{question}
""".strip()
# ======================================================================
# =========================== MAIN PIPELINE ============================
# ================ Postprocess SQL & Normalize Pipeline ================
# ======================================================================  
def _postprocess_sql(sql: str, question: str, context: str = "") -> str:
    DEBUG_WR = True
    if not sql:
        return sql
    s = sql.strip()
    
    # CRITICAL FIX: Normalize ALL major clause boundaries FIRST
    # Pattern: ensure space before WHERE/GROUP BY/ORDER BY/HAVING/LIMIT
    # This must handle: "col1,col2HAVING" and "col1 HAVING" cases
    s = re.sub(
    r"(\b[A-Za-z_][A-Za-z0-9_]*)(WHERE|GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT)\b",
    r"\1 \2",
    s,
    flags=re.I)   
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
    s = _canonicalize_table_names(s, context)
    s = ensure_join_aliases(s)
    # ---------------------------------------------------------------------------
    # Phase 1 — JOIN Semantics & Structural Rewrites (FINAL)
    # Decide JOIN necessity and semantics before touching aliases
    # ---------------------------------------------------------------------------
    s = _rewrite_left_join_to_inner_if_not_needed(s, question)
    s = _rewrite_left_join_to_inner_if_rhs_filtered(s, question)
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
    # Phase 4 — Winner-Row & Aggregate Guardrails
    # Ensure semantic correctness for ranking-style queries
    # ---------------------------------------------------------------------------
    before_phase4 = s

    if DEBUG_WR: print("\n[PH4] start ----")
    s = re.sub(
    r"(\b[A-Za-z_][A-Za-z0-9_]*)(WHERE|GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT)\b",
    r"\1 \2",
    s,
    flags=re.I)
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

        s = _rewrite_having_alias_to_expr(s)
        s = _winner_row_final_sanity(s, question)

        # fail-safe restore group by if pipeline broke it
        s = _force_restore_groupby_if_aggregate_without_groupby(before_phase4, s)

        if DEBUG_WR: print("[PH4] end ----", s, "\n")
    # ---------------------------------------------------------------------------
    # Phase 5 — Pattern, REGEXP, and Predicate Normalization
    # ---------------------------------------------------------------------------
    s = _guardrail_fix_like_regex_anchors(s, question)
    s = _rewrite_simple_regexp_to_like(s, question)
    s = _guardrail_fix_mysql_regexp_literals(s)
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