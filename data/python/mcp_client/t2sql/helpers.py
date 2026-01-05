# ============================================================
# ====================== helpers.py ==========================
# ============================================================
# Purpose:
# - Query classification & state-derived flags (qtype, requires_regex)
# - SQL whitelist parsing/validation (ALLOWED IDENTIFIERS)
# - Dense probe builder for retrieval
# - Generation prompt builder (LLM instruction)
# - SQL postprocessing pipeline (formatting, DISTINCT/LIMIT/ORDER, join pruning)
# - Alias repair using Context alias_map + classic fallback
# ============================================================

from __future__ import annotations

# --------------------------------------------------
# Standard library imports
# --------------------------------------------------
import re
import json
from typing import TYPE_CHECKING, Dict, List, Optional, Set
from collections import defaultdict

# --------------------------------------------------
# Local imports
# --------------------------------------------------
if TYPE_CHECKING:
    from .state import RAGState

# ======================================================================
# =========== QUERY CLASSIFICATION & STATE FLAGS (qtype/regex) =========
# ======================================================================
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
    
def classify_query(question: str) -> str:
    """
    Heuristik sederhana untuk klasifikasi tipe pertanyaan.
    Kembalikan string qtype yang dipakai di pipeline kamu.
    """
    q = (question or "").strip().lower()

    # aggregate / summary
    if any(k in q for k in ["count", "total", "sum", "average", "avg", "berapa banyak", "jumlah", "rata-rata"]):
        return "aggregate"

    # ranking / top-k
    if any(k in q for k in ["top", "highest", "lowest", "largest", "smallest", "terbesar", "terkecil", "paling"]):
        return "rank"

    # filter / lookup default
    return "lookup"

def _question_requires_word_boundary_regex(question: str) -> bool:
    q = (question or "").lower()

    # strong boundary intent signals
    boundary_phrases = [
        "contain a word that starts with",
        "contains a word that starts with",
        "a word that starts with",
        "word starts with",
        "kata yang diawali",
        "kata yang dimulai",
        "awal kata",
        "whole word",
        "kata utuh",
        "word boundary",
        "batas kata",
        "token",
    ]
    return any(p in q for p in boundary_phrases)

def _question_requires_regex(question: str) -> bool:
    q = (question or "").strip().lower()

    # explicit regex request
    if any(k in q for k in ["regex", "regexp", "regular expression"]):
        return True

    # word-boundary intent
    if _question_requires_word_boundary_regex(question):
        return True

    # strict structure / exact length / digits-only / mask/format
    if re.search(r"\bexactly\s+\d+\b", q):
        return True
    if re.search(r"\b\d+\s+(digits|digit|characters|chars|huruf|karakter)\b", q):
        return True
    if re.search(r"\bexactly\s+(one|two|three|four|five|six|seven|eight|nine|ten)\b", q):
        return True
    if re.search(r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+(digits|digit|characters|chars|huruf|karakter)\b", q):
        return True
    if any(k in q for k in ["only digits", "hanya angka", "numeric only", "format", "mask"]):
        return True

    # default: prefix/suffix/contains biasa => LIKE cukup
    return False

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

# ======================================================================
# ==================== SQL ALIAS EXTRACTION (FROM/JOIN) ================
# ======================================================================
_ALIAS_OVERRIDES = {
    "payments": "p",
    "employees": "e",
    "customers": "c",
    "orderdetails": "od",
    "orders": "o",
    "offices": "of",
    "products": "pr",
    "productlines": "pl",
}

_RESERVED_ALIASES = {
    "or", "and", "on", "as", "in", "is", "by", "from", "join", "where",
    "group", "order", "limit", "select", "left", "right", "inner", "outer",
    "having", "union", "cross", "natural"
}

def _split_outside_quotes(sql: str) -> List[str]:
    # reuse your existing approach (same regex style you already use)
    return re.split(r"('(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`[^`]*`)", sql)


def _alias_for_table(table: str, used: set[str]) -> str:
    """
    Deterministic alias:
    - use overrides first
    - fallback: 1-2 letters from table name, increasing until unique
    """
    t = (table or "").strip("`").split(".")[-1]  # drop schema if any
    tl = t.lower()

    base = _ALIAS_OVERRIDES.get(tl)
    if base and base not in used:
        used.add(base)
        return base

    # generic fallback
    # e.g. products -> pr, productlines -> pl, etc.
    # keep it short but avoid collisions
    for k in range(1, min(4, len(tl)) + 1):
        cand = tl[:k]
        if cand not in used:
            used.add(cand)
            return cand

    # last resort
    i = 2
    while f"{tl[0]}{i}" in used:
        i += 1
    cand = f"{tl[0]}{i}"
    used.add(cand)
    return cand


def ensure_join_aliases(sql: str) -> str:
    """
    Enforce: if query has JOIN, every FROM/JOIN table must have an alias.
    Also rewrite qualifiers: table.col -> alias.col (outside quotes).
    Safe-only: no CTE, no subqueries/derived tables.
    """
    if not sql:
        return sql

    # Only enforce if there is a join
    if not has_join(sql):
        return sql

    # Skip complex forms (keep it non-brittle)
    if re.match(r"^\s*with\b", sql, flags=re.I):
        return sql
    if re.search(r"\bfrom\s*\(", sql, flags=re.I) or re.search(r"\bjoin\s*\(", sql, flags=re.I):
        return sql

    # Find FROM/JOIN tables + aliases (if present)
    # Matches: FROM table [AS] alias
    #          JOIN table [AS] alias
    tj_pat = re.compile(
        r"(?P<kw>\bFROM\b|\bJOIN\b)\s+"
        r"(?P<table>`?[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?`?)"
        r"(?:\s+(?:AS\s+)?(?P<alias>[A-Za-z_][A-Za-z0-9_]*))?",
        flags=re.I,
    )

    # 1) First pass: collect existing aliases & decide missing aliases
    used_aliases: set[str] = set()
    table_to_alias: dict[str, str] = {}   # lower(table)->alias
    spans_to_patch: list[tuple[int, int, str]] = []

    for m in tj_pat.finditer(sql):
        kw = m.group("kw")
        table_raw = m.group("table")
        alias = m.group("alias")

        table_norm = table_raw.strip("`")
        table_key = table_norm.split(".")[-1].lower()

        if alias and alias.lower() in _RESERVED_ALIASES:
            alias = None

        if alias:
            used_aliases.add(alias)
            table_to_alias[table_key] = alias
        else:
            spans_to_patch.append((m.start(), m.end(), table_raw))

    # 2) Second pass: assign aliases for missing ones
    # We need stable alias assignment order: FROM first, then JOINs in appearance order.
    used_aliases = set(used_aliases)  # ensure mutable
    patches: list[tuple[int, int, str]] = []

    for start, end, table_raw in spans_to_patch:
        table_key = table_raw.strip("`").split(".")[-1].lower()
        if table_key in table_to_alias:
            continue
        alias = _alias_for_table(table_key, used_aliases)
        table_to_alias[table_key] = alias

        # Inject " <alias>" right after table token (no AS needed)
        # We patch by reconstructing the matched segment.
        seg = sql[start:end]
        # seg looks like: "FROM customers" or "JOIN payments"
        seg_patched = re.sub(
            rf"(\b(?:FROM|JOIN)\b\s+{re.escape(table_raw)})(?!\s+\bAS\b|\s+[A-Za-z_])",
            rf"\1 {alias}",
            seg,
            flags=re.I,
        )
        patches.append((start, end, seg_patched))

    # Apply patches from the back to keep indices valid
    out = sql
    for start, end, seg_patched in sorted(patches, key=lambda x: x[0], reverse=True):
        out = out[:start] + seg_patched + out[end:]

    # 3) Rewrite qualifiers table.col -> alias.col (outside quotes/backticks/strings)
    parts = _split_outside_quotes(out)
    for i in range(0, len(parts), 2):  # code segments only
        seg = parts[i]
        for tbl, al in table_to_alias.items():
            # rewrite schema.tbl.col and tbl.col forms conservatively:
            # - if already alias.col, leave it
            seg = re.sub(rf"\b{re.escape(tbl)}\s*\.\s*", f"{al}.", seg, flags=re.I)
        parts[i] = seg

    return "".join(parts)

def _parse_from_join_aliases(sql: str) -> Dict[str, str]:
    """
    FROM `products` pr  -> {"pr": "products"}
    JOIN db.orders `or` -> {"or": "orders"}
    """
    if not sql:
        return {}

    pat = re.compile(
        r"\b(?:from|join)\s+"
        r"(?P<table>`?[A-Za-z_][\w]*`?(?:\.`?[A-Za-z_][\w]*`?)?)"
        r"\s+(?:as\s+)?(?P<alias>[A-Za-z_]\w*)\b",
        flags=re.I,
    )

    out: Dict[str, str] = {}
    for m in pat.finditer(sql):
        table_raw = m.group("table")
        alias = m.group("alias")
        # normalize: drop schema and backticks
        if alias and alias.lower() in _RESERVED_ALIASES:
            continue
        tbl = table_raw.replace("`", "")
        tbl = tbl.split(".")[-1]
        out[alias] = tbl
        out[alias.lower()] = tbl  # convenience
    return out

# ======================================================================
# ============================ WHITELIST ===============================
# ======================================================================
def _parse_allowed_identifiers(ctx: str) -> Dict[str, Set[str]]:
    tables: Dict[str, Set[str]] = {}
    m = re.search(r"### ALLOWED IDENTIFIERS(.*?)(?:\n### |\Z)", ctx, flags=re.S)
    if not m:
        return tables
    block = m.group(1)
    for line in block.splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        parts = line[2:].split(":", 1)
        if len(parts) != 2:
            continue
        table = parts[0].strip().strip("`")
        cols = [c.strip().strip("`") for c in parts[1].split(",") if c.strip()]
        if table:
            tables.setdefault(table, set()).update(cols)
    return tables

def get_single_allowed_table(ctx: str) -> Optional[str]:
    """
    Jika ALLOWED IDENTIFIERS hanya berisi 1 tabel, return nama tabel itu.
    Kalau >1 atau 0 tabel, return None.
    """
    allowed = _parse_allowed_identifiers(ctx)
    if len(allowed) == 1:
        return next(iter(allowed.keys()))
    return None

def _extract_tables_from_sql(sql: str) -> Set[str]:
    tables: Set[str] = set()
    for m in re.finditer(r"\bfrom\s+([A-Za-z_][A-Za-z0-9_]*)", sql, flags=re.I):
        tables.add(m.group(1))
    for m in re.finditer(r"\bjoin\s+([A-Za-z_][A-Za-z0-9_]*)", sql, flags=re.I):
        tables.add(m.group(1))
    return tables

def _identify_used_identifiers(sql: str) -> Dict[Optional[str], Set[str]]:
    qualified = re.findall(
        r"`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\.\s*`?([A-Za-z_][A-Za-z0-9_]*)`?",
        sql,
    )
    out: Dict[Optional[str], Set[str]] = {}
    for t, c in qualified:
        out.setdefault(t, set()).add(c)

    table_names = _extract_tables_from_sql(sql)
    for t in table_names:
        out.setdefault(t, set())

    keywords = {
        "select","from","where","group","by","order","limit","join","left","right",
        "inner","on","and","or","not","as","distinct","count","sum","avg","min","max",
        "having","asc","desc","like","in","between","is","null","case","when","then",
        "else","end","regexp","rlike"
    }

    # Remove string literals so we don't treat regex contents as identifiers
    code_only = re.sub(r"'(?:''|[^'])*'", "''", sql)
    code_only = re.sub(r'"(?:""|[^"])*"', '""', code_only)

    for tok in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", code_only):
        if tok.lower() in keywords:
            continue
        if tok in table_names:
            continue
        out.setdefault(None, set()).add(tok)

    return out

def validate_identifiers(sql: str, ctx: str) -> List[str]:
    allowed = _parse_allowed_identifiers(ctx)
    used = _identify_used_identifiers(sql)
    errs: List[str] = []

    alias2table = _parse_from_join_aliases(sql)  # alias -> table (canonical)
    # build reverse: table names set (lower) to help mapping
    allowed_lower = {t.lower(): t for t in allowed.keys()}

    # Remap used qualifiers (aliases) to real tables
    remapped: Dict[Optional[str], Set[str]] = {}
    for q, cols in used.items():
        if q is None:
            remapped.setdefault(None, set()).update(cols)
            continue

        q_norm = q.strip("`")
        # if qualifier is alias -> table
        t = alias2table.get(q_norm) or alias2table.get(q_norm.lower())
        if t:
            # normalize to ALLOWED canonical casing if possible
            canon = allowed_lower.get(t.lower(), t)
            remapped.setdefault(canon, set()).update(cols)
        else:
            # qualifier might already be a table name
            canon = allowed_lower.get(q_norm.lower(), q_norm)
            remapped.setdefault(canon, set()).update(cols)

    used = remapped

    explicit_tables = {t for t in used.keys() if t is not None}

    single_table: Optional[str] = None
    if len(explicit_tables) == 1:
        single_table = next(iter(explicit_tables))
    elif len(explicit_tables) == 0 and len(allowed) == 1:
        single_table = next(iter(allowed.keys()))

    for tbl, cols in used.items():
        if tbl is None:
            continue
        if tbl not in allowed:
            errs.append(f"table `{tbl}` not in ALLOWED")
            continue
        bad = [c for c in cols if c not in allowed[tbl]]
        if bad:
            errs.append(f"{tbl} -> invalid columns: {', '.join(bad)}")

    if single_table:
        allowed_cols = allowed.get(single_table, set())
        for col in used.get(None, set()):
            if col not in allowed_cols:
                errs.append(f"{single_table} -> invalid column (unqualified): {col}")

    return errs

# ======================================================================
# =================== FRIENDLY NAME PREFERENCE RULES ===================
# ======================================================================
def _wants_name_only(question: str, entity_words: List[str]) -> bool:
    """
    Generic helper: True kalau user ngomongin entity (customer/product/...) 
    tapi tidak eksplisit minta id/number/code.
    """
    q = (question or "").lower()
    # buang "number of" supaya tidak dianggap id-like
    q = re.sub(r"\b(total\s+)?number\s+of\b", "", q)

    mention_entity = any(w in q for w in entity_words)
    mention_id_like = (
        re.search(r"\b(id|number|nomor|no|code|kode)\b", q)
        or any(f"{w} code" in q for w in entity_words)
    )
    return mention_entity and not mention_id_like

def _wants_customer_name_only(question: str) -> bool:
    # pelanggan/customer tapi tidak minta id/number/code
    return _wants_name_only(question, ["customer", "customers", "pelanggan"])

def _wants_product_name_only(question: str) -> bool:
    # produk/product tapi tidak minta id/number/code
    return _wants_name_only(question, ["product", "products", "produk"])

# ======================================================================
# =========================== DENSE PROBE ==============================
# ======================================================================
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

    # --------------------------------------------------
    # 1) Tentukan base tokens
    # --------------------------------------------------
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

    # --------------------------------------------------
    # 2) Entity-specific hints
    # --------------------------------------------------
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

    # --------------------------------------------------
    # 2b) Geo hints (lebih dinamis)
    # --------------------------------------------------
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

    # --------------------------------------------------
    # 3) Dedup + truncate
    # --------------------------------------------------
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

# ======================================================================
# ======================== CONTEXT PROMPT ==============================
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
# ===================== POSTPROCESS HEURISTIC ==========================
# ======================================================================
_JOIN_KW = r"(?:inner\s+join|left\s+join|right\s+join|full\s+join|join)"

def has_join(sql: str) -> bool:
    return re.search(rf"\b{_JOIN_KW}\b", sql, flags=re.I) is not None

def join_count(sql: str) -> int:
    if not sql:
        return 0
    return len(re.findall(r"\bjoin\b", sql, flags=re.I))

def has_non_inner_join(sql: str) -> bool:
    if not sql:
        return False
    return re.search(r"\b(left|right|full)\s+join\b", sql, flags=re.I) is not None

TRANSACTION_TABLES = {"payments", "orderdetails", "orders"}

_ANCHOR_SIGNALS = {
    "payments": {"amount", "paymentDate", "customerNumber"},
    "orderdetails": {"quantityOrdered", "priceEach", "orderLineNumber", "orderNumber", "productCode"},
    "orders": {"orderDate", "status", "shippedDate", "requiredDate", "customerNumber"},
}

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

def _rewrite_left_join_to_inner_if_rhs_filtered(sql: str, question: str) -> str:
    """
    Generic Fix 2:
    If a LEFT JOIN's RHS is filtered in WHERE, it is semantically an INNER JOIN.
    Convert LEFT JOIN -> JOIN, unless the question explicitly wants include-zero rows.
    Works with or without aliases.
    """
    if not sql:
        return sql
    if _question_wants_include_zero(question):
        return sql

    # Only handle the simplest, safest case: exactly one JOIN, and it's a LEFT JOIN.
    if join_count(sql) != 1:
        return sql
    if not re.search(r"\bleft\s+join\b", sql, flags=re.I):
        return sql

    # Capture: FROM <t1> [a1] LEFT JOIN <t2> [a2] ON ... (tail includes WHERE/etc.)
    pat = re.compile(
        r"""
        ^(?P<prefix>.*?)
        \bFROM\s+(?P<t1>`?[A-Za-z_][A-Za-z0-9_]*`?)
        (?:\s+(?:AS\s+)?(?P<a1>[A-Za-z_][A-Za-z0-9_]*))?
        \s+LEFT\s+JOIN\s+(?P<t2>`?[A-Za-z_][A-Za-z0-9_]*`?)
        (?:\s+(?:AS\s+)?(?P<a2>[A-Za-z_][A-Za-z0-9_]*))?
        \s+\bON\s+(?P<on>.*?)
        (?P<tail>\s+(?:WHERE|GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT)\b.*?|;?\s*)$
        """,
        flags=re.IGNORECASE | re.DOTALL | re.VERBOSE,
    )
    m = pat.match(sql.strip())
    if not m:
        return sql

    t2 = (m.group("t2") or "").strip().strip("`")
    a2 = (m.group("a2") or "").strip()
    tail = m.group("tail") or ""

    # Determine RHS qualifier candidates: alias if exists, else table name.
    rhs_qualifiers = []
    if a2:
        rhs_qualifiers.append(a2)
    if t2:
        rhs_qualifiers.append(t2)

    # Must have WHERE, and WHERE must reference RHS (alias/table). Otherwise keep LEFT JOIN.
    where_m = re.search(r"\bWHERE\b", tail, flags=re.I)
    if not where_m:
        return sql

    # Check RHS usage in WHERE (e.g., "p.paymentDate", "payments.paymentDate")
    rhs_referenced = any(
        re.search(rf"\b{re.escape(q)}\s*\.", tail, flags=re.I) for q in rhs_qualifiers
    )
    if not rhs_referenced:
        return sql

    # Safe rewrite: LEFT JOIN -> JOIN (results preserved due to WHERE filtering RHS)
    prefix = m.group("prefix")
    t1_raw = m.group("t1")
    a1 = m.group("a1")
    t2_raw = m.group("t2")
    on_clause = m.group("on").strip()

    from_part = f"FROM {t1_raw}" + (f" {a1}" if a1 else "")
    join_part = f" JOIN {t2_raw}" + (f" {a2}" if a2 else "")

    return f"{prefix}{from_part}{join_part} ON {on_clause}{tail}"

def _canonicalize_table_names(sql: str, ctx: str) -> str:
    """
    Rewrite nama table di FROM/JOIN agar sesuai persis dengan ALLOWED IDENTIFIERS (termasuk casing).
    Ini mengatasi kasus LLM menulis Orders padahal tabelnya orders.
    """
    if not sql or not ctx:
        return sql

    allowed = _parse_allowed_identifiers(ctx)
    if not allowed:
        return sql

    # lower(table) -> canonical table name dari ALLOWED
    canon_by_lower = {t.lower(): t for t in allowed.keys()}

    # Match FROM/JOIN <db.?>table <alias?>
    pat = re.compile(
        r"(\bFROM\s+|\bJOIN\s+)"
        r"((?:[A-Za-z_][A-Za-z0-9_]*\.)?[A-Za-z_][A-Za-z0-9_]*)"
        r"(\s+(?:AS\s+)?[A-Za-z_][A-Za-z0-9_]*\b)?",
        flags=re.IGNORECASE,
    )

    def repl(m: re.Match) -> str:
        kw = m.group(1)
        full = m.group(2)
        alias = m.group(3) or ""

        if "." in full:
            db, tbl = full.split(".", 1)
            canon = canon_by_lower.get(tbl.lower())
            if not canon:
                return m.group(0)
            return f"{kw}{db}.{canon}{alias}"

        canon = canon_by_lower.get(full.lower())
        if not canon:
            return m.group(0)
        return f"{kw}{canon}{alias}"

    return pat.sub(repl, sql)

def sanitize_notes(sql: str, notes: str) -> str:
    if not notes:
        return notes
    if not has_join(sql or ""):
        return "No joins used."
    return notes.strip()

def _should_use_backtick(identifier: str) -> bool:
    if not identifier: return False
    clean = identifier.strip('`"')
    if not clean: return False
    first_char = clean[0]
    if first_char.isupper() or first_char.isdigit():
        return True
    if not re.match(r'^[a-z_][a-z0-9_]*$', clean, re.I):
        return True
    return False

def _smart_backtick(identifier: str) -> str:
    clean = identifier.strip('`"')
    return f"`{clean}`" if _should_use_backtick(clean) else clean

def _split_qualified(name: str) -> List[str]:
    return [p.strip('`" ') for p in name.split(".") if p.strip("\"` ")]

def _fix_bad_backticks(sql: str) -> str:
    def repl_fqn(m: re.Match) -> str:
        inside = m.group(1)
        if "." not in inside:
            return _smart_backtick(inside)
        parts = _split_qualified(inside)
        if len(parts) == 0:
            return inside
        return ".".join(_smart_backtick(p) for p in parts)
    sql = re.sub(r"`([^`]+?\.[^`]+?)`", repl_fqn, sql)
    sql = re.sub(r"(\b[A-Za-z_][\w]*)\.`([^`]+?)`",
                 lambda m: f"{_smart_backtick(m.group(1))}."
                           + ".".join(_smart_backtick(p) for p in _split_qualified(m.group(2))), sql)
    sql = re.sub(r"`([^`]+?)`\.(\b[A-Za-z_][\w]*)",
                 lambda m: ".".join(_smart_backtick(p) for p in _split_qualified(m.group(1)))
                           + f".{_smart_backtick(m.group(2))}", sql)
    sql = re.sub(r'`([^`]+?)`(?!\s*\.)', lambda m: _smart_backtick(m.group(1)), sql, flags=re.I)
    return sql

def _explicit_limit_from_question(question: str) -> Optional[int]:
    q = (question or "").lower()
    m = re.search(r"\blimit\s+(\d{1,4})\b", q)
    if m: return max(1, min(int(m.group(1)), 1_000_000))
    m = re.search(r"\btop\s+(\d{1,4})\b", q)
    if m: return max(1, min(int(m.group(1)), 1_000_000))
    m = re.search(r"\b(show|first|tampilkan|ambil)\s+(\d{1,4})\b", q)
    if m: return max(1, min(int(m.group(2)), 1_000_000))
    return None

def _maybe_add_limit(question: str, sql: str) -> str:
    if re.search(r"\blimit\s+\d+\b", sql, flags=re.I):
        return sql
    n = _explicit_limit_from_question(question)
    if n is None:
        return sql
    s = sql.rstrip().rstrip(";")
    return s + f" LIMIT {n};"

def _wants_distinct_from_question(question: str) -> bool:
    q = (question or "").lower()
    hard_signals = [
        "distinct",
        "no duplicate",
        "without duplicates",
        "tanpa duplikat",
        "tanpa duplicate",
        "hanya yang unik",
        "dedup",
        "de-dup",
    ]
    if any(s in q for s in hard_signals):
        return True
    if "unique" in q and ("rows" in q or "records" in q or "result" in q):
        return True
    return False

def _has_distinct(sql: str) -> bool:
    return bool(re.search(r"^\s*select\s+distinct\b", sql, flags=re.I))

def _add_distinct(sql: str) -> str:
    if _has_distinct(sql):
        return sql
    return re.sub(r"^\s*select\s+", "SELECT DISTINCT ", sql, flags=re.I)

def _remove_distinct(sql: str) -> str:
    return re.sub(r"^\s*select\s+distinct\s+", "SELECT ", sql, flags=re.I)

def _ensure_reasonable_distinct(sql: str, question: str) -> str:
    if not sql:
        return sql
    if _wants_distinct_from_question(question):
        return _add_distinct(sql)
    return _remove_distinct(sql)

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

def _collapse_spaces_outside_quotes(s: str) -> str:
    parts = re.split(r"('(?:''|[^'])*')", s)
    for i in range(len(parts)):
        if i % 2 == 0:
            parts[i] = re.sub(r"[ \t]{2,}", " ", parts[i])
    return "".join(parts)

def _extract_order_by_clause(sql: str) -> Optional[re.Match]:
    return re.search(r"\border\s+by\s+(.*?)(?=(\blimit\b|;|$))", sql, flags=re.I | re.S)

def _remove_order_by_clause(sql: str) -> str:
    return re.sub(r"\s+order\s+by\s+.*?(?=(\blimit\b|;|$))", "", sql, flags=re.I | re.S)

def _question_requests_ordering(question: str) -> bool:
    q = (question or "").lower()
    keywords = [
        "order by", "sort", "sorted", "ascending", "descending", "asc", "desc",
        "alphabetical", "alphabetically", "latest", "newest", "oldest",
        "highest", "lowest", "top", "bottom", "most expensive", "cheapest",
        "urut", "urutkan", "diurutkan", "menaik", "menurun",
        "terbesar", "terkecil", "tertinggi", "terendah", "terbaru", "terlama",
        "termahal", "termurah",
    ]
    return any(k in q for k in keywords)

def _strip_unneeded_count_distinct(sql: str) -> str:
    low = sql.lower()
    if "from customers" in low and "join orders" in low:
        a2t = _parse_from_join_aliases(sql)
        # cari alias untuk orders
        orders_alias = None
        for a, t in a2t.items():
            if t.lower() == "orders" and a.islower():  # ambil versi normal
                orders_alias = a
                break
        if orders_alias:
            sql = re.sub(
                rf"COUNT\s*\(\s*DISTINCT\s+{re.escape(orders_alias)}\.orderNumber\s*\)",
                f"COUNT({orders_alias}.orderNumber)",
                sql,
                flags=re.I,
            )
    return sql

def _should_prefer_like_over_regexp(question: str) -> bool:
    q = (question or "").lower()

    # kalau boundary atau strict structure → jangan rewrite
    if _question_requires_word_boundary_regex(question):
        return False
    if _question_requires_regex(question):
        return False

    # prefix/suffix/contains sederhana → LIKE ok
    simple_signals = [
        "starts with", "start with", "begins with", "begin with", "diawali", "awalan", "prefix",
        "ends with", "end with", "diakhiri", "suffix",
        "contains", "contain", "mengandung",
    ]
    return any(s in q for s in simple_signals)


def _rewrite_simple_regexp_to_like(sql: str, question: str) -> str:
    """
    Jika pertanyaan hanya butuh LIKE, tapi LLM menghasilkan:
      col REGEXP '^X'  -> col LIKE 'X%'
      col REGEXP 'X$'  -> col LIKE '%X'
    Hanya rewrite untuk pattern sederhana tanpa metachar kompleks.
    """
    if not sql or not _should_prefer_like_over_regexp(question):
        return sql

    # REGEXP '^196' di atas: replace("REGEXP","LIKE") belum terjadi karena r"\1" literal.
    # Jadi kita lakukan rewrite yang lebih eksplisit agar tidak ambigu:
    sql = re.sub(
        r"(\bWHERE\b.*?\b)((?:[A-Za-z_][A-Za-z0-9_]*\.)?[A-Za-z_][A-Za-z0-9_]*)\s+REGEXP\s+'\^([A-Za-z0-9 _-]+)'",
        r"\1\2 LIKE '\3%'",
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # REGEXP '196$' => LIKE '%196'
    sql = re.sub(
        r"(\bWHERE\b.*?\b)((?:[A-Za-z_][A-Za-z0-9_]*\.)?[A-Za-z_][A-Za-z0-9_]*)\s+REGEXP\s+'([A-Za-z0-9 _-]+)\$'",
        r"\1\2 LIKE '%\3'",
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    )

    return sql

def _guardrail_fix_like_regex_anchors(sql: str, question: str) -> str:
    """
    Fix kesalahan umum LLM: memasukkan anchor regex (^, $) ke LIKE.
    Contoh salah:
      LIKE '^196'   (harusnya starts with -> LIKE '196%')
      LIKE '196$'   (harusnya ends with   -> LIKE '%196')
    """
    if not sql:
        return sql

    q = (question or "").lower()

    starts = any(k in q for k in [
        "starts with", "start with",
        "begins with", "begin with",
        "diawali", "awalan", "awal", "prefix"
    ])
    ends = any(k in q for k in [
        "ends with", "end with",
        "diakhiri", "akhir", "suffix"
    ])
    contains = any(k in q for k in [
        "contains", "contain",
        "mengandung"
    ])

    # Rewrite LIKE '^X' -> LIKE 'X%'
    def repl_starts(m: re.Match) -> str:
        col = m.group(1)
        val = m.group(2)
        return f"{col}LIKE '{val}%'"

    # Rewrite LIKE 'X$' -> LIKE '%X'
    def repl_ends(m: re.Match) -> str:
        col = m.group(1)
        val = m.group(2)
        return f"{col}LIKE '%{val}'"

    if starts:
        sql = re.sub(
            r"(\b(?:[A-Za-z_][A-Za-z0-9_]*\.)?[A-Za-z_][A-Za-z0-9_]*\s+)LIKE\s+'\^([^']*)'",
            repl_starts,
            sql,
            flags=re.IGNORECASE,
        )

    if ends:
        sql = re.sub(
            r"(\b(?:[A-Za-z_][A-Za-z0-9_]*\.)?[A-Za-z_][A-Za-z0-9_]*\s+)LIKE\s+'([^']*)\$'",
            repl_ends,
            sql,
            flags=re.IGNORECASE,
        )

    if contains:
        sql = re.sub(
            r"(\b(?:[A-Za-z_][A-Za-z0-9_]*\.)?[A-Za-z_][A-Za-z0-9_]*\s+)LIKE\s+'\^([^']*)\$'",
            lambda m: f"{m.group(1)}LIKE '%{m.group(2)}%'",
            sql,
            flags=re.IGNORECASE,
        )
        sql = re.sub(
            r"(\b(?:[A-Za-z_][A-Za-z0-9_]*\.)?[A-Za-z_][A-Za-z0-9_]*\s+)LIKE\s+'\^([^']*)'",
            lambda m: f"{m.group(1)}LIKE '{m.group(2)}%'",
            sql,
            flags=re.IGNORECASE,
        )

    return sql

def _question_wants_include_zero(question: str) -> bool:
    q = (question or "").lower()
    signals = [
        "including customers with no", "customers with no", "with no payments", "never paid",
        "all customers", "even if no", "include those without",
        "termasuk", "meskipun tidak", "walaupun tidak", "yang tidak punya", "tanpa pembayaran",
        "pelanggan yang belum", "customer yang belum",
    ]
    return any(s in q for s in signals)

def _rewrite_regexp_literal_to_posix_if_needed(lit: str) -> str:
    r"""
    Input: isi string literal untuk REGEXP (tanpa quote luar), mis: r"\bCo\w*"
    Output: versi aman untuk MySQL:
      - \b atau \\b -> (^|[^[:alnum:]_])
      - \w atau \\w -> [[:alnum:]_]
      - (opsional) \d/\s
    Lalu double-backslash untuk sisa backslash single (mis: \. -> \\.)
    Catatan: ini mengasumsikan MySQL default (backslash escapes aktif).
    """
    if not lit:
        return lit

    # Trigger jika ada \b/\w baik single maupun double (\\b/\\w)
    # Pattern ini match: \b atau \\b (jumlah backslash 1 atau 2)
    has_b = re.search(r"(?:\\\\|\\)b", lit) is not None
    has_w = re.search(r"(?:\\\\|\\)w", lit) is not None

    # Kalau tidak ada indikasi, jangan sentuh (biar tidak mengubah regex valid)
    if not (has_b or has_w):
        return lit

    out = lit

    # 1) Rewrite boundary + wordchar (handle 1 atau 2 backslash)
    out = re.sub(r"(?:\\\\|\\)b", r"(^|[^[:alnum:]_])", out)
    out = re.sub(r"(?:\\\\|\\)w", r"[[:alnum:]_]", out)

    # Optional tambahan umum:
    # out = re.sub(r"(?:\\\\|\\)d", r"[0-9]", out)
    # out = re.sub(r"(?:\\\\|\\)s", r"[[:space:]]", out)

    # 2) Sisa backslash tunggal → jadikan double
    # Contoh: \. \- \( dll agar lolos parsing string MySQL dan sampai ke regex engine.
    # Jangan ubah yang sudah double.
    out = re.sub(r"(?<!\\)\\(?!\\)", r"\\\\", out)

    return out

def _guardrail_fix_mysql_regexp_literals(sql: str) -> str:
    r"""
    Repair literal untuk REGEXP/RLIKE yang mengandung \b/\w ala PCRE.
    Support single-quote literal (dan optional double-quote jika kamu aktifkan pattern kedua).
    Preserves doubled single-quotes inside literal.
    """
    if not sql:
        return sql

    # Single-quoted: REGEXP '...'
    pat_sq = re.compile(
        r"(\b(?:REGEXP|RLIKE)\b\s*)(N?')((?:''|[^'])*)(')",
        flags=re.IGNORECASE,
    )

    def _repl_sq(m: re.Match) -> str:
        prefix = m.group(1)
        qopen  = m.group(2)  # "'" or "N'"
        body   = m.group(3)  # content
        qclose = m.group(4)

        new_body = _rewrite_regexp_literal_to_posix_if_needed(body)
        return f"{prefix}{qopen}{new_body}{qclose}"

    out = pat_sq.sub(_repl_sq, sql)

    # (Opsional) Double-quoted: REGEXP "..."
    # Kalau kamu yakin LLM selalu pakai single quote, blok ini boleh dihapus.
    pat_dq = re.compile(
        r'(\b(?:REGEXP|RLIKE)\b\s*)(")((?:[^"\\]|\\.)*)(")',
        flags=re.IGNORECASE,
    )

    def _repl_dq(m: re.Match) -> str:
        prefix = m.group(1)
        qopen  = m.group(2)
        body   = m.group(3)
        qclose = m.group(4)

        new_body = _rewrite_regexp_literal_to_posix_if_needed(body)
        return f"{prefix}{qopen}{new_body}{qclose}"

    out = pat_dq.sub(_repl_dq, out)

    return out

# ======================================================================
# ========== SQL SEMANTIC TWEAKS (FRIENDLY NAME + JOIN PRUNING) ========
# ======================================================================
def _prefer_name_over_id_generic(
    sql: str,
    question: str,
    wants_name_fn,
    name_column: str,
    id_columns: List[str],
) -> str:
    """
    Ganti penggunaan kolom ID menjadi kolom nama yang lebih 'friendly'
    untuk SELECT & GROUP BY, kalau pertanyaan memang prefer 'name'.
    Contoh:
      - customerNumber -> customerName
      - productCode    -> productName
    """
    if not wants_name_fn(question):
        return sql

    # contoh: "customerNumber|customer_id|..."
    id_pattern = "|".join(re.escape(c) for c in id_columns)

    # -------------------------------------------------------------
    # CASE 1: SELECT alias.idCol → SELECT alias.nameCol AS nameCol
    # -------------------------------------------------------------
    sql = re.sub(
        rf"SELECT\s+([A-Za-z_][A-Za-z0-9_]*)\.({id_pattern})(\s+AS\s+\2)?",
        rf"SELECT \1.{name_column} AS {name_column}",
        sql,
        flags=re.I,
    )
    # -------------------------------------------------------------
    # CASE 2: SELECT alias.idCol, ... → SELECT alias.nameCol, .....
    # -------------------------------------------------------------
    sql = re.sub(
        rf"SELECT\s+([A-Za-z_][A-Za-z0-9_]*)\.({id_pattern})\s*,",
        rf"SELECT \1.{name_column},",
        sql,
        flags=re.I,
    )
    # -------------------------------------------------------------
    # CASE 3: GROUP BY alias.idCol → GROUP BY alias.nameCol
    # -------------------------------------------------------------
    sql = re.sub(
        rf"GROUP BY\s+([A-Za-z_][A-Za-z0-9_]*)\.({id_pattern})\b",
        rf"GROUP BY \1.{name_column}",
        sql,
        flags=re.I,
    )

    return sql

def _prefer_friendly_names(sql: str, question: str) -> str:
    """
    DRY: apply rule yang sama untuk beberapa entity sekaligus.
    - customers: customerNumber -> customerName
    - products : productCode    -> productName
    (hanya kalau user tidak minta id/number/code)
    """
    # customers
    sql = _prefer_name_over_id_generic(
        sql=sql,
        question=question,
        wants_name_fn=_wants_customer_name_only,
        name_column="customerName",
        id_columns=["customerNumber"],
    )

    # products
    sql = _prefer_name_over_id_generic(
        sql=sql,
        question=question,
        wants_name_fn=_wants_product_name_only,
        name_column="productName",
        id_columns=["productCode"],
    )

    return sql

def _collect_alias_column_usage(sql: str) -> Dict[str, set]:
    """
    Kembalikan mapping alias -> set kolom yang dipakai di query,
    berdasarkan pola alias.column (tidak bedakan posisi SELECT/WHERE/etc.).
    """
    usage: Dict[str, set] = defaultdict(set)
    for m in re.finditer(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)",
        sql,
        flags=re.IGNORECASE,
    ):
        alias, col = m.group(1), m.group(2)
        usage[alias].add(col)
    return usage

def _question_wants_existence(question: str) -> bool:
    q = (question or "").lower()

    eng = [
        "who have", "who has", "that have", "that has",
        "with at least one", "having at least one",
        "have at least one", "has at least one",
        "with any", "have any", "has any",
    ]

    return any(p in q for p in eng)


def _question_wants_anti_existence(question: str) -> bool:
    q = (question or "").lower()

    eng = [
        "with no", "without any", "who have no", "who has no",
        "that have no", "that has no",
        "never", "have never", "has never",
        "no orders", "no payments",
    ]

    return any(p in q for p in eng)

def _rewrite_inner_join_existence_to_exists(sql: str, question: str) -> str:
    if not sql:
        return sql
    if join_count(sql) != 1:
        return sql
    if re.search(r"\b(left|right|full)\s+join\b", sql, flags=re.I):
        return sql
    if _question_wants_anti_existence(question):
        return sql
    if not _question_wants_existence(question):
        return sql

    s0 = sql.strip()

    pat = re.compile(
        r"""
        ^(?P<prefix>.*?)
        \bFROM\s+(?P<t1>`?[A-Za-z_][A-Za-z0-9_]*`?)\s+(?:AS\s+)?(?P<a1>[A-Za-z_][A-Za-z0-9_]*)
        \s+\b(?:INNER\s+)?JOIN\s+(?P<t2>`?[A-Za-z_][A-Za-z0-9_]*`?)\s+(?:AS\s+)?(?P<a2>[A-Za-z_][A-Za-z0-9_]*)
        \s+\bON\s+(?P<on>.*?)
        (?P<tail>\s+(?:WHERE|GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT)\b.*?|;?\s*)$
        """,
        flags=re.I | re.S | re.VERBOSE,
    )
    m = pat.match(s0)
    if not m:
        return sql

    prefix = m.group("prefix")
    t1, a1 = m.group("t1"), m.group("a1")
    t2, a2 = m.group("t2"), m.group("a2")
    on_clause = (m.group("on") or "").strip()
    tail = m.group("tail") or ""

    # RHS alias must not be used outside ON (mask ON on s0, not sql)
    start_on = m.start("on")
    end_on = m.end("on")
    masked = s0[:start_on] + (" " * (end_on - start_on)) + s0[end_on:]
    if re.search(rf"\b{re.escape(a2)}\s*\.", masked, flags=re.I):
        return sql

    exists_expr = f"EXISTS (SELECT 1 FROM {t2} {a2} WHERE {on_clause})"

    if re.search(r"\bWHERE\b", tail, flags=re.I):
        tail2 = re.sub(r"\bWHERE\b", f"WHERE {exists_expr} AND ", tail, count=1, flags=re.I)
        return f"{prefix}FROM {t1} {a1}{tail2}"
    else:
        return f"{prefix}FROM {t1} {a1} WHERE {exists_expr}{tail}"

def _split_where_and_rest(tail: str) -> tuple[Optional[str], str]:
    """
    tail mulai dari WHERE/GROUP/ORDER/HAVING/LIMIT atau kosong.
    Return: (where_body_or_None, rest_after_where)
    """
    if not tail:
        return None, ""
    m = re.search(r"\bWHERE\b\s+(.*)$", tail, flags=re.I | re.S)
    if not m:
        return None, tail

    after_where = m.group(1)

    # potong saat ketemu clause berikutnya (GROUP BY/ORDER BY/HAVING/LIMIT)
    m2 = re.search(r"\b(GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT)\b", after_where, flags=re.I)
    if not m2:
        return after_where.strip(), ""

    where_body = after_where[:m2.start()].strip()
    rest = after_where[m2.start():]  # include keyword
    return where_body, " " + rest.strip()


def _split_top_level_and(expr: str) -> List[str]:
    """
    Split expr by AND di level top (tidak di dalam tanda kutip / kurung).
    Kalau ada OR top-level, kita abort di caller (biar safe).
    """
    if not expr:
        return []

    s = expr.strip()
    parts = []
    buf = []
    depth = 0
    in_sq = False
    in_dq = False
    i = 0
    while i < len(s):
        ch = s[i]

        # handle quotes
        if ch == "'" and not in_dq:
            # handle escaped '' (double single quote)
            if in_sq and i + 1 < len(s) and s[i + 1] == "'":
                buf.append("''")
                i += 2
                continue
            in_sq = not in_sq
            buf.append(ch)
            i += 1
            continue
        if ch == '"' and not in_sq:
            in_dq = not in_dq
            buf.append(ch)
            i += 1
            continue

        if in_sq or in_dq:
            buf.append(ch)
            i += 1
            continue

        # parentheses nesting
        if ch == "(":
            depth += 1
            buf.append(ch)
            i += 1
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            buf.append(ch)
            i += 1
            continue

        # detect AND at top-level
        if depth == 0 and s[i:i+3].lower() == "and":
            # boundary check
            left_ok = (i == 0) or not s[i-1].isalnum()
            right_ok = (i+3 >= len(s)) or not s[i+3].isalnum()
            if left_ok and right_ok:
                parts.append("".join(buf).strip())
                buf = []
                i += 3
                continue

        buf.append(ch)
        i += 1

    last = "".join(buf).strip()
    if last:
        parts.append(last)
    return [p for p in parts if p]


def _has_top_level_or(expr: str) -> bool:
    # konservatif: jika ada OR di luar quote/paren, kita tidak rewrite
    s = expr.strip()
    depth = 0
    in_sq = False
    in_dq = False
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "'" and not in_dq:
            if in_sq and i + 1 < len(s) and s[i + 1] == "'":
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
        if depth == 0 and s[i:i+2].lower() == "or":
            left_ok = (i == 0) or not s[i-1].isalnum()
            right_ok = (i+2 >= len(s)) or not s[i+2].isalnum()
            if left_ok and right_ok:
                return True
        i += 1
    return False


def _rewrite_inner_join_filter_to_exists(sql: str) -> str:
    """
    Rewrite filter-only INNER JOIN -> EXISTS.
    Syarat aman:
    - exactly 1 JOIN
    - join inner/join (bukan LEFT/RIGHT/FULL)
    - RHS alias tidak dipakai di SELECT/GROUP/ORDER/HAVING (boleh dipakai di WHERE)
    - WHERE tidak punya top-level OR (hanya AND)
    - ada minimal 1 predicate RHS-only di WHERE yang bisa dipindah
    """
    if not sql:
        return sql
    if join_count(sql) != 1:
        return sql
    if re.search(r"\b(left|right|full)\s+join\b", sql, flags=re.I):
        return sql

    s0 = sql.strip()

    pat = re.compile(
        r"""
        ^(?P<prefix_select>.*?)
        \bFROM\s+(?P<t1>`?[A-Za-z_][A-Za-z0-9_]*`?)\s+(?:AS\s+)?(?P<a1>[A-Za-z_][A-Za-z0-9_]*)
        \s+\b(?:INNER\s+)?JOIN\s+(?P<t2>`?[A-Za-z_][A-Za-z0-9_]*`?)\s+(?:AS\s+)?(?P<a2>[A-Za-z_][A-Za-z0-9_]*)
        \s+\bON\s+(?P<on>.*?)
        (?P<tail>\s+(?:WHERE|GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT)\b.*?|;?\s*)$
        """,
        flags=re.I | re.S | re.VERBOSE,
    )
    m = pat.match(s0)
    if not m:
        return sql

    prefix_select = m.group("prefix_select")
    t1, a1 = m.group("t1"), m.group("a1")
    t2, a2 = m.group("t2"), m.group("a2")
    on_clause = (m.group("on") or "").strip()
    tail = m.group("tail") or ""

    where_body, rest_after_where = _split_where_and_rest(tail)

    # kalau tidak ada WHERE, berarti bukan filter-only join -> skip
    if not where_body:
        return sql

    # jangan rewrite kalau ada OR top-level (biar tidak salah distribusi)
    if _has_top_level_or(where_body):
        return sql

    # Pastikan RHS alias tidak dipakai di SELECT/GROUP/ORDER/HAVING
    # Cek bagian SELECT (prefix_select) dan rest_after_where
    if re.search(rf"\b{re.escape(a2)}\s*\.", prefix_select, flags=re.I):
        return sql
    if re.search(rf"\b{re.escape(a2)}\s*\.", rest_after_where, flags=re.I):
        return sql

    # Ambil conjunct AND top-level
    conjuncts = _split_top_level_and(where_body)

    rhs_only = []
    keep = []
    for c in conjuncts:
        c0 = c.strip()
        has_rhs = re.search(rf"\b{re.escape(a2)}\s*\.", c0, flags=re.I) is not None
        has_lhs = re.search(rf"\b{re.escape(a1)}\s*\.", c0, flags=re.I) is not None

        # rhs-only: refer RHS, tidak refer LHS, dan bukan subquery SELECT (konservatif)
        if has_rhs and (not has_lhs) and (re.search(r"\bselect\b", c0, flags=re.I) is None):
            rhs_only.append(c0)
        else:
            keep.append(c0)

    if not rhs_only:
        return sql

    rhs_filters = " AND ".join(rhs_only)
    on_has_rhs_filter = _on_clause_has_rhs_filter(on_clause, a2)

    # Kalau tidak ada rhs_only dari WHERE, tapi ON sudah filter-like, tetap boleh rewrite
    if (not rhs_only) and (not on_has_rhs_filter):
        return sql

    exists_where = on_clause
    if rhs_filters:
        exists_where = f"{on_clause} AND {rhs_filters}"
        
    exists_expr = f"EXISTS (SELECT 1 FROM {t2} {a2} WHERE {on_clause} AND {rhs_filters})"

    new_where_parts = []
    if keep:
        new_where_parts.append(" AND ".join(keep))
    new_where_parts.append(exists_expr)

    new_where = " WHERE " + " AND ".join(new_where_parts)

    return f"{prefix_select}FROM {t1} {a1}{new_where}{(' ' + rest_after_where.strip()) if rest_after_where.strip() else ''}"

def _strip_string_literals(s: str) -> str:
    s = re.sub(r"'(?:''|[^'])*'", "''", s)
    s = re.sub(r'"(?:""|[^"])*"', '""', s)
    return s

def _on_clause_has_rhs_filter(on_clause: str, rhs_alias: str) -> bool:
    if not on_clause or not rhs_alias:
        return False

    oc = _strip_string_literals(on_clause)

    # 1) operator non-equality yang jelas filter-like
    if re.search(rf"\b{re.escape(rhs_alias)}\.\w+\s*(>=|<=|<>|!=|>|<)\s*", oc, flags=re.I):
        return True
    if re.search(rf"\b{re.escape(rhs_alias)}\.\w+\s+\b(BETWEEN|LIKE|IN|REGEXP|RLIKE)\b", oc, flags=re.I):
        return True
    if re.search(rf"\b{re.escape(rhs_alias)}\.\w+\s+IS\s+(NOT\s+)?NULL\b", oc, flags=re.I):
        return True

    # 2) equality ke literal/angka/param/fungsi juga filter-like
    # rhs.col = 'X' / 123 / (subexpr) / func(...)
    if re.search(rf"\b{re.escape(rhs_alias)}\.\w+\s*=\s*('|\d|\(|\bNULL\b)", oc, flags=re.I):
        return True
    if re.search(rf"\b{re.escape(rhs_alias)}\.\w+\s*=\s*\w+\s*\(", oc, flags=re.I):
        return True

    return False

def _drop_redundant_inner_join_same_key(sql: str) -> str:
    """
    Drop INNER JOIN jika:
    - hanya 1 JOIN
    - ON adalah equality join-key sederhana: a1.k = a2.k (nama kolom sama)
    - RHS alias/table tidak dipakai untuk kolom lain di SELECT/WHERE/GROUP/ORDER/HAVING (kecuali join key)
    Lalu rewrite: rhs.k -> lhs.k di luar ON, dan buang JOIN.
    """
    if not sql or join_count(sql) != 1:
        return sql
    if has_non_inner_join(sql):
        return sql

    pat = re.compile(
        r"""
        ^(?P<prefix>.*?)
        \bFROM\s+(?P<t1>`?[A-Za-z_][A-Za-z0-9_]*`?)
        (?:\s+(?:AS\s+)?(?P<a1>[A-Za-z_][A-Za-z0-9_]*))?
        \s+\b(?:INNER\s+)?JOIN\s+(?P<t2>`?[A-Za-z_][A-Za-z0-9_]*`?)
        (?:\s+(?:AS\s+)?(?P<a2>[A-Za-z_][A-Za-z0-9_]*))?
        \s+\bON\s+(?P<on>.*?)
        (?P<tail>\s+(?:WHERE|GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT)\b.*?|;?\s*)$
        """,
        flags=re.IGNORECASE | re.DOTALL | re.VERBOSE,
    )
    m = pat.match(sql.strip())
    if not m:
        return sql

    a1 = (m.group("a1") or "").strip()
    a2 = (m.group("a2") or "").strip()
    t1 = (m.group("t1") or "").strip().strip("`")
    t2 = (m.group("t2") or "").strip().strip("`")
    on_txt = (m.group("on") or "").strip()
    tail = m.group("tail") or ""

    lhs = a1 or t1
    rhs = a2 or t2

    # ON harus bentuk sederhana: lhs.k = rhs.k (atau kebalik), dengan k sama
    on_clean = _strip_string_literals(on_txt)
    on_m = re.search(
        rf"\b{re.escape(lhs)}\.(?P<k1>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*{re.escape(rhs)}\.(?P<k2>[A-Za-z_][A-Za-z0-9_]*)\b"
        rf"|\b{re.escape(rhs)}\.(?P<k3>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*{re.escape(lhs)}\.(?P<k4>[A-Za-z_][A-Za-z0-9_]*)\b",
        on_clean,
        flags=re.I,
    )
    if not on_m:
        return sql

    k_left = (on_m.group("k1") or on_m.group("k4") or "")
    k_right = (on_m.group("k2") or on_m.group("k3") or "")
    if not k_left or not k_right:
        return sql

    # konservatif: hanya kalau nama kolom sama (case-insensitive)
    if k_left.lower() != k_right.lower():
        return sql

    join_key = k_left  # canonical from lhs side

    # Cari semua rhs.<col> di tail (dan select/prefix) kecuali join_key
    # Pakai sql full (di luar ON) dengan masking berbasis segment:
    sql_no_strings = _strip_string_literals(sql)
    # ambil bagian di luar ON untuk cek penggunaan RHS
    outside_on = (m.group("prefix") or "") + (m.group("tail") or "")
    outside_on = _strip_string_literals(outside_on)

    # Jika RHS dipakai untuk kolom lain selain join_key → tidak aman drop
    bad_rhs_use = re.search(
        rf"\b{re.escape(rhs)}\.(?!{re.escape(join_key)}\b)[A-Za-z_][A-Za-z0-9_]*\b",
        outside_on,
        flags=re.I,
    )
    if bad_rhs_use:
        return sql

    # Rewrite rhs.join_key -> lhs.join_key di luar ON
    def _rewrite_outside_on(text: str) -> str:
        parts = re.split(r"('(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`[^`]*`)", text)
        for i in range(0, len(parts), 2):
            parts[i] = re.sub(
                rf"\b{re.escape(rhs)}\s*\.\s*{re.escape(join_key)}\b",
                f"{lhs}.{join_key}",
                parts[i],
                flags=re.I,
            )
        return "".join(parts)

    prefix = (m.group("prefix") or "")
    t1_raw = m.group("t1")
    from_part = f"FROM {t1_raw}" + (f" {a1}" if a1 else "")
    new_tail = _rewrite_outside_on(tail)

    return f"{prefix}{from_part}{new_tail}"

def _prune_ornamental_joins_safe(sql: str, question: str = "") -> str:
    if not sql:
        return sql

    out = sql

    join_pat = re.compile(
        r"""
        (?P<full>
            \s+
            (?P<jtype>(?:INNER\s+JOIN|LEFT\s+JOIN|RIGHT\s+JOIN|FULL\s+JOIN|JOIN))
            \s+
            (?P<table>`?[A-Za-z_][A-Za-z0-9_]*`?)
            (?:\s+(?:AS\s+)?(?P<alias>[A-Za-z_][A-Za-z0-9_]*))?
            \s+\bON\b\s+(?P<on>.*?)
        )
        (?=
            \b(?:INNER\s+JOIN|LEFT\s+JOIN|RIGHT\s+JOIN|FULL\s+JOIN|JOIN)\b
            |\bWHERE\b|\bGROUP\s+BY\b|\bORDER\s+BY\b|\bHAVING\b|\bLIMIT\b
            |;|$
        )
        """,
        flags=re.IGNORECASE | re.DOTALL | re.VERBOSE,
    )

    # ...lanjutkan logic kamu yang existing,
    # tapi pakai m.group("table") bila alias None
    matches = list(join_pat.finditer(out))
    for m in reversed(matches):
        alias = m.group("alias") or (m.group("table") or "").strip("`")
        jtype = (m.group("jtype") or "").upper()
        on_txt = m.group("on") or ""

        start_on = m.start("on")
        end_on = m.end("on")
        masked = out[:start_on] + (" " * (end_on - start_on)) + out[end_on:]

        if re.search(rf"\b{re.escape(alias)}\s*\.", masked, flags=re.I):
            continue

        is_inner = ("LEFT" not in jtype and "RIGHT" not in jtype and "FULL" not in jtype)
        if is_inner:
            if _on_clause_has_rhs_filter(on_txt, alias):
                continue

        out = out[:m.start("full")] + " " + out[m.end("full"):]
    return out

# ======================================================================
# ===================== SQL NORMALIZATION PIPELINE =====================
# ======================================================================
def _strip_table_prefix_if_single_table(sql: str) -> str:
    """
    Hapus prefix table/alias (e.g., payments.amount -> amount) HANYA jika:
    - tidak ada JOIN
    - tidak ada CTE (WITH ...)
    - FROM hanya 1 table biasa (bukan subquery / derived table)
    Aman untuk kasus single-table lookup/rank sederhana.
    """
    if not sql:
        return sql

    # 1) Jangan sentuh kalau ada JOIN
    if has_join(sql):
        return sql

    # 2) Jangan sentuh kalau ada CTE
    if re.match(r"^\s*with\b", sql, flags=re.I):
        return sql

    # 3) Pastikan FROM table biasa (bukan FROM (SELECT ...))
    m_from = re.search(
        r"\bfrom\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s+(?:as\s+)?([A-Za-z_][A-Za-z0-9_]*))?\b",
        sql,
        flags=re.I,
    )
    if not m_from:
        return sql

    table = m_from.group(1)
    alias = m_from.group(2)

    # 4) Pastikan cuma 1 tabel (tidak ada join/extra from table lain)
    tables = _extract_tables_from_sql(sql)  # ini sudah ada di file kamu
    if len(tables) != 1:
        return sql

    # 5) Strip qualifier hanya untuk table/alias ini, dan hanya di code segment (bukan string/backtick)
    qualifiers = [table]
    if alias:
        qualifiers.append(alias)

    # split supaya tidak merusak string literal / quoted identifiers
    parts = re.split(r"('(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`[^`]*`)", sql)
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            out.append(part)  # quoted segment: biarkan
            continue

        seg = part
        for q in qualifiers:
            # hapus "payments." atau "p." (opsional ada whitespace)
            seg = re.sub(rf"\b{re.escape(q)}\s*\.\s*", "", seg, flags=re.I)
        out.append(seg)

    return "".join(out)

def _postprocess_sql(sql: str, question: str, context: str = "") -> str:
    if not sql:
        return sql
    s = sql.strip()

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
    if not s.endswith(";"):
        s += ";"

    s = _fix_bad_backticks(s)
    s = _canonicalize_table_names(s, context)
    s = ensure_join_aliases(s)
    s = _rewrite_left_join_to_inner_if_rhs_filtered(s, question)
    s = _rewrite_inner_join_filter_to_exists(s)
    s = _rewrite_inner_join_existence_to_exists(s, question)
    s = _drop_redundant_inner_join_same_key(s)
    s = _prune_ornamental_joins_safe(s, question)
    

    s = _guardrail_fix_like_regex_anchors(s, question)
    s = _rewrite_simple_regexp_to_like(s, question)
    s = _guardrail_fix_mysql_regexp_literals(s)
    s = _ensure_reasonable_distinct(s, question)
    s = _prefer_friendly_names(s, question)
    ob_match = _extract_order_by_clause(s)
    if ob_match and not _question_requests_ordering(question):
        s = _remove_order_by_clause(s)

    # dynamic join pruning
    # strip qualifier kalau benar-benar single-table tanpa JOIN/CTE
    s = _strip_table_prefix_if_single_table(s)
    s = _maybe_add_limit(question, s)
    s = _fix_missing_space_before_clauses(s)
    s = _collapse_spaces_outside_quotes(s)
    s = _strip_unneeded_count_distinct(s)
    return s

def normalize_sql_pipeline(raw_sql: str, question: str, ctx: str, ranked_cols=None) -> str:
    """
    Pipeline standar untuk normalisasi SQL:
    - postprocess (normalize keyword, limit, distinct, dsb)
    - repair alias berdasarkan alias_map
    - postprocess lagi setelah alias diganti
    """
    if not raw_sql:
        return raw_sql

    s = _postprocess_sql(raw_sql, question, ctx)

    if ranked_cols:
        pref = pick_preferred_anchor_from_ranked_cols(ranked_cols, top_n=12)
        s = normalize_anchor_for_simple_inner_join(s, pref)

    s = _repair_with_aliases(s, ctx)
    s = _postprocess_sql(s, question, ctx)
    return s

# ======================================================================
# =========================== ALIAS REPAIR =============================
# ======================================================================
_ALNUM = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

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

    keywords = {
        "select","from","where","group","by","order","limit","join","left","right","inner",
        "on","and","or","not","as","distinct","count","sum","avg","min","max","having",
        "asc","desc","like","in","between","is","null","case","when","then","else","end", "regexp","rlike"
    }
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