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
def _parse_from_join_aliases(sql: str) -> Dict[str, str]:
    """
    Parse alias dari FROM/JOIN:
    - FROM products p  -> {"p": "products"}
    - JOIN orders o    -> {"o": "orders"}
    Return: mapping alias -> table_name
    """
    if not sql:
        return {}

    pattern = re.compile(
        r"\b(?:from|join)\s+([a-zA-Z_][\w\.]*)\s+(?:as\s+)?([a-zA-Z_]\w*)\b",
        re.IGNORECASE,
    )
    out: Dict[str, str] = {}
    for table, alias in pattern.findall(sql):
        out[alias] = table
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

Return a runnable query using ONLY tables/columns listed in the ALLOWED IDENTIFIERS section of Context.
If a needed column name appears as an alias in Context, use its canonical column name.
If the query is about customers and the user does NOT explicitly ask for an id/number/code, prefer customerName over customerNumber AND also prefer customerName over contactFirstName/contactLastName when both are available in ALLOWED IDENTIFIERS.

IMPORTANT JOIN RULE:
- If all required columns live in a SINGLE table from ALLOWED IDENTIFIERS, DO NOT use any JOIN at all. Select only from that single table.
- Only introduce JOINs when you really need columns from more than one table.

PATTERN / REGEXP RULES (VERY IMPORTANT):
1) Use LIKE for simple wildcard intent only:
   - starts with X  -> col LIKE 'X%'
   - ends with X    -> col LIKE '%X'
   - contains X     -> col LIKE '%X%'
   IMPORTANT: LIKE does NOT support regex anchors (^, $). Never put ^ or $ inside LIKE.

2) Use REGEXP (MySQL 8) only when:
   A) Word-boundary intent: "contains a word that starts with X" / "kata yang diawali X"
   B) Strict structure/format: "exactly N digits/chars", "letter + digits", masks, "only digits"
   C) User explicitly requests regex/regexp.

3) REGEXP word-boundary style (do NOT use \b or \w):
   - Use boundary: (^|[^[:alnum:]_]) as the left word boundary.
   - After the prefix, use word chars: [[:alnum:]_]* (NOT '.*').
   Example: col REGEXP '(^|[^[:alnum:]_])Xy[[:alnum:]_]*'

   Note: '^Xy' matches only the beginning of the full string, not the beginning of a word inside the string.

4) REGEXP strict structure:
   - Always anchor with ^ and $ to enforce full-string structure.
   Example: col REGEXP '^x[0-9]{3}$'

########################
# STRICT OUTPUT FORMAT #
########################
{{ "sql": "<final MySQL query>", "notes": "<brief rationale; for EACH JOIN, name the exact selected/grouped/filtered column it contributes>" }}
# No extra keys. No Markdown. End SQL with a semicolon.
# If the final SQL has NO JOINs, set "notes" exactly to: "No joins used."

##################
# SQL CONSTRAINT #
##################
- MySQL 8.0; backticks only when needed; ANSI joins.

################
# CONTEXT
################
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
        sql = re.sub(
            r"COUNT\s*\(\s*DISTINCT\s+o\.orderNumber\s*\)",
            "COUNT(o.orderNumber)",
            sql,
            flags=re.IGNORECASE,
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

    # REGEXP '^196' => LIKE '196%'
    sql = re.sub(
        r"(\bWHERE\b.*?\b([A-Za-z_][A-Za-z0-9_]*\.)?[A-Za-z_][A-Za-z0-9_]*\s+REGEXP\s+)'\^([A-Za-z0-9 _-]+)'",
        r"\1'\3%'".replace("REGEXP", "LIKE"),  # aman karena group(1) masih mengandung REGEXP
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    )

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
    "diawali", "awalan", "awal", "prefix"])
    ends = any(k in q for k in [
        "ends with", "end with",
        "diakhiri", "akhir", "suffix"])
    contains = any(k in q for k in [
        "contains", "contain",
        "mengandung"])

    # Rewrite LIKE '^X' -> LIKE 'X%'
    def repl_starts(m: re.Match) -> str:
        col = m.group(1)
        val = m.group(2)
        return f"{col} LIKE '{val}%'"

    # Rewrite LIKE 'X$' -> LIKE '%X'
    def repl_ends(m: re.Match) -> str:
        col = m.group(1)
        val = m.group(2)
        return f"{col} LIKE '%{val}'"

    # Generic: kalau question "starts with" tapi pattern like '^...' muncul
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

    # Optional: kalau "contains" tapi dia bikin '^X$' atau '^X'
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

def _dynamic_prune_joins_if_single_table_possible(sql: str, question: str, ctx: str = "") -> str:
    """
    Coba hilangkan JOIN yang tidak perlu secara dinamis:
    - Lihat alias -> table dari FROM/JOIN
    - Lihat kolom yang dipakai per alias
    - Lihat schema dari ALLOWED IDENTIFIERS
    - Kalau semua kolom yang dipakai lewat alias lain juga ada di 1 tabel anchor,
      alias itu dianggap redundant dan JOIN-nya bisa dihapus.

    Hanya aktif untuk query jenis 'lookup' dan 'aggregate_all' (lebih aman).
    """
    if not sql or not has_join(sql):
        return sql

    allowed = _parse_allowed_identifiers(ctx)
    if not allowed:
        return sql

    alias2table = _parse_from_join_aliases(sql)
    if len(alias2table) <= 1:
        return sql  # nggak ada join beneran

    usage = _collect_alias_column_usage(sql)

    # Pilih alias anchor: yang paling banyak dipakai kolomnya
    alias_counts = {a: len(cols) for a, cols in usage.items() if a in alias2table}
    if not alias_counts:
        return sql

    primary_alias = max(alias_counts.items(), key=lambda x: x[1])[0]
    primary_table = alias2table.get(primary_alias)
    if not primary_table or primary_table not in allowed:
        return sql

    # schema lower-case
    allowed_lower = {t.lower(): {c.lower() for c in cols} for t, cols in allowed.items()}
    primary_cols = allowed_lower.get(primary_table.lower())
    if not primary_cols:
        return sql

    redundant_aliases: List[tuple[str, str]] = []

    for alias, cols in usage.items():
        if alias == primary_alias:
            continue
        table = alias2table.get(alias)
        if not table:
            continue
        if table not in allowed:
            continue  # kita nggak tahu schema-nya, jangan diutak-atik

        # semua kolom yang dipakai via alias ini harus tersedia juga di tabel anchor
        if all(col.lower() in primary_cols for col in cols):
            redundant_aliases.append((alias, table))

    if not redundant_aliases:
        return sql

    new_sql = sql

    for alias, table in redundant_aliases:
        # ----------------------------------------------------------------------------
        # 1) ganti alias.col -> primary_alias.col untuk kolom yang ada di tabel anchor
        # ----------------------------------------------------------------------------
        pattern = re.compile(
            rf"\b{re.escape(alias)}\.\s*([A-Za-z_][A-Za-z0-9_]*)",
            flags=re.IGNORECASE,
        )

        def _repl(m: re.Match) -> str:
            col = m.group(1)
            if col.lower() in primary_cols:
                return f"{primary_alias}.{col}"
            return m.group(0)

        new_sql = pattern.sub(_repl, new_sql)
        # ----------------------------------------------------------------------------
        # 2) hapus JOIN <table> <alias> ON ...
        # ----------------------------------------------------------------------------
        join_pat = re.compile(
            rf"""
            \s+
            (?:INNER\s+|LEFT\s+|RIGHT\s+|FULL\s+OUTER\s+|FULL\s+)?JOIN
            \s+{re.escape(table)}\s+{re.escape(alias)}
            \s+ON
            \s+
            .*?
            (?=(\s+(?:INNER\s+|LEFT\s+|RIGHT\s+|FULL\s+OUTER\s+)?JOIN\b
                |\s+WHERE\b
                |\s+GROUP\s+BY\b
                |\s+ORDER\s+BY\b
                |\s+LIMIT\b
                |;|$))
            """,
            flags=re.IGNORECASE | re.VERBOSE | re.DOTALL,
        )
        new_sql = join_pat.sub(" ", new_sql, count=1)

    return new_sql

# ======================================================================
# ===================== SQL NORMALIZATION PIPELINE =====================
# ======================================================================
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
    s = _guardrail_fix_like_regex_anchors(s, question)
    s = _rewrite_simple_regexp_to_like(s, question)
    s = _guardrail_fix_mysql_regexp_literals(s)
    s = _ensure_reasonable_distinct(s, question)
    s = _prefer_friendly_names(s, question)
    ob_match = _extract_order_by_clause(s)
    if ob_match and not _question_requests_ordering(question):
        s = _remove_order_by_clause(s)

    # dynamic join pruning
    s = _dynamic_prune_joins_if_single_table_possible(s, question, context)

    s = _maybe_add_limit(question, s)
    s = _fix_missing_space_before_clauses(s)
    s = _collapse_spaces_outside_quotes(s)
    s = _strip_unneeded_count_distinct(s)
    return s

def normalize_sql_pipeline(raw_sql: str, question: str, ctx: str) -> str:
    """
    Pipeline standar untuk normalisasi SQL:
    - postprocess (normalize keyword, limit, distinct, dsb)
    - repair alias berdasarkan alias_map
    - postprocess lagi setelah alias diganti
    """
    if not raw_sql:
        return raw_sql
    s = _postprocess_sql(raw_sql, question, ctx)
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