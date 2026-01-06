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
from .utils import _dbg_print_two

# ======================================================================
# ======================= SQL TOKEN KEYWORDS ===========================
# ======================================================================
_SQL_KEYWORDS: Set[str] = {
    "select","from","where","group","by","order","limit","join","left","right","inner",
    "on","and","or","not","as","distinct","count","sum","avg","min","max","having",
    "asc","desc","like","in","between","is","null","case","when","then","else","end",
    "regexp","rlike","over","partition","row_number"
}

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
    q = " " + (question or "").strip().lower() + " "
    
    # --- 1. DETEKSI RANK (The "Winner-Row" Pattern) ---
    # Syarat: Mencari 'baris data' tunggal dalam sebuah kelompok.
    # Pattern: [Group Context] + [Superlative] + [Object, not count]
    has_group = re.search(r"\b(per|each|every|within|partition)\b", q)
    has_superlative = re.search(r"\b(largest|highest|latest|best|most|max|min|top)\b", q)
    # Mencari objek (record/payment), bukan mencari jumlah angka (total/count)
    asks_for_object = re.search(r"\b(record|entry|row|payment|order|details|info)\b", q)
    is_math_agg = re.search(r"\b(count|sum|total|average|avg|number of)\b", q)

    if has_group and has_superlative and asks_for_object and not is_math_agg:
        return "rank"

    # --- 2. DETEKSI AGGREGATE (The "Math/Summary" Pattern) ---
    # Ciri: Melakukan operasi matematika pada sekumpulan baris.
    if is_math_agg or re.search(r"\b(classify|bands?|grouping|distribution)\b", q):
        return "aggregate"

    # --- 3. DEFAULT (The "Data Retrieval" Pattern) ---
    # Jika hanya filter (WHERE) dan sorting (ORDER BY), masuk ke Lookup.
    return "lookup"

def _question_wants_winner_row(question: str) -> bool:
    """
    True jika pertanyaan meminta nilai terbesar/terkecil dan juga meminta atribut lain
    yang harus berasal dari baris yang sama (mis. tanggal kejadian).
    
    Contoh POSITIF (winner-row):
      - "largest payment and its payment date"
      - "highest amount and when it occurred"
      - "pembayaran terbesar dan tanggalnya"
      - "customer with highest payment and when they paid"
    
    Contoh NEGATIF (true aggregation):
      - "largest payment per customer" (hanya minta nilai, bukan atribut baris)
      - "total pembayaran terbesar" (aggregate, bukan row selection)
      - "berapa pembayaran tertinggi" (hanya minta nilai max)
    """
    q = (question or "").lower()

    # Guard: kalau ada "total", "sum", "count" -> ini aggregation, bukan winner-row
    if any(k in q for k in ["total", "sum of", "count", "berapa banyak", "jumlah total"]):
        return False

    rank_sig = any(k in q for k in [
        "largest", "highest", "maximum", "max", "biggest",
        "smallest", "lowest", "minimum", "min",
        "terbesar", "tertinggi", "maksimum",
        "terkecil", "terendah", "minimum",
        "paling besar", "paling kecil",
    ])

    if not rank_sig:
        return False

    # Sinyal 1: eksplisit minta atribut dari "baris yang sama"
    same_row_sig = any(k in q for k in [
        "and its", "and their", "and the", "along with",
        "dan", "beserta", "serta",
        "which", "that", "who",
        "yang mana", "yang",
    ])

    # Sinyal 2: minta informasi temporal/kontekstual
    context_sig = any(k in q for k in [
        "when", "date when", "occurred", "on what date", "at what date",
        "what date", "which date",
        "kapan", "tanggal", "pada tanggal", "waktu", "tanggalnya", "kapan terjadinya",
        # kolom-kolom tipikal yang sering diminta sebagai pasangan max/min
        "payment date", "order date", "shipped date", "required date",
        "customer", "product", "employee",  # entity context
        "pelanggan", "produk", "karyawan",
    ])

    # Sinyal 3: pola "X and Y" atau "X with Y" (multi-attribute request)
    multi_attr_sig = bool(re.search(
        r"\b(largest|highest|maximum|biggest|terbesar|tertinggi)\s+\w+\s+(and|with|beserta|dan)\s+",
        q
    ))

    # Winner-row jika ada rank + (same_row ATAU context ATAU multi_attr)
    return rank_sig and (same_row_sig or context_sig or multi_attr_sig)

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
    if any(p in q for p in ["only digits", "hanya angka", "numeric only"]):
        return True
    # "format" dan "mask" harus whole-word, biar "information" tidak kena "format"
    if re.search(r"\bformat\b", q) or re.search(r"\bmask\b", q):
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
            seg = re.sub(
                rf"(?:\b[A-Za-z_][A-Za-z0-9_]*\s*\.\s*)?\b{re.escape(tbl)}\s*\.\s*",
                f"{al}.",
                seg,
                flags=re.I,
        )
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
    """
    Extract table tokens from FROM/JOIN.
    Supports:
      - schema.table
      - `schema`.`table`
      - `table`
    Returns base table names only (no schema), lower/upper preserved as written.
    """
    if not sql:
        return set()

    pat = re.compile(
        r"\b(?:from|join)\s+"
        r"(?P<table>`?[A-Za-z_][A-Za-z0-9_]*`?(?:\s*\.\s*`?[A-Za-z_][A-Za-z0-9_]*`?)?)",
        flags=re.I,
    )

    tables: Set[str] = set()
    for m in pat.finditer(sql):
        raw = (m.group("table") or "").replace("`", "")
        base = raw.split(".")[-1].strip()
        if base:
            tables.add(base)
    return tables

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

    # 1) Hapus string literals/double quotes supaya token scan tidak ke-distract
    code_only = re.sub(r"'(?:''|[^'])*'", "''", sql)
    code_only = re.sub(r'"(?:""|[^"])*"', '""', code_only)

    keywords = _SQL_KEYWORDS  # FIX: selalu defined

    # 2) Ambil table names yang muncul di FROM/JOIN (untuk exclude)
    #    Support schema.table dan backticks secara sederhana.
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

    # =========================
    # Exclude aliases (table + derived + output)
    # =========================
    exclude: Set[str] = set()

    # 3) table aliases from FROM/JOIN
    a2t = _parse_from_join_aliases(sql)  # alias -> table
    exclude.update({a.lower() for a in a2t.keys() if a})

    # 4) derived-table aliases: FROM ( ... ) AS t / JOIN ( ... ) AS t
    for m in re.finditer(
        r"\b(?:from|join)\s*\([^)]*\)\s*(?:as\s+)?([A-Za-z_]\w*)\b",
        code_only,
        flags=re.I | re.S,
    ):
        exclude.add(m.group(1).lower())

    # 5) output aliases: SELECT ... AS alias (hanya area SELECT ... FROM)
    m_sel = re.search(r"\bselect\b(.*?)\bfrom\b", code_only, flags=re.I | re.S)
    if m_sel:
        select_part = m_sel.group(1)
        for m in re.finditer(r"\bas\s+([A-Za-z_]\w*)\b", select_part, flags=re.I):
            exclude.add(m.group(1).lower())

    # =========================
    # 6) Collect qualified usages: alias.col / table.col
    # =========================
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

    # =========================
    # 7) Unqualified token scan (konservatif)
    # =========================
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

# def validate_identifiers(sql: str, ctx: str) -> List[str]:
#     allowed = _parse_allowed_identifiers(ctx)
#     used = _identify_used_identifiers(sql)
#     errs: List[str] = []

#     alias2table = _parse_from_join_aliases(sql)  # alias -> table (canonical)
#     # build reverse: table names set (lower) to help mapping
#     allowed_lower = {t.lower(): t for t in allowed.keys()}

#     # Remap used qualifiers (aliases) to real tables
#     remapped: Dict[Optional[str], Set[str]] = {}
#     for q, cols in used.items():
#         if q is None:
#             remapped.setdefault(None, set()).update(cols)
#             continue

#         q_norm = q.strip("`")
#         # if qualifier is alias -> table
#         t = alias2table.get(q_norm) or alias2table.get(q_norm.lower())
#         if t:
#             # normalize to ALLOWED canonical casing if possible
#             canon = allowed_lower.get(t.lower(), t)
#             remapped.setdefault(canon, set()).update(cols)
#         else:
#             # qualifier might already be a table name
#             canon = allowed_lower.get(q_norm.lower(), q_norm)
#             remapped.setdefault(canon, set()).update(cols)

#     used = remapped

#     explicit_tables = {t for t in used.keys() if t is not None}

#     single_table: Optional[str] = None
#     if len(explicit_tables) == 1:
#         single_table = next(iter(explicit_tables))
#     elif len(explicit_tables) == 0 and len(allowed) == 1:
#         single_table = next(iter(allowed.keys()))

#     for tbl, cols in used.items():
#         if tbl is None:
#             continue
#         if tbl not in allowed:
#             errs.append(f"table `{tbl}` not in ALLOWED")
#             continue
#         bad = [c for c in cols if c not in allowed[tbl]]
#         if bad:
#             errs.append(f"{tbl} -> invalid columns: {', '.join(bad)}")

#     if single_table:
#         allowed_cols = allowed.get(single_table, set())
#         for col in used.get(None, set()):
#             if col not in allowed_cols:
#                 errs.append(f"{single_table} -> invalid column (unqualified): {col}")

#     return errs
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

def _expand_missing_null_to_null_or_empty(sql: str, question: str) -> str:
    """
    Policy: untuk pertanyaan yang bermakna 'missing', treat sebagai:
      col IS NULL  -> (col IS NULL OR col = '')
    Tanpa TRIM, sesuai benchmark.
    Sangat konservatif: jalan hanya jika memang ada predicate 'IS NULL'
    dan belum ada '= ''' untuk kolom tsb.
    """
    if not sql or not _question_wants_missing_value(question):
        return sql

    # Jangan ganggu query yang sudah eksplisit handle empty-string
    if re.search(r"=\s*''", sql):
        return sql

    # split supaya tidak rewrite di dalam string literal/backtick
    parts = re.split(r"('(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`[^`]*`)", sql)

    for i in range(0, len(parts), 2):
        seg = parts[i]

        # Rewrite kolom string yang umum: state (qualified/unqualified)
        # state IS NULL  -> (state IS NULL OR state = '')
        seg = re.sub(
            r"\b((?:[A-Za-z_][A-Za-z0-9_]*\s*\.\s*)?state)\s+IS\s+NULL\b",
            r"(\1 IS NULL OR \1 = '')",
            seg,
            flags=re.I,
        )

        parts[i] = seg

    return "".join(parts)

def _rewrite_left_join_to_inner_if_not_needed(sql: str, question: str) -> str:
    if not sql or _question_wants_include_zero(question):
        return sql
    if join_count(sql) != 1:
        return sql
    if not re.search(r"\bleft\s+join\b", sql, flags=re.I):
        return sql

    # Kalau ada referensi RHS alias/table di SELECT/GROUP/ORDER/HAVING,
    # itu biasanya artinya user memang ingin baris RHS (bukan preserve-zeros).
    # Maka LEFT JOIN tidak diperlukan.
    m = re.search(
        r"\bFROM\b.*?\bLEFT\s+JOIN\b\s+(?P<t2>`?[A-Za-z_]\w*`?)(?:\s+(?:AS\s+)?(?P<a2>[A-Za-z_]\w*))?",
        sql,
        flags=re.I | re.S,
    )
    if not m:
        return sql

    t2 = (m.group("t2") or "").strip("`")
    a2 = (m.group("a2") or "").strip()
    rhs = a2 or t2
    if not rhs:
        return sql

    # cek penggunaan rhs.<col> di luar ON
    masked = re.sub(r"\bON\b\s+.*?(?=(\bWHERE\b|\bGROUP\s+BY\b|\bORDER\s+BY\b|\bHAVING\b|\bLIMIT\b|;|$))",
                    " ", sql, flags=re.I | re.S)
    if re.search(rf"\b{re.escape(rhs)}\s*\.", masked, flags=re.I):
        return re.sub(r"\bLEFT\s+JOIN\b", "JOIN", sql, count=1, flags=re.I)

    return sql

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

# def _question_wants_include_zero(question: str) -> bool:
#     q = (question or "").lower()
#     signals = [
#         "including customers with no", "customers with no", "with no payments", "never paid",
#         "all customers", "even if no", "include those without",
#         "termasuk", "meskipun tidak", "walaupun tidak", "yang tidak punya", "tanpa pembayaran",
#         "pelanggan yang belum", "customer yang belum",
#     ]
#     return any(s in q for s in signals)
def _question_wants_include_zero(question: str) -> bool:
    q = (question or "").lower()
    
    # Explicit zero-inclusion signals
    explicit_signals = [
        "including customers with no", "customers with no", "with no payments", 
        "never paid", "all customers", "even if no", "include those without",
        "termasuk", "meskipun tidak", "walaupun tidak", "yang tidak punya", 
        "tanpa pembayaran", "pelanggan yang belum", "customer yang belum",
    ]
    
    if any(s in q for s in explicit_signals):
        return True
    
    # ✅ NEW: Implicit preserve-all patterns
    entity_kw = r"(customer|product|employee|order|office|payment|pelanggan|produk|karyawan|pesanan|pembayaran)"
    
    # Pattern 1: "for each X" / "per X" / "by X"
    if re.search(rf"\b(for\s+each|per|setiap|tiap)\s+{entity_kw}", q, flags=re.I):
        return True
    
    # Pattern 2: "X with <aggregate condition>"
    if re.search(
        rf"\b{entity_kw}s?\s+with\s+(total|sum|count|average|avg|jumlah)\s+.*?\b(exceed|over|greater|more\s+than|above|>|melebihi|lebih\s+dari)",
        q,
        flags=re.I
    ):
        return True
    
    # Pattern 3: "group by X" dalam konteks non-filter
    # (ini tricky, tapi "group by customer" biasanya preserve-all)
    if re.search(rf"\bgroup\s+by\s+{entity_kw}", q, flags=re.I):
        # HANYA jika tidak ada pre-filter intent ("who paid", "that ordered", dll)
        if not re.search(r"\b(who|that|which|yang)\s+(paid|ordered|purchased|bought)", q, flags=re.I):
            return True
    
    return False

def _question_wants_missing_value(question: str) -> bool:
    q = (question or "").lower()
    return any(s in q for s in [
        "missing", "is missing", "missing information",
        "empty", "blank", "not provided", "unknown",
        "kosong", "tidak ada", "belum diisi", "null"
    ])

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

def _split_top_level_commas(expr: str) -> List[str]:
    """
    Split string by comma pada level top (tidak di dalam quote/paren).
    Dipakai untuk memecah SELECT list dan GROUP BY list.
    """
    if not expr:
        return []

    s = expr.strip()
    parts: List[str] = []
    buf: List[str] = []
    depth = 0
    in_sq = False
    in_dq = False
    i = 0

    while i < len(s):
        ch = s[i]

        # single quotes
        if ch == "'" and not in_dq:
            if in_sq and i + 1 < len(s) and s[i + 1] == "'":
                buf.append("''")
                i += 2
                continue
            in_sq = not in_sq
            buf.append(ch)
            i += 1
            continue

        # double quotes
        if ch == '"' and not in_sq:
            in_dq = not in_dq
            buf.append(ch)
            i += 1
            continue

        if in_sq or in_dq:
            buf.append(ch)
            i += 1
            continue

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

        if depth == 0 and ch == ",":
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
            i += 1
            continue

        buf.append(ch)
        i += 1

    last = "".join(buf).strip()
    if last:
        parts.append(last)

    return parts


def _is_simple_column_expr(x: str) -> bool:
    """
    Hanya izinkan bentuk kolom sederhana:
      - col
      - alias.col
      - `alias`.`col` atau `col`
    Tidak izinkan fungsi/aritmetika kompleks (konservatif).
    """
    if not x:
        return False
    t = x.strip()
    # buang alias AS ... kalau ada (caller biasanya sudah pisah, tapi aman)
    t = re.sub(r"\s+AS\s+\w+\s*$", "", t, flags=re.I).strip()
    # match: `a`.`b` | a.b | `b` | b
    return re.fullmatch(
        r"(?:`?[A-Za-z_][A-Za-z0-9_]*`?\s*\.\s*)?`?[A-Za-z_][A-Za-z0-9_]*`?",
        t
    ) is not None


def _extract_alias_or_colname(select_item: str) -> Optional[str]:
    """
    Ambil output column name yang bisa dipakai outer select.
    Prefer:
      1) alias dari 'AS xxx'
      2) kalau bentuk alias.col => col
      3) selain itu: None (caller boleh abort)
    """
    if not select_item:
        return None
    s = select_item.strip()

    m_as = re.search(r"\bAS\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", s, flags=re.I)
    if m_as:
        return m_as.group(1)

    # kalau "t.col" tanpa AS
    m_col = re.fullmatch(
        r"(?:`?[A-Za-z_][A-Za-z0-9_]*`?\s*\.\s*)?`?([A-Za-z_][A-Za-z0-9_]*)`?\s*$",
        s
    )
    if m_col:
        return m_col.group(1)

    return None

def _maybe_keep_or_alias_simple(expr: str, out_name: Optional[str]) -> str:
    """
    Kalau expr itu kolom sederhana dan nama output natural-nya sudah sama dengan out_name,
    jangan bikin 'expr AS out_name' (menghindari: customerNumber AS customerNumber).

    Alias hanya dipasang kalau:
    - out_name berbeda dari nama natural kolom, atau
    - expr bukan kolom sederhana (mis. fungsi/aritmetika), atau
    - memang perlu stabilisasi nama.
    """
    e = (expr or "").strip()
    if not e or not out_name:
        return e

    # kalau sudah ada AS, jangan ganggu
    if re.search(r"\bAS\b", e, flags=re.I):
        return e

    # nama natural yang akan keluar (mis. p.customerNumber -> customerNumber)
    natural = _extract_alias_or_colname(e)

    # kalau kolom sederhana dan natural == out_name -> jangan alias
    if natural and natural.lower() == out_name.lower() and _is_simple_column_expr(e):
        return e

    return f"{e} AS {out_name}"

# ✅ AFTER (replace with)
def _parse_select_from_groupby(sql: str) -> Optional[dict]:
    """
    Parse sangat konservatif untuk pola:
      SELECT ... FROM ... [WHERE ...] GROUP BY ... [ORDER BY/LIMIT/HAVING...]

    Return dict berisi potongan string mentah.
    Abort jika:
      - WITH
      - subquery FROM ( ... )
      - HAVING (karena semantik aggregate sulit dipertahankan aman)
      - ROW_NUMBER (already windowed)
      - Malformed syntax (spacing issues detected)
    """
    if not sql:
        return None

    s = sql.strip().rstrip(";").strip()
    
    # ✅ SAFETY: Detect malformed syntax and abort early
    # Pattern: clause keywords stuck together without space
    if re.search(r"\b[A-Za-z0-9_]+(?:WHERE|GROUP|ORDER|HAVING|LIMIT)\b", s, flags=re.I):
        # Try one normalization pass
        s = re.sub(r"(?<=[A-Za-z0-9_)])(?=\b(?:WHERE|GROUP|ORDER|HAVING|LIMIT)\b)", " ", s, flags=re.I)
        # If still malformed, abort
        if re.search(r"\b[A-Za-z0-9_]+(?:WHERE|GROUP|ORDER|HAVING|LIMIT)\b", s, flags=re.I):
            return None

    # ✅ PRE-NORMALIZE: Fix common spacing issues
    s = re.sub(r"(\b[A-Za-z_][A-Za-z0-9_]*)\s*(?=\b(?:HAVING|ORDER|LIMIT|WHERE|GROUP)\b)", r"\1 ", s, flags=re.I)
    s = re.sub(r"\s{2,}", " ", s)  # collapse multiple spaces

    if re.match(r"^\s*with\b", s, flags=re.I):
        return None
    if re.search(r"\bfrom\s*\(", s, flags=re.I):
        return None
    if re.search(r"\bhaving\b", s, flags=re.I):
        return None
    if re.search(r"\bROW_NUMBER\s*\(", s, flags=re.I):
        return None

    # tangkap SELECT ... FROM ... GROUP BY ... (sisanya optional)
    pat = re.compile(
        r"""
        ^\s*SELECT\s+(?P<select>.*?)
        \s+FROM\s+(?P<from>.*?)
        \s+GROUP\s+BY\s+(?P<groupby>.*?)
        (?:\s+HAVING\s+(?P<having>.*?))?
        (?P<tail>\s+(?:ORDER\s+BY|LIMIT)\b.*?|)\s*$
        """,
        flags=re.I | re.S | re.VERBOSE
    )
    m = pat.match(s)
    if not m:
        return None

    select_part = m.group("select").strip()
    from_part = m.group("from").strip()
    groupby_part = m.group("groupby").strip()
    having_part = (m.group("having") or "").strip()
    tail = (m.group("tail") or "").strip()

    # Split FROM menjadi: from_core + where_clause (optional)
    # Kita cari WHERE yang top-level (konservatif: pakai regex pertama, karena FROM (subquery) sudah di-abort).
    where_m = re.search(r"\bWHERE\b", from_part, flags=re.I)
    if where_m:
        from_core = from_part[:where_m.start()].strip()
        where_clause = from_part[where_m.start():].strip()  # include WHERE
    else:
        from_core = from_part
        where_clause = ""

    return {
        "select": select_part,
        "from_core": from_core,
        "where": where_clause,
        "groupby": groupby_part,
        "having": having_part,
        "tail": tail,
    }

def _select_has_multiple_aggregates_on_same_row_intent(sel):
    """
    Detect cases like:
      SELECT key, MAX(x), MAX(y)
    which usually indicate winner-row misuse,
    because aggregates are being used to proxy row attributes.
    """

    from sqlglot import exp

    aggregates = []

    for item in sel.expressions:
        base = item.this if isinstance(item, exp.Alias) else item
        agg = (
            base.find(exp.Max)
            or base.find(exp.Min)
            or (base if isinstance(base, (exp.Max, exp.Min)) else None)
        )
        if agg:
            aggregates.append(agg)

    # only suspicious when >1 aggregate appears in SELECT
    if len(aggregates) <= 1:
            return False

    # if *any* of the aggregates refer to time/order semantics,
    # treat as winner-row misuse
    time_tokens = ("date", "time", "stamp", "update", "created")

    for agg in aggregates:
        try:
            col = agg.this.sql(dialect="mysql").lower()
        except Exception:
            continue

        if any(tok in col for tok in time_tokens):
            return True

    # otherwise: still allow (conservative)
    return False

# ensure tie-break expr is RAW ROW VALUE (not MAX(...))
def _to_raw_expr(expr: str) -> str:
    # remove AS <alias>
    expr = re.sub(r"\s+AS\s+\w+$", "", expr, flags=re.I).strip()
    # take inner of MAX(x) / MIN(x)
    m = re.match(r"\b(MAX|MIN)\s*\((.+)\)", expr, flags=re.I)
    return m.group(2).strip() if m else expr

# ✅ AFTER (replace with)
def _rewrite_winner_row_guardrail(sql: str, question: str, force: bool = False) -> str:
    """
    Guardrail untuk pola salah:

      SELECT k, MAX(metric), attr_from_same_row
      FROM t
      GROUP BY k

    → rewrite menjadi winner-row via ROW_NUMBER()
    """

    import re
    from typing import List

    if not sql:
        return sql

    s0 = sql.strip()

    # ✅ CRITICAL: Detect truly malformed queries (not false positives from subqueries)
    # Pattern: ROW_NUMBER appears BEFORE GROUP BY (malformed window)
    if re.search(r"\bROW_NUMBER\s*\([^)]*\)\s+[^;]*\bGROUP\s+BY\b", s0, flags=re.I):
        return sql  # Already corrupted window + group mix
    
    # Pattern: Multiple FROM at same nesting level (structural corruption)
    # Count top-level FROM (excluding those in subqueries)
    s_no_sub = re.sub(r"\([^)]*\)", "", s0)  # Remove all subqueries
    from_count = len(re.findall(r"\bFROM\b", s_no_sub, flags=re.I))
    if from_count > 1:
        return sql  # Multiple FROM without proper nesting = broken

    # 🚧 intent-level filter — boleh dilewati saat FORCE MODE
    if not _question_wants_winner_row(question) and not force:
        return sql

    # 🚫 jangan rewrite kalau sudah windowed
    if re.search(r"\bROW_NUMBER\s*\(", s0, flags=re.I):
        return sql

    if "GROUP BY" in s0.upper() and "ORDER BY" in s0.upper():
        # Check if GROUP BY comes AFTER ORDER BY (invalid/corrupted)
        group_pos = s0.upper().find("GROUP BY")
        order_pos = s0.upper().find("ORDER BY")
        if group_pos > order_pos and order_pos > 0:
            return sql  # SQL already corrupted, don't make it worse

    parsed = _parse_select_from_groupby(s0)
    if not parsed:
        return sql   # parsing fail = hard stop

    select_items = _split_top_level_commas(parsed["select"])
    group_items  = _split_top_level_commas(parsed["groupby"])

    if not select_items or not group_items:
        return sql   # struktur rusak → jangan rewrite

    # GROUP BY harus kolom sederhana (kecuali FORCE MODE)
    if any(not _is_simple_column_expr(g) for g in group_items):
        if not force:
            return sql

    # =====================================================
    # PARTITION KEY (entity key)
    # =====================================================
    q_low = (question or "").lower()
    partition_items: List[str] = []

    def looks_like_customer(g: str) -> bool:
        gn = g.replace("`","").lower()
        return "customer" in gn

    if "customer" in q_low:
        partition_items = [g for g in group_items if looks_like_customer(g)]

    # fallback — tetap harus ada PARTITION
    if not partition_items:
        partition_items = [group_items[0]]

    # =====================================================
    # DETECT AGGREGATES
    # =====================================================

    agg_pat = re.compile(
        r"^(?P<prefix>.*?)\b(?P<fn>MAX|MIN)\s*\(\s*(?P<expr>.*?)\s*\)"
        r"\s*(?P<alias>\bAS\s+[A-Za-z_][A-Za-z0-9_]*\s*)?$",
        flags=re.I | re.S
    )

    agg_idx = -1
    agg_kind = None
    agg_expr = None
    agg_out_name = None
    all_aggs = []

    for i, item in enumerate(select_items):
        it = item.strip()
        m = agg_pat.match(it)
        if not m:
            continue

        if (m.group("prefix") or "").strip():
            if not force:
                return sql

        fn = (m.group("fn") or "").upper()
        expr = (m.group("expr") or "").strip()
        if not expr:
            return sql

        all_aggs.append(expr)

        # multiple aggregates → treat as misuse candidate
        if agg_idx != -1:
            agg_idx = -2
            break

        agg_idx = i
        agg_kind = fn
        agg_expr = expr

        m_as = re.search(r"\bAS\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", it, flags=re.I)
        agg_out_name = m_as.group(1) if m_as else _extract_alias_or_colname(expr)

    # =====================================================
    # CASE A — multiple MAX/MIN → likely winner-row misuse
    # =====================================================
    if agg_idx == -2:

        if not force and not _looks_like_winner_row_multi_agg(select_items):
            return sql

        # pilih metric = aggregate yang paling "nilai"
        metric_candidates = [
            x for x in all_aggs
            if any(tok in x.lower() for tok in ["amount","price","total","score","quantity","revenue","sales"])
        ]

        agg_expr = metric_candidates[0] if metric_candidates else all_aggs[0]
        agg_kind = "MAX"
        agg_out_name = _extract_alias_or_colname(agg_expr)

        # semua kolom lain = kandidat row attributes
        non_agg_items = [x for x in select_items]

    # =====================================================
    # CASE B — single aggregate (normal)
    # =====================================================
    else:
        if agg_idx == -1 or not agg_kind or not agg_expr:
            if not force:
                return sql

        # pilih metric yang paling masuk akal
        metric_candidates = [
            x for x in all_aggs
            if any(tok in x.lower() for tok in ["amount","price","total","score","quantity","revenue","sales"])
        ]
        if metric_candidates:
            agg_expr = metric_candidates[0]

        if not _is_simple_column_expr(agg_expr):
            if not force:
                return sql

        # ambil kolom non-aggregate lain
        non_agg_items = [x for j,x in enumerate(select_items) if j != agg_idx]

        if not non_agg_items and not force:
            return sql

    # =====================================================
    # TIE BREAKER SELECTION - ROBUST VERSION
    # =====================================================

    def norm_col(z: str) -> str:
        z2 = re.sub(r"\s+","", z.replace("`","").lower())
        z2 = re.sub(r"\bas\b.*$","", z2, flags=re.I)
        return z2

    group_norm = {norm_col(g) for g in group_items}

    # Collect all candidate tie-breakers (not in GROUP BY)
    candidates = []
    for x in non_agg_items:
        base = re.sub(r"\s+AS\s+\w+\s*$","", x, flags=re.I).strip()
        
        # Extract from MAX/MIN wrapper
        agg_match = re.match(r"^(MAX|MIN)\s*\(\s*(.+?)\s*\)$", base, flags=re.I)
        if agg_match:
            base = agg_match.group(2)
        
        if norm_col(base) not in group_norm and _is_simple_column_expr(base):
            candidates.append(base)

    # ✅ Strategy: collect ALL date/time columns as tie-breakers
    tie_exprs: List[str] = []
    
    # Priority 1: Date columns (most common tie-breaker)
    for c in candidates:
        c_low = c.lower()
        if any(k in c_low for k in ["date", "time", "timestamp", "created", "updated"]):
            tie_exprs.append(_to_raw_expr(c))
    
    # Priority 2: If no date found, use first non-date candidate
    if not tie_exprs and candidates:
        tie_exprs.append(_to_raw_expr(candidates[0]))

    # =====================================================
    # ORDERING - ROBUST VERSION
    # =====================================================

    metric_dir = "DESC" if agg_kind == "MAX" else "ASC"

    # ✅ Detect tie-breaker direction from question
    ql = (question or "").lower()
    
    # Explicit signals
    if any(k in ql for k in ["latest", "most recent", "newest", "terbaru", "paling baru", "terakhir"]):
        tie_dir = "DESC"
    elif any(k in ql for k in ["earliest", "oldest", "first", "terlama", "paling lama", "pertama"]):
        tie_dir = "ASC"
    else:
        # ✅ Smart default based on context
        # For date/time columns: default DESC (latest is usually desired)
        # For other columns: ASC
        if tie_exprs and any(k in tie_exprs[0].lower() for k in ["date", "time", "timestamp"]):
            tie_dir = "DESC"
        else:
            tie_dir = "ASC"

    # Strip table prefixes
    agg_expr = _strip_table_prefix(agg_expr)
    tie_exprs = [_strip_table_prefix(t) for t in tie_exprs]

    # Build ORDER BY: metric first, then tie-breakers (limit to 2 for stability)
    order_parts = [f"{agg_expr} {metric_dir}"]
    for tie in tie_exprs[:2]:  # ✅ Take up to 2 tie-breakers
        order_parts.append(f"{tie} {tie_dir}")

    order_window = ", ".join(order_parts)

    # =====================================================
    # BUILD INNER + OUTER QUERY
    # =====================================================

    inner_items = []
    outer_items = []

    for j, x in enumerate(select_items):
        orig_alias = _extract_alias_or_colname(x)

        # --- 🔥 untuk kolom aggregate: GANTI MAX(...) -> raw metric expr ---
        if j == agg_idx:
            base = agg_expr
            # ✅ Use natural column name, not the aggregate alias
            out_name = base  # "amount" instead of "max_amount"
            
        else:
            base = re.sub(
                r"\s+AS\s+[A-Za-z_][A-Za-z0-9_]*\s*$",
                "",
                x,
                flags=re.I
            ).strip()
            
            # Extract from aggregate functions for other items
            agg_match = re.match(r"^(MAX|MIN)\s*\(\s*(.+?)\s*\)$", base, flags=re.I)
            if agg_match:
                base = agg_match.group(2)
                # ✅ Use natural column name for extracted aggregates
                out_name = _strip_table_prefix(base)
            else:
                # Keep original alias for true non-aggregate columns
                out_name = orig_alias if orig_alias else _strip_table_prefix(base)
            
            base = _strip_table_prefix(base)

        inner_items.append(
            _maybe_keep_or_alias_simple(base, out_name)
        )
        outer_items.append(out_name)

    partition_clean = [_strip_table_prefix(p) for p in partition_items]

    inner_items.append(
        f"ROW_NUMBER() OVER "
        f"(PARTITION BY {', '.join(partition_clean)} "
        f"ORDER BY {order_window}) AS rn"
    )

    from_core = parsed["from_core"]
    where_clause = parsed["where"]
    tail = parsed["tail"]

    inner_sql = "SELECT " + ", ".join(inner_items) + " FROM " + from_core
    if where_clause:
        inner_sql += " " + where_clause

    inner_sql = _strip_alias_if_single_table(inner_sql)

    out_sql = (
        "SELECT " + ", ".join(outer_items) +
        " FROM (" + inner_sql + ") t WHERE rn = 1"
    )

    if tail:
        out_sql += " " + tail

    return out_sql

def _rewrite_winner_row_aggregate_misuse(sql: str, question: str) -> str:
    """
    Catch winner-row questions where the SQL still uses MAX/MIN (often with HAVING hacks)
    and rewrite to ROW_NUMBER winner-row selection.

    This handles cases WITHOUT GROUP BY (your current guardrail requires GROUP BY).
    Very conservative:
    - triggers only if _question_wants_winner_row(question)
    - triggers only if query contains exactly one MAX/MIN in SELECT
    - aborts if query already has ROW_NUMBER
    """
    if not sql or not _question_wants_winner_row(question):
        return sql
    if re.search(r"\bROW_NUMBER\s*\(", sql, flags=re.I):
        return sql
    if not re.search(r"\b(MAX|MIN)\s*\(", sql, flags=re.I):
        return sql

    try:
        import sqlglot
        from sqlglot import exp
    except Exception:
        return sql

    try:
        ast = sqlglot.parse_one(sql, read="mysql")
    except Exception:
        return sql

    sel = ast.find(exp.Select)
    if not sel:
        return sql

    # Identify exactly one MAX/MIN in SELECT list
    agg_item = None
    agg_kind = None
    metric_expr = None

    def find_maxmin(e: exp.Expression):
        m = e.find(exp.Max) or e.find(exp.Min)
        return m

    for item in sel.expressions:
        base = item.this if isinstance(item, exp.Alias) else item
        m = find_maxmin(base)
        if not m:
            continue
        if agg_item is not None:
            return sql  # >1 aggregate => abort (conservative)
        agg_item = item
        if isinstance(m, exp.Max):
            agg_kind = "MAX"
            metric_expr = m.this
        elif isinstance(m, exp.Min):
            agg_kind = "MIN"
            metric_expr = m.this

    if agg_item is None or metric_expr is None or agg_kind is None:
        return sql

    # Collect non-aggregate SELECT expressions (potential partition/tie/output)
    nonagg_items = []
    for item in sel.expressions:
        if item is agg_item:
            continue
        base = item.this if isinstance(item, exp.Alias) else item
        # skip literals/stars
        if isinstance(base, (exp.Literal, exp.Star)):
            continue
        # if it contains an agg function, abort
        if base.find(exp.AggFunc):
            return sql
        nonagg_items.append(item)

    if not nonagg_items:
        return sql

    q_low = (question or "").lower()

    def norm_name(e: exp.Expression) -> str:
        try:
            return e.sql(dialect="mysql").lower()
        except Exception:
            return ""

    # Partition columns:
    # Prefer columns mentioning the entity in the question (e.g., customer)
    partition_exprs = []
    for item in nonagg_items:
        base = item.this if isinstance(item, exp.Alias) else item
        s = norm_name(base)
        if "customer" in q_low and "customer" in s:
            partition_exprs.append(base)

    # Fallback: id/number/code-like
    if not partition_exprs:
        for item in nonagg_items:
            base = item.this if isinstance(item, exp.Alias) else item
            s = norm_name(base)
            if any(k in s for k in ["id", "number", "code"]):
                partition_exprs.append(base)
                break

    # Final fallback: first non-agg expression
    if not partition_exprs:
        base0 = nonagg_items[0].this if isinstance(nonagg_items[0], exp.Alias) else nonagg_items[0]
        partition_exprs = [base0]

    # Tie-breaker: prefer a date-like column among non-agg outputs (excluding partition)
    tie_exprs = []
    for item in nonagg_items:
        base = item.this if isinstance(item, exp.Alias) else item
        s = norm_name(base)
        if base in partition_exprs:
            continue
        if "date" in s:
            tie_exprs.append(base)

    # ORDER BY for window
    metric_desc = (agg_kind == "MAX")
    order_exprs = [
        exp.Ordered(this=metric_expr.copy(), desc=metric_desc)
    ]
    for te in tie_exprs[:2]:  # keep it short/conservative
        order_exprs.append(exp.Ordered(this=te.copy(), desc=False))

    rn = exp.Window(
        this=exp.RowNumber(),
        partition_by=exp.Partition(expressions=[p.copy() for p in partition_exprs]),
        order=exp.Order(expressions=order_exprs),
    )

    # Determine output alias for metric
    metric_alias = "value"
    if isinstance(agg_item, exp.Alias) and agg_item.alias:
        metric_alias = agg_item.alias
    else:
        # If it's MAX(amount) with no alias, try to name it reasonably
        metric_alias = "amount" if "amount" in norm_name(metric_expr) else "value"

    # Build inner select with stable aliases
    inner_exprs = []
    outer_exprs = []

    for item in nonagg_items:
        if isinstance(item, exp.Alias):
            inner_exprs.append(item.copy())
            outer_exprs.append(exp.column(item.alias, table="t"))
        else:
            # Require it to be a Column for safety; otherwise abort
            if not isinstance(item, exp.Column):
                return sql
            name = item.name
            inner_exprs.append(exp.alias_(item.copy(), name, quoted=False))
            outer_exprs.append(exp.column(name, table="t"))

    # Add metric
    inner_exprs.append(exp.alias_(metric_expr.copy(), metric_alias, quoted=False))
    outer_exprs.append(exp.column(metric_alias, table="t"))

    # Add rn
    inner_exprs.append(exp.alias_(rn, "rn", quoted=False))

    # Rebuild inner FROM/WHERE from original query
    if not sel.args.get("from"):
        return sql
    inner = exp.Select(expressions=inner_exprs)
    inner.set("from", sel.args["from"].copy())

    if sel.args.get("where"):
        inner.set("where", sel.args["where"].copy())

    # Outer query
    outer = exp.Select(expressions=outer_exprs).from_(
        exp.Subquery(this=inner, alias=exp.TableAlias(this=exp.to_identifier("t")))
    ).where(exp.EQ(this=exp.column("rn", table="t"), expression=exp.Literal.number(1)))

    return outer.sql(dialect="mysql")

def _remove_top_level_order_by_if_not_requested(sql: str, question: str) -> str:
    if not sql:
        return sql
    if _question_requests_ordering(question):
        return sql

    try:
        import sqlglot
        from sqlglot import exp
    except Exception:
        return sql  # jangan rusak pipeline kalau sqlglot tidak ada

    try:
        ast = sqlglot.parse_one(sql, read="mysql")
    except Exception:
        return sql

    # ast biasanya Select untuk query normal
    sel = ast if isinstance(ast, exp.Select) else None
    if sel is None:
        # fallback: cari Select terluar (bukan inner). Cari yang parent-nya bukan Select.
        # (cukup aman untuk kasus kamu)
        candidates = list(ast.find_all(exp.Select))
        if candidates:
            sel = candidates[0]

    if sel is None:
        return sql

    # Hapus ORDER BY hanya di select terluar
    if sel.args.get("order"):
        sel.set("order", None)

    return ast.sql(dialect="mysql")

def _looks_like_winner_row_multi_agg(select_items):
    """
    Detect patterns like:
      MAX(amount), MAX(paymentDate)
    which strongly indicate 'pick row with max value + its attributes'
    rather than true aggregation.
    """
    cols = []

    for it in select_items:
        # extract column inside MAX(...)
        m = re.search(r"max\s*\(\s*([A-Za-z0-9_.`]+)\s*\)",
                      it, flags=re.I)
        if not m:
            continue

        col = m.group(1).lower()
        cols.append(col)

    # only suspicious if there are ≥2 MAX()
    if len(cols) < 2:
        return False

    # Very strong signal of winner-row intent
    # e.g. MAX(amount) + MAX(paymentDate)
    if any("date" in c for c in cols):
        return True

    # allow more conservative fallback later
    return False

def _winner_row_sanity_date_in_groupby(sql: str, question: str) -> str:
    if not sql or not _question_wants_winner_row(question):
        return sql

    if re.search(r"\bROW_NUMBER\s*\(", sql, flags=re.I):
        return sql

    s = sql.lower()

    # only flag suspicious — do NOT modify query
    if "group by" not in s:
        return sql

    if not re.search(r"\b(max|min)\s*\(", s):
        return sql

    # if GROUP BY contains date column → risky aggregate misuse
    gb = re.search(
        r"\bgroup\s+by\s+(.*?)(?=(\border\s+by\b|\bhaving\b|\blimit\b|;|$))",
        sql,
        flags=re.I | re.S,
    )

    if gb and re.search(r"\b\w*date\w*\b", gb.group(1), flags=re.I):
        # do NOT rewrite — only let guardrail force rewrite later
        return sql

    return sql

def _strip_having_max_self_subquery(sql: str) -> str:
    """
    Remove HAVING MAX(x) = (SELECT MAX(x) ...) pattern.
    Two-pass: first normalize spacing, then remove.
    """
    if not sql:
        return sql
    
    # Pass 1: Normalize spacing around HAVING
    sql = re.sub(r"\s*\bHAVING\b\s*", " HAVING ", sql, flags=re.I)
    
    # Pass 2: Remove the HAVING clause if it matches winner-row hack pattern
    # Pattern: HAVING MAX(...) = (SELECT MAX(...) ...)
    pat = re.compile(
        r"\s+HAVING\s+MAX\s*\([^)]+\)\s*=\s*\(\s*SELECT\s+MAX\s*\([^)]+\)[^)]+\)",
        flags=re.I | re.S
    )
    
    if pat.search(sql):
        # Remove the HAVING clause
        sql = pat.sub(" ", sql)
        
        # Clean up multiple spaces
        sql = re.sub(r"\s{2,}", " ", sql)
    
    return sql

def _winner_row_final_sanity(sql: str, question: str) -> str:
    if not sql or not _question_wants_winner_row(question):
        return sql
    if re.search(r"\bROW_NUMBER\s*\(", sql, flags=re.I):
        return sql

    # Kalau query sudah masuk pola HAVING-subquery, kita anggap "suspect"
    # dan kita coba tetap rewrite, bukan abort.
    try:
        import sqlglot
        from sqlglot import exp
    except Exception:
        return sql

    try:
        ast = sqlglot.parse_one(sql, read="mysql")
    except Exception:
        return sql

    sel = ast.find(exp.Select)
    if not sel:
        return sql

    grp = sel.args.get("group")
    if not grp or not grp.expressions:
        return sql

    # Cari tepat 1 MAX/MIN aggregate di SELECT list
    agg_items = []
    nonagg_items = []
    for item in sel.expressions:
        base = item.this if isinstance(item, exp.Alias) else item
        if base.find(exp.Max) or base.find(exp.Min):
            agg_items.append(item)
        else:
            nonagg_items.append(item)

    if len(agg_items) != 1:
        return sql

    agg_item = agg_items[0]
    agg_base = agg_item.this if isinstance(agg_item, exp.Alias) else agg_item

    # Ambil expr metric di dalam MAX/MIN
    metric_expr = None
    agg_kind = None
    if isinstance(agg_base, exp.Max):
        metric_expr = agg_base.this
        agg_kind = "MAX"
    elif isinstance(agg_base, exp.Min):
        metric_expr = agg_base.this
        agg_kind = "MIN"
    else:
        # bisa juga Max/Min nested; fallback
        m = agg_base.find(exp.Max) or agg_base.find(exp.Min)
        if isinstance(m, exp.Max):
            metric_expr = m.this
            agg_kind = "MAX"
        elif isinstance(m, exp.Min):
            metric_expr = m.this
            agg_kind = "MIN"

    if metric_expr is None:
        return sql

    # group-by set (string normalize)
    def norm(e: exp.Expression) -> str:
        return e.sql(dialect="mysql").strip().lower()

    group_set = {norm(g) for g in grp.expressions}

    # Deteksi ada non-agg select yang bukan group-by => winner-row misuse
    nonagg_not_in_group = []
    for item in nonagg_items:
        base = item.this if isinstance(item, exp.Alias) else item
        # skip literal/star
        if isinstance(base, (exp.Literal, exp.Star)):
            continue
        if norm(base) not in group_set:
            nonagg_not_in_group.append(item)

    if not nonagg_not_in_group:
        return sql  # bukan winner-row misuse

    # Tentukan partition columns: pilih group-by yang nampak "key"
    key_like = []
    for g in grp.expressions:
        s = norm(g)
        if any(k in s for k in ["id", "number", "code"]):
            key_like.append(g)

    partition_exprs = key_like or [grp.expressions[0]]

    # Tie-breaker: cari kolom non-agg yang mengandung 'date'
    tie_expr = None
    for item in nonagg_not_in_group:
        base = item.this if isinstance(item, exp.Alias) else item
        if "date" in norm(base):
            tie_expr = base
            break
    if tie_expr is None:
        # fallback: pakai nonagg pertama yang "not in group"
        base = nonagg_not_in_group[0].this if isinstance(nonagg_not_in_group[0], exp.Alias) else nonagg_not_in_group[0]
        tie_expr = base

    # Build ORDER BY window
    metric_dir = "DESC" if agg_kind == "MAX" else "ASC"

    order_by = exp.Order(expressions=[
        exp.Ordered(this=metric_expr.copy(), desc=(metric_dir == "DESC")),
        exp.Ordered(this=tie_expr.copy(), desc=False),
    ])

    rn = exp.Window(
        this=exp.RowNumber(),
        partition_by=exp.Partition(expressions=[p.copy() for p in partition_exprs]),
        order=order_by,
    )

    # Inner select: semua output yang diminta, tapi MAX/MIN diganti jadi metric_expr
    inner_select_exprs = []
    outer_select_exprs = []

    for item in sel.expressions:
        if item is agg_item:
            alias = item.alias_or_name or "value"
            inner_select_exprs.append(exp.alias_(metric_expr.copy(), alias, quoted=False))
            outer_select_exprs.append(exp.column(alias, table="t"))
        else:
            # pastikan tiap item punya alias supaya outer stabil
            if isinstance(item, exp.Alias):
                alias = item.alias
                inner_select_exprs.append(item.copy())
                outer_select_exprs.append(exp.column(alias, table="t"))
            else:
                # derive alias dari column name kalau memungkinkan
                name = item.name if isinstance(item, exp.Column) else None
                if not name:
                    return sql  # konservatif: abort jika kompleks
                inner_select_exprs.append(exp.alias_(item.copy(), name, quoted=False))
                outer_select_exprs.append(exp.column(name, table="t"))

    inner_select_exprs.append(exp.alias_(rn, "rn", quoted=False))

    # Rebuild query: SELECT outer FROM (SELECT inner FROM ... WHERE ...) t WHERE rn=1
    inner = exp.Select(expressions=inner_select_exprs).from_(sel.args["from"].copy())
    if sel.args.get("where"):
        inner.set("where", sel.args["where"].copy())

    outer = exp.Select(expressions=outer_select_exprs).from_(
        exp.Subquery(this=inner, alias=exp.TableAlias(this=exp.to_identifier("t")))
    ).where(exp.EQ(this=exp.column("rn", table="t"), expression=exp.Literal.number(1)))

    return outer.sql(dialect="mysql")

def _force_restore_groupby_if_aggregate_without_groupby(sql_before, sql_after):
    """
    Fail-safe untuk ONLY_FULL_GROUP_BY:
    jika setelah rewrite GROUP BY hilang,
    tetapi masih ada aggregate non-window,
    kembalikan sql_before (jangan eksekusi sql_after).
    """
    if not sql_after:
        return sql_after

    low = sql_after.lower()

    # sudah aman / windowed
    if "row_number(" in low:
        return sql_after

    # tidak ada aggregate → aman
    if not re.search(r"\b(max|min|sum|avg|count)\s*\(", sql_after, flags=re.I):
        return sql_after

    # ada aggregate tapi tidak ada GROUP BY → rollback
    if "group by" not in low:
        return sql_before

    return sql_after

def _looks_like_max_date_tiebreaker(sql: str) -> bool:
    """
    Deteksi pattern sangat spesifik yang PASTI winner-row misuse:
    - Multiple MAX() dimana salah satunya adalah date-like
    - ATAU: HAVING dengan MAX = subquery self-join hack
    
    Ini dipakai untuk FORCE MODE, jadi harus sangat konservatif.
    """
    s = sql.lower()

    # must be aggregate query
    if "group by" not in s:
        return False

    # must involve MAX() 
    if "max(" not in s:
        return False

    # Count berapa banyak MAX()
    max_count = len(re.findall(r"\bmax\s*\(", s, flags=re.I))

    # ===== SIGNAL 1: Multiple MAX() with date-like (very suspicious) =====
    if max_count >= 2:
        date_like = re.compile(
            r"max\s*\([^)]*date[^)]*\)",
            re.I,
        )
        if date_like.search(s):
            return True

    # ===== SIGNAL 2: HAVING with MAX = subquery (classic hack) =====
    if re.search(r"\bhaving\s+max\s*\(", s):
        return True

    # ===== SIGNAL 3: Multiple MAX() on different semantic types =====
    # e.g., MAX(amount), MAX(paymentDate) - clearly trying to get row attributes
    if max_count >= 2:
        # Check if we have value-like and date-like columns
        has_value = bool(re.search(
            r"max\s*\([^)]*(?:amount|price|total|revenue|sales|quantity)[^)]*\)",
            s, flags=re.I
        ))
        has_date = bool(re.search(
            r"max\s*\([^)]*date[^)]*\)",
            s, flags=re.I
        ))
        if has_value and has_date:
            return True

    return False

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

    on_has_rhs_filter = _on_clause_has_rhs_filter(on_clause, a2)

    # kalau tidak ada rhs_only dari WHERE dan ON juga tidak filter-like → bukan filter-only join
    if (not rhs_only) and (not on_has_rhs_filter):
        return sql

    rhs_filters = " AND ".join(rhs_only) if rhs_only else ""

    exists_where = on_clause
    if rhs_filters:
        exists_where = f"{on_clause} AND {rhs_filters}"
        
    if rhs_filters:
        exists_expr = f"EXISTS (SELECT 1 FROM {t2} {a2} WHERE {on_clause} AND {rhs_filters})"
    else:
        exists_expr = f"EXISTS (SELECT 1 FROM {t2} {a2} WHERE {on_clause})"

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

    Diperkuat:
    - tahan backtick di ON / qualifiers
    - guard: jika ON mengandung RHS filter (mis. a2.status='X'), jangan drop (biarkan EXISTS logic)
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

    # Guard: kalau ON mengandung filter RHS (bukan pure equi-join key), jangan drop.
    # Ini mencegah bentrok dengan logic join->exists (ON-filter-only).
    if _on_clause_has_rhs_filter(on_txt, rhs):
        return sql

    # Normalisasi untuk parsing ON:
    # - strip string literals supaya tidak false positive
    # - remove backtick supaya regex join-key match stabil
    on_clean = _strip_string_literals(on_txt).replace("`", "")
    lhs_clean = lhs.replace("`", "")
    rhs_clean = rhs.replace("`", "")

    # ON harus bentuk sederhana: lhs.k = rhs.k (atau kebalik), dengan k sama
    on_m = re.search(
        rf"\b{re.escape(lhs_clean)}\s*\.\s*(?P<k1>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*{re.escape(rhs_clean)}\s*\.\s*(?P<k2>[A-Za-z_][A-Za-z0-9_]*)\b"
        rf"|\b{re.escape(rhs_clean)}\s*\.\s*(?P<k3>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*{re.escape(lhs_clean)}\s*\.\s*(?P<k4>[A-Za-z_][A-Za-z0-9_]*)\b",
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

    # Ambil bagian di luar ON untuk cek pemakaian RHS
    outside_on = (m.group("prefix") or "") + (m.group("tail") or "")
    outside_on = _strip_string_literals(outside_on).replace("`", "")

    # Jika RHS dipakai untuk kolom lain selain join_key → tidak aman drop
    bad_rhs_use = re.search(
        rf"\b{re.escape(rhs_clean)}\s*\.\s*(?!{re.escape(join_key)}\b)[A-Za-z_][A-Za-z0-9_]*\b",
        outside_on,
        flags=re.I,
    )
    if bad_rhs_use:
        return sql

    # Rewrite rhs.join_key -> lhs.join_key di luar ON (jangan sentuh string/backtick segments)
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

    prefix_raw = (m.group("prefix") or "")
    t1_raw = m.group("t1")
    from_part = f"FROM {t1_raw}" + (f" {a1}" if a1 else "")

    # FIX: rewrite rhs.join_key -> lhs.join_key juga di SELECT/prefix
    new_prefix = _rewrite_outside_on(prefix_raw)
    new_tail   = _rewrite_outside_on(tail)

    return f"{new_prefix}{from_part}{new_tail}"

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
def _strip_alias_if_single_table(sql: str) -> str:
    """
    If query has only one base table in FROM and no JOIN,
    strip alias prefixes like e.col -> col.
    DO NOT touch queries with GROUP BY or subqueries.
    """
    if not sql:
        return sql

    s = sql.lower()

    # HARD GUARDS — jangan sentuh query kompleks
    if " join " in s:
        return sql
    if " group by " in s:
        return sql
    if " from (" in s:          # subquery / derived table
        return sql
    if " row_number" in s:      # winner-row pipeline
        return sql

    m = re.search(r"\bfrom\s+([a-zA-Z_][a-zA-Z0-9_]*)\b", s)
    if not m:
        return sql

    # split to avoid touching literals
    parts = re.split(r"('(?:''|[^'])*')", sql)

    for i in range(0, len(parts), 2):
        parts[i] = re.sub(
            r"\b([a-zA-Z_][a-zA-Z0-9_]*)\.",
            "",
            parts[i],
        )

    return "".join(parts)

def _strip_table_prefix(expr: str) -> str:
    """
    Remove leading table aliases like  p.col -> col
    Only when expression is simple column form.
    """
    if not expr:
        return expr

    # do not touch function calls / expressions
    if "(" in expr or ")" in expr:
        return expr

    # p.customerNumber  -> customerNumber
    return re.sub(
        r"\b[A-Za-z_][A-Za-z0-9_]*\.",
        "",
        expr
    )

def _expand_group_by_aliases_regex(sql: str) -> str:
    """
    Regex-based GROUP BY alias expansion (simpler but less robust).
    """
    if not sql or "group by" not in sql.lower():
        return sql

    # Extract SELECT ... FROM
    m_select = re.search(
        r"\bSELECT\b(.*?)\bFROM\b",
        sql,
        flags=re.I | re.S
    )
    if not m_select:
        return sql

    select_part = m_select.group(1)

    # Build alias -> expression map
    alias_map = {}
    
    # Pattern: <expression> AS <alias>
    for m in re.finditer(
        r"(CASE\b.*?\bEND|\w+\s*\([^)]*\)|[A-Za-z_][A-Za-z0-9_.]*)\s+AS\s+([A-Za-z_][A-Za-z0-9_]*)",
        select_part,
        flags=re.I | re.S
    ):
        expr = m.group(1).strip()
        alias = m.group(2).strip()
        alias_map[alias.lower()] = expr

    if not alias_map:
        return sql

    # Extract GROUP BY clause
    m_group = re.search(
        r"\bGROUP\s+BY\b\s+(.*?)(?=(\bORDER\s+BY\b|\bHAVING\b|\bLIMIT\b|;|$))",
        sql,
        flags=re.I | re.S
    )
    if not m_group:
        return sql

    group_clause = m_group.group(1).strip()
    
    # Replace aliases in GROUP BY
    def replace_alias(match):
        col = match.group(1).strip()
        col_lower = col.lower()
        if col_lower in alias_map:
            return alias_map[col_lower]
        return col

    # Pattern: column names in GROUP BY (comma-separated)
    new_group = re.sub(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\b",
        replace_alias,
        group_clause
    )

    # Replace in original SQL
    return sql[:m_group.start(1)] + new_group + sql[m_group.end(1):]

def _enforce_explicit_group_by(sql: str) -> str:
    """
    Benchmark style:
    If query has GROUP BY, every SELECT expression that is NOT:
      - aggregate (SUM/COUNT/AVG/MIN/MAX/...),
      - window (... OVER ...),
      - literal,
      - star
    must appear explicitly in GROUP BY.
    """
    if not sql:
        return sql

    try:
        import sqlglot
        from sqlglot import exp
    except Exception:
        return sql

    try:
        ast = sqlglot.parse_one(sql, read="mysql")
    except Exception:
        return sql

    sel = ast.find(exp.Select)
    if not sel:
        return sql

    grp = sel.args.get("group")
    # SAFETY: do nothing if no GROUP BY exists
    if not grp or not grp.expressions:
        return sql

    group_set = {g.sql(dialect="mysql").strip().lower() for g in grp.expressions}
    missing = []

    for item in sel.expressions:
        base = item.this if isinstance(item, exp.Alias) else item

        if base.find(exp.AggFunc):
            continue
        if base.find(exp.Window):
            continue
        if isinstance(base, (exp.Literal, exp.Star)):
            continue

        base_sql = base.sql(dialect="mysql").strip().lower()
        if base_sql not in group_set:
            missing.append(base)
            group_set.add(base_sql)

    if missing:
        grp.set("expressions", grp.expressions + missing)

    return ast.sql(dialect="mysql")

def _rewrite_having_alias_to_expr(sql: str) -> str:
    """
    Benchmark style:
    Avoid using SELECT aliases in HAVING (e.g. HAVING totalPaymentAmount > 50000).
    Rewrite HAVING to use the original aggregate expression (e.g. HAVING SUM(p.amount) > 50000)
    when the alias maps to a SELECT expression.
    """
    if not sql:
        return sql

    try:
        import sqlglot
        from sqlglot import exp
    except Exception:
        return sql

    try:
        ast = sqlglot.parse_one(sql, read="mysql")
    except Exception:
        return sql

    sel = ast.find(exp.Select)
    if not sel:
        return sql

    having = sel.args.get("having")
    if not having:
        return ast.sql(dialect="mysql")

    # Build alias -> expression map from SELECT list
    alias_map = {}
    for item in sel.expressions:
        if isinstance(item, exp.Alias):
            alias = item.alias
            if alias:
                # Keep the raw expression (not the alias wrapper)
                alias_map[alias.lower()] = item.this

    if not alias_map:
        return ast.sql(dialect="mysql")

    # Replace column references in HAVING that match aliases
    def transform(node: exp.Expression) -> exp.Expression:
        # We only rewrite simple identifier references: HAVING totalPaymentAmount > 50000
        if isinstance(node, exp.Column):
            # exp.Column can be qualified; alias usually appears unqualified
            if node.table:
                return node
            name = (node.name or "").lower()
            if name in alias_map:
                return alias_map[name].copy()
        return node

    new_having = having.transform(transform)
    sel.set("having", new_having)

    return ast.sql(dialect="mysql")

def _drop_redundant_column_aliases(sql: str) -> str:
    """
    Hapus alias yang redundant untuk kolom sederhana:
      - customerNumber AS customerNumber  -> customerNumber
      - p.amount AS amount                -> p.amount
    TIDAK menyentuh:
      - fungsi/window/expr (harus tetap ber-alias jika butuh nama output stabil)
      - alias yang beda nama
    """
    if not sql:
        return sql

    try:
        import sqlglot
        from sqlglot import exp
    except Exception:
        # Fallback regex konservatif (hanya untuk bentuk kolom sederhana)
        # alias.col AS col  / col AS col
        def _repl(m: re.Match) -> str:
            expr = m.group("expr").strip()
            al = m.group("al").strip()
            # hanya drop jika expr adalah kolom sederhana dan nama akhirnya sama dgn alias
            if not _is_simple_column_expr(expr):
                return m.group(0)
            last = expr.replace("`", "").split(".")[-1].strip()
            return expr if last.lower() == al.lower() else m.group(0)

        return re.sub(
            r"(?P<expr>(?:`?[A-Za-z_]\w*`?\s*\.\s*)?`?[A-Za-z_]\w*`?)\s+AS\s+(?P<al>[A-Za-z_]\w*)\b",
            _repl,
            sql,
            flags=re.I,
        )

    try:
        ast = sqlglot.parse_one(sql, read="mysql")
    except Exception:
        return sql

    def transform(node: "exp.Expression") -> "exp.Expression":
        # Drop only: Alias(Column, alias_name) where alias_name == column.name
        if isinstance(node, exp.Alias) and isinstance(node.this, exp.Column):
            alias = (node.alias or "").strip()
            colname = (node.this.name or "").strip()
            if alias and colname and alias.lower() == colname.lower():
                return node.this
        return node

    ast2 = ast.transform(transform)
    return ast2.sql(dialect="mysql")

def _drop_unused_from_table_aliases(sql: str) -> str:
    """
    Hapus alias di FROM <table> [AS] <alias> jika alias itu tidak pernah dipakai sebagai qualifier (alias.col)
    di sisa query.

    Sangat konservatif:
    - hanya jalan kalau query TIDAK punya JOIN (biar tidak ganggu join semantics)
    - tidak menyentuh derived table: FROM (SELECT ...) t  (karena MySQL butuh alias derived table)
    - tidak menyentuh alias reserved
    """
    if not sql:
        return sql

    # Kalau ada JOIN di mana pun, jangan sentuh (biar aman).
    if has_join(sql):
        return sql

    # Build versi "scan" yang aman dari string literal + backticks untuk deteksi alias.col
    scan = _strip_string_literals(sql).replace("`", "")

    pat = re.compile(
        r"""
        \bFROM\s+
        (?P<table>`?[A-Za-z_][A-Za-z0-9_]*`?(?:\.[A-Za-z_][A-Za-z0-9_]*)?)
        \s+
        (?:AS\s+)?(?P<alias>[A-Za-z_][A-Za-z0-9_]*)\b
        """,
        flags=re.IGNORECASE | re.VERBOSE,
    )

    def repl(m: re.Match) -> str:
        table_raw = m.group("table")
        alias = m.group("alias")

        # derived table tidak match di sini (karena table token bukan '('), jadi aman
        if not alias or alias.lower() in _RESERVED_ALIASES:
            return m.group(0)

        # cek apakah alias dipakai sebagai qualifier setelah posisi FROM ini
        # (kalau dipakai, jangan dihapus)
        if re.search(rf"\b{re.escape(alias)}\s*\.", scan[m.end():], flags=re.I):
            return m.group(0)

        # alias tidak dipakai -> drop alias
        return f"FROM {table_raw}"

    return pat.sub(repl, sql)

def _is_potential_winner_row(sql: str, question: str, force: bool = False) -> bool:
    """
    Detect whether a query *should* be treated as a winner-row query.

    force=True → skip intent filter (used when guardrail already confirmed pattern)
    """

    if not sql:
        return False

    # intent-aware mode unless forced
    if not _question_wants_winner_row(question) and not force:
        return False

    s = sql.lower()

    # must mention GROUP BY or an aggregate candidate
    if "group by" not in s and "max(" not in s and "min(" not in s:
        return False

    # classic misuse patterns
    if "max(" in s and "paymentdate" in s:
        return True
    if "max(" in s and "orderdate" in s:
        return True
    if "having" in s and "max(" in s:
        return True

    return True

def _postprocess_sql(sql: str, question: str, context: str = "") -> str:
    DEBUG_WR = True
    if not sql:
        return sql
    s = sql.strip()
    
    # ✅ CRITICAL FIX: Normalize ALL major clause boundaries FIRST
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
    # ✅ Debug checkpoint - verify spacing fixed
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
    # ------------------------------------------------------------
    # Phase 0 — Syntax & Canonicalization
    # Normalize tokens and table naming early
    # ------------------------------------------------------------
    s = _fix_bad_backticks(s)
    s = _canonicalize_table_names(s, context)
    s = ensure_join_aliases(s)
    # ------------------------------------------------------------
    # Phase 1 — JOIN Semantics & Structural Rewrites (FINAL)
    # Decide JOIN necessity and semantics before touching aliases
    # ------------------------------------------------------------
    s = _rewrite_left_join_to_inner_if_not_needed(s, question)
    s = _rewrite_left_join_to_inner_if_rhs_filtered(s, question)
    s = _rewrite_inner_join_filter_to_exists(s)
    s = _rewrite_inner_join_existence_to_exists(s, question)

    s = _drop_redundant_inner_join_same_key(s)
    s = _prune_ornamental_joins_safe(s, question)
    # ------------------------------------------------------------
    # Phase 2 — Alias Hygiene & FROM Cleanup
    # Safe only after JOIN structure is frozen
    # ------------------------------------------------------------
    s = _drop_redundant_column_aliases(s)
    s = _drop_unused_from_table_aliases(s)

    s = _strip_alias_if_single_table(s)
    # ------------------------------------------------------------
    # Phase 3 — Column Naming & GROUP BY Consistency
    # Operate on final column identities
    # ------------------------------------------------------------
    s = _prefer_friendly_names(s, question)
    s = _expand_group_by_aliases_regex(s)
    if not _is_potential_winner_row(s, question, force=True):
        s = _enforce_explicit_group_by(s)
    # ------------------------------------------------------------
    # Phase 4 — Winner-Row & Aggregate Guardrails
    # Ensure semantic correctness for ranking-style queries
    # ------------------------------------------------------------
    before_phase4 = s

    if DEBUG_WR: print("\n[PH4] start ----")
    
    # ✅ CRITICAL: Re-normalize spacing BEFORE Phase 4
    # (Aliases/joins might have corrupted spacing)
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

    # ============================================================
    # 🚨 CASE 1 — pattern clearly indicates MAX + date tie-breaker
    # → enforce ROW_NUMBER rewrite (FORCE MODE)
    # ============================================================
    if _looks_like_max_date_tiebreaker(s):
        # ✅ SAFETY: Double-check question intent
        # Jangan force rewrite kalau question JELAS bukan winner-row
        if _question_wants_winner_row(question):
            if DEBUG_WR: print("[PH4] detected max-date tie-breaker + winner-row intent -> forcing guardrail")
            
            # ✅ CRITICAL: Check if query is already corrupted
            s_forced = _rewrite_winner_row_guardrail(s, question, force=True)
            
            # ✅ SANITY: If forced rewrite produced garbage, rollback to original
            if not s_forced or ("GROUP BY" in s_forced.upper() and "ROW_NUMBER" in s_forced.upper()):
                if DEBUG_WR: print("[PH4] ABORT: forced rewrite produced malformed SQL, keeping original")
                s = before_phase4
            else:
                s = s_forced
                
            if DEBUG_WR: print("[PH4] FINAL after forced guardrail:", s)
            
            # ✅ Skip normal heuristic flow - we've made our choice
            s = _rewrite_having_alias_to_expr(s)
            s = _force_restore_groupby_if_aggregate_without_groupby(before_phase4, s)
            if DEBUG_WR: print("[PH4] end ----", s, "\n")
            
            # ✅ DON'T return early - continue to Phase 5-7
            goto_phase5 = True
        else:
            if DEBUG_WR: print("[PH4] max-date pattern detected BUT question wants true aggregation -> skip force")
            goto_phase5 = False
    else:
        goto_phase5 = False

    # ============================================================
    # 🚥 CASE 2 — normal heuristic winner-row handling
    # ============================================================
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
    # ------------------------------------------------------------
    # Phase 5 — Pattern, REGEXP, and Predicate Normalization
    # ------------------------------------------------------------
    s = _guardrail_fix_like_regex_anchors(s, question)
    s = _rewrite_simple_regexp_to_like(s, question)
    s = _guardrail_fix_mysql_regexp_literals(s)
    # ------------------------------------------------------------
    # Phase 6 — NULL / DISTINCT / Result Cardinality Policies
    # ------------------------------------------------------------
    s = _expand_missing_null_to_null_or_empty(s, question)
    s = _ensure_reasonable_distinct(s, question)
    s = _strip_unneeded_count_distinct(s)
    # ------------------------------------------------------------
    # Phase 7 — Ordering, Limits, and Output Formatting
    # Cosmetic and execution-safety adjustments only
    # ------------------------------------------------------------
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