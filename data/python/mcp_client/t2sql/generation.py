import re
import json
from typing import Dict, Any, List, Optional
from collections import defaultdict
from .config import *
from .helpers import _wants_customer_name_only, _wants_product_name_only, _parse_allowed_identifiers, _parse_from_join_aliases

# ======================================================================
# ========================== REGEX HEURISTIC ===========================
# ======================================================================
def _looks_like_mask_example(s: str) -> bool:
    """
    Heuristik ringan untuk mendeteksi contoh pola yang lebih cocok REGEXP:
    - Mengandung kombinasi huruf + digit + pemisah ( -, /, _ )
      contoh: 'AAA-9999', 'AB-123', '2024-01-01'
    - Atau mengandung karakter placeholder umum: 9, #, X, N, A
    """
    if not s:
        return False

    # kalau terlalu pendek, biasanya bukan mask serius
    if len(s.strip()) < 4:
        return False

    # placeholder-style
    if any(ch in s for ch in ["9", "#", "X", "N", "A"]):
        return True

    # pola huruf2 + pemisah + digit2
    if re.search(r"[A-Za-z]{2,}[-_/][0-9]{2,}", s):
        return True

    # pola digit2 + pemisah + digit2
    if re.search(r"[0-9]{2,}[-_/][0-9]{2,}", s):
        return True

    return False

# ======================================================================
# ======================== CONTEXT PROMPT ==============================
# ======================================================================
def make_generation_prompt(ctx: str, question: str) -> str:
    return f"""
You are a senior SQL assistant for MySQL 8.0.

Return a runnable query using ONLY tables/columns listed in the ALLOWED IDENTIFIERS section of Context.
If a needed column name appears as an alias in Context, use its canonical column name.
If the query is about customers and the user does NOT explicitly ask for an id/number/code, prefer customerName over customerNumber AND also prefer customerName over contactFirstName/contactLastName when both are available in ALLOWED IDENTIFIERS.

IMPORTANT JOIN RULE:
- If all required columns live in a SINGLE table from ALLOWED IDENTIFIERS, DO NOT use any JOIN at all. Select only from that single table.
- Only introduce JOINs when you really need columns from more than one table.

PATTERN / REGEXP RULES (VERY IMPORTANT):
- When the user describes a string **pattern**, such as:
  - "starts with ..."
  - "ends with ..."
  - "contains ..."
  - "matches pattern ..."
  - "exactly N digits/characters"
  - "format ...", "mask ...", "only digits"/"hanya angka"
  you MUST use proper string pattern operators:
  - For structured or exact patterns (exact length, digit/letter structure, specific format),
    implement the core condition using REGEXP with anchors ^ and $.
  - You may use LIKE (or simple LEFT/RIGHT/SUBSTRING) only for naive wildcard cases
    where there is no requirement about exact length or digit/letter structure
    (e.g. "name starts with 'A'", "contains 'ABC'").
  - DO NOT try to emulate "exactly N digits/characters" using arithmetic tricks such as:
    expressions based on substrings compared with BETWEEN or numeric ranges.

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

def sanitize_notes(sql: str, notes: str) -> str:
    if not notes:
        return notes
    if not has_join(sql or ""):
        return "No joins used."
    return notes.strip()

# ======================================================================
# ===================== POSTPROCESS HEURISTIC ==========================
# ======================================================================
_JOIN_KW = r"(?:inner\s+join|left\s+join|right\s+join|full\s+join|join)"

def has_join(sql: str) -> bool:
    return re.search(rf"\b{_JOIN_KW}\b", sql, flags=re.I) is not None

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

    # CASE 1: SELECT alias.idCol        → SELECT alias.nameCol AS nameCol
    sql = re.sub(
        rf"SELECT\s+([A-Za-z_][A-Za-z0-9_]*)\.({id_pattern})(\s+AS\s+\2)?",
        rf"SELECT \1.{name_column} AS {name_column}",
        sql,
        flags=re.I,
    )

    # CASE 2: SELECT alias.idCol, ...   → SELECT alias.nameCol, ...
    sql = re.sub(
        rf"SELECT\s+([A-Za-z_][A-Za-z0-9_]*)\.({id_pattern})\s*,",
        rf"SELECT \1.{name_column},",
        sql,
        flags=re.I,
    )

    # CASE 3: GROUP BY alias.idCol      → GROUP BY alias.nameCol
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
        # 1) ganti alias.col -> primary_alias.col untuk kolom yang ada di tabel anchor
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

        # 2) hapus JOIN <table> <alias> ON ...
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
    s = _ensure_reasonable_distinct(s, question)
    s = _prefer_friendly_names(s, question)
    ob_match = _extract_order_by_clause(s)
    if ob_match and not _question_requests_ordering(question):
        s = _remove_order_by_clause(s)

    # NEW: dynamic join pruning
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
        "asc","desc","like","in","between","is","null","case","when","then","else","end"
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
            out_parts.append(part)  # quoted → jangan disentuh
        else:
            out_parts.append(replace_in_code_segment(part))  # code → boleh direpair
    return "".join(out_parts)