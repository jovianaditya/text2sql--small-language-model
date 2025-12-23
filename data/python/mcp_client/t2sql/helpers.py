from __future__ import annotations

import re
from typing import TYPE_CHECKING, Dict, List, Optional, Set

if TYPE_CHECKING:
    from .state import RAGState

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


def _question_requires_regex(question: str) -> bool:
    """
    Deteksi kebutuhan regex / pattern matching.
    (Sengaja heuristik ringan; nanti bisa kamu upgrade.)
    """
    q = (question or "").lower()

    # kata kunci eksplisit
    if any(k in q for k in ["regex", "regexp", "regular expression", "pola", "pattern"]):
        return True

    # indikasi pattern matching natural language
    if any(k in q for k in ["starts with", "ends with", "contains", "match", "matches", "diawali", "diakhiri", "mengandung"]):
        return True

    # karakter pola umum
    if any(ch in q for ch in ["*", "^", "$", "[", "]", "(", ")", "|"]):
        return True

    return False

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
        "else","end"
    }

    for tok in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", sql):
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

def _get_requires_regex(state: "RAGState") -> bool:
    """
    Hitung satu kali apakah pertanyaan butuh REGEXP,
    lalu simpan ke state['requires_regex'].
    """
    val = state.get("requires_regex")
    if isinstance(val, bool):
        return val

    q = state.get("question", "")
    needs = _question_requires_regex(q)
    state["requires_regex"] = needs
    return needs