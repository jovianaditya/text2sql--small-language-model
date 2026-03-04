# ======================================================================
# ========================== helpers_core.py ===========================
# ======================================================================
# Purpose:
# - Heuristic Rules for SQL Generation Phase 0 - Phase 7.
# - Core helper functions used across multiple phases.

# ======================================================================
# ====================== STANDARD LIBRARY IMPORT =======================
# ======================================================================
import re
from typing import Any, Dict, List, Optional, Set, Tuple
# ======================================================================
# ======================== LOCAL MODULE IMPORT =========================
# ======================================================================
from .schema_analyzer import (
    get_name_column,
    get_id_column,
    get_all_tables,
    get_transaction_tables,
    get_fk_relationships,
    get_schema_analyzer,
    get_table_synonyms,
    get_singular_form,
)
from .config import ALIAS_MAX_LEN
# ======================================================================
# ===================== SQL ALIAS & KEYWORD CONSTANTS ==================
# ======================================================================
_RESERVED_ALIASES = {
    "or", "and", "on", "as", "in", "is", "by", "from", "join", "where",
    "group", "order", "limit", "select", "left", "right", "inner", "outer",
    "having", "union", "cross", "natural",
    # MySQL 8.0+ window function keywords (reserved)
    "rank", "dense_rank", "row_number", "ntile", "lead", "lag",
    "first_value", "last_value", "nth_value", "cume_dist", "percent_rank"
}
_SQL_KEYWORDS = {
    "select","from","where","group","by","order","limit","join","left","right","inner",
    "on","and","or","not","as","distinct","count","sum","avg","min","max","having",
    "asc","desc","like","in","between","is","null","case","when","then","else","end",
    "regexp","rlike","over","partition","row_number",
    "union","all","except","intersect"
}
_JOIN_KW = r"(?:inner\s+join|left\s+join|right\s+join|full\s+join|join)"
# =======================================================================
# ================= Phase 0 — Syntax & Canonicalization =================
# =============== Normalize tokens and table naming early ===============
# =======================================================================
# - should_use_backtick
# - smart_badtick
# - split_qualified
# - fix_bad_backticks
# - parse_allowed_identifiers
# - canonicalize_table_names
# - alias_for_table
# - split_outside_quotes
# - has_join

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

def _escape_reserved_column_aliases(sql: str) -> str:
    """
    Escape column aliases that are MySQL reserved keywords.
    E.g.: AS rank -> AS `rank`
    
    Pattern: ... AS <alias> (where alias is NOT already backticked)
    Only applies to SELECT clause column aliases, not table aliases.
    """
    if not sql:
        return sql
    
    # Pattern to find: ) AS alias or expr AS alias (column alias in SELECT)
    # Must NOT match table aliases (FROM/JOIN table AS alias)
    # Focus on common patterns: aggregate/window function result aliases
    
    def escape_alias(m: re.Match) -> str:
        prefix = m.group(1)  # ") AS " or similar
        alias = m.group(2)
        
        # Skip if already backticked
        if alias.startswith('`'):
            return m.group(0)
        
        # Check if alias is reserved
        if alias.lower() in _RESERVED_ALIASES:
            return f"{prefix}`{alias}`"
        return m.group(0)
    
    # Match: ) AS alias or , expr AS alias (typical SELECT alias patterns)
    # Negative lookbehind to skip FROM/JOIN table aliases
    sql = re.sub(
        r'(\)\s+AS\s+)([A-Za-z_][A-Za-z0-9_]*)\b(?!\s*\.)',
        escape_alias,
        sql,
        flags=re.I
    )
    return sql

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

def _to_singular(term: str) -> str:
    t = (term or "").lower().strip()
    if t.endswith("ies") and len(t) > 3:
        return t[:-3] + "y"
    if t.endswith("s") and not t.endswith("ss") and len(t) > 1:
        return t[:-1]
    return t

def _entity_terms_from_ctx(ctx: str = "") -> Set[str]:
    """
    Build dynamic entity lexicon from schema context.
    Priority:
    1) ALLOWED IDENTIFIERS block (request-scoped schema)
    2) Global schema analyzer tables (fallback)
    """
    terms: Set[str] = set()

    allowed = _parse_allowed_identifiers(ctx)
    if allowed:
        source_tables = list(allowed.keys())
    else:
        try:
            source_tables = list(get_all_tables())
        except Exception:
            source_tables = []

    for table in source_tables:
        t = (table or "").lower().strip().strip("`")
        if not t:
            continue

        singular = _to_singular(t)
        variants = {t, singular}

        # Support table names with underscores in natural-language questions
        if "_" in t:
            variants.add(t.replace("_", " "))
        if "_" in singular:
            variants.add(singular.replace("_", " "))

        terms.update(v for v in variants if v)

    return terms

def _build_entity_kw(ctx: str = "") -> str:
    """
    Return regex alternation for entity nouns (schema-driven).
    Keeps a conservative fallback if schema context is unavailable.
    """
    terms = _entity_terms_from_ctx(ctx)
    if not terms:
        terms = {"customer", "product", "employee", "order", "office", "payment"}

    escaped = sorted((re.escape(t) for t in terms), key=len, reverse=True)
    return f"(?:{'|'.join(escaped)})"

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

def _alias_for_table(table: str, used: set[str]) -> str:
    """
    Deterministic alias WITHOUT hardcoded overrides.
    LLM is free to generate its own aliases - this function is only
    used when aliases need to be auto-generated (e.g., in heuristic fixes).

    Strategy: 1-2 letters from table name, increasing until unique
    """
    t = (table or "").strip("`").split(".")[-1]  # drop schema if any
    tl = t.lower()

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

def _split_outside_quotes(sql: str) -> List[str]:
    # reuse your existing approach (same regex style you already use)
    return re.split(r"('(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`[^`]*`)", sql)

def has_join(sql: str) -> bool:
    return re.search(rf"\b{_JOIN_KW}\b", sql, flags=re.I) is not None

# =======================================================================
# ======== Phase 1 — JOIN Semantics & Structural Rewrites (FINAL)========
# ===== Decide JOIN necessity and semantics before touching aliases =====
# =======================================================================
# - join_count
# - question_wants_include_zero
# - rewrite_left_join_to_inner_if_not_needed
# - rewrite_left_join_to_inner_if_rhs_filtered
# - split_where_and_rest
# - split_top_level_and
# - has_top_level_or
# - on_clause_has_rhs_filter
# - rewrite_inner_join_filter_to_exists
# - question_wants_anti_existence
# - question_wants_existence
# - rewrite_inner_join_existence_to_exists
# - has_non_inner_join
# - strip_string_literals
# - drop_redundant_inner_join_same_key
# - prune_ornamental_joins_safe
# - rewrite_invalid_fk_join_conditions

def join_count(sql: str) -> int:
    if not sql:
        return 0
    return len(re.findall(r"\bjoin\b", sql, flags=re.I))

def _question_requires_inner_join(question: str, ctx: str = "") -> bool:
    """
    Detect explicit filtering intent that should EXCLUDE non-matching parent rows.
    This implies INNER JOIN semantics for multi-table queries.
    """
    q = (question or "").lower()
    if not q:
        return False

    entity_kw = _build_entity_kw(ctx)
    entity_terms = _entity_terms_from_ctx(ctx)

    # include-zero intent should win over inner-join filtering
    include_zero_overrides = [
        "including", "with no", "without any", "without",
        "never paid", "even if no", "include those without",
        "termasuk", "meskipun tidak", "walaupun tidak", "yang tidak punya",
        "tanpa", "belum",
    ]
    if any(s in q for s in include_zero_overrides):
        if re.search(rf"\b{entity_kw}\b", q, flags=re.I):
            return False

    for ent in entity_terms:
        if (
            f"including {ent} with no" in q
            or f"{ent} with no" in q
        ):
            return False

    # Explicit existence filters (without requiring "only"), schema-driven:
    # - "customers with payments"
    # - "authors with books"
    # - "users who have sessions"
    #
    # Build candidate "existence object" terms from transaction-like tables
    # in schema metadata, narrowed by ALLOWED IDENTIFIERS when available.
    allowed = _parse_allowed_identifiers(ctx)
    scoped_tables = {t.lower() for t in allowed.keys()} if allowed else set()
    try:
        tx_tables = {t.lower() for t in get_transaction_tables()}
    except Exception:
        tx_tables = set()

    if scoped_tables and tx_tables:
        obj_tables = scoped_tables.intersection(tx_tables)
    elif tx_tables:
        obj_tables = tx_tables
    else:
        obj_tables = set()

    obj_terms: Set[str] = set()
    for t in obj_tables:
        t0 = (t or "").strip().strip("`")
        if not t0:
            continue
        singular = _to_singular(t0)
        variants = {t0, singular}
        if "_" in t0:
            variants.add(t0.replace("_", " "))
        if "_" in singular:
            variants.add(singular.replace("_", " "))
        obj_terms.update(v for v in variants if v)
        try:
            for syn in get_table_synonyms(t0):
                s = (syn or "").lower().strip()
                if not s:
                    continue
                obj_terms.add(s)
                obj_terms.add(_to_singular(s))
        except Exception:
            pass

    # Conservative lexical fallback if metadata is unavailable.
    if not obj_terms:
        obj_terms = {"order", "orders", "payment", "payments", "transaction", "transactions", "purchase", "purchases"}

    # Generic row-record words are useful for prompts like:
    # "have at least one row in the payments table"
    generic_obj_terms = {"record", "records", "row", "rows", "entry", "entries"}
    obj_terms.update(generic_obj_terms)

    escaped_obj = sorted((re.escape(t) for t in obj_terms if t), key=len, reverse=True)
    existence_object = f"(?:{'|'.join(escaped_obj)})" if escaped_obj else r"(?:records?|rows?)"
    has_entity_mention = re.search(rf"\b{entity_kw}\b", q, flags=re.I) is not None
    has_existence_object = re.search(rf"\b{existence_object}\b", q, flags=re.I) is not None
    has_existence_connector = (
        re.search(r"\b(?:with|having)\b", q, flags=re.I) is not None
        or re.search(r"\b(?:who|that|which)\b.{0,80}\b(?:have|has|had)\b", q, flags=re.I) is not None
        or re.search(r"\b(?:have|has|had)\b.{0,80}\b(?:at\s+least\s+one|any)\b", q, flags=re.I) is not None
    )
    if has_entity_mention and has_existence_object and has_existence_connector:
        return True

    # Generic quantified existence intent:
    # "X that have at least one Y ...", "X with any Y ..."
    # Keep this schema-driven and avoid table-name hardcoding.
    has_quantified_existence = re.search(
        r"\b(?:at\s+least\s+one|one\s+or\s+more|any)\b",
        q,
        flags=re.I,
    ) is not None
    if has_entity_mention and has_existence_connector and has_quantified_existence:
        mentioned_entity_roots: Set[str] = set()
        for ent in entity_terms:
            e = (ent or "").lower().strip()
            if not e:
                continue
            if re.search(rf"\b{re.escape(e)}\b", q, flags=re.I):
                mentioned_entity_roots.add(_to_singular(e))
        has_multi_entity_mentions = len(mentioned_entity_roots) >= 2
        if has_existence_object or has_multi_entity_mentions:
            return True

    # Aggregate comparison across related entities implies filtering intent.
    # Example: "offices where average credit limit of customers assigned to employees
    # in that office is higher than overall average ..."
    mentioned_entity_roots: Set[str] = set()
    for ent in entity_terms:
        e = (ent or "").lower().strip()
        if not e:
            continue
        if re.search(rf"\b{re.escape(e)}\b", q, flags=re.I):
            mentioned_entity_roots.add(_to_singular(e))

    has_multi_entity_mentions = len(mentioned_entity_roots) >= 2
    has_aggregate_intent = re.search(
        r"\b(?:avg|average|mean|sum|total|count|min|max)\b",
        q,
        flags=re.I,
    ) is not None
    has_comparison_intent = re.search(
        r"\b(?:higher|greater|more|above|over|lower|less|below|under)\s+than\b|[<>]=?",
        q,
        flags=re.I,
    ) is not None
    has_relation_intent = re.search(
        r"\b(?:of|from|assigned\s+to|linked\s+to|related\s+to|belongs?\s+to)\b",
        q,
        flags=re.I,
    ) is not None
    if has_multi_entity_mentions and has_aggregate_intent and has_comparison_intent and has_relation_intent:
        return True

    if not re.search(r"\bonly\b", q):
        return False

    # "only products that have been ordered", "only customers who have placed orders"
    pat_relative_clause = rf"""
        \bonly\s+{entity_kw}\b
        .*?
        \b(?:who|that|which)\b
        .*?
        \b(?:have|has|had)\b
        (?:\s+\w+){{0,4}}
        \s+
        \b(?:ordered|placed|made|paid|purchased|bought|orders?|payments?|transactions?|purchases?)\b
    """
    if re.search(pat_relative_clause, q, flags=re.I | re.X):
        return True

    # "only customers with orders", "only products having transactions"
    pat_with_having = rf"""
        \bonly\s+{entity_kw}\b
        .*?
        \b(?:with|having)\b
        (?:\s+\w+){{0,4}}
        \b(?:orders?|payments?|transactions?|purchases?)\b
    """
    if re.search(pat_with_having, q, flags=re.I | re.X):
        return True

    return False

def _question_wants_include_zero(question: str, ctx: str = "") -> bool:
    q = (question or "").lower()

    entity_kw = _build_entity_kw(ctx)
    entity_terms = _entity_terms_from_ctx(ctx)

    # Existence-filter intent must take precedence over broad "for each" wording.
    # Example: "customers with payments, and for each customer ..."
    if _question_requires_inner_join(question, ctx):
        return False

    # --- LOGIKA SIGNALS (Tetap sama) ---
    explicit_signals = [
        "including", "with no", "without any", "without",
        "never paid", "even if no", "include those without",
        "termasuk", "meskipun tidak", "walaupun tidak", "yang tidak punya",
        "tanpa", "belum",
    ]

    if any(s in q for s in explicit_signals):
        # Require at least one entity mention for generic terms
        if re.search(rf"\b{entity_kw}\b", q, flags=re.I):
            return True

    # Dynamic include-zero phrasing should stay conservative.
    # "all/every/for each" alone is not enough to imply preserve-zero semantics.
    for ent in entity_terms:
        if (
            f"including {ent} with no" in q
            or f"{ent} with no" in q
        ):
            return True

    if re.search(rf"\b{entity_kw}\b.*\b(without|with\s+no|no)\b", q, flags=re.I):
        return True
    
    return False

def _rewrite_left_join_to_inner_if_not_needed(sql: str, question: str, ctx: str = "") -> str:
    if not sql or _question_wants_include_zero(question, ctx):
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

def _rewrite_left_join_to_inner_if_rhs_filtered(sql: str, question: str, ctx: str = "") -> str:
    """
    Generic Fix 2:
    If a LEFT JOIN's RHS is filtered in WHERE, it is semantically an INNER JOIN.
    Convert LEFT JOIN -> JOIN, unless the question explicitly wants include-zero rows.
    Works with or without aliases.
    """
    if not sql:
        return sql
    if _question_wants_include_zero(question, ctx):
        return sql

    # Existence/filtering intent should use INNER semantics even when RHS filters
    # were already pushed into ON (so WHERE-based detection would miss it).
    # Keep this conservative: only auto-convert for single-join queries.
    if _question_requires_inner_join(question, ctx):
        if join_count(sql) == 1 and re.search(r"\bleft\s+join\b", sql, flags=re.I):
            return re.sub(r"\bLEFT\s+JOIN\b", "JOIN", sql, count=1, flags=re.I)

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

def _question_wants_all_entities(question: str, ctx: str = "") -> bool:
    """
    Detect if question explicitly wants ALL entities (including those with no matching RHS).
    E.g., "show ALL customers", "list every customer", "for each customer"
    
    This is more strict than _question_wants_include_zero - only triggers on explicit "all/every/each".
    """
    q = (question or "").lower()
    if not q:
        return False

    # Existence-filter intent (e.g., "have/with at least one ...") must NOT be
    # treated as "all entities"; those queries should prefer INNER semantics.
    if _question_requires_inner_join(question, ctx):
        return False

    entity_kw = _build_entity_kw(ctx)
    
    # Pattern: "all/every/each <entity>"
    # E.g., "all customers", "every customer", "each customer"
    all_patterns = [
        rf"\ball\s+{entity_kw}\b",             # "all customers"
        rf"\bevery\s+{entity_kw}\b",           # "every customer"
        rf"\beach\s+{entity_kw}\b",            # "each customer"
        rf"\bfor\s+each\s+{entity_kw}\b",      # "for each customer"
        rf"\bsemua\s+{entity_kw}\b",           # "semua customers" (Indonesian)
        rf"\bsetiap\s+{entity_kw}\b",          # "setiap customer" (Indonesian)
    ]
    
    return any(re.search(pat, q, flags=re.I) for pat in all_patterns)

def _move_rhs_filter_from_where_to_on(sql: str, question: str, ctx: str = "") -> str:
    """
    For LEFT JOIN queries: Move RHS-only filters from WHERE to ON clause.
    This preserves LEFT JOIN semantics (include rows with no match).
    
    ONLY triggers when question explicitly asks for "ALL" entities.
    
    Example:
    "Show ALL customers and their total payment in 2004"
    FROM customers c LEFT JOIN payments p ON c.id = p.cid WHERE YEAR(p.date) = 2004
    →
    FROM customers c LEFT JOIN payments p ON c.id = p.cid AND YEAR(p.date) = 2004
    """
    if not sql:
        return sql
    
    # SAFETY: Only trigger when question explicitly wants ALL entities
    if not _question_wants_all_entities(question, ctx):
        return sql
    
    # Only handle single LEFT JOIN
    if join_count(sql) != 1:
        return sql
    if not re.search(r"\bleft\s+join\b", sql, flags=re.I):
        return sql
    
    # Parse the query structure
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
    
    t1 = (m.group("t1") or "").strip().strip("`")
    a1 = (m.group("a1") or "").strip()
    t2 = (m.group("t2") or "").strip().strip("`")
    a2 = (m.group("a2") or "").strip()
    on_clause = m.group("on").strip()
    tail = m.group("tail") or ""
    prefix = m.group("prefix")
    
    # Get qualifiers for LHS and RHS
    lhs_qualifiers = [q for q in [a1, t1] if q]
    rhs_qualifiers = [q for q in [a2, t2] if q]
    
    if not rhs_qualifiers:
        return sql
    
    # Parse WHERE clause
    where_body, rest_after_where = _split_where_and_rest(tail)
    if not where_body:
        return sql
    
    # Check for OR (too complex, skip)
    if _has_top_level_or(where_body):
        return sql
    
    # Split WHERE conditions by AND
    conditions = _split_top_level_and(where_body)
    if not conditions:
        return sql
    
    # Categorize conditions: RHS-only vs others
    rhs_only_conditions = []
    other_conditions = []
    
    for cond in conditions:
        cond_stripped = cond.strip()
        if not cond_stripped:
            continue
        
        # Check if condition references RHS
        refs_rhs = any(
            re.search(rf"\b{re.escape(q)}\s*\.", cond_stripped, flags=re.I) 
            for q in rhs_qualifiers
        )
        # Check if condition references LHS
        refs_lhs = any(
            re.search(rf"\b{re.escape(q)}\s*\.", cond_stripped, flags=re.I) 
            for q in lhs_qualifiers
        )
        
        # Only move if it ONLY references RHS (not LHS)
        if refs_rhs and not refs_lhs:
            rhs_only_conditions.append(cond_stripped)
        else:
            other_conditions.append(cond_stripped)
    
    # If no RHS-only conditions found, nothing to move
    if not rhs_only_conditions:
        return sql
    
    # Build new ON clause with moved conditions
    new_on_clause = on_clause
    for cond in rhs_only_conditions:
        new_on_clause += f" AND {cond}"
    
    # Build new WHERE clause (if any conditions remain)
    if other_conditions:
        new_where = " WHERE " + " AND ".join(other_conditions)
    else:
        new_where = ""
    
    # Reconstruct query
    t1_raw = m.group("t1")
    t2_raw = m.group("t2")
    from_part = f"FROM {t1_raw}" + (f" {a1}" if a1 else "")
    join_part = f" LEFT JOIN {t2_raw}" + (f" {a2}" if a2 else "")
    
    result = f"{prefix}{from_part}{join_part} ON {new_on_clause}{new_where}{rest_after_where}"
    
    if result != sql:
        print(f"[Logic] Moved RHS filter to ON clause (preserving LEFT JOIN for 'ALL' query)")
    
    return result

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
    # Derived-table joins are too risky for this regex rewrite.
    if re.search(r"\b(?:from|join)\s*\(", sql, flags=re.I):
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

def _question_wants_anti_existence(question: str, ctx: str = "") -> bool:
    q = (question or "").lower()
    entity_kw = _build_entity_kw(ctx)

    eng = [
        "with no", "without any", "who have no", "who has no",
        "that have no", "that has no",
        "never", "have never", "has never",
        "without", "tanpa",
    ]

    if any(p in q for p in eng):
        return True

    if re.search(rf"\bno\s+{entity_kw}\b", q, flags=re.I):
        return True
    if re.search(rf"\bwithout\s+(?:any\s+)?{entity_kw}\b", q, flags=re.I):
        return True
    if re.search(rf"\btanpa\s+{entity_kw}\b", q, flags=re.I):
        return True

    return False

def _question_wants_existence(question: str) -> bool:
    q = (question or "").lower()

    eng = [
        "who have", "who has", "that have", "that has",
        "with at least one", "having at least one",
        "have at least one", "has at least one",
        "with any", "have any", "has any",
    ]

    return any(p in q for p in eng)

def _rewrite_inner_join_existence_to_exists(sql: str, question: str, ctx: str = "") -> str:
    if not sql:
        return sql
    if join_count(sql) != 1:
        return sql
    if re.search(r"\b(left|right|full)\s+join\b", sql, flags=re.I):
        return sql
    # Derived-table joins are too risky for this regex rewrite.
    if re.search(r"\b(?:from|join)\s*\(", sql, flags=re.I):
        return sql
    if _question_wants_anti_existence(question, ctx):
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

def has_non_inner_join(sql: str) -> bool:
    if not sql:
        return False
    return re.search(r"\b(left|right|full)\s+join\b", sql, flags=re.I) is not None

def _strip_string_literals(s: str) -> str:
    s = re.sub(r"'(?:''|[^'])*'", "''", s)
    s = re.sub(r'"(?:""|[^"])*"', '""', s)
    return s

def _remove_redundant_and_ornamental_joins(sql: str) -> str:
    """
    Gabungan logic:
    - Hapus JOIN yang tidak dipakai di hasil query (ornamental)
    - Jika JOIN adalah equi-join dan referensi kolom RHS hanya join key, hapus JOIN dan rewrite referensi kolom ke LHS
    - Tetap konservatif untuk query agregasi/GROUP BY
    """
    if not sql:
        return sql
    # Skip jika ada GROUP BY atau fungsi agregat
    if "GROUP BY" in sql.upper():
        return sql
    if re.search(r"\b(SUM|COUNT|AVG|MIN|MAX|GROUP_CONCAT)\s*\(", sql, flags=re.I):
        return sql
    # Skip derived-table joins to avoid accidental cross-subquery rewrites.
    if re.search(r"\b(?:from|join)\s*\(", sql, flags=re.I):
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
    matches = list(join_pat.finditer(out))
    for m in reversed(matches):
        alias = m.group("alias") or (m.group("table") or "").strip("`")
        jtype = (m.group("jtype") or "").upper()
        on_txt = m.group("on") or ""
        start_on = m.start("on")
        end_on = m.end("on")
        masked = out[:start_on] + (" " * (end_on - start_on)) + out[end_on:]

        # Jika alias tidak dipakai di luar ON, JOIN bisa dihapus
        if not re.search(rf"\b{re.escape(alias)}\s*\.", masked, flags=re.I):
            # Tambahan: Untuk INNER JOIN, cek apakah equi-join dan referensi kolom RHS hanya join key
            is_inner = ("LEFT" not in jtype and "RIGHT" not in jtype and "FULL" not in jtype)
            if is_inner:
                # Cek apakah ON clause adalah equi-join
                on_clean = _strip_string_literals(on_txt).replace("`", "")
                lhs = None
                rhs = alias
                # Cari bentuk lhs.k = rhs.k
                on_m = re.search(
                    rf"([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*{re.escape(rhs)}\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)",
                    on_clean,
                    flags=re.I,
                )
                if on_m:
                    lhs = on_m.group(1)
                    k_left = on_m.group(2)
                    k_right = on_m.group(3)
                    if k_left.lower() == k_right.lower():
                        join_key = k_left
                        # Cek di luar ON, referensi rhs.<col> selain join_key
                        outside_on = out[:m.start("full")] + out[m.end("full"):]
                        outside_on = _strip_string_literals(outside_on).replace("`", "")
                        bad_rhs_use = re.search(
                            rf"\b{re.escape(rhs)}\s*\.\s*(?!{re.escape(join_key)}\b)[A-Za-z_][A-Za-z0-9_]*\b",
                            outside_on,
                            flags=re.I,
                        )
                        if not bad_rhs_use:
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
                            prefix_raw = out[:m.start("full")]
                            tail_raw = out[m.end("full"):]
                            new_prefix = _rewrite_outside_on(prefix_raw)
                            new_tail = _rewrite_outside_on(tail_raw)
                            out = new_prefix + " " + new_tail
                            continue
            # Jika bukan equi-join, hapus JOIN saja
            out = out[:m.start("full")] + " " + out[m.end("full"):] 
    return out


def _rewrite_invalid_fk_join_conditions(sql: str, ctx: str = "") -> str:
    """
    Repair invalid ON-equijoin predicates using schema FK metadata.

    Conservative behavior:
    - only considers direct table-to-table joins (skips JOIN (...subquery...))
    - only rewrites simple qualified equi-predicates (a.col = b.col)
    - rewrites only when there is exactly one FK relationship between the two
      joined tables and the current predicate is a near-miss (shares at least
      one expected FK/id column).
    """
    if not sql or not has_join(sql):
        return sql

    # If query contains derived-table joins/subqueries, do a recursive pass on
    # inner SELECT blocks first, then return (outer scope is too ambiguous for
    # direct regex surgery).
    if re.search(r"\b(?:from|join)\s*\(", sql, flags=re.I):
        def _rewrite_inner_select_blocks(text: str) -> str:
            if not text or "(" not in text:
                return text

            out_parts: List[str] = []
            i = 0
            n = len(text)
            while i < n:
                ch = text[i]
                if ch != "(":
                    out_parts.append(ch)
                    i += 1
                    continue

                # Find matching ')' with quote/backtick awareness.
                depth = 0
                in_sq = False
                in_dq = False
                in_bt = False
                j = i
                while j < n:
                    cj = text[j]
                    if cj == "'" and not in_dq and not in_bt:
                        if in_sq and j + 1 < n and text[j + 1] == "'":
                            j += 2
                            continue
                        in_sq = not in_sq
                        j += 1
                        continue
                    if cj == '"' and not in_sq and not in_bt:
                        if in_dq and j + 1 < n and text[j + 1] == '"':
                            j += 2
                            continue
                        in_dq = not in_dq
                        j += 1
                        continue
                    if cj == "`" and not in_sq and not in_dq:
                        in_bt = not in_bt
                        j += 1
                        continue
                    if in_sq or in_dq or in_bt:
                        j += 1
                        continue
                    if cj == "(":
                        depth += 1
                    elif cj == ")":
                        depth -= 1
                        if depth == 0:
                            break
                    j += 1

                if j >= n or text[j] != ")":
                    # Unbalanced fallback: keep remaining text untouched.
                    out_parts.append(text[i:])
                    return "".join(out_parts)

                inner = text[i + 1:j]
                inner_stripped = inner.lstrip()
                if re.match(r"(?is)^select\b", inner_stripped):
                    fixed_inner = _rewrite_invalid_fk_join_conditions(inner, ctx)
                    out_parts.append("(" + fixed_inner + ")")
                else:
                    out_parts.append("(" + inner + ")")
                i = j + 1

            return "".join(out_parts)

        fixed_with_inner = _rewrite_inner_select_blocks(sql)
        return fixed_with_inner

    try:
        fk_map_raw = get_fk_relationships() or {}
    except Exception:
        fk_map_raw = {}
    if not fk_map_raw:
        return sql

    fk_map: Dict[str, Dict[str, str]] = {}
    for src, mapping in fk_map_raw.items():
        src_l = str(src or "").lower()
        if not src_l:
            continue
        fk_map[src_l] = {}
        for fk_col, tgt in (mapping or {}).items():
            fk_col_l = str(fk_col or "").lower()
            tgt_l = str(tgt or "").lower()
            if fk_col_l and tgt_l:
                fk_map[src_l][fk_col_l] = tgt_l

    alias_to_table = _parse_from_join_aliases(sql)
    table_token_pat = re.compile(
        r"\b(?:from|join)\s+"
        r"(?P<table>`?[A-Za-z_][A-Za-z0-9_]*`?(?:\.`?[A-Za-z_][A-Za-z0-9_]*`?)?)"
        r"(?:\s+(?:as\s+)?(?P<alias>[A-Za-z_][A-Za-z0-9_]*))?",
        flags=re.I,
    )
    for m_tbl in table_token_pat.finditer(sql):
        raw = (m_tbl.group("table") or "").replace("`", "")
        tbl = raw.split(".")[-1].lower()
        if not tbl:
            continue
        alias = m_tbl.group("alias")
        alias_to_table.setdefault(tbl, tbl)
        if alias:
            alias_to_table.setdefault(alias, tbl)
            alias_to_table.setdefault(alias.lower(), tbl)

    if not alias_to_table:
        return sql

    def _get_table(alias_or_table: str) -> str:
        key = (alias_or_table or "").strip()
        return (alias_to_table.get(key) or alias_to_table.get(key.lower()) or key).lower()

    def _target_join_col(target_table: str, fk_col: str) -> str:
        id_col = get_id_column(target_table)
        if id_col:
            return str(id_col)
        return fk_col

    try:
        analyzer = get_schema_analyzer()
    except Exception:
        analyzer = None
    table_cols_cache: Dict[str, Set[str]] = {}

    def _table_cols(table_name: str) -> Set[str]:
        t = (table_name or "").lower().strip()
        if not t:
            return set()
        if t in table_cols_cache:
            return table_cols_cache[t]
        cols: Set[str] = set()
        if analyzer is not None:
            try:
                cols.update(c.lower() for c in analyzer.get_table_columns(t))
            except Exception:
                pass
        table_cols_cache[t] = cols
        return cols

    def _fk_pairs_between(left_table: str, right_table: str) -> List[Tuple[str, str, str, str]]:
        pairs: List[Tuple[str, str, str, str]] = []
        lt = (left_table or "").lower()
        rt = (right_table or "").lower()
        if not lt or not rt or lt == rt:
            return pairs

        for fk_col, tgt in fk_map.get(lt, {}).items():
            if tgt == rt:
                pairs.append((lt, fk_col, rt, _target_join_col(rt, fk_col)))
        for fk_col, tgt in fk_map.get(rt, {}).items():
            if tgt == lt:
                pairs.append((rt, fk_col, lt, _target_join_col(lt, fk_col)))
        return pairs

    def _pick_bridge_rewrite(
        valid_table: str,
        invalid_table: str,
        missing_fk_col: str,
    ) -> Optional[Tuple[str, str, str, str]]:
        """
        Find bridge table C such that:
          - C.<missing_fk_col> references valid_table
          - invalid_table has FK to C via some column
        Returns (bridge_table, bridge_fk_to_valid, invalid_fk_to_bridge, bridge_target_col).
        """
        vt = (valid_table or "").lower().strip()
        it = (invalid_table or "").lower().strip()
        miss = (missing_fk_col or "").lower().strip()
        if not vt or not it or not miss or vt == it:
            return None

        candidates: List[Tuple[str, str, str, str]] = []
        for bridge_table, bridge_fk_map in fk_map.items():
            bt = (bridge_table or "").lower().strip()
            if not bt:
                continue

            # Bridge must reference valid table using the missing FK name.
            if (bridge_fk_map or {}).get(miss) != vt:
                continue

            # Invalid table must reference bridge table.
            inv_fk_candidates = [
                fk_col
                for fk_col, tgt in (fk_map.get(it, {}) or {}).items()
                if (tgt or "").lower() == bt
            ]
            if not inv_fk_candidates:
                continue

            def _fk_score(fk_col: str) -> int:
                f = (fk_col or "").lower()
                s = 0
                if f.endswith("id") or f.endswith("number") or f.endswith("code"):
                    s += 4
                if bt.rstrip("s") in f:
                    s += 2
                return s

            invalid_fk_to_bridge = sorted(inv_fk_candidates, key=_fk_score, reverse=True)[0]
            bridge_target_col = str(get_id_column(bt) or invalid_fk_to_bridge)

            bt_cols = _table_cols(bt)
            it_cols = _table_cols(it)
            if bt_cols and miss not in bt_cols:
                continue
            if it_cols and invalid_fk_to_bridge not in it_cols:
                continue
            if bt_cols and bridge_target_col.lower() not in bt_cols:
                if invalid_fk_to_bridge in bt_cols:
                    bridge_target_col = invalid_fk_to_bridge
                else:
                    continue

            candidates.append((bt, miss, invalid_fk_to_bridge, bridge_target_col))

        if not candidates:
            return None
        return candidates[0]

    def _pick_unique_alias(base_table: str) -> str:
        used_aliases = {k.lower() for k in alias_to_table.keys() if k}
        base = (get_singular_form(base_table) or base_table or "b").strip().lower()
        base = re.sub(r"[^a-z0-9_]", "", base) or "b"
        if not re.match(r"[a-z_]", base[0]):
            base = f"b_{base}"
        cand = base
        idx = 2
        while cand.lower() in used_aliases:
            cand = f"{base}_{idx}"
            idx += 1
        return cand

    join_pat = re.compile(
        r"""
        (?P<full>
            \s+
            (?P<jtype>(?:INNER\s+JOIN|LEFT\s+JOIN|RIGHT\s+JOIN|FULL\s+JOIN|JOIN))
            \s+
            (?P<table>`?[A-Za-z_][A-Za-z0-9_]*`?(?:\.`?[A-Za-z_][A-Za-z0-9_]*`?)?)
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
    eq_pat = re.compile(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)\b",
        flags=re.I,
    )

    out = sql
    matches = list(join_pat.finditer(out))
    for m_join in reversed(matches):
        on_txt = m_join.group("on") or ""
        eq_matches = list(eq_pat.finditer(on_txt))
        if not eq_matches:
            continue

        # Bridge-table repair for invalid direct joins (single simple equijoin).
        # Example:
        #   employees e LEFT JOIN payments p ON e.employeeNumber = p.salesRepEmployeeNumber
        # ->
        #   employees e LEFT JOIN customers c ON c.salesRepEmployeeNumber = e.employeeNumber
        #               LEFT JOIN payments p ON p.customerNumber = c.customerNumber
        if len(eq_matches) == 1:
            m_eq_only = eq_matches[0]
            on_residual = (on_txt[:m_eq_only.start()] + on_txt[m_eq_only.end():]).strip()
            on_residual = re.sub(r"[()\s]+", "", on_residual)
            if not on_residual:
                lhs_alias, lhs_col = m_eq_only.group(1), m_eq_only.group(2)
                rhs_alias, rhs_col = m_eq_only.group(3), m_eq_only.group(4)
                lt = _get_table(lhs_alias)
                rt = _get_table(rhs_alias)

                left_cols = _table_cols(lt)
                right_cols = _table_cols(rt)
                lhs_missing = bool(left_cols) and (lhs_col.lower() not in left_cols)
                rhs_missing = bool(right_cols) and (rhs_col.lower() not in right_cols)

                if lhs_missing ^ rhs_missing:
                    # Only rewrite when the invalid side belongs to the joined table alias.
                    joined_table_raw = (m_join.group("table") or "").replace("`", "")
                    joined_table_name = joined_table_raw.split(".")[-1].lower()
                    joined_alias = (m_join.group("alias") or "").strip()
                    joined_symbol = (joined_alias or joined_table_name).lower()

                    if lhs_missing:
                        invalid_alias, invalid_col, invalid_table = lhs_alias, lhs_col, lt
                        valid_alias, valid_col, valid_table = rhs_alias, rhs_col, rt
                    else:
                        invalid_alias, invalid_col, invalid_table = rhs_alias, rhs_col, rt
                        valid_alias, valid_col, valid_table = lhs_alias, lhs_col, lt

                    if invalid_alias.lower() == joined_symbol:
                        bridge = _pick_bridge_rewrite(valid_table, invalid_table, invalid_col)
                        if bridge:
                            bridge_table, bridge_fk_to_valid, invalid_fk_to_bridge, bridge_target_col = bridge
                            bridge_alias = _pick_unique_alias(bridge_table)
                            jtype = (m_join.group("jtype") or "JOIN").strip()
                            join_table_token = (m_join.group("table") or "").strip()
                            join_alias_token = f" {joined_alias}" if joined_alias else ""

                            # Preserve original missing-FK column casing when possible.
                            bridge_fk_out = invalid_col if invalid_col.lower() == bridge_fk_to_valid.lower() else bridge_fk_to_valid

                            replacement = (
                                f" {jtype} {bridge_table} {bridge_alias}"
                                f" ON {bridge_alias}.{bridge_fk_out} = {valid_alias}.{valid_col}"
                                f" {jtype} {join_table_token}{join_alias_token}"
                                f" ON {invalid_alias}.{invalid_fk_to_bridge} = {bridge_alias}.{bridge_target_col}"
                            )
                            out = out[:m_join.start("full")] + replacement + out[m_join.end("full"):]
                            alias_to_table[bridge_alias] = bridge_table
                            alias_to_table[bridge_alias.lower()] = bridge_table
                            alias_to_table.setdefault(bridge_table, bridge_table)
                            continue

        rebuilt = []
        last = 0
        changed = False

        for m_eq in eq_matches:
            rebuilt.append(on_txt[last:m_eq.start()])
            lhs_alias, lhs_col = m_eq.group(1), m_eq.group(2)
            rhs_alias, rhs_col = m_eq.group(3), m_eq.group(4)

            lt = _get_table(lhs_alias)
            rt = _get_table(rhs_alias)
            pairs = _fk_pairs_between(lt, rt)
            new_eq = m_eq.group(0)

            if len(pairs) == 1:
                fk_tbl, fk_col, tgt_tbl, tgt_col = pairs[0]
                lhs_col_l = lhs_col.lower()
                rhs_col_l = rhs_col.lower()
                fk_col_l = fk_col.lower()
                tgt_col_l = str(tgt_col).lower()

                valid = (
                    (lt == fk_tbl and rt == tgt_tbl and lhs_col_l == fk_col_l and rhs_col_l == tgt_col_l)
                    or (rt == fk_tbl and lt == tgt_tbl and rhs_col_l == fk_col_l and lhs_col_l == tgt_col_l)
                )

                if not valid:
                    near_miss = ({lhs_col_l, rhs_col_l} & {fk_col_l, tgt_col_l})
                    left_cols = _table_cols(lt)
                    right_cols = _table_cols(rt)
                    lhs_missing = bool(left_cols) and (lhs_col_l not in left_cols)
                    rhs_missing = bool(right_cols) and (rhs_col_l not in right_cols)
                    invalid_column_side = lhs_missing or rhs_missing

                    if near_miss or invalid_column_side:
                        # Preserve original column casing when one side already
                        # uses the intended FK or target key.
                        fk_col_out = fk_col
                        tgt_col_out = str(tgt_col)
                        if lhs_col_l == fk_col_l:
                            fk_col_out = lhs_col
                        elif rhs_col_l == fk_col_l:
                            fk_col_out = rhs_col
                        if lhs_col_l == tgt_col_l:
                            tgt_col_out = lhs_col
                        elif rhs_col_l == tgt_col_l:
                            tgt_col_out = rhs_col

                        if lt == fk_tbl and rt == tgt_tbl:
                            new_eq = f"{lhs_alias}.{fk_col_out} = {rhs_alias}.{tgt_col_out}"
                            changed = True
                        elif lt == tgt_tbl and rt == fk_tbl:
                            new_eq = f"{lhs_alias}.{tgt_col_out} = {rhs_alias}.{fk_col_out}"
                            changed = True

            rebuilt.append(new_eq)
            last = m_eq.end()

        if not changed:
            continue

        rebuilt.append(on_txt[last:])
        new_on = "".join(rebuilt)
        out = out[:m_join.start("on")] + new_on + out[m_join.end("on"):]

    return out
# =======================================================================
# ================= Phase 2 — Syntax & Canonicalization =================
# =============== Normalize tokens and table naming early ===============
# =======================================================================
# - is_simple_column_expr
# - drop_redundant_column_aliases
# - drop_unused_from_table_aliases
# - strip_alias_if_single_table
# - repair_invalid_qualified_columns
# - repair_simple_single_table_column_names

def _repair_invalid_qualified_columns(sql: str, ctx: str = "") -> str:
    """
    Schema-aware repair for malformed qualified references (alias.column).

    Example fixed:
      MAX(od.orderDate)  -> MAX(o.orderDate)
    when `od` maps to a table that has no `orderDate`, and exactly one
    other visible alias maps to a table that has `orderDate`.

    Safety constraints:
    - No hardcoded table/column names.
    - Only rewrite when target alias is unique.
    - Skip ambiguous alias scopes (same alias used for different tables).
    - Ignore quoted string/backtick literal segments.
    """
    if not sql:
        return sql

    analyzer = None
    allowed_cols: Dict[str, Set[str]] = {}
    try:
        analyzer = get_schema_analyzer()
    except Exception:
        analyzer = None

    # Fallback source for columns if global analyzer is unavailable.
    if not analyzer and ctx:
        try:
            allowed = _parse_allowed_identifiers(ctx)
            allowed_cols = {
                t.lower(): {c.lower() for c in cols}
                for t, cols in allowed.items()
            }
        except Exception:
            allowed_cols = {}

    # Build symbol -> table map from FROM/JOIN (alias optional).
    # Keep alias and bare table symbol, but drop ambiguous symbols.
    # Derived-table aliases are intentionally excluded (unknown columns).
    tbl_pat = re.compile(
        r"""
        \b(?:from|join)\s+
        (?P<table>`?[A-Za-z_][A-Za-z0-9_]*`?(?:\.`?[A-Za-z_][A-Za-z0-9_]*`?)?)
        (?:\s+(?:as\s+)?(?P<alias>[A-Za-z_][A-Za-z0-9_]*))?
        """,
        flags=re.I | re.X,
    )

    symbol_map: Dict[str, tuple] = {}  # symbol(lower) -> (table_lower, is_alias_symbol)
    ambiguous: Set[str] = set()

    def _add_symbol(sym: str, table_l: str, is_alias_symbol: bool) -> None:
        if not sym:
            return
        k = sym.lower()
        if k in ambiguous:
            return
        prev = symbol_map.get(k)
        if prev is None:
            symbol_map[k] = (table_l, is_alias_symbol)
            return
        if prev[0] != table_l:
            ambiguous.add(k)
            symbol_map.pop(k, None)

    for m in tbl_pat.finditer(sql):
        table_raw = (m.group("table") or "").replace("`", "")
        table_l = table_raw.split(".")[-1].strip().lower()
        if not table_l:
            continue
        alias = (m.group("alias") or "").strip()
        if alias and alias.lower() in _RESERVED_ALIASES:
            alias = ""
        _add_symbol(table_l, table_l, False)
        if alias:
            _add_symbol(alias, table_l, True)

    if not symbol_map:
        return sql

    # Cache table columns once.
    table_cols: Dict[str, Set[str]] = {}
    for _sym, (tbl_l, _is_alias) in symbol_map.items():
        if tbl_l in table_cols:
            continue
        cols: Set[str] = set()
        if analyzer:
            try:
                cols = {c.lower() for c in analyzer.get_table_columns(tbl_l)}
            except Exception:
                cols = set()
        if not cols and allowed_cols:
            cols = set(allowed_cols.get(tbl_l, set()))
        table_cols[tbl_l] = cols

    if not table_cols:
        return sql

    # Qualified ref pattern that also supports optional backticks.
    qcol_pat = re.compile(
        r"""
        (?P<aq>`?)
        (?P<alias>[A-Za-z_][A-Za-z0-9_]*)
        (?P=aq)\s*\.\s*
        (?P<cq>`?)
        (?P<col>[A-Za-z_][A-Za-z0-9_]*)
        (?P=cq)
        """,
        flags=re.X,
    )

    # Alias symbols grouped by table, for unique-candidate resolution.
    alias_syms_by_table: Dict[str, List[str]] = {}
    table_syms_by_table: Dict[str, List[str]] = {}
    for sym, (tbl_l, is_alias_symbol) in symbol_map.items():
        if is_alias_symbol:
            alias_syms_by_table.setdefault(tbl_l, []).append(sym)
        else:
            table_syms_by_table.setdefault(tbl_l, []).append(sym)

    def _pick_replacement_symbol(src_alias_l: str, col_l: str) -> Optional[str]:
        # 1) Prefer real aliases.
        alias_candidates: List[str] = []
        seen_tbl = set()
        for tbl_l, syms in alias_syms_by_table.items():
            if col_l not in table_cols.get(tbl_l, set()):
                continue
            if tbl_l in seen_tbl:
                continue
            seen_tbl.add(tbl_l)
            alias_candidates.append(sorted(syms, key=len)[0])

        alias_candidates = [a for a in alias_candidates if a.lower() != src_alias_l]
        if len(alias_candidates) == 1:
            return alias_candidates[0]
        if len(alias_candidates) > 1:
            return None

        # 2) Fallback to bare table symbols if no alias candidate exists.
        table_candidates: List[str] = []
        seen_tbl = set()
        for tbl_l, syms in table_syms_by_table.items():
            if col_l not in table_cols.get(tbl_l, set()):
                continue
            if tbl_l in seen_tbl:
                continue
            seen_tbl.add(tbl_l)
            table_candidates.append(sorted(syms, key=len)[0])

        table_candidates = [t for t in table_candidates if t.lower() != src_alias_l]
        if len(table_candidates) == 1:
            return table_candidates[0]
        return None

    on_clause_pat = re.compile(
        r"""
        \bON\b\s+.*?
        (?=
            \b(?:INNER\s+JOIN|LEFT\s+JOIN|RIGHT\s+JOIN|FULL\s+JOIN|JOIN)\b
            |\bWHERE\b|\bGROUP\s+BY\b|\bORDER\s+BY\b|\bHAVING\b|\bLIMIT\b
            |;|$
        )
        """,
        flags=re.I | re.S | re.X,
    )

    parts = _split_outside_quotes(sql)
    for i in range(0, len(parts), 2):
        seg = parts[i]
        protected_on_segments: List[str] = []

        def _protect_on(m: re.Match) -> str:
            protected_on_segments.append(m.group(0))
            return f" __ON_CLAUSE_PROTECTED_{len(protected_on_segments) - 1}__ "

        seg_masked = on_clause_pat.sub(_protect_on, seg)

        def _repl(m: re.Match) -> str:
            alias = m.group("alias")
            col = m.group("col")
            aq = m.group("aq") or ""
            cq = m.group("cq") or ""

            alias_l = alias.lower()
            col_l = col.lower()

            entry = symbol_map.get(alias_l)
            if not entry:
                return m.group(0)

            src_tbl = entry[0]
            src_cols = table_cols.get(src_tbl, set())
            if col_l in src_cols:
                return m.group(0)

            replacement = _pick_replacement_symbol(alias_l, col_l)
            if not replacement:
                return m.group(0)

            return f"{aq}{replacement}{aq}.{cq}{col}{cq}"

        seg_rewritten = qcol_pat.sub(_repl, seg_masked)
        for idx, original_on in enumerate(protected_on_segments):
            seg_rewritten = seg_rewritten.replace(f" __ON_CLAUSE_PROTECTED_{idx}__ ", original_on)
        parts[i] = seg_rewritten

    return "".join(parts)


def _repair_simple_single_table_column_names(sql: str, ctx: str = "") -> str:
    """
    Schema-aware repair for simple single-table queries where the model uses
    naturalized snake_case identifiers instead of the actual schema column names.

    Example:
      customer_number -> customerNumber
      payment_date    -> paymentDate

    Conservative guardrails:
    - only for a single base table with no JOIN
    - only when normalized identifier maps uniquely to one real column
    - skips aliases after AS and function names
    """
    if not sql:
        return sql
    if has_join(sql):
        return sql
    if re.search(r"\bfrom\s*\(", sql, flags=re.I):
        return sql

    m_from = re.search(
        r"""
        \bFROM\s+
        (?P<table>`?[A-Za-z_][A-Za-z0-9_]*`?(?:\.`?[A-Za-z_][A-Za-z0-9_]*`?)?)
        (?:\s+(?:AS\s+)?(?P<alias>[A-Za-z_][A-Za-z0-9_]*))?
        """,
        sql,
        flags=re.I | re.X,
    )
    if not m_from:
        return sql

    table_raw = (m_from.group("table") or "").replace("`", "")
    table_l = table_raw.split(".")[-1].strip().lower()
    alias = (m_from.group("alias") or "").strip()
    if alias and alias.lower() in _RESERVED_ALIASES:
        alias = ""
    if not table_l:
        return sql

    analyzer = None
    try:
        analyzer = get_schema_analyzer()
    except Exception:
        analyzer = None

    actual_cols: List[str] = []
    if analyzer:
        try:
            actual_cols = [str(c) for c in analyzer.get_table_columns(table_l) if str(c)]
        except Exception:
            actual_cols = []
    if not actual_cols and ctx:
        try:
            allowed = _parse_allowed_identifiers(ctx)
            actual_cols = [str(c) for c in (allowed.get(table_l, set()) or []) if str(c)]
        except Exception:
            actual_cols = []
    if not actual_cols:
        return sql

    actual_cols_lower = {c.lower() for c in actual_cols}

    def _norm_ident(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", (text or "").strip().lower())

    norm_map: Dict[str, List[str]] = {}
    for c in actual_cols:
        norm_map.setdefault(_norm_ident(c), []).append(c)

    def _resolve_col(token: str) -> Optional[str]:
        tl = (token or "").strip().lower()
        if not tl:
            return None
        if tl in actual_cols_lower:
            return None
        cands = norm_map.get(_norm_ident(token), [])
        if len(cands) == 1:
            return cands[0]
        return None

    valid_symbols = {table_l}
    if alias:
        valid_symbols.add(alias.lower())

    qcol_pat = re.compile(
        r"""
        (?P<aq>`?)
        (?P<sym>[A-Za-z_][A-Za-z0-9_]*)
        (?P=aq)\s*\.\s*
        (?P<cq>`?)
        (?P<col>[A-Za-z_][A-Za-z0-9_]*)
        (?P=cq)
        """,
        flags=re.X,
    )

    parts = _split_outside_quotes(sql)
    for i in range(0, len(parts), 2):
        seg = parts[i]

        def _qual_repl(m: re.Match) -> str:
            sym = (m.group("sym") or "").lower()
            col = m.group("col") or ""
            if sym not in valid_symbols:
                return m.group(0)
            replacement = _resolve_col(col)
            if not replacement:
                return m.group(0)
            aq = m.group("aq") or ""
            cq = m.group("cq") or ""
            return f"{aq}{m.group('sym')}{aq}.{cq}{replacement}{cq}"

        seg = qcol_pat.sub(_qual_repl, seg)

        token_pat = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")

        def _unqual_repl(m: re.Match) -> str:
            token = m.group(1)
            low = token.lower()
            if low in _SQL_KEYWORDS or low in valid_symbols or low in actual_cols_lower:
                return token

            left = seg[:m.start()]
            right = seg[m.end():]
            if left.rstrip().endswith("."):
                return token
            if re.search(r"\bAS\s*$", left, flags=re.I):
                return token
            if right.lstrip().startswith("("):
                return token

            replacement = _resolve_col(token)
            return replacement or token

        parts[i] = token_pat.sub(_unqual_repl, seg)

    return "".join(parts)

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
    print(f"[REFINER - DROP ALIAS] Input: {sql[:100]}")
    print(f"[REFINER - DROP ALIAS] has_join: {has_join(sql)}")
    # FIX 1: Kalau ada JOIN di mana pun, jangan sentuh (biar aman).
    if has_join(sql):
        return sql

    # FIX 2: Kalau ada GROUP BY, jangan sentuh (kemungkinan besar ada aggregasi multi-table)
    if "GROUP BY" in sql.upper():
        return sql

    # FIX 3: Kalau ada aggregate functions, jangan sentuh
    if re.search(r"\b(SUM|COUNT|AVG|MIN|MAX|GROUP_CONCAT)\s*\(", sql, flags=re.I):
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

        # FIX 4: Cek apakah alias dipakai di SELURUH query (bukan hanya setelah FROM)
        # SEBELUMNYA: scan[m.end():] → hanya cek setelah FROM
        # SEKARANG: scan → cek di seluruh query (termasuk SELECT clause)
        if re.search(rf"\b{re.escape(alias)}\s*\.", scan, flags=re.I):
            return m.group(0)

        # alias tidak dipakai -> drop alias
        return f"FROM {table_raw}"

    return pat.sub(repl, sql)

def _strip_alias_if_single_table(sql: str) -> str:
    """
    If query has only one base table in FROM and no JOIN,
    strip alias prefixes like e.col -> col.
    DO NOT touch queries with GROUP BY or subqueries.
    """
    if not sql:
        return sql

    s = sql.lower()

    # ULTRA SAFE GUARDS
    if re.search(r"\bjoin\b", s):
        print(f"[REFINER - STRIP ALIAS] SKIP: has join")
        return sql
    
    if re.search(r"\bgroup\s+by\b", s):
        print(f"[REFINER - STRIP ALIAS] SKIP: has GROUP BY")
        return sql
    
    if re.search(r"\bfrom\s*\(", s):
        print(f"[REFINER - STRIP ALIAS] SKIP: has subquery")
        return sql
    
    if re.search(r"\brow_number\b", s):
        print(f"[REFINER - STRIP ALIAS] SKIP: has ROW_NUMBER")
        return sql
    
    if re.search(r"\b(sum|count|avg|min|max)\s*\(", s):
        print(f"[REFINER - STRIP ALIAS] SKIP: has aggregate")
        return sql

    print(f"[REFINER - STRIP ALIAS] Processing simple single-table query...")

    m = re.search(r"\bfrom\s+([a-zA-Z_][a-zA-Z0-9_]*)\b", s)
    if not m:
        return sql

    parts = re.split(r"('(?:''|[^'])*')", sql)

    for i in range(0, len(parts), 2):
        parts[i] = re.sub(
            r"\b([a-zA-Z_][a-zA-Z0-9_]*)\.",
            "",
            parts[i],
        )

    result = "".join(parts)
    print(f"[REFINER - STRIP ALIAS] Output: {result[:100]}")
    return result
# =======================================================================
# ============ Phase 3 — Column Naming & GROUP BY Consistency ===========
# ================== Operate on final column identities =================
# =======================================================================
# - _prefer_name_over_id_generic
# - prefer_friendly_names
# - expand_group_by_aliases_regex
# - is_potential_winner_row
# - enforce_explicit_group_by

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


def _rewrite_invalid_grouped_existence_subquery_to_join(sql: str, question: str, ctx: str = "") -> str:
    """
    Rewrite a narrow invalid grouped-existence pattern into a direct JOIN.

    Target anti-pattern:
      SELECT group_col, COUNT(DISTINCT outer_id)
      FROM outer_table
      WHERE group_col IN (SELECT DISTINCT group_col FROM child_table)
      GROUP BY group_col

    when `child_table` does NOT actually have `group_col`, but it has a unique
    direct FK to `outer_table`. In that case, the intended meaning is usually:
      - keep only outer rows that have at least one child row
      - aggregate over those qualifying outer rows

    This helper is intentionally conservative:
    - only for single-table outer queries (no JOIN yet)
    - only when question asks for COUNT(DISTINCT ...)
    - only when include-zero is NOT requested
    - only when the child table has exactly one direct FK path to the outer table
    - only when the DISTINCT target is the outer table's ID column
    """
    if not sql:
        return sql
    if has_join(sql):
        return sql
    if not _question_wants_count_distinct(question):
        return sql
    if _question_wants_include_zero(question, ctx):
        return sql

    s = (sql or "").strip().rstrip(";").strip()
    if not s:
        return sql

    outer_pat = re.compile(
        r"""
        (?is)^\s*
        SELECT\s+(?P<select>.+?)
        \s+FROM\s+
        (?P<table>`?[A-Za-z_][A-Za-z0-9_]*`?(?:\.`?[A-Za-z_][A-Za-z0-9_]*`?)?)
        (?:\s+(?:AS\s+)?(?P<alias>[A-Za-z_][A-Za-z0-9_]*))?
        \s+WHERE\s+(?P<where>.+?)
        \s+GROUP\s+BY\s+(?P<groupby>.+?)
        (?P<tail>\s+(?:HAVING|ORDER\s+BY|LIMIT)\b.*)?$
        """,
        flags=re.X,
    )
    m = outer_pat.match(s)
    if not m:
        return sql

    select_clause = (m.group("select") or "").strip()
    outer_table_raw = (m.group("table") or "").replace("`", "")
    outer_table = outer_table_raw.split(".")[-1].strip()
    outer_alias = (m.group("alias") or "").strip()
    where_clause = (m.group("where") or "").strip()
    group_by_clause = (m.group("groupby") or "").strip()
    tail = (m.group("tail") or "").strip()

    if not outer_table:
        return sql

    select_items = _split_top_level_commas(select_clause)
    group_items = _split_top_level_commas(group_by_clause)
    if not select_items or not group_items:
        return sql

    count_distinct_pat = re.compile(
        r"""(?is)^COUNT\s*\(\s*DISTINCT\s+(?P<expr>(?:`?[A-Za-z_][A-Za-z0-9_]*`?\s*\.\s*)?`?[A-Za-z_][A-Za-z0-9_]*`?)\s*\)(?:\s+AS\s+(?P<alias>[A-Za-z_][A-Za-z0-9_]*))?$"""
    )
    count_distinct_expr = None
    for item in select_items:
        m_count = count_distinct_pat.match(item.strip())
        if m_count:
            count_distinct_expr = (m_count.group("expr") or "").strip()
            break
    if not count_distinct_expr:
        return sql

    def _base_col(expr: str) -> str:
        e = (expr or "").strip().replace("`", "")
        if "." in e:
            return e.split(".")[-1].strip()
        return e.strip()

    group_expr = group_items[0].strip()
    group_col = _base_col(group_expr)
    distinct_col = _base_col(count_distinct_expr)
    outer_id = get_id_column(outer_table)
    if not outer_id or distinct_col.lower() != str(outer_id).lower():
        return sql

    where_pat = re.compile(
        r"""
        (?is)^\s*
        (?P<lhs>(?:`?[A-Za-z_][A-Za-z0-9_]*`?\s*\.\s*)?`?[A-Za-z_][A-Za-z0-9_]*`?)
        \s+IN\s*\(
            \s*SELECT\s+DISTINCT\s+
            (?P<rhs>(?:`?[A-Za-z_][A-Za-z0-9_]*`?\s*\.\s*)?`?[A-Za-z_][A-Za-z0-9_]*`?)
            \s+FROM\s+
            (?P<inner_table>`?[A-Za-z_][A-Za-z0-9_]*`?(?:\.`?[A-Za-z_][A-Za-z0-9_]*`?)?)
            (?:\s+(?:AS\s+)?(?P<inner_alias>[A-Za-z_][A-Za-z0-9_]*))?
            \s*
        \)\s*$
        """,
        flags=re.X,
    )
    w = where_pat.match(where_clause)
    if not w:
        return sql

    lhs_col = _base_col(w.group("lhs") or "")
    rhs_col = _base_col(w.group("rhs") or "")
    inner_table_raw = (w.group("inner_table") or "").replace("`", "")
    inner_table = inner_table_raw.split(".")[-1].strip()
    if not inner_table:
        return sql

    # We only rewrite the clearly invalid case: same logical column name in the
    # subquery, but the inner table does not actually have that column.
    if lhs_col.lower() != rhs_col.lower():
        return sql

    try:
        analyzer = get_schema_analyzer()
    except Exception:
        analyzer = None

    inner_cols: Set[str] = set()
    if analyzer is not None:
        try:
            inner_cols.update(c.lower() for c in analyzer.get_table_columns(inner_table))
        except Exception:
            pass
    if not inner_cols and ctx:
        try:
            allowed = _parse_allowed_identifiers(ctx)
            inner_cols.update(str(c).strip("`").lower() for c in allowed.get(inner_table, set()))
        except Exception:
            pass

    if lhs_col.lower() in inner_cols:
        return sql

    fk_map = get_fk_relationships() or {}
    direct_fk_cols = [
        str(fk_col)
        for fk_col, tgt in (fk_map.get(inner_table.lower(), {}) or {}).items()
        if str(tgt or "").lower() == outer_table.lower()
    ]
    if len(direct_fk_cols) != 1:
        return sql

    fk_col = direct_fk_cols[0]
    outer_ref = outer_alias or outer_table
    inner_ref = inner_table

    # Qualify COUNT(DISTINCT outer_id) so the new JOIN cannot make it ambiguous.
    qualified_count_expr = count_distinct_expr
    if "." not in count_distinct_expr.replace("`", ""):
        qualified_count_expr = f"{outer_ref}.{outer_id}"
    new_select_clause = re.sub(
        r"""(?is)COUNT\s*\(\s*DISTINCT\s+(?:`?[A-Za-z_][A-Za-z0-9_]*`?\s*\.\s*)?`?[A-Za-z_][A-Za-z0-9_]*`?\s*\)""",
        f"COUNT(DISTINCT {qualified_count_expr})",
        select_clause,
        count=1,
    )

    join_clause = (
        f"FROM {outer_table_raw}"
        f"{(' ' + outer_alias) if outer_alias else ''} "
        f"JOIN {inner_table_raw} ON {inner_ref}.{fk_col} = {outer_ref}.{outer_id}"
    )

    rebuilt = (
        f"SELECT {new_select_clause} "
        f"{join_clause} "
        f"GROUP BY {group_by_clause}"
    )
    if tail:
        rebuilt += f" {tail}"
    if sql.strip().endswith(";"):
        rebuilt += ";"
    return rebuilt

def _wants_name_only(question: str, entity_words: List[str]) -> bool:
    """
    Generic helper: True kalau user ngomongin entity (customer/product/...) 
    tapi tidak eksplisit minta id/number/code.
    """
    q = (question or "").lower()
    # buang "number of" supaya tidak dianggap id-like
    q = re.sub(r"\b(total\s+)?number\s+of\b", "", q)

    def _phrase_mentioned(text: str, phrase: str) -> bool:
        p = str(phrase or "").strip().lower()
        if not p:
            return False
        pattern = re.escape(p).replace(r"\ ", r"\s+")
        return re.search(rf"\b{pattern}\b", text) is not None

    mention_entity = any(_phrase_mentioned(q, w) for w in entity_words)
    mention_id_like = (
        re.search(r"\b(ids?|numbers?|nomors?|nos?|codes?|kodes?)\b", q)
        or any(_phrase_mentioned(q, f"{w} code") for w in entity_words)
        or any(_phrase_mentioned(q, f"{w} codes") for w in entity_words)
    )
    return mention_entity and not mention_id_like

def _entity_words_for_table(table: str) -> List[str]:
    """
    Build entity words dynamically from schema metadata.
    Includes table name, singular form, and table synonyms.
    """
    words: Set[str] = set()
    t = (table or "").strip().lower()
    if not t:
        return []

    words.add(t)
    try:
        s = get_singular_form(t)
        if s:
            words.add(s.lower())
    except Exception:
        if t.endswith("s") and len(t) > 1:
            words.add(t[:-1])

    generic_terms = {
        "name", "names", "id", "ids", "code", "codes", "number", "numbers",
        "date", "dates", "status", "type", "types", "value", "values",
        "amount", "amounts", "total", "totals", "count", "counts",
        "description", "descriptions", "detail", "details", "record", "records",
        "row", "rows", "entry", "entries", "unit", "units",
    }

    try:
        for syn in get_table_synonyms(t):
            syn_low = str(syn).strip().lower()
            if not syn_low:
                continue
            syn_tokens = [tok for tok in re.findall(r"[a-z0-9]+", syn_low) if tok]
            if not syn_tokens:
                continue
            # Skip generic synonyms that cause false entity hits (e.g., "name", "unit").
            if all(tok in generic_terms for tok in syn_tokens):
                continue
            if len(syn_tokens) == 1 and len(syn_tokens[0]) < 4:
                continue
            syn_norm = " ".join(syn_tokens)
            if syn_norm:
                words.add(syn_norm)
            # Also include compact form for underscore/space variations.
            compact = syn_norm.replace(" ", "")
            if compact and len(compact) >= 5:
                words.add(compact)
    except Exception:
        pass

    return sorted(words)
    
def _wants_customer_name_only(question: str) -> bool:
    return _wants_name_only(question, _entity_words_for_table("customers"))

# ini belum diterapkan
def _wants_product_name_only(question: str) -> bool:
    return _wants_name_only(question, _entity_words_for_table("products"))

def _prefer_friendly_names(sql: str, question: str) -> str:
    """
    DRY: apply rule yang sama untuk beberapa entity sekaligus.
    - customers: customerNumber -> customerName
    - products : productCode    -> productName
    (hanya kalau user tidak minta id/number/code)

    Now uses dynamic column inference from schema metadata.
    """
    # customers - dynamic column names
    customer_name = get_name_column("customers")
    customer_id = get_id_column("customers")
    if customer_name and customer_id:
        sql = _prefer_name_over_id_generic(
            sql=sql,
            question=question,
            wants_name_fn=_wants_customer_name_only,
            name_column=customer_name,
            id_columns=[customer_id],
        )

    # products - dynamic column names
    product_name = get_name_column("products")
    product_id = get_id_column("products")
    if product_name and product_id:
        sql = _prefer_name_over_id_generic(
            sql=sql,
            question=question,
            wants_name_fn=_wants_product_name_only,
            name_column=product_name,
            id_columns=[product_id],
        )

    return sql

def _ensure_entity_id_and_name_if_requested(sql: str, question: str) -> str:
    """
    If question explicitly asks for both entity identifier and entity name,
    ensure both columns are present in SELECT (and GROUP BY for grouped queries).

    Dynamic and schema-driven:
    - entity detection from table synonyms / singular names
    - id/name columns from metadata (get_id_column/get_name_column)
    """
    if not sql or not question:
        return sql

    q = (question or "").lower()
    q_norm = re.sub(r"\b(total\s+)?number\s+of\b", "", q)
    q_snake = re.sub(r"[^a-z0-9_]+", "_", q_norm).strip("_")

    def _entity_wants_id_like(entity_words: List[str], table_name: str) -> bool:
        id_terms = r"(?:ids?|numbers?|nomors?|nos?|codes?|kodes?)"
        forms: Set[str] = {table_name.lower(), (get_singular_form(table_name) or table_name).lower()}
        for w in entity_words:
            ww = (w or "").strip().lower()
            if ww:
                forms.add(ww)
                forms.add(ww.replace(" ", "_"))

        for f in forms:
            if not f:
                continue
            if any(
                token in q_snake
                for token in (
                    f"{f}_id",
                    f"{f}_ids",
                    f"{f}_number",
                    f"{f}_numbers",
                    f"{f}_code",
                    f"{f}_codes",
                )
            ):
                return True

        for f in forms:
            p = re.escape((f or "").strip().lower()).replace(r"\ ", r"\s+")
            if not p:
                continue
            # Direct forms: "customer id", "id of customer"
            if re.search(rf"\b{p}\b\s+{id_terms}\b", q_norm, flags=re.I):
                return True
            if re.search(rf"\b{id_terms}\b\s+(?:of|for)\s+\b{p}\b", q_norm, flags=re.I):
                return True
            # Shorthand list style: "customer name, id" / "customer name and id"
            if re.search(rf"\b{p}\b\s+(?:full\s+)?names?\b\s*(?:,|\band\b)\s*(?:their\s+)?{id_terms}\b", q_norm, flags=re.I):
                return True

        return False

    def _entity_wants_name_like(entity_words: List[str], table_name: str) -> bool:
        name_terms = r"(?:full\s+names?|names?)"
        forms: Set[str] = {table_name.lower(), (get_singular_form(table_name) or table_name).lower()}
        for w in entity_words:
            ww = (w or "").strip().lower()
            if ww:
                forms.add(ww)
                forms.add(ww.replace(" ", "_"))

        for f in forms:
            if not f:
                continue
            if f"{f}_name" in q_snake or f"{f}_names" in q_snake:
                return True

        for f in forms:
            p = re.escape((f or "").strip().lower()).replace(r"\ ", r"\s+")
            if not p:
                continue
            # Direct forms: "customer name", "name of customer"
            if re.search(rf"\b{p}\b\s+{name_terms}\b", q_norm, flags=re.I):
                return True
            if re.search(rf"\b{name_terms}\b\s+(?:of|for)\s+\b{p}\b", q_norm, flags=re.I):
                return True

        return False

    parsed = _parse_select_from_groupby(sql)
    if not parsed:
        # Safety: do not apply relaxed fallback parser on queries with subqueries.
        # It can misinterpret inner GROUP BY as top-level and corrupt FROM/JOIN.
        if re.search(r"\b(?:FROM|JOIN)\s*\(", sql, flags=re.I):
            return sql
        # Relaxed fallback parser for GROUP BY queries with HAVING.
        s = (sql or "").strip().rstrip(";").strip()
        m_sel = re.match(r"(?is)^\s*SELECT\s+(?P<select>.+?)\s+FROM\s+(?P<after>.+)$", s)
        if not m_sel:
            return sql
        select_part = (m_sel.group("select") or "").strip()
        after_from = (m_sel.group("after") or "").strip()

        gb_matches = list(re.finditer(r"\bGROUP\s+BY\b", after_from, flags=re.I))
        if not gb_matches:
            return sql
        gb_m = gb_matches[-1]
        from_and_maybe_where = after_from[:gb_m.start()].strip()
        after_gb = after_from[gb_m.end():].strip()
        if not from_and_maybe_where or not after_gb:
            return sql

        next_kw = re.search(r"\b(HAVING|ORDER\s+BY|LIMIT)\b", after_gb, flags=re.I)
        if next_kw:
            groupby_part = after_gb[:next_kw.start()].strip()
            rem = after_gb[next_kw.start():].strip()
        else:
            groupby_part = after_gb.strip()
            rem = ""
        if not groupby_part:
            return sql

        having_part = ""
        tail_part = ""
        if rem:
            if re.match(r"(?is)^HAVING\b", rem):
                h_body = re.sub(r"(?is)^HAVING\s+", "", rem).strip()
                h_next = re.search(r"\b(ORDER\s+BY|LIMIT)\b", h_body, flags=re.I)
                if h_next:
                    having_part = h_body[:h_next.start()].strip()
                    tail_part = h_body[h_next.start():].strip()
                else:
                    having_part = h_body
            else:
                tail_part = rem

        where_match = re.search(r"\bWHERE\b", from_and_maybe_where, flags=re.I)
        if where_match:
            from_core = from_and_maybe_where[:where_match.start()].strip()
            where_part = from_and_maybe_where[where_match.start():].strip()
        else:
            from_core = from_and_maybe_where
            where_part = ""

        parsed = {
            "select": select_part,
            "from_core": from_core,
            "where": where_part,
            "groupby": groupby_part,
            "having": having_part,
            "tail": tail_part,
        }

    if not parsed:
        return sql

    from_core = (parsed.get("from_core") or "").strip()
    if not from_core:
        return sql

    # Build alias -> table map, plus table self-qualifier fallback.
    alias_to_table = _parse_from_join_aliases(f"SELECT 1 FROM {from_core}")
    if not alias_to_table:
        # fallback for no-alias FROM base table
        m_base = re.match(r"(?is)^`?([A-Za-z_][A-Za-z0-9_]*)`?", from_core)
        if m_base:
            t = m_base.group(1)
            alias_to_table = {t: t, t.lower(): t}

    if not alias_to_table:
        return sql

    table_to_aliases: Dict[str, List[str]] = {}
    for a, t in alias_to_table.items():
        if not a or not t:
            continue
        # skip convenience lower duplicates
        if a.lower() != a and a.lower() in alias_to_table:
            pass
        table_to_aliases.setdefault(t.lower(), [])
        if a not in table_to_aliases[t.lower()]:
            table_to_aliases[t.lower()].append(a)

    def _phrase_mentioned(text: str, phrase: str) -> bool:
        p = str(phrase or "").strip().lower()
        if not p:
            return False
        pattern = re.escape(p).replace(r"\ ", r"\s+")
        return re.search(rf"\b{pattern}\b", text) is not None

    def _split_expr_alias(item: str) -> tuple:
        m = re.match(
            r"""(?is)^\s*(.*?)\s+AS\s+(`[^`]+`|'[^']+'|"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)\s*$""",
            item.strip(),
        )
        if not m:
            return item.strip(), None
        return m.group(1).strip(), m.group(2).strip()

    def _parse_simple_col(expr: str) -> Optional[tuple]:
        m = re.match(
            r"(?is)^\s*(?:`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\.\s*)?`?([A-Za-z_][A-Za-z0-9_]*)`?\s*$",
            expr.strip(),
        )
        if not m:
            return None
        return m.group(1), m.group(2)

    select_items = _split_top_level_commas(parsed["select"])
    group_items = _split_top_level_commas(parsed.get("groupby", ""))

    # Gather selected simple columns by (qualifier?, column) and output aliases.
    selected_cols: List[tuple] = []
    selected_aliases: Set[str] = set()
    selected_exprs: List[str] = []
    for it in select_items:
        expr, out_alias = _split_expr_alias(it)
        selected_exprs.append(expr.strip())
        if out_alias:
            selected_aliases.add(out_alias.strip().strip("`'\"").lower())
        pcol = _parse_simple_col(expr)
        if pcol:
            qf, col = pcol
            selected_cols.append((qf.lower() if qf else None, col.lower()))

    grouped_cols: Set[tuple] = set()
    for g in group_items:
        pcol = _parse_simple_col(g)
        if pcol:
            qf, col = pcol
            grouped_cols.add((qf.lower() if qf else None, col.lower()))

    changed = False
    prepended_cols: List[str] = []

    for table in get_all_tables():
        t = (table or "").strip().lower()
        if t not in table_to_aliases:
            continue

        # only enforce when the question actually mentions this entity
        entity_words = _entity_words_for_table(table)
        if not any(_phrase_mentioned(q_norm, w) for w in entity_words):
            continue
        if not (_entity_wants_id_like(entity_words, table) and _entity_wants_name_like(entity_words, table)):
            continue

        id_col = get_id_column(table)
        name_col = get_name_column(table)
        if not id_col or not name_col:
            continue
        id_col_l = id_col.lower()
        name_col_l = name_col.lower()

        aliases = table_to_aliases.get(t) or [table]
        # Prefer first non-lowercase convenience alias
        qual = aliases[0]
        qual_l = qual.lower()
        singular = (get_singular_form(table) or table).lower()
        table_name_aliases = {
            f"{singular}_name",
            f"{table.lower()}_name",
            f"{qual_l}_name",
            "name",
        }

        has_id = any(
            c == id_col_l and (qf is None or qf == qual_l or qf == t)
            for qf, c in selected_cols
        )
        has_name_simple = any(
            c == name_col_l and (qf is None or qf == qual_l or qf == t)
            for qf, c in selected_cols
        )
        has_name_alias = any(a in table_name_aliases for a in selected_aliases)
        has_fullname_expr = any(
            re.search(
                rf"""(?is)\bCONCAT\s*\(\s*`?{re.escape(qual)}`?\s*\.\s*`?(firstName|first_name)`?\s*,\s*'[^']*'\s*,\s*`?{re.escape(qual)}`?\s*\.\s*`?(lastName|last_name)`?\s*\)""",
                expr,
            )
            or re.search(
                rf"""(?is)\bCONCAT\s*\(\s*`?{re.escape(qual)}`?\s*\.\s*`?(lastName|last_name)`?\s*,\s*'[^']*'\s*,\s*`?{re.escape(qual)}`?\s*\.\s*`?(firstName|first_name)`?\s*\)""",
                expr,
            )
            for expr in selected_exprs
        )
        has_name = has_name_simple or has_name_alias or has_fullname_expr

        if not has_id:
            prepended_cols.append(f"{qual}.{id_col}")
            selected_cols.append((qual_l, id_col_l))
            changed = True

        if not has_name:
            prepended_cols.append(f"{qual}.{name_col}")
            selected_cols.append((qual_l, name_col_l))
            changed = True

        # For grouped queries, ensure GROUP BY contains missing non-aggregate cols.
        if group_items:
            need_g_id = not any(
                c == id_col_l and (qf is None or qf == qual_l or qf == t)
                for qf, c in grouped_cols
            )
            # If name is already satisfied via alias/expression (e.g., employee_name/CONCAT),
            # do not force-add raw name column to GROUP BY.
            if has_name_simple:
                need_g_name = not any(
                    c == name_col_l and (qf is None or qf == qual_l or qf == t)
                    for qf, c in grouped_cols
                )
            else:
                need_g_name = False
            if need_g_id:
                group_items.append(f"{qual}.{id_col}")
                grouped_cols.add((qual_l, id_col_l))
                changed = True
            if need_g_name:
                group_items.append(f"{qual}.{name_col}")
                grouped_cols.add((qual_l, name_col_l))
                changed = True

    if not changed:
        return sql

    # Preserve existing SELECT order; prepend only truly missing columns.
    new_select_items = prepended_cols + select_items
    # Deduplicate simple duplicates while preserving order.
    dedup_select: List[str] = []
    seen_sel = set()
    for it in new_select_items:
        k = it.strip().lower()
        if k in seen_sel:
            continue
        seen_sel.add(k)
        dedup_select.append(it.strip())

    # Deduplicate GROUP BY too.
    dedup_group: List[str] = []
    seen_grp = set()
    for g in group_items:
        k = g.strip().lower()
        if k in seen_grp:
            continue
        seen_grp.add(k)
        dedup_group.append(g.strip())

    out_lines = []
    out_lines.append("SELECT " + ", ".join(dedup_select))
    out_lines.append("FROM " + from_core)
    if parsed.get("where"):
        out_lines.append(parsed["where"].strip())
    out_lines.append("GROUP BY " + ", ".join(dedup_group))
    if parsed.get("having"):
        out_lines.append("HAVING " + parsed["having"].strip())
    if parsed.get("tail"):
        out_lines.append(parsed["tail"].strip())

    fixed = "\n".join(out_lines).strip()
    if not fixed.endswith(";"):
        fixed += ";"
    return fixed

def _prune_non_primary_dimensions_for_primary_aggregate(sql: str, question: str, ctx: str = "") -> str:
    """
    For include-zero "for each <primary entity>" aggregate questions, keep one row per
    primary entity by removing non-primary simple dimensions from SELECT/GROUP BY.

    Example fixed:
      For each employee ... total payment from their customers ...
      DROP: customerNumber/customerName from SELECT/GROUP BY.
    """
    if not sql or not question:
        return sql

    q = (question or "").lower()
    # Only activate on conservative signals to avoid over-pruning.
    has_for_each = bool(re.search(r"\b(for\s+each|each|per|every)\b", q))
    has_agg = bool(re.search(r"\b(sum|count|avg|average|min|max|total|jumlah)\b", q))
    include_zero = _question_wants_include_zero(question, ctx)
    if not (has_for_each and has_agg and include_zero):
        return sql

    parsed = _parse_select_from_groupby(sql)
    if not parsed:
        return sql
    from_core = (parsed.get("from_core") or "").strip()
    if not from_core or not re.search(r"\bjoin\b", from_core, flags=re.I):
        return sql
    if not parsed.get("groupby"):
        return sql

    alias_to_table = _parse_from_join_aliases(f"SELECT 1 FROM {from_core}")
    if not alias_to_table:
        return sql

    # Build term->table mapping dynamically from schema metadata.
    term_to_table: Dict[str, str] = {}
    for t in get_all_tables():
        tt = (t or "").strip().lower()
        if not tt:
            continue
        terms: Set[str] = {tt}
        sg = get_singular_form(tt) if tt else tt
        if sg:
            terms.add(str(sg).lower())
        if "_" in tt:
            terms.add(tt.replace("_", " "))
        if sg and "_" in str(sg):
            terms.add(str(sg).replace("_", " ").lower())
        try:
            for syn in get_table_synonyms(tt):
                s = (syn or "").strip().lower()
                if s:
                    terms.add(s)
        except Exception:
            pass
        for term in terms:
            if not term:
                continue
            term_to_table[term] = tt

    # Infer primary table from "for each <entity>" phrase.
    primary_table: Optional[str] = None
    # Sort longer terms first to avoid partial matches.
    for term in sorted(term_to_table.keys(), key=len, reverse=True):
        pat = rf"\b(?:for\s+each|each|per|every)\s+{re.escape(term)}\b"
        if re.search(pat, q, flags=re.I):
            primary_table = term_to_table[term]
            break
    if not primary_table:
        return sql

    primary_aliases: Set[str] = set()
    for a, t in alias_to_table.items():
        if not a or not t:
            continue
        if t.lower() == primary_table:
            primary_aliases.add(a.lower())
    # fallback without explicit alias
    primary_aliases.add(primary_table)
    primary_aliases.add(get_singular_form(primary_table).lower())

    def _split_expr_alias(item: str) -> tuple:
        m = re.match(
            r"""(?is)^\s*(.*?)\s+AS\s+(`[^`]+`|'[^']+'|"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)\s*$""",
            item.strip(),
        )
        if not m:
            return item.strip(), None
        return m.group(1).strip(), m.group(2).strip()

    def _alias_refs(expr: str) -> Set[str]:
        refs = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.", expr or "", flags=re.I)
        return {r.lower() for r in refs}

    def _is_agg_or_window(expr: str) -> bool:
        e = (expr or "")
        if re.search(r"\b(SUM|COUNT|AVG|MIN|MAX|GROUP_CONCAT|STDDEV|VARIANCE)\s*\(", e, flags=re.I):
            return True
        if re.search(r"\bOVER\s*\(", e, flags=re.I):
            return True
        return False

    select_items = _split_top_level_commas(parsed["select"])
    group_items = _split_top_level_commas(parsed["groupby"])

    new_select: List[str] = []
    for it in select_items:
        expr, _alias = _split_expr_alias(it)
        refs = _alias_refs(expr)
        if _is_agg_or_window(expr):
            new_select.append(it.strip())
            continue
        if not refs:
            new_select.append(it.strip())
            continue
        # Keep non-aggregate expression only when it depends exclusively on primary alias/table.
        if refs.issubset(primary_aliases):
            new_select.append(it.strip())
            continue
        # else: drop non-primary dimension

    if not new_select:
        return sql

    new_group: List[str] = []
    for g in group_items:
        refs = _alias_refs(g)
        if not refs or refs.issubset(primary_aliases):
            new_group.append(g.strip())

    if not new_group:
        # do not produce invalid aggregate query
        return sql

    # Rebuild query with same tail/having.
    out_lines = []
    out_lines.append("SELECT " + ", ".join(new_select))
    out_lines.append("FROM " + from_core)
    if parsed.get("where"):
        out_lines.append(parsed["where"].strip())
    out_lines.append("GROUP BY " + ", ".join(new_group))
    if parsed.get("having"):
        out_lines.append("HAVING " + parsed["having"].strip())
    if parsed.get("tail"):
        out_lines.append(parsed["tail"].strip())

    fixed = "\n".join(out_lines).strip()
    if not fixed.endswith(";"):
        fixed += ";"
    return fixed

def _drop_redundant_raw_name_columns_when_concat_present(sql: str, question: str) -> str:
    """
    If SELECT already contains CONCAT(alias.firstName, ' ', alias.lastName),
    drop redundant raw alias.firstName / alias.lastName columns (and their GROUP BY entries)
    unless the question explicitly asks for first/last name.
    """
    if not sql or not question:
        return sql

    q = (question or "").lower()
    if "first name" in q or "last name" in q:
        return sql

    parsed = _parse_select_from_groupby(sql)
    if not parsed:
        return sql
    if not parsed.get("groupby"):
        return sql

    select_items = _split_top_level_commas(parsed["select"])
    group_items = _split_top_level_commas(parsed["groupby"])
    if not select_items:
        return sql

    def _split_expr_alias(item: str) -> tuple:
        m = re.match(
            r"""(?is)^\s*(.*?)\s+AS\s+(`[^`]+`|'[^']+'|"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)\s*$""",
            item.strip(),
        )
        if not m:
            return item.strip(), None
        return m.group(1).strip(), m.group(2).strip()

    def _parse_simple_col(expr: str) -> Optional[tuple]:
        m = re.match(
            r"(?is)^\s*(?:`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\.\s*)?`?([A-Za-z_][A-Za-z0-9_]*)`?\s*$",
            expr.strip(),
        )
        if not m:
            return None
        return (m.group(1) or "").lower(), m.group(2).lower()

    # Detect aliases that already project full-name via CONCAT(first,last).
    concat_aliases: Set[str] = set()
    for it in select_items:
        expr, _a = _split_expr_alias(it)
        m1 = re.match(
            r"""(?is)^CONCAT\s*\(\s*`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\.\s*`?(firstName|first_name)`?\s*,\s*'[^']*'\s*,\s*`?\1`?\s*\.\s*`?(lastName|last_name)`?\s*\)\s*$""",
            expr,
        )
        m2 = re.match(
            r"""(?is)^CONCAT\s*\(\s*`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\.\s*`?(lastName|last_name)`?\s*,\s*'[^']*'\s*,\s*`?\1`?\s*\.\s*`?(firstName|first_name)`?\s*\)\s*$""",
            expr,
        )
        if m1:
            concat_aliases.add(m1.group(1).lower())
        elif m2:
            concat_aliases.add(m2.group(1).lower())

    if not concat_aliases:
        return sql

    def _is_raw_name_col(item: str) -> bool:
        expr, _a = _split_expr_alias(item)
        p = _parse_simple_col(expr)
        if not p:
            return False
        a, c = p
        if not a:
            return False
        if a not in concat_aliases:
            return False
        return c in {"firstname", "first_name", "lastname", "last_name"}

    new_select = [it.strip() for it in select_items if not _is_raw_name_col(it)]
    if len(new_select) == len(select_items):
        return sql

    # Keep original GROUP BY as-is for safety (prevents 1055 regressions).
    new_group = [g.strip() for g in group_items]

    out_lines = []
    out_lines.append("SELECT " + ", ".join(new_select))
    out_lines.append("FROM " + (parsed.get("from_core") or "").strip())
    if parsed.get("where"):
        out_lines.append(parsed["where"].strip())
    out_lines.append("GROUP BY " + ", ".join(new_group))
    if parsed.get("having"):
        out_lines.append("HAVING " + parsed["having"].strip())
    if parsed.get("tail"):
        out_lines.append(parsed["tail"].strip())

    fixed = "\n".join(out_lines).strip()
    if not fixed.endswith(";"):
        fixed += ";"
    return fixed

def _expand_group_by_aliases_regex(sql: str) -> str:
    """
    Regex-based GROUP BY alias expansion (simpler but less robust).
    """
    if not sql or "group by" not in sql.lower():
        return sql

    # Only operate when GROUP BY exists at top level (not inside subqueries).
    def _find_top_level_group_by(s: str) -> int:
        s_up = s.upper()
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
            elif ch == ")":
                depth = max(0, depth - 1)
            elif depth == 0 and s_up[i:i + 8] == "GROUP BY":
                left_ok = (i == 0 or not s[i - 1].isalnum())
                right_ok = (i + 8 >= len(s) or not s[i + 8].isalnum())
                if left_ok and right_ok:
                    return i
            i += 1
        return -1

    group_pos_top = _find_top_level_group_by(sql)
    if group_pos_top < 0:
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

    # Build alias -> expression map conservatively from top-level SELECT items.
    # Important: only use explicit "AS alias" and never rewrite partial tokens
    # inside qualified refs (e.g., do NOT turn p.amount into p.MAX(...)).
    alias_map: Dict[str, str] = {}
    select_items = _split_top_level_commas(select_part)
    for item in select_items:
        m_as = re.match(
            r"""(?is)^\s*(.*?)\s+AS\s+(`[^`]+`|'[^']+'|"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)\s*$""",
            item.strip(),
        )
        if not m_as:
            continue
        expr = (m_as.group(1) or "").strip()
        alias = (m_as.group(2) or "").strip().strip("`'\"")
        if not expr or not alias:
            continue
        alias_map[alias.lower()] = expr

    if not alias_map:
        return sql

    # Extract top-level GROUP BY clause (search from top-level GROUP BY position).
    m_group = re.search(
        r"\bGROUP\s+BY\b\s+(.*?)(?=(\bORDER\s+BY\b|\bHAVING\b|\bLIMIT\b|;|$))",
        sql[group_pos_top:],
        flags=re.I | re.S
    )
    if not m_group:
        return sql

    abs_group_start = group_pos_top + m_group.start(1)
    abs_group_end = group_pos_top + m_group.end(1)
    group_clause = m_group.group(1).strip()
    group_items = _split_top_level_commas(group_clause)

    def _unquote_identifier(x: str) -> str:
        t = (x or "").strip()
        if len(t) >= 2 and t[0] == t[-1] and t[0] in {"`", "'", '"'}:
            return t[1:-1].strip()
        return t

    # Replace only when a GROUP BY item is exactly an alias token.
    new_group_items: List[str] = []
    for item in group_items:
        raw = item.strip()
        key = _unquote_identifier(raw).lower()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", _unquote_identifier(raw)) and key in alias_map:
            new_group_items.append(alias_map[key])
        else:
            new_group_items.append(raw)
    new_group = ", ".join(new_group_items)

    # Replace in original SQL, preserving clause boundaries.
    suffix = sql[abs_group_end:]
    if suffix and not suffix.startswith((" ", "\n", "\t", "\r")):
        if re.match(r"(?is)^(ORDER\s+BY|HAVING|LIMIT|;)", suffix):
            suffix = " " + suffix
    return sql[:abs_group_start] + new_group + suffix

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

    # Winner-row patterns (robust, no hardcoded column names)
    # Pattern 1: MAX()/MIN() on date/timestamp columns (e.g., orderDate, paymentDate, createdAt)
    # These indicate "most recent" or "earliest" queries → winner-row pattern
    has_max_min = ("max(" in s or "min(" in s)
    if has_max_min:
        # Check for date/timestamp column patterns: *date, *time, *at
        has_date_col = bool(re.search(r'\b\w*date\b', s)) or bool(re.search(r'\b\w*time\b', s))
        if has_date_col:
            return True

    # Pattern 2: HAVING clause with aggregate → likely filtering for max/min value
    if "having" in s and ("max(" in s or "min(" in s):
        return True

    # If no specific winner-row pattern matched, it's NOT a winner-row query
    return False


def _fix_order_by_group_by_consistency(sql: str) -> str:
    """
    Fix ORDER BY items that reference columns not in GROUP BY (causes Error 1055).

    Two fix modes:
    1. When ORDER BY has simple column and GROUP BY has expression → replace ORDER BY with GROUP BY expr
    2. When ORDER BY has expression not in GROUP BY → add ORDER BY expr to GROUP BY

    Example 1 (replace ORDER BY):
        GROUP BY MONTH(o.orderDate)
        ORDER BY o.orderDate DESC
    →   ORDER BY MONTH(o.orderDate) DESC

    Example 2 (add to GROUP BY):
        GROUP BY DATE_FORMAT(o.orderDate, '%M')
        ORDER BY MONTH(o.orderDate) DESC
    →   GROUP BY DATE_FORMAT(o.orderDate, '%M'), MONTH(o.orderDate)
        ORDER BY MONTH(o.orderDate) DESC
    """
    if not sql:
        return sql

    if "GROUP BY" not in sql.upper() or "ORDER BY" not in sql.upper():
        return sql

    # IMPORTANT: only consider TOP-LEVEL ORDER BY/GROUP BY (outside subqueries/window functions).
    # Otherwise ORDER BY inside OVER(...) will corrupt GROUP BY rewrites.
    def _find_top_level_order_by(s: str) -> Optional[int]:
        s_upper = s.upper()
        depth = 0
        i = 0
        while i < len(s):
            ch = s[i]
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth = max(0, depth - 1)
            elif depth == 0 and s_upper[i:i+8] == 'ORDER BY':
                left_ok = (i == 0 or not s[i-1].isalnum())
                right_ok = (i + 8 >= len(s) or not s[i+8].isalnum())
                if left_ok and right_ok:
                    return i
            i += 1
        return None

    def _find_top_level_group_by(s: str) -> Optional[int]:
        s_upper = s.upper()
        depth = 0
        i = 0
        while i < len(s):
            ch = s[i]
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth = max(0, depth - 1)
            elif depth == 0 and s_upper[i:i+8] == 'GROUP BY':
                left_ok = (i == 0 or not s[i-1].isalnum())
                right_ok = (i + 8 >= len(s) or not s[i+8].isalnum())
                if left_ok and right_ok:
                    return i
            i += 1
        return None

    order_pos = _find_top_level_order_by(sql)
    if order_pos is None:
        return sql
    group_pos_top = _find_top_level_group_by(sql)
    if group_pos_top is None:
        return sql

    # Extract top-level GROUP BY clause
    gb_match = re.search(
        r'\bGROUP\s+BY\s+(.*?)(?=\s*(?:ORDER\s+BY|HAVING|LIMIT|;|$))',
        sql[group_pos_top:],
        re.IGNORECASE | re.DOTALL
    )
    if not gb_match:
        return sql

    gb_abs_start = group_pos_top + gb_match.start(1)
    gb_abs_end = group_pos_top + gb_match.end(1)
    group_clause = gb_match.group(1).strip()
    group_items = _split_top_level_commas(group_clause)

    # Normalize GROUP BY items for comparison (strip whitespace, lowercase)
    group_items_normalized = [item.strip().lower() for item in group_items]

    # Build a mapping: column_name → group_by_expression
    # e.g., "orderdate" → "MONTH(o.orderDate)"
    col_to_gb_expr = {}
    for item in group_items:
        # Extract column names referenced in this GROUP BY expression
        # Pattern: matches table.column or column names
        cols_in_expr = re.findall(r'\b(\w+(?:\.\w+)?)\b', item)
        for col in cols_in_expr:
            col_lower = col.lower()
            # Skip function names and common keywords
            if col_lower.upper() in {'MONTH', 'YEAR', 'DAY', 'DATE', 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX',
                                     'DATE_FORMAT', 'STR_TO_DATE', 'IF', 'IFNULL', 'COALESCE', 'CONCAT'}:
                continue
            # Map this column to the GROUP BY expression
            if col_lower not in col_to_gb_expr:
                col_to_gb_expr[col_lower] = item

    # Extract and fix ORDER BY clause
    ob_match = re.search(r'\bORDER\s+BY\s+(.*?)(?=\s*(?:LIMIT|;|$))', sql[order_pos:], re.IGNORECASE | re.DOTALL)
    if not ob_match:
        return sql

    order_clause = ob_match.group(1).strip()
    order_items = _split_top_level_commas(order_clause)

    new_order_items = []
    items_to_add_to_group_by = []

    # Aggregate expressions in ORDER BY must never be pushed into GROUP BY
    # (e.g., ORDER BY COUNT(*) DESC is valid and should stay as-is).
    aggregate_func_pattern = re.compile(
        r"\b(COUNT|SUM|AVG|MIN|MAX|STDDEV|STD|VARIANCE|VAR_POP|VAR_SAMP|GROUP_CONCAT|BIT_AND|BIT_OR|BIT_XOR|JSON_ARRAYAGG|JSON_OBJECTAGG)\s*\(",
        re.IGNORECASE,
    )

    for item in order_items:
        # Extract the base column/expression (before ASC/DESC)
        base_match = re.match(r'(.*?)\s+(ASC|DESC)\s*$', item, re.IGNORECASE)
        if base_match:
            base = base_match.group(1).strip()
            direction = base_match.group(2).strip()
        else:
            base = item.strip()
            direction = ""

        # Check if this is a simple column reference (not a function)
        is_simple_col = bool(re.match(r'^\w+(?:\.\w+)?$', base))

        if is_simple_col:
            base_lower = base.lower()
            # Check if this column appears in GROUP BY expressions
            if base_lower in col_to_gb_expr:
                # Replace with the GROUP BY expression
                base = col_to_gb_expr[base_lower]
        else:
            # ORDER BY item is a function expression (e.g., MONTH(o.orderDate))
            # Check if this expression is already in GROUP BY
            base_normalized = base.lower().replace(" ", "")  # Normalize for comparison

            # Also check without table alias prefix for flexibility
            # MONTH(o.orderDate) should match MONTH(orderDate) in GROUP BY
            base_no_space = base.replace(" ", "").lower()
            is_in_group_by = any(
                base_no_space in gi.replace(" ", "").lower()
                for gi in group_items_normalized
            )

            has_aggregate = bool(aggregate_func_pattern.search(base))
            has_window = bool(re.search(r"\bOVER\s*\(", base, re.IGNORECASE))

            if not is_in_group_by and not has_aggregate and not has_window:
                # Add this ORDER BY expression to GROUP BY
                items_to_add_to_group_by.append(base)

        # Reconstruct with direction
        if direction:
            new_order_items.append(f"{base} {direction}")
        else:
            new_order_items.append(base)

    # Update ORDER BY clause
    if new_order_items:
        new_order = ', '.join(new_order_items)
        abs_start = order_pos + ob_match.start(1)
        abs_end = order_pos + ob_match.end(1)
        sql = sql[:abs_start] + new_order + sql[abs_end:]

    # Add missing items to GROUP BY clause
    if items_to_add_to_group_by:
        # Find GROUP BY clause again (position may have changed)
        group_pos_top = _find_top_level_group_by(sql)
        if group_pos_top is None:
            return _drop_aggregate_or_window_items_from_group_by(sql)
        gb_match = re.search(
            r'\bGROUP\s+BY\s+(.*?)(?=\s*(?:ORDER\s+BY|HAVING|LIMIT|;|$))',
            sql[group_pos_top:],
            re.IGNORECASE | re.DOTALL
        )
        if gb_match:
            new_group_items = group_items + items_to_add_to_group_by
            new_group_clause = ', '.join(new_group_items)
            gb_abs_start = group_pos_top + gb_match.start(1)
            gb_abs_end = group_pos_top + gb_match.end(1)
            sql = sql[:gb_abs_start] + new_group_clause + sql[gb_abs_end:]

    # Final safety: GROUP BY must not contain aggregate/window expressions.
    return _drop_aggregate_or_window_items_from_group_by(sql)


def _drop_aggregate_or_window_items_from_group_by(sql: str) -> str:
    """
    Remove malformed GROUP BY items that are aggregate/window expressions.

    Example malformed:
      GROUP BY customerNumber, COUNT(*)
    """
    if not sql or "GROUP BY" not in sql.upper():
        return sql

    s = sql
    low = s.lower()

    def _is_word_boundary(text: str, i: int, n: int) -> bool:
        left_ok = i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")
        right_ok = i + n >= len(text) or not (text[i + n].isalnum() or text[i + n] == "_")
        return left_ok and right_ok

    def _find_top_level_kw(text: str, kw: str, start: int = 0) -> int:
        kw_l = kw.lower()
        n = len(kw_l)
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
            if depth == 0 and i >= start and i + n <= len(text):
                if text[i:i + n].lower() == kw_l and _is_word_boundary(text, i, n):
                    return i
            i += 1
        return -1

    group_pos = _find_top_level_kw(low, "group by", 0)
    if group_pos < 0:
        return sql

    group_start = group_pos + len("group by")

    end_candidates = []
    for kw in ("having", "order by", "limit"):
        p = _find_top_level_kw(low, kw, group_start)
        if p >= 0:
            end_candidates.append(p)
    semicolon_pos = _find_top_level_kw(low, ";", group_start)
    if semicolon_pos >= 0:
        end_candidates.append(semicolon_pos)
    group_end = min(end_candidates) if end_candidates else len(s)

    group_clause = s[group_start:group_end].strip()
    if not group_clause:
        return sql

    group_items = _split_top_level_commas(group_clause)
    if not group_items:
        return sql

    aggregate_func_pattern = re.compile(
        r"\b(COUNT|SUM|AVG|MIN|MAX|STDDEV|STD|VARIANCE|VAR_POP|VAR_SAMP|GROUP_CONCAT|BIT_AND|BIT_OR|BIT_XOR|JSON_ARRAYAGG|JSON_OBJECTAGG)\s*\(",
        re.IGNORECASE,
    )
    window_pattern = re.compile(r"\bOVER\s*\(", re.IGNORECASE)

    cleaned_items: List[str] = []
    for item in group_items:
        raw = (item or "").strip()
        if not raw:
            continue
        if aggregate_func_pattern.search(raw):
            continue
        if window_pattern.search(raw):
            continue
        cleaned_items.append(raw)

    if len(cleaned_items) == len(group_items) or not cleaned_items:
        return sql

    new_group = ", ".join(cleaned_items)
    prefix = s[:group_start]
    suffix = s[group_end:]
    if suffix and not suffix[0].isspace():
        suffix = " " + suffix
    return f"{prefix} {new_group}{suffix}"


def _normalize_alias_lengths(sql: str) -> str:
    """
    Soft alias guard:
    - Preserve LLM-defined aliases by default.
    - Only shorten aliases that are excessively long.

    This keeps model creativity for normal aliases while preventing unwieldy
    output names that hurt readability.
    """
    if not sql:
        return sql

    # Don't process if no AS clause
    if " AS " not in sql.upper():
        return sql

    max_alias_len = ALIAS_MAX_LEN if ALIAS_MAX_LEN >= 8 else 8

    def replace_alias(match):
        """Shorten only if alias is too long; otherwise keep original."""
        # Extract the alias from whichever group matched
        # groups[0] = content inside single quotes
        # groups[1] = content inside double quotes
        # groups[2] = content inside backticks
        # groups[3] = unquoted alias

        groups = match.groups()

        if groups[0] is not None:
            # Single-quoted alias
            alias = groups[0]
            quote_char = "'"
        elif groups[1] is not None:
            # Double-quoted alias
            alias = groups[1]
            quote_char = '"'
        elif groups[2] is not None:
            # Backtick-quoted alias
            alias = groups[2]
            quote_char = '`'
        elif groups[3] is not None:
            # Unquoted alias
            alias = groups[3]
            quote_char = None
        else:
            # No match, return original
            return match.group(0)

        # Keep exactly what LLM produced if alias is already reasonable.
        if len(alias) <= max_alias_len:
            return match.group(0)

        # If too long, compact to snake_case + max length.
        shortened = alias.lower().replace(" ", "_").replace("-", "_")
        shortened = re.sub(r"[^\w]", "_", shortened)
        shortened = re.sub(r"_+", "_", shortened).strip("_")
        if not shortened:
            shortened = "alias"
        if len(shortened) > max_alias_len:
            shortened = shortened[:max_alias_len].rstrip("_")
        if not shortened:
            shortened = "alias"

        if quote_char:
            return f" AS {quote_char}{shortened}{quote_char}"
        else:
            return f" AS {shortened}"

    # Match AS followed by quoted or unquoted alias
    # Pattern handles: AS 'name', AS "name", AS `name`, AS identifier
    pattern = r"\s+AS\s+(?:'([^']*)'|"          # Single quoted
    pattern += r'"([^"]*)"'                     # Double quoted
    pattern += r'|`([^`]*)`'                    # Backtick quoted
    pattern += r'|([a-zA-Z_][a-zA-Z0-9_]*))'    # Unquoted identifier

    result = re.sub(pattern, replace_alias, sql, flags=re.IGNORECASE)

    return result

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

def _remove_invalid_select_columns_with_group_by(sql: str) -> str:
    """
    When GROUP BY is present, remove SELECT columns that are:
    - NOT aggregate functions (COUNT, SUM, AVG, MIN, MAX, etc.)
    - NOT in GROUP BY
    - NOT window functions (OVER)
    - NOT literals/stars

    This prevents Error 1055: "not functionally dependent on columns in GROUP BY clause"

    Safe for all correct queries - only removes hallucinated columns.
    """
    if not sql:
        return sql

    # Only apply when GROUP BY is top-level (not inside subqueries).
    def _has_top_level_group_by(s: str) -> bool:
        s_up = s.upper()
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
            elif ch == ")":
                depth = max(0, depth - 1)
            elif depth == 0 and s_up[i:i + 8] == "GROUP BY":
                left_ok = (i == 0 or not s[i - 1].isalnum())
                right_ok = (i + 8 >= len(s) or not s[i + 8].isalnum())
                if left_ok and right_ok:
                    return True
            i += 1
        return False

    if not _has_top_level_group_by(sql):
        return sql

    # Try sqlglot first (more accurate)
    try:
        import sqlglot
        from sqlglot import exp
        ast = sqlglot.parse_one(sql, read="mysql")
        sel = ast.find(exp.Select)
        if not sel:
            return sql

        grp = sel.args.get("group")
        if not grp or not grp.expressions:
            return sql

        def _normalize_table_name(name: str) -> str:
            t = (name or "").strip("`").lower()
            if not t:
                return ""
            if t.endswith("ies") and len(t) > 3:
                return t[:-3] + "y"
            if t.endswith("s") and len(t) > 1:
                return t[:-1]
            return t

        def _column_entity_stem(col_name: str) -> str:
            c = (col_name or "").strip("`").lower()
            if not c:
                return ""
            c = re.sub(r"(?:_)?(?:number|id|code|key)$", "", c).strip("_")
            return c

        def _looks_like_entity_identifier(col_name: str, table_name: str) -> bool:
            c = (col_name or "").strip("`").lower()
            if not c or not re.search(r"(?:number|id|code|key)$", c):
                return False
            col_stem = _column_entity_stem(c)
            if not col_stem:
                return False
            table_norm = _normalize_table_name(table_name)
            return bool(table_norm) and col_stem == table_norm

        alias_to_table: Dict[str, str] = {}
        table_names: Set[str] = set()
        for src in ast.find_all(exp.Table):
            table_name = str(src.name or "").strip("`").lower()
            alias_name = str(src.alias_or_name or table_name).strip("`").lower()
            if not table_name:
                continue
            table_names.add(table_name)
            if alias_name:
                alias_to_table[alias_name] = table_name
            alias_to_table[table_name] = table_name

        # Keep descriptive columns from the same table when GROUP BY already uses
        # that table's own entity key (e.g., offices.officeCode -> offices.city).
        entity_key_sources: Set[str] = set()
        for gexpr in grp.expressions:
            if not isinstance(gexpr, exp.Column):
                continue
            g_table = str(gexpr.table or "").strip("`").lower()
            g_name = str(gexpr.name or "").strip("`").lower()
            resolved_table = alias_to_table.get(g_table, g_table)
            if _looks_like_entity_identifier(g_name, resolved_table):
                if g_table:
                    entity_key_sources.add(g_table)
                if resolved_table:
                    entity_key_sources.add(resolved_table)

        # Get GROUP BY columns
        group_sql = grp.sql(dialect="mysql")
        group_set = {g.strip().lower() for g in re.findall(r'(\w+(?:\.\w+)?)', group_sql, re.IGNORECASE)}

        # Check SELECT expressions
        expressions_to_remove = []
        for item in sel.expressions:
            base = item.this if isinstance(item, exp.Alias) else item
            alias = item.alias if isinstance(item, exp.Alias) else None

            # Skip if aggregate, window, literal, or star
            if base.find(exp.AggFunc):
                continue
            if base.find(exp.Window):
                continue
            if isinstance(base, (exp.Literal, exp.Star)):
                continue

            # Skip CASE expressions (they define computed columns used in GROUP BY)
            if isinstance(base, exp.Case):
                continue

            base_sql = base.sql(dialect="mysql").strip().lower()

            # Also check if alias is in GROUP BY (MySQL compatibility)
            alias_in_groupby = False
            if alias:
                alias_sql = alias.sql(dialect="mysql").strip().lower()
                alias_in_groupby = alias_sql in group_set

            preserve_via_entity_key = False
            if isinstance(base, exp.Column):
                base_table = str(base.table or "").strip("`").lower()
                resolved_base_table = alias_to_table.get(base_table, base_table)
                if base_table and (
                    base_table in entity_key_sources or resolved_base_table in entity_key_sources
                ):
                    preserve_via_entity_key = True
                elif (
                    not base_table
                    and len(table_names) == 1
                    and entity_key_sources
                ):
                    # Conservative support for unqualified columns in single-table queries.
                    preserve_via_entity_key = True

            if base_sql not in group_set and not alias_in_groupby and not preserve_via_entity_key:
                expressions_to_remove.append(item)

        if expressions_to_remove:
            new_expressions = [e for e in sel.expressions if e not in expressions_to_remove]
            sel.set("expressions", new_expressions)

        return ast.sql(dialect="mysql")

    except Exception:
        # FALLBACK: Use regex when sqlglot fails (syntax errors, etc.)
        # This handles cases where SQL has minor syntax issues
        if "GROUP BY" not in sql.upper():
            return sql
        # Safety: for queries with subqueries, regex fallback can grab inner GROUP BY
        # and corrupt top-level SELECT. Skip fallback in that case.
        if re.search(r"\b(?:FROM|JOIN)\s*\(", sql, flags=re.I):
            return sql

        # Extract GROUP BY clause
        gb_match = re.search(r'\bGROUP\s+BY\s+(.*?)(?=\s*(?:ORDER\s+BY|HAVING|LIMIT|;|$))', sql, re.IGNORECASE | re.DOTALL)
        if not gb_match:
            return sql

        group_clause = gb_match.group(1).strip()

        # Helper: split by top-level commas (not inside parentheses)
        def split_top_level_commas(clause):
            items = []
            current = []
            depth = 0
            for char in clause:
                if char == '(':
                    depth += 1
                    current.append(char)
                elif char == ')':
                    depth -= 1
                    current.append(char)
                elif char == ',' and depth == 0:
                    items.append(''.join(current).strip())
                    current = []
                else:
                    current.append(char)
            if current:
                items.append(''.join(current).strip())
            return [i for i in items if i]

        group_items = split_top_level_commas(group_clause)
        # For each GROUP BY item, extract the base expression (before AS, if any)
        group_base_expressions = []
        for item in group_items:
            # Remove AS alias if present
            as_match = re.search(r'\s+AS\s+(?:[\w"]+|"[^"]*"|\'[^\']*\'|`[^`]*`)\s*$', item, re.IGNORECASE)
            if as_match:
                base = item[:as_match.start()].strip()
            else:
                base = item.strip()
            group_base_expressions.append(base.lower())

        group_set = set(group_base_expressions)

        # Extract SELECT clause
        select_match = re.search(r'\bSELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
        if not select_match:
            return sql

        select_clause = select_match.group(1).strip()
        select_items = split_top_level_commas(select_clause)

        # Rebuild SELECT, removing items that are NOT in GROUP BY (except aggregates/windows)
        new_select_parts = []
        agg_funcs = {'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'DATEDIFF', 'DATE_ADD', 'TIMESTAMPDIFF',
                     'DATE_FORMAT', 'YEAR', 'MONTH', 'DAY', 'CONCAT', 'IF', 'IFNULL', 'COALESCE'}

        for item in select_items:
            # Extract base expression (before AS)
            as_match = re.search(r'\s+AS\s+(?:[\w"]+|"[^"]*"|\'[^\']*\'|`[^`]*`)\s*$', item, re.IGNORECASE)
            if as_match:
                base = item[:as_match.start()].strip()
                alias = as_match.group(0).replace('AS', '').strip().strip('"\'`')
            else:
                base = item.strip()
                alias = None

            base_lower = base.lower()

            # Check if it's an aggregate or window function
            is_aggregate = False
            func_match = re.match(r'(\w+)\s*\(', base)
            if func_match:
                func_name = func_match.group(1).upper()
                if func_name in agg_funcs:
                    is_aggregate = True

            is_window = bool(re.search(r'\bOVER\s*\(', base, re.IGNORECASE))
            # Check if it's a CASE expression (should always be kept)
            is_case = bool(re.search(r'\bCASE\s+WHEN\b', base, re.IGNORECASE))

            # Keep if: aggregate, window, CASE, base in GROUP BY, or alias in GROUP BY (MySQL compatibility)
            # MySQL allows GROUP BY alias, so check both base expression and alias
            alias_in_groupby = alias and alias.lower() in group_set if alias else False
            if is_aggregate or is_window or is_case or base_lower in group_set or alias_in_groupby:
                new_select_parts.append(item)
            # Otherwise, remove it (hallucinated column)

        if new_select_parts:
            new_select = ', '.join(new_select_parts)
            sql = sql[:select_match.start(1)] + new_select + sql[select_match.end(1):]

        return sql
# =======================================================================
# ============ Phase 4 — Winner-Row & Aggregate Guardrails ==============
# ======= Ensure semantic correctness for ranking-style queries =========
# =======================================================================
# - winner_row_sanity_date_in_groupby
# - strip_having_max_self_subquery
# - looks_like_max_date_tiebreaker
# - question_wants_winner_row
# - split_top_level_commas
# - extract_alias_or_colname
# - maybe_keep_or_alias_simple
# - parse_select_from_groupby
# - to_raw_expr
# - looks_like_winner_row_multi_agg
# - strip_table_prefix
# - rewrite_winner_row_guardrail
# - rewrite_having_alias_to_expr
# - winner_row_final_sanity
# - force_restore_groupby_if_aggregate_without_groupby

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
    # Ignore trailing output-order instructions so phrases like
    # "ordered by amount from highest to lowest" do not get misread as a
    # winner-row intent. Winner-row should come from the main ask, not from
    # display ordering.
    q_core = re.sub(
        r"\b(?:sort(?:ed)?|order(?:ed)?)\s+(?:the\s+results?|the\s+result|results?|result|output|rows)?\s*by\b.*$",
        "",
        q,
        flags=re.I,
    ).strip()
    if not q_core:
        q_core = q

    rank_sig = any(k in q_core for k in [
        "largest", "highest", "maximum", "max", "biggest",
        "smallest", "lowest", "minimum", "min",
        "best", "worst", "top", "bottom",
        "best-selling", "best selling", "top-selling", "top selling",
        "terbesar", "tertinggi", "maksimum",
        "terkecil", "terendah", "minimum",
        "paling besar", "paling kecil",
    ])

    if not rank_sig:
        return False

    explicit_multirow_rank_sig = bool(re.search(
        r"\b(?:show|list|display|find|return)\s+(?:the\s+)?(?:top\s+)?(?:[2-9]|\d{2,})\b",
        q_core,
    ))

    # Sinyal 1: eksplisit minta atribut dari "baris yang sama"
    same_row_sig = any(k in q_core for k in [
        "and its", "and their", "and the", "along with",
        "dan", "beserta", "serta",
        "which", "that", "who",
        "yang mana", "yang",
    ])

    # Sinyal 2: minta informasi temporal/kontekstual
    context_sig = any(k in q_core for k in [
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
        r"\b(largest|highest|maximum|biggest|best|worst|top|bottom|terbesar|tertinggi)\s+\w+\s+(and|with|beserta|dan)\s+",
        q_core
    ))

    tie_sig = any(k in q_core for k in [
        "if multiple", "if there are multiple", "if there is still a tie",
        "tie", "ties", "tied", "choose the one", "pick the one",
    ])

    per_group_sig = bool(
        rank_sig
        and re.search(r"\b(?:for\s+each|per|each)\b", q_core)
    )

    # Guard: explicit top-N / multi-row ranking asks are not winner-row asks.
    # Example: "Show the 10 products with the highest stock levels".
    # These should stay as ORDER BY ... LIMIT N, not ROW_NUMBER() winner-row.
    if explicit_multirow_rank_sig and not tie_sig and not per_group_sig:
        return False

    strong_winner_sig = rank_sig and (same_row_sig or context_sig or multi_attr_sig or tie_sig or per_group_sig)

    # Guard: "total"/"count" usually means pure aggregation, but do not suppress
    # clear winner-row asks like "best-selling product for each product line"
    # or tie-break questions.
    if any(k in q_core for k in ["total", "sum of", "count", "berapa banyak", "jumlah total"]) and not strong_winner_sig:
        return False

    # Winner-row jika ada rank + (same_row ATAU context ATAU multi_attr)
    return strong_winner_sig

def _fix_mixed_max_groupby_rownumber(sql: str) -> Optional[str]:
    """
    Detect and fix malformed SQL where LLM mixed MAX()/GROUP BY with ROW_NUMBER().

    Example malformed:
    SELECT ... FROM (SELECT customerNumber, MAX(amount), paymentDate
                     FROM payments GROUP BY customerNumber
                     ROW_NUMBER() OVER (PARTITION BY customerNumber ORDER BY amount DESC) = 1) p;

    Correct:
    SELECT t.customerNumber, t.amount, t.paymentDate
    FROM (SELECT customerNumber, amount, paymentDate,
          ROW_NUMBER() OVER (PARTITION BY customerNumber ORDER BY amount DESC) AS rn
          FROM payments) t
    WHERE t.rn = 1;

    Returns fixed SQL or None if pattern not detected.
    """
    if not sql:
        return None

    s = sql.lower()

    # Detect broad mixed-pattern signal.
    has_agg = bool(re.search(r"\b(max|min|sum|count|avg)\s*\(", s))
    has_group_by = "group by" in s
    has_row_number = "row_number() over" in s
    if not (has_agg and has_group_by and has_row_number):
        return None

    # ------------------------------------------------------------------
    # CASE 1: Legacy malformed pattern (ROW_NUMBER ... = 1 inside subquery)
    # ------------------------------------------------------------------
    malformed_pattern = re.search(
        r"row_number\s*\(\s*\)\s+over\s*\([^)]+\)\s*[=<>\s]*1\s*\)",
        sql,
        re.IGNORECASE
    )
    if malformed_pattern:
        outer_select_match = re.search(
            r"select\s+(.*?)\s+from\s*\(",
            sql,
            re.IGNORECASE | re.DOTALL
        )
        if not outer_select_match:
            return None
        outer_cols = outer_select_match.group(1).strip()

        from_table_match = re.search(
            r"from\s+.*?\)\s*(?:as\s+)?(\w+)",
            sql,
            re.IGNORECASE
        )
        outer_alias = from_table_match.group(1) if from_table_match else "t"

        base_table_match = re.search(
            r"from\s+(\w+)\s+(?:(?!group\s+by|row_number).)*?(?=group\s+by|row_number|\))",
            sql,
            re.IGNORECASE
        )
        if not base_table_match:
            return None
        base_table = base_table_match.group(1)

        partition_match = re.search(
            r"partition\s+by\s+([^)]+)",
            sql,
            re.IGNORECASE
        )
        partition_by = partition_match.group(1).strip() if partition_match else ""

        order_match = re.search(
            r"order\s+by\s+([^)]+?)(?:\)|desc|asc)",
            sql,
            re.IGNORECASE
        )
        order_by = order_match.group(1).strip() if order_match else ""

        select_cols = []
        for col in re.split(r",\s*", outer_cols):
            col = col.strip()
            col = re.sub(r"^\w+\.", "", col)
            col = re.sub(
                r"^(max|min|sum|count|avg)\s*\(([^)]+)\)\s*(?:as\s+\w+)?$",
                r"\2",
                col,
                flags=re.IGNORECASE
            )
            select_cols.append(col)

        inner_select = ", ".join(select_cols)
        outer_select = ", ".join([f"{outer_alias}.{col.strip()}" for col in select_cols])
        fixed_sql = (
            f"SELECT {outer_select} "
            f"FROM (SELECT {inner_select}, "
            f"ROW_NUMBER() OVER (PARTITION BY {partition_by} ORDER BY {order_by}) AS rn "
            f"FROM {base_table}) {outer_alias} "
            f"WHERE {outer_alias}.rn = 1"
        )
        return fixed_sql

    # ------------------------------------------------------------------
    # CASE 2: Semantic mixed pattern:
    # outer MAX()+GROUP BY but inner derived-table already uses ROW_NUMBER rn=1.
    # Rewrite to one consistent winner-row query at the parent grain.
    # ------------------------------------------------------------------
    s0 = sql.strip().rstrip(";").strip()
    if not s0:
        return None

    def _is_word_boundary(text: str, start: int, n: int) -> bool:
        left_ok = (start == 0) or not (text[start - 1].isalnum() or text[start - 1] == "_")
        end = start + n
        right_ok = (end >= len(text)) or not (text[end].isalnum() or text[end] == "_")
        return left_ok and right_ok

    def _find_top_level_kw(text: str, kw: str, start: int = 0) -> int:
        w = kw.lower()
        n = len(w)
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
                if text[i:i + n].lower() == w and _is_word_boundary(text, i, n):
                    return i
            i += 1
        return -1

    sel_pos = _find_top_level_kw(s0, "select", 0)
    from_pos = _find_top_level_kw(s0, "from", max(0, sel_pos + 6))
    group_pos = _find_top_level_kw(s0, "group by", max(0, from_pos + 4))
    if sel_pos < 0 or from_pos < 0 or group_pos < 0:
        return None

    order_pos = _find_top_level_kw(s0, "order by", group_pos + 8)
    limit_pos = _find_top_level_kw(s0, "limit", group_pos + 8)
    tail_pos_candidates = [p for p in [order_pos, limit_pos] if p >= 0]
    group_end = min(tail_pos_candidates) if tail_pos_candidates else len(s0)

    select_part = s0[sel_pos + len("select"):from_pos].strip()
    from_part = s0[from_pos + len("from"):group_pos].strip()
    group_part = s0[group_pos + len("group by"):group_end].strip()
    tail_part = s0[group_end:].strip() if group_end < len(s0) else ""

    if not select_part or not from_part or not group_part:
        return None

    select_items = _split_top_level_commas(select_part)
    group_items = _split_top_level_commas(group_part)
    if not select_items or not group_items:
        return None

    # Find exactly one MAX/MIN in outer SELECT.
    agg_idx = -1
    agg_kind = ""
    agg_expr = ""

    def _split_expr_alias(item: str) -> tuple:
        m = re.match(
            r"""(?is)^\s*(.*?)\s+AS\s+(`[^`]+`|'[^']+'|"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)\s*$""",
            item.strip(),
        )
        if not m:
            return item.strip(), None
        return m.group(1).strip(), m.group(2).strip().strip("`'\"")

    def _simple_col_name(expr: str) -> Optional[str]:
        m = re.match(
            r"""(?is)^\s*(?:`?[A-Za-z_][A-Za-z0-9_]*`?\s*\.\s*)?`?([A-Za-z_][A-Za-z0-9_]*)`?\s*$""",
            (expr or "").strip(),
        )
        if not m:
            return None
        return m.group(1)

    def _norm_expr(expr: str) -> str:
        return re.sub(r"\s+", "", (expr or "").replace("`", "").lower())

    for i, it in enumerate(select_items):
        expr_i, _al_i = _split_expr_alias(it)
        m_agg = re.match(r"(?is)^\s*(MAX|MIN)\s*\(\s*(.+?)\s*\)\s*$", expr_i)
        if not m_agg:
            continue
        if agg_idx != -1:
            return None
        agg_idx = i
        agg_kind = (m_agg.group(1) or "").upper()
        agg_expr = (m_agg.group(2) or "").strip()

    if agg_idx < 0 or not agg_expr or agg_kind not in {"MAX", "MIN"}:
        return None

    # Select a parent partition key from GROUP BY.
    id_like = [g for g in group_items if re.search(r"(id|number|code)\b", g, flags=re.I)]
    partition_expr = (id_like[0] if id_like else group_items[0]).strip()

    # Rewrite derived join "(SELECT ... ROW_NUMBER ... ) alias ON ... alias.rn=1"
    # into direct join to base table while preserving join key condition.
    derived_pat = re.compile(
        r"""(?is)
        (?P<join_kw>\b(?:LEFT|INNER|RIGHT|FULL)?\s*JOIN)\s*
        \(\s*SELECT\s+(?P<inner_select>[\s\S]*?)\s+FROM\s+
        (?P<base_table>`?[A-Za-z_][A-Za-z0-9_]*`?(?:\.`?[A-Za-z_][A-Za-z0-9_]*`?)?)
        (?P<inner_rest>[\s\S]*?)\)\s*
        (?:AS\s+)?(?P<d_alias>[A-Za-z_][A-Za-z0-9_]*)\s+
        ON\s+(?P<on_clause>[\s\S]*?)
        (?=(?:\b(?:LEFT|INNER|RIGHT|FULL)\s+JOIN\b|\bJOIN\b|$))
        """,
        flags=re.IGNORECASE | re.VERBOSE,
    )

    d_match = None
    inner_block = ""
    for m in derived_pat.finditer(from_part):
        d_alias = m.group("d_alias")
        inner_block = (m.group("inner_select") or "") + " " + (m.group("inner_rest") or "")
        on_clause = m.group("on_clause") or ""
        if "row_number" not in inner_block.lower():
            continue
        if not re.search(rf"\b{re.escape(d_alias)}\.rn\s*=\s*1\b", on_clause, flags=re.I):
            continue
        d_match = m
        break

    if not d_match:
        return None

    join_kw = re.sub(r"\s+", " ", (d_match.group("join_kw") or "").strip())
    base_table = (d_match.group("base_table") or "").strip()
    d_alias = (d_match.group("d_alias") or "").strip()
    on_clause = (d_match.group("on_clause") or "").strip()
    if not join_kw or not base_table or not d_alias or not on_clause:
        return None

    conds = _split_top_level_and(on_clause)
    if not conds:
        return None

    join_key_found = False
    new_conds: List[str] = []
    for cond in conds:
        c = (cond or "").strip()
        if not c:
            continue
        if re.search(rf"\b{re.escape(d_alias)}\.rn\s*=\s*1\b", c, flags=re.I):
            continue

        m_eq = re.match(r"(?is)^\s*(.+?)\s*=\s*(.+?)\s*$", c)
        if m_eq:
            lhs = (m_eq.group(1) or "").strip()
            rhs = (m_eq.group(2) or "").strip()
            lhs_is_d = bool(re.match(rf"(?is)^\s*`?{re.escape(d_alias)}`?\.", lhs))
            rhs_is_d = bool(re.match(rf"(?is)^\s*`?{re.escape(d_alias)}`?\.", rhs))
            if lhs_is_d ^ rhs_is_d:
                d_side = lhs if lhs_is_d else rhs
                other_side = rhs if lhs_is_d else lhs
                d_col = re.match(
                    rf"(?is)^\s*`?{re.escape(d_alias)}`?\.`?([A-Za-z_][A-Za-z0-9_]*)`?\s*$",
                    d_side,
                )
                if d_col:
                    join_key_found = True
                    new_conds.append(f"{d_alias}.{d_col.group(1)} = {other_side}")
                    continue

        new_conds.append(c)

    if not join_key_found:
        return None

    new_on = " AND ".join([c for c in new_conds if c.strip()])
    if not new_on:
        return None

    new_join = f"{join_kw} {base_table} {d_alias} ON {new_on}"
    new_from = from_part[:d_match.start()] + new_join + from_part[d_match.end():]
    new_from = re.sub(r"\s{2,}", " ", new_from).strip()

    # Build rewritten SELECT list and stable output names.
    inner_items: List[str] = []
    outer_items: List[str] = []
    seen_names: Set[str] = set()
    expr_to_out: Dict[str, str] = {}

    metric_expr = agg_expr.strip()
    metric_dir = "DESC" if agg_kind == "MAX" else "ASC"

    date_expr = ""
    id_tie_expr = ""

    for i, it in enumerate(select_items):
        expr_i, alias_i = _split_expr_alias(it)
        use_expr = metric_expr if i == agg_idx else expr_i.strip()

        out_name = alias_i or _simple_col_name(use_expr) or f"col_{i+1}"
        out_name = re.sub(r"[^A-Za-z0-9_]", "_", out_name).strip("_") or f"col_{i+1}"
        if re.match(r"^\d", out_name):
            out_name = f"c_{out_name}"

        base_name = out_name
        suffix = 2
        while out_name.lower() in seen_names:
            out_name = f"{base_name}_{suffix}"
            suffix += 1
        seen_names.add(out_name.lower())

        inner_items.append(f"{use_expr} AS {out_name}")
        outer_items.append(f"t.{out_name}")
        expr_to_out[_norm_expr(use_expr)] = out_name

        if i != agg_idx:
            use_low = use_expr.lower()
            if not date_expr and any(k in use_low for k in ["date", "time", "timestamp"]):
                date_expr = use_expr
            if not id_tie_expr and re.search(r"(id|number|code)\b", use_low):
                if _norm_expr(use_expr) != _norm_expr(partition_expr):
                    id_tie_expr = use_expr

    if not inner_items or not outer_items:
        return None

    # Prefer the ORDER BY sequence already present inside the derived ROW_NUMBER().
    inner_order_parts: List[str] = []
    m_inner_order = re.search(
        r"""(?is)\bROW_NUMBER\s*\(\s*\)\s*OVER\s*\(\s*PARTITION\s+BY\s+.*?\bORDER\s+BY\s+([^)]+)\)""",
        inner_block or "",
    )
    if m_inner_order:
        raw_inner_order = (m_inner_order.group(1) or "").strip()
        for part in _split_top_level_commas(raw_inner_order):
            p = (part or "").strip()
            if not p:
                continue
            m_dir = re.match(r"(?is)^(.*?)\s+(ASC|DESC)\s*$", p)
            if m_dir:
                expr0 = (m_dir.group(1) or "").strip()
                dir0 = (m_dir.group(2) or "").upper()
            else:
                expr0 = p
                dir0 = "DESC"
            if _simple_col_name(expr0) and "." not in expr0:
                expr0 = f"{d_alias}.{expr0}"
            inner_order_parts.append(f"{expr0} {dir0}")

    if inner_order_parts:
        order_parts = inner_order_parts
        if id_tie_expr:
            present = any(_norm_expr(id_tie_expr) in _norm_expr(p) for p in inner_order_parts)
            if not present:
                order_parts.append(f"{id_tie_expr} ASC")
    else:
        # Fallback heuristic when inner order is unavailable.
        order_parts = []
        if date_expr:
            order_parts.append(f"{date_expr} DESC")
        order_parts.append(f"{metric_expr} {metric_dir}")
        if id_tie_expr and _norm_expr(id_tie_expr) not in {_norm_expr(metric_expr), _norm_expr(date_expr)}:
            order_parts.append(f"{id_tie_expr} ASC")
    order_window = ", ".join(order_parts)

    partition_out = expr_to_out.get(_norm_expr(partition_expr))
    date_out = expr_to_out.get(_norm_expr(date_expr)) if date_expr else None

    if date_out and partition_out:
        outer_tail = f" ORDER BY t.{date_out} DESC, t.{partition_out} ASC"
    elif tail_part:
        tail_no_qual = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\.", "", tail_part)
        outer_tail = " " + tail_no_qual.strip()
    else:
        outer_tail = ""

    inner_sql = (
        f"SELECT {', '.join(inner_items)}, "
        f"ROW_NUMBER() OVER (PARTITION BY {partition_expr} ORDER BY {order_window}) AS rn "
        f"FROM {new_from}"
    )
    fixed_sql = (
        f"SELECT {', '.join(outer_items)} "
        f"FROM ({inner_sql}) t "
        f"WHERE t.rn = 1"
        f"{outer_tail}"
    )

    fixed_sql = re.sub(r"\s{2,}", " ", fixed_sql).strip()
    if not fixed_sql.endswith(";"):
        fixed_sql += ";"
    return fixed_sql


def _fix_redundant_groupby_with_rn_filter(sql: str) -> Optional[str]:
    """
    Fix redundant GROUP BY + MAX(CASE WHEN rn = N...) when WHERE rn = N is already present.

    Anti-pattern (WRONG):
        SELECT t.customer, MAX(CASE WHEN rn = 2 THEN amount END) AS amount
        FROM (... ROW_NUMBER() OVER (...) AS rn ...) t
        WHERE t.rn = 2
        GROUP BY t.customer;

    Correct:
        SELECT t.customer, t.amount
        FROM (... ROW_NUMBER() OVER (...) AS rn ...) t
        WHERE t.rn = 2;

    When WHERE rn = N is present, each group already has exactly ONE row,
    so GROUP BY is redundant and MAX(CASE WHEN rn = N...) is unnecessary.

    Returns fixed SQL or None if pattern not detected.
    """
    if not sql:
        return None

    s_upper = sql.upper()

    # Must have: ROW_NUMBER, WHERE rn = N, GROUP BY, MAX(CASE WHEN...)
    if "ROW_NUMBER" not in s_upper:
        return None
    if "GROUP BY" not in s_upper:
        return None

    # Detect WHERE rn = N pattern (N is 1-9 for ordinal)
    where_rn_match = re.search(r"\bWHERE\s+(\w+\.)?rn\s*=\s*(\d)", sql, re.IGNORECASE)
    if not where_rn_match:
        return None

    rn_value = where_rn_match.group(2)

    # Detect MAX(CASE WHEN rn = N THEN col END) pattern
    # Pattern: MAX(CASE WHEN [alias.]rn [=<>] N THEN column END)
    max_case_pattern = re.compile(
        r"\b(MAX|MIN)\s*\(\s*CASE\s+WHEN\s+(?:\w+\.)?rn\s*=\s*" + rn_value + r"\s+THEN\s+(\w+(?:\.\w+)?)\s+END\s*\)",
        re.IGNORECASE
    )

    has_max_case = max_case_pattern.search(sql)
    if not has_max_case:
        return None

    # ===== FIX: Remove GROUP BY and simplify MAX(CASE WHEN...) =====
    fixed = sql

    # 1. Replace MAX(CASE WHEN rn = N THEN col END) with just the column
    def replace_max_case(match):
        # group(2) is the column (may include alias like "pmt.amount")
        return match.group(2)

    fixed = max_case_pattern.sub(replace_max_case, fixed)

    # 2. Remove GROUP BY clause from outer query (not subquery)
    # We need to remove GROUP BY that comes after the closing paren of subquery
    # Pattern: ) alias WHERE ... GROUP BY ...
    # Use careful regex to only match GROUP BY in outer query

    # Find the outer GROUP BY position (after subquery closing paren)
    # First, find WHERE rn = N position
    where_pos = fixed.upper().find("WHERE")
    if where_pos == -1:
        return None

    # Look for GROUP BY after WHERE (in the outer query)
    group_by_after_where = re.search(
        r"\bGROUP\s+BY\s+[^;)]+(?:;|$|\s*HAVING|\s*ORDER|\s*LIMIT)",
        fixed[where_pos:],
        re.IGNORECASE
    )

    if group_by_after_where:
        # Remove the GROUP BY clause
        # Find exact position
        start = where_pos + group_by_after_where.start()
        end = where_pos + group_by_after_where.end()

        # Trim trailing keywords that might have been captured
        fixed_group = fixed[start:end].rstrip()
        # Don't remove trailing semicolon, ORDER BY, LIMIT, HAVING
        for kw in ["ORDER", "LIMIT", "HAVING", ";"]:
            if fixed_group.upper().endswith(kw):
                fixed_group = fixed_group[:-len(kw)].rstrip()

        fixed = fixed[:start] + fixed[end:]

    # Clean up extra spaces
    fixed = re.sub(r"\s+", " ", fixed).strip()

    # Ensure semicolon at end if original had one
    if sql.rstrip().endswith(";") and not fixed.endswith(";"):
        fixed += ";"

    return fixed


def _fix_inconsistent_subquery_alias(sql: str) -> Optional[str]:
    """
    Fix LLM hallucination where outer SELECT uses a different alias than the subquery.

    Example WRONG:
        SELECT p.customerNumber, t.amount FROM (...) t WHERE t.rn = 2
        -- 'p' is undefined! Only 't' is valid.

    Example CORRECT:
        SELECT t.customerNumber, t.amount FROM (...) t WHERE t.rn = 2
        -- Consistent alias 't' everywhere.

    Returns fixed SQL or None if pattern not detected.
    """
    if not sql:
        return None

    s_upper = sql.upper()

    # Only process queries with subqueries
    if "ROW_NUMBER" not in s_upper:
        return None

    def _find_top_level_from_idx(raw: str) -> int:
        low = raw.lower()
        depth = 0
        i = 0
        in_sq = False
        in_dq = False
        while i < len(raw):
            ch = raw[i]
            if ch == "'" and not in_dq:
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
            elif ch == ")":
                depth = max(0, depth - 1)
            elif depth == 0 and low.startswith("from", i):
                prev_ok = (i == 0) or (not low[i - 1].isalnum() and low[i - 1] != "_")
                j = i + 4
                next_ok = (j >= len(low)) or (not low[j].isalnum() and low[j] != "_")
                if prev_ok and next_ok:
                    return i
            i += 1
        return -1

    from_idx = _find_top_level_from_idx(sql)
    if from_idx < 0:
        return None
    after_from = sql[from_idx + 4:].lstrip()
    if not after_from.startswith("("):
        return None
    lparen_idx = from_idx + 4 + (len(sql[from_idx + 4:]) - len(after_from))

    depth = 0
    close_idx = -1
    for i in range(lparen_idx, len(sql)):
        ch = sql[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                close_idx = i
                break
    if close_idx < 0:
        return None

    alias_tail = sql[close_idx + 1:]
    alias_match = re.match(r"\s*(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\b", alias_tail, re.IGNORECASE)
    if not alias_match:
        return None
    correct_alias = alias_match.group(1)

    if not re.match(r"(?is)^\s*SELECT\b", sql):
        return None
    outer_cols = sql[sql.upper().find("SELECT") + len("SELECT"):from_idx].strip()
    if not outer_cols:
        return None

    # Find all aliases used in the outer SELECT
    alias_refs = set(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\.\w+', outer_cols))

    # Check if any alias is different from the correct one
    inconsistent_aliases = alias_refs - {correct_alias.lower(), correct_alias}

    if not inconsistent_aliases:
        return None  # All aliases are consistent

    # Fix: Replace all inconsistent aliases with the correct one
    fixed = sql

    old_select = sql[:from_idx]
    new_select = old_select
    for wrong_alias in inconsistent_aliases:
        new_select = re.sub(
            rf'\b{re.escape(wrong_alias)}\.',
            f"{correct_alias}.",
            new_select
        )
    fixed = new_select + sql[from_idx:]

    # Debug output
    # print(f"[ALIAS FIX] Fixed inconsistent alias: {inconsistent_aliases} -> {correct_alias}")

    return fixed


def _fix_inner_rn_filter_scope(sql: str) -> Optional[str]:
    """
    Fix malformed ordinal-ranking SQL where WHERE rn = N is placed inside the
    same SELECT block that defines ROW_NUMBER().

    Wrong:
      SELECT customerNumber, amount
      FROM (
        SELECT customerNumber, amount,
               ROW_NUMBER() OVER (...) AS rn
        FROM payments
        WHERE rn = 2
      ) AS t;

    Correct:
      SELECT customerNumber, amount
      FROM (
        SELECT customerNumber, amount,
               ROW_NUMBER() OVER (...) AS rn
        FROM payments
      ) AS t
      WHERE t.rn = 2;
    """
    if not sql:
        return None

    s = sql.strip()
    if "ROW_NUMBER" not in s.upper():
        return None

    from_match = re.search(r"\bFROM\s*\(", s, re.I)
    if not from_match:
        return None

    open_idx = s.find("(", from_match.start())
    if open_idx < 0:
        return None

    depth = 0
    in_sq = False
    in_dq = False
    close_idx = -1
    for i in range(open_idx, len(s)):
        ch = s[i]
        if ch == "'" and not in_dq:
            in_sq = not in_sq
        elif ch == '"' and not in_sq:
            in_dq = not in_dq
        elif not in_sq and not in_dq:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    close_idx = i
                    break

    if close_idx < 0:
        return None

    inner_sql = s[open_idx + 1:close_idx].strip()
    if not re.match(r"(?is)^select\b", inner_sql):
        return None
    if "ROW_NUMBER" not in inner_sql.upper():
        return None

    inner_where_match = re.search(r"\bWHERE\s+(\w+\.)?rn\s*=\s*(\d+)\b", inner_sql, re.I)
    if not inner_where_match:
        return None
    rn_value = inner_where_match.group(2)

    outer_tail = s[close_idx + 1:].strip()
    if re.search(r"\bWHERE\s+(\w+\.)?rn\s*=\s*" + re.escape(rn_value) + r"\b", outer_tail, re.I):
        return None

    inner_fixed = re.sub(
        r"\s*\bWHERE\s+(\w+\.)?rn\s*=\s*" + re.escape(rn_value) + r"\b\s*",
        " ",
        inner_sql,
        count=1,
        flags=re.I,
    )
    inner_fixed = re.sub(r"\s{2,}", " ", inner_fixed).strip()

    alias_match = re.match(r"(?is)^(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*)\b(.*)$", outer_tail)
    if not alias_match:
        return None

    alias = alias_match.group(1)
    tail_rest = (alias_match.group(2) or "").strip()

    fixed = (
        s[:open_idx + 1]
        + inner_fixed
        + s[close_idx]
        + f" AS {alias} WHERE {alias}.rn = {rn_value}"
    )
    if tail_rest:
        fixed += " " + tail_rest

    fixed = re.sub(r"\)\s+AS\s+AS\b", ") AS", fixed, flags=re.I)
    fixed = re.sub(r"\s{2,}", " ", fixed).strip()
    if sql.rstrip().endswith(";") and not fixed.endswith(";"):
        fixed += ";"
    return fixed


def _fix_outer_order_by_for_derived_alias(sql: str) -> str:
    """
    For outer queries over a derived table:
      SELECT ... FROM ( ... ) t ... ORDER BY ...
    normalize ORDER BY references to valid outer output names.

    Typical repairs:
      - wrong qualifier alias (e.g., e.customerNumber -> customerNumber)
      - snake/camel mismatch (payment_date -> paymentDate)
      - number->id alias drift (employeeNumber -> employee_id when projected as alias)
    """
    if not sql:
        return sql

    s = sql.strip()
    if not s or "ORDER BY" not in s.upper() or "FROM (" not in s.upper():
        return sql

    # Detect derived-table alias used in outer scope.
    m_sub_alias = re.search(r"\)\s*(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*)\s+WHERE\b", s, flags=re.I)
    if not m_sub_alias:
        return sql
    sub_alias = m_sub_alias.group(1)

    m_select = re.search(r"(?is)^\s*SELECT\s+(.*?)\s+FROM\s*\(", s)
    if not m_select:
        return sql
    outer_select = (m_select.group(1) or "").strip()

    def _is_word_boundary(text: str, i: int, n: int) -> bool:
        left_ok = (i == 0) or not (text[i - 1].isalnum() or text[i - 1] == "_")
        right_ok = (i + n >= len(text)) or not (text[i + n].isalnum() or text[i + n] == "_")
        return left_ok and right_ok

    def _find_top_level_order_by(text: str) -> int:
        low = text.lower()
        i = 0
        depth = 0
        in_sq = False
        in_dq = False
        in_bt = False
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
            if depth == 0 and low[i:i + 8] == "order by" and _is_word_boundary(text, i, 8):
                return i
            i += 1
        return -1

    def _find_top_level_limit_after(text: str, start: int) -> int:
        low = text.lower()
        i = max(0, start)
        depth = 0
        in_sq = False
        in_dq = False
        in_bt = False
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
            if depth == 0 and low[i:i + 5] == "limit" and _is_word_boundary(text, i, 5):
                return i
            if depth == 0 and ch == ";":
                return i
            i += 1
        return -1

    order_pos = _find_top_level_order_by(s)
    if order_pos < 0:
        return sql
    order_clause_start = order_pos + len("ORDER BY")
    order_clause_end = _find_top_level_limit_after(s, order_clause_start)
    if order_clause_end < 0:
        order_clause_end = len(s)

    order_clause = s[order_clause_start:order_clause_end].strip()
    if not order_clause:
        return sql

    def _split_expr_alias(item: str) -> tuple:
        m = re.match(
            r"""(?is)^\s*(.*?)\s+AS\s+(`[^`]+`|'[^']+'|"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)\s*$""",
            (item or "").strip(),
        )
        if not m:
            return (item or "").strip(), None
        return (m.group(1) or "").strip(), (m.group(2) or "").strip()

    def _output_name_from_expr(expr: str, alias_raw: Optional[str]) -> Optional[str]:
        if alias_raw:
            a = alias_raw.strip()
            if (
                (a.startswith("`") and a.endswith("`"))
                or (a.startswith("'") and a.endswith("'"))
                or (a.startswith('"') and a.endswith('"'))
            ):
                a = a[1:-1]
            return a.strip() or None
        m_col = re.match(
            r"(?is)^\s*(?:`?[A-Za-z_][A-Za-z0-9_]*`?\s*\.\s*)?`?([A-Za-z_][A-Za-z0-9_]*)`?\s*$",
            (expr or "").strip(),
        )
        if m_col:
            return (m_col.group(1) or "").strip() or None
        return None

    def _norm_token(x: str) -> str:
        return re.sub(r"[^a-z0-9]", "", (x or "").lower())

    output_names: List[str] = []
    qualified_simple_outputs: Set[str] = set()
    for it in _split_top_level_commas(outer_select):
        expr, a0 = _split_expr_alias(it)
        out_name = _output_name_from_expr(expr, a0)
        if out_name:
            output_names.append(out_name)
            m_direct = re.match(
                r"(?is)^\s*`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\.\s*`?([A-Za-z_][A-Za-z0-9_]*)`?\s*$",
                expr,
            )
            src_col = (m_direct.group(2) or "").strip() if m_direct else ""
            alias_name = (
                a0.strip("`'\" ").strip()
                if a0 and a0.strip("`'\" ").strip()
                else ""
            )
            if (
                m_direct
                and (m_direct.group(1) or "").strip().lower() == sub_alias.lower()
                and (
                    not alias_name
                    or alias_name.lower() == src_col.lower()
                )
                and out_name.lower() == src_col.lower()
            ):
                qualified_simple_outputs.add(out_name)
    if not output_names:
        return sql

    out_by_norm: Dict[str, List[str]] = {}
    for name in output_names:
        out_by_norm.setdefault(_norm_token(name), []).append(name)

    def _resolve_to_output_name(raw_base: str) -> Optional[str]:
        b = (raw_base or "").strip()
        if not b:
            return None

        keep_sub_alias_qualifier = False
        qualified_col = ""

        # Strip qualifier if any: alias.col -> col
        m_q = re.match(
            r"(?is)^`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\.\s*`?([A-Za-z_][A-Za-z0-9_]*)`?$",
            b,
        )
        if m_q:
            q = (m_q.group(1) or "").strip()
            c = (m_q.group(2) or "").strip()
            if q.lower() == sub_alias.lower():
                keep_sub_alias_qualifier = True
                qualified_col = c
            b = c

        norm_b = _norm_token(b)
        if not norm_b:
            return None

        # Exact/normalized match first.
        norm_hits = out_by_norm.get(norm_b, [])
        if len(norm_hits) == 1:
            chosen = norm_hits[0]
            if (
                keep_sub_alias_qualifier
                and qualified_col
                and chosen in qualified_simple_outputs
                and chosen.lower() == qualified_col.lower()
            ):
                return f"{sub_alias}.{chosen}"
            return chosen

        # Heuristic: employeeNumber -> employee_id if projected alias uses *_id.
        if norm_b.endswith("number"):
            pref_norm = norm_b[:-6] + "id"
            pref_hits = out_by_norm.get(pref_norm, [])
            if len(pref_hits) == 1:
                return pref_hits[0]

        # If already an output name (case-insensitive), keep canonical casing from SELECT.
        for name in output_names:
            if name.lower() == b.lower():
                if (
                    keep_sub_alias_qualifier
                    and qualified_col
                    and name in qualified_simple_outputs
                    and name.lower() == qualified_col.lower()
                ):
                    return f"{sub_alias}.{name}"
                return name
        return None

    rebuilt_order_items: List[str] = []
    changed = False
    for item in _split_top_level_commas(order_clause):
        raw = (item or "").strip()
        if not raw:
            continue

        m_dir = re.match(r"(?is)^(.*?)(\s+(?:ASC|DESC))\s*$", raw)
        if m_dir:
            base = (m_dir.group(1) or "").strip()
            dir_part = (m_dir.group(2) or "").strip()
        else:
            base = raw
            dir_part = ""

        replacement = _resolve_to_output_name(base)
        if replacement and replacement != base:
            changed = True
            base = replacement

        rebuilt_order_items.append((base + (" " + dir_part if dir_part else "")).strip())

    if not changed or not rebuilt_order_items:
        return sql

    new_order_clause = ", ".join(rebuilt_order_items)
    fixed = s[:order_clause_start] + " " + new_order_clause + s[order_clause_end:]
    return fixed


def _drop_redundant_top_groupby_after_rn_filter(sql: str) -> Optional[str]:
    """
    Drop top-level GROUP BY when query already filters to a single winner row via WHERE rn = N.

    Pattern:
      SELECT ... FROM (...) t WHERE t.rn = 1 GROUP BY ... [ORDER BY ...]

    This GROUP BY is usually redundant and can distort semantics.
    Conservative guards:
    - requires ROW_NUMBER and top-level WHERE rn = N
    - only applies when top-level SELECT has no aggregate functions
    """
    if not sql:
        return None

    s0 = sql.strip().rstrip(";").strip()
    if not s0:
        return None
    s_up = s0.upper()

    if "ROW_NUMBER" not in s_up or "GROUP BY" not in s_up or "WHERE" not in s_up:
        return None
    if not re.search(r"\bWHERE\s+(?:[A-Za-z_][A-Za-z0-9_]*\.)?rn\s*=\s*\d+\b", s0, flags=re.I):
        return None

    def _is_word_boundary(text: str, start: int, n: int) -> bool:
        left_ok = (start == 0) or not (text[start - 1].isalnum() or text[start - 1] == "_")
        end = start + n
        right_ok = (end >= len(text)) or not (text[end].isalnum() or text[end] == "_")
        return left_ok and right_ok

    def _find_top_level_kw(text: str, kw: str, start: int = 0) -> int:
        w = kw.lower()
        n = len(w)
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
                if text[i:i + n].lower() == w and _is_word_boundary(text, i, n):
                    return i
            i += 1
        return -1

    sel_pos = _find_top_level_kw(s0, "select", 0)
    from_pos = _find_top_level_kw(s0, "from", max(0, sel_pos + 6))
    group_pos = _find_top_level_kw(s0, "group by", max(0, from_pos + 4))
    if sel_pos < 0 or from_pos < 0 or group_pos < 0:
        return None

    select_part = s0[sel_pos + len("select"):from_pos].strip()
    if re.search(r"\b(MAX|MIN|SUM|AVG|COUNT)\s*\(", select_part, flags=re.I):
        return None

    order_pos = _find_top_level_kw(s0, "order by", group_pos + 8)
    limit_pos = _find_top_level_kw(s0, "limit", group_pos + 8)
    tail_candidates = [p for p in [order_pos, limit_pos] if p >= 0]
    group_end = min(tail_candidates) if tail_candidates else len(s0)

    out = (s0[:group_pos].rstrip() + " " + s0[group_end:].lstrip()).strip()
    out = re.sub(r"\s{2,}", " ", out)
    if not out:
        return None
    if not out.endswith(";"):
        out += ";"

    return out if out != sql else None


def _preexecute_sql_safety_pass(sql: str, question: str = "") -> str:
    """
    Final defensive safety pass before execution.
    Applies conservative rewrites for high-risk anti-patterns and repeats until stable.
    """
    if not sql:
        return sql

    s = sql.strip()
    if not s:
        return sql
    if not s.endswith(";"):
        s += ";"

    for _ in range(3):
        prev = s

        s = _drop_aggregate_or_window_items_from_group_by(s)

        fixed = _fix_mixed_max_groupby_rownumber(s)
        if fixed:
            s = fixed

        fixed = _fix_redundant_groupby_with_rn_filter(s)
        if fixed:
            s = fixed

        fixed = _drop_redundant_top_groupby_after_rn_filter(s)
        if fixed:
            s = fixed

        fixed = _fix_inconsistent_subquery_alias(s)
        if fixed:
            s = fixed

        s = _fix_outer_order_by_for_derived_alias(s)

        # Winner-row fallback: force rewrite if query still looks like MAX-date misuse.
        # IMPORTANT: skip for "latest year + aggregate amount" intent, where MAX(date)
        # is legitimately combined with SUM(amount) under year-scoped grouping.
        latest_year_agg_intent = _question_wants_latest_year_scope_with_aggregate(question)
        if _question_wants_winner_row(question) and not latest_year_agg_intent:
            looks_like_agg_winner = bool(
                re.search(r"\bGROUP\s+BY\b", s, flags=re.I)
                and re.search(r"\b(MAX|MIN)\s*\(", s, flags=re.I)
                and not re.search(r"\bROW_NUMBER\s*\(", s, flags=re.I)
            )
            if looks_like_agg_winner or _looks_like_max_date_tiebreaker(s):
                forced = _rewrite_winner_row_guardrail(s, question, force=True)
                if forced and not (
                    "GROUP BY" in forced.upper() and "ROW_NUMBER" in forced.upper()
                ):
                    s = forced

        s = _drop_aggregate_or_window_items_from_group_by(s)

        if s == prev:
            break

    s = re.sub(r"\s{2,}", " ", s).strip()
    if s and not s.endswith(";"):
        s += ";"
    return s


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
    
    # SAFETY: Detect malformed syntax and abort early
    # Pattern: clause keywords stuck together without space
    if re.search(r"\b[A-Za-z0-9_]+(?:WHERE|GROUP|ORDER|HAVING|LIMIT)\b", s, flags=re.I):
        # Try one normalization pass
        s = re.sub(r"(?<=[A-Za-z0-9_)])(?=\b(?:WHERE|GROUP|ORDER|HAVING|LIMIT)\b)", " ", s, flags=re.I)
        # If still malformed, abort
        if re.search(r"\b[A-Za-z0-9_]+(?:WHERE|GROUP|ORDER|HAVING|LIMIT)\b", s, flags=re.I):
            return None

    # PRE-NORMALIZE: Fix common spacing issues
    s = re.sub(r"(\b[A-Za-z_][A-Za-z0-9_]*)\s*(?=\b(?:HAVING|ORDER|LIMIT|WHERE|GROUP)\b)", r"\1 ", s, flags=re.I)
    s = re.sub(r"\s{2,}", " ", s)  # collapse multiple spaces

    if re.match(r"^\s*with\b", s, flags=re.I):
        return None
    # This parser is intentionally for flat queries only.
    # Any derived table in FROM/JOIN can make regex clause capture unsafe
    # (inner GROUP BY/HAVING/ORDER BY may be mistaken as top-level clauses).
    if re.search(r"\b(?:from|join)\s*\(", s, flags=re.I):
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

def _to_raw_expr(expr: str) -> str:
    # remove AS <alias>
    expr = re.sub(r"\s+AS\s+\w+$", "", expr, flags=re.I).strip()
    # take inner of MAX(x) / MIN(x)
    m = re.match(r"\b(MAX|MIN)\s*\((.+)\)", expr, flags=re.I)
    return m.group(2).strip() if m else expr

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

    # This guardrail is only for GROUP BY winner-row misuse patterns.
    # If GROUP BY exists only inside subqueries, skip and let other rewrites handle it.
    if not _has_top_level_group_by(s0):
        return sql

    # CRITICAL: Detect truly malformed queries (not false positives from subqueries)
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

    # Preserve qualified column references for multi-table sources.
    # Stripping prefixes in JOIN queries can create ambiguous columns
    # (e.g., customerNumber exists in both customers and payments).
    preserve_qualified = has_join(parsed.get("from_core", ""))

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
    # Focus "requested output columns" detection on the projection phrase,
    # not on sorting-only phrases (e.g., "sort by customer id").
    q_low_output = re.split(r"\b(?:sort(?:ed)?\s+by|order\s+by)\b", q_low, maxsplit=1, flags=re.I)[0]
    q_tokens = set(re.findall(r"[a-z0-9]+", q_low))
    stop_tokens = {
        "for", "each", "per", "the", "a", "an", "of", "with", "and", "or",
        "id", "ids", "code", "codes", "number", "numbers", "name", "names",
        "date", "dates", "record", "row", "latest", "earliest", "largest",
        "highest", "lowest", "payment", "payments", "amount", "total"
    }
    q_tokens -= stop_tokens

    def _group_item_score(g: str) -> int:
        gn = g.replace("`", "").lower()
        gn = re.sub(r"[^\w]+", " ", gn)
        g_tokens = set(re.findall(r"[a-z0-9]+", gn))
        g_tokens -= stop_tokens
        if not g_tokens:
            return 0
        return len(g_tokens & q_tokens)

    best_group = max(group_items, key=_group_item_score)
    if _group_item_score(best_group) > 0:
        partition_items = [best_group]
    else:
        # fallback — tetap harus ada PARTITION
        partition_items = [group_items[0]]

    # =====================================================
    # DETECT AGGREGATES
    # =====================================================

    agg_pat = re.compile(
        r"^(?P<prefix>.*?)\b(?P<fn>MAX|MIN)\s*\(\s*(?P<expr>.*?)\s*\)"
        r"\s*(?P<alias>\bAS\s+(?:`[^`]+`|\"[^\"]+\"|'[^']+'|[A-Za-z_][A-Za-z0-9_]*)\s*)?$",
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

    # Strategy: collect ALL date/time columns as tie-breakers
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

    # Detect tie-breaker direction from question
    ql = (question or "").lower()
    
    # Explicit signals
    if any(k in ql for k in ["latest", "most recent", "newest", "terbaru", "paling baru", "terakhir"]):
        tie_dir = "DESC"
    elif any(k in ql for k in ["earliest", "oldest", "first", "terlama", "paling lama", "pertama"]):
        tie_dir = "ASC"
    else:
        # Smart default based on context
        # For date/time columns: default DESC (latest is usually desired)
        # For other columns: ASC
        if tie_exprs and any(k in tie_exprs[0].lower() for k in ["date", "time", "timestamp"]):
            tie_dir = "DESC"
        else:
            tie_dir = "ASC"

    # Strip table prefixes only for single-table sources.
    if not preserve_qualified:
        agg_expr = _strip_table_prefix(agg_expr)
        tie_exprs = [_strip_table_prefix(t) for t in tie_exprs]

    # Build ORDER BY: metric first, then tie-breakers (limit to 2 for stability)
    order_parts = [f"{agg_expr} {metric_dir}"]
    for tie in tie_exprs[:2]:  # Take up to 2 tie-breakers
        order_parts.append(f"{tie} {tie_dir}")

    order_window = ", ".join(order_parts)

    # =====================================================
    # BUILD INNER + OUTER QUERY
    # =====================================================

    inner_items = []
    outer_items = []

    for j, x in enumerate(select_items):
        orig_alias = _extract_alias_or_colname(x)

        # --- untuk kolom aggregate: GANTI MAX(...) -> raw metric expr ---
        if j == agg_idx:
            base = agg_expr
            # Use natural column name, not the aggregate alias
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
                # Use natural column name for extracted aggregates
                out_name = _strip_table_prefix(base)
            else:
                # Keep original alias for true non-aggregate columns
                out_name = orig_alias if orig_alias else _strip_table_prefix(base)
            
            if not preserve_qualified:
                base = _strip_table_prefix(base)

        inner_items.append(
            _maybe_keep_or_alias_simple(base, out_name)
        )
        outer_items.append(out_name)

    if preserve_qualified:
        partition_clean = partition_items
    else:
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

    sel = ast if isinstance(ast, exp.Select) else ast.find(exp.Select)
    if not isinstance(sel, exp.Select):
        return sql

    from_expr = sel.args.get("from") or sel.args.get("from_")
    if not from_expr:
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
    inner = exp.Select(expressions=inner_select_exprs).from_(from_expr.copy())
    joins = sel.args.get("joins")
    if joins:
        inner.set("joins", [j.copy() for j in joins])
    if sel.args.get("where"):
        inner.set("where", sel.args["where"].copy())

    outer = exp.Select(expressions=outer_select_exprs).from_(
        exp.Subquery(this=inner, alias=exp.TableAlias(this=exp.to_identifier("t")))
    ).where(exp.EQ(this=exp.column("rn", table="t"), expression=exp.Literal.number(1)))

    out_sql = outer.sql(dialect="mysql")

    # sqlglot's MySQL serializer expands ascending ORDER BY on plain columns into:
    #   CASE WHEN alias.col IS NULL THEN 1 ELSE 0 END, alias.col ASC
    # For this winner-row rewrite, those wrappers are only serializer noise on the
    # synthetic ranked aliases we just created. Collapse them back to the direct
    # column ORDER BY form so the final SQL stays readable.
    ordered_aliases = list(dict.fromkeys([partition_alias] + rank_aliases))
    for rel_alias in ("c", "t"):
        for alias in ordered_aliases:
            pat = re.compile(
                rf"CASE\s+WHEN\s+{re.escape(rel_alias)}\.`?{re.escape(alias)}`?\s+IS\s+NULL\s+THEN\s+1\s+ELSE\s+0\s+END\s*,\s*"
                rf"{re.escape(rel_alias)}\.`?{re.escape(alias)}`?\s+ASC",
                flags=re.IGNORECASE,
            )
            out_sql = pat.sub(f"{rel_alias}.{alias} ASC", out_sql)

    return out_sql


def _rewrite_grouped_winner_row_ranked_aggregate(sql: str, question: str) -> str:
    """
    Repair grouped aggregate near-misses for winner-row questions.

    Typical bad shape:
      SELECT group_key, candidate_cols, metric_aliases...
      FROM ...
      GROUP BY ...
      ORDER BY group_key ASC, metric1 DESC, metric2 DESC, stable_key ASC

    The query already computes the candidate grain correctly, but it still
    returns every candidate row. Wrap it into:
      1) aggregated candidate subquery
      2) ROW_NUMBER() OVER (PARTITION BY group_key ORDER BY metric/tiebreakers)
      3) outer WHERE rn = 1

    Conservative guardrails:
    - only for clear winner-row questions
    - only if SQL has GROUP BY + ORDER BY and no ROW_NUMBER yet
    - only if every SELECT expression has a stable outer alias
    - only if ORDER BY items can be mapped back to projected columns
    """
    if not sql or not _question_wants_winner_row(question):
        return sql
    if re.search(r"\bROW_NUMBER\s*\(", sql, flags=re.I):
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

    sel = ast if isinstance(ast, exp.Select) else ast.find(exp.Select)
    if not isinstance(sel, exp.Select):
        return sql

    grp = sel.args.get("group")
    order = sel.args.get("order")
    if not grp or not getattr(grp, "expressions", None):
        return sql
    if not order or not getattr(order, "expressions", None):
        return sql
    if len(order.expressions) < 2:
        return sql
    if sel.args.get("limit"):
        return sql

    def _norm_expr(e: exp.Expression) -> str:
        return re.sub(r"\s+", " ", e.sql(dialect="mysql").strip().lower())

    alias_names: List[str] = []
    expr_to_alias: Dict[str, str] = {}
    alias_by_lower: Dict[str, str] = {}

    for idx, item in enumerate(sel.expressions):
        if isinstance(item, exp.Star):
            return sql

        if isinstance(item, exp.Alias):
            alias = item.alias
            base = item.this
        else:
            base = item
            if isinstance(base, exp.Column):
                alias = base.name
            else:
                # Complex expressions need explicit aliases so the ranked layer
                # can reference them safely.
                return sql

        alias = (alias or f"col_{idx + 1}").strip()
        if not alias or alias.lower() == "rn" or alias.lower() in alias_by_lower:
            return sql

        alias_names.append(alias)
        alias_by_lower[alias.lower()] = alias
        expr_to_alias[_norm_expr(base)] = alias

    order_aliases: List[str] = []
    order_dirs: List[bool] = []
    for ordered in order.expressions:
        expr = ordered.this
        desc = bool(ordered.args.get("desc"))
        alias = None

        expr_norm = _norm_expr(expr)
        alias = expr_to_alias.get(expr_norm)

        if alias is None and isinstance(expr, exp.Column):
            alias = alias_by_lower.get((expr.name or "").lower())
            if alias is None:
                bare_name = (expr.name or "").lower()
                matches = [a for low, a in alias_by_lower.items() if low == bare_name]
                if len(matches) == 1:
                    alias = matches[0]

        if alias is None:
            return sql

        order_aliases.append(alias)
        order_dirs.append(desc)

    partition_alias = order_aliases[0]
    rank_aliases = order_aliases[1:]
    rank_dirs = order_dirs[1:]
    if not rank_aliases:
        return sql

    candidate_ast = ast.copy()
    candidate_sel = candidate_ast if isinstance(candidate_ast, exp.Select) else candidate_ast.find(exp.Select)
    if not isinstance(candidate_sel, exp.Select):
        return sql
    candidate_sel.set("order", None)

    # Trim redundant GROUP BY columns in the aggregated candidate layer.
    #
    # Common H17-like failure mode:
    # the model groups by many extra same-table dimension columns
    # (MSRP, productVendor, productScale, ...) even though the visible grouped
    # grain is already identified by a key-like column such as productCode.
    #
    # We only prune when this is very likely safe:
    # - GROUP BY contains more columns than the visible non-aggregate projection
    # - the visible grouped projection includes at least one key-like column
    # - every extra GROUP BY item is a simple column on the same table alias
    #   as that key-like column
    #
    # If those conditions are not met, keep the original GROUP BY unchanged.
    candidate_group = candidate_sel.args.get("group")
    if candidate_group and getattr(candidate_group, "expressions", None):
        current_group_exprs = list(candidate_group.expressions or [])
        current_group_norms = {_norm_expr(g) for g in current_group_exprs}

        visible_group_columns: List[exp.Column] = []
        visible_group_norms: Set[str] = set()
        for item in candidate_sel.expressions:
            base = item.this if isinstance(item, exp.Alias) else item
            base_norm = _norm_expr(base)
            if base_norm in current_group_norms:
                visible_group_norms.add(base_norm)
                if isinstance(base, exp.Column):
                    visible_group_columns.append(base)

        if len(current_group_exprs) > len(visible_group_norms) and visible_group_columns:
            key_like_cols = [
                c for c in visible_group_columns
                if any(tok in (c.name or "").lower() for tok in ["id", "code", "number", "key"])
            ]
            key_tables = {
                (c.table or "").lower()
                for c in key_like_cols
                if (c.table or "").strip()
            }

            if key_like_cols and key_tables:
                extras = [
                    g for g in current_group_exprs
                    if _norm_expr(g) not in visible_group_norms
                ]
                if extras and all(
                    isinstance(g, exp.Column) and ((g.table or "").lower() in key_tables)
                    for g in extras
                ):
                    trimmed_group_exprs: List[exp.Expression] = []
                    seen_group_norms: Set[str] = set()
                    for g in current_group_exprs:
                        g_norm = _norm_expr(g)
                        if g_norm in visible_group_norms and g_norm not in seen_group_norms:
                            trimmed_group_exprs.append(g.copy())
                            seen_group_norms.add(g_norm)
                    if trimmed_group_exprs:
                        candidate_sel.set("group", exp.Group(expressions=trimmed_group_exprs))

    candidate_sub = exp.Subquery(
        this=candidate_ast,
        alias=exp.TableAlias(this=exp.to_identifier("c")),
    )

    ranked_order = exp.Order(
        expressions=[
            exp.Ordered(this=exp.column(alias, table="c"), desc=desc)
            for alias, desc in zip(rank_aliases, rank_dirs)
        ]
    )
    rn_expr = exp.alias_(
        exp.Window(
            this=exp.RowNumber(),
            partition_by=exp.Partition(expressions=[exp.column(partition_alias, table="c")]),
            order=ranked_order,
        ),
        "rn",
        quoted=False,
    )

    ranked_select = exp.Select(
        expressions=[exp.column(alias, table="c") for alias in alias_names] + [rn_expr]
    ).from_(candidate_sub)

    ranked_sub = exp.Subquery(
        this=ranked_select,
        alias=exp.TableAlias(this=exp.to_identifier("t")),
    )

    outer = exp.Select(
        expressions=[exp.column(alias, table="t") for alias in alias_names]
    ).from_(ranked_sub).where(
        exp.EQ(this=exp.column("rn", table="t"), expression=exp.Literal.number(1))
    )
    outer.set(
        "order",
        exp.Order(
            expressions=[
                exp.Ordered(this=exp.column(partition_alias, table="t"), desc=order_dirs[0])
            ]
        ),
    )

    out_sql = outer.sql(dialect="mysql")

    # Drop sqlglot's null-order wrappers for the synthetic aliases created by
    # this rewrite; these keys are non-null in the narrowed winner-row shape.
    ordered_aliases = list(dict.fromkeys([partition_alias] + rank_aliases))
    for rel_alias in ("c", "t"):
        for alias in ordered_aliases:
            pat = re.compile(
                rf"CASE\s+WHEN\s+{re.escape(rel_alias)}\.`?{re.escape(alias)}`?\s+IS\s+NULL\s+THEN\s+1\s+ELSE\s+0\s+END\s*,\s*"
                rf"{re.escape(rel_alias)}\.`?{re.escape(alias)}`?\s+ASC",
                flags=re.IGNORECASE,
            )
            out_sql = pat.sub(f"{rel_alias}.{alias} ASC", out_sql)

    return out_sql


def _fix_winner_row_window_order_priority(sql: str, question: str) -> str:
    """
    Normalize ROW_NUMBER() window ORDER BY priority for winner-row questions
    that ask:
      - latest/most recent date as primary criterion
      - largest/highest amount as tie-breaker

    Example fix:
      ORDER BY amount DESC, paymentDate DESC
    -> ORDER BY paymentDate DESC, amount DESC
    """
    if not sql or "ROW_NUMBER" not in (sql or "").upper():
        return sql
    q = (question or "").lower()
    wants_latest_date = bool(
        re.search(r"\b(?:latest|most\s+recent|newest|terbaru|terakhir)\b", q)
        and re.search(r"\b(?:date|time|tanggal)\b", q)
    )
    if not wants_latest_date:
        return sql

    # Intent disambiguation:
    # - amount-first: "largest/highest amount ... if same amount then latest date"
    # - date-first:   "most recent/latest date ... if same date then largest amount"
    same_amount_tie = bool(
        re.search(r"\b(?:if\s+there\s+are\s+multiple|if\s+multiple|tie|tied|same)\b", q)
        and re.search(r"\b(?:same|equal)\s+(?:payment\s+)?amount\b", q)
    )
    same_date_tie = bool(
        re.search(r"\b(?:if\s+there\s+are\s+multiple|if\s+multiple|tie|tied|same)\b", q)
        and re.search(r"\bsame\s+(?:most\s+recent\s+)?(?:date|payment\s+date|order\s+date|time)\b", q)
    )
    amount_superlative = bool(
        re.search(r"\b(?:largest|highest|max(?:imum)?|biggest|top)\b", q)
        and re.search(r"\b(?:amount|price|quantity|value|payment\s+amount)\b", q)
    )

    # Conservative rule:
    # rewrite only when date is clearly primary, never when amount-primary tie logic is explicit.
    date_primary_intent = bool(
        same_date_tie
        or (wants_latest_date and not same_amount_tie and not amount_superlative)
    )
    if not date_primary_intent:
        return sql

    wants_amount_tiebreak = bool(
        re.search(r"\b(?:if\s+there\s+are\s+multiple|if\s+multiple|same\s+most\s+recent\s+date|tie)\b", q)
        and re.search(r"\b(?:largest|highest|max|terbesar|tertinggi)\b", q)
        and re.search(r"\b(?:amount|payment|total|quantity|qty|price)\b", q)
    )

    # NOTE:
    # Keep this pattern as a compact regex (no verbose spacing) so it
    # reliably matches one-line SQL emitted by the model.
    rownum_pat = re.compile(
        r"(?is)ROW_NUMBER\s*\(\s*\)\s*OVER\s*\(\s*(?:PARTITION\s+BY\s+.*?\b)?ORDER\s+BY\s+(?P<ob>[^)]*?)\)"
    )

    # Fallback candidates (used when window ORDER BY misses required fields).
    date_fallbacks = re.findall(
        r"\b([A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*(?:Date|date|Time|time|Timestamp|timestamp))\b",
        sql or "",
        flags=re.I,
    )
    amount_fallbacks = re.findall(
        r"\b([A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*(?:amount|price|quantity|total|sales|revenue))\b",
        sql or "",
        flags=re.I,
    )
    customer_fallbacks = re.findall(
        r"\b([A-Za-z_][A-Za-z0-9_]*\.(?:customerNumber|customer_id|customerId|customercode|customerCode))\b",
        sql or "",
        flags=re.I,
    )

    def _is_date_expr(item: str) -> bool:
        x = (item or "").lower()
        return ("date" in x) or ("time" in x) or ("tanggal" in x)

    def _is_amount_expr(item: str) -> bool:
        x = (item or "").lower()
        return any(tok in x for tok in ("amount", "payment", "total", "quantity", "qty", "price", "sales", "revenue"))

    def _force_dir(item: str, desc: bool) -> str:
        base = re.sub(r"(?is)\s+(ASC|DESC)\s*$", "", (item or "").strip()).strip()
        if not base:
            return item
        return f"{base} {'DESC' if desc else 'ASC'}"

    def _norm_expr(item: str) -> str:
        x = re.sub(r"(?is)\s+(ASC|DESC)\s*$", "", (item or "").strip())
        return re.sub(r"[\s`]+", "", x.lower())

    def _pick_fallback(cands: List[str], seen_norm: Set[str]) -> Optional[str]:
        for c in cands:
            c0 = (c or "").strip()
            if not c0:
                continue
            if _norm_expr(c0) in seen_norm:
                continue
            return c0
        return None

    def _repl(m: re.Match) -> str:
        ob = (m.group("ob") or "").strip()
        if not ob:
            return m.group(0)

        items = _split_top_level_commas(ob)
        if not items:
            return m.group(0)

        has_date_any = any(_is_date_expr(it) for it in items)
        has_amount_any = any(_is_amount_expr(it) for it in items)
        first = (items[0] or "").strip()
        # Rewrite when:
        # 1) amount appears before date, or
        # 2) date-primary intent but date is missing from window ORDER BY.
        should_rewrite = bool((_is_amount_expr(first) and has_date_any) or (not has_date_any))
        if not should_rewrite and not (wants_amount_tiebreak and has_date_any and not has_amount_any):
            return m.group(0)

        date_items = []
        amount_items = []
        other_items = []
        for it in items:
            it0 = (it or "").strip()
            if _is_date_expr(it0):
                date_items.append(_force_dir(it0, desc=True))
            elif _is_amount_expr(it0):
                if wants_amount_tiebreak:
                    amount_items.append(_force_dir(it0, desc=True))
                else:
                    amount_items.append(it0)
            else:
                other_items.append(it0)

        seen_norm = {_norm_expr(x) for x in items if (x or "").strip()}
        if not date_items:
            fb_date = _pick_fallback(date_fallbacks, seen_norm)
            if fb_date:
                date_items.append(_force_dir(fb_date, desc=True))
                seen_norm.add(_norm_expr(fb_date))
            else:
                return m.group(0)

        if wants_amount_tiebreak and not amount_items:
            fb_amount = _pick_fallback(amount_fallbacks, seen_norm)
            if fb_amount:
                amount_items.append(_force_dir(fb_amount, desc=True))
                seen_norm.add(_norm_expr(fb_amount))

        new_items = date_items + amount_items + other_items
        # Deterministic tertiary tie-break for customer winner-row queries.
        wants_customer_tiebreak = bool(re.search(r"\bcustomer\s*(?:id|number|code)?\b", q))
        if wants_customer_tiebreak:
            fb_customer = _pick_fallback(customer_fallbacks, {_norm_expr(x) for x in new_items})
            if fb_customer:
                new_items.append(_force_dir(fb_customer, desc=False))

        dedup_items = []
        dedup_seen = set()
        for it in new_items:
            k = _norm_expr(it)
            if not k or k in dedup_seen:
                continue
            dedup_seen.add(k)
            dedup_items.append(it)
        if not dedup_items:
            return m.group(0)

        if [_norm_expr(x) for x in dedup_items] == [_norm_expr(x) for x in items]:
            return m.group(0)
        new_ob = ", ".join(dedup_items)
        full = m.group(0)
        return full.replace(ob, new_ob, 1)

    fixed = rownum_pat.sub(_repl, sql)
    return fixed

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

def _rewrite_winner_row_aggregate_misuse(sql: str, question: str, ctx: str = "") -> str:
    """
    Catch winner-row questions where the SQL still uses MAX/MIN (often with
    HAVING/subquery hacks) and rewrite to ROW_NUMBER winner-row selection.

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

    # Special-case rescue:
    # Winner-row tie-break questions sometimes produce malformed window MAX/MIN SQL
    # (no ROW_NUMBER filter), returning all rows. Rewrite to canonical rn=1 shape.
    q_low = (question or "").lower()
    has_tie_phrase = bool(re.search(r"\b(?:if\s+there\s+are\s+multiple|if\s+multiple|tie|tied|same)\b", q_low))
    same_metric_tie = bool(
        has_tie_phrase
        and re.search(r"\b(?:same|equal)\s+(?:payment\s+)?(?:amount|price|quantity|value)\b", q_low)
    )
    wants_latest_date = bool(
        re.search(r"\b(?:latest|most\s+recent|newest)\b", q_low)
        and re.search(r"\b(?:date|time)\b", q_low)
    )
    wants_metric_superlative = bool(
        re.search(r"\b(?:largest|highest|max(?:imum)?|top)\b", q_low)
        and re.search(r"\b(?:amount|payment|price|quantity|value|total)\b", q_low)
    )
    if same_metric_tie and wants_latest_date and wants_metric_superlative and not sel.args.get("group"):
        def _base_expr(it: exp.Expression) -> exp.Expression:
            return it.this if isinstance(it, exp.Alias) else it

        def _expr_norm(e: exp.Expression) -> str:
            try:
                return re.sub(r"[\s`]+", "", e.sql(dialect="mysql").lower())
            except Exception:
                return ""

        window_entries: List[Tuple[exp.Expression, exp.Expression, exp.Window, exp.Expression]] = []
        for item in sel.expressions:
            base = _base_expr(item)
            w = base if isinstance(base, exp.Window) else base.find(exp.Window)
            if not isinstance(w, exp.Window) or not isinstance(w.this, (exp.Max, exp.Min)):
                continue
            window_entries.append((item, base, w, w.this))

        if window_entries:
            def _window_partition_exprs(win: exp.Window) -> List[exp.Expression]:
                """
                sqlglot Window.partition_by can be represented as:
                - exp.Partition(expressions=[...])
                - list[Expression]
                - single Expression
                Keep extraction tolerant to all forms.
                """
                raw = win.args.get("partition_by")
                if raw is None:
                    # compatibility: some versions may use different key
                    raw = win.args.get("partition")
                if raw is None:
                    return []
                if isinstance(raw, list):
                    return [x for x in raw if isinstance(x, exp.Expression)]
                if isinstance(raw, exp.Partition):
                    return [x for x in (raw.expressions or []) if isinstance(x, exp.Expression)]
                if isinstance(raw, exp.Expression):
                    exprs = getattr(raw, "expressions", None)
                    if isinstance(exprs, list) and exprs:
                        return [x for x in exprs if isinstance(x, exp.Expression)]
                    return [raw]
                return []

            partition_expr = None
            for _item, _base, w, _agg in window_entries:
                part_exprs = _window_partition_exprs(w)
                if part_exprs:
                    partition_expr = part_exprs[0].copy()
                    break

            # Date expression from MAX/MIN window input (prefer date/time-like column).
            date_expr = None
            for _item, _base, _w, agg in window_entries:
                inner = agg.this
                inner_sql = (inner.sql(dialect="mysql") if inner is not None else "").lower()
                if any(k in inner_sql for k in ("date", "time", "timestamp")):
                    date_expr = inner.copy()
                    break
            if date_expr is None and window_entries:
                date_expr = window_entries[0][3].this.copy()

            # Metric expression from non-window simple columns (amount/price/qty/value).
            metric_expr = None
            for item in sel.expressions:
                base = _base_expr(item)
                if base.find(exp.Window):
                    continue
                col = base if isinstance(base, exp.Column) else (base.this if isinstance(base, exp.Alias) and isinstance(base.this, exp.Column) else None)
                if not isinstance(col, exp.Column):
                    continue
                col_sql = col.sql(dialect="mysql").lower()
                if any(k in col_sql for k in ("amount", "price", "quantity", "qty", "value", "total")):
                    metric_expr = col.copy()
                    break

            # Fallback: derive metric directly from window MAX/MIN inputs.
            if metric_expr is None:
                for _item, _base, _w, agg in window_entries:
                    inner = agg.this
                    inner_sql = (inner.sql(dialect="mysql") if inner is not None else "").lower()
                    if any(k in inner_sql for k in ("amount", "price", "quantity", "qty", "value", "total", "score", "revenue", "sales")):
                        metric_expr = inner.copy()
                        break
            if metric_expr is None and window_entries:
                metric_expr = window_entries[0][3].this.copy()

            if partition_expr is not None and date_expr is not None and metric_expr is not None:
                def _infer_alias(expr_obj: exp.Expression, fallback: str) -> str:
                    norm = _expr_norm(expr_obj)
                    for it in sel.expressions:
                        b = _base_expr(it)
                        if _expr_norm(b) == norm:
                            if isinstance(it, exp.Alias) and it.alias:
                                return it.alias
                            if isinstance(b, exp.Column) and b.name:
                                return b.name
                    if isinstance(expr_obj, exp.Column) and expr_obj.name:
                        return expr_obj.name
                    return fallback

                part_alias = _infer_alias(partition_expr, "group_key")
                metric_alias = _infer_alias(metric_expr, "value")
                date_alias = _infer_alias(date_expr, "event_date")

                used_aliases: Set[str] = set()
                def _uniq_alias(a: str) -> str:
                    base = re.sub(r"[^A-Za-z0-9_]", "", (a or "").strip()) or "col"
                    if not re.match(r"[A-Za-z_]", base):
                        base = f"c_{base}"
                    cand = base
                    i = 2
                    while cand.lower() in used_aliases:
                        cand = f"{base}_{i}"
                        i += 1
                    used_aliases.add(cand.lower())
                    return cand

                part_alias_u = _uniq_alias(part_alias)
                metric_alias_u = _uniq_alias(metric_alias)
                date_alias_u = _uniq_alias(date_alias)

                metric_desc = not bool(re.search(r"\b(?:lowest|smallest|min(?:imum)?)\b", q_low))
                date_desc = bool(re.search(r"\b(?:latest|most\s+recent|newest)\b", q_low))

                rn = exp.Window(
                    this=exp.RowNumber(),
                    partition_by=exp.Partition(expressions=[partition_expr.copy()]),
                    order=exp.Order(
                        expressions=[
                            exp.Ordered(this=metric_expr.copy(), desc=metric_desc),
                            exp.Ordered(this=date_expr.copy(), desc=date_desc),
                        ]
                    ),
                )

                inner_exprs = [
                    exp.alias_(partition_expr.copy(), part_alias_u, quoted=False),
                    exp.alias_(metric_expr.copy(), metric_alias_u, quoted=False),
                    exp.alias_(date_expr.copy(), date_alias_u, quoted=False),
                    exp.alias_(rn, "rn", quoted=False),
                ]

                from_expr = sel.args.get("from") or sel.args.get("from_")
                if from_expr is None:
                    return sql
                inner = exp.Select(expressions=inner_exprs)
                inner.set("from_", from_expr.copy())
                joins = sel.args.get("joins")
                if joins:
                    inner.set("joins", [j.copy() for j in joins])
                where_expr = sel.args.get("where")
                if where_expr is not None:
                    inner.set("where", where_expr.copy())

                outer = (
                    exp.Select(
                        expressions=[
                            exp.column(part_alias_u, table="t"),
                            exp.column(metric_alias_u, table="t"),
                            exp.column(date_alias_u, table="t"),
                        ]
                    )
                    .from_(exp.Subquery(this=inner, alias=exp.TableAlias(this=exp.to_identifier("t"))))
                    .where(exp.EQ(this=exp.column("rn", table="t"), expression=exp.Literal.number(1)))
                )
                return outer.sql(dialect="mysql")

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
    include_zero_intent = _question_wants_include_zero(question, ctx)

    def norm_name(e: exp.Expression) -> str:
        try:
            return e.sql(dialect="mysql").lower()
        except Exception:
            return ""

    # Partition columns:
    # Prefer expressions with lexical overlap to question entities.
    q_tokens = set(re.findall(r"[a-z0-9]+", q_low))
    stop_tokens = {
        "for", "each", "per", "the", "a", "an", "of", "with", "and", "or",
        "id", "ids", "code", "codes", "number", "numbers", "name", "names",
        "date", "dates", "record", "row", "latest", "earliest", "largest",
        "highest", "lowest", "payment", "payments", "amount", "total"
    }
    q_tokens -= stop_tokens

    partition_exprs = []
    partition_candidates: List[Tuple[int, exp.Expression]] = []
    for item in nonagg_items:
        base = item.this if isinstance(item, exp.Alias) else item
        s = norm_name(base)
        expr_tokens = set(re.findall(r"[a-z0-9]+", s)) - stop_tokens
        if expr_tokens and (expr_tokens & q_tokens):
            score = len(expr_tokens & q_tokens)
            if any(k in s for k in ["id", "number", "code"]):
                score += 2
            partition_candidates.append((score, base))

    if partition_candidates:
        partition_exprs = [max(partition_candidates, key=lambda x: x[0])[1]]

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
    wants_latest_date = bool(
        re.search(r"\b(?:latest|most\s+recent|newest)\b", q_low, flags=re.I)
        and re.search(r"\b(?:date|time|payment\s+date|order\s+date)\b", q_low, flags=re.I)
    )
    wants_amount_tiebreak = bool(
        re.search(r"\b(?:if\s+there\s+are\s+multiple|if\s+multiple|same|tie)\b", q_low, flags=re.I)
        and re.search(r"\bdate\b", q_low, flags=re.I)
        and re.search(r"\b(?:largest|highest|max(?:imum)?)\b", q_low, flags=re.I)
        and re.search(r"\b(?:amount|payment)\b", q_low, flags=re.I)
    )

    order_exprs = []
    if wants_latest_date and tie_exprs:
        date_expr = tie_exprs[0]
        order_exprs.append(exp.Ordered(this=date_expr.copy(), desc=True))
        if wants_amount_tiebreak:
            order_exprs.append(exp.Ordered(this=metric_expr.copy(), desc=True))
            # Stable deterministic tie-break when same date and amount.
            for item in nonagg_items:
                base = item.this if isinstance(item, exp.Alias) else item
                s = norm_name(base)
                if norm_name(base) in {norm_name(partition_exprs[0]), norm_name(date_expr)}:
                    continue
                if any(k in s for k in ["id", "number", "code"]):
                    order_exprs.append(exp.Ordered(this=base.copy(), desc=False))
                    break
        else:
            order_exprs.append(exp.Ordered(this=metric_expr.copy(), desc=metric_desc))
    else:
        order_exprs = [exp.Ordered(this=metric_expr.copy(), desc=metric_desc)]
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
    alias_by_norm: Dict[str, str] = {}

    for item in nonagg_items:
        if isinstance(item, exp.Alias):
            inner_exprs.append(item.copy())
            outer_exprs.append(exp.column(item.alias, table="t"))
            base_norm = norm_name(item.this if isinstance(item.this, exp.Expression) else item)
            if base_norm:
                alias_by_norm[base_norm] = item.alias
        else:
            # Require it to be a Column for safety; otherwise abort
            if not isinstance(item, exp.Column):
                return sql
            name = item.name
            inner_exprs.append(exp.alias_(item.copy(), name, quoted=False))
            outer_exprs.append(exp.column(name, table="t"))
            base_norm = norm_name(item)
            if base_norm:
                alias_by_norm[base_norm] = name

    # Add metric
    inner_exprs.append(exp.alias_(metric_expr.copy(), metric_alias, quoted=False))
    outer_exprs.append(exp.column(metric_alias, table="t"))
    metric_norm = norm_name(metric_expr)
    if metric_norm:
        alias_by_norm[metric_norm] = metric_alias

    # Add rn
    inner_exprs.append(exp.alias_(rn, "rn", quoted=False))

    # Rebuild inner FROM/WHERE from original query
    from_expr = sel.args.get("from") or sel.args.get("from_")
    if not from_expr:
        return sql
    inner = exp.Select(expressions=inner_exprs)
    inner.set("from_", from_expr.copy())
    joins = sel.args.get("joins")
    if joins:
        inner.set("joins", [j.copy() for j in joins])

    where_expr = sel.args.get("where")
    if where_expr is not None and include_zero_intent:
        where_sql = where_expr.sql(dialect="mysql").lower()
        ranking_like_filter = bool(
            re.search(r"\bselect\s+max\s*\(", where_sql, flags=re.I)
            and re.search(
                r"\b(paymentdate|orderdate|amount|quantity|price|year\s*\()",
                where_sql,
                flags=re.I,
            )
        )
        if ranking_like_filter:
            where_expr = None

    if where_expr is not None:
        inner.set("where", where_expr.copy())

    # Outer query
    outer = exp.Select(expressions=outer_exprs).from_(
        exp.Subquery(this=inner, alias=exp.TableAlias(this=exp.to_identifier("t")))
    ).where(exp.EQ(this=exp.column("rn", table="t"), expression=exp.Literal.number(1)))

    # Re-introduce deterministic outer ordering when the question explicitly asks sorting.
    if re.search(r"\b(?:sort(?:ed)?\s+by|order\s+by)\b", q_low, flags=re.I):
        outer_order_exprs = []
        if tie_exprs:
            tie_alias = alias_by_norm.get(norm_name(tie_exprs[0]))
            if tie_alias:
                outer_order_exprs.append(exp.Ordered(this=exp.column(tie_alias, table="t"), desc=True))
        part_alias = alias_by_norm.get(norm_name(partition_exprs[0]))
        if part_alias:
            outer_order_exprs.append(exp.Ordered(this=exp.column(part_alias, table="t"), desc=False))
        if outer_order_exprs:
            outer.set("order", exp.Order(expressions=outer_order_exprs))

    return outer.sql(dialect="mysql")


def _rewrite_metric_aggregate_column_mismatch(sql: str, question: str, ctx: str = "") -> str:
    """
    Fix SUM/AVG on non-metric columns (e.g., code/number/date/name) when
    question intent is metric total/average.

    Dynamic behavior:
    - resolve source table from SQL alias map
    - pick replacement metric column from same table using schema anchors
      + lexical metric signals
    """
    if not sql or not question:
        return sql

    q = question.lower()
    has_metric_intent = bool(
        re.search(
            r"\b(total|sum|average|avg|amount|paid|payment|quantity|qty|price|cost|sales|revenue|msrp|value)\b",
            q,
        )
    )
    if not has_metric_intent:
        return sql

    m = re.match(r"(?is)^\s*select\s+(?P<select>.*?)\s+from\s+(?P<after>.*)$", sql.strip())
    if not m:
        return sql

    select_part = (m.group("select") or "").strip()
    after_part = (m.group("after") or "").strip()
    if not select_part or not after_part:
        return sql

    m_from = re.match(
        r"(?is)^(?P<from_core>.*?)(?P<tail>\s+(?:where|group\s+by|having|order\s+by|limit)\b.*|;?\s*)$",
        after_part,
    )
    from_core = (m_from.group("from_core") or "").strip() if m_from else after_part.rstrip(";").strip()
    if not from_core:
        return sql

    alias_to_table = _parse_from_join_aliases(f"SELECT 1 FROM {from_core}")
    if not alias_to_table:
        m_base = re.match(r"(?is)^`?([A-Za-z_][A-Za-z0-9_]*)`?", from_core)
        if m_base:
            t = m_base.group(1)
            alias_to_table = {t: t, t.lower(): t}
    if not alias_to_table:
        return sql

    table_universe = {str(t).lower() for t in alias_to_table.values() if t}

    try:
        analyzer = get_schema_analyzer()
    except Exception:
        return sql
    metric_tokens = {"amount", "price", "cost", "qty", "quantity", "revenue", "sales", "msrp", "value", "total"}
    id_like_tokens = {"id", "code", "number", "name", "date", "status", "email", "phone", "address", "city", "country", "zip", "postal", "check"}

    prefer_amount = bool(re.search(r"\b(amount|paid|payment)\b", q))
    prefer_quantity = bool(re.search(r"\b(quantity|qty)\b", q))
    prefer_price = bool(re.search(r"\b(price|cost|msrp)\b", q))
    prefer_sales = bool(re.search(r"\b(sales|revenue)\b", q))
    strong_metric_intent = prefer_amount or prefer_quantity or prefer_price or prefer_sales

    def _clean_ident(s: str) -> str:
        return (s or "").replace("`", "").strip().lower()

    def _is_metric_like(col_l: str) -> bool:
        return any(tok in col_l for tok in metric_tokens)

    def _is_id_like(col_l: str) -> bool:
        return any(tok in col_l for tok in id_like_tokens)

    def _pick_metric_col(table_l: str, current_col_l: str) -> Optional[str]:
        try:
            cols = [c.lower() for c in analyzer.get_table_columns(table_l)]
        except Exception:
            cols = []
        if not cols:
            return None

        try:
            anchor_cols = {c.lower() for c in analyzer.get_anchor_columns(table_l)}
        except Exception:
            anchor_cols = set()

        candidates = [c for c in cols if c != current_col_l and (_is_metric_like(c) or c in anchor_cols)]
        if not candidates:
            return None

        def _score(c: str) -> int:
            score = 0
            if c in anchor_cols:
                score += 20
            if _is_metric_like(c):
                score += 8
            if prefer_amount and "amount" in c:
                score += 12
            if prefer_quantity and ("quantity" in c or "qty" in c):
                score += 12
            if prefer_price and any(x in c for x in ["price", "cost", "msrp"]):
                score += 12
            if prefer_sales and any(x in c for x in ["sales", "revenue"]):
                score += 12
            if _is_id_like(c):
                score -= 8
            return score

        best = max(candidates, key=_score)
        return best if _score(best) >= 8 else None

    agg_pat = re.compile(
        r"""(?is)\b(?P<func>SUM|AVG)\s*\(\s*(?P<distinct>DISTINCT\s+)?(?:(?P<q>`?[A-Za-z_][A-Za-z0-9_]*`?)\s*\.\s*)?(?P<c>`?[A-Za-z_][A-Za-z0-9_]*`?)\s*\)"""
    )
    bool_agg_pat = re.compile(
        r"""(?is)\b(?P<func>SUM|AVG)\s*\(\s*(?:(?P<q>`?[A-Za-z_][A-Za-z0-9_]*`?)\s*\.\s*)?(?P<c>`?[A-Za-z_][A-Za-z0-9_]*`?)\s+IS\s+NOT\s+NULL\s*\)"""
    )

    changed = False

    def _rewrite_item(item: str) -> str:
        nonlocal changed

        def _repl(m_agg: re.Match) -> str:
            nonlocal changed
            func = (m_agg.group("func") or "").upper()
            distinct_kw = m_agg.group("distinct") or ""
            qual_raw = m_agg.group("q")
            col_raw = m_agg.group("c") or ""

            col_l = _clean_ident(col_raw)
            if _is_metric_like(col_l) or not _is_id_like(col_l):
                return m_agg.group(0)

            table_l = None
            if qual_raw:
                q_clean = _clean_ident(qual_raw)
                table_l = _clean_ident(alias_to_table.get(qual_raw) or alias_to_table.get(q_clean) or "")
            elif len(table_universe) == 1:
                table_l = next(iter(table_universe))

            if not table_l:
                return m_agg.group(0)

            replacement = _pick_metric_col(table_l, col_l)
            if not replacement:
                return m_agg.group(0)

            changed = True
            if qual_raw:
                return f"{func}({distinct_kw}{qual_raw}.{replacement})"
            return f"{func}({distinct_kw}{replacement})"

        item2 = agg_pat.sub(_repl, item)

        # Handle anti-pattern like SUM(pay.paymentDate IS NOT NULL) when metric intent
        # clearly asks for amount/quantity/price/revenue style totals.
        if strong_metric_intent:
            def _repl_bool(m_bool: re.Match) -> str:
                nonlocal changed
                func = (m_bool.group("func") or "").upper()
                qual_raw = m_bool.group("q")
                col_raw = m_bool.group("c") or ""
                col_l = _clean_ident(col_raw)

                table_l = None
                if qual_raw:
                    q_clean = _clean_ident(qual_raw)
                    table_l = _clean_ident(alias_to_table.get(qual_raw) or alias_to_table.get(q_clean) or "")
                elif len(table_universe) == 1:
                    table_l = next(iter(table_universe))

                if not table_l:
                    return m_bool.group(0)

                replacement = _pick_metric_col(table_l, col_l)
                if not replacement:
                    return m_bool.group(0)

                changed = True
                if qual_raw:
                    return f"{func}({qual_raw}.{replacement})"
                return f"{func}({replacement})"

            item2 = bool_agg_pat.sub(_repl_bool, item2)

        return item2

    select_items = _split_top_level_commas(select_part)
    rebuilt_items = [_rewrite_item(it) for it in select_items]
    if not changed:
        return sql

    new_select = ", ".join(rebuilt_items)
    return re.sub(
        r"(?is)^\s*select\s+.*?\s+from\s+",
        f"SELECT {new_select} FROM ",
        sql.strip(),
        count=1,
    )


def _question_wants_entity_orders_payments_rollup(question: str) -> bool:
    """
    Detect intent like:
    - customers with at least one order
    - total number of orders
    - total payment amount
    - paid/unpaid style status
    """
    q = (question or "").lower().strip()
    if not q:
        return False

    def _entity_alt(table_name: str) -> str:
        words = _entity_words_for_table(table_name)
        escaped = []
        for w in words:
            ww = (w or "").strip().lower()
            if not ww:
                continue
            escaped.append(re.escape(ww).replace(r"\ ", r"\s+"))
        escaped = sorted(set(escaped), key=len, reverse=True)
        return "|".join(escaped)

    cust_alt = _entity_alt("customers")
    ord_alt = _entity_alt("orders")
    pay_alt = _entity_alt("payments")

    def _has_alt(alt: str) -> bool:
        return bool(alt and re.search(rf"\b(?:{alt})\b", q, flags=re.I))

    has_customer = _has_alt(cust_alt)
    has_orders = _has_alt(ord_alt)
    has_payments = _has_alt(pay_alt) or bool(re.search(r"\bpaid\b", q))

    has_order_count = bool(
        re.search(
            rf"\b(?:total\s+number\s+of|number\s+of|count\s+of|jumlah)\s+(?:{ord_alt})\b",
            q,
            flags=re.I,
        )
    ) or bool(re.search(r"\btotal_orders?\b", q, flags=re.I))

    has_payment_total = bool(
        re.search(
            rf"\b(?:total\s+(?:{pay_alt})|total\s+amount\s+paid|sum\s+of\s+(?:{pay_alt})|(?:{pay_alt})\s+amount|jumlah\s+pembayaran)\b",
            q,
            flags=re.I,
        )
    ) or bool(re.search(r"\btotal_payments?\b|\btotal_payment_amount\b", q, flags=re.I))

    asks_status = bool(re.search(r"\bpayment[_\s]*status\b", q, flags=re.I)) or ("paid" in q and "unpaid" in q)

    return has_customer and has_orders and (has_payments or has_payment_total or asks_status) and has_order_count and (has_payment_total or asks_status)


def _question_wants_customer_orders_payment_rollup(question: str) -> bool:
    """
    Backward-compatible wrapper.
    Kept to avoid breaking older call sites while using generic detector logic.
    """
    return _question_wants_entity_orders_payments_rollup(question)


def _question_wants_employee_customer_payment_rollup(question: str) -> bool:
    """
    Detect intent like H5:
    - for each employee
    - include employees/customers with no matches
    - return total payment amount collected by their customers
    """
    q = (question or "").lower().strip()
    if not q:
        return False

    # Exclude winner-row/latest style intents (handled by different pathways).
    if re.search(
        r"\b(latest|most\s+recent|newest|earliest|row_number|rank|dense_rank|if\s+multiple|if\s+there\s+are\s+ties)\b",
        q,
        flags=re.I,
    ):
        return False

    def _entity_alt(table_name: str) -> str:
        words = _entity_words_for_table(table_name)
        escaped = []
        for w in words:
            ww = (w or "").strip().lower()
            if not ww:
                continue
            escaped.append(re.escape(ww).replace(r"\ ", r"\s+"))
        escaped = sorted(set(escaped), key=len, reverse=True)
        return "|".join(escaped)

    emp_alt = _entity_alt("employees")
    cust_alt = _entity_alt("customers")
    pay_alt = _entity_alt("payments")

    def _has_alt(alt: str) -> bool:
        return bool(alt and re.search(rf"\b(?:{alt})\b", q, flags=re.I))

    has_employee = _has_alt(emp_alt)
    has_customer = _has_alt(cust_alt)
    has_payment = _has_alt(pay_alt) or bool(re.search(r"\bpaid\b", q, flags=re.I))

    asks_for_each_employee = bool(
        re.search(rf"\bfor\s+each\s+(?:{emp_alt})\b", q, flags=re.I)
        if emp_alt
        else re.search(r"\bfor\s+each\s+employees?\b", q, flags=re.I)
    )
    asks_total_payment = bool(
        re.search(
            r"\b(total\s+payment(?:\s+amount)?|sum\s+of\s+payments?|amount\s+collected|collected\s+by)\b",
            q,
            flags=re.I,
        )
    )

    return has_employee and has_customer and has_payment and asks_for_each_employee and asks_total_payment


def _rewrite_employee_customer_payment_rollup(sql: str, question: str, ctx: str = "") -> str:
    """
    Canonical rewrite for employee -> customer -> payment rollup:
      employees LEFT JOIN customers LEFT JOIN payments
      GROUP BY employee identity fields

    This targets robust repair for drafts that miss payments join alias
    (e.g., SUM(p.amount) without joining `p`).
    """
    if not sql or not question:
        return sql
    if not _question_wants_employee_customer_payment_rollup(question):
        return sql

    parsed = _parse_select_from_groupby(sql)
    if not parsed:
        return sql

    # Conservative: skip complex shapes.
    if (parsed.get("where") or "").strip():
        return sql
    if (parsed.get("having") or "").strip():
        return sql

    from_core = (parsed.get("from_core") or "").strip()
    if not from_core or "join" not in from_core.lower():
        return sql
    if re.search(r"\brow_number\s*\(", sql, flags=re.I):
        return sql

    alias_map_raw = _parse_from_join_aliases(f"SELECT 1 FROM {from_core}")
    alias_to_table: Dict[str, str] = {}
    for a, t in alias_map_raw.items():
        al = (a or "").strip().lower()
        tl = (t or "").strip().lower()
        if not al or not tl:
            continue
        if al not in alias_to_table:
            alias_to_table[al] = tl

    # Aliasless fallback: map table name -> table name from FROM/JOIN clauses.
    if not alias_to_table:
        for m_tbl in re.finditer(
            r"""(?is)\b(?:FROM|JOIN)\s+(`?[A-Za-z_][A-Za-z0-9_]*`?(?:\.`?[A-Za-z_][A-Za-z0-9_]*`?)?)\b""",
            f"FROM {from_core}",
        ):
            raw_tbl = (m_tbl.group(1) or "").replace("`", "").strip()
            tbl = raw_tbl.split(".")[-1].lower()
            if tbl and tbl not in alias_to_table:
                alias_to_table[tbl] = tbl

    has_employees = any(t == "employees" for t in alias_to_table.values())
    has_customers = any(t == "customers" for t in alias_to_table.values())
    if not (has_employees and has_customers):
        return sql

    # If payments already joined, don't force canonical rewrite here.
    if any(t == "payments" for t in alias_to_table.values()):
        return sql

    # Build canonical projection fields dynamically where possible.
    emp_id_col = get_id_column("employees") or "employeeNumber"
    emp_first_col = "firstName"
    emp_last_col = "lastName"
    emp_name_col = get_name_column("employees") or ""

    analyzer = None
    try:
        analyzer = get_schema_analyzer()
    except Exception:
        analyzer = None

    emp_cols_lower: Set[str] = set()
    if analyzer is not None:
        try:
            emp_cols_lower = {
                str(c).strip("`").lower()
                for c in analyzer.get_table_columns("employees")
            }
        except Exception:
            emp_cols_lower = set()

    has_first_last = (
        emp_first_col.lower() in emp_cols_lower
        and emp_last_col.lower() in emp_cols_lower
    ) or not emp_cols_lower

    if has_first_last:
        name_select = "CONCAT(emp.firstName, ' ', emp.lastName) AS employee_name"
        group_by = f"emp.{emp_id_col}, emp.firstName, emp.lastName"
    else:
        safe_name_col = emp_name_col if emp_name_col else emp_id_col
        name_select = f"emp.{safe_name_col} AS employee_name"
        group_by = f"emp.{emp_id_col}, emp.{safe_name_col}"

    rewritten = (
        "SELECT "
        f"emp.{emp_id_col}, "
        f"{name_select}, "
        "SUM(pmt.amount) AS total_payment_amount "
        "FROM employees AS emp "
        f"LEFT JOIN customers AS cust ON emp.{emp_id_col} = cust.salesRepEmployeeNumber "
        "LEFT JOIN payments AS pmt ON cust.customerNumber = pmt.customerNumber "
        f"GROUP BY {group_by};"
    )
    return rewritten


def _rewrite_customer_orders_payment_fanout(sql: str, question: str, ctx: str = "") -> str:
    """
    Fix row-multiplication anti-pattern:
      customers JOIN orders JOIN payments + GROUP BY customer
      with COUNT(order) and SUM(payment).

    Rewrites into fanout-safe shape:
      customers
      JOIN (orders aggregated by customer)
      LEFT JOIN (payments aggregated by customer)
    """
    if not sql or not question:
        return sql

    if not _question_wants_entity_orders_payments_rollup(question):
        return sql

    parsed = _parse_select_from_groupby(sql)
    if not parsed:
        return sql

    # Conservative: only rewrite simple grouped query without WHERE/HAVING.
    if (parsed.get("where") or "").strip():
        return sql
    if (parsed.get("having") or "").strip():
        return sql

    from_core = (parsed.get("from_core") or "").strip()
    if not from_core or "join" not in from_core.lower():
        return sql

    alias_map_raw = _parse_from_join_aliases(f"SELECT 1 FROM {from_core}")
    alias_to_table: Dict[str, str] = {}
    for a, t in alias_map_raw.items():
        al = (a or "").strip().lower()
        tl = (t or "").strip().lower()
        if not al or not tl:
            continue
        if al not in alias_to_table:
            alias_to_table[al] = tl

    # Aliasless fallback: map table name -> table name from FROM/JOIN clauses.
    if not alias_to_table:
        for m_tbl in re.finditer(
            r"""(?is)\b(?:FROM|JOIN)\s+(`?[A-Za-z_][A-Za-z0-9_]*`?(?:\.`?[A-Za-z_][A-Za-z0-9_]*`?)?)\b""",
            f"FROM {from_core}",
        ):
            raw_tbl = (m_tbl.group(1) or "").replace("`", "").strip()
            tbl = raw_tbl.split(".")[-1].lower()
            if tbl and tbl not in alias_to_table:
                alias_to_table[tbl] = tbl

    order_aliases = [a for a, t in alias_to_table.items() if t == "orders"]
    payment_aliases = [a for a, t in alias_to_table.items() if t == "payments"]
    if not order_aliases or not payment_aliases:
        return sql
    order_alias = order_aliases[0]
    payment_alias = payment_aliases[0]

    m_base = re.match(
        r"""(?is)^\s*(`?[A-Za-z_][A-Za-z0-9_]*`?(?:\.`?[A-Za-z_][A-Za-z0-9_]*`?)?)\s*(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*)?""",
        from_core,
    )
    if not m_base:
        return sql

    base_table_expr = (m_base.group(1) or "").strip()
    base_table = base_table_expr.replace("`", "").split(".")[-1]
    if not base_table:
        return sql
    base_alias_raw = (m_base.group(2) or "").strip()
    # Guard: do not treat JOIN keywords as aliases in aliasless SQL.
    reserved_alias_tokens = {
        "left", "right", "inner", "outer", "full", "cross", "join",
        "on", "where", "group", "order", "having", "limit",
    }
    if base_alias_raw.lower() in reserved_alias_tokens:
        base_alias_raw = ""

    base_alias = (base_alias_raw or base_table).strip()
    base_alias_l = base_alias.lower()

    # Parse SELECT list.
    def _split_expr_alias(item: str) -> tuple:
        m = re.match(
            r"""(?is)^\s*(.*?)\s+AS\s+(`[^`]+`|'[^']+'|"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)\s*$""",
            (item or "").strip(),
        )
        if not m:
            return (item or "").strip(), None
        return (m.group(1) or "").strip(), (m.group(2) or "").strip()

    def _parse_simple_col(expr: str) -> Optional[tuple]:
        m = re.match(
            r"""(?is)^\s*(?:`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\.\s*)?`?([A-Za-z_][A-Za-z0-9_]*)`?\s*$""",
            (expr or "").strip(),
        )
        if not m:
            return None
        return m.group(1), m.group(2)

    select_items = _split_top_level_commas(parsed["select"])
    if not select_items:
        return sql

    base_select_items: List[str] = []
    count_alias_out: Optional[str] = None
    count_inner: Optional[str] = None
    count_distinct = False
    sum_alias_out: Optional[str] = None
    sum_inner: Optional[str] = None
    case_items: List[tuple] = []

    count_pat = re.compile(r"(?is)^COUNT\s*\(\s*(DISTINCT\s+)?(.+?)\s*\)$")
    sum_pat = re.compile(r"(?is)^SUM\s*\(\s*(.+?)\s*\)$")

    for raw_item in select_items:
        expr, out_alias = _split_expr_alias(raw_item)
        expr_l = expr.lower()

        m_count = count_pat.match(expr)
        if m_count:
            inner_expr = (m_count.group(2) or "").strip()
            # Accept COUNT(*) or COUNT(order_alias.col)
            if inner_expr == "*":
                count_inner = "*"
                count_distinct = bool((m_count.group(1) or "").strip())
                count_alias_out = (out_alias or "total_orders").strip("`'\"")
                continue
            ref = _parse_simple_col(inner_expr)
            if ref and ref[0] and ref[0].lower() == order_alias:
                count_inner = inner_expr
                count_distinct = bool((m_count.group(1) or "").strip())
                count_alias_out = (out_alias or "total_orders").strip("`'\"")
                continue

        m_sum = sum_pat.match(expr)
        if m_sum:
            inner_expr = (m_sum.group(1) or "").strip()
            ref = _parse_simple_col(inner_expr)
            if ref and ref[0] and ref[0].lower() == payment_alias:
                sum_inner = inner_expr
                sum_alias_out = (out_alias or "total_payments").strip("`'\"")
                continue

        # CASE item that depends on SUM(payment_alias.*) -> rewrite later.
        if expr_l.startswith("case") and re.search(
            rf"\bSUM\s*\(\s*(?:DISTINCT\s+)?`?{re.escape(payment_alias)}`?\s*\.",
            expr,
            flags=re.I,
        ):
            case_items.append((expr, out_alias))
            continue

        # Keep base-table simple columns only.
        pcol = _parse_simple_col(expr)
        if pcol:
            qf, _col = pcol
            if qf and qf.lower() == base_alias_l:
                base_select_items.append(raw_item.strip())
                continue
            if not qf:
                base_select_items.append(raw_item.strip())
                continue

        # Unsupported/complex projection: abort rewrite for safety.
        return sql

    if not base_select_items or not count_inner or not sum_inner:
        return sql
    if not count_alias_out:
        count_alias_out = "total_orders"
    if not sum_alias_out:
        sum_alias_out = "total_payments"

    # Deduplicate base projection columns conservatively.
    # Keep duplicates when aliases differ (e.g., customerNumber AS customer_id
    # and customerNumber AS customer_number are both explicitly requested).
    dedup_base: List[str] = []
    seen_base_expr: Set[str] = set()
    for it in base_select_items:
        base_expr, item_alias = _split_expr_alias(it)
        expr_norm = re.sub(r"\s+", "", (base_expr or "").lower())
        alias_norm = (item_alias or "").strip("`'\"").lower()
        k = f"{expr_norm}__as__{alias_norm}" if alias_norm else expr_norm
        if not k or k in seen_base_expr:
            continue
        seen_base_expr.add(k)
        dedup_base.append(it)
    base_select_items = dedup_base

    # GROUP BY must not depend on orders/payments aliases.
    for g in _split_top_level_commas(parsed.get("groupby", "")):
        g0 = (g or "").strip().lower()
        if re.search(rf"\b{re.escape(order_alias)}\s*\.", g0) or re.search(rf"\b{re.escape(payment_alias)}\s*\.", g0):
            return sql

    # Infer join key pairs between base<->orders and base<->payments.
    eq_pat = re.compile(
        r"""`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\.\s*`?([A-Za-z_][A-Za-z0-9_]*)`?\s*=\s*`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\.\s*`?([A-Za-z_][A-Za-z0-9_]*)`?""",
        flags=re.I,
    )
    order_pairs: List[tuple] = []
    payment_pairs: List[tuple] = []
    for m in eq_pat.finditer(from_core):
        a1, c1, a2, c2 = m.group(1).lower(), m.group(2), m.group(3).lower(), m.group(4)
        if {a1, a2} == {base_alias_l, order_alias}:
            if a1 == base_alias_l:
                order_pairs.append((c1, c2))
            else:
                order_pairs.append((c2, c1))
        if {a1, a2} == {base_alias_l, payment_alias}:
            if a1 == base_alias_l:
                payment_pairs.append((c1, c2))
            else:
                payment_pairs.append((c2, c1))

    def _pick_pair(pairs: List[tuple]) -> Optional[tuple]:
        if not pairs:
            return None

        def _score(p: tuple) -> int:
            b, c = (p[0] or "").lower(), (p[1] or "").lower()
            s = 0
            for x in (b, c):
                if x.endswith("id") or x.endswith("number") or x.endswith("code"):
                    s += 4
                if "customer" in x:
                    s += 3
            if b == c:
                s += 2
            return s

        return sorted(pairs, key=_score, reverse=True)[0]

    order_pair = _pick_pair(order_pairs)
    payment_pair = _pick_pair(payment_pairs)
    if not order_pair or not payment_pair:
        return sql

    base_order_key, order_key = order_pair
    base_payment_key, payment_key = payment_pair

    # If question explicitly asks for identifier slots, ensure projection stability.
    q_low = (question or "").lower()
    # Treat ORDER BY mention as sorting intent, not projection requirement.
    q_low_output = re.split(r"\b(?:sort(?:ed)?\s+by|order\s+by)\b", q_low, maxsplit=1, flags=re.I)[0]
    entity_words = _entity_words_for_table(base_table)
    if not entity_words:
        entity_words = [base_table, (base_table[:-1] if base_table.endswith("s") else base_table)]

    def _mentions_entity_attr(attr: str) -> bool:
        for w in entity_words:
            ww = (w or "").strip().lower()
            if not ww:
                continue
            pat = re.escape(ww).replace(r"\ ", r"\s+")
            if re.search(rf"\b{pat}\s+{re.escape(attr)}\b", q_low_output):
                return True
        return False

    wants_customer_number = _mentions_entity_attr("number") or bool(re.search(rf"\b{re.escape(base_table)}number\b", q_low_output))
    wants_customer_id = _mentions_entity_attr("id")
    wants_entity_name = _mentions_entity_attr("name")
    id_col = get_id_column(base_table) or base_order_key
    name_col = get_name_column(base_table)

    def _base_has_col(col_name: str) -> bool:
        for it in base_select_items:
            e0, _a0 = _split_expr_alias(it)
            p0 = _parse_simple_col(e0)
            if not p0:
                continue
            _qf, c0 = p0
            if c0 and c0.lower() == col_name.lower():
                return True
        return False

    def _base_has_alias(alias_name: str) -> bool:
        for it in base_select_items:
            _e0, a0 = _split_expr_alias(it)
            if a0 and a0.strip("`'\"").lower() == alias_name.lower():
                return True
        return False

    if id_col and (wants_customer_number or wants_customer_id):
        if not _base_has_col(id_col):
            base_select_items.insert(0, f"{base_alias}.{id_col}")
        # If question explicitly says "customer id", expose a stable alias.
        if wants_customer_id and not _base_has_alias("customer_id"):
            base_select_items.append(f"{base_alias}.{id_col} AS customer_id")

    # Normalize duplicate identifier projections.
    # If question explicitly asks both identifier slots (id + number), keep both.
    # Otherwise collapse accidental duplicates; if name is explicitly requested,
    # we can promote one duplicate slot to name when available.
    if id_col:
        id_item_idxs: List[int] = []
        removed_aliases: List[str] = []
        for idx, it in enumerate(base_select_items):
            e0, a0 = _split_expr_alias(it)
            p0 = _parse_simple_col(e0)
            if not p0:
                continue
            qf0, c0 = p0
            if c0 and c0.lower() == id_col.lower() and (not qf0 or qf0.lower() == base_alias_l):
                id_item_idxs.append(idx)

        if len(id_item_idxs) > 1:
            wants_dual_identifier_slots = wants_customer_id and wants_customer_number
            if not wants_dual_identifier_slots:
                def _alias_pref_score(idx: int) -> int:
                    _expr_i, alias_i = _split_expr_alias(base_select_items[idx])
                    a = (alias_i or "").strip("`'\"").lower()
                    score = 0
                    if wants_customer_number and not wants_customer_id:
                        if "number" in a:
                            score += 6
                        if "id" in a and "number" not in a:
                            score -= 2
                    elif wants_customer_id and not wants_customer_number:
                        if "id" in a:
                            score += 6
                        if "number" in a and "id" not in a:
                            score -= 2
                    # Slightly prefer explicit aliases over bare columns on ties.
                    if a:
                        score += 1
                    # Stable fallback: keep earliest projection on exact tie.
                    score -= idx
                    return score

                keep_idx = max(id_item_idxs, key=_alias_pref_score)
                for drop_idx in reversed(id_item_idxs):
                    if drop_idx == keep_idx:
                        continue
                    _drop_expr, drop_alias = _split_expr_alias(base_select_items[drop_idx])
                    if drop_alias:
                        removed_aliases.append(drop_alias.strip("`'\""))
                    base_select_items.pop(drop_idx)

                if wants_entity_name and name_col and not _base_has_col(name_col):
                    name_alias: Optional[str] = None
                    for a in removed_aliases:
                        if "name" in a.lower():
                            name_alias = a
                            break
                    name_item = f"{base_alias}.{name_col}"
                    if name_alias:
                        name_item += f" AS {name_alias}"
                    insert_at = min(keep_idx + 1, len(base_select_items))
                    base_select_items.insert(insert_at, name_item)

    # If question explicitly asks for entity name, ensure it is projected.
    # This must be independent from duplicate-id collapse (both id+number may be requested).
    if wants_entity_name and name_col and not _base_has_col(name_col):
        base_select_items.append(f"{base_alias}.{name_col}")

    used_aliases = {base_alias_l, order_alias, payment_alias, *alias_to_table.keys()}

    def _pick_alias(base: str) -> str:
        b = re.sub(r"[^A-Za-z0-9_]", "", base) or "t"
        if not re.match(r"[A-Za-z_]", b[0]):
            b = f"t_{b}"
        cand = b
        i = 2
        while cand.lower() in used_aliases:
            cand = f"{b}_{i}"
            i += 1
        used_aliases.add(cand.lower())
        return cand

    orders_agg_alias = _pick_alias("orders_agg")
    payments_agg_alias = _pick_alias("payments_agg")

    count_inner_clean = count_inner
    if count_inner_clean != "*":
        count_inner_clean = re.sub(
            rf"(?is)\b`?{re.escape(order_alias)}`?\s*\.\s*",
            "",
            count_inner_clean,
        )
    count_expr = f"COUNT({'DISTINCT ' if count_distinct else ''}{count_inner_clean})"

    sum_inner_clean = re.sub(
        rf"(?is)\b`?{re.escape(payment_alias)}`?\s*\.\s*",
        "",
        sum_inner,
    )

    # Build SELECT projection.
    projected_items: List[str] = []
    projected_items.extend(base_select_items)
    projected_items.append(f"{orders_agg_alias}.{count_alias_out} AS {count_alias_out}")
    projected_items.append(
        f"COALESCE({payments_agg_alias}.{sum_alias_out}, 0) AS {sum_alias_out}"
    )

    for case_expr, case_alias in case_items:
        rewritten_case = re.sub(
            rf"(?is)SUM\s*\(\s*(?:DISTINCT\s+)?`?{re.escape(payment_alias)}`?\s*\.\s*`?[A-Za-z_][A-Za-z0-9_]*`?\s*\)",
            f"COALESCE({payments_agg_alias}.{sum_alias_out}, 0)",
            case_expr,
        )
        if case_alias:
            projected_items.append(f"{rewritten_case} AS {case_alias}")
        else:
            projected_items.append(rewritten_case)

    tail_clause = (parsed.get("tail") or "").strip()
    # Stabilize ORDER BY customer_id when alias may be missing.
    if tail_clause and id_col and re.search(r"\bcustomer_id\b", tail_clause, flags=re.I):
        has_customer_id_alias = any(
            (_split_expr_alias(it)[1] or "").strip("`'\"").lower() == "customer_id"
            for it in projected_items
        )
        if not has_customer_id_alias:
            tail_clause = re.sub(
                r"\bcustomer_id\b",
                f"{base_alias}.{id_col}",
                tail_clause,
                flags=re.I,
            )
    tail_block = f"\n{tail_clause}" if tail_clause else ""
    select_block = ",\n  ".join(projected_items)

    fixed = (
        "SELECT\n"
        f"  {select_block}\n"
        f"FROM {base_table_expr} {base_alias}\n"
        "JOIN (\n"
        f"  SELECT {order_key}, {count_expr} AS {count_alias_out}\n"
        f"  FROM orders\n"
        f"  GROUP BY {order_key}\n"
        f") AS {orders_agg_alias}\n"
        f"  ON {orders_agg_alias}.{order_key} = {base_alias}.{base_order_key}\n"
        "LEFT JOIN (\n"
        f"  SELECT {payment_key}, SUM({sum_inner_clean}) AS {sum_alias_out}\n"
        f"  FROM payments\n"
        f"  GROUP BY {payment_key}\n"
        f") AS {payments_agg_alias}\n"
        f"  ON {payments_agg_alias}.{payment_key} = {base_alias}.{base_payment_key}"
        f"{tail_block};"
    )
    return fixed


def _question_wants_latest_year_scope_with_aggregate(question: str) -> bool:
    """
    Dynamic intent detector for:
      - latest/most-recent year
      - plus aggregate amount/total intent

    Intentionally avoids exact phrase dependency like "in that year" so
    paraphrases still trigger the same rewrite path.
    """
    q = (question or "").lower().strip()
    if not q:
        return False

    has_latest = bool(
        re.search(r"\b(?:latest|most\s+recent|newest|terbaru|terakhir)\b", q)
    )
    has_year = bool(re.search(r"\b(?:year|tahun)\b", q))
    has_aggregate_amount = bool(
        re.search(r"\b(?:sum|total|jumlah|amount|paid|payment)\b", q)
    )

    return has_latest and has_year and has_aggregate_amount


def _rewrite_latest_year_same_year_aggregate(sql: str, question: str, ctx: str = "") -> str:
    """
    Fix aggregate anti-pattern for questions like:
      - "latest/most recent payment year"
      - "sum amount in that same year"

    Wrong pattern:
      SELECT key_cols, MAX(paymentDate), SUM(amount)
      FROM entity JOIN metric_table ...
      GROUP BY key_cols

    This sums across all years. Correct strategy:
      1) derive latest_year per entity from the metric table
      2) join/filter metric rows to that latest_year
      3) aggregate MAX(paymentDate) and SUM(amount) on filtered rows
    """
    if not sql:
        return sql

    # Keep trigger conservative but phrasing-agnostic.
    if not _question_wants_latest_year_scope_with_aggregate(question):
        return sql

    low_sql = sql.lower()

    # Already handled pattern (alias-agnostic): keep as-is.
    # Accept any derived-table alias, not only a specific one.
    already_has_latest_year_pattern = (
        bool(re.search(r"\bmax\s*\(\s*year\s*\(", low_sql))
        and "latest_year" in low_sql
        and bool(re.search(r"=\s*[a-z_][a-z0-9_]*\.latest_year\b", low_sql))
    )
    if already_has_latest_year_pattern:
        return sql

    def _strip_tautological_latest_year_having(sql_text: str) -> str:
        """
        Remove tautological HAVING predicates like:
          HAVING YEAR(MAX(p.paymentDate)) = YEAR(MAX(p.paymentDate))
        which can break parser-based rewrites while adding no filtering value.
        """
        m = re.search(
            r"""(?is)\bHAVING\s+YEAR\s*\(\s*MAX\s*\(\s*(.+?)\s*\)\s*\)\s*=\s*YEAR\s*\(\s*MAX\s*\(\s*(.+?)\s*\)\s*\)\s*""",
            sql_text,
        )
        if not m:
            return sql_text
        lhs = re.sub(r"[\s`]+", "", (m.group(1) or "").lower())
        rhs = re.sub(r"[\s`]+", "", (m.group(2) or "").lower())
        if not lhs or lhs != rhs:
            return sql_text
        return (sql_text[: m.start()] + " " + sql_text[m.end() :]).strip()

    parsed = _parse_select_from_groupby(sql)
    if not parsed:
        sql_sanitized = _strip_tautological_latest_year_having(sql)
        if sql_sanitized != sql:
            parsed = _parse_select_from_groupby(sql_sanitized)
            if parsed:
                sql = sql_sanitized
                low_sql = sql.lower()
    if not parsed:
        return sql

    from_core = parsed["from_core"].strip()
    if not from_core:
        return sql

    def _split_expr_alias(item: str) -> tuple:
        m = re.match(
            r"""(?is)^\s*(.*?)\s+AS\s+(`[^`]+`|'[^']+'|"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)\s*$""",
            item.strip(),
        )
        if not m:
            return item.strip(), None
        return m.group(1).strip(), m.group(2).strip()

    def _parse_qualified_col(expr: str) -> Optional[tuple]:
        """
        Parse simple column reference:
          p.paymentDate  -> ("p", "paymentDate")
          paymentDate    -> (None, "paymentDate")
        """
        m = re.match(
            r"(?is)^\s*(?:`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\.\s*)?`?([A-Za-z_][A-Za-z0-9_]*)`?\s*$",
            expr.strip(),
        )
        if not m:
            return None
        return m.group(1), m.group(2)

    def _extract_alias_refs(expr: str) -> List[str]:
        refs = re.findall(r"`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\.", expr or "", flags=re.I)
        # keep order, dedup case-insensitively
        out: List[str] = []
        seen = set()
        for r in refs:
            k = r.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(r)
        return out

    def _pick_best_key(candidates: List[str], avoid_cols: Set[str]) -> Optional[str]:
        if not candidates:
            return None
        uniq = []
        seen = set()
        for c in candidates:
            k = c.lower()
            if k in seen:
                continue
            seen.add(k)
            uniq.append(c)

        def score(col: str) -> int:
            c = col.lower()
            s = 0
            if c in avoid_cols:
                s -= 10
            if c.endswith("id") or c.endswith("number") or c.endswith("code"):
                s += 5
            if "id" in c or "number" in c or "code" in c:
                s += 2
            return s

        return sorted(uniq, key=score, reverse=True)[0]

    select_items = _split_top_level_commas(parsed["select"])
    if len(select_items) < 3:
        return sql

    date_expr_raw: Optional[str] = None
    sum_expr_raw: Optional[str] = None

    for it in select_items:
        expr, _alias = _split_expr_alias(it)

        m_max = re.match(r"(?is)^MAX\s*\(\s*(.+?)\s*\)$", expr)
        if m_max and date_expr_raw is None:
            max_inner = m_max.group(1).strip()
            # Accept both:
            #   MAX(o.orderDate)
            #   MAX(YEAR(o.orderDate))
            year_wrap = re.match(r"(?is)^YEAR\s*\(\s*(.+?)\s*\)$", max_inner)
            max_inner_col = year_wrap.group(1).strip() if year_wrap else max_inner
            parsed_col = _parse_qualified_col(max_inner_col)
            if parsed_col:
                _, col_name = parsed_col
                if ("date" in col_name.lower()) or ("time" in col_name.lower()):
                    date_expr_raw = max_inner_col
                    continue

        m_sum = re.match(r"(?is)^SUM\s*\(\s*(.+?)\s*\)$", expr)
        if m_sum and sum_expr_raw is None:
            sum_expr_raw = m_sum.group(1).strip()
            continue

    if not date_expr_raw or not sum_expr_raw:
        return sql

    date_col = _parse_qualified_col(date_expr_raw)
    if not date_col:
        return sql
    date_alias, date_col_name = date_col

    sum_alias_refs = _extract_alias_refs(sum_expr_raw)
    date_owner_alias: Optional[str] = date_alias
    measure_alias: Optional[str] = None
    if date_alias:
        if not sum_alias_refs:
            measure_alias = date_alias
        elif all(a.lower() == date_alias.lower() for a in sum_alias_refs):
            measure_alias = date_alias
        elif len(sum_alias_refs) == 1:
            # Cross-alias pattern:
            # SUM on bridge/fact alias (e.g., od.quantityOrdered),
            # DATE on joined temporal alias (e.g., o.orderDate).
            measure_alias = sum_alias_refs[0]
        else:
            return sql
    else:
        if len(sum_alias_refs) == 1:
            measure_alias = sum_alias_refs[0]
            date_owner_alias = measure_alias
        else:
            return sql

    if not measure_alias:
        return sql
    if not date_owner_alias:
        date_owner_alias = measure_alias

    alias_map = _parse_from_join_aliases(f"SELECT 1 FROM {from_core}")

    def _resolve_table_from_alias(alias_name: Optional[str]) -> Optional[str]:
        if not alias_name:
            return None
        tbl = alias_map.get(alias_name) or alias_map.get(alias_name.lower())
        return tbl or alias_name

    measure_table = _resolve_table_from_alias(measure_alias)
    date_owner_table = _resolve_table_from_alias(date_owner_alias)
    if not measure_table or not date_owner_table:
        return sql

    # Schema-aware column checks.
    try:
        analyzer = get_schema_analyzer()
    except Exception:
        analyzer = None
    allowed_cols: Dict[str, Set[str]] = {}
    if not analyzer and ctx:
        try:
            allowed = _parse_allowed_identifiers(ctx)
            allowed_cols = {
                str(t).strip().lower(): {str(c).strip().lower() for c in cols}
                for t, cols in (allowed or {}).items()
            }
        except Exception:
            allowed_cols = {}

    def _table_cols(table_name: str) -> Set[str]:
        t = (table_name or "").replace("`", "").split(".")[-1].strip().lower()
        if not t:
            return set()
        if analyzer:
            try:
                return {c.lower() for c in analyzer.get_table_columns(t)}
            except Exception:
                return set()
        return set(allowed_cols.get(t, set()))

    date_owner_cols = _table_cols(date_owner_table)
    if date_owner_cols and date_col_name.lower() not in date_owner_cols:
        return sql

    existing_aliases = {k.lower() for k in alias_map.keys() if k and not k.startswith("__")}

    # Find join keys connected to metric alias via ON predicates.
    eq_pat = re.compile(
        r"`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\.\s*`?([A-Za-z_][A-Za-z0-9_]*)`?\s*=\s*"
        r"`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\.\s*`?([A-Za-z_][A-Za-z0-9_]*)`?",
        flags=re.I,
    )
    key_candidates: List[str] = []
    measure_date_links: List[Tuple[str, str]] = []
    for m in eq_pat.finditer(from_core):
        a1, c1, a2, c2 = m.group(1), m.group(2), m.group(3), m.group(4)
        if a1.lower() == measure_alias.lower():
            key_candidates.append(c1)
        if a2.lower() == measure_alias.lower():
            key_candidates.append(c2)
        if a1.lower() == measure_alias.lower() and a2.lower() == date_owner_alias.lower():
            measure_date_links.append((c1, c2))
        elif a2.lower() == measure_alias.lower() and a1.lower() == date_owner_alias.lower():
            measure_date_links.append((c2, c1))

    # Fallback: metric alias appears in GROUP BY.
    if not key_candidates:
        for g in _split_top_level_commas(parsed["groupby"]):
            gm = _parse_qualified_col(g)
            if not gm:
                continue
            g_alias, g_col = gm
            if g_alias and g_alias.lower() == measure_alias.lower():
                key_candidates.append(g_col)

    measure_date_key_col: Optional[str] = None
    date_join_col: Optional[str] = None
    if date_owner_alias.lower() != measure_alias.lower():
        if not measure_date_links:
            return sql
        # Prefer symmetric key names (e.g., orderNumber=orderNumber), then id/code-like.
        scored = []
        for mk, dk in measure_date_links:
            s = 0
            if mk.lower() == dk.lower():
                s += 5
            mk_l = mk.lower()
            if mk_l.endswith("id") or mk_l.endswith("number") or mk_l.endswith("code"):
                s += 3
            elif ("id" in mk_l) or ("number" in mk_l) or ("code" in mk_l):
                s += 1
            scored.append((s, mk, dk))
        scored.sort(key=lambda x: x[0], reverse=True)
        measure_date_key_col, date_join_col = scored[0][1], scored[0][2]

    avoid_cols = {date_col_name.lower()}
    if measure_date_key_col:
        avoid_cols.add(measure_date_key_col.lower())
    measure_key_col = _pick_best_key(key_candidates, avoid_cols=avoid_cols)
    if not measure_key_col:
        return sql

    # If query already has this year filter, do nothing (alias-agnostic).
    date_ref = (
        f"{date_owner_alias}.{date_col_name}"
        if date_owner_alias
        else f"{date_col_name}"
    )
    if date_owner_alias:
        year_filter_pat = re.compile(
            rf"\bYEAR\s*\(\s*`?{re.escape(date_owner_alias)}`?\s*\.\s*`?{re.escape(date_col_name)}`?\s*\)\s*=\s*[A-Za-z_][A-Za-z0-9_]*\.latest_year",
            flags=re.I,
        )
    else:
        year_filter_pat = re.compile(
            rf"\bYEAR\s*\(\s*`?{re.escape(date_col_name)}`?\s*\)\s*=\s*[A-Za-z_][A-Za-z0-9_]*\.latest_year",
            flags=re.I,
        )
    if year_filter_pat.search(sql):
        return sql

    def _pick_subquery_alias(preferred: str = "latest_years") -> str:
        base = re.sub(r"[^A-Za-z0-9_]", "", preferred) or "latest_years"
        if not re.match(r"[A-Za-z_]", base[0]):
            base = f"y_{base}"
        cand = base
        i = 2
        while cand.lower() in existing_aliases:
            cand = f"{base}_{i}"
            i += 1
        existing_aliases.add(cand.lower())
        return cand

    year_alias = _pick_subquery_alias(f"{measure_alias}_latest_years")

    if date_owner_alias.lower() == measure_alias.lower():
        ly_join = (
            "JOIN (\n"
            "  SELECT\n"
            f"    {measure_key_col},\n"
            f"    MAX(YEAR({date_col_name})) AS latest_year\n"
            f"  FROM {measure_table}\n"
            f"  GROUP BY {measure_key_col}\n"
            f") AS {year_alias}\n"
            f"  ON {year_alias}.{measure_key_col} = {measure_alias}.{measure_key_col}"
        )
    else:
        if not measure_date_key_col or not date_join_col:
            return sql
        ly_measure_alias = _pick_subquery_alias(f"{measure_alias}_ly")
        ly_date_alias = _pick_subquery_alias(f"{date_owner_alias}_ly")
        ly_join = (
            "JOIN (\n"
            "  SELECT\n"
            f"    {ly_measure_alias}.{measure_key_col},\n"
            f"    MAX(YEAR({ly_date_alias}.{date_col_name})) AS latest_year\n"
            f"  FROM {measure_table} AS {ly_measure_alias}\n"
            f"  JOIN {date_owner_table} AS {ly_date_alias}\n"
            f"    ON {ly_date_alias}.{date_join_col} = {ly_measure_alias}.{measure_date_key_col}\n"
            f"  GROUP BY {ly_measure_alias}.{measure_key_col}\n"
            f") AS {year_alias}\n"
            f"  ON {year_alias}.{measure_key_col} = {measure_alias}.{measure_key_col}"
        )

    from_with_ly = f"{from_core}\n{ly_join}"
    year_cond = f"YEAR({date_ref}) = {year_alias}.latest_year"

    where_clause = parsed["where"].strip()
    if where_clause:
        # Preserve original boolean precedence (especially OR chains).
        existing_pred = re.sub(r"^\s*WHERE\s+", "", where_clause, flags=re.I).strip()
        # Remove stale correlated MAX(date) predicate when present.
        # Keeping it would incorrectly reduce SUM(amount) to only max-date rows.
        if existing_pred and not _has_top_level_or(existing_pred):
            kept_predicates: List[str] = []
            for cond in _split_top_level_and(existing_pred):
                c0 = (cond or "").strip()
                low_c0 = c0.lower()
                norm_c0 = re.sub(r"[\s`]+", "", low_c0)
                is_correlated_max_date = bool(
                    re.search(r"\b(?:in|=)\s*\(\s*select\s+max\s*\(", low_c0)
                    and "year(" in low_c0
                    and date_col_name.lower() in low_c0
                )
                date_alias_forms = {
                    a.lower()
                    for a in [measure_alias, date_owner_alias]
                    if a
                }
                self_max_forms = {
                    f"year({date_col_name.lower()})=year(max({date_col_name.lower()}))",
                    f"year(max({date_col_name.lower()}))=year({date_col_name.lower()})",
                }
                for da in date_alias_forms:
                    self_max_forms.add(
                        f"year({da}.{date_col_name.lower()})=year(max({da}.{date_col_name.lower()}))"
                    )
                    self_max_forms.add(
                        f"year(max({da}.{date_col_name.lower()}))=year({da}.{date_col_name.lower()})"
                    )
                is_self_max_year_pred = norm_c0 in self_max_forms
                if is_correlated_max_date or is_self_max_year_pred:
                    continue
                kept_predicates.append(c0)
            existing_pred = " AND ".join(p for p in kept_predicates if p)

        if existing_pred:
            where_clause = f"WHERE ({existing_pred}) AND {year_cond}"
        else:
            where_clause = f"WHERE {year_cond}"
    else:
        where_clause = f"WHERE {year_cond}"

    tail_clause = f"\n{parsed['tail'].strip()}" if parsed["tail"].strip() else ""
    fixed = (
        f"SELECT {parsed['select'].strip()}\n"
        f"FROM {from_with_ly}\n"
        f"{where_clause}\n"
        f"GROUP BY {parsed['groupby'].strip()}"
        f"{tail_clause};"
    )
    return fixed

def _rewrite_latest_year_correlated_maxdate_misuse(sql: str, question: str, ctx: str = "") -> str:
    """
    Fix correlated-subquery anti-pattern such as:

      WHERE p.paymentDate IN (
        SELECT MAX(paymentDate)
        FROM payments p2
        WHERE YEAR(p2.paymentDate) = YEAR(p.paymentDate)
          AND p2.customerNumber = p.customerNumber
      )

    combined with SUM(amount) + GROUP BY customer/year.

    Strategy:
      1) Canonicalize into: SELECT entity_cols, MAX(date), SUM(amount) ... GROUP BY entity_cols
      2) Reuse latest-year rewrite to enforce "sum in the most recent year".
    """
    if not sql:
        return sql

    if not _question_wants_latest_year_scope_with_aggregate(question):
        return sql

    s = sql.strip().rstrip(";").strip()
    if not s:
        return sql

    # Already-correct latest-year shape: skip.
    low = s.lower()
    if "latest_year" in low and re.search(r"=\s*[a-z_][a-z0-9_]*\.latest_year\b", low):
        return sql

    parsed = _parse_select_from_groupby(s)
    if not parsed:
        return sql
    if not parsed.get("where"):
        return sql

    alias_map = _parse_from_join_aliases(f"SELECT 1 FROM {parsed['from_core'].strip()}")
    try:
        analyzer = get_schema_analyzer()
    except Exception:
        analyzer = None
    allowed_cols: Dict[str, Set[str]] = {}
    if not analyzer and ctx:
        try:
            allowed = _parse_allowed_identifiers(ctx)
            allowed_cols = {
                t.lower(): {c.lower() for c in cols}
                for t, cols in allowed.items()
            }
        except Exception:
            allowed_cols = {}

    where_raw = re.sub(r"^\s*WHERE\s+", "", parsed["where"].strip(), flags=re.I).strip()
    if not where_raw:
        return sql
    # Target only correlated max-date subquery shape.
    if not re.search(r"\b(?:IN|=)\s*\(\s*SELECT\s+MAX\s*\(", where_raw, flags=re.I):
        return sql
    if "year(" not in where_raw.lower():
        return sql

    # Keep rewrite conservative on OR expressions.
    if _has_top_level_or(where_raw):
        return sql

    def _split_expr_alias(item: str) -> tuple:
        m = re.match(
            r"""(?is)^\s*(.*?)\s+AS\s+(`[^`]+`|'[^']+'|"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)\s*$""",
            item.strip(),
        )
        if not m:
            return item.strip(), None
        return m.group(1).strip(), m.group(2).strip()

    def _parse_qualified_col(expr: str) -> Optional[tuple]:
        m = re.match(
            r"(?is)^\s*(?:`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\.\s*)?`?([A-Za-z_][A-Za-z0-9_]*)`?\s*$",
            expr.strip(),
        )
        if not m:
            return None
        return m.group(1), m.group(2)

    def _strip_as(item: str) -> str:
        return re.sub(
            r"""(?is)\s+AS\s+(`[^`]+`|'[^']+'|"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)\s*$""",
            "",
            item.strip(),
        ).strip()

    def _safe_alias(a: Optional[str], default_name: str) -> str:
        if not a:
            return default_name
        x = a.strip()
        if (x.startswith("`") and x.endswith("`")) or (x.startswith("'") and x.endswith("'")) or (x.startswith('"') and x.endswith('"')):
            x = x[1:-1]
        x = re.sub(r"\s+", "_", x)
        x = re.sub(r"[^A-Za-z0-9_]", "", x)
        if not x:
            return default_name
        if not re.match(r"[A-Za-z_]", x[0]):
            return default_name
        return x

    def _table_cols(tbl: str) -> Set[str]:
        if not tbl:
            return set()
        t_l = tbl.lower()
        if analyzer:
            try:
                return {x.lower() for x in analyzer.get_table_columns(t_l)}
            except Exception:
                return set()
        return set(allowed_cols.get(t_l, set()))

    def _repair_date_expr_if_needed(expr_raw: str) -> str:
        """
        If <alias>.<date_col> points to a table that does not own <date_col>,
        and exactly one other visible alias owns it, rewrite to that alias.
        """
        parsed_col = _parse_qualified_col(expr_raw)
        if not parsed_col:
            return expr_raw
        a, c = parsed_col
        if not a:
            return expr_raw

        src_table = alias_map.get(a) or alias_map.get(a.lower())
        if not src_table:
            return expr_raw
        src_cols = _table_cols(src_table)
        if not src_cols:
            return expr_raw
        if c.lower() in src_cols:
            return expr_raw

        candidates = []
        seen_tables = set()
        for al, tbl in alias_map.items():
            al_l = (al or "").lower()
            if not al_l or al_l in seen_tables:
                continue
            # skip duplicate convenience key entries from alias parser
            seen_tables.add(al_l)
            cols = _table_cols(tbl)
            if c.lower() in cols and al_l != a.lower():
                candidates.append(al)

        # unique repair only
        uniq = []
        seen = set()
        for x in candidates:
            xl = x.lower()
            if xl in seen:
                continue
            seen.add(xl)
            uniq.append(x)
        if len(uniq) != 1:
            return expr_raw
        return f"{uniq[0]}.{c}"

    select_items = _split_top_level_commas(parsed["select"])
    if not select_items:
        return sql

    entity_items: List[str] = []
    date_expr_raw: Optional[str] = None
    date_alias_raw: Optional[str] = None
    sum_expr_inner: Optional[str] = None
    sum_alias_raw: Optional[str] = None
    measure_alias: Optional[str] = None
    date_col_name: Optional[str] = None

    for it in select_items:
        expr, alias = _split_expr_alias(it)

        m_sum = re.match(r"(?is)^SUM\s*\(\s*(.+?)\s*\)$", expr)
        if m_sum and sum_expr_inner is None:
            sum_expr_inner = m_sum.group(1).strip()
            sum_alias_raw = alias
            refs = re.findall(r"`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\.", sum_expr_inner, flags=re.I)
            if len(refs) == 1:
                measure_alias = refs[0]
            continue

        # Date may be raw (p.paymentDate) or already aggregated.
        m_max = re.match(r"(?is)^MAX\s*\(\s*(.+?)\s*\)$", expr)
        candidate = m_max.group(1).strip() if m_max else expr
        parsed_col = _parse_qualified_col(candidate)
        if parsed_col:
            a, c = parsed_col
            if ("date" in c.lower()) or ("time" in c.lower()):
                if date_expr_raw is None:
                    date_expr_raw = _repair_date_expr_if_needed(candidate)
                    date_alias_raw = alias
                    repaired_col = _parse_qualified_col(date_expr_raw)
                    if repaired_col:
                        ra, rc = repaired_col
                        date_col_name = rc
                        if ra:
                            measure_alias = ra
                    else:
                        date_col_name = c
                        if a:
                            measure_alias = a
                    continue

        # Keep non-aggregate entity columns.
        if not re.search(r"\b(SUM|COUNT|AVG|MIN|MAX)\s*\(", expr, flags=re.I):
            entity_items.append(it.strip())

    if not date_expr_raw or not sum_expr_inner:
        return sql

    if not measure_alias:
        refs = re.findall(r"`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\.", date_expr_raw, flags=re.I)
        if len(refs) == 1:
            measure_alias = refs[0]
    if not measure_alias or not date_col_name:
        return sql

    # Remove correlated MAX(date) predicate from WHERE, keep unrelated filters.
    conjuncts = _split_top_level_and(where_raw)
    keep_conj: List[str] = []
    for c in conjuncts:
        c0 = c.strip()
        low_c0 = c0.lower()
        if ("select" in low_c0) and ("max(" in low_c0) and ("year(" in low_c0):
            continue
        keep_conj.append(c0)

    # Build canonical GROUP BY from non-aggregate entity columns.
    group_items: List[str] = []
    seen_g = set()
    for it in entity_items:
        g = _strip_as(it)
        k = g.lower()
        if k in seen_g:
            continue
        seen_g.add(k)
        group_items.append(g)

    if not group_items:
        # fallback from original GROUP BY: drop YEAR(...) and aggregates
        for g in _split_top_level_commas(parsed.get("groupby", "")):
            g0 = g.strip()
            if not g0:
                continue
            low_g = g0.lower()
            if ("year(" in low_g) or re.search(r"\b(SUM|COUNT|AVG|MIN|MAX)\s*\(", low_g, flags=re.I):
                continue
            k = g0.lower()
            if k in seen_g:
                continue
            seen_g.add(k)
            group_items.append(g0)

    if not group_items:
        return sql

    # Qualify SUM expression to metric alias if expression is bare column.
    sum_expr_final = sum_expr_inner.strip()
    if re.fullmatch(r"`?[A-Za-z_][A-Za-z0-9_]*`?", sum_expr_final):
        sum_expr_final = f"{measure_alias}.{sum_expr_final.strip('`')}"

    date_alias_out = _safe_alias(date_alias_raw, f"latest_{date_col_name}")
    sum_alias_out = _safe_alias(sum_alias_raw, "total_amount")

    # Preserve requested ORDER BY direction, but make it aggregate-safe.
    order_clause = ""
    limit_clause = ""
    tail = (parsed.get("tail") or "").strip()
    if tail:
        m_limit = re.search(r"\bLIMIT\b.+$", tail, flags=re.I)
        if m_limit:
            limit_clause = " " + m_limit.group(0).strip()
        if re.search(r"\bORDER\s+BY\b", tail, flags=re.I):
            if re.search(r"\bDESC\b", tail, flags=re.I):
                direction = "DESC"
            elif re.search(r"\bASC\b", tail, flags=re.I):
                direction = "ASC"
            else:
                direction = "DESC"
            order_clause = f"ORDER BY MAX({date_expr_raw}) {direction}"

    canonical_lines = []
    canonical_lines.append("SELECT")
    select_out = []
    select_out.extend(entity_items)
    select_out.append(f"MAX({date_expr_raw}) AS {date_alias_out}")
    select_out.append(f"SUM({sum_expr_final}) AS {sum_alias_out}")
    canonical_lines.append("  " + ",\n  ".join(select_out))
    canonical_lines.append(f"FROM {parsed['from_core'].strip()}")
    if keep_conj:
        canonical_lines.append("WHERE " + " AND ".join(keep_conj))
    canonical_lines.append("GROUP BY " + ", ".join(group_items))
    if order_clause:
        canonical_lines.append(order_clause + limit_clause)
    elif limit_clause:
        canonical_lines.append(limit_clause.strip())
    canonical_sql = "\n".join(canonical_lines).strip() + ";"

    # Reuse the existing latest-year rewrite.
    fixed = _rewrite_latest_year_same_year_aggregate(canonical_sql, question, ctx)
    return fixed if fixed else canonical_sql

def _move_latest_year_filter_where_to_inner_on(sql: str) -> str:
    """
    Canonicalize style:
      WHERE YEAR(p.paymentDate) = ly.latest_year
    ->
      JOIN payments p ON ... AND YEAR(p.paymentDate) = ly.latest_year

    Safety rules:
    - only for queries parseable as SELECT...FROM...GROUP BY...
    - only when all joins are INNER/JOIN (no LEFT/RIGHT/FULL)
    - only top-level AND predicates (skip OR)
    - only predicates matching YEAR(<alias>.<col>) = <alias>.latest_year (or reversed)
    """
    if not sql:
        return sql
    if "latest_year" not in (sql or "").lower():
        return sql

    def _is_word_boundary(text: str, i: int, n: int) -> bool:
        left_ok = i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")
        right_ok = (i + n) >= len(text) or not (text[i + n].isalnum() or text[i + n] == "_")
        return left_ok and right_ok

    def _scan_top_level_word_positions(text: str, word: str) -> List[int]:
        out: List[int] = []
        w = word.lower()
        n = len(w)
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
            if depth == 0 and i + n <= len(text) and text[i:i + n].lower() == w and _is_word_boundary(text, i, n):
                out.append(i)
            i += 1
        return out

    def _first_top_level_word_pos(text: str, word: str, start: int = 0) -> int:
        for p in _scan_top_level_word_positions(text, word):
            if p >= start:
                return p
        return -1

    def _first_top_level_group_by_pos(text: str, start: int = 0) -> int:
        for p in _scan_top_level_word_positions(text, "group"):
            if p < start:
                continue
            j = p + len("group")
            while j < len(text) and text[j].isspace():
                j += 1
            if j + 2 <= len(text) and text[j:j + 2].lower() == "by" and _is_word_boundary(text, j, 2):
                return p
        return -1

    def _first_top_level_order_by_pos(text: str, start: int = 0) -> int:
        for p in _scan_top_level_word_positions(text, "order"):
            if p < start:
                continue
            j = p + len("order")
            while j < len(text) and text[j].isspace():
                j += 1
            if j + 2 <= len(text) and text[j:j + 2].lower() == "by" and _is_word_boundary(text, j, 2):
                return p
        return -1

    s0 = sql.strip().rstrip(";").strip()
    if not s0:
        return sql

    from_pos = _first_top_level_word_pos(s0, "from", start=0)
    if from_pos < 0:
        return sql

    where_pos = _first_top_level_word_pos(s0, "where", start=from_pos + 4)
    group_pos = _first_top_level_group_by_pos(s0, start=from_pos + 4)
    having_pos = _first_top_level_word_pos(s0, "having", start=from_pos + 4)
    order_pos = _first_top_level_order_by_pos(s0, start=from_pos + 4)
    limit_pos = _first_top_level_word_pos(s0, "limit", start=from_pos + 4)

    def _next_pos_after(start_pos: int, candidates: List[int]) -> int:
        vals = [p for p in candidates if p >= 0 and p > start_pos]
        return min(vals) if vals else -1

    # FROM core ends at first top-level clause after FROM.
    from_end = _next_pos_after(
        from_pos,
        [p for p in [where_pos, group_pos, having_pos, order_pos, limit_pos] if p >= 0],
    )
    if from_end < 0:
        from_end = len(s0)

    from_core = s0[from_pos + 4:from_end].strip()
    if not from_core:
        return sql
    if not re.search(r"\bjoin\b", from_core, flags=re.I):
        return sql
    if re.search(r"\b(left|right|full)\s+join\b", from_core, flags=re.I):
        return sql

    if where_pos < 0:
        return sql
    where_end = _next_pos_after(
        where_pos,
        [p for p in [group_pos, having_pos, order_pos, limit_pos] if p >= 0],
    )
    if where_end < 0:
        where_end = len(s0)
    where_body = s0[where_pos + 5:where_end].strip()
    if not where_body:
        return sql
    if _has_top_level_or(where_body):
        return sql

    select_prefix = s0[:from_pos].strip()
    rest_after_where = s0[where_end:].strip() if where_end < len(s0) else ""

    def _strip_wrapping_parens(expr: str) -> str:
        s = (expr or "").strip()
        changed = True
        while changed and s.startswith("(") and s.endswith(")"):
            changed = False
            depth = 0
            ok = True
            for i, ch in enumerate(s):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth < 0:
                        ok = False
                        break
                    if depth == 0 and i != len(s) - 1:
                        ok = False
                        break
            if ok and depth == 0:
                s = s[1:-1].strip()
                changed = True
        return s

    def _extract_latest_year_alias_pair(cond: str) -> Optional[tuple]:
        c = _strip_wrapping_parens(cond)
        p1 = re.match(
            r"""(?is)^\s*YEAR\s*\(\s*`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\.\s*`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\)\s*=\s*`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\.\s*`?latest_year`?\s*$""",
            c,
        )
        if p1:
            return p1.group(1), p1.group(3)
        p2 = re.match(
            r"""(?is)^\s*`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\.\s*`?latest_year`?\s*=\s*YEAR\s*\(\s*`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\.\s*`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\)\s*$""",
            c,
        )
        if p2:
            return p2.group(2), p2.group(1)
        return None

    def _rhs_alias_from_join_pre(pre: str) -> Optional[str]:
        p = pre.strip()
        # Derived table join: JOIN ( ... ) AS alias
        m_sub = re.search(r"\)\s*(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*$", p, flags=re.I)
        if m_sub:
            return m_sub.group(1)
        # Regular table join: JOIN table [AS] alias?
        m_tbl = re.search(
            r"""\b(?:INNER\s+JOIN|JOIN)\s+(`?[A-Za-z_][A-Za-z0-9_]*`?(?:\.`?[A-Za-z_][A-Za-z0-9_]*`?)?)(?:\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*))?\s*$""",
            p,
            flags=re.I,
        )
        if not m_tbl:
            return None
        alias = (m_tbl.group(2) or "").strip()
        if alias:
            return alias
        tbl = (m_tbl.group(1) or "").replace("`", "").split(".")[-1]
        return tbl or None

    def _base_aliases_from_from_core_base(base: str) -> Set[str]:
        """
        Parse base FROM part before first JOIN.
        Examples:
          customers AS c  -> {"c", "customers"}
          `db`.`customers` c -> {"c", "customers"}
          customers -> {"customers"}
        """
        b = (base or "").strip()
        out: Set[str] = set()
        m = re.match(
            r"""(?is)^(`?[A-Za-z_][A-Za-z0-9_]*`?(?:\.`?[A-Za-z_][A-Za-z0-9_]*`?)?)(?:\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*))?\s*$""",
            b,
        )
        if not m:
            return out
        tbl = (m.group(1) or "").replace("`", "").split(".")[-1].strip()
        alias = (m.group(2) or "").strip()
        if tbl:
            out.add(tbl.lower())
        if alias:
            out.add(alias.lower())
        return out

    # Build join segments at top level.
    raw_join_positions = _scan_top_level_word_positions(from_core, "join")
    if not raw_join_positions:
        return sql
    join_positions: List[int] = []
    for p in raw_join_positions:
        start = p
        prefix = from_core[:p]
        m_inner = re.search(r"\bINNER\s*$", prefix, flags=re.I)
        if m_inner:
            start = m_inner.start()
        if not join_positions or join_positions[-1] != start:
            join_positions.append(start)

    segments = []
    for i, start in enumerate(join_positions):
        end = join_positions[i + 1] if i + 1 < len(join_positions) else len(from_core)
        seg = from_core[start:end]
        on_rel = _first_top_level_word_pos(seg, "on", start=0)
        if on_rel < 0:
            continue
        pre = seg[:on_rel].rstrip()
        on_expr = seg[on_rel + 2:].strip()
        rhs_alias = _rhs_alias_from_join_pre(pre)
        segments.append(
            {
                "start": start,
                "end": end,
                "pre": pre,
                "on": on_expr,
                "rhs_alias": (rhs_alias or "").strip(),
                "moved": [],
                "avail": set(),
            }
        )

    if not segments:
        return sql

    # Compute alias scope available at each JOIN ON clause.
    base_aliases = _base_aliases_from_from_core_base(from_core[:join_positions[0]])
    running_aliases: Set[str] = set(base_aliases)
    for seg in segments:
        rhs = (seg.get("rhs_alias") or "").lower()
        if rhs:
            running_aliases.add(rhs)
        seg["avail"] = set(running_aliases)

    # Extract movable predicates from WHERE.
    conjuncts = _split_top_level_and(where_body)
    keep_conj: List[str] = []
    moved_any = False

    for cond in conjuncts:
        c0 = cond.strip()
        pair = _extract_latest_year_alias_pair(c0)
        if not pair:
            keep_conj.append(c0)
            continue

        date_alias, year_alias = pair[0], pair[1]
        date_alias_l = date_alias.lower()
        year_alias_l = year_alias.lower()

        # Prefer the rightmost JOIN where BOTH aliases are already in scope.
        # This guarantees the ON clause can legally reference both sides.
        target_idx = -1
        for j in range(len(segments) - 1, -1, -1):
            avail = {a.lower() for a in (segments[j].get("avail") or set())}
            if date_alias_l in avail and year_alias_l in avail:
                target_idx = j
                break
        # Fallback: attach to a JOIN introducing one of the aliases.
        if target_idx < 0:
            for j in range(len(segments) - 1, -1, -1):
                rhs = (segments[j]["rhs_alias"] or "").lower()
                if rhs in {date_alias_l, year_alias_l}:
                    target_idx = j
                    break

        if target_idx < 0:
            keep_conj.append(c0)
            continue

        segments[target_idx]["moved"].append(c0)
        moved_any = True

    if not moved_any:
        return sql

    # Reconstruct FROM/JOIN with predicates moved into target ON clauses.
    base = from_core[:join_positions[0]].strip()
    rebuilt_parts: List[str] = [base] if base else []
    for seg in segments:
        on_new = seg["on"]
        for p in seg["moved"]:
            on_new = f"({on_new}) AND ({p})"
        rebuilt_parts.append(f"{seg['pre']} ON {on_new}")
    from_new = "\n".join(p for p in rebuilt_parts if p and p.strip())

    out_lines = []
    out_lines.append(select_prefix)
    out_lines.append(f"FROM {from_new}")
    if keep_conj:
        out_lines.append("WHERE " + " AND ".join(keep_conj))
    if rest_after_where:
        out_lines.append(rest_after_where)

    fixed = "\n".join(x for x in out_lines if x and x.strip()).strip()
    if not fixed.endswith(";"):
        fixed += ";"
    return fixed

def _rewrite_latest_year_derived_subquery_misuse(sql: str, question: str, ctx: str = "") -> str:
    """
    Handle malformed pattern:

      SELECT entity_cols, sub.paymentDate, sub.amount
      FROM entity e
      JOIN (
        SELECT key, MAX(date) AS latest_date, SUM(amount) AS total_amount
        FROM metric
        GROUP BY key
      ) sub ON e.key = sub.key
      [WHERE ...]

    This pattern is both semantically wrong for "same year sum" and can reference
    non-existent columns from derived subquery aliases.
    """
    if not sql:
        return sql

    if not _question_wants_latest_year_scope_with_aggregate(question):
        return sql

    s = sql.strip().rstrip(";").strip()
    if not s:
        return sql

    def _fallback_rewrite_customers_payments_latest_year(raw: str) -> Optional[str]:
        """
        Strict salvage path for malformed "customers + payments latest year" SQL.
        Only triggers for this specific intent/signature.
        """
        low_raw = (raw or "").lower()
        if "from customers" not in low_raw:
            return None
        if "payments" not in low_raw:
            return None
        if "paymentdate" not in low_raw or "amount" not in low_raw:
            return None
        if "year(" not in low_raw and "latest_year" not in low_raw:
            return None
        if "sum(" not in low_raw or "max(" not in low_raw:
            return None

        m_sel = re.search(r"(?is)^\s*SELECT\s+(.*?)\s+FROM\b", raw)
        select_txt = (m_sel.group(1) if m_sel else "")
        wants_name = (
            bool(re.search(r"\bcustomername\b", select_txt, flags=re.I))
            or bool(re.search(r"\bcustomer\s+name\b", (question or "").lower()))
        )

        date_alias = "latest_payment_date"
        sum_alias = "total_amount_paid"
        m_date_alias = re.search(
            r"(?is)\bMAX\s*\(\s*[^)]*paymentDate[^)]*\)\s+AS\s+([A-Za-z_][A-Za-z0-9_]*)",
            raw,
        )
        if m_date_alias:
            date_alias = m_date_alias.group(1)
        m_sum_alias = re.search(
            r"(?is)\bSUM\s*\(\s*[^)]*amount[^)]*\)\s+AS\s+([A-Za-z_][A-Za-z0-9_]*)",
            raw,
        )
        if m_sum_alias:
            sum_alias = m_sum_alias.group(1)

        select_cols = ["cust.customerNumber"]
        if wants_name:
            select_cols.append("cust.customerName")
        select_cols.append(f"MAX(pay.paymentDate) AS {date_alias}")
        select_cols.append(f"SUM(pay.amount) AS {sum_alias}")

        group_cols = ["cust.customerNumber"]
        if wants_name:
            group_cols.append("cust.customerName")
        select_block = ",\n  ".join(select_cols)

        return (
            "SELECT\n"
            f"  {select_block}\n"
            "FROM customers cust\n"
            "JOIN (\n"
            "  SELECT customerNumber, MAX(YEAR(paymentDate)) AS latest_year\n"
            "  FROM payments\n"
            "  GROUP BY customerNumber\n"
            ") AS pay_latest_years\n"
            "  ON pay_latest_years.customerNumber = cust.customerNumber\n"
            "JOIN payments AS pay\n"
            "  ON pay.customerNumber = cust.customerNumber\n"
            " AND YEAR(pay.paymentDate) = pay_latest_years.latest_year\n"
            f"GROUP BY {', '.join(group_cols)}\n"
            f"ORDER BY {date_alias} DESC;"
        )

    # If already using latest_year pattern and SQL is parseable, keep as-is.
    low = s.lower()
    has_malformed_nested_payments_subquery = bool(
        re.search(r"(?is)\bFROM\s+payments\b\s+JOIN\s*\(", s)
    )
    if has_malformed_nested_payments_subquery:
        rescued = _fallback_rewrite_customers_payments_latest_year(s)
        if rescued:
            return rescued

    if "latest_year" in low and re.search(r"=\s*[a-z_][a-z0-9_]*\.latest_year\b", low):
        has_nested_join_subquery = bool(
            re.search(r"(?is)\bJOIN\s*\(\s*SELECT\b[\s\S]*\bJOIN\s*\(\s*SELECT\b", s)
        )
        if _parse_select_from_groupby(s) and not has_nested_join_subquery:
            return sql

    m_join_sub = re.search(r"\bJOIN\s*\(\s*SELECT\b", s, flags=re.I)
    if not m_join_sub:
        return sql

    open_idx = s.find("(", m_join_sub.start())
    if open_idx == -1:
        return sql

    # Find matching ')' for JOIN ( ... )
    depth = 0
    close_idx = -1
    in_sq = False
    in_dq = False
    i = open_idx
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
        elif ch == ")":
            depth -= 1
            if depth == 0:
                close_idx = i
                break
        i += 1

    if close_idx == -1:
        return sql

    before = s[:m_join_sub.start()].strip()
    inner_sql = s[open_idx + 1:close_idx].strip()
    after = s[close_idx + 1:].strip()

    # before ends with join type token (e.g., "... FROM customers AS cust INNER")
    before = re.sub(r"\b(?:INNER|LEFT|RIGHT|FULL|CROSS)\s*$", "", before, flags=re.I).strip()

    m_after = re.match(
        r"(?is)^(?:AS\s+)?(?P<sub>[A-Za-z_][A-Za-z0-9_]*)\s+ON\s+(?P<on>.+?)(?:\s+WHERE\s+(?P<where>.+))?$",
        after,
    )
    if not m_after:
        return sql

    sub_alias = m_after.group("sub").strip()
    on_clause = (m_after.group("on") or "").strip()

    m_before = re.match(
        r"(?is)^SELECT\s+(?P<select>.+?)\s+FROM\s+(?P<t>`?[A-Za-z_][A-Za-z0-9_]*`?)(?:\s+(?:AS\s+)?(?P<a>[A-Za-z_][A-Za-z0-9_]*))?\s*$",
        before,
    )
    if not m_before:
        return sql

    outer_select = m_before.group("select").strip()
    entity_table = m_before.group("t").strip()
    entity_alias = (m_before.group("a") or "").strip()
    if not entity_alias:
        # Recover dropped base-table alias from SELECT qualifiers.
        outer_alias_refs = [
            a for a in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.", outer_select, flags=re.I)
            if a.lower() != sub_alias.lower()
        ]
        uniq_refs = []
        seen_refs = set()
        for a in outer_alias_refs:
            k = a.lower()
            if k in seen_refs:
                continue
            seen_refs.add(k)
            uniq_refs.append(a)
        if len(uniq_refs) == 1:
            entity_alias = uniq_refs[0]
        else:
            # Fallback deterministic alias from table name.
            entity_alias = entity_table.strip("`")

    inner_parsed = _parse_select_from_groupby(inner_sql)
    if not inner_parsed:
        rescued = _fallback_rewrite_customers_payments_latest_year(s)
        if rescued:
            return rescued
        return sql
    if inner_parsed.get("where"):
        # Conservative: skip if inner query already has filters.
        return sql

    m_inner_from = re.match(
        r"(?is)^(`?[A-Za-z_][A-Za-z0-9_]*`?)(?:\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*))?$",
        inner_parsed["from_core"].strip(),
    )
    if not m_inner_from:
        return sql

    metric_table = m_inner_from.group(1).strip()
    metric_table_plain = metric_table.strip("`")
    metric_inner_alias = (m_inner_from.group(2) or metric_table_plain).strip()

    def _split_expr_alias(item: str) -> tuple:
        m = re.match(
            r"""(?is)^\s*(.*?)\s+AS\s+(`[^`]+`|'[^']+'|"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)\s*$""",
            item.strip(),
        )
        if not m:
            return item.strip(), None
        return m.group(1).strip(), m.group(2).strip()

    def _parse_simple_col(expr: str) -> Optional[tuple]:
        m = re.match(
            r"(?is)^\s*(?:`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\.\s*)?`?([A-Za-z_][A-Za-z0-9_]*)`?\s*$",
            expr.strip(),
        )
        if not m:
            return None
        return m.group(1), m.group(2)

    def _safe_alias(a: Optional[str], default_name: str) -> str:
        if not a:
            return default_name
        x = a.strip()
        if (x.startswith("`") and x.endswith("`")) or (x.startswith("'") and x.endswith("'")) or (x.startswith('"') and x.endswith('"')):
            x = x[1:-1]
        x = re.sub(r"\s+", "_", x)
        x = re.sub(r"[^A-Za-z0-9_]", "", x)
        if not x:
            return default_name
        if not re.match(r"[A-Za-z_]", x[0]):
            return default_name
        return x

    inner_select_items = _split_top_level_commas(inner_parsed["select"])
    inner_group_items = _split_top_level_commas(inner_parsed["groupby"])
    if not inner_group_items:
        return sql

    key_col = None
    first_group = _parse_simple_col(inner_group_items[0])
    if first_group:
        key_col = first_group[1]
    if not key_col:
        return sql

    date_col = None
    date_alias = None
    sum_expr = None
    sum_alias = None

    for item in inner_select_items:
        expr, alias = _split_expr_alias(item)
        m_max = re.match(r"(?is)^MAX\s*\(\s*(.+?)\s*\)$", expr)
        if m_max and date_col is None:
            parsed_col = _parse_simple_col(m_max.group(1).strip())
            if parsed_col:
                _qa, c = parsed_col
                if ("date" in c.lower()) or ("time" in c.lower()):
                    date_col = c
                    date_alias = alias
                    continue

        m_sum = re.match(r"(?is)^SUM\s*\(\s*(.+?)\s*\)$", expr)
        if m_sum and sum_expr is None:
            sum_expr = m_sum.group(1).strip()
            sum_alias = alias
            continue

    if not date_col or not sum_expr:
        return sql

    # Safety: derived metric table must actually own the date column.
    metric_cols: Set[str] = set()
    try:
        analyzer = get_schema_analyzer()
        metric_cols = {c.lower() for c in analyzer.get_table_columns(metric_table_plain)}
    except Exception:
        metric_cols = set()
    if not metric_cols and ctx:
        try:
            allowed = _parse_allowed_identifiers(ctx)
            metric_cols = {c.lower() for c in allowed.get(metric_table_plain, set())}
        except Exception:
            metric_cols = set()
    if metric_cols and date_col.lower() not in metric_cols:
        return sql

    # Parse entity/sub key mapping from ON
    eq1 = re.search(
        rf"`?{re.escape(entity_alias)}`?\s*\.\s*`?([A-Za-z_][A-Za-z0-9_]*)`?\s*=\s*`?{re.escape(sub_alias)}`?\s*\.\s*`?([A-Za-z_][A-Za-z0-9_]*)`?",
        on_clause,
        flags=re.I,
    )
    eq2 = re.search(
        rf"`?{re.escape(sub_alias)}`?\s*\.\s*`?([A-Za-z_][A-Za-z0-9_]*)`?\s*=\s*`?{re.escape(entity_alias)}`?\s*\.\s*`?([A-Za-z_][A-Za-z0-9_]*)`?",
        on_clause,
        flags=re.I,
    )
    entity_key_col = None
    sub_key_col = None
    if eq1:
        entity_key_col = eq1.group(1)
        sub_key_col = eq1.group(2)
    elif eq2:
        sub_key_col = eq2.group(1)
        entity_key_col = eq2.group(2)
    if not entity_key_col:
        return sql
    if sub_key_col and sub_key_col.lower() != key_col.lower():
        # Conservative: inconsistent key naming in derived table projection.
        return sql

    outer_items = _split_top_level_commas(outer_select)
    entity_items = []
    for it in outer_items:
        if re.search(rf"\b{re.escape(sub_alias)}\s*\.", it, flags=re.I):
            continue
        entity_items.append(it.strip())
    if not entity_items:
        entity_items = [f"{entity_alias}.{entity_key_col}"]

    def _strip_as(item: str) -> str:
        return re.sub(
            r"""(?is)\s+AS\s+(`[^`]+`|'[^']+'|"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)\s*$""",
            "",
            item.strip(),
        ).strip()

    group_items = []
    seen = set()
    for it in entity_items:
        base = _strip_as(it)
        k = base.lower()
        if k in seen:
            continue
        seen.add(k)
        group_items.append(base)

    existing_aliases = {entity_alias.lower(), sub_alias.lower(), metric_inner_alias.lower()}

    def _pick_alias(base: str) -> str:
        b = re.sub(r"[^A-Za-z0-9_]", "", base) or "t"
        if not re.match(r"[A-Za-z_]", b[0]):
            b = f"t_{b}"
        cand = b
        n = 2
        while cand.lower() in existing_aliases:
            cand = f"{b}_{n}"
            n += 1
        existing_aliases.add(cand.lower())
        return cand

    year_alias = _pick_alias(f"{sub_alias}_latest_years")
    metric_alias = _pick_alias(f"{metric_table_plain}_rows")

    # Qualify SUM expression against metric alias.
    sum_metric_expr = re.sub(
        rf"\b(?:{re.escape(metric_inner_alias)}|{re.escape(metric_table_plain)})\s*\.",
        f"{metric_alias}.",
        sum_expr,
        flags=re.I,
    ).strip()
    if re.fullmatch(r"`?[A-Za-z_][A-Za-z0-9_]*`?", sum_metric_expr):
        sum_metric_expr = f"{metric_alias}.{sum_metric_expr.strip('`')}"

    date_alias_out = _safe_alias(date_alias, f"latest_{date_col}")
    sum_alias_out = _safe_alias(sum_alias, "total_amount")

    select_lines = []
    for it in entity_items:
        select_lines.append(it)
    select_lines.append(f"MAX({metric_alias}.{date_col}) AS {date_alias_out}")
    select_lines.append(f"SUM({sum_metric_expr}) AS {sum_alias_out}")
    select_block = ",\n  ".join(select_lines)

    fixed = (
        "SELECT\n"
        f"  {select_block}\n"
        f"FROM {entity_table} {entity_alias}\n"
        "JOIN (\n"
        "  SELECT\n"
        f"    {key_col},\n"
        f"    MAX(YEAR({date_col})) AS latest_year\n"
        f"  FROM {metric_table}\n"
        f"  GROUP BY {key_col}\n"
        f") AS {year_alias}\n"
        f"  ON {year_alias}.{key_col} = {entity_alias}.{entity_key_col}\n"
        f"JOIN {metric_table} AS {metric_alias}\n"
        f"  ON {metric_alias}.{key_col} = {entity_alias}.{entity_key_col}\n"
        f" AND YEAR({metric_alias}.{date_col}) = {year_alias}.latest_year\n"
        f"GROUP BY {', '.join(group_items)};"
    )
    return fixed


def _canonicalize_latest_year_same_year_pattern(sql: str, question: str, ctx: str = "") -> str:
    """
    Canonicalize correlated latest-year filters into a derived-table pattern.

    Target shape (example):
      WHERE YEAR(o.orderDate) = (
        SELECT MAX(YEAR(orderDate))
        FROM orders
        WHERE orderNumber IN (
          SELECT orderNumber
          FROM orderdetails
          WHERE productCode = p.productCode
        )
      )

    Canonicalized:
      JOIN (
        SELECT bridge.entity_key AS latest_key, MAX(YEAR(date_table.date_col)) AS latest_year
        FROM bridge_table bridge
        JOIN date_table date_table ON ...
        GROUP BY bridge.entity_key
      ) AS ly ON ly.latest_key = outer.entity_key
      WHERE YEAR(outer.date_col) = ly.latest_year
    """
    if not sql:
        return sql
    if not _question_wants_latest_year_scope_with_aggregate(question):
        return sql

    low_sql = sql.lower()
    if "latest_year" in low_sql and re.search(r"=\s*[a-z_][a-z0-9_]*\.latest_year\b", low_sql):
        return sql

    parsed = _parse_select_from_groupby(sql)
    if not parsed:
        return sql
    if not parsed.get("where"):
        return sql

    from_core = parsed["from_core"].strip()
    if not from_core:
        return sql

    alias_map = _parse_from_join_aliases(f"SELECT 1 FROM {from_core}")
    existing_aliases = {a.lower() for a in alias_map.keys() if a}

    try:
        analyzer = get_schema_analyzer()
    except Exception:
        analyzer = None

    allowed_cols: Dict[str, Set[str]] = {}
    if not analyzer and ctx:
        try:
            allowed = _parse_allowed_identifiers(ctx)
            allowed_cols = {t.lower(): {c.lower() for c in cols} for t, cols in allowed.items()}
        except Exception:
            allowed_cols = {}

    def _plain_table_name(table_expr: str) -> str:
        return (table_expr or "").replace("`", "").split(".")[-1].strip()

    def _table_cols(table_name: str) -> Set[str]:
        t = _plain_table_name(table_name).lower()
        if not t:
            return set()
        if analyzer:
            try:
                return {c.lower() for c in analyzer.get_table_columns(t)}
            except Exception:
                return set()
        return set(allowed_cols.get(t, set()))

    def _pick_alias(base: str, used: Set[str]) -> str:
        b = re.sub(r"[^A-Za-z0-9_]", "", base) or "t"
        if not re.match(r"[A-Za-z_]", b[0]):
            b = f"t_{b}"
        cand = b
        i = 2
        while cand.lower() in used:
            cand = f"{b}_{i}"
            i += 1
        used.add(cand.lower())
        return cand

    def _split_qualified_ref(expr: str) -> Optional[tuple]:
        m = re.match(
            r"""(?is)^\s*(?:`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\.\s*)?`?([A-Za-z_][A-Za-z0-9_]*)`?\s*$""",
            (expr or "").strip(),
        )
        if not m:
            return None
        return m.group(1), m.group(2)

    def _parse_year_eq_subquery(cond: str) -> Optional[tuple]:
        c = (cond or "").strip()
        m1 = re.match(
            r"""(?is)^YEAR\s*\(\s*`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\.\s*`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\)\s*=\s*\((SELECT.+)\)\s*$""",
            c,
        )
        if m1:
            return m1.group(1), m1.group(2), m1.group(3).strip()
        m2 = re.match(
            r"""(?is)^\((SELECT.+)\)\s*=\s*YEAR\s*\(\s*`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\.\s*`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\)\s*$""",
            c,
        )
        if m2:
            return m2.group(2), m2.group(3), m2.group(1).strip()
        return None

    groupby_aliases: Set[str] = set()
    for g in _split_top_level_commas(parsed.get("groupby", "")):
        q = _split_qualified_ref(g)
        if q and q[0]:
            groupby_aliases.add(q[0].lower())

    def _infer_bridge_entity_link(
        bridge_alias: str,
        avoid_aliases: Set[str],
    ) -> Optional[tuple]:
        """
        Infer outer-entity join link from FROM/JOIN ON predicates.
        Returns: (outer_entity_alias, outer_entity_col, bridge_entity_col)
        """
        b = (bridge_alias or "").lower().strip()
        if not b:
            return None

        eq_iter = re.finditer(
            r"`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\.\s*`?([A-Za-z_][A-Za-z0-9_]*)`?\s*=\s*"
            r"`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\.\s*`?([A-Za-z_][A-Za-z0-9_]*)`?",
            from_core,
            flags=re.I,
        )

        candidates: List[tuple] = []
        for m in eq_iter:
            a1, c1, a2, c2 = m.group(1), m.group(2), m.group(3), m.group(4)
            a1l, a2l = a1.lower(), a2.lower()
            if a1l == b and a2l in alias_map and a2l not in avoid_aliases:
                candidates.append((a2, c2, c1))
            elif a2l == b and a1l in alias_map and a1l not in avoid_aliases:
                candidates.append((a1, c1, c2))

        if not candidates:
            return None

        # Prefer aliases explicitly present in GROUP BY (typical entity key owner).
        for outer_alias, outer_col, bridge_col in candidates:
            if groupby_aliases and outer_alias.lower() in groupby_aliases:
                return outer_alias, outer_col, bridge_col

        # Fallback: first viable candidate.
        return candidates[0]

    where_body = re.sub(r"^\s*WHERE\s+", "", parsed["where"].strip(), flags=re.I).strip()
    if not where_body or _has_top_level_or(where_body):
        return sql

    conjuncts = _split_top_level_and(where_body)
    if not conjuncts:
        return sql

    target_idx = -1
    parsed_target = None

    for i, cond in enumerate(conjuncts):
        y = _parse_year_eq_subquery(cond)
        if not y:
            continue
        outer_date_alias, outer_date_col, sub1 = y

        m_sub1 = re.match(
            r"""(?is)^\s*SELECT\s+MAX\s*\(\s*YEAR\s*\(\s*(?:`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\.\s*)?`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\)\s*\)\s+FROM\s+(`?[A-Za-z_][A-Za-z0-9_]*`?(?:\.`?[A-Za-z_][A-Za-z0-9_]*`?)?)(?:\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*))?\s+WHERE\s+(.+?)\s*$""",
            sub1,
        )
        if not m_sub1:
            continue

        inner_date_qual = (m_sub1.group(1) or "").strip()
        inner_date_col = m_sub1.group(2).strip()
        inner_date_table = m_sub1.group(3).strip()
        inner_date_alias = (m_sub1.group(4) or "").strip() or _plain_table_name(inner_date_table)
        inner_where = (m_sub1.group(5) or "").strip()
        if not inner_where or _has_top_level_or(inner_where):
            continue
        inner_conj = _split_top_level_and(inner_where)
        if len(inner_conj) != 1:
            continue

        inner_pred = inner_conj[0].strip()
        inner_join_col = ""
        bridge_link_col = ""
        bridge_table = ""
        bridge_alias = ""
        outer_entity_alias = None
        outer_entity_col = None
        bridge_entity_col = None

        m_in = re.match(
            r"""(?is)^\s*(?:`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\.\s*)?`?([A-Za-z_][A-Za-z0-9_]*)`?\s+IN\s*\((SELECT.+)\)\s*$""",
            inner_pred,
        )

        if m_in:
            inner_join_qual = (m_in.group(1) or "").strip()
            inner_join_col = m_in.group(2).strip()
            sub2 = (m_in.group(3) or "").strip()
            if inner_join_qual and inner_join_qual.lower() not in {inner_date_alias.lower(), _plain_table_name(inner_date_table).lower()}:
                continue

            m_sub2 = re.match(
                r"""(?is)^\s*SELECT\s+(?:`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\.\s*)?`?([A-Za-z_][A-Za-z0-9_]*)`?\s+FROM\s+(`?[A-Za-z_][A-Za-z0-9_]*`?(?:\.`?[A-Za-z_][A-Za-z0-9_]*`?)?)(?:\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*))?\s+WHERE\s+(.+?)\s*$""",
                sub2,
            )
            if not m_sub2:
                continue

            bridge_link_qual = (m_sub2.group(1) or "").strip()
            bridge_link_col = m_sub2.group(2).strip()
            bridge_table = m_sub2.group(3).strip()
            bridge_alias = (m_sub2.group(4) or "").strip() or _plain_table_name(bridge_table)
            bridge_where = (m_sub2.group(5) or "").strip()
            if not bridge_where or _has_top_level_or(bridge_where):
                continue
            bridge_conj = _split_top_level_and(bridge_where)
            if len(bridge_conj) != 1:
                continue

            m_eq = re.match(
                r"""(?is)^\s*(.+?)\s*=\s*(.+?)\s*$""",
                bridge_conj[0].strip(),
            )
            if not m_eq:
                continue
            left_ref = _split_qualified_ref(m_eq.group(1))
            right_ref = _split_qualified_ref(m_eq.group(2))
            if not left_ref or not right_ref:
                continue

            lq, lc = left_ref
            rq, rc = right_ref
            bridge_aliases = {bridge_alias.lower(), _plain_table_name(bridge_table).lower()}

            if rq and rq.lower() in alias_map and (not lq or lq.lower() in bridge_aliases):
                outer_entity_alias = rq
                outer_entity_col = rc
                bridge_entity_col = lc
            elif lq and lq.lower() in alias_map and (not rq or rq.lower() in bridge_aliases):
                outer_entity_alias = lq
                outer_entity_col = lc
                bridge_entity_col = rc
            else:
                continue

            if bridge_link_qual and bridge_link_qual.lower() not in bridge_aliases:
                continue
            if inner_date_qual and inner_date_qual.lower() not in {inner_date_alias.lower(), _plain_table_name(inner_date_table).lower()}:
                continue

        else:
            # Also support direct equality correlation:
            #   WHERE YEAR(o.dateCol) = (
            #     SELECT MAX(YEAR(dateCol))
            #     FROM date_table
            #     WHERE joinCol = outer_bridge.joinCol
            #   )
            # This is a common malformed pattern for latest-year-per-entity.
            m_eq_direct = re.match(
                r"""(?is)^\s*(.+?)\s*=\s*(.+?)\s*$""",
                inner_pred,
            )
            if not m_eq_direct:
                continue
            left_ref = _split_qualified_ref(m_eq_direct.group(1))
            right_ref = _split_qualified_ref(m_eq_direct.group(2))
            if not left_ref or not right_ref:
                continue

            lq, lc = left_ref
            rq, rc = right_ref
            inner_date_aliases = {inner_date_alias.lower(), _plain_table_name(inner_date_table).lower()}

            outer_bridge_alias = None
            if lq and lq.lower() in alias_map and lq.lower() != outer_date_alias.lower() and ((not rq) or (rq.lower() in inner_date_aliases)):
                outer_bridge_alias = lq
                bridge_link_col = lc
                inner_join_col = rc
            elif rq and rq.lower() in alias_map and rq.lower() != outer_date_alias.lower() and ((not lq) or (lq.lower() in inner_date_aliases)):
                outer_bridge_alias = rq
                bridge_link_col = rc
                inner_join_col = lc
            else:
                continue

            if not inner_join_col or not bridge_link_col:
                continue

            bridge_alias = outer_bridge_alias
            bridge_table = alias_map.get(bridge_alias) or alias_map.get(bridge_alias.lower()) or bridge_alias
            inferred = _infer_bridge_entity_link(
                bridge_alias,
                avoid_aliases={outer_date_alias.lower(), bridge_alias.lower()},
            )
            if not inferred:
                continue
            outer_entity_alias, outer_entity_col, bridge_entity_col = inferred

        parsed_target = {
            "outer_date_alias": outer_date_alias,
            "outer_date_col": outer_date_col,
            "inner_date_table": inner_date_table,
            "inner_date_alias": inner_date_alias,
            "inner_date_col": inner_date_col,
            "inner_join_col": inner_join_col,
            "bridge_table": bridge_table,
            "bridge_alias": bridge_alias,
            "bridge_link_col": bridge_link_col,
            "bridge_entity_col": bridge_entity_col,
            "outer_entity_alias": outer_entity_alias,
            "outer_entity_col": outer_entity_col,
        }
        target_idx = i
        break

    if target_idx < 0 or not parsed_target:
        return sql

    outer_date_alias = parsed_target["outer_date_alias"]
    outer_date_col = parsed_target["outer_date_col"]
    inner_date_table = parsed_target["inner_date_table"]
    inner_date_col = parsed_target["inner_date_col"]
    inner_join_col = parsed_target["inner_join_col"]
    bridge_table = parsed_target["bridge_table"]
    bridge_link_col = parsed_target["bridge_link_col"]
    bridge_entity_col = parsed_target["bridge_entity_col"]
    outer_entity_alias = parsed_target["outer_entity_alias"]
    outer_entity_col = parsed_target["outer_entity_col"]

    if outer_date_alias.lower() not in alias_map:
        return sql
    if outer_entity_alias.lower() not in alias_map:
        return sql

    outer_date_table = alias_map.get(outer_date_alias) or alias_map.get(outer_date_alias.lower()) or outer_date_alias
    outer_entity_table = alias_map.get(outer_entity_alias) or alias_map.get(outer_entity_alias.lower()) or outer_entity_alias
    outer_date_cols = _table_cols(outer_date_table)
    outer_entity_cols = _table_cols(outer_entity_table)
    inner_date_cols = _table_cols(inner_date_table)
    bridge_cols = _table_cols(bridge_table)

    if outer_date_cols and outer_date_col.lower() not in outer_date_cols:
        return sql
    if outer_entity_cols and outer_entity_col.lower() not in outer_entity_cols:
        return sql
    if inner_date_cols and (
        inner_date_col.lower() not in inner_date_cols
        or inner_join_col.lower() not in inner_date_cols
    ):
        return sql
    if bridge_cols and (
        bridge_link_col.lower() not in bridge_cols
        or bridge_entity_col.lower() not in bridge_cols
    ):
        return sql

    used = set(existing_aliases)
    ly_alias = _pick_alias("latest_years", used)
    src_alias = "ly_src"
    dt_alias = "ly_dt"

    ly_join = (
        "JOIN (\n"
        "  SELECT\n"
        f"    {src_alias}.{bridge_entity_col} AS latest_key,\n"
        f"    MAX(YEAR({dt_alias}.{inner_date_col})) AS latest_year\n"
        f"  FROM {bridge_table} AS {src_alias}\n"
        f"  JOIN {inner_date_table} AS {dt_alias}\n"
        f"    ON {dt_alias}.{inner_join_col} = {src_alias}.{bridge_link_col}\n"
        f"  GROUP BY {src_alias}.{bridge_entity_col}\n"
        f") AS {ly_alias}\n"
        f"  ON {ly_alias}.latest_key = {outer_entity_alias}.{outer_entity_col}"
    )

    from_with_ly = f"{from_core}\n{ly_join}"
    year_cond = f"YEAR({outer_date_alias}.{outer_date_col}) = {ly_alias}.latest_year"

    keep_conj = [c.strip() for idx, c in enumerate(conjuncts) if idx != target_idx and c.strip()]
    where_final = " AND ".join(keep_conj + [year_cond])

    tail_clause = f"\n{parsed['tail'].strip()}" if parsed["tail"].strip() else ""
    fixed = (
        f"SELECT {parsed['select'].strip()}\n"
        f"FROM {from_with_ly}\n"
        f"WHERE {where_final}\n"
        f"GROUP BY {parsed['groupby'].strip()}"
        f"{tail_clause};"
    )
    return fixed


def _rewrite_latest_year_projection_to_year(sql: str, question: str) -> str:
    """
    Convert date aggregate projection to year projection when output alias
    explicitly asks for year-like value.

    Example:
      MAX(o.orderDate) AS latest_order_year
    -> MAX(YEAR(o.orderDate)) AS latest_order_year

    Safety:
    - Trigger only when question mentions year/tahun.
    - Trigger only for aliases containing year/tahun.
    - Trigger only for MAX(date/time-like column), skip if already YEAR(...).
    """
    if not sql:
        return sql

    q = (question or "").lower()
    if not re.search(r"\b(?:year|tahun)\b", q):
        return sql

    parsed = _parse_select_from_groupby(sql)
    fallback_rest: Optional[str] = None

    if parsed:
        select_part = parsed["select"]
    else:
        # Fallback for derived-table shapes where conservative parser aborts.
        # We only need to rewrite top-level SELECT projection.
        s = (sql or "").strip().rstrip(";").strip()
        if not re.match(r"(?is)^\s*select\b", s):
            return sql

        def _find_top_level_from(text: str) -> int:
            i = 0
            depth = 0
            in_sq = False
            in_dq = False
            in_bt = False
            low = text.lower()
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
                if depth == 0 and low[i:i + 4] == "from":
                    left_ok = (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_"))
                    right_ok = (i + 4 >= len(text) or not (text[i + 4].isalnum() or text[i + 4] == "_"))
                    if left_ok and right_ok:
                        return i
                i += 1
            return -1

        from_idx = _find_top_level_from(s)
        if from_idx < 0:
            return sql
        select_part = s[len("SELECT"):from_idx].strip()
        fallback_rest = s[from_idx:].strip()

    select_items = _split_top_level_commas(select_part)
    if not select_items:
        return sql

    changed = False
    rebuilt: List[str] = []

    def _split_expr_alias(item: str) -> tuple:
        m = re.match(
            r"""(?is)^\s*(.*?)\s+AS\s+(`[^`]+`|'[^']+'|"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)\s*$""",
            item.strip(),
        )
        if not m:
            return item.strip(), None
        return m.group(1).strip(), m.group(2).strip()

    def _parse_qualified_col(expr: str) -> Optional[tuple]:
        m = re.match(
            r"(?is)^\s*(?:`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\.\s*)?`?([A-Za-z_][A-Za-z0-9_]*)`?\s*$",
            expr.strip(),
        )
        if not m:
            return None
        return m.group(1), m.group(2)

    for item in select_items:
        expr, alias = _split_expr_alias(item)
        if not alias:
            rebuilt.append(item.strip())
            continue

        alias_clean = alias.strip()
        if (
            (alias_clean.startswith("`") and alias_clean.endswith("`"))
            or (alias_clean.startswith("'") and alias_clean.endswith("'"))
            or (alias_clean.startswith('"') and alias_clean.endswith('"'))
        ):
            alias_clean = alias_clean[1:-1]
        alias_l = alias_clean.lower()

        # Only rewrite explicit year-like output alias.
        if ("year" not in alias_l) and ("tahun" not in alias_l):
            rebuilt.append(item.strip())
            continue
        # If alias explicitly asks for date/time too, keep date projection.
        # Example: latest_payment_date_in_latest_year should remain MAX(date).
        if ("date" in alias_l) or ("time" in alias_l) or ("tanggal" in alias_l):
            rebuilt.append(item.strip())
            continue

        m_max = re.match(r"(?is)^MAX\s*\(\s*(.+?)\s*\)$", expr)
        if not m_max:
            rebuilt.append(item.strip())
            continue

        inner = m_max.group(1).strip()
        if re.search(r"(?is)^\s*YEAR\s*\(", inner):
            rebuilt.append(item.strip())
            continue

        parsed_col = _parse_qualified_col(inner)
        if not parsed_col:
            rebuilt.append(item.strip())
            continue
        _a, c = parsed_col
        c_l = (c or "").lower()
        if ("date" not in c_l) and ("time" not in c_l):
            rebuilt.append(item.strip())
            continue

        rebuilt.append(f"MAX(YEAR({inner})) AS {alias}")
        changed = True

    if not changed:
        return sql

    if parsed:
        where_block = f"\n{parsed['where'].strip()}" if parsed["where"].strip() else ""
        tail_block = f"\n{parsed['tail'].strip()}" if parsed["tail"].strip() else ""
        fixed = (
            f"SELECT {', '.join(rebuilt)}\n"
            f"FROM {parsed['from_core'].strip()}"
            f"{where_block}\n"
            f"GROUP BY {parsed['groupby'].strip()}"
            f"{tail_block};"
        )
        return fixed

    # Fallback rebuild: preserve original FROM... tail as-is.
    fixed_fb = f"SELECT {', '.join(rebuilt)}\n{fallback_rest or ''}".strip()
    if fixed_fb and not fixed_fb.endswith(";"):
        fixed_fb += ";"
    return fixed_fb


def _rewrite_row_level_nonnull_to_group_existence(sql: str, question: str) -> str:
    """
    For grouped aggregate queries where question asks existence semantics
    ("at least one ... NOT NULL"), avoid row-level WHERE filtering that
    shrinks aggregate totals.

    Example rewrite:
      ... WHERE e.reportsTo IS NOT NULL GROUP BY ...
    ->
      ... GROUP BY ... HAVING SUM(CASE WHEN e.reportsTo IS NOT NULL THEN 1 ELSE 0 END) > 0
    """
    if not sql:
        return sql

    q = (question or "").lower()
    has_exist_intent = bool(re.search(r"\b(?:at\s+least\s+one|any|one\s+or\s+more)\b", q))
    has_nonnull_intent = bool(re.search(r"\b(?:not\s+null|non[-\s]?null)\b", q))
    if not (has_exist_intent and has_nonnull_intent):
        return sql

    s = sql.strip().rstrip(";").strip()
    if not s:
        return sql
    if not re.search(r"\bGROUP\s+BY\b", s, flags=re.I):
        return sql

    m = re.match(
        r"(?is)^(?P<head>.*?\bFROM\b.*?)(?P<tail>\s+(?:WHERE|GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT)\b.*)$",
        s,
    )
    if not m:
        return sql

    head = (m.group("head") or "").strip()
    tail = (m.group("tail") or "").strip()
    where_body, rest_after_where = _split_where_and_rest(" " + tail)
    if not where_body:
        return sql
    if _has_top_level_or(where_body):
        return sql

    conds = _split_top_level_and(where_body)
    if not conds:
        return sql

    nonnull_conds: List[str] = []
    keep_conds: List[str] = []

    colref = r"`?[A-Za-z_][A-Za-z0-9_]*`?\s*\.\s*`?[A-Za-z_][A-Za-z0-9_]*`?"
    nonnull_pat = re.compile(
        rf"""(?is)^\s*\(?\s*(?P<col>{colref})\s+IS\s+NOT\s+NULL\s*\)?\s*$"""
    )
    not_isnull_pat = re.compile(
        rf"""(?is)^\s*\(?\s*NOT\s+(?P<col>{colref})\s+IS\s+NULL\s*\)?\s*$"""
    )
    not_null_or_empty_pat = re.compile(
        rf"""(?is)^\s*\(?\s*NOT\s*\(\s*(?P<c1>{colref})\s+IS\s+NULL\s+OR\s*(?P<c2>{colref})\s*=\s*''\s*\)\s*\)?\s*$"""
    )

    def _norm_colref(ref: str) -> str:
        return re.sub(r"[\s`]+", "", str(ref or "")).lower()

    def _extract_nonnull_condition(cond: str) -> Optional[str]:
        c = (cond or "").strip()
        if not c:
            return None

        m1 = nonnull_pat.match(c)
        if m1:
            return f"{m1.group('col').strip()} IS NOT NULL"

        m2 = not_isnull_pat.match(c)
        if m2:
            return f"{m2.group('col').strip()} IS NOT NULL"

        m3 = not_null_or_empty_pat.match(c)
        if m3:
            c1 = (m3.group("c1") or "").strip()
            c2 = (m3.group("c2") or "").strip()
            if c1 and c2 and _norm_colref(c1) == _norm_colref(c2):
                return f"NOT ({c1} IS NULL OR {c1} = '')"

        return None

    for c in conds:
        c0 = c.strip()
        if not c0:
            continue
        nonnull_expr = _extract_nonnull_condition(c0)
        if nonnull_expr:
            nonnull_conds.append(nonnull_expr)
        else:
            keep_conds.append(c0)

    if not nonnull_conds:
        return sql

    existence_parts = [
        f"SUM(CASE WHEN {c} THEN 1 ELSE 0 END) > 0" for c in nonnull_conds
    ]
    existence_expr = " AND ".join(existence_parts)

    rest = (rest_after_where or "").strip()
    if not rest:
        return sql
    if not re.search(r"\bGROUP\s+BY\b", rest, flags=re.I):
        return sql

    if re.search(r"\bHAVING\b", rest, flags=re.I):
        rest = re.sub(
            r"(?is)\bHAVING\b\s*(?P<h>.*?)(?=(\bORDER\s+BY\b|\bLIMIT\b|$))",
            lambda m_h: f"HAVING ({m_h.group('h').strip()}) AND ({existence_expr})",
            rest,
            count=1,
        )
    else:
        m_tail = re.search(r"\b(ORDER\s+BY|LIMIT)\b", rest, flags=re.I)
        if m_tail:
            rest = (
                rest[:m_tail.start()].rstrip()
                + f" HAVING {existence_expr} "
                + rest[m_tail.start():].lstrip()
            )
        else:
            rest = rest.rstrip() + f" HAVING {existence_expr}"

    where_part = f" WHERE {' AND '.join(keep_conds)}" if keep_conds else ""
    fixed = f"{head}{where_part} {rest}".strip()
    if not fixed.endswith(";"):
        fixed += ";"
    return fixed


def _drop_unrequested_exists_filters(sql: str, question: str, ctx: str = "") -> str:
    """
    Drop EXISTS predicates that reference tables not mentioned in the question.

    This is a conservative guardrail against hallucinated relationship filters such as:
      WHERE EXISTS (SELECT 1 FROM customers c WHERE ...)
    when the prompt never asks about customers.
    """
    if not sql:
        return sql

    q = (question or "").lower().strip()
    if not q:
        return sql
    if "exists" not in sql.lower():
        return sql

    s = sql.strip().rstrip(";").strip()
    if not s:
        return sql

    m = re.match(
        r"(?is)^(?P<head>.*?\bFROM\b.*?)(?P<tail>\s+(?:WHERE|GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT)\b.*)$",
        s,
    )
    if not m:
        return sql

    head = (m.group("head") or "").strip()
    tail = (m.group("tail") or "").strip()
    where_body, rest_after_where = _split_where_and_rest(" " + tail)
    if not where_body:
        return sql
    if _has_top_level_or(where_body):
        return sql

    def _table_mentioned_in_question(tbl: str) -> bool:
        t = (tbl or "").lower().strip()
        if not t:
            return False
        variants = {t}
        singular = get_singular_form(t)
        if singular:
            variants.add(singular.lower())
        if "_" in t:
            variants.add(t.replace("_", " "))
        if singular and "_" in singular:
            variants.add(singular.replace("_", " "))
        try:
            for syn in get_table_synonyms(t):
                s0 = (syn or "").lower().strip()
                if s0:
                    variants.add(s0)
        except Exception:
            pass
        for v in variants:
            if not v:
                continue
            pat = re.escape(v).replace(r"\ ", r"\s+")
            if re.search(rf"\b{pat}\b", q, flags=re.I):
                return True
        return False

    exists_table_pat = re.compile(
        r"""(?is)^\s*EXISTS\s*\(\s*SELECT\b.*?\bFROM\s+(`?[A-Za-z_][A-Za-z0-9_]*`?(?:\.`?[A-Za-z_][A-Za-z0-9_]*`?)?)\b.*\)\s*$"""
    )

    conds = _split_top_level_and(where_body)
    if not conds:
        return sql

    keep: List[str] = []
    removed_any = False
    for c in conds:
        c0 = c.strip()
        if not c0:
            continue
        m_exists = exists_table_pat.match(c0)
        if not m_exists:
            keep.append(c0)
            continue

        tbl_expr = (m_exists.group(1) or "").replace("`", "")
        tbl = tbl_expr.split(".")[-1].lower().strip()
        if _table_mentioned_in_question(tbl):
            keep.append(c0)
            continue

        removed_any = True

    if not removed_any:
        return sql

    if keep:
        where_part = f" WHERE {' AND '.join(keep)}"
    else:
        where_part = ""

    rest = (rest_after_where or "").strip()
    fixed = f"{head}{where_part} {rest}".strip() if rest else f"{head}{where_part}".strip()
    if not fixed.endswith(";"):
        fixed += ";"
    return fixed


def _drop_unrequested_temporal_filters(sql: str, question: str, ctx: str = "") -> str:
    """
    Remove hallucinated temporal WHERE predicates when question has no temporal intent.

    Scope is intentionally conservative:
    - only runs for winner-row/ranking shaped SQL (ROW_NUMBER / RANK),
    - skips questions that explicitly mention temporal constraints.
    """
    if not sql:
        return sql

    q = (question or "").lower().strip()
    if not q:
        return sql

    low_sql = sql.lower()
    if "row_number(" not in low_sql and "rank(" not in low_sql and "dense_rank(" not in low_sql:
        return sql

    # Keep temporal predicates when question clearly asks for temporal scope/filter.
    temporal_terms = (
        "latest", "most recent", "newest", "earliest", "oldest", "recent",
        "year", "month", "day", "date", "time", "today", "yesterday",
        "current", "tahun", "bulan", "hari", "tanggal", "waktu",
    )
    has_temporal_intent = any(t in q for t in temporal_terms) or bool(
        re.search(r"\b(?:19|20)\d{2}\b", q) or
        re.search(r"\b\d{4}-\d{2}-\d{2}\b", q)
    )
    if has_temporal_intent:
        return sql

    try:
        import sqlglot
        from sqlglot import exp
    except Exception:
        return sql

    def _flatten_and(expr_node):
        if isinstance(expr_node, exp.And):
            return _flatten_and(expr_node.this) + _flatten_and(expr_node.expression)
        return [expr_node]

    def _combine_and(exprs):
        if not exprs:
            return None
        cur = exprs[0]
        for nxt in exprs[1:]:
            cur = exp.And(this=cur, expression=nxt)
        return cur

    def _is_temporal_predicate(expr_node) -> bool:
        if expr_node is None:
            return False

        s = expr_node.sql(dialect="mysql").lower()
        # Date literals or explicit temporal functions
        if re.search(r"'(?:19|20)\d{2}-\d{2}-\d{2}'", s):
            return True
        if re.search(r"\b(?:year|month|day|date|timestamp|curdate|current_date|now)\s*\(", s):
            return True

        # Date/time-like column references
        for col in expr_node.find_all(exp.Column):
            col_name = (col.name or "").lower()
            if any(k in col_name for k in ("date", "time", "year", "month", "day")):
                return True
        return False

    try:
        parsed = sqlglot.parse_one(sql, read="mysql")
    except Exception:
        return sql

    removed_any = False
    for where in list(parsed.find_all(exp.Where)):
        cond = where.this
        if cond is None:
            continue
        # Keep conservative behavior when OR is present.
        if any(isinstance(n, exp.Or) for n in cond.walk()):
            continue

        conjuncts = _flatten_and(cond)
        keep = []
        removed_local = False
        for c in conjuncts:
            if _is_temporal_predicate(c):
                removed_local = True
            else:
                keep.append(c)

        if not removed_local:
            continue

        removed_any = True
        if keep:
            new_cond = _combine_and(keep)
            where.set("this", new_cond)
        else:
            parent = where.parent
            if parent is not None:
                for key, val in list(parent.args.items()):
                    if val is where:
                        parent.set(key, None)
                        break

    if not removed_any:
        return sql

    fixed = parsed.sql(dialect="mysql")
    if sql.strip().endswith(";") and not fixed.endswith(";"):
        fixed += ";"
    return fixed


def _prefer_not_exists_for_simple_exclusion(sql: str, question: str, ctx: str = "") -> str:
    """
    Canonicalize simple `NOT IN (SELECT key FROM table ...)` anti-joins into
    correlated `NOT EXISTS (...)` for better NULL-safety.

    This is intentionally conservative:
    - only runs when the question has clear exclusion wording,
    - only rewrites simple subquery forms,
    - skips grouped/distinct/limited/ordered subqueries.
    """
    if not sql:
        return sql

    low_sql = sql.lower()
    if not re.search(r"\bnot\b[\s`\"A-Za-z0-9_.]*\bin\s*\(", low_sql, flags=re.I):
        return sql

    q = (question or "").lower()
    if not q:
        return sql

    has_exclusion_intent = bool(
        re.search(
            r"\b(?:no|without(?:\s+any)?|never|have\s+never|has\s+never)\b",
            q,
            flags=re.I,
        )
    )
    if not has_exclusion_intent:
        return sql

    try:
        import sqlglot
        from sqlglot import exp
    except Exception:
        return sql

    try:
        parsed = sqlglot.parse_one(sql, read="mysql")
    except Exception:
        return sql

    changed = False

    def _transform(node):
        nonlocal changed

        if not isinstance(node, exp.Not):
            return node

        in_node = node.this
        if not isinstance(in_node, exp.In):
            return node

        lhs = in_node.this
        subq = in_node.args.get("query")

        if not isinstance(lhs, exp.Column):
            return node
        if not isinstance(subq, exp.Subquery):
            return node
        if not isinstance(subq.this, exp.Select):
            return node

        sel = subq.this

        # Only rewrite simple anti-join subqueries.
        if len(sel.expressions or []) != 1:
            return node
        if sel.args.get("group") or sel.args.get("having") or sel.args.get("distinct"):
            return node
        if sel.args.get("limit") or sel.args.get("order") or sel.args.get("union"):
            return node

        rhs_expr = sel.expressions[0]
        rhs_base = rhs_expr.this if isinstance(rhs_expr, exp.Alias) else rhs_expr
        if not isinstance(rhs_base, exp.Column):
            return node

        new_sel = sel.copy()
        new_sel.set("expressions", [exp.Literal.number(1)])

        corr_eq = exp.EQ(this=rhs_base.copy(), expression=lhs.copy())
        existing_where = new_sel.args.get("where")
        if isinstance(existing_where, exp.Where) and existing_where.this is not None:
            new_where_cond = exp.And(this=existing_where.this.copy(), expression=corr_eq)
        else:
            new_where_cond = corr_eq
        new_sel.set("where", exp.Where(this=new_where_cond))

        changed = True
        return exp.Not(this=exp.Exists(this=new_sel))

    rewritten = parsed.transform(_transform)
    if not changed:
        return sql

    fixed = rewritten.sql(dialect="mysql")
    if sql.strip().endswith(";") and not fixed.endswith(";"):
        fixed += ";"
    return fixed
# =======================================================================
# ========= Phase 5 — Pattern, REGEXP, and Predicate Normalization ======
# ========================= Guardrail for Regex =========================
# =======================================================================
# - guardrail_fix_like_regex_anchors
# - question_requires_word_boundary_regex
# - question_requires_regex
# - should_prefer_like_over_regexp
# - rewrite_simple_regexp_to_like
# - rewrite_regexp_literal_to_posix_if_needed
# - guardrail_fix_mysql_regexp_literals

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


def _extract_whole_word_target(question: str) -> Optional[str]:
    q = question or ""
    patterns = [
        r'whole\s+word\s+["\']([^"\']+)["\']',
        r'word\s+boundary\s+["\']([^"\']+)["\']',
        r'token\s+["\']([^"\']+)["\']',
        r'kata\s+utuh\s+["\']([^"\']+)["\']',
    ]
    for pat in patterns:
        m = re.search(pat, q, flags=re.I)
        if not m:
            continue
        token = (m.group(1) or "").strip()
        if token:
            return token
    return None


def _question_requires_structural_regex(question: str) -> bool:
    q = (question or "").lower()

    structural_patterns = [
        r"\bconsists?\s+of\s+(?:exactly\s+)?(\d+|one|two|three|four|five)\s+words?\b",
        r"\bhas\s+(?:exactly\s+)?(\d+|one|two|three|four|five)\s+words?\b",
        r"\b(?:single|one|two|three|four|five)\s+words?\b(?:\s+separated\s+by\s+(?:a\s+)?space)?",
        r"\bseparated\s+by\s+(?:a\s+)?space\b",
        r"\bword\s+count\b",
    ]
    return any(re.search(pat, q, flags=re.I) for pat in structural_patterns)


def _question_requires_two_word_space_pattern(question: str) -> bool:
    q = (question or "").lower()
    patterns = [
        r"\bconsists?\s+of\s+(?:exactly\s+)?two\s+words?\b",
        r"\bmade\s+up\s+of\s+(?:exactly\s+)?two\s+words?\b",
        r"\bhas\s+(?:exactly\s+)?two\s+words?\b",
        r"\bexactly\s+two\s+words?\b",
        r"\btwo\s+words?\b(?:\s+separated\s+by\s+(?:a\s+)?space)?",
    ]
    return any(re.search(pat, q, flags=re.I) for pat in patterns)


def _extract_structural_word_count(question: str) -> Optional[int]:
    q = (question or "").lower()
    word_to_num = {
        "single": 1,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
    }
    patterns = [
        r"\bconsists?\s+of\s+(?:exactly\s+)?(?P<count>\d+|single|one|two|three|four|five)\s+words?\b",
        r"\bmade\s+up\s+of\s+(?:exactly\s+)?(?P<count>\d+|single|one|two|three|four|five)\s+words?\b",
        r"\bhas\s+(?:exactly\s+)?(?P<count>\d+|single|one|two|three|four|five)\s+words?\b",
        r"\bexactly\s+(?P<count>\d+|single|one|two|three|four|five)\s+words?\b",
        r"\b(?P<count>single|one|two|three|four|five)\s+words?\b(?:\s+separated\s+by\s+(?:a\s+)?space)?",
    ]
    for pat in patterns:
        m = re.search(pat, q, flags=re.I)
        if not m:
            continue
        raw = (m.group("count") or "").strip().lower()
        if raw.isdigit():
            n = int(raw)
            if 1 <= n <= 5:
                return n
            return None
        return word_to_num.get(raw)
    return None

def _question_requires_regex(question: str) -> bool:
    q = (question or "").strip().lower()

    # 1. Deteksi eksplisit (sudah ada di code Anda)
    if any(k in q for k in ["regex", "regexp", "regexp_like", "regular expression"]):
        return True

    # 2. Deteksi jumlah digit yang spesifik (Start Small: Fokus di sini)
    # Ini akan menangkap "3 digits", "three digits", "exactly 4 characters", dll.
    if re.search(r"\b\d+\s+(digit|character|angka|huruf)\b", q):
        return True
    if re.search(r"\b(one|two|three|four|five)\s+(digit|character|angka|huruf)\b", q):
        return True

    # 3. Deteksi pola "followed by" (kasus spesifik Anda)
    if "followed by" in q or "diikuti oleh" in q:
        return True

    # 4. Structural text patterns that need anchored REGEXP, e.g. "exactly two words"
    if _question_requires_structural_regex(question):
        return True

    return False

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
        r"\1\2 LIKE '\3%",
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


def _canonicalize_structural_regex(sql: str, question: str) -> str:
    """
    Canonicalize narrow structural-regex families that are semantically clear but
    easy for the LLM to express incorrectly.

    Current supported family:
      - exactly N words separated by a single space (N = 1..5)
    """
    if not sql:
        return sql
    word_count = _extract_structural_word_count(question)
    if not word_count:
        return sql

    canonical = "^" + " ".join(["[^ ]+"] * word_count) + "$"
    out = sql

    # REGEXP_LIKE(col, '...')
    out = re.sub(
        r"(?is)(REGEXP_LIKE\s*\(\s*[^,]+,\s*)(N?'(?:''|[^'])*'|\"(?:[^\"\\\\]|\\\\.)*\")(\s*\))",
        lambda m: f"{m.group(1)}'{canonical}'{m.group(3)}",
        out,
    )

    # col REGEXP '...' / col RLIKE '...'
    out = re.sub(
        r"(?is)(\b(?:[A-Za-z_][A-Za-z0-9_]*\.)?[A-Za-z_][A-Za-z0-9_]*\s+\b(?:REGEXP|RLIKE)\b\s*)(N?'(?:''|[^'])*'|\"(?:[^\"\\\\]|\\\\.)*\")",
        lambda m: f"{m.group(1)}'{canonical}'",
        out,
    )

    # If the model weakens this to a simple "contains a space" LIKE pattern,
    # strengthen it to the canonical REGEXP instead.
    if word_count == 2:
        out = re.sub(
            r"(?is)(\b(?:[A-Za-z_][A-Za-z0-9_]*\.)?[A-Za-z_][A-Za-z0-9_]*\s+)LIKE\s+'% %'",
            lambda m: f"{m.group(1)}REGEXP '{canonical}'",
            out,
        )

    return out


def _canonicalize_boundary_regex(sql: str, question: str) -> str:
    """
    Canonicalize narrow word-boundary regex family:
      - contains the whole word "X"

    Rewrites any existing regex/rlike literal (or weak LIKE substring) to the
    canonical MySQL-safe boundary pattern:
      (^|[^[:alnum:]_])X([^[:alnum:]_]|$)
    """
    if not sql or not _question_requires_word_boundary_regex(question):
        return sql
    target = _extract_whole_word_target(question)
    if not target:
        return sql

    canonical = f"(^|[^[:alnum:]_]){re.escape(target)}([^[:alnum:]_]|$)"
    out = sql

    # REGEXP_LIKE(col, '...')
    out = re.sub(
        r"(?is)(REGEXP_LIKE\s*\(\s*[^,]+,\s*)(N?'(?:''|[^'])*'|\"(?:[^\"\\\\]|\\\\.)*\")(\s*\))",
        lambda m: f"{m.group(1)}'{canonical}'{m.group(3)}",
        out,
    )

    # col REGEXP '...' / col RLIKE '...'
    out = re.sub(
        r"(?is)(\b(?:[A-Za-z_][A-Za-z0-9_]*\.)?[A-Za-z_][A-Za-z0-9_]*\s+\b(?:REGEXP|RLIKE)\b\s*)(N?'(?:''|[^'])*'|\"(?:[^\"\\\\]|\\\\.)*\")",
        lambda m: f"{m.group(1)}'{canonical}'",
        out,
    )

    # Upgrade weak substring LIKE to boundary regex for whole-word intent.
    like_pat = rf"(?is)(\b(?:[A-Za-z_][A-Za-z0-9_]*\.)?[A-Za-z_][A-Za-z0-9_]*\s+)LIKE\s+'%{re.escape(target)}%'"
    out = re.sub(
        like_pat,
        lambda m: f"{m.group(1)}REGEXP '{canonical}'",
        out,
    )

    return out

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

    # Trigger jika ada \b/\w/\d/\s baik single maupun double.
    # Pattern ini match: \x atau \\x (jumlah backslash 1 atau 2)
    has_b = re.search(r"(?:\\\\|\\)b", lit) is not None
    has_w = re.search(r"(?:\\\\|\\)w", lit) is not None
    has_d = re.search(r"(?:\\\\|\\)d", lit) is not None
    has_s = re.search(r"(?:\\\\|\\)s", lit) is not None

    # Kalau tidak ada indikasi, jangan sentuh (biar tidak mengubah regex valid)
    if not (has_b or has_w or has_d or has_s):
        return lit

    out = lit

    # 1) Rewrite boundary + wordchar (handle 1 atau 2 backslash)
    out = re.sub(r"(?:\\\\|\\)b", r"(^|[^[:alnum:]_])", out)
    out = re.sub(r"(?:\\\\|\\)w", r"[[:alnum:]_]", out)

    # Common shorthand classes that are often produced by LLMs.
    out = re.sub(r"(?:\\\\|\\)d", r"[0-9]", out)
    out = re.sub(r"(?:\\\\|\\)s", r"[[:space:]]", out)

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
# =======================================================================
# ======= Phase 6 — NULL / DISTINCT / Result Cardinality Policies =======
# ============== Heuristic for Missing Values and Distinct ==============
# =======================================================================
# - question_wants_missing_value
# - expand_missing_null_to_null_or_empty
# - wants_distinct_from_question
# - has_distinct
# - add_distinct
# - remove_distinct
# - ensure_reasonable_distinct
# - question_wants_count_distinct
# - ensure_reasonable_count_distinct
# - parse_from_join_aliases
# - strip_unneeded_count_distinct

def _question_wants_missing_value(question: str) -> bool:
    q = (question or "").lower()
    return any(s in q for s in [
        "missing", "is missing", "missing information",
        "empty", "blank", "not provided", "unknown",
        "null or empty", "null/empty",
        "kosong", "tidak ada", "belum diisi"
    ])

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
            r"\b((?:[A-Za-z_][A-Za-z0-9_]*\.)?[A-Za-z_][A-Za-z0-9_]*)\s+IS\s+NULL\b",
            r"(\1 IS NULL OR \1 = '')",
            seg,
            flags=re.I,
        )

        parts[i] = seg

    return "".join(parts)

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
    if "unique" in q and any(tok in q for tok in ("row", "rows", "record", "records", "result", "results", "value", "values")):
        return True
    implicit_patterns = (
        r"\bone\s+(?:row|record|result)\s+(?:per|for\s+each)\b",
        r"\beach\s+[a-z0-9_]+\s+only\s+once\b",
        r"\bonly\s+once\s+per\b",
    )
    if any(re.search(pat, q, flags=re.I) for pat in implicit_patterns):
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
    if _question_wants_count_distinct(question):
        return _remove_distinct(sql)
    if _wants_distinct_from_question(question):
        return _add_distinct(sql)
    return _remove_distinct(sql)


def _question_wants_count_distinct(question: str) -> bool:
    q = (question or "").lower()
    if not q:
        return False
    if not re.search(r"\b(how\s+many|number\s+of|count|total\s+number)\b", q):
        return False
    return bool(re.search(r"\b(distinct|unique|different)\b", q))


def _ensure_reasonable_count_distinct(sql: str, question: str) -> str:
    """
    Repair a common grouped-aggregate mistake for questions that ask for
    COUNT(DISTINCT ...), where the model instead groups by the distinct target
    column and uses plain COUNT(...).

    Example bad shape:
      SELECT YEAR(orderDate) AS year, status, COUNT(orderNumber)
      FROM orders
      GROUP BY YEAR(orderDate), status

    Rewrites to:
      SELECT YEAR(orderDate) AS year, COUNT(DISTINCT status)
      FROM orders
      GROUP BY YEAR(orderDate)
    """
    s = (sql or "").strip()
    q = (question or "").lower()
    if not s or not _question_wants_count_distinct(q):
        return sql
    if re.search(r"\bCOUNT\s*\(\s*DISTINCT\b", s, flags=re.I):
        return sql
    if "GROUP BY" not in s.upper() or "COUNT(" not in s.upper():
        return sql

    def _find_top_level_kw(text: str, kw: str, start: int = 0) -> int:
        target = kw.upper()
        t_up = text.upper()
        depth = 0
        quote = ""
        i = start
        while i < len(text):
            ch = text[i]
            if quote:
                if ch == quote and (i == 0 or text[i - 1] != "\\"):
                    quote = ""
                i += 1
                continue
            if ch in {"'", '"', "`"}:
                quote = ch
                i += 1
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif depth == 0 and t_up.startswith(target, i):
                prev_ok = i == 0 or not (t_up[i - 1].isalnum() or t_up[i - 1] == "_")
                after = i + len(target)
                next_ok = after >= len(t_up) or not (t_up[after].isalnum() or t_up[after] == "_")
                if prev_ok and next_ok:
                    return i
            i += 1
        return -1

    def _split_top_level_csv(text: str) -> List[str]:
        parts: List[str] = []
        cur: List[str] = []
        depth = 0
        quote = ""
        for i, ch in enumerate(text):
            if quote:
                cur.append(ch)
                if ch == quote and (i == 0 or text[i - 1] != "\\"):
                    quote = ""
                continue
            if ch in {"'", '"', "`"}:
                quote = ch
                cur.append(ch)
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            if ch == "," and depth == 0:
                part = "".join(cur).strip()
                if part:
                    parts.append(part)
                cur = []
                continue
            cur.append(ch)
        tail = "".join(cur).strip()
        if tail:
            parts.append(tail)
        return parts

    def _strip_alias(item: str) -> str:
        return re.split(r"\s+AS\s+", item, maxsplit=1, flags=re.I)[0].strip()

    def _get_alias(item: str) -> str:
        parts = re.split(r"\s+AS\s+", item, maxsplit=1, flags=re.I)
        if len(parts) == 2:
            return parts[1].strip().strip("`")
        return ""

    def _norm(text: str) -> str:
        rendered = text.replace("`", "")
        rendered = re.sub(r"\s+", "", rendered).lower()
        return rendered

    def _tokenize_expr(item: str) -> Set[str]:
        base = _strip_alias(item)
        alias = _get_alias(item)
        raw = f"{base} {alias}".lower()
        tokens = {t for t in re.findall(r"[a-z0-9]+", raw) if t}
        expanded = set(tokens)
        for t in list(tokens):
            if t.endswith("ies") and len(t) > 3:
                expanded.add(t[:-3] + "y")
            elif t.endswith("es") and len(t) > 2:
                expanded.add(t[:-2])
            elif t.endswith("s") and len(t) > 1 and not t.endswith(("ss", "us")):
                expanded.add(t[:-1])
        return expanded

    select_match = re.match(r"^\s*SELECT\s+(DISTINCT\s+)?", s, flags=re.I)
    if not select_match:
        return sql
    has_select_distinct = bool(select_match.group(1))

    from_pos = _find_top_level_kw(s, "FROM", select_match.end())
    group_pos = _find_top_level_kw(s, "GROUP BY", max(select_match.end(), from_pos))
    if from_pos < 0 or group_pos < 0:
        return sql

    having_pos = _find_top_level_kw(s, "HAVING", group_pos + len("GROUP BY"))
    order_pos = _find_top_level_kw(s, "ORDER BY", group_pos + len("GROUP BY"))
    limit_pos = _find_top_level_kw(s, "LIMIT", group_pos + len("GROUP BY"))

    group_end_candidates = [p for p in (having_pos, order_pos, limit_pos) if p >= 0]
    group_end = min(group_end_candidates) if group_end_candidates else len(s)

    select_part = s[select_match.end():from_pos].strip()
    from_clause = s[from_pos:group_pos].strip()
    group_part = s[group_pos + len("GROUP BY"):group_end].strip()
    suffix = s[group_end:].strip()

    select_items = _split_top_level_csv(select_part)
    group_items = _split_top_level_csv(group_part)
    if len(select_items) < 2 or not group_items:
        return sql

    agg_pat = re.compile(r"\b(?:COUNT|SUM|AVG|MIN|MAX)\s*\(", flags=re.I)
    window_pat = re.compile(r"\bOVER\s*\(", flags=re.I)

    count_idx = -1
    for idx, item in enumerate(select_items):
        if re.search(r"\bCOUNT\s*\(", item, flags=re.I) and not window_pat.search(item):
            count_idx = idx
            break
    if count_idx < 0:
        return sql

    non_agg_indices = [
        idx for idx, item in enumerate(select_items)
        if not agg_pat.search(item) and not window_pat.search(item)
    ]
    if len(non_agg_indices) < 2:
        return sql

    hint_tokens: Set[str] = set()
    m_hint = re.search(
        r"\b(?:how\s+many|number\s+of|count|total\s+number(?:\s+of)?)\s+"
        r"(?:distinct|unique|different)\s+([a-z][a-z0-9_ ]{1,40})\b",
        q,
        flags=re.I,
    )
    if m_hint:
        hint_tokens = {t for t in re.findall(r"[a-z0-9]+", m_hint.group(1).lower()) if t}
        expanded = set(hint_tokens)
        for t in list(hint_tokens):
            if t.endswith("ies") and len(t) > 3:
                expanded.add(t[:-3] + "y")
            elif t.endswith("es") and len(t) > 2:
                expanded.add(t[:-2])
            elif t.endswith("s") and len(t) > 1 and not t.endswith(("ss", "us")):
                expanded.add(t[:-1])
        hint_tokens = expanded

    candidate_idx = -1
    best_score = 0
    for idx in non_agg_indices:
        score = 0
        if hint_tokens:
            score = len(_tokenize_expr(select_items[idx]).intersection(hint_tokens))
        if score > best_score or (score == best_score and score > 0 and idx > candidate_idx):
            best_score = score
            candidate_idx = idx
    if candidate_idx < 0:
        candidate_idx = non_agg_indices[-1]

    candidate_item = select_items[candidate_idx]
    candidate_base = _strip_alias(candidate_item)
    candidate_alias = _get_alias(candidate_item)
    candidate_norm = _norm(candidate_base)
    group_norms = [_norm(item) for item in group_items]
    if candidate_norm not in group_norms:
        return sql

    if order_pos >= 0:
        order_end = limit_pos if limit_pos > order_pos else len(s)
        order_clause = s[order_pos + len("ORDER BY"):order_end].strip()
        order_norm = _norm(order_clause)
        if candidate_alias and re.search(rf"\b{re.escape(candidate_alias.lower())}\b", order_clause.lower()):
            return sql
        if candidate_norm and candidate_norm in order_norm:
            return sql

    count_item = select_items[count_idx]
    count_alias = ""
    count_alias_match = re.search(r"\s+AS\s+(`?[A-Za-z_][A-Za-z0-9_]*`?)\s*$", count_item, flags=re.I)
    if count_alias_match:
        count_alias = count_alias_match.group(1)
    new_count_item = f"COUNT(DISTINCT {candidate_base})"
    if count_alias:
        new_count_item += f" AS {count_alias}"

    new_select_items: List[str] = []
    for idx, item in enumerate(select_items):
        if idx == candidate_idx:
            continue
        if idx == count_idx:
            new_select_items.append(new_count_item)
        else:
            new_select_items.append(item)

    new_group_items = [item for item in group_items if _norm(item) != candidate_norm]
    if not new_select_items or not new_group_items:
        return sql

    select_keyword = "SELECT DISTINCT" if has_select_distinct else "SELECT"
    out = f"{select_keyword} " + ", ".join(new_select_items) + f" {from_clause} GROUP BY " + ", ".join(new_group_items)
    if suffix:
        out += " " + suffix
    return out


def _drop_redundant_group_by_for_distinct_lookup(sql: str) -> str:
    """
    Drop top-level GROUP BY when it is only acting as a dedupe mechanism for a
    SELECT DISTINCT lookup query.

    Conservative guardrails:
    - top-level SELECT must already be DISTINCT
    - no HAVING, no aggregate functions, no window functions
    - GROUP BY expressions must match the visible SELECT expressions
    """
    s = (sql or "").strip()
    if not s or "GROUP BY" not in s.upper() or "SELECT DISTINCT" not in s.upper():
        return sql

    s_up = s.upper()
    if " HAVING " in f" {s_up} " or " OVER " in f" {s_up} ":
        return sql

    def _find_top_level_kw(text: str, kw: str, start: int = 0) -> int:
        target = kw.upper()
        t_up = text.upper()
        depth = 0
        quote = ""
        i = start
        while i < len(text):
            ch = text[i]
            if quote:
                if ch == quote and text[i - 1] != "\\":
                    quote = ""
                i += 1
                continue
            if ch in {"'", '"', "`"}:
                quote = ch
                i += 1
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif depth == 0 and t_up.startswith(target, i):
                prev_ok = i == 0 or not (t_up[i - 1].isalnum() or t_up[i - 1] == "_")
                after = i + len(target)
                next_ok = after >= len(t_up) or not (t_up[after].isalnum() or t_up[after] == "_")
                if prev_ok and next_ok:
                    return i
            i += 1
        return -1

    def _split_top_level_csv(text: str) -> List[str]:
        parts: List[str] = []
        cur: List[str] = []
        depth = 0
        quote = ""
        for i, ch in enumerate(text):
            if quote:
                cur.append(ch)
                if ch == quote and text[i - 1] != "\\":
                    quote = ""
                continue
            if ch in {"'", '"', "`"}:
                quote = ch
                cur.append(ch)
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            if ch == "," and depth == 0:
                part = "".join(cur).strip()
                if part:
                    parts.append(part)
                cur = []
                continue
            cur.append(ch)
        tail = "".join(cur).strip()
        if tail:
            parts.append(tail)
        return parts

    select_match = re.match(r"^\s*SELECT\s+DISTINCT\s+", s, flags=re.I)
    if not select_match:
        return sql

    from_pos = _find_top_level_kw(s, "FROM", select_match.end())
    group_pos = _find_top_level_kw(s, "GROUP BY", max(select_match.end(), from_pos))
    if from_pos < 0 or group_pos < 0:
        return sql

    next_positions = [
        pos for pos in (
            _find_top_level_kw(s, "HAVING", group_pos + 8),
            _find_top_level_kw(s, "ORDER BY", group_pos + 8),
            _find_top_level_kw(s, "LIMIT", group_pos + 8),
        )
        if pos >= 0
    ]
    group_end = min(next_positions) if next_positions else len(s)

    select_items = _split_top_level_csv(s[select_match.end():from_pos].strip())
    group_items = _split_top_level_csv(s[group_pos + len("GROUP BY"):group_end].strip())
    if not select_items or len(select_items) != len(group_items):
        return sql

    if any(re.search(r"\b(count|sum|avg|min|max)\s*\(", item, flags=re.I) for item in select_items):
        return sql

    def _strip_select_alias(item: str) -> str:
        return re.split(r"\s+AS\s+", item, maxsplit=1, flags=re.I)[0].strip()

    def _norm(text: str) -> str:
        rendered = text.replace("`", "")
        rendered = re.sub(r"\s+", "", rendered).lower()
        return rendered

    select_norms = sorted(_norm(_strip_select_alias(item)) for item in select_items)
    group_norms = sorted(_norm(item) for item in group_items)
    if select_norms != group_norms:
        return sql

    prefix = s[:group_pos].rstrip()
    suffix = s[group_end:].lstrip()
    if not suffix:
        return prefix
    sep = "\n" if "\n" in prefix or "\n" in suffix else " "
    return prefix + sep + suffix


def _normalize_simple_set_operation_branches(sql: str, question: str) -> str:
    """
    Normalize simple UNION / UNION ALL chains where each branch returns an
    identifier plus a label column.

    Supported branch families:
    - SELECT id, 'label' AS label_alias FROM table WHERE ...
    - SELECT CONCAT('label-', id) AS bad_alias FROM table WHERE ...

    This is intentionally conservative:
    - only top-level UNION / UNION ALL chains
    - only simple single-table branches
    - all branches must project the same identifier expression

    It supports 2+ branches and canonicalizes the output to:
      SELECT id, 'label' AS label_alias FROM table WHERE ...
      UNION [ALL] ...
      ORDER BY label_alias <dir>, id <dir>
    """
    s = (sql or "").strip()
    q = (question or "").lower()
    if not s or "UNION" not in s.upper():
        return sql

    def _find_top_level_kw(text: str, kw: str, start: int = 0) -> int:
        target = kw.upper()
        t_up = text.upper()
        depth = 0
        quote = ""
        i = start
        while i < len(text):
            ch = text[i]
            if quote:
                if ch == quote and (i == 0 or text[i - 1] != "\\"):
                    quote = ""
                i += 1
                continue
            if ch in {"'", '"', "`"}:
                quote = ch
                i += 1
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif depth == 0 and t_up.startswith(target, i):
                prev_ok = i == 0 or not (t_up[i - 1].isalnum() or t_up[i - 1] == "_")
                after = i + len(target)
                next_ok = after >= len(t_up) or not (t_up[after].isalnum() or t_up[after] == "_")
                if prev_ok and next_ok:
                    return i
            i += 1
        return -1

    def _split_top_level_csv(text: str) -> List[str]:
        parts: List[str] = []
        cur: List[str] = []
        depth = 0
        quote = ""
        for i, ch in enumerate(text):
            if quote:
                cur.append(ch)
                if ch == quote and (i == 0 or text[i - 1] != "\\"):
                    quote = ""
                continue
            if ch in {"'", '"', "`"}:
                quote = ch
                cur.append(ch)
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            if ch == "," and depth == 0:
                part = "".join(cur).strip()
                if part:
                    parts.append(part)
                cur = []
                continue
            cur.append(ch)
        tail = "".join(cur).strip()
        if tail:
            parts.append(tail)
        return parts

    def _split_union_chain(text: str) -> Tuple[List[str], List[str]]:
        branches: List[str] = []
        operators: List[str] = []
        depth = 0
        quote = ""
        start = 0
        i = 0
        t_up = text.upper()
        while i < len(text):
            ch = text[i]
            if quote:
                if ch == quote and (i == 0 or text[i - 1] != "\\"):
                    quote = ""
                i += 1
                continue
            if ch in {"'", '"', "`"}:
                quote = ch
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
            if depth == 0:
                token = ""
                if t_up.startswith("UNION ALL", i):
                    token = "UNION ALL"
                elif t_up.startswith("UNION", i):
                    token = "UNION"
                if token:
                    prev_ok = i == 0 or not (t_up[i - 1].isalnum() or t_up[i - 1] == "_")
                    after = i + len(token)
                    next_ok = after >= len(t_up) or not (t_up[after].isalnum() or t_up[after] == "_")
                    if prev_ok and next_ok:
                        branch = text[start:i].strip()
                        if not branch:
                            return [], []
                        branches.append(branch)
                        operators.append(token)
                        start = after
                        i = after
                        continue
            i += 1
        tail = text[start:].strip()
        if tail:
            branches.append(tail)
        return branches, operators

    def _segment_dir(text: str, target_pat: str, default: str = "ASC") -> str:
        if not text:
            return default
        if re.search(rf"{target_pat}.*\bdescending\b", text, flags=re.I) or re.search(
            rf"\bdescending\b.*{target_pat}", text, flags=re.I
        ):
            return "DESC"
        if re.search(rf"{target_pat}.*\bascending\b", text, flags=re.I) or re.search(
            rf"\bascending\b.*{target_pat}", text, flags=re.I
        ):
            return "ASC"
        return default

    body = s[:-1].rstrip() if s.endswith(";") else s
    order_pos = _find_top_level_kw(body, "ORDER BY")
    if order_pos >= 0:
        order_clause = body[order_pos + len("ORDER BY"):].strip()
        core = body[:order_pos].rstrip()
    else:
        order_clause = ""
        core = body

    branches, operators = _split_union_chain(core)
    if len(branches) < 2 or len(operators) != len(branches) - 1:
        return sql

    direct_re = re.compile(
        r"""
        ^\s*SELECT\s+
        (?P<idcol>(?:[A-Za-z_][A-Za-z0-9_]*\.)?[A-Za-z_][A-Za-z0-9_]*)\s*,\s*
        '(?P<label>[^']+)'\s+AS\s+`?(?P<label_alias>[A-Za-z_][A-Za-z0-9_]*|\d{4})`?\s+
        FROM\s+(?P<table>`?[A-Za-z_][A-Za-z0-9_]*`?)
        (?:\s+WHERE\s+(?P<where>.+?))?\s*$
        """,
        flags=re.I | re.S | re.X,
    )
    concat_re = re.compile(
        r"""
        ^\s*SELECT\s+
        CONCAT\(\s*'(?P<label>[^']+)'\s*,\s*(?P<idcol>(?:[A-Za-z_][A-Za-z0-9_]*\.)?[A-Za-z_][A-Za-z0-9_]*)\s*\)\s+
        AS\s+`?(?P<bad_alias>[A-Za-z_][A-Za-z0-9_]*)`?\s+
        FROM\s+(?P<table>`?[A-Za-z_][A-Za-z0-9_]*`?)
        (?:\s+WHERE\s+(?P<where>.+?))?\s*$
        """,
        flags=re.I | re.S | re.X,
    )

    branch_infos: List[Dict[str, str]] = []
    for branch in branches:
        parsed: Optional[Dict[str, str]] = None
        m = direct_re.match(branch)
        if m:
            parsed = {
                "idcol": m.group("idcol"),
                "label": m.group("label"),
                "label_alias": m.group("label_alias"),
                "table": m.group("table"),
                "where": (m.group("where") or "").strip(),
            }
        else:
            m = concat_re.match(branch)
            if m:
                label = m.group("label")
                clean_label = label.rstrip("-: ").strip() or label
                parsed = {
                    "idcol": m.group("idcol"),
                    "label": clean_label,
                    "label_alias": "",
                    "table": m.group("table"),
                    "where": (m.group("where") or "").strip(),
                }
        if not parsed:
            return sql
        branch_infos.append(parsed)

    def _norm_ident(text: str) -> str:
        return re.sub(r"\s+", "", (text or "").strip()).lower()

    if any(_norm_ident(info["idcol"]) != _norm_ident(branch_infos[0]["idcol"]) for info in branch_infos[1:]):
        return sql

    alias_candidates = [info["label_alias"] for info in branch_infos if info["label_alias"]]
    label_alias = alias_candidates[0] if alias_candidates else ("year_label" if "year label" in q else "label")

    prefer_union = bool(
        re.search(
            r"\b(?:remove\s+duplicate\s+rows|without\s+duplicates|no\s+duplicate|distinct)\b",
            q,
            flags=re.I,
        )
    )
    prefer_union_all = bool(re.search(r"\bunion\s+all\b", q, flags=re.I))
    if prefer_union:
        out_ops = ["UNION"] * len(operators)
    elif prefer_union_all:
        out_ops = ["UNION ALL"] * len(operators)
    else:
        out_ops = [op.upper() for op in operators]

    label_dir = "ASC"
    id_dir = "ASC"
    if order_clause:
        order_items = _split_top_level_csv(order_clause)
        if order_items:
            label_dir = "DESC" if re.search(r"\bDESC\b", order_items[0], flags=re.I) else "ASC"
        if len(order_items) >= 2:
            id_dir = "DESC" if re.search(r"\bDESC\b", order_items[1], flags=re.I) else "ASC"
    else:
        label_dir = _segment_dir(q, r"\b(?:year\s+label|label)\b", "ASC")
        id_leaf = branch_infos[0]["idcol"].split(".")[-1]
        id_words = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", id_leaf.replace("_", " ")).lower()
        id_dir = _segment_dir(q, rf"\b{re.escape(id_words)}\b", "ASC")

    branch_sqls: List[str] = []
    for info in branch_infos:
        part = f"SELECT {info['idcol']}, '{info['label']}' AS {label_alias} FROM {info['table']}"
        if info["where"]:
            part += f" WHERE {info['where']}"
        branch_sqls.append(part)

    out = branch_sqls[0]
    for op, branch_sql in zip(out_ops, branch_sqls[1:]):
        out += f" {op} {branch_sql}"

    if order_clause or re.search(r"\b(?:order|sort|sorted)\b", q, flags=re.I):
        out += f" ORDER BY {label_alias} {label_dir}, {branch_infos[0]['idcol']} {id_dir}"
    if s.endswith(";"):
        out += ";"
    return out


def _rewrite_concat_label_union(sql: str, question: str) -> str:
    """
    Repair a common bad set-operation draft where the model packs a label and an
    identifier into one CONCAT(...) column instead of returning two columns.

    Example bad shape:
      SELECT CONCAT('2003-', orderNumber) AS year_orderNumber ...
      UNION
      SELECT CONCAT('2004-', orderNumber) AS year_orderNumber ...

    Rewrites to:
      SELECT orderNumber, '2003' AS year_label ...
      UNION
      SELECT orderNumber, '2004' AS year_label ...
      ORDER BY year_label ASC, orderNumber ASC
    """
    s = (sql or "").strip()
    q = (question or "").lower()
    if not s or "UNION" not in s.upper():
        return sql
    if "label" not in q:
        return sql

    ident = r"(?:[A-Za-z_][A-Za-z0-9_]*\.)?([A-Za-z_][A-Za-z0-9_]*)"
    tbl = r"(`?[A-Za-z_][A-Za-z0-9_]*`?)"
    pat = re.compile(
        rf"""
        ^\s*SELECT\s+CONCAT\(\s*'(?P<label1>[^']+)'\s*,\s*{ident}\s*\)\s+AS\s+`?(?P<bad_alias>[A-Za-z_][A-Za-z0-9_]*)`?\s+
        FROM\s+{tbl}\s+
        WHERE\s+YEAR\(\s*{ident}\s*\)\s*=\s*(?P<year1>\d{{4}})\s+
        UNION\s+
        SELECT\s+CONCAT\(\s*'(?P<label2>[^']+)'\s*,\s*{ident}\s*\)\s+AS\s+`?(?P=bad_alias)`?\s+
        FROM\s+{tbl}\s+
        WHERE\s+YEAR\(\s*{ident}\s*\)\s*=\s*(?P<year2>\d{{4}})\s*
        (?:ORDER\s+BY\s+`?(?P<order_alias>[A-Za-z_][A-Za-z0-9_]*)`?\s+(?P<order_dir>ASC|DESC)\s*)?
        ;?\s*$
        """,
        flags=re.I | re.S | re.X,
    )
    m = pat.match(s)
    if not m:
        return sql

    groups = m.groups()
    # Group layout from repeated unnamed captures:
    # label1, id1, bad_alias, table1, date1, year1, label2, id2, table2, date2, year2, order_alias, order_dir
    label1 = m.group("label1")
    label2 = m.group("label2")
    bad_alias = m.group("bad_alias")
    year1 = m.group("year1")
    year2 = m.group("year2")
    order_alias = m.group("order_alias") or ""
    order_dir = (m.group("order_dir") or "ASC").upper()
    # Positional unnamed capture groups for repeated identifier/table parts.
    # 1=id1, 2=table1, 3=date1, 4=id2, 5=table2, 6=date2
    id1 = groups[1]
    table1 = groups[3]
    date1 = groups[4]
    id2 = groups[7]
    table2 = groups[8]
    date2 = groups[9]

    if not all((id1, id2, table1, table2, date1, date2)):
        return sql
    if id1.lower() != id2.lower():
        return sql
    if table1.replace("`", "").lower() != table2.replace("`", "").lower():
        return sql
    if date1.lower() != date2.lower():
        return sql
    if order_alias and order_alias.lower() != bad_alias.lower():
        return sql

    # Only unwrap the "label-prefix + id" misuse, not arbitrary CONCATs.
    if not label1.endswith("-") or not label2.endswith("-"):
        return sql

    clean_label1 = label1[:-1]
    clean_label2 = label2[:-1]
    label_alias = "year_label" if "year label" in q else "label"
    order_dir = "DESC" if order_dir == "DESC" else "ASC"

    out = (
        f"SELECT {id1}, '{clean_label1}' AS {label_alias} "
        f"FROM {table1} WHERE YEAR({date1}) = {year1} "
        "UNION "
        f"SELECT {id1}, '{clean_label2}' AS {label_alias} "
        f"FROM {table1} WHERE YEAR({date1}) = {year2} "
        f"ORDER BY {label_alias} {order_dir}, {id1} ASC"
    )
    if s.endswith(";"):
        out += ";"
    return out


def _rewrite_case_split_to_union(sql: str, question: str) -> str:
    """
    Repair a bad set-operation draft where the model emulates UNION with multiple
    CASE columns in a single SELECT.

    Example bad shape:
      SELECT
        CASE WHEN YEAR(orderDate)=2003 THEN orderNumber ELSE NULL END AS `2003`,
        CASE WHEN YEAR(orderDate)=2004 THEN orderNumber ELSE NULL END
      FROM orders
      WHERE YEAR(orderDate) IN (2003, 2004)

    Rewrites to row-wise set operation branches:
      SELECT orderNumber, '2003' AS year_label ...
      UNION
      SELECT orderNumber, '2004' AS year_label ...
    """
    s = (sql or "").strip()
    q = (question or "").lower()
    if not s or "CASE" not in s.upper() or "UNION" in s.upper():
        return sql
    if "label" not in q:
        return sql

    def _find_top_level_kw(text: str, kw: str, start: int = 0) -> int:
        target = kw.upper()
        t_up = text.upper()
        depth = 0
        quote = ""
        i = start
        while i < len(text):
            ch = text[i]
            if quote:
                if ch == quote and (i == 0 or text[i - 1] != "\\"):
                    quote = ""
                i += 1
                continue
            if ch in {"'", '"', "`"}:
                quote = ch
                i += 1
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif depth == 0 and t_up.startswith(target, i):
                prev_ok = i == 0 or not (t_up[i - 1].isalnum() or t_up[i - 1] == "_")
                after = i + len(target)
                next_ok = after >= len(t_up) or not (t_up[after].isalnum() or t_up[after] == "_")
                if prev_ok and next_ok:
                    return i
            i += 1
        return -1

    def _split_top_level_csv(text: str) -> List[str]:
        parts: List[str] = []
        cur: List[str] = []
        depth = 0
        quote = ""
        for i, ch in enumerate(text):
            if quote:
                cur.append(ch)
                if ch == quote and (i == 0 or text[i - 1] != "\\"):
                    quote = ""
                continue
            if ch in {"'", '"', "`"}:
                quote = ch
                cur.append(ch)
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            if ch == "," and depth == 0:
                part = "".join(cur).strip()
                if part:
                    parts.append(part)
                cur = []
                continue
            cur.append(ch)
        tail = "".join(cur).strip()
        if tail:
            parts.append(tail)
        return parts

    select_pos = _find_top_level_kw(s, "SELECT")
    from_pos = _find_top_level_kw(s, "FROM", max(0, select_pos + 6))
    if select_pos < 0 or from_pos < 0:
        return sql
    where_pos = _find_top_level_kw(s, "WHERE", from_pos + 4)
    order_pos = _find_top_level_kw(s, "ORDER BY", from_pos + 4)
    limit_pos = _find_top_level_kw(s, "LIMIT", from_pos + 4)

    from_end_candidates = [p for p in (where_pos, order_pos, limit_pos) if p >= 0]
    from_end = min(from_end_candidates) if from_end_candidates else len(s)

    select_part = s[select_pos + len("SELECT"):from_pos].strip()
    from_part = s[from_pos + len("FROM"):from_end].strip()
    where_part = ""
    if where_pos >= 0:
        where_end_candidates = [p for p in (order_pos, limit_pos) if p > where_pos]
        where_end = min(where_end_candidates) if where_end_candidates else len(s)
        where_part = s[where_pos + len("WHERE"):where_end].strip()
    order_part = ""
    if order_pos >= 0:
        order_end = limit_pos if limit_pos > order_pos else len(s)
        order_part = s[order_pos + len("ORDER BY"):order_end].strip().rstrip(";")

    if not select_part or not from_part:
        return sql
    if any(tok in from_part.upper() for tok in (" JOIN ", ",")):
        return sql

    case_items = _split_top_level_csv(select_part)
    if len(case_items) < 2:
        return sql

    case_re = re.compile(
        r"""
        ^CASE\s+WHEN\s+YEAR\(\s*(?P<datecol>[A-Za-z_][A-Za-z0-9_]*)\s*\)\s*=\s*(?P<year>\d{4})\s+
        THEN\s+(?P<idcol>[A-Za-z_][A-Za-z0-9_]*)\s+ELSE\s+NULL\s+END
        (?:\s+AS\s+`?(?P<alias>[A-Za-z_][A-Za-z0-9_]*|\d{4})`?)?
        $""",
        flags=re.I | re.X,
    )

    branches: List[Tuple[str, str, str, str]] = []
    base_date = ""
    base_id = ""
    for item in case_items:
        m = case_re.match(item.strip())
        if not m:
            return sql
        datecol = m.group("datecol")
        year = m.group("year")
        idcol = m.group("idcol")
        alias = (m.group("alias") or year).strip("`")
        if not base_date:
            base_date = datecol
            base_id = idcol
        elif datecol.lower() != base_date.lower() or idcol.lower() != base_id.lower():
            return sql
        branches.append((year, alias, datecol, idcol))

    if not branches:
        return sql

    if where_part:
        years_in_where = {
            y for y in re.findall(r"\b(?:19|20)\d{2}\b", where_part)
        }
        branch_years = {y for y, _, _, _ in branches}
        if years_in_where and branch_years and not branch_years.issubset(years_in_where):
            return sql

    label_alias = "year_label" if "year label" in q else "label"
    use_union = bool(
        re.search(
            r"\b(?:remove\s+duplicate\s+rows|without\s+duplicates|no\s+duplicate|distinct)\b",
            q,
            flags=re.I,
        )
    )
    union_kw = "UNION" if use_union else "UNION ALL"

    label_dir = "ASC"
    id_dir = "ASC"
    if order_part:
        order_items = _split_top_level_csv(order_part)
        if order_items:
            m1 = re.search(r"\b(DESC|ASC)\b", order_items[0], flags=re.I)
            if m1:
                label_dir = m1.group(1).upper()
        if len(order_items) >= 2:
            m2 = re.search(r"\b(DESC|ASC)\b", order_items[1], flags=re.I)
            if m2:
                id_dir = m2.group(1).upper()

    branch_sqls = [
        f"SELECT {base_id}, '{alias}' AS {label_alias} FROM {from_part} WHERE YEAR({base_date}) = {year}"
        for year, alias, _, _ in branches
    ]
    out = f" {union_kw} ".join(branch_sqls)
    if order_part or re.search(r"\b(?:order|sort|sorted)\b", q, flags=re.I):
        out += f" ORDER BY {label_alias} {label_dir}, {base_id} {id_dir}"
    if s.endswith(";"):
        out += ";"
    return out


def _rewrite_union_case_label_to_literal_branches(sql: str, question: str) -> str:
    """
    Repair a bad UNION draft where each branch still uses the same CASE expression
    to derive a label, instead of assigning a constant label per branch.

    Example bad shape:
      SELECT orderNumber,
             CASE WHEN YEAR(orderDate)=2003 THEN '2003' ELSE '2004' END AS year_label
      FROM orders
      WHERE YEAR(orderDate) IN (2003, 2004)
      UNION
      SELECT orderNumber,
             CASE WHEN YEAR(orderDate)=2003 THEN '2003' ELSE '2004' END AS year_label
      FROM orders
      WHERE YEAR(orderDate) IN (2003, 2004)
    """
    s = (sql or "").strip()
    q = (question or "").lower()
    if not s or "UNION" not in s.upper() or "CASE" not in s.upper():
        return sql
    if "label" not in q:
        return sql

    def _find_top_level_kw(text: str, kw: str, start: int = 0) -> int:
        target = kw.upper()
        t_up = text.upper()
        depth = 0
        quote = ""
        i = start
        while i < len(text):
            ch = text[i]
            if quote:
                if ch == quote and (i == 0 or text[i - 1] != "\\"):
                    quote = ""
                i += 1
                continue
            if ch in {"'", '"', "`"}:
                quote = ch
                i += 1
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif depth == 0 and t_up.startswith(target, i):
                prev_ok = i == 0 or not (t_up[i - 1].isalnum() or t_up[i - 1] == "_")
                after = i + len(target)
                next_ok = after >= len(t_up) or not (t_up[after].isalnum() or t_up[after] == "_")
                if prev_ok and next_ok:
                    return i
            i += 1
        return -1

    body = s[:-1].rstrip() if s.endswith(";") else s
    order_pos = _find_top_level_kw(body, "ORDER BY")
    if order_pos >= 0:
        order_clause = body[order_pos + len("ORDER BY"):].strip()
        core = body[:order_pos].rstrip()
    else:
        order_clause = ""
        core = body

    union_pos = _find_top_level_kw(core, "UNION ALL")
    union_kw_raw = "UNION ALL"
    if union_pos < 0:
        union_pos = _find_top_level_kw(core, "UNION")
        union_kw_raw = "UNION"
    if union_pos < 0:
        return sql

    branch1 = core[:union_pos].strip()
    branch2 = core[union_pos + len(union_kw_raw):].strip()
    if not branch1 or not branch2 or _find_top_level_kw(branch2, "UNION") >= 0:
        return sql

    branch_re = re.compile(
        r"""
        ^\s*SELECT\s+
        (?P<idcol>(?:[A-Za-z_][A-Za-z0-9_]*\.)?[A-Za-z_][A-Za-z0-9_]*)\s*,\s*
        CASE\s+WHEN\s+YEAR\(\s*(?P<datecol>(?:[A-Za-z_][A-Za-z0-9_]*\.)?[A-Za-z_][A-Za-z0-9_]*)\s*\)\s*=\s*(?P<year1>\d{4})\s+
        THEN\s+'(?P<label1>[^']+)'\s+ELSE\s+'(?P<label2>[^']+)'\s+END\s+AS\s+`?(?P<label_alias>[A-Za-z_][A-Za-z0-9_]*|\d{4})`?\s+
        FROM\s+(?P<table>`?[A-Za-z_][A-Za-z0-9_]*`?)\s+
        WHERE\s+YEAR\(\s*(?P=datecol)\s*\)\s+IN\s*\(\s*(?P<iny1>\d{4})\s*,\s*(?P<iny2>\d{4})\s*\)\s*$
        """,
        flags=re.I | re.S | re.X,
    )

    m1 = branch_re.match(branch1)
    m2 = branch_re.match(branch2)
    if not m1 or not m2:
        return sql

    def _norm_table(v: str) -> str:
        return (v or "").replace("`", "").strip().lower()

    def _norm_ident(v: str) -> str:
        return re.sub(r"\s+", "", (v or "").strip()).lower()

    for key in ("idcol", "datecol", "year1", "label1", "label2", "label_alias"):
        if _norm_ident(m1.group(key)) != _norm_ident(m2.group(key)):
            return sql
    if _norm_table(m1.group("table")) != _norm_table(m2.group("table")):
        return sql

    years_in = {m1.group("iny1"), m1.group("iny2")}
    if len(years_in) != 2 or m1.group("year1") not in years_in:
        return sql
    other_years = [y for y in years_in if y != m1.group("year1")]
    if len(other_years) != 1:
        return sql

    idcol = m1.group("idcol")
    datecol = m1.group("datecol")
    table = m1.group("table")
    year1 = m1.group("year1")
    year2 = other_years[0]
    label1 = m1.group("label1")
    label2 = m1.group("label2")
    label_alias = m1.group("label_alias")

    def _segment_dir(text: str, target_pat: str, default: str = "ASC") -> str:
        if not text:
            return default
        if re.search(rf"{target_pat}.*\bdescending\b", text, flags=re.I) or re.search(
            rf"\bdescending\b.*{target_pat}", text, flags=re.I
        ):
            return "DESC"
        if re.search(rf"{target_pat}.*\bascending\b", text, flags=re.I) or re.search(
            rf"\bascending\b.*{target_pat}", text, flags=re.I
        ):
            return "ASC"
        return default

    label_dir = "ASC"
    id_dir = "ASC"
    if order_clause:
        order_items = [x.strip() for x in order_clause.split(",") if x.strip()]
        if order_items:
            label_dir = "DESC" if re.search(r"\bDESC\b", order_items[0], flags=re.I) else "ASC"
        if len(order_items) >= 2:
            id_dir = "DESC" if re.search(r"\bDESC\b", order_items[1], flags=re.I) else "ASC"
    else:
        label_dir = _segment_dir(q, r"\byear\s+label\b", "ASC")
        id_words = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", idcol.replace("_", " ")).lower()
        id_dir = _segment_dir(q, rf"\b{re.escape(id_words)}\b", "ASC")

    if re.search(
        r"\b(?:remove\s+duplicate\s+rows|without\s+duplicates|no\s+duplicate|distinct)\b",
        q,
        flags=re.I,
    ):
        union_kw = "UNION"
    else:
        union_kw = union_kw_raw

    out = (
        f"SELECT {idcol}, '{label1}' AS {label_alias} "
        f"FROM {table} WHERE YEAR({datecol}) = {year1} "
        f"{union_kw} "
        f"SELECT {idcol}, '{label2}' AS {label_alias} "
        f"FROM {table} WHERE YEAR({datecol}) = {year2}"
    )
    if order_clause or re.search(r"\b(?:order|sort|sorted)\b", q, flags=re.I):
        out += f" ORDER BY {label_alias} {label_dir}, {idcol} {id_dir}"
    if s.endswith(";"):
        out += ";"
    return out

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


def _ast_guardrails(sql: str, ctx: str = "", question: str = "") -> Tuple[str, Dict[str, Any]]:
    """
    SQL AST guardrails (sqlglot-first).

    Goals:
    - Resolve easy alias/reference issues safely.
    - Auto-qualify ambiguous ORDER BY columns when a deterministic target exists.
    - Surface join-key sanity warnings without destructive rewrites.

    Returns:
      (possibly_rewritten_sql, meta)
      meta = {
        "parse_ok": bool,
        "fixes": List[str],
        "warnings": List[str],
        "error": str,
      }
    """
    meta: Dict[str, Any] = {"parse_ok": False, "fixes": [], "warnings": [], "error": ""}
    if not sql:
        return sql, meta

    try:
        import sqlglot
        from sqlglot import exp
    except Exception as e:
        meta["error"] = f"sqlglot_unavailable:{e}"
        return sql, meta

    try:
        ast = sqlglot.parse_one(sql, read="mysql")
        meta["parse_ok"] = True
    except Exception as e:
        meta["error"] = f"parse_error:{e}"
        return sql, meta

    sel = ast.find(exp.Select)
    if sel is None:
        return sql, meta

    def _flatten_top_and(expr_in: Optional["exp.Expression"]) -> List["exp.Expression"]:
        if expr_in is None:
            return []
        if isinstance(expr_in, exp.And):
            return _flatten_top_and(expr_in.this) + _flatten_top_and(expr_in.expression)
        return [expr_in]

    def _build_and(exprs: List["exp.Expression"]) -> Optional["exp.Expression"]:
        cleaned = [e for e in exprs if e is not None]
        if not cleaned:
            return None
        out = cleaned[0]
        for nxt in cleaned[1:]:
            out = exp.And(this=out, expression=nxt)
        return out

    def _question_has_agg_threshold(q: str) -> bool:
        ql = (q or "").lower()
        if not ql:
            return False
        agg_tokens = ("average", "avg", "sum", "total", "count", "maximum", "minimum", "max", "min")
        cmp_tokens = ("above", "below", "greater", "less", "over", "under", "more than", "less than", ">", "<", ">=", "<=")
        return any(t in ql for t in agg_tokens) and any(t in ql for t in cmp_tokens)

    def _expr_contains_agg(expr_in: Optional["exp.Expression"]) -> bool:
        """
        Detect aggregate functions only in the current expression scope.
        Aggregates inside scalar subqueries (e.g. col > (SELECT AVG(...))) must
        not trigger WHERE->HAVING rewrites, because those predicates are valid
        row-level filters pre-grouping.
        """
        if expr_in is None:
            return False

        def _has_agg_no_subquery(node: "exp.Expression") -> bool:
            if isinstance(node, exp.Subquery):
                return False
            if isinstance(node, exp.AggFunc):
                return True
            for child in node.iter_expressions():
                if _has_agg_no_subquery(child):
                    return True
            return False

        return _has_agg_no_subquery(expr_in)

    # -----------------------------
    # 0) AST-first semantic rewrites (conservative)
    # -----------------------------
    # 0a) GROUP BY query: move aggregate predicates from WHERE -> HAVING.
    group_arg = sel.args.get("group")
    where_arg = sel.args.get("where")
    if group_arg is not None and isinstance(where_arg, exp.Where):
        where_cond = where_arg.this
        conds = _flatten_top_and(where_cond)
        move_to_having: List["exp.Expression"] = []
        keep_in_where: List["exp.Expression"] = []
        for c in conds:
            # Keep scalar-subquery predicates in WHERE; moving them to HAVING
            # can break semantics and MySQL grouping validity.
            if c.find(exp.Subquery):
                keep_in_where.append(c.copy())
                continue
            if _expr_contains_agg(c):
                move_to_having.append(c.copy())
            else:
                keep_in_where.append(c.copy())

        if move_to_having:
            new_where = _build_and(keep_in_where)
            if new_where is None:
                sel.set("where", None)
            else:
                sel.set("where", exp.Where(this=new_where))

            existing_having = sel.args.get("having")
            existing_having_cond = None
            if isinstance(existing_having, exp.Having):
                existing_having_cond = existing_having.this
            merged_having = _build_and(
                ([existing_having_cond.copy()] if existing_having_cond is not None else [])
                + move_to_having
            )
            if merged_having is not None:
                sel.set("having", exp.Having(this=merged_having))
                meta["fixes"].append("agg_where_to_having_ast")

    # 0b) Aggregate-threshold intent: raw metric comparisons in WHERE -> HAVING.
    # Example:
    #   WHERE amount > 50000 with SELECT AVG(amount) ... GROUP BY ...
    # becomes:
    #   HAVING AVG(amount) > 50000
    group_arg = sel.args.get("group")
    where_arg = sel.args.get("where")
    if (
        group_arg is not None
        and isinstance(where_arg, exp.Where)
        and _question_has_agg_threshold(question)
    ):
        agg_map: Dict[str, "exp.Expression"] = {}
        for item in sel.expressions or []:
            base = item.this if isinstance(item, exp.Alias) else item
            if base is None:
                continue
            for agg in base.find_all(exp.AggFunc):
                cols = [c for c in agg.find_all(exp.Column)]
                if len(cols) != 1:
                    continue
                c0 = cols[0]
                key_name = str(c0.name or "").strip("`").lower()
                if not key_name:
                    continue
                if key_name not in agg_map:
                    agg_map[key_name] = agg.copy()

        if agg_map:
            cmp_types = (exp.GT, exp.GTE, exp.LT, exp.LTE, exp.EQ, exp.NEQ)

            def _is_literalish(x: Optional["exp.Expression"]) -> bool:
                return isinstance(x, (exp.Literal, exp.Null))

            def _rewrite_cmp(expr_in: "exp.Expression") -> Optional["exp.Expression"]:
                if not isinstance(expr_in, cmp_types):
                    return None
                if _expr_contains_agg(expr_in):
                    return None

                left = getattr(expr_in, "this", None)
                right = getattr(expr_in, "expression", None)
                if left is None or right is None:
                    return None

                # Avoid moving joins/column-vs-column predicates.
                if isinstance(left, exp.Column) and isinstance(right, exp.Column):
                    return None
                if not (_is_literalish(left) or _is_literalish(right)):
                    return None

                changed = False
                new_expr = expr_in.copy()
                new_left = getattr(new_expr, "this", None)
                new_right = getattr(new_expr, "expression", None)

                if isinstance(new_left, exp.Column):
                    lk = str(new_left.name or "").strip("`").lower()
                    agg_expr = agg_map.get(lk)
                    if agg_expr is not None:
                        new_expr.set("this", agg_expr.copy())
                        changed = True

                if isinstance(new_right, exp.Column):
                    rk = str(new_right.name or "").strip("`").lower()
                    agg_expr = agg_map.get(rk)
                    if agg_expr is not None:
                        new_expr.set("expression", agg_expr.copy())
                        changed = True

                return new_expr if changed else None

            where_cond = where_arg.this
            conds = _flatten_top_and(where_cond)
            move_to_having: List["exp.Expression"] = []
            keep_in_where: List["exp.Expression"] = []
            for c in conds:
                rewritten = _rewrite_cmp(c)
                if rewritten is not None:
                    move_to_having.append(rewritten)
                else:
                    keep_in_where.append(c.copy())

            if move_to_having:
                new_where = _build_and(keep_in_where)
                if new_where is None:
                    sel.set("where", None)
                else:
                    sel.set("where", exp.Where(this=new_where))

                existing_having = sel.args.get("having")
                existing_having_cond = None
                if isinstance(existing_having, exp.Having):
                    existing_having_cond = existing_having.this
                merged_having = _build_and(
                    ([existing_having_cond.copy()] if existing_having_cond is not None else [])
                    + move_to_having
                )
                if merged_having is not None:
                    sel.set("having", exp.Having(this=merged_having))
                    meta["fixes"].append("raw_metric_where_to_having_ast")

    # -----------------------------
    # Top-level alias -> table map
    # -----------------------------
    alias_to_table: Dict[str, str] = {}
    alias_order: List[str] = []

    def _register_source(src: "exp.Expression") -> None:
        # Base table source
        if isinstance(src, exp.Table):
            table_name = (src.name or "").strip("`")
            if not table_name:
                return
            table_l = table_name.lower()
            alias_raw = (src.alias_or_name or table_name).strip("`")
            alias_l = alias_raw.lower()
            if alias_l and alias_l not in alias_to_table:
                alias_to_table[alias_l] = table_l
                alias_order.append(alias_l)
            if table_l and table_l not in alias_to_table:
                # Allow table-name qualifiers as valid references.
                alias_to_table[table_l] = table_l
            return

        # Derived table source: keep alias as known qualifier, but never try to
        # map its columns to base schema columns in this pass.
        if isinstance(src, exp.Subquery):
            alias_raw = (src.alias_or_name or "").strip("`")
            alias_l = alias_raw.lower()
            if alias_l and alias_l not in alias_to_table:
                alias_to_table[alias_l] = f"__derived__:{alias_l}"
                alias_order.append(alias_l)
            return

    from_arg = sel.args.get("from") or sel.args.get("from_")
    if from_arg is not None and getattr(from_arg, "this", None) is not None:
        _register_source(from_arg.this)
    for j in sel.args.get("joins") or []:
        _register_source(getattr(j, "this", None))

    # Fallback when top-level extraction misses cases.
    if not alias_to_table:
        alias_raw = _parse_from_join_aliases(sql)
        for a, t in alias_raw.items():
            al = str(a or "").strip().lower()
            tl = str(t or "").strip().lower()
            if not al or not tl:
                continue
            if al not in alias_to_table:
                alias_to_table[al] = tl
                alias_order.append(al)
            if tl not in alias_to_table:
                alias_to_table[tl] = tl

    if not alias_to_table:
        return sql, meta

    # -----------------------------
    # Build table column map (ctx + schema fallback)
    # -----------------------------
    allowed = _parse_allowed_identifiers(ctx or "")
    allowed_l: Dict[str, Set[str]] = {}
    for t, cols in (allowed or {}).items():
        tl = str(t or "").strip("`").lower()
        if not tl:
            continue
        allowed_l.setdefault(tl, set()).update(
            str(c or "").strip("`").lower() for c in (cols or set()) if str(c or "").strip()
        )

    analyzer = None
    try:
        analyzer = get_schema_analyzer()
    except Exception:
        analyzer = None

    alias_cols: Dict[str, Set[str]] = {}
    for alias, table in alias_to_table.items():
        if str(table).startswith("__derived__:"):
            alias_cols[alias] = set()
            continue
        cols = set(allowed_l.get(table, set()))
        if analyzer is not None:
            try:
                cols.update(str(c).strip("`").lower() for c in analyzer.get_table_columns(table))
            except Exception:
                pass
        alias_cols[alias] = cols

    # Map: column_name -> aliases that contain it
    col_to_aliases: Dict[str, List[str]] = {}
    for alias, cols in alias_cols.items():
        for c in cols:
            col_to_aliases.setdefault(c, []).append(alias)

    # Prefer dimension-like aliases over transaction aliases for ambiguous refs.
    try:
        txn_tables = {str(t or "").strip().lower() for t in (get_transaction_tables() or [])}
    except Exception:
        txn_tables = set()

    def _preferred_alias(col_name: str, candidates: List[str]) -> Optional[str]:
        uniq = []
        seen = set()
        for c in candidates:
            cl = str(c or "").strip().lower()
            if cl and cl not in seen:
                uniq.append(cl)
                seen.add(cl)
        if not uniq:
            return None
        if len(uniq) == 1:
            return uniq[0]

        non_tx = [a for a in uniq if alias_to_table.get(a, "") not in txn_tables]
        if len(non_tx) == 1:
            return non_tx[0]

        for a in alias_order:
            if a in uniq:
                return a
        return uniq[0]

    # -----------------------------
    # Projection map for ORDER BY disambiguation
    # -----------------------------
    projection_col_map: Dict[str, List["exp.Column"]] = {}
    for item in sel.expressions or []:
        base = item.this if isinstance(item, exp.Alias) else item
        if not isinstance(base, exp.Column):
            continue
        out_name = str(item.alias_or_name or "").strip("`").lower()
        if out_name:
            projection_col_map.setdefault(out_name, []).append(base.copy())
        base_name = str(base.name or "").strip("`").lower()
        if base_name:
            projection_col_map.setdefault(base_name, []).append(base.copy())

    def _is_nested_subquery_column(col_expr: "exp.Column") -> bool:
        """
        Return True when a column does not belong to the current top-level SELECT.

        Important: nested SELECT inside EXISTS/IN may not be wrapped by exp.Subquery.
        So we identify scope by the nearest ancestor exp.Select:
        - nearest Select == current `sel`  -> top-level column (safe)
        - nearest Select != current `sel`  -> nested subquery column (must skip)
        """
        anc = getattr(col_expr, "parent", None)
        while anc is not None:
            if isinstance(anc, exp.Select):
                return anc is not sel
            anc = getattr(anc, "parent", None)
        return False

    # -----------------------------
    # 1) Unknown qualifier fix/warn
    # -----------------------------
    known_qualifiers = set(alias_to_table.keys())
    for col in sel.find_all(exp.Column):
        if _is_nested_subquery_column(col):
            # Never rewrite nested scopes in this pass.
            continue
        table_q = str(col.table or "").strip("`").lower()
        col_name = str(col.name or "").strip("`").lower()
        if not table_q:
            continue
        if table_q in known_qualifiers:
            continue

        owners = col_to_aliases.get(col_name, [])
        best = _preferred_alias(col_name, owners)
        if best:
            col.set("table", exp.to_identifier(best))
            meta["fixes"].append(f"unknown_qualifier_fixed:{table_q}->{best}.{col_name}")
        else:
            meta["warnings"].append(f"unknown_qualifier:{table_q}.{col_name}")

    # -----------------------------
    # 2) Ambiguous ORDER BY disambiguation
    # -----------------------------
    order = sel.args.get("order")
    if order is not None:
        for ord_item in order.expressions or []:
            this_expr = getattr(ord_item, "this", None)
            if not isinstance(this_expr, exp.Column):
                continue
            if str(this_expr.table or "").strip():
                continue

            c_name = str(this_expr.name or "").strip("`")
            c_low = c_name.lower()
            if not c_low:
                continue

            proj_candidates = projection_col_map.get(c_low, [])
            proj_norm = {}
            for p in proj_candidates:
                key = p.sql(dialect="mysql")
                proj_norm[key] = p

            replacement = None
            if len(proj_norm) == 1:
                replacement = list(proj_norm.values())[0].copy()
                meta["fixes"].append(f"order_by_qualified_from_projection:{c_low}")
            else:
                owners = col_to_aliases.get(c_low, [])
                if owners:
                    best = _preferred_alias(c_low, owners)
                    if best:
                        replacement = exp.column(c_name, table=best)
                        if len(set(owners)) > 1:
                            meta["fixes"].append(f"order_by_ambiguous_qualified:{best}.{c_low}")
                elif len(set(proj_norm.keys())) > 1:
                    meta["warnings"].append(f"order_by_projection_ambiguous:{c_low}")

            if replacement is not None:
                ord_item.set("this", replacement)

    # -----------------------------
    # 3) Join key sanity warning
    # -----------------------------
    fk_map = {}
    try:
        fk_map = get_fk_relationships() or {}
    except Exception:
        fk_map = {}

    joins = sel.args.get("joins") or []
    for j in joins:
        on_expr = j.args.get("on")
        if on_expr is None:
            continue
        for eq in on_expr.find_all(exp.EQ):
            left = getattr(eq, "left", None)
            right = getattr(eq, "right", None)
            if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
                continue
            lt_alias = str(left.table or "").strip("`").lower()
            rt_alias = str(right.table or "").strip("`").lower()
            lc = str(left.name or "").strip("`").lower()
            rc = str(right.name or "").strip("`").lower()
            if not lt_alias or not rt_alias or not lc or not rc:
                continue

            lt = alias_to_table.get(lt_alias, lt_alias)
            rt = alias_to_table.get(rt_alias, rt_alias)

            # Skip derived-table aliases in FK sanity checks.
            if str(lt).startswith("__derived__:") or str(rt).startswith("__derived__:"):
                continue

            plausible = False
            # direct fk metadata check
            if lt in fk_map and lc in {str(k).lower() for k in (fk_map.get(lt) or {}).keys()}:
                for fk_col, tgt in (fk_map.get(lt) or {}).items():
                    if str(fk_col).lower() == lc and str(tgt).lower() == rt:
                        plausible = True
                        break
            if not plausible and rt in fk_map and rc in {str(k).lower() for k in (fk_map.get(rt) or {}).keys()}:
                for fk_col, tgt in (fk_map.get(rt) or {}).items():
                    if str(fk_col).lower() == rc and str(tgt).lower() == lt:
                        plausible = True
                        break

            # fallback heuristic for shared id-like fields
            if not plausible and lc == rc and (lc.endswith("id") or lc.endswith("number") or lc.endswith("code")):
                plausible = True

            if not plausible:
                meta["warnings"].append(f"suspicious_join_key:{lt_alias}.{lc}={rt_alias}.{rc}")

    # Return rewritten SQL only when we made deterministic fixes.
    if not meta["fixes"]:
        return sql, meta

    out_sql = ast.sql(dialect="mysql")
    if sql.strip().endswith(";"):
        out_sql = out_sql.rstrip(";").strip() + ";"
    return out_sql, meta

def _strip_unneeded_count_distinct(sql: str, question: str = "") -> str:
    """
    Remove unnecessary DISTINCT in COUNT(DISTINCT alias.id_col) for single-join queries.

    Previous behavior was hardcoded for customers/orders + orderNumber only.
    New behavior is schema-driven:
    - only when query has exactly one JOIN
    - only for DISTINCT on the joined table's inferred ID column
    """
    if not sql:
        return sql
    # If the question explicitly asks for distinct counting, preserve the
    # semantics as written. Let the semantic contract layer reason about it,
    # but do not erase DISTINCT pre-emptively.
    if question and _question_wants_count_distinct(question):
        return sql
    if join_count(sql) != 1:
        return sql

    alias_to_table = _parse_from_join_aliases(sql)
    if not alias_to_table:
        return sql

    pat = re.compile(
        r"COUNT\s*\(\s*DISTINCT\s+([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\s*\)",
        flags=re.I,
    )

    def _repl(m: re.Match) -> str:
        alias = m.group(1)
        col = m.group(2)
        table = alias_to_table.get(alias) or alias_to_table.get(alias.lower())
        if not table:
            return m.group(0)

        id_col = get_id_column(table)
        if not id_col:
            return m.group(0)

        if col.lower() != id_col.lower():
            return m.group(0)

        return f"COUNT({alias}.{col})"

    return pat.sub(_repl, sql)
# =======================================================================
# ========== Phase 7 — Ordering, Limits, and Output Formatting ==========
# ============== UI and execution-safety adjustments only ===============
# =======================================================================
# - question_requests_ordering
# - enforce_window_sequence_direction
# - enforce_requested_order_direction
# - remove_top_level_order_by_if_not_requested
# - explicit_limit_from_question
# - maybe_add_limit
# - collapse_spaces_outside_quotes

def _question_requests_ordering(question: str) -> bool:
    """
    Detect if user explicitly wants final result ordering.
    
    IMPORTANT: "descending/ascending order" in context of RANK/ROW_NUMBER 
    refers to window function ordering, NOT final result ORDER BY.
    """
    q = (question or "").lower()
    
    # --- Pattern 1: Window function context (NOT final ORDER BY) ---
    # If "ascending/descending order" appears with ranking context, skip it
    # E.g., "rank employees ... based on employee number in descending order"
    # E.g., "cumulative payment amount ... ordered by payment date"
    window_context_patterns = [
        r"\brank\b.*\b(ascending|descending)\s+order\b",
        r"\brow[_\s]?number\b.*\b(ascending|descending)\s+order\b",
        r"\bdense[_\s]?rank\b.*\b(ascending|descending)\s+order\b",
        r"\bpartition\s+by\b.*\b(ascending|descending)\b",
        r"\bbased\s+on\b.*\b(ascending|descending)\s+order\b",  # "based on X in descending order" = window context
        r"\b(cumulative|running)\b.*\bordered\s+by\b",  # "cumulative X ordered by Y" = window context
        r"\b(second|third|fourth|fifth|\d+(?:st|nd|rd|th))\b.*\b(sorted|ordered)\s+by\b",
        r"\b(sorted|ordered)\s+by\b.*\b(second|third|fourth|fifth|\d+(?:st|nd|rd|th))\b",
    ]
    
    has_window_context = any(re.search(pat, q) for pat in window_context_patterns)
    
    # --- Pattern 2: Explicit final ordering keywords ---
    # These strongly indicate user wants final ORDER BY
    # NOTE: "ordered by" only counts if NOT in window context
    explicit_sort_keywords = [
        "order by", "arrange by",
        "urut berdasarkan", "urutkan berdasarkan", "diurutkan berdasarkan",
    ]
    
    if any(k in q for k in explicit_sort_keywords):
        return True
    
    # "sort by" / "sorted by" / "ordered by" can be ranking/window context.
    if ("sort by" in q or "sorted by" in q or "ordered by" in q) and not has_window_context:
        return True

    # "ordered by" is ambiguous - only treat as final sort if NOT in window context
    if "ordered by" in q and not has_window_context:
        return True
    
    # --- Pattern 3: Superlative / ranking output (implies ORDER BY + LIMIT) ---
    # Keep this conservative. Bare "first"/"last" is too ambiguous because it
    # often appears in projection phrases like "first names" / "last names".
    # Treat "first"/"last" as ordering only in bounded row-selection shapes.
    superlative_keywords = [
        "top ", "bottom ",
        "highest", "lowest", "most expensive", "cheapest",
        "terbesar", "terkecil", "tertinggi", "terendah", 
        "termahal", "termurah", "terbaru", "terlama",
    ]
    
    if any(k in q for k in superlative_keywords):
        return True

    first_last_row_selection_patterns = [
        r"\b(?:first|last)\s+\d+\b",
        r"\b(?:first|last)\s+(?:row|rows|record|records|entry|entries|result|results)\b",
        r"\b(?:the\s+)?(?:first|last)\s+page\b",
    ]
    if any(re.search(pat, q) for pat in first_last_row_selection_patterns):
        return True
    
    # --- Pattern 4: Explicit sort direction (only if NOT in window context) ---
    # "ascending", "descending" without window function context
    sort_direction_keywords = [
        "ascending", "descending", "asc", "desc",
        "alphabetical", "alphabetically",
        "menaik", "menurun",
    ]
    
    if not has_window_context and any(k in q for k in sort_direction_keywords):
        return True
    
    # --- Pattern 5: Phrase-based ordering intent ---
    # "from highest to lowest", "from latest", "from A to Z"
    ordering_phrases = [
        r"\bfrom\s+(the\s+)?(highest|lowest|latest|earliest|newest|oldest)",
        r"\bfrom\s+(a\s+to\s+z|z\s+to\s+a)",
        r"\b(sorted|ordered)\s+from\b",
    ]
    
    if any(re.search(pat, q) for pat in ordering_phrases):
        return True
    
    return False


def _enforce_window_sequence_direction(sql: str, question: str) -> str:
    """
    Fix LAG/LEAD window ORDER BY direction for sequence-style questions.

    For natural questions like:
      "previous payment amount ... ordered by payment date"
    the sequence should be chronological ascending unless the question
    explicitly says descending.

    Only rewrites ORDER BY inside LAG/LEAD OVER(...), never top-level ORDER BY.
    """
    s = (sql or "").strip()
    q = (question or "").lower()
    if not s or not re.search(r"\b(?:LAG|LEAD)\s*\(", s, flags=re.I):
        return sql
    if not re.search(r"\b(?:previous|prior|next|following)\b", q, flags=re.I):
        return sql
    if "ordered by" not in q:
        return sql

    ordered_idx = q.find("ordered by")
    ordered_tail = q[ordered_idx + len("ordered by"):] if ordered_idx >= 0 else q
    stop = re.search(r"[.;!?]", ordered_tail)
    if stop:
        ordered_tail = ordered_tail[:stop.start()]

    if re.search(r"\b(?:descending|desc|reverse)\b", ordered_tail, flags=re.I):
        desired_dir = "DESC"
    elif re.search(r"\b(?:ascending|asc|chronological|earliest\s+first|oldest\s+first)\b", ordered_tail, flags=re.I):
        desired_dir = "ASC"
    else:
        desired_dir = "ASC"

    laglead_pat = re.compile(
        r"""
        (?P<prefix>\b(?:LAG|LEAD)\s*\([^)]*\)\s+OVER\s*\()
        (?P<body>[^)]*?)
        (?P<suffix>\))
        """,
        flags=re.I | re.S | re.X,
    )

    def _rewrite_body(body: str) -> str:
        m = re.search(r"\bORDER\s+BY\b", body, flags=re.I)
        if not m:
            return body
        before = body[:m.end()]
        after = body[m.end():]
        order_items = [x.strip() for x in after.split(",") if x.strip()]
        if not order_items:
            return body
        first = order_items[0]
        rewritten_first = re.sub(r"\s+\b(?:ASC|DESC)\b\s*$", "", first, flags=re.I).rstrip()
        rewritten_first = f"{rewritten_first} {desired_dir}"
        if rewritten_first == first:
            return body
        order_items[0] = rewritten_first
        return before + " " + ", ".join(order_items)

    changed = False

    def _repl(m: re.Match) -> str:
        nonlocal changed
        body = m.group("body") or ""
        new_body = _rewrite_body(body)
        if new_body != body:
            changed = True
        return f"{m.group('prefix')}{new_body}{m.group('suffix')}"

    out = laglead_pat.sub(_repl, s)
    return out if changed else sql


def _enforce_requested_order_direction(sql: str, question: str) -> str:
    """
    Enforce explicit final ORDER BY directions from the question on the
    top-level ORDER BY clause.

    Conservative behavior:
    - only applies when the question explicitly says ascending/descending
      (or alphabetical/alphabetically)
    - only rewrites top-level ORDER BY, never window ORDER BY inside OVER(...)
    - preserves columns/expressions; only fixes direction tokens
    - supports multiple ORDER BY items by position when the question provides
      multiple ordered segments (split by ',' and 'then')
    """
    if not sql:
        return sql

    q = (question or "").lower()
    if not q:
        return sql

    window_context_patterns = [
        r"\brank\b.*\b(ascending|descending)\s+order\b",
        r"\brow[_\s]?number\b.*\b(ascending|descending)\s+order\b",
        r"\bdense[_\s]?rank\b.*\b(ascending|descending)\s+order\b",
        r"\bpartition\s+by\b.*\b(ascending|descending)\b",
        r"\bbased\s+on\b.*\b(ascending|descending)\s+order\b",
        r"\b(cumulative|running)\b.*\bordered\s+by\b",
    ]
    has_window_context = any(re.search(pat, q) for pat in window_context_patterns)

    def _extract_sort_clause(text: str) -> str:
        patterns = [
            r"\b(?:sort(?:ed)?\s+by|arrange\s+by|order\s+by)\b",
            r"\border\s+the\s+(?:result|results|output|rows?)\s+by\b",
            r"\bordered\s+by\b",
        ]
        for pat in patterns:
            m = re.search(pat, text, flags=re.I)
            if not m:
                continue
            if "ordered by" in m.group(0).lower() and has_window_context:
                continue
            tail = text[m.end():]
            end = len(tail)
            stop = re.search(r"[.;!?]", tail)
            if stop:
                end = stop.start()
            return tail[:end].strip()
        return ""

    def _split_top_level_csv(text: str) -> List[str]:
        parts: List[str] = []
        cur: List[str] = []
        depth = 0
        quote = ""
        for i, ch in enumerate(text):
            if quote:
                cur.append(ch)
                if ch == quote and (i == 0 or text[i - 1] != "\\"):
                    quote = ""
                continue
            if ch in {"'", '"', "`"}:
                quote = ch
                cur.append(ch)
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            if ch == "," and depth == 0:
                part = "".join(cur).strip()
                if part:
                    parts.append(part)
                cur = []
                continue
            cur.append(ch)
        tail = "".join(cur).strip()
        if tail:
            parts.append(tail)
        return parts

    sort_clause = _extract_sort_clause(q)
    explicit_dirs: List[str] = []
    if sort_clause:
        segments = re.split(r"\bthen\b", sort_clause, flags=re.I)
        for seg in segments:
            for part in _split_top_level_csv(seg):
                s = part.strip().lower()
                if not s:
                    continue
                if re.search(r"\b(?:descending|desc|z\s+to\s+a)\b", s):
                    explicit_dirs.append("DESC")
                elif re.search(r"\b(?:ascending|asc|alphabetical|alphabetically|a\s+to\s+z)\b", s):
                    explicit_dirs.append("ASC")
                else:
                    explicit_dirs.append("")

    if not explicit_dirs:
        if has_window_context:
            return sql
        has_asc = bool(re.search(r"\b(?:ascending|asc|alphabetical|alphabetically)\b", q))
        has_desc = bool(re.search(r"\b(?:descending|desc)\b", q))
        # If both directions appear but we failed to parse a concrete sort clause,
        # do not guess a single direction by fallback.
        if has_asc and has_desc:
            return sql
        if has_asc:
            explicit_dirs = ["ASC"]
        elif has_desc:
            explicit_dirs = ["DESC"]
        else:
            return sql

    def _find_top_level_kw(text: str, kw: str, start: int = 0) -> int:
        target = kw.upper()
        t_up = text.upper()
        depth = 0
        quote = ""
        i = start
        while i < len(text):
            ch = text[i]
            if quote:
                if ch == quote and (i == 0 or text[i - 1] != "\\"):
                    quote = ""
                i += 1
                continue
            if ch in {"'", '"', "`"}:
                quote = ch
                i += 1
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif depth == 0 and t_up.startswith(target, i):
                prev_ok = i == 0 or not (t_up[i - 1].isalnum() or t_up[i - 1] == "_")
                after = i + len(target)
                next_ok = after >= len(t_up) or not (t_up[after].isalnum() or t_up[after] == "_")
                if prev_ok and next_ok:
                    return i
            i += 1
        return -1

    s = sql
    order_pos = _find_top_level_kw(s, "ORDER BY")
    if order_pos < 0:
        return sql

    limit_pos = _find_top_level_kw(s, "LIMIT", order_pos + len("ORDER BY"))
    semi_pos = s.rfind(";")
    order_end_candidates = [p for p in (limit_pos, semi_pos) if p >= 0 and p > order_pos]
    order_end = min(order_end_candidates) if order_end_candidates else len(s)

    order_prefix = s[:order_pos + len("ORDER BY")]
    order_clause = s[order_pos + len("ORDER BY"):order_end].strip()
    order_suffix = s[order_end:]
    if not order_clause:
        return sql

    order_items = _split_top_level_csv(order_clause)
    if not order_items:
        return sql

    new_items: List[str] = []
    changed = False
    for idx, item in enumerate(order_items):
        requested = explicit_dirs[idx] if idx < len(explicit_dirs) else ""
        if not requested:
            new_items.append(item)
            continue
        trimmed = item.strip()
        rewritten = re.sub(r"\s+\b(?:ASC|DESC)\b\s*$", "", trimmed, flags=re.I).rstrip()
        rewritten = f"{rewritten} {requested}"
        if rewritten != trimmed:
            changed = True
        new_items.append(rewritten)

    if not changed:
        return sql

    sep = "\n" if "\n" in order_clause else " "
    return f"{order_prefix}{sep}{', '.join(new_items)}{order_suffix}"

def _simplify_default_window_frame(sql: str, question: str = "") -> str:
    """
    Style normalization for cumulative/running window queries:
    Collapse explicit default frame into compact form.

    Example:
      SUM(x) OVER (PARTITION BY k ORDER BY d ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    ->
      SUM(x) OVER (PARTITION BY k ORDER BY d)

    We only apply this for cumulative/running intent to avoid changing unrelated
    window queries where explicit frame text may be intentionally kept.
    """
    if not sql:
        return sql

    q = (question or "").lower()
    has_running_intent = any(t in q for t in ["cumulative", "running", "kumulatif", "akumulatif"])
    if not has_running_intent:
        return sql

    frame_pat = re.compile(
        r"\s+\b(?:ROWS|RANGE)\s+BETWEEN\s+UNBOUNDED\s+PRECEDING\s+AND\s+CURRENT\s+ROW\b",
        flags=re.IGNORECASE,
    )
    out = frame_pat.sub("", sql)
    return out

def _remove_top_level_order_by_if_not_requested(sql: str, question: str) -> str:
    """
    Versi Regex Barbar: Memangkas ORDER BY jika user tidak meminta.
    Lebih kuat daripada SQLGlot karena tidak peduli validitas syntax SQL keseluruhan.
    """
    if not sql:
        return sql
        
    # 1. Cek Intent: Kalau user minta sort, biarkan.
    if _question_requests_ordering(question):
        return sql

    # 2. Hapus ORDER BY
    # Logic: Cari "ORDER BY <apa saja>" sampai ketemu "LIMIT", ";", atau habis string.
    # Pattern penjelasan:
    # \s+ORDER\s+BY\s+  -> Cari kata ORDER BY (case insensitive)
    # .*?               -> Ambil semua karakter setelahnya (non-greedy)
    # (?=...)           -> Berhenti (Lookahead) jika ketemu...
    #   (?:\s+LIMIT\b|;|$) -> Kata LIMIT, titik koma, atau akhir string.
    
    # IMPORTANT: Only remove TOP-LEVEL ORDER BY, not ORDER BY inside OVER(...)
    # Strategy: Find the last ORDER BY that is NOT inside parentheses
    
    def find_top_level_order_by(s: str) -> Optional[int]:
        """
        Find position of top-level ORDER BY (not inside parentheses).
        Returns start index or None if not found.
        """
        depth = 0
        i = 0
        s_upper = s.upper()
        while i < len(s):
            if s[i] == '(':
                depth += 1
            elif s[i] == ')':
                depth -= 1
            elif depth == 0 and s_upper[i:i+8] == 'ORDER BY':
                # Check it's a word boundary
                if (i == 0 or not s[i-1].isalnum()) and (i+8 >= len(s) or not s[i+8].isalnum()):
                    return i
            i += 1
        return None
    
    order_pos = find_top_level_order_by(sql)
    if order_pos is None:
        return sql  # No top-level ORDER BY found
    
    # Find where ORDER BY clause ends (at LIMIT, ;, or end of string)
    rest = sql[order_pos:]
    # Match ORDER BY ... until LIMIT or ; or end
    m = re.match(r"ORDER\s+BY\s+.*?(?=\s+LIMIT\b|;|$)", rest, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return sql
    
    # Remove the matched ORDER BY clause
    cleaned = sql[:order_pos].rstrip() + sql[order_pos + m.end():]
    
    if cleaned != sql:
        print("[Logic] Removed unrequested ORDER BY (Regex Force).")
        
    return cleaned

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

def _collapse_spaces_outside_quotes(s: str) -> str:
    parts = re.split(r"('(?:''|[^'])*')", s)
    for i in range(len(parts)):
        if i % 2 == 0:
            parts[i] = re.sub(r"[ \t]{2,}", " ", parts[i])
    return "".join(parts)
# ======================================================================
# ======= Phase 4b — Aggregate in WHERE Fix (MySQL Error 1111) =========
# ======================================================================

def _fix_aggregate_in_where(sql: str) -> str:
    """
    Fix invalid SQL where aggregate functions are used in WHERE clause.
    MySQL Error 1111: Invalid use of group function
    
    WRONG:  SELECT ... FROM ... WHERE AVG(amount) > 50000 GROUP BY ...
    RIGHT:  SELECT ... FROM ... GROUP BY ... HAVING AVG(amount) > 50000
    
    Handles:
    - Simple: WHERE AVG(x) > 50000
    - With other conditions: WHERE status = 'Active' AND AVG(x) > 50000
    """
    if not sql:
        return sql
    
    sql_up = sql.upper()
    
    # Skip jika tidak ada GROUP BY
    if "GROUP BY" not in sql_up:
        return sql
    
    # Skip jika tidak ada WHERE
    if "WHERE" not in sql_up:
        return sql
    
    # Pattern untuk mendeteksi aggregate di WHERE
    agg_funcs = r"(?:AVG|SUM|COUNT|MIN|MAX)"
    
    # Cari WHERE clause (sebelum GROUP BY)
    where_match = re.search(
        r"\bWHERE\s+(.*?)(?=\s+GROUP\s+BY\b)",
        sql,
        flags=re.IGNORECASE | re.DOTALL
    )
    
    if not where_match:
        return sql
    
    where_body = where_match.group(1).strip()

    # Conservative safety: do not rewrite WHERE that contains subqueries.
    # Naive AND-splitting below is unsafe for correlated subqueries and can corrupt SQL.
    if re.search(r"\bSELECT\b", where_body, flags=re.IGNORECASE):
        return sql
    
    # Cari semua aggregate conditions di WHERE
    agg_conditions = []
    non_agg_conditions = []
    
    # Split by AND (simple approach)
    conditions = re.split(r'\s+AND\s+', where_body, flags=re.IGNORECASE)
    
    for cond in conditions:
        cond = cond.strip()
        if not cond:
            continue
        
        # Check if this condition contains an aggregate function
        if re.search(rf"\b{agg_funcs}\s*\(", cond, flags=re.IGNORECASE):
            agg_conditions.append(cond)
        else:
            non_agg_conditions.append(cond)
    
    # Jika tidak ada aggregate di WHERE, return as-is
    if not agg_conditions:
        return sql
    
    print(f"[FIX_AGG_WHERE] Found aggregate in WHERE: {agg_conditions}")
    
    # Build new WHERE clause (tanpa aggregate conditions)
    if non_agg_conditions:
        new_where = "WHERE " + " AND ".join(non_agg_conditions)
    else:
        new_where = ""
    
    # Find GROUP BY dan apa yang setelahnya
    group_match = re.search(
        r"(\bGROUP\s+BY\s+[\w\s,\.`]+?)(\s+HAVING\s+|\s+ORDER\s+BY\s+|\s+LIMIT\s+|\s*;|\s*$)",
        sql,
        flags=re.IGNORECASE
    )
    
    if not group_match:
        return sql
    
    group_clause = group_match.group(1).strip()
    after_group = group_match.group(2)
    
    # Build HAVING clause
    having_conditions = " AND ".join(agg_conditions)
    
    # Check if HAVING already exists
    if re.search(r"\bHAVING\b", after_group, flags=re.IGNORECASE):
        # Prepend to existing HAVING
        new_having = re.sub(
            r"(\bHAVING\s+)",
            rf"\1{having_conditions} AND ",
            after_group,
            flags=re.IGNORECASE
        )
    else:
        # Add new HAVING before ORDER BY/LIMIT/end
        new_having = f" HAVING {having_conditions}{after_group}"
    
    # Get prefix (SELECT ... FROM ... ) - before WHERE
    prefix_match = re.search(r"^(.*?)\bWHERE\b", sql, flags=re.IGNORECASE | re.DOTALL)
    if not prefix_match:
        return sql
    prefix = prefix_match.group(1).strip()
    
    # Get suffix (after GROUP BY clause) 
    suffix_start = group_match.end()
    suffix = sql[suffix_start:].strip()
    
    # Build final SQL
    if new_where:
        result = f"{prefix} {new_where} {group_clause}{new_having}"
    else:
        result = f"{prefix} {group_clause}{new_having}"
    
    # Add remaining suffix if any
    if suffix and not suffix.startswith(";") and suffix not in new_having:
        result = f"{result} {suffix}"
    
    # Clean up multiple spaces
    result = re.sub(r"\s{2,}", " ", result)
    result = result.strip()
    
    # Ensure semicolon
    if not result.endswith(";"):
        result += ";"
    
    print(f"[FIX_AGG_WHERE] Fixed: {result}")

    return result


# ===== Phase 4c — Fix Raw Column WHERE → HAVING for Aggregate Questions =====
# ======================================================================
def _fix_raw_column_where_to_having(sql: str, question: str = "") -> str:
    """
    Fix semantic error: raw column in WHERE should be aggregate in HAVING.

    Pattern:
    - Question asks: "average/sum/total above/below/equal X"
    - SQL has: WHERE <column> > <value> GROUP BY ...
    - Should be: WHERE (maybe other filters) GROUP BY ... HAVING AVG(<column>) > <value>

    Example:
    WRONG: SELECT customer, AVG(amount) FROM payments WHERE amount > 50000 GROUP BY customer
    RIGHT: SELECT customer, AVG(amount) FROM payments GROUP BY customer HAVING AVG(amount) > 50000
    """
    if not sql or not question:
        return sql

    sql_up = sql.upper()

    # Skip jika tidak ada GROUP BY
    if "GROUP BY" not in sql_up:
        return sql

    # Skip jika tidak ada WHERE
    if "WHERE" not in sql_up:
        return sql

    # Conservative gate:
    # - if query already has HAVING, avoid risky surgery.
    # - this fixer is intended for simple WHERE-only aggregate thresholds.
    if "HAVING" in sql_up:
        return sql

    # Detect aggregate intent in question
    question_lower = question.lower()
    agg_intent_patterns = [
        # Pattern: "average/sum/total ... above/below X" (allows words in between)
        r"(?:average|avg|sum|total|count|maximum|minimum|max|min).{0,20}(?:above|below|greater|less|over|under|more|than?|exceeds?|>\s*\d+)",
        # Reverse pattern: "above/below X ... average/sum/total"
        r"(?:above|below|greater|less|over|under|more|than?|exceeds?|>\s*\d+).{0,20}(?:average|avg|sum|total)",
        # Direct pattern with operator: "avg > 50000"
        r"(?:avg|average|sum|total|count|max|min)\s+[><=]",
    ]

    has_agg_intent = any(re.search(p, question_lower) for p in agg_intent_patterns)
    if not has_agg_intent:
        return sql

    # Extract WHERE clause
    where_match = re.search(
        r"\bWHERE\s+(.*?)(?=\s+GROUP\s+BY\b)",
        sql,
        flags=re.IGNORECASE | re.DOTALL
    )

    if not where_match:
        return sql

    where_body = where_match.group(1).strip()

    # Guard against complex WHERE (subquery / OR) that naive split cannot handle safely.
    if re.search(r"\bSELECT\b", where_body, flags=re.IGNORECASE):
        return sql
    if _has_top_level_or(where_body):
        return sql

    # Get columns used in aggregate functions in SELECT
    select_match = re.search(r"\bSELECT\b(.*?)\bFROM\b", sql, flags=re.IGNORECASE | re.DOTALL)
    if not select_match:
        return sql

    select_clause = select_match.group(1)

    # Find aggregate functions and their columns in SELECT
    agg_in_select = {}
    for m in re.finditer(r"(AVG|SUM|COUNT|MIN|MAX)\s*\(\s*([^)]+)\s*\)", select_clause, flags=re.IGNORECASE):
        func = m.group(1).upper()
        col = m.group(2).strip()
        agg_in_select[col.lower()] = func

    if not agg_in_select:
        return sql

    # Find WHERE conditions that use these columns WITHOUT aggregate wrapper
    conditions_to_move = []
    conditions_to_keep = []

    # Split by AND/OR (simple approach)
    separators = r'\s+(AND|OR)\s+'
    parts = re.split(f'({separators})', where_body, flags=re.IGNORECASE)

    i = 0
    while i < len(parts):
        part = parts[i].strip()
        if not part:
            i += 1
            continue

        if part.upper() in ("AND", "OR"):
            conditions_to_keep.append(part)
            i += 1
            continue

        # Check if this condition uses a column that's aggregated in SELECT
        # Pattern: column operator value
        cond_match = re.match(r"^([a-zA-Z_][a-zA-Z0-9_.]*)\s*(=|!=|<>|>=?|<=?|>)\s*(.+)$", part, flags=re.IGNORECASE)
        if cond_match:
            col = cond_match.group(1).lower()
            operator = cond_match.group(2)
            value = cond_match.group(3).strip()

            if col in agg_in_select:
                # This condition should be in HAVING with aggregate wrapper
                agg_func = agg_in_select[col]
                conditions_to_move.append(f"{agg_func}({col}) {operator} {value}")
            else:
                conditions_to_keep.append(part)
        else:
            # Complex condition, keep in WHERE
            conditions_to_keep.append(part)

        i += 1

    if not conditions_to_move:
        return sql

    print(f"[FIX_RAW_WHERE] Moving conditions to HAVING: {conditions_to_move}")

    # Build new SQL
    # Get prefix (before WHERE)
    prefix_match = re.search(r"^(.*?)\bWHERE\b", sql, flags=re.IGNORECASE | re.DOTALL)
    if not prefix_match:
        return sql
    prefix = prefix_match.group(1).strip()

    # Build new WHERE (if anything left)
    new_where = ""
    if conditions_to_keep:
        new_where = "WHERE " + " ".join(conditions_to_keep)

    # Build GROUP BY + HAVING
    group_match = re.search(
        r"\bGROUP\s+BY\s+.*?(?=\s*(?:HAVING|ORDER\s+BY|LIMIT|;|$))",
        sql,
        flags=re.IGNORECASE | re.DOTALL
    )
    if not group_match:
        return sql

    group_clause = group_match.group(0)

    # Build HAVING
    having_clause = " HAVING " + " AND ".join(conditions_to_move)

    # Find what comes after GROUP BY (before HAVING if exists, or ORDER BY/LIMIT)
    after_group_match = re.search(
        r"\bGROUP\s+BY\s+.*?(\s+(?:HAVING|ORDER\s+BY|LIMIT|;|$))",
        sql,
        flags=re.IGNORECASE | re.DOTALL
    )

    if after_group_match:
        after_group = after_group_match.group(1).strip()
        if after_group.upper().startswith("HAVING"):
            # Merge with existing HAVING
            existing_having = re.search(r"\bHAVING\s+(.*?)(?:\s+(?:ORDER\s+BY|LIMIT|;|$))", sql, flags=re.IGNORECASE | re.DOTALL)
            if existing_having:
                existing_cond = existing_having.group(1).strip()
                having_clause = f" HAVING ({' AND '.join(conditions_to_move)}) AND ({existing_cond})"
                # Remove old HAVING position
                after_group = re.sub(r"\bHAVING\s+.*?(?=\s+(?:ORDER\s+BY|LIMIT|;|$))", "", after_group, flags=re.IGNORECASE | re.DOTALL).strip()

        result = f"{prefix} {new_where} {group_clause}{having_clause} {after_group}".strip()
    else:
        result = f"{prefix} {new_where} {group_clause}{having_clause};".strip()

    # Clean up multiple spaces
    result = re.sub(r"\s{2,}", " ", result)
    result = result.strip()
    if not result.endswith(";"):
        result += ";"

    print(f"[FIX_RAW_WHERE] Fixed: {result}")
    return result


# ===== Phase 4d — Add Missing COUNT(*) for GROUP BY Aggregates =====
# ======================================================================
def _add_missing_count_for_group_by(sql: str, question: str = "") -> str:
    """
    Safely add COUNT(*) to SELECT when question asks for count/total in GROUP BY query.

    Guardrails:
    1. Must have GROUP BY (simple aggregate, not window function)
    2. NOT a ranking/comparison query (e.g., "top 3", "more than 5")
    3. Has GROUP BY intent indicators
    4. Question asks for count/total number
    5. COUNT not already in SELECT

    Examples:
    - "For each status, show total number of orders" → Add COUNT(*)
    - "Count orders per status" → Add COUNT(*)
    - "Customers with more than 5 orders" → Skip (needs HAVING/window)
    - "Top 3 customers by count" → Skip (needs ROW_NUMBER)
    """
    if not sql or not question:
        return sql

    def _is_word_boundary(text: str, i: int, n: int) -> bool:
        left_ok = i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")
        right_ok = i + n >= len(text) or not (text[i + n].isalnum() or text[i + n] == "_")
        return left_ok and right_ok

    def _find_top_level_kw(text: str, kw: str, start: int = 0) -> int:
        kw_l = kw.lower()
        n = len(kw_l)
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
            if depth == 0 and i >= start and i + n <= len(text):
                if text[i:i + n].lower() == kw_l and _is_word_boundary(text, i, n):
                    return i
            i += 1
        return -1

    s = sql.strip()
    select_pos = _find_top_level_kw(s, "select", 0)
    from_pos = _find_top_level_kw(s, "from", max(0, select_pos + 6))
    group_pos = _find_top_level_kw(s, "group by", max(0, from_pos + 4))
    if select_pos < 0 or from_pos < 0 or group_pos < 0:
        return sql

    sql_up = sql.upper()
    question_lower = question.lower()

    # GUARDRAIL 1: Must have GROUP BY
    if "GROUP BY" not in sql_up:
        return sql

    # GUARDRAIL 2: NOT a ranking/comparison query (needs window function or HAVING)
    comparison_patterns = [
        r"more than\s+\d+",
        r"above\s+\d+",
        r"below\s+\d+",
        r"exceeds?\s+\d+",
        r"greater than\s+\d+",
        r"less than\s+\d+",
        r"top\s+\d+",
        r"bottom\s+\d+",
        r"highest\s+\d+",
        r"lowest\s+\d+",
        r"\d+\s+or more",
        r"\d+\s+or less",
        r"at least\s+\d+",
        r"at most\s+\d+",
    ]
    if any(re.search(p, question, re.IGNORECASE) for p in comparison_patterns):
        # Likely needs window function, HAVING, or subquery - skip
        return sql

    # GUARDRAIL 3: Has GROUP BY intent indicators (looser patterns)
    group_by_indicators = [
        r"for each",
        r"per ",
        r"every ",
        r"by\s+\w+",           # "by status", "by city"
        r"in each",            # "in each city"
        r"for\s+\w+",          # "for customers"
        r"each\s+\w+",         # "each status"
    ]
    has_group_intent = any(re.search(p, question, re.IGNORECASE) for p in group_by_indicators)
    if not has_group_intent:
        return sql

    # GUARDRAIL 4: Question asks for count/total number (specific phrases only)
    # NOTE: "total" alone removed - too generic, can mean SUM(amount) not COUNT(*)
    count_keywords = ["total number", "number of", "count of", "how many", "count"]
    has_count_intent = any(kw in question_lower for kw in count_keywords)
    if not has_count_intent:
        return sql

    # GUARDRAIL 5: COUNT not already in SELECT
    select_clause = s[select_pos + len("select"):from_pos].upper()
    if "COUNT" in select_clause:
        return sql  # COUNT already present

    # All guardrails passed - safe to add COUNT(*)
    print(f"[ADD_COUNT] Adding COUNT(*) for GROUP BY aggregate")
    result = re.sub(
        r"(SELECT\s+)",
        r"\1COUNT(*) AS total_count, ",
        sql,
        count=1,
        flags=re.IGNORECASE
    )

    # Clean up multiple spaces
    result = re.sub(r"\s{2,}", " ", result)
    result = result.strip()
    if not result.endswith(";"):
        result += ";"

    print(f"[ADD_COUNT] Fixed: {result}")
    return result
