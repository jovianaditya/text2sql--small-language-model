# ======================================================================
# =========================== sql_planner.py ============================
# ======================================================================
# Purpose:
# - Deterministic SQL planner from Intent IR + schema graph.
# - Fanout-safe strategy: aggregate child tables by subject key first.
# - Safety validation loop before handing SQL to execution/refinement.

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .schema_analyzer import get_fk_relationships, get_id_column, get_schema_analyzer


def _safe_get_fk_relationships() -> Dict[str, Dict[str, str]]:
    try:
        return get_fk_relationships() or {}
    except Exception:
        return {}


def _safe_get_id_column(table: str) -> Optional[str]:
    try:
        return get_id_column(table or "")
    except Exception:
        return None


def _parse_allowed_identifiers(ctx: str) -> Dict[str, Set[str]]:
    tables: Dict[str, Set[str]] = {}
    m = re.search(r"### ALLOWED IDENTIFIERS(.*?)(?:\n### |\Z)", ctx or "", flags=re.S)
    if not m:
        return tables

    block = m.group(1)
    for line in block.splitlines():
        s = line.strip()
        if not s.startswith("- "):
            continue
        table_col = s[2:].split(":", 1)
        if len(table_col) != 2:
            continue
        table = table_col[0].strip().strip("`").lower()
        cols = [c.strip().strip("`") for c in table_col[1].split(",") if c.strip()]
        if table:
            tables.setdefault(table, set()).update(cols)
    return tables


def _load_allowed_or_schema(ctx: str) -> Dict[str, Set[str]]:
    allowed = _parse_allowed_identifiers(ctx)
    if allowed:
        return allowed

    out: Dict[str, Set[str]] = {}
    try:
        analyzer = get_schema_analyzer()
        for t in analyzer.get_tables():
            cols = {str(c) for c in analyzer.get_table_columns(t) if str(c)}
            if cols:
                out[str(t).lower()] = cols
    except Exception:
        pass
    return out


def _quote_ident(s: str) -> str:
    x = str(s or "").strip().strip("`")
    return f"`{x}`"


def _quote_literal(v: Any) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v).replace("'", "''")
    return f"'{s}'"


def _id_like_column(table: str, allowed: Dict[str, Set[str]]) -> Optional[str]:
    id_col = _safe_get_id_column(table or "")
    if id_col and id_col in allowed.get((table or "").lower(), set()):
        return id_col

    cols = list(allowed.get((table or "").lower(), set()))
    for c in cols:
        cl = c.lower()
        if cl.endswith("id") or cl.endswith("number") or cl.endswith("code"):
            return c
    return cols[0] if cols else None


def _build_fk_maps(allowed_cols: Dict[str, Set[str]]) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Set[str]]]:
    allowed = {str(t).lower() for t in (allowed_cols or {}).keys() if str(t)}
    fk_raw = _safe_get_fk_relationships()

    fk_map: Dict[str, Dict[str, str]] = {}
    adj: Dict[str, Set[str]] = {}
    for src, fmap in fk_raw.items():
        s = str(src).lower()
        if s not in allowed:
            continue
        for fk_col, tgt in (fmap or {}).items():
            t = str(tgt).lower()
            if t not in allowed:
                continue
            fk_map.setdefault(s, {})[str(fk_col)] = t
            adj.setdefault(s, set()).add(t)
            adj.setdefault(t, set()).add(s)

    # Fallback graph edges from shared id-like columns.
    # This keeps planner operational when FK metadata is sparse/missing.
    tbls = sorted(allowed)
    for i, a in enumerate(tbls):
        cols_a = {c.lower(): c for c in allowed_cols.get(a, set())}
        id_like_a = {
            k for k in cols_a
            if k.endswith("id") or k.endswith("number") or k.endswith("code")
        }
        for b in tbls[i + 1:]:
            cols_b = {c.lower(): c for c in allowed_cols.get(b, set())}
            id_like_b = {
                k for k in cols_b
                if k.endswith("id") or k.endswith("number") or k.endswith("code")
            }
            if id_like_a.intersection(id_like_b):
                adj.setdefault(a, set()).add(b)
                adj.setdefault(b, set()).add(a)
    return fk_map, adj


def _find_path(start: str, end: str, adj: Dict[str, Set[str]]) -> Optional[List[str]]:
    s = (start or "").lower()
    e = (end or "").lower()
    if not s or not e:
        return None
    if s == e:
        return [s]
    if s not in adj or e not in adj:
        return None

    queue: List[Tuple[str, List[str]]] = [(s, [s])]
    seen = {s}
    while queue:
        cur, path = queue.pop(0)
        for nxt in adj.get(cur, set()):
            if nxt == e:
                return path + [nxt]
            if nxt in seen:
                continue
            seen.add(nxt)
            queue.append((nxt, path + [nxt]))
    return None


def _find_fk_col_between(child: str, parent: str, fk_map: Dict[str, Dict[str, str]]) -> Optional[str]:
    for fk_col, tgt in (fk_map.get((child or "").lower(), {}) or {}).items():
        if str(tgt).lower() == (parent or "").lower():
            return fk_col
    return None


def _join_cond_between(
    left_table: str,
    left_alias: str,
    right_table: str,
    right_alias: str,
    fk_map: Dict[str, Dict[str, str]],
    allowed: Dict[str, Set[str]],
) -> Optional[str]:
    # left -> right (left is child)
    fk_l = _find_fk_col_between(left_table, right_table, fk_map)
    if fk_l:
        right_pk = _id_like_column(right_table, allowed)
        if right_pk:
            return f"{left_alias}.{_quote_ident(fk_l)} = {right_alias}.{_quote_ident(right_pk)}"

    # right -> left (right is child)
    fk_r = _find_fk_col_between(right_table, left_table, fk_map)
    if fk_r:
        left_pk = _id_like_column(left_table, allowed)
        if left_pk:
            return f"{right_alias}.{_quote_ident(fk_r)} = {left_alias}.{_quote_ident(left_pk)}"

    # Fallback: shared id-like column.
    left_cols = {c.lower(): c for c in allowed.get(left_table.lower(), set())}
    right_cols = {c.lower(): c for c in allowed.get(right_table.lower(), set())}
    shared = [k for k in left_cols.keys() if k in right_cols]
    for c_low in shared:
        if c_low.endswith("id") or c_low.endswith("number") or c_low.endswith("code"):
            return (
                f"{left_alias}.{_quote_ident(left_cols[c_low])} = "
                f"{right_alias}.{_quote_ident(right_cols[c_low])}"
            )
    return None


# ======================================================================
# ================== FK PATH VALIDATION (SCHEMA-AGNOSTIC) ==============
# ======================================================================

def validate_join_paths(sql: str, ctx: str = "") -> Dict[str, Any]:
    """
    Validate that all JOINs in SQL use valid FK relationships.
    
    Schema-agnostic - uses FK metadata from schema analyzer.
    Falls back to shared ID column detection when FK metadata is sparse.
    
    Args:
        sql: SQL query to validate
        ctx: Context string with ALLOWED IDENTIFIERS
    
    Returns:
        Dict with:
        - valid: bool - True if all join paths are valid
        - errors: List[str] - Description of invalid joins
        - warnings: List[str] - Non-fatal issues
        - join_count: int - Number of joins found
        - validated_pairs: List[Tuple[str, str]] - Successfully validated table pairs
    """
    result: Dict[str, Any] = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "join_count": 0,
        "validated_pairs": [],
    }
    
    if not sql:
        return result
    
    s_up = sql.upper()
    if "JOIN" not in s_up:
        return result
    
    # Extract tables from SQL
    table_pattern = r'\b(?:FROM|JOIN)\s+(`?[A-Za-z_][A-Za-z0-9_]*`?)(?:\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*))?'
    matches = list(re.finditer(table_pattern, sql, flags=re.I))
    
    if len(matches) < 2:
        return result
    
    # Build table list (table_name, alias)
    tables: List[Tuple[str, str]] = []
    for m in matches:
        table = m.group(1).strip('`').lower()
        alias = (m.group(2) or table).lower()
        tables.append((table, alias))
    
    # Get FK relationships from schema
    fk_rels = _safe_get_fk_relationships()
    
    # Build adjacency graph (schema-agnostic)
    adjacent: Set[Tuple[str, str]] = set()
    
    # Add explicit FK relationships
    for src_table, fk_map in fk_rels.items():
        for fk_col, target_table in fk_map.items():
            adjacent.add((src_table.lower(), target_table.lower()))
            adjacent.add((target_table.lower(), src_table.lower()))
    
    # Add implicit relationships (shared ID columns)
    allowed = _parse_allowed_identifiers(ctx)
    table_names = [t[0] for t in tables]
    
    for i, t1 in enumerate(table_names):
        cols1 = set(c.lower() for c in allowed.get(t1, set()))
        if not cols1:
            try:
                analyzer = get_schema_analyzer()
                cols1 = set(c.lower() for c in analyzer.get_table_columns(t1))
            except Exception:
                pass
        
        id_cols1 = {c for c in cols1 if c.endswith(('id', 'number', 'code'))}
        
        for t2 in table_names[i+1:]:
            cols2 = set(c.lower() for c in allowed.get(t2, set()))
            if not cols2:
                try:
                    analyzer = get_schema_analyzer()
                    cols2 = set(c.lower() for c in analyzer.get_table_columns(t2))
                except Exception:
                    pass
            
            id_cols2 = {c for c in cols2 if c.endswith(('id', 'number', 'code'))}
            
            if id_cols1.intersection(id_cols2):
                adjacent.add((t1, t2))
                adjacent.add((t2, t1))
    
    # Count actual JOINs
    join_count = len(re.findall(r'\bJOIN\b', sql, flags=re.I))
    result["join_count"] = join_count
    
    # Validate each consecutive pair of tables
    errors: List[str] = []
    warnings: List[str] = []
    validated_pairs: List[Tuple[str, str]] = []
    
    for i in range(len(table_names) - 1):
        t1 = table_names[i]
        t2 = table_names[i + 1]
        
        # Check if tables are connected
        if (t1, t2) in adjacent or (t2, t1) in adjacent:
            validated_pairs.append((t1, t2))
        else:
            # Check if there's a path via BFS
            fk_map, adj = _build_fk_maps(allowed)
            path = _find_path(t1, t2, adj)
            
            if path:
                if len(path) > 2:
                    warnings.append(f"Tables {t1} and {t2} require intermediate tables: {' -> '.join(path)}")
                validated_pairs.append((t1, t2))
            else:
                errors.append(f"No FK relationship between {t1} and {t2}")
    
    result["errors"] = errors
    result["warnings"] = warnings
    result["validated_pairs"] = validated_pairs
    result["valid"] = len(errors) == 0
    
    return result


def suggest_join_path(
    source_table: str,
    target_table: str,
    ctx: str = "",
) -> Optional[List[str]]:
    """
    Suggest optimal FK path between two tables.
    
    Schema-agnostic - uses FK metadata and shared ID column detection.
    
    Args:
        source_table: Starting table
        target_table: Ending table
        ctx: Context string with ALLOWED IDENTIFIERS
    
    Returns:
        List of table names forming the path, or None if no path exists
    """
    s = (source_table or "").lower().strip()
    t = (target_table or "").lower().strip()
    
    if not s or not t:
        return None
    
    if s == t:
        return [s]
    
    allowed = _parse_allowed_identifiers(ctx)
    fk_map, adj = _build_fk_maps(allowed)
    
    return _find_path(s, t, adj)


def _render_filter_sql(f: Dict[str, Any], alias: str) -> Optional[str]:
    col = str(f.get("column") or "").strip()
    op = str(f.get("op") or "").strip().lower()
    if not col or not op:
        return None

    ref = f"{alias}.{_quote_ident(col)}"
    other_col = str(f.get("value_column") or "").strip()
    other_ref = f"{alias}.{_quote_ident(other_col)}" if other_col else ""
    v = f.get("value")
    v2 = f.get("value2")

    if op == "eq":
        return f"{ref} = {_quote_literal(v)}"
    if op == "gt":
        return f"{ref} > {_quote_literal(v)}"
    if op == "gte":
        return f"{ref} >= {_quote_literal(v)}"
    if op == "lt":
        return f"{ref} < {_quote_literal(v)}"
    if op == "lte":
        return f"{ref} <= {_quote_literal(v)}"
    if op == "between":
        return f"{ref} BETWEEN {_quote_literal(v)} AND {_quote_literal(v2)}"
    if op == "in":
        values = v if isinstance(v, list) else []
        if not values:
            return None
        inside = ", ".join(_quote_literal(x) for x in values)
        return f"{ref} IN ({inside})"
    if op == "year_eq":
        return f"YEAR({ref}) = {_quote_literal(v)}"
    if op == "like_prefix":
        return f"{ref} LIKE {_quote_literal(str(v) + '%')}"
    if op == "regexp":
        return f"{ref} REGEXP {_quote_literal(v)}"
    if op == "is_null_or_empty":
        return f"({ref} IS NULL OR {ref} = '')"
    if op == "col_gt":
        return f"{ref} > {other_ref}" if other_ref else None
    if op == "col_gte":
        return f"{ref} >= {other_ref}" if other_ref else None
    if op == "col_lt":
        return f"{ref} < {other_ref}" if other_ref else None
    if op == "col_lte":
        return f"{ref} <= {other_ref}" if other_ref else None
    return None


_METRIC_THRESHOLD_OPS = {"eq", "gt", "gte", "lt", "lte", "between"}
_METRIC_THRESHOLD_AGGS = {"count", "sum", "avg", "min", "max"}


def _is_numeric_literal(v: Any) -> bool:
    if isinstance(v, (int, float)):
        return True
    s = str(v or "").strip().replace(",", "")
    if not s:
        return False
    try:
        float(s)
        return True
    except Exception:
        return False


def _filter_targets_metric_threshold(f: Dict[str, Any], metric: Dict[str, Any]) -> bool:
    if not isinstance(f, dict) or not isinstance(metric, dict):
        return False

    op = str(f.get("op") or "").strip().lower()
    if op not in _METRIC_THRESHOLD_OPS:
        return False

    agg = str(metric.get("agg") or "").strip().lower()
    if agg not in _METRIC_THRESHOLD_AGGS:
        return False

    if op == "between":
        if not (_is_numeric_literal(f.get("value")) and _is_numeric_literal(f.get("value2"))):
            return False
    elif not _is_numeric_literal(f.get("value")):
        return False

    f_table = str(f.get("table") or "").strip().lower()
    m_table = str(metric.get("table") or "").strip().lower()
    if not f_table or not m_table or f_table != m_table:
        return False

    f_col = str(f.get("column") or "").strip().lower()
    m_col = str(metric.get("column") or "").strip().lower()
    if not f_col or not m_col:
        return False
    if m_col == "*":
        return f_col == "*"
    return f_col == m_col


def _render_metric_having_sql(f: Dict[str, Any], metric: Dict[str, Any], source_alias: str) -> Optional[str]:
    if not _filter_targets_metric_threshold(f, metric):
        return None

    agg = str(metric.get("agg") or "").strip().lower()
    col = str(metric.get("column") or "*").strip()
    distinct = bool(metric.get("distinct"))
    col_sql = "*" if col == "*" else f"{source_alias}.{_quote_ident(col)}"
    agg_sql = _agg_expr(agg, col_sql, distinct=distinct)
    if not agg_sql:
        return None

    op = str(f.get("op") or "").strip().lower()
    v = f.get("value")
    v2 = f.get("value2")
    if op == "eq":
        return f"{agg_sql} = {_quote_literal(v)}"
    if op == "gt":
        return f"{agg_sql} > {_quote_literal(v)}"
    if op == "gte":
        return f"{agg_sql} >= {_quote_literal(v)}"
    if op == "lt":
        return f"{agg_sql} < {_quote_literal(v)}"
    if op == "lte":
        return f"{agg_sql} <= {_quote_literal(v)}"
    if op == "between":
        return f"{agg_sql} BETWEEN {_quote_literal(v)} AND {_quote_literal(v2)}"
    return None


def _agg_expr(agg: str, col_sql: str, distinct: bool = False) -> str:
    a = (agg or "").lower()
    if a == "count":
        if col_sql == "*":
            return "COUNT(*)"
        if distinct:
            return f"COUNT(DISTINCT {col_sql})"
        return f"COUNT({col_sql})"
    if a == "sum":
        return f"SUM({col_sql})"
    if a == "avg":
        return f"AVG({col_sql})"
    if a == "min":
        return f"MIN({col_sql})"
    if a == "max":
        return f"MAX({col_sql})"
    return ""


def _build_path_subquery(
    source_table: str,
    subject_table: str,
    subject_id_col: str,
    fk_map: Dict[str, Dict[str, str]],
    adj: Dict[str, Set[str]],
    allowed: Dict[str, Set[str]],
    filters: List[Dict[str, Any]],
    metric: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Build subquery that projects `__subject_key` for rows sourced from `source_table`.
    If metric is provided, subquery also computes aggregate value as metric alias.
    """
    src = (source_table or "").lower()
    subj = (subject_table or "").lower()
    path = _find_path(src, subj, adj)
    if not path:
        return None, {"error": f"no_fk_path:{src}->{subj}"}

    alias_by_table: Dict[str, str] = {}
    for i, t in enumerate(path):
        alias_by_table[t] = f"p{i}"

    from_sql = f"FROM {_quote_ident(path[0])} AS {alias_by_table[path[0]]}"
    joins: List[str] = []
    for i in range(len(path) - 1):
        left_t = path[i]
        right_t = path[i + 1]
        left_a = alias_by_table[left_t]
        right_a = alias_by_table[right_t]
        cond = _join_cond_between(left_t, left_a, right_t, right_a, fk_map, allowed)
        if not cond:
            return None, {"error": f"missing_join_condition:{left_t}-{right_t}"}
        joins.append(f"JOIN {_quote_ident(right_t)} AS {right_a} ON {cond}")

    subject_alias = alias_by_table[subj]
    subject_key_sql = f"{subject_alias}.{_quote_ident(subject_id_col)}"

    where_parts: List[str] = []
    having_parts: List[str] = []
    for f in filters:
        if str(f.get("table") or "").lower() not in alias_by_table:
            continue
        fa = alias_by_table[str(f.get("table")).lower()]
        if metric and str(f.get("table") or "").strip().lower() == src:
            having_pred = _render_metric_having_sql(f, metric, fa)
            if having_pred:
                having_parts.append(having_pred)
                continue
        pred = _render_filter_sql(f, fa)
        if pred:
            where_parts.append(pred)
    where_sql = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

    if metric:
        agg = str(metric.get("agg") or "").lower()
        col = str(metric.get("column") or "*").strip()
        alias_out = str(metric.get("alias") or "metric_value").strip()
        distinct = bool(metric.get("distinct"))

        source_alias = alias_by_table[src]
        col_sql = "*" if col == "*" else f"{source_alias}.{_quote_ident(col)}"
        agg_sql = _agg_expr(agg, col_sql, distinct=distinct)
        if not agg_sql:
            return None, {"error": "unsupported_aggregate"}

        having_parts = list(dict.fromkeys(having_parts))
        query = (
            "SELECT\n"
            f"  {subject_key_sql} AS __subject_key,\n"
            f"  {agg_sql} AS {_quote_ident(alias_out)}\n"
            f"{from_sql}\n"
            + ("\n".join(joins) + "\n" if joins else "")
            + (f"{where_sql}\n" if where_sql else "")
            + f"GROUP BY {subject_key_sql}"
            + (f"\nHAVING {' AND '.join(having_parts)}" if having_parts else "")
        )
        return query, {"path": path, "has_metric_having": bool(having_parts)}

    # Existence/filter subquery.
    query = (
        "SELECT DISTINCT\n"
        f"  {subject_key_sql} AS __subject_key\n"
        f"{from_sql}\n"
        + ("\n".join(joins) + "\n" if joins else "")
        + (f"{where_sql}" if where_sql else "")
    )
    return query, {"path": path}


def _render_dimension_expr(d: Dict[str, Any], base_alias: str) -> Optional[Tuple[str, Optional[str], bool]]:
    col = str(d.get("column") or "").strip()
    alias = str(d.get("alias") or "").strip()
    transform = str(d.get("transform") or "none").strip().lower()
    visible = bool(d.get("visible", True))
    if not col:
        return None

    col_ref = f"{base_alias}.{_quote_ident(col)}"
    if transform == "none":
        expr = col_ref
    elif transform == "month_name":
        expr = f"DATE_FORMAT({col_ref}, '%M')"
        if not alias:
            alias = "month_name"
    elif transform == "month_num":
        expr = f"MONTH({col_ref})"
        if not alias:
            alias = "__month_num"
    elif transform == "year":
        expr = f"YEAR({col_ref})"
        if not alias:
            alias = "year"
    else:
        return None

    if alias:
        return f"{expr} AS {_quote_ident(alias)}", expr, visible
    return expr, expr, visible


def _sql_has_banned_pattern(sql: str) -> bool:
    if not sql:
        return True
    s = sql.strip()
    # single-statement only
    if s.count(";") > 1:
        return True
    # no destructive/DDL patterns in generated SQL
    banned = (
        r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke)\b",
        r"--",
        r"/\*",
    )
    return any(re.search(p, s, flags=re.I) for p in banned)


def _format_with_sqlglot(sql: str) -> str:
    try:
        import sqlglot

        ast = sqlglot.parse_one(sql, read="mysql")
        return ast.sql(dialect="mysql")
    except Exception:
        return sql


def _strip_sql_wrappers(sql: str) -> Tuple[str, List[str]]:
    warnings: List[str] = []
    s = (sql or "").strip()
    if not s:
        return "", warnings

    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
        warnings.append("stripped_code_fences")

    lines = s.splitlines()
    if lines and lines[0].strip().lower() in {"sql", "mysql"}:
        s = "\n".join(lines[1:]).strip()
        warnings.append("stripped_sql_language_tag")

    prefix_match = re.match(r"^(?:sql|mysql|query)\s*:\s*", s, flags=re.I)
    if prefix_match:
        s = s[prefix_match.end() :].strip()
        warnings.append("stripped_sql_prefix_label")

    return s, warnings


def _extract_first_sql_statement(sql: str) -> Tuple[str, List[str]]:
    warnings: List[str] = []
    s = (sql or "").strip()
    if not s:
        return "", warnings

    start_match = re.search(r"(?:^|[\r\n])\s*((?:with|select)\b)", s, flags=re.I)
    if start_match and start_match.start(1) > 0:
        s = s[start_match.start(1):].strip()
        warnings.append("trimmed_non_sql_prefix")

    depth = 0
    in_single = False
    in_double = False
    in_backtick = False
    escaped = False

    for idx, ch in enumerate(s):
        if in_single:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == "'":
                in_single = False
            continue

        if in_double:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_double = False
            continue

        if in_backtick:
            if ch == "`":
                in_backtick = False
            continue

        if ch == "'":
            in_single = True
            continue
        if ch == '"':
            in_double = True
            continue
        if ch == "`":
            in_backtick = True
            continue
        if ch == "(":
            depth += 1
            continue
        if ch == ")" and depth > 0:
            depth -= 1
            continue
        if ch == ";" and depth == 0:
            tail = s[idx + 1 :].strip()
            if tail:
                warnings.append("truncated_trailing_sql_text")
            return s[: idx + 1].strip(), warnings

    return s, warnings


def _validate_generated_sql(sql: str, intent_ir: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if not sql:
        return False, ["empty_sql"]
    if _sql_has_banned_pattern(sql):
        return False, ["banned_pattern"]

    # Parse validation when sqlglot is available.
    try:
        import sqlglot
    except Exception:
        sqlglot = None
    if sqlglot is not None:
        try:
            sqlglot.parse_one(sql, read="mysql")
        except Exception as e:
            errors.append(f"parse_error:{e}")

    low = sql.lower()

    # Ensure requested output aliases/columns are present.
    for d in intent_ir.get("dimensions", []) or []:
        if not bool(d.get("visible", True)):
            continue
        alias = str(d.get("alias") or "").strip().lower()
        col = str(d.get("column") or "").strip().lower()
        if alias:
            if f" as `{alias}`".lower() not in low and f" as {alias}" not in low:
                errors.append(f"missing_dimension_alias:{alias}")
        elif col and col not in low:
            errors.append(f"missing_dimension_column:{col}")

    for m in intent_ir.get("metrics", []) or []:
        alias = str(m.get("alias") or "").strip().lower()
        if alias and (f" as `{alias}`" not in low and f" as {alias}" not in low):
            errors.append(f"missing_metric_alias:{alias}")

    include_zero = bool(intent_ir.get("include_zero", False))
    has_exists_requirement = any(
        str(e.get("mode") or "").lower() == "at_least_one"
        for e in (intent_ir.get("existence", []) or [])
    )
    metric_specs = [m for m in (intent_ir.get("metrics", []) or []) if isinstance(m, dict)]
    filter_specs = [f for f in (intent_ir.get("filters", []) or []) if isinstance(f, dict)]
    has_metric_threshold = any(
        _filter_targets_metric_threshold(f, m)
        for f in filter_specs
        for m in metric_specs
    )
    metric_tables = {
        str(m.get("table") or "").lower()
        for m in (intent_ir.get("metrics", []) or [])
        if str(m.get("table") or "").strip()
    }
    subject = str(intent_ir.get("subject_entity") or "").lower()
    child_metric_tables = {t for t in metric_tables if t and t != subject}

    # include_zero semantics should preserve outer subject rows.
    if include_zero and child_metric_tables and not has_exists_requirement and not has_metric_threshold:
        if not re.search(r"\bleft\s+join\b", low, flags=re.I):
            errors.append("include_zero_without_left_join")

    # Fanout guard: multiple child metric tables should be joined as aggregated subqueries.
    if len(child_metric_tables) >= 2:
        if low.count("__subject_key") < 2:
            errors.append("fanout_risk_missing_preaggregation")

    return (len(errors) == 0), errors


def _group_filters_by_table(filters: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for f in filters or []:
        t = str(f.get("table") or "").lower()
        if not t:
            continue
        out.setdefault(t, []).append(f)
    return out


def _validate_ir_identifiers(intent_ir: Dict[str, Any], allowed: Dict[str, Set[str]]) -> List[str]:
    errors: List[str] = []
    allowed_lower = {
        str(t).lower(): {str(c).lower() for c in (cols or set()) if str(c)}
        for t, cols in (allowed or {}).items()
        if str(t)
    }

    visible_aliases = {
        str(d.get("alias") or "").strip().lower()
        for d in (intent_ir.get("dimensions") or [])
        if isinstance(d, dict) and bool(d.get("visible", True)) and str(d.get("alias") or "").strip()
    }
    visible_aliases.update(
        str(m.get("alias") or "").strip().lower()
        for m in (intent_ir.get("metrics") or [])
        if isinstance(m, dict) and str(m.get("alias") or "").strip()
    )

    def _require_table(table: str, scope: str) -> bool:
        t = str(table or "").strip().lower()
        if not t:
            errors.append(f"{scope}:missing_table")
            return False
        if t not in allowed_lower:
            errors.append(f"{scope}:table_not_allowed:{t}")
            return False
        return True

    def _require_column(table: str, column: str, scope: str, allow_star: bool = False) -> bool:
        col = str(column or "").strip()
        if allow_star and col == "*":
            return True
        if not col:
            errors.append(f"{scope}:missing_column")
            return False

        t = str(table or "").strip().lower()
        if t not in allowed_lower:
            errors.append(f"{scope}:table_not_allowed:{t or 'unknown'}")
            return False
        if col.lower() not in allowed_lower[t]:
            errors.append(f"{scope}:column_not_allowed:{t}.{col}")
            return False
        return True

    subject = str(intent_ir.get("subject_entity") or "").strip().lower()
    if subject and subject not in allowed_lower:
        errors.append(f"subject_not_allowed:{subject}")

    for d in (intent_ir.get("dimensions") or []):
        if not isinstance(d, dict):
            continue
        table = str(d.get("table") or subject).strip().lower()
        if _require_table(table, "dimension"):
            _require_column(table, str(d.get("column") or ""), "dimension")

    for m in (intent_ir.get("metrics") or []):
        if not isinstance(m, dict):
            continue
        table = str(m.get("table") or "").strip().lower()
        if _require_table(table, "metric"):
            _require_column(table, str(m.get("column") or "*"), "metric", allow_star=True)

    for f in (intent_ir.get("filters") or []):
        if not isinstance(f, dict):
            continue
        table = str(f.get("table") or "").strip().lower()
        if _require_table(table, "filter"):
            _require_column(table, str(f.get("column") or ""), "filter")
            op = str(f.get("op") or "").strip().lower()
            if op in {"col_gt", "col_gte", "col_lt", "col_lte"}:
                _require_column(table, str(f.get("value_column") or ""), "filter_value_column")

    for ex in (intent_ir.get("existence") or []):
        if not isinstance(ex, dict):
            continue
        _require_table(str(ex.get("table") or "").strip().lower(), "existence")

    for o in (intent_ir.get("ordering") or []):
        if not isinstance(o, dict):
            continue
        kind = str(o.get("kind") or "").strip().lower()
        by = str(o.get("by") or "").strip()
        if not by:
            errors.append("ordering:missing_target")
            continue
        if kind == "column":
            table = str(o.get("table") or subject).strip().lower()
            if _require_table(table, "ordering"):
                _require_column(table, by, "ordering")
        elif kind == "alias" and by.lower() not in visible_aliases:
            errors.append(f"ordering:unknown_alias:{by}")

    return list(dict.fromkeys(errors))


def build_sql_from_intent(intent_ir: Dict[str, Any], ctx: str) -> Tuple[Optional[str], Dict[str, Any]]:
    meta: Dict[str, Any] = {"mode": "deterministic_ir", "errors": [], "warnings": [], "join_plan": {}}
    if not intent_ir:
        return None, {"mode": "deterministic_ir", "errors": ["empty_intent_ir"]}

    subject = str(intent_ir.get("subject_entity") or "").strip().lower()
    if not subject:
        return None, {"mode": "deterministic_ir", "errors": ["missing_subject_entity"]}

    allowed = _load_allowed_or_schema(ctx)
    if not allowed:
        return None, {"mode": "deterministic_ir", "errors": ["missing_schema_context"]}
    if subject not in allowed:
        return None, {"mode": "deterministic_ir", "errors": [f"subject_not_allowed:{subject}"]}

    identifier_errors = _validate_ir_identifiers(intent_ir, allowed)
    if identifier_errors:
        return None, {
            "mode": "deterministic_ir",
            "errors": identifier_errors,
            "warnings": ["planner_fallback_due_to_ir_identifier_validation"],
        }

    fk_map, adj = _build_fk_maps(allowed)
    subject_id_col = _id_like_column(subject, allowed)
    if not subject_id_col:
        return None, {"mode": "deterministic_ir", "errors": [f"subject_id_not_found:{subject}"]}

    dims = list(intent_ir.get("dimensions", []) or [])
    metrics = list(intent_ir.get("metrics", []) or [])
    filters = list(intent_ir.get("filters", []) or [])
    existence = list(intent_ir.get("existence", []) or [])
    ordering = list(intent_ir.get("ordering", []) or [])
    include_zero = bool(intent_ir.get("include_zero", False))
    select_distinct = bool(intent_ir.get("select_distinct", False))
    limit = intent_ir.get("limit")

    # Guard: planner currently supports dimensions on subject table only.
    for d in dims:
        dt = str(d.get("table") or subject).lower()
        if dt != subject:
            meta["errors"].append(f"unsupported_dimension_table:{dt}")
            return None, meta

    filters_by_table = _group_filters_by_table(filters)

    # Determine whether single-table aggregate mode is valid.
    all_metric_subject = bool(metrics) and all(str(m.get("table") or "").lower() == subject for m in metrics)
    only_subject_filters = all(str(f.get("table") or "").lower() == subject for f in filters)
    no_existence = not existence
    single_table_agg_mode = all_metric_subject and only_subject_filters and no_existence

    base_alias = "s"
    select_items: List[str] = []
    where_items: List[str] = []
    having_items: List[str] = []
    group_items: List[str] = []
    join_items: List[str] = []

    # Dimensions.
    month_expr_for_order = ""
    for d in dims:
        rendered = _render_dimension_expr(d, base_alias)
        if not rendered:
            meta["errors"].append(f"invalid_dimension:{d}")
            return None, meta
        select_expr, group_expr, visible = rendered
        if visible:
            select_items.append(select_expr)
        if metrics and group_expr:
            group_items.append(group_expr)
        if str(d.get("transform") or "").lower() == "month_num":
            month_expr_for_order = group_expr or ""

    # Subject filters.
    for f in filters_by_table.get(subject, []):
        pred = _render_filter_sql(f, base_alias)
        if pred:
            where_items.append(pred)

    # Query modes.
    if not metrics:
        # Lookup mode.
        if not select_items:
            # stable fallback projection
            id_col = _id_like_column(subject, allowed)
            if id_col:
                select_items.append(f"{base_alias}.{_quote_ident(id_col)}")
        query_from = f"FROM {_quote_ident(subject)} AS {base_alias}"

    elif single_table_agg_mode:
        # Single-table aggregate mode (e.g., orders per month).
        for m in metrics:
            agg = str(m.get("agg") or "").lower()
            col = str(m.get("column") or "*").strip()
            alias = str(m.get("alias") or "").strip() or f"{agg}_value"
            distinct = bool(m.get("distinct"))
            col_sql = "*" if col == "*" else f"{base_alias}.{_quote_ident(col)}"
            if agg == "pct_non_null":
                if col == "*":
                    meta["errors"].append(f"unsupported_metric:{m}")
                    return None, meta
                expr = (
                    "(\n"
                    f"    SUM(CASE WHEN {col_sql} IS NOT NULL THEN 1 ELSE 0 END) * 100.0\n"
                    "    / NULLIF(COUNT(*), 0)\n"
                    "  )"
                )
            else:
                expr = _agg_expr(agg, col_sql, distinct=distinct)
            if not expr:
                meta["errors"].append(f"unsupported_metric:{m}")
                return None, meta
            select_items.append(f"{expr} AS {_quote_ident(alias)}")

        # Promote aggregate-threshold predicates to HAVING.
        # Example: AVG(amount) > 50000 should not remain as row-level WHERE amount > 50000.
        where_drop: Set[str] = set()
        for m in metrics:
            for f in filters_by_table.get(subject, []):
                having_pred = _render_metric_having_sql(f, m, base_alias)
                if not having_pred:
                    continue
                having_items.append(having_pred)
                row_pred = _render_filter_sql(f, base_alias)
                if row_pred:
                    where_drop.add(row_pred)
        if where_drop:
            where_items = [w for w in where_items if w not in where_drop]
        if having_items:
            having_items = list(dict.fromkeys(having_items))
        query_from = f"FROM {_quote_ident(subject)} AS {base_alias}"

    else:
        # Subject-grain mode + child pre-aggregated subqueries.
        query_from = f"FROM {_quote_ident(subject)} AS {base_alias}"
        if not select_items:
            id_col = _id_like_column(subject, allowed)
            if id_col:
                select_items.append(f"{base_alias}.{_quote_ident(id_col)}")

        # Flat grouped-join mode for simple aggregate-threshold intents.
        # This keeps SQL shape closer to canonical LEFT JOIN + GROUP BY + HAVING.
        child_metric_tables = sorted(
            {
                str(m.get("table") or "").strip().lower()
                for m in metrics
                if str(m.get("table") or "").strip() and str(m.get("table") or "").strip().lower() != subject
            }
        )
        non_subject_filter_tables = {
            str(t or "").strip().lower()
            for t in filters_by_table.keys()
            if str(t or "").strip().lower() and str(t or "").strip().lower() != subject
        }
        if (
            len(metrics) == 1
            and len(child_metric_tables) == 1
            and not existence
            and not ordering
            and not isinstance(limit, int)
            and non_subject_filter_tables.issubset(set(child_metric_tables))
        ):
            m0 = metrics[0]
            mt = child_metric_tables[0]
            if str(m0.get("table") or "").strip().lower() == mt:
                metric_alias = "m0"
                join_cond = _join_cond_between(mt, metric_alias, subject, base_alias, fk_map, allowed)
                if join_cond:
                    on_filters: List[str] = []
                    local_having: List[str] = []
                    for f in filters_by_table.get(mt, []):
                        hp = _render_metric_having_sql(f, m0, metric_alias)
                        if hp:
                            local_having.append(hp)
                            continue
                        rp = _render_filter_sql(f, metric_alias)
                        if rp:
                            on_filters.append(rp)

                    local_having = list(dict.fromkeys(local_having))
                    on_filters = list(dict.fromkeys(on_filters))

                    if local_having:
                        agg = str(m0.get("agg") or "").lower()
                        col = str(m0.get("column") or "*").strip()
                        distinct = bool(m0.get("distinct"))
                        alias = str(m0.get("alias") or "metric_value").strip()
                        col_sql = "*" if col == "*" else f"{metric_alias}.{_quote_ident(col)}"
                        metric_expr = _agg_expr(agg, col_sql, distinct=distinct)
                        if metric_expr:
                            if on_filters:
                                join_cond = join_cond + " AND " + " AND ".join(on_filters)
                            join_items.append(
                                f"LEFT JOIN {_quote_ident(mt)} AS {metric_alias}\n"
                                f"  ON {join_cond}"
                            )
                            select_items.append(f"{metric_expr} AS {_quote_ident(alias)}")
                            group_by_items = list(group_items) or [f"{base_alias}.{_quote_ident(subject_id_col)}"]

                            sql = (
                                "SELECT\n"
                                "  " + ",\n  ".join(select_items) + "\n"
                                f"{query_from}\n"
                                + "\n".join(join_items)
                                + "\n"
                            )
                            if where_items:
                                sql += "WHERE " + " AND ".join(where_items) + "\n"
                            sql += "GROUP BY " + ", ".join(group_by_items) + "\n"
                            sql += "HAVING " + " AND ".join(local_having) + "\n"

                            sql = _format_with_sqlglot(sql.strip())
                            if not sql.endswith(";"):
                                sql += ";"

                            ok, val_errs = _validate_generated_sql(sql, intent_ir)
                            if ok:
                                meta["join_plan"] = {
                                    "subject": subject,
                                    "subject_id": subject_id_col,
                                    "metric_tables": [mt],
                                    "existence_tables": [],
                                    "single_table_agg_mode": False,
                                    "flat_metric_having_mode": True,
                                    "include_zero": include_zero,
                                    "limit": limit,
                                }
                                return sql, meta
                            meta["warnings"].append(
                                f"flat_metric_having_mode_rejected:{'|'.join(val_errs)}"
                            )

        # Flat inner-join mode for simple "has at least one child row" + aggregate-on-same-child intents.
        # Example:
        # - show customers who have placed orders, and count their orders
        # - show products that have been ordered, and sum ordered quantity
        #
        # For this narrow family, the aggregate over the child table already enforces existence,
        # so a single INNER JOIN + GROUP BY is cleaner than separate existence/metric subqueries.
        if (
            len(metrics) == 1
            and len(child_metric_tables) == 1
            and len(existence) == 1
            and not include_zero
            and not ordering
            and not isinstance(limit, int)
            and non_subject_filter_tables.issubset(set(child_metric_tables))
        ):
            m0 = metrics[0]
            ex0 = existence[0]
            mt = child_metric_tables[0]
            ex_table = str(ex0.get("table") or "").strip().lower()
            ex_mode = str(ex0.get("mode") or "").strip().lower()
            if (
                ex_mode == "at_least_one"
                and ex_table == mt
                and str(m0.get("table") or "").strip().lower() == mt
            ):
                metric_alias = "m0"
                join_cond = _join_cond_between(mt, metric_alias, subject, base_alias, fk_map, allowed)
                if join_cond:
                    on_filters: List[str] = []
                    local_having: List[str] = []
                    for f in filters_by_table.get(mt, []):
                        hp = _render_metric_having_sql(f, m0, metric_alias)
                        if hp:
                            local_having.append(hp)
                            continue
                        rp = _render_filter_sql(f, metric_alias)
                        if rp:
                            on_filters.append(rp)

                    local_having = list(dict.fromkeys(local_having))
                    on_filters = list(dict.fromkeys(on_filters))

                    agg = str(m0.get("agg") or "").lower()
                    col = str(m0.get("column") or "*").strip()
                    distinct = bool(m0.get("distinct"))
                    alias = str(m0.get("alias") or "metric_value").strip()
                    col_sql = "*" if col == "*" else f"{metric_alias}.{_quote_ident(col)}"
                    metric_expr = _agg_expr(agg, col_sql, distinct=distinct)
                    if metric_expr:
                        trial_join_cond = join_cond
                        if on_filters:
                            trial_join_cond = trial_join_cond + " AND " + " AND ".join(on_filters)
                        trial_select_items = list(select_items) + [
                            f"{metric_expr} AS {_quote_ident(alias)}"
                        ]
                        trial_join_items = [
                            f"JOIN {_quote_ident(mt)} AS {metric_alias}\n"
                            f"  ON {trial_join_cond}"
                        ]
                        group_by_items = list(group_items) or [f"{base_alias}.{_quote_ident(subject_id_col)}"]

                        sql = (
                            "SELECT\n"
                            "  " + ",\n  ".join(trial_select_items) + "\n"
                            f"{query_from}\n"
                            + "\n".join(trial_join_items)
                            + "\n"
                        )
                        if where_items:
                            sql += "WHERE " + " AND ".join(where_items) + "\n"
                        sql += "GROUP BY " + ", ".join(group_by_items) + "\n"
                        if local_having:
                            sql += "HAVING " + " AND ".join(local_having) + "\n"

                        sql = _format_with_sqlglot(sql.strip())
                        if not sql.endswith(";"):
                            sql += ";"

                        ok, val_errs = _validate_generated_sql(sql, intent_ir)
                        if ok:
                            meta["join_plan"] = {
                                "subject": subject,
                                "subject_id": subject_id_col,
                                "metric_tables": [mt],
                                "existence_tables": [mt],
                                "single_table_agg_mode": False,
                                "flat_existence_metric_join_mode": True,
                                "include_zero": include_zero,
                                "limit": limit,
                            }
                            return sql, meta
                        meta["warnings"].append(
                            f"flat_existence_metric_join_mode_rejected:{'|'.join(val_errs)}"
                        )

        # Existence constraints first.
        ex_index = 1
        for ex in existence:
            t = str(ex.get("table") or "").strip().lower()
            mode = str(ex.get("mode") or "").strip().lower()
            if not t or t == subject:
                continue
            ex_sql, ex_meta = _build_path_subquery(
                source_table=t,
                subject_table=subject,
                subject_id_col=subject_id_col,
                fk_map=fk_map,
                adj=adj,
                allowed=allowed,
                filters=[],
                metric=None,
            )
            if not ex_sql:
                meta["errors"].append(ex_meta.get("error", f"existence_path_failed:{t}"))
                return None, meta
            ex_alias = f"ex_{ex_index}"
            ex_index += 1
            if mode == "at_least_one":
                join_items.append(
                    f"JOIN (\n{ex_sql}\n) AS {ex_alias}\n"
                    f"  ON {ex_alias}.__subject_key = {base_alias}.{_quote_ident(subject_id_col)}"
                )
            elif mode == "none":
                join_items.append(
                    f"LEFT JOIN (\n{ex_sql}\n) AS {ex_alias}\n"
                    f"  ON {ex_alias}.__subject_key = {base_alias}.{_quote_ident(subject_id_col)}"
                )
                where_items.append(f"{ex_alias}.__subject_key IS NULL")
            else:
                meta["errors"].append(f"unsupported_existence_mode:{mode}")
                return None, meta

        # Metric subqueries.
        metric_index = 1
        for m in metrics:
            t = str(m.get("table") or "").strip().lower()
            if not t:
                meta["errors"].append("metric_missing_table")
                return None, meta

            if t == subject:
                # fallback: aggregate subject by subject key to keep outer shape stable
                m_copy = dict(m)
                m_copy.setdefault("alias", f"{m_copy.get('agg')}_value")
                m_sql, m_meta = _build_path_subquery(
                    source_table=subject,
                    subject_table=subject,
                    subject_id_col=subject_id_col,
                    fk_map=fk_map,
                    adj={subject: {subject}},
                    allowed=allowed,
                    filters=filters_by_table.get(subject, []),
                    metric=m_copy,
                )
            else:
                m_sql, m_meta = _build_path_subquery(
                    source_table=t,
                    subject_table=subject,
                    subject_id_col=subject_id_col,
                    fk_map=fk_map,
                    adj=adj,
                    allowed=allowed,
                    filters=filters_by_table.get(t, []),
                    metric=m,
                )
            if not m_sql:
                meta["errors"].append(m_meta.get("error", f"metric_path_failed:{t}"))
                return None, meta

            m_alias = f"m_{metric_index}"
            metric_index += 1
            has_metric_having = bool((m_meta or {}).get("has_metric_having", False))
            # Aggregate-threshold metric should filter subject rows, so always prefer INNER JOIN.
            join_type = "JOIN" if has_metric_having else "LEFT JOIN"
            join_items.append(
                f"{join_type} (\n{m_sql}\n) AS {m_alias}\n"
                f"  ON {m_alias}.__subject_key = {base_alias}.{_quote_ident(subject_id_col)}"
            )

            alias = str(m.get("alias") or "metric_value").strip()
            agg = str(m.get("agg") or "").lower()
            if include_zero and agg in {"count", "sum"} and not has_metric_having:
                select_items.append(f"COALESCE({m_alias}.{_quote_ident(alias)}, 0) AS {_quote_ident(alias)}")
            else:
                select_items.append(f"{m_alias}.{_quote_ident(alias)} AS {_quote_ident(alias)}")

        # Non-subject filters (table-level existence predicates).
        for t, t_filters in filters_by_table.items():
            if t == subject:
                continue
            if any(str(m.get("table") or "").lower() == t for m in metrics):
                # already applied in metric subquery
                continue
            if any(str(e.get("table") or "").lower() == t for e in existence):
                # already applied by explicit existence clause
                continue
            fx_sql, fx_meta = _build_path_subquery(
                source_table=t,
                subject_table=subject,
                subject_id_col=subject_id_col,
                fk_map=fk_map,
                adj=adj,
                allowed=allowed,
                filters=t_filters,
                metric=None,
            )
            if not fx_sql:
                meta["errors"].append(fx_meta.get("error", f"filter_path_failed:{t}"))
                return None, meta
            where_items.append(
                "EXISTS (\n"
                f"  SELECT 1 FROM (\n{fx_sql}\n  ) AS fx\n"
                f"  WHERE fx.__subject_key = {base_alias}.{_quote_ident(subject_id_col)}\n"
                ")"
            )

    # ORDER BY.
    order_items: List[str] = []
    for o in ordering:
        kind = str(o.get("kind") or "").strip().lower()
        by = str(o.get("by") or "").strip()
        direction = str(o.get("direction") or "ASC").strip().upper()
        direction = "DESC" if direction == "DESC" else "ASC"

        if kind == "alias" and by:
            order_items.append(f"{_quote_ident(by)} {direction}")
        elif kind == "column" and by:
            table = str(o.get("table") or subject).strip().lower()
            alias = base_alias if table == subject else base_alias
            order_items.append(f"{alias}.{_quote_ident(by)} {direction}")
        elif kind == "expression" and by.upper() == "MONTH" and month_expr_for_order:
            order_items.append(f"{month_expr_for_order} {direction}")

    # Default ordering for top/bottom semantics without explicit order mapping.
    if not order_items and metrics and isinstance(limit, int):
        m0_alias = str(metrics[0].get("alias") or "").strip()
        if m0_alias:
            order_items.append(f"{_quote_ident(m0_alias)} DESC")

    if not select_items:
        meta["errors"].append("empty_select_projection")
        return None, meta

    select_keyword = "SELECT DISTINCT" if select_distinct and not metrics else "SELECT"
    sql = (
        f"{select_keyword}\n"
        "  " + ",\n  ".join(select_items) + "\n"
        f"{query_from}\n"
    )
    if join_items:
        sql += "\n".join(join_items) + "\n"
    if where_items:
        sql += "WHERE " + " AND ".join(where_items) + "\n"
    if metrics and group_items and single_table_agg_mode:
        sql += "GROUP BY " + ", ".join(group_items) + "\n"
        if having_items:
            sql += "HAVING " + " AND ".join(having_items) + "\n"
    if order_items:
        sql += "ORDER BY " + ", ".join(order_items) + "\n"
    if isinstance(limit, int) and limit > 0:
        sql += f"LIMIT {limit}\n"

    sql = _format_with_sqlglot(sql.strip())
    if not sql.endswith(";"):
        sql += ";"

    ok, val_errs = _validate_generated_sql(sql, intent_ir)
    if not ok:
        meta["errors"].extend(val_errs)
        return None, meta

    meta["join_plan"] = {
        "subject": subject,
        "subject_id": subject_id_col,
        "metric_tables": [str(m.get("table") or "").lower() for m in metrics],
        "existence_tables": [str(e.get("table") or "").lower() for e in existence],
        "single_table_agg_mode": single_table_agg_mode,
        "include_zero": include_zero,
        "select_distinct": select_distinct and not metrics,
        "limit": limit,
    }
    return sql, meta


def safety_normalize_sql(sql: str) -> Tuple[str, List[str]]:
    warnings: List[str] = []
    s = (sql or "").strip()
    if not s:
        return s, ["empty_sql"]

    s, wrapper_warnings = _strip_sql_wrappers(s)
    warnings.extend(wrapper_warnings)

    s, statement_warnings = _extract_first_sql_statement(s)
    warnings.extend(statement_warnings)

    if _sql_has_banned_pattern(s):
        warnings.append("banned_pattern_detected")

    # collapse spaces, keep line breaks mostly stable
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    s = _format_with_sqlglot(s)
    if not s.endswith(";"):
        s += ";"
    return s, warnings
