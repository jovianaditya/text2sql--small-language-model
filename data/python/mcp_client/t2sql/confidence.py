# ======================================================================
# =========================== confidence.py ============================
# ======================================================================
# Purpose:
# - Schema-agnostic confidence scoring for generated SQL
# - Multi-signal uncertainty quantification
# - Dynamic validation using schema metadata (no hardcoded table names)

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple

from .schema_analyzer import is_numeric_column, is_date_column, get_all_tables
from .helpers_main import validate_identifiers
from .config import SQL_PARSE_DIALECT

# ======================================================================
# ======================= SIGNAL WEIGHTS (TUNABLE) =====================
# ======================================================================
# Weights for combining individual confidence signals into overall score.
# All weights are normalized to sum to 1.0 at runtime.

SIGNAL_WEIGHTS = {
    "syntax_valid": 0.25,         # SQL parses without error
    "identifiers_valid": 0.30,    # All tables/columns exist in schema
    "entity_coverage": 0.15,      # Question entities appear in SQL
    "join_path_valid": 0.15,      # FK relationships are respected
    "structure_plausible": 0.15,  # Query structure + basic semantic matches question intent
}


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Normalize weights to sum to 1.0."""
    total = sum(weights.values())
    if total <= 0:
        return {k: 1.0 / len(weights) for k in weights}
    return {k: v / total for k, v in weights.items()}


# ======================================================================
# ======================= SIGNAL EXTRACTORS ============================
# ======================================================================

def _check_syntax_valid(sql: str) -> Tuple[float, List[str]]:
    """
    Check if SQL parses without syntax errors.
    
    Returns:
        (score, warnings) where score is 1.0 if valid, 0.0 if invalid
    """
    if not sql or not sql.strip():
        return 0.0, ["Empty SQL"]
    
    try:
        import sqlglot
        parsed = sqlglot.parse_one(sql.strip(), read=SQL_PARSE_DIALECT)
        if parsed is None:
            return 0.5, ["SQL parsed but returned None"]
        return 1.0, []
    except Exception as e:
        return 0.0, [f"Syntax error: {str(e)[:100]}"]


def _extract_sql_tables(sql: str) -> Set[str]:
    """Extract table names from FROM/JOIN clauses (schema-agnostic)."""
    tables: Set[str] = set()
    if not sql:
        return tables
    
    # Pattern: FROM <table> or JOIN <table> (with optional alias)
    pattern = r'\b(?:FROM|JOIN)\s+(`?[A-Za-z_][A-Za-z0-9_]*`?)(?:\s+(?:AS\s+)?[A-Za-z_][A-Za-z0-9_]*)?'
    for m in re.finditer(pattern, sql, flags=re.I):
        tbl = m.group(1).strip('`').lower()
        tables.add(tbl)
    
    return tables


def _extract_sql_columns(sql: str) -> Set[str]:
    """Extract column identifiers from SQL (schema-agnostic)."""
    columns: Set[str] = set()
    if not sql:
        return columns
    
    # Remove string literals to avoid false matches
    clean = re.sub(r"'(?:''|[^'])*'", "''", sql)
    clean = re.sub(r'"(?:""|[^"])*"', '""', clean)
    
    # Pattern: table.column or just column (qualified references)
    for m in re.finditer(r'\b[A-Za-z_][A-Za-z0-9_]*\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)\b', clean, flags=re.I):
        columns.add(m.group(1).lower())
    
    return columns


def _check_identifiers_valid(sql: str, ctx: str) -> Tuple[float, List[str]]:
    """
    Check if all identifiers in SQL exist in schema context.
    Uses ALLOWED IDENTIFIERS block from context (schema-agnostic).
    
    Returns:
        (score, warnings) where score is ratio of valid identifiers
    """
    warnings: List[str] = []

    if not sql or not ctx:
        return 0.5, ["Missing SQL or context"]

    # Reuse the same identifier validator used by the postprocess/gate flow
    # so confidence scoring remains aligned with acceptance decisions.
    errors = list(validate_identifiers(sql, ctx) or [])
    if not errors:
        return 1.0, []

    sql_tables = _extract_sql_tables(sql)
    sql_columns = _extract_sql_columns(sql)
    total_ids = len(sql_tables) + len(sql_columns)
    if total_ids <= 0:
        total_ids = max(1, len(errors))

    hard_errors = [
        e for e in errors
        if "not in ALLOWED" in str(e)
        or "invalid column" in str(e).lower()
        or "invalid columns" in str(e).lower()
    ]

    weighted_invalid = float(len(hard_errors)) + 0.5 * float(max(0, len(errors) - len(hard_errors)))
    score = max(0.0, 1.0 - (weighted_invalid / float(total_ids)))
    warnings.extend(f"Identifier validation: {str(e)}" for e in errors[:8])
    if len(errors) > 8:
        warnings.append(f"Identifier validation: ... and {len(errors) - 8} more")
    return score, warnings


def _extract_question_entities(question: str, ctx: str) -> Set[str]:
    """
    Extract entity mentions from question using schema metadata.
    Dynamically uses table names from ALLOWED IDENTIFIERS (schema-agnostic).
    """
    entities: Set[str] = set()
    q = (question or "").lower()
    
    if not q:
        return entities
    
    # Parse table names from context
    allowed_tables: Set[str] = set()
    m = re.search(r"### ALLOWED IDENTIFIERS(.*?)(?:\n### |\Z)", ctx or "", flags=re.S)
    if m:
        block = m.group(1)
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("- "):
                parts = line[2:].split(":", 1)
                if parts:
                    table = parts[0].strip().strip("`").lower()
                    if table:
                        allowed_tables.add(table)
    
    # Build entity variants dynamically
    for table in allowed_tables:
        # Check table name and singular form
        singular = table
        if table.endswith("ies") and len(table) > 3:
            singular = table[:-3] + "y"
        elif table.endswith("s") and not table.endswith("ss") and len(table) > 1:
            singular = table[:-1]
        
        variants = {table, singular}
        if "_" in table:
            variants.add(table.replace("_", " "))
        if "_" in singular:
            variants.add(singular.replace("_", " "))
        
        # Check if any variant appears in question
        for v in variants:
            if v and re.search(rf"\b{re.escape(v)}\b", q):
                entities.add(table)
                break
    
    return entities


def _check_entity_coverage(sql: str, question: str, ctx: str) -> Tuple[float, List[str]]:
    """
    Check if entities mentioned in question appear in SQL.
    
    Returns:
        (score, warnings) where score is coverage ratio
    """
    warnings: List[str] = []
    
    question_entities = _extract_question_entities(question, ctx)
    sql_tables = _extract_sql_tables(sql)
    
    if not question_entities:
        return 1.0, []  # No entities to check
    
    # Check coverage
    covered = question_entities.intersection(sql_tables)
    missing = question_entities - sql_tables
    
    if missing:
        warnings.append(f"Question mentions entities not in SQL: {', '.join(missing)}")
    
    score = len(covered) / len(question_entities) if question_entities else 1.0
    return score, warnings


def _check_join_path_valid(sql: str, ctx: str) -> Tuple[float, List[str]]:
    """
    Check if JOIN conditions use valid FK relationships from schema.
    Extracts FK info dynamically from context or schema analyzer.
    
    Returns:
        (score, warnings) where score is 1.0 if valid, lower if issues found
    """
    warnings: List[str] = []
    
    if not sql or "JOIN" not in sql.upper():
        return 1.0, []  # No joins to validate
    
    # Extract tables from SQL
    sql_tables = _extract_sql_tables(sql)
    if len(sql_tables) < 2:
        return 1.0, []  # Single table, no join validation needed
    
    # Try to get FK relationships from schema analyzer
    try:
        from .schema_analyzer import get_fk_relationships
        fk_rels = get_fk_relationships()
    except Exception:
        fk_rels = {}
    
    if not fk_rels:
        # No FK metadata available, assume valid but warn
        return 0.8, ["FK relationships not available for validation"]
    
    # Build adjacency graph from FK relationships
    adjacent: Set[Tuple[str, str]] = set()
    for src_table, fk_map in fk_rels.items():
        for fk_col, target_table in fk_map.items():
            adjacent.add((src_table.lower(), target_table.lower()))
            adjacent.add((target_table.lower(), src_table.lower()))
    
    # Also add implicit relationships (shared ID columns)
    # This is schema-agnostic - looks for columns ending in 'id', 'number', 'code'
    try:
        from .schema_analyzer import get_schema_analyzer
        analyzer = get_schema_analyzer()
        tables = list(sql_tables)
        for i, t1 in enumerate(tables):
            cols1 = set(analyzer._table_columns.get(t1, []))
            id_cols1 = {c for c in cols1 if c.endswith(('id', 'number', 'code'))}
            for t2 in tables[i+1:]:
                cols2 = set(analyzer._table_columns.get(t2, []))
                id_cols2 = {c for c in cols2 if c.endswith(('id', 'number', 'code'))}
                if id_cols1.intersection(id_cols2):
                    adjacent.add((t1, t2))
                    adjacent.add((t2, t1))
    except Exception:
        pass
    
    # Check if all joined tables are connected
    tables_list = list(sql_tables)
    all_connected = True
    for i, t1 in enumerate(tables_list):
        for t2 in tables_list[i+1:]:
            if (t1, t2) not in adjacent and (t2, t1) not in adjacent:
                # Check if path exists via BFS
                path_exists = _bfs_path_exists(t1, t2, adjacent)
                if not path_exists:
                    warnings.append(f"No FK path between {t1} and {t2}")
                    all_connected = False
    
    return 1.0 if all_connected else 0.6, warnings


def _bfs_path_exists(start: str, end: str, adjacent: Set[Tuple[str, str]]) -> bool:
    """BFS to check if path exists between two tables."""
    if start == end:
        return True
    
    # Build adjacency list
    adj_list: Dict[str, Set[str]] = {}
    for a, b in adjacent:
        adj_list.setdefault(a, set()).add(b)
    
    visited: Set[str] = {start}
    queue: List[str] = [start]
    
    while queue:
        current = queue.pop(0)
        for neighbor in adj_list.get(current, set()):
            if neighbor == end:
                return True
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return False


def _validate_aggregate_columns(sql: str) -> List[str]:
    """
    Validate that columns used in SUM/AVG are numeric.
    
    Extracts column references from SUM(col), AVG(col) and checks
    if they are actually numeric columns using schema metadata.
    
    Returns:
        List of warning messages for non-numeric aggregate columns
    """
    warnings: List[str] = []
    s = sql.upper()
    
    # Pattern to extract SUM(table.col) or SUM(col) or SUM(alias.col)
    agg_pattern = re.compile(r'\b(SUM|AVG)\s*\(\s*([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)\s*\)', re.IGNORECASE)
    
    # Get all known tables
    all_tables = {t.lower() for t in get_all_tables()}
    
    for match in agg_pattern.finditer(sql):
        agg_func = match.group(1).upper()
        col_ref = match.group(2)
        
        # Parse table.column or just column
        if '.' in col_ref:
            table_or_alias, col = col_ref.split('.', 1)
            table_or_alias = table_or_alias.lower()
            col = col.lower()
        else:
            table_or_alias = None
            col = col_ref.lower()
        
        # Try to find the table
        table = None
        if table_or_alias and table_or_alias in all_tables:
            table = table_or_alias
        elif not table_or_alias:
            # Try to find the table from FROM clause
            from_match = re.search(r'\bFROM\s+([A-Za-z_][A-Za-z0-9_]*)', sql, re.IGNORECASE)
            if from_match:
                t = from_match.group(1).lower()
                if t in all_tables:
                    table = t
        
        # Validate if we found the table
        if table:
            if not is_numeric_column(table, col):
                warnings.append(f"{agg_func}({col}) may use non-numeric column")
    
    return warnings


def _validate_order_columns(sql: str) -> List[str]:
    """
    Validate that columns used in ORDER BY are orderable (numeric or date).
    
    Returns:
        List of warning messages for non-orderable ORDER BY columns
    """
    warnings: List[str] = []
    
    # Extract ORDER BY clause
    order_match = re.search(r'ORDER\s+BY\s+(.+?)(?:LIMIT|$)', sql, re.IGNORECASE | re.DOTALL)
    if not order_match:
        return warnings
    
    order_clause = order_match.group(1).strip()
    
    # Get all known tables
    all_tables = {t.lower() for t in get_all_tables()}
    
    # Extract column references (skip ASC, DESC, numbers, expressions)
    col_pattern = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)\b')
    
    for match in col_pattern.finditer(order_clause):
        col_ref = match.group(1)
        if col_ref.upper() in ('ASC', 'DESC', 'NULLS', 'FIRST', 'LAST'):
            continue
        
        # Parse table.column or just column
        if '.' in col_ref:
            table_or_alias, col = col_ref.split('.', 1)
            table_or_alias = table_or_alias.lower()
            col = col.lower()
        else:
            table_or_alias = None
            col = col_ref.lower()
        
        # Try to find the table
        table = None
        if table_or_alias and table_or_alias in all_tables:
            table = table_or_alias
        elif not table_or_alias:
            # Try to find the table from FROM clause
            from_match = re.search(r'\bFROM\s+([A-Za-z_][A-Za-z0-9_]*)', sql, re.IGNORECASE)
            if from_match:
                t = from_match.group(1).lower()
                if t in all_tables:
                    table = t
        
        # Validate if we found the table
        if table:
            if not is_numeric_column(table, col) and not is_date_column(table, col):
                # Only warn if column looks like it should be orderable
                name_like = col.endswith('name') or col.endswith('code')
                if not name_like:
                    # This column might not be ideal for ORDER BY, but not always a problem
                    pass  # Don't warn - could be intentional

    return warnings


def _check_structure_plausible(sql: str, question: str) -> Tuple[float, List[str]]:
    """
    Check if SQL structure matches question intent (schema-agnostic).
    Includes basic semantic contract checks (COUNT, SUM, AVG, ORDER BY, LIMIT, GROUP BY).
    
    Returns:
        (score, warnings)
    """
    warnings: List[str] = []
    q = (question or "").lower()
    s = (sql or "").upper()
    
    if not q or not s:
        return 0.5, ["Missing question or SQL"]
    
    checks_total = 0
    checks_passed = 0
    
    # --- Semantic contract checks (merged from _check_semantic_contract) ---
    
    # Check: COUNT when question asks for count/how many
    if re.search(r"\b(count|how\s+many|number\s+of|berapa)\b", q):
        checks_total += 1
        if "COUNT(" in s:
            checks_passed += 1
        else:
            warnings.append("Question asks for count but SQL lacks COUNT()")
    
    # Check: SUM when question asks for total
    if re.search(r"\b(total|sum|jumlah(?!\s*(?:dari)))\b", q):
        checks_total += 1
        if "SUM(" in s:
            checks_passed += 1
        else:
            warnings.append("Question asks for total but SQL lacks SUM()")
    
    # Check: AVG when question asks for average
    if re.search(r"\b(average|avg|rata-rata|mean)\b", q):
        checks_total += 1
        if "AVG(" in s:
            checks_passed += 1
        else:
            warnings.append("Question asks for average but SQL lacks AVG()")
    
    # Check: ORDER BY when question asks for ranking/sorting
    if re.search(r"\b(top|highest|lowest|best|worst|largest|smallest|order(?:ed)?\s+by|sort(?:ed)?\s+by)\b", q):
        checks_total += 1
        if "ORDER BY" in s:
            checks_passed += 1
        else:
            warnings.append("Question implies ordering but SQL lacks ORDER BY")
    
    # Check: LIMIT when question asks for top N
    if re.search(r"\b(top\s+\d+|first\s+\d+|limit\s+\d+)\b", q):
        checks_total += 1
        if "LIMIT" in s:
            checks_passed += 1
        else:
            warnings.append("Question asks for top N but SQL lacks LIMIT")
    
    # Check: GROUP BY when question implies grouping with aggregation
    has_agg = any(agg in s for agg in ["COUNT(", "SUM(", "AVG(", "MIN(", "MAX("])
    has_group_by = "GROUP BY" in s
    asks_for_group = re.search(r"\b(per|for\s+each|each|every|by\s+\w+)\b", q)
    if asks_for_group and re.search(r"\b(count|sum|avg|total|average)\b", q):
        checks_total += 1
        if has_group_by:
            checks_passed += 1
        else:
            warnings.append("Question implies grouping but SQL lacks GROUP BY")
    
    # --- Structure plausibility checks ---
    
    # Check: Aggregate with grouping intent
    if has_agg and asks_for_group and not has_group_by:
        checks_total += 1
        warnings.append("Aggregate with grouping intent but no GROUP BY")
    else:
        if has_agg and asks_for_group:
            checks_total += 1
            checks_passed += 1
    
    # Check: Subquery/CTE for complex patterns
    asks_for_nested = re.search(r"\b(second|third|nth|within|among|compared\s+to)\b", q)
    has_subquery = "(" in s and "SELECT" in s[s.find("("):] if "(" in s else False
    has_cte = "WITH" in s
    if asks_for_nested:
        checks_total += 1
        if has_subquery or has_cte:
            checks_passed += 1
        else:
            warnings.append("Complex pattern may require subquery")
    
    # Check: WHERE clause when question has filters
    filter_patterns = r"\b(where|with|having|that\s+have|who\s+have|which\s+have|greater\s+than|less\s+than|more\s+than|equal\s+to|in\s+\d{4}|before|after|between)\b"
    if re.search(filter_patterns, q):
        checks_total += 1
        if "WHERE" in s or "HAVING" in s:
            checks_passed += 1
        else:
            warnings.append("Question implies filtering but SQL lacks WHERE/HAVING")
    
    # --- Column type validation ---
    # Validate aggregate columns are numeric
    agg_warnings = _validate_aggregate_columns(sql)
    if agg_warnings:
        checks_total += 1
        warnings.extend(agg_warnings)
        # Partial credit if some aggregates are valid
        if len(agg_warnings) == 0:
            checks_passed += 1
    
    # Compute score as ratio of passed checks
    if checks_total == 0:
        return 1.0, []
    
    score = checks_passed / checks_total
    return score, warnings


# ======================================================================
# ======================= MAIN CONFIDENCE FUNCTION =====================
# ======================================================================

def compute_sql_confidence(
    sql: str,
    question: str,
    ctx: str,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Compute multi-signal confidence score for generated SQL.
    
    All signals are schema-agnostic - derived from:
    - SQL structure (syntax parsing)
    - ALLOWED IDENTIFIERS in context (dynamic schema)
    - Question wording patterns (intent detection)
    - FK relationships from schema analyzer (dynamic)
    
    Args:
        sql: Generated SQL query
        question: Original natural language question
        ctx: Context string with ALLOWED IDENTIFIERS
        weights: Optional custom weights for signals
    
    Returns:
        Dict with:
        - confidence: float (0.0 - 1.0) overall confidence score
        - signals: Dict[str, float] individual signal scores
        - warnings: List[str] issues found
        - details: Dict[str, Any] per-signal details
    """
    effective_weights = _normalize_weights(weights or SIGNAL_WEIGHTS)
    
    signals: Dict[str, float] = {}
    all_warnings: List[str] = []
    details: Dict[str, Any] = {}
    
    # Signal 1: Syntax validity
    score, warns = _check_syntax_valid(sql)
    signals["syntax_valid"] = score
    all_warnings.extend(warns)
    details["syntax_valid"] = {"score": score, "warnings": warns}
    
    # Signal 2: Identifier validity
    score, warns = _check_identifiers_valid(sql, ctx)
    signals["identifiers_valid"] = score
    all_warnings.extend(warns)
    details["identifiers_valid"] = {"score": score, "warnings": warns}
    
    # Signal 3: Entity coverage
    score, warns = _check_entity_coverage(sql, question, ctx)
    signals["entity_coverage"] = score
    all_warnings.extend(warns)
    details["entity_coverage"] = {"score": score, "warnings": warns}
    
    # Signal 4: Join path validity
    score, warns = _check_join_path_valid(sql, ctx)
    signals["join_path_valid"] = score
    all_warnings.extend(warns)
    details["join_path_valid"] = {"score": score, "warnings": warns}
    
    # Signal 5: Structure plausibility (includes semantic contract checks)
    score, warns = _check_structure_plausible(sql, question)
    signals["structure_plausible"] = score
    all_warnings.extend(warns)
    details["structure_plausible"] = {"score": score, "warnings": warns}
    
    # Compute weighted overall confidence
    overall = sum(signals[k] * effective_weights.get(k, 0) for k in signals)
    
    return {
        "confidence": round(overall, 4),
        "signals": signals,
        "warnings": all_warnings,
        "details": details,
    }


def format_confidence_report(result: Dict[str, Any], verbose: bool = False) -> str:
    """
    Format confidence result as human-readable string.
    
    Args:
        result: Output from compute_sql_confidence
        verbose: Include per-signal details
    
    Returns:
        Formatted string report
    """
    lines = [
        f"SQL Confidence: {result['confidence']:.1%}",
    ]
    
    if verbose:
        lines.append("Signal Scores:")
        for signal, score in result.get("signals", {}).items():
            lines.append(f"  - {signal}: {score:.1%}")
    
    warnings = result.get("warnings", [])
    if warnings:
        lines.append(f"Warnings ({len(warnings)}):")
        for w in warnings[:5]:  # Limit to first 5
            lines.append(f"  - {w}")
        if len(warnings) > 5:
            lines.append(f"  ... and {len(warnings) - 5} more")
    
    return "\n".join(lines)
