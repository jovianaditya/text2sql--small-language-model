"""
Dynamic Schema Analyzer - Extracts schema metadata from JSONL for agnostic Text-to-SQL.

This module parses the JSONL metadata file to build a dynamic schema configuration,
eliminating hardcoded table names, synonyms, and patterns.

Key Features:
- Auto-detect table types (transaction, dimension, bridge) from column patterns
- Extract synonyms from metadata
- Build FK relationship graph
- Infer table-specific rules dynamically
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict


# ======================================================================
# ==================== SCHEMA ANALYZER =================================
# ======================================================================

class SchemaAnalyzer:
    """
    Analyzes JSONL schema metadata to build dynamic configuration.

    Replaces hardcoded table names, synonyms, and patterns with schema-driven inference.
    """

    def __init__(self, jsonl_path: str):
        """
        Initialize with path to JSONL metadata file.

        Args:
            jsonl_path: Path to dense JSONL file (e.g., classicmodels_dense.jsonl)
        """
        self.jsonl_path = jsonl_path
        self._tables: Set[str] = set()
        self._table_columns: Dict[str, List[str]] = defaultdict(list)
        self._table_synonyms: Dict[str, List[str]] = defaultdict(list)
        self._column_synonyms: Dict[str, List[str]] = defaultdict(list)
        self._table_types: Dict[str, str] = {}  # table -> type
        self._fk_relationships: Dict[str, Dict[str, str]] = defaultdict(dict)  # {table: {fk_col: target_table}}
        self._column_roles: Dict[str, Dict[str, str]] = defaultdict(dict)  # {table: {col: role}}
        self._column_datatypes: Dict[str, Dict[str, str]] = defaultdict(dict)  # {table: {col: datatype}}
        # Store original case for column names (lowercase -> original)
        self._column_original_case: Dict[str, str] = {}

        self._parse_jsonl()

    def _parse_jsonl(self):
        """Parse JSONL file and extract all metadata."""
        with open(self.jsonl_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    self._process_entry(data)
                except json.JSONDecodeError:
                    continue

        # Post-processing: infer table types
        self._infer_table_types()

    def _process_entry(self, data: Dict[str, Any]):
        """Process a single JSONL entry."""
        metadata = data.get("metadata", {})
        table = metadata.get("table", "")
        column = metadata.get("column", "")

        if not table or not column:
            return

        self._tables.add(table.lower())
        self._table_columns[table.lower()].append(column.lower())

        # Store original case for column name
        col_key = f"{table.lower()}.{column.lower()}"
        self._column_original_case[col_key] = column

        # Extract synonyms
        text = data.get("text", "")
        synonyms_match = re.search(r'Synonyms:\s*(.+?)\.', text)
        if synonyms_match:
            synonyms_str = synonyms_match.group(1).strip()
            synonyms = [s.strip().lower() for s in synonyms_str.split(",")]
            self._column_synonyms[f"{table.lower()}.{column.lower()}"].extend(synonyms)

        # Extract FK info
        description = metadata.get("description", "")
        if "Foreign Key" in description:
            # Extract target table from "FK to {table}.{column}"
            fk_match = re.search(r'FK to\s+(\w+)\.(\w+)', description, re.IGNORECASE)
            if fk_match:
                target_table = fk_match.group(1).lower()
                self._fk_relationships[table.lower()][column.lower()] = target_table

        # Extract column role
        role = self._next_field_after(text, "Role:")
        if role:
            self._column_roles[table.lower()][column.lower()] = role.strip().lower()

        # Extract datatype
        datatype = metadata.get("datatype", "")
        self._column_datatypes[table.lower()][column.lower()] = datatype

    def _next_field_after(self, text: str, field: str) -> Optional[str]:
        """Extract the value of a field after a given marker."""
        pattern = rf'{re.escape(field)}\s+([^.]+?)(?:\.|$)'
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        return None

    def _infer_table_types(self):
        """
        Infer table types from column patterns and metadata.

        Table Types:
        - transaction: Tables with amount/quantity columns (payments, orderdetails)
        - dimension: Master tables (customers, products, employees)
        - bridge: Junction tables with composite FKs (orderdetails)
        - reference: Lookup tables (productlines, offices)
        """
        for table in self._tables:
            columns = self._table_columns[table]

            # Check for transaction indicators
            has_amount = any('amount' in col for col in columns)
            has_quantity = any('quantity' in col for col in columns)
            has_price = any('price' in col for col in columns)

            # Check for bridge table (multiple FKs)
            fk_count = len(self._fk_relationships.get(table, {}))

            # Check for dimension indicators
            has_name = any('name' in col for col in columns)
            is_master = table.endswith('s') and fk_count == 0 and has_name

            if has_amount or has_quantity:
                self._table_types[table] = "transaction"
            elif fk_count >= 2:
                self._table_types[table] = "bridge"
            elif is_master:
                self._table_types[table] = "dimension"
            else:
                self._table_types[table] = "reference"

    # ====================================================================
    # ==================== PUBLIC API =====================================
    # ====================================================================

    def get_tables(self) -> Set[str]:
        """Return all table names (lowercase)."""
        return self._tables

    def get_table_columns(self, table: str) -> List[str]:
        """Return columns for a table (lowercase)."""
        return self._table_columns.get(table.lower(), [])

    def get_table_type(self, table: str) -> str:
        """
        Return inferred table type.

        Types: transaction, dimension, bridge, reference
        """
        return self._table_types.get(table.lower(), "unknown")

    def get_transaction_tables(self) -> Set[str]:
        """Return all transaction tables (have amount/quantity)."""
        return {t for t, typ in self._table_types.items() if typ == "transaction"}

    def get_dimension_tables(self) -> Set[str]:
        """Return all dimension/master tables."""
        return {t for t, typ in self._table_types.items() if typ == "dimension"}

    def get_bridge_tables(self) -> Set[str]:
        """Return all bridge/junction tables."""
        return {t for t, typ in self._table_types.items() if typ == "bridge"}

    def get_fk_relationships(self) -> Dict[str, Dict[str, str]]:
        """Return FK relationships: {table: {fk_col: target_table}}."""
        return self._fk_relationships

    def get_table_synonyms(self, table: str) -> List[str]:
        """
        Get all synonyms for a table (from column synonyms).

        Aggregates synonyms from all columns in the table.
        """
        table = table.lower()
        synonyms = set()

        for col in self._table_columns.get(table, []):
            key = f"{table}.{col}"
            if key in self._column_synonyms:
                synonyms.update(self._column_synonyms[key])

        return list(synonyms)

    def get_column_synonyms(self, table: str, column: str) -> List[str]:
        """
        Get synonyms for a specific column from metadata.

        Returns synonyms from JSONL metadata for the given table.column.
        If no synonyms found, returns empty list (safe fallback).

        Args:
            table: Table name
            column: Column name

        Returns:
            List of synonyms for the column
        """
        key = f"{table.lower()}.{column.lower()}"
        return self._column_synonyms.get(key, [])

    def get_anchor_columns(self, table: str) -> List[str]:
        """
        Get key columns for scoring boost (Role: Metric).

        These columns are used for boosting table relevance in retrieval.
        """
        table = table.lower()
        anchors = []

        for col, role in self._column_roles.get(table, {}).items():
            if role == "metric":
                anchors.append(col)

        return anchors

    def is_transaction_table(self, table: str) -> bool:
        """Check if table is a transaction table."""
        return self.get_table_type(table) == "transaction"

    def get_primary_key_column(self, table: str) -> Optional[str]:
        """
        Get primary key column for a table.

        Finds column with description containing "primary key".
        """
        table = table.lower()
        for col in self._table_columns.get(table, []):
            col_synonyms = self._column_synonyms.get(f"{table}.{col}", [])
            if any('primary key' in s or 'unique' in s or 'pk' in s for s in col_synonyms):
                return col
        return None

    def get_singular_form(self, table: str) -> str:
        """
        Get singular form of table name.

        Uses simple rules: removes trailing 's', handles 'ies' -> 'y'.
        """
        table = table.lower()
        if table.endswith('ies'):
            return table[:-3] + 'y'
        elif table.endswith('s') and not table.endswith('ss'):
            return table[:-1]
        return table

    def find_table_by_synonym(self, keyword: str) -> Optional[str]:
        """
        Find table by searching through all synonyms.

        Args:
            keyword: Search keyword (lowercase)

        Returns:
            Table name if found, None otherwise
        """
        keyword = keyword.lower()

        for table in self._tables:
            # Check direct table name match
            if keyword == table or keyword == self.get_singular_form(table):
                return table

            # Check column synonyms
            for col in self._table_columns.get(table, []):
                col_synonyms = self._column_synonyms.get(f"{table}.{col}", [])
                if keyword in col_synonyms:
                    return table

        return None


# ======================================================================
# ==================== GLOBAL INSTANCE ==================================
# ======================================================================

_analyzer_instance: Optional[SchemaAnalyzer] = None


def get_schema_analyzer(jsonl_path: str = None) -> SchemaAnalyzer:
    """
    Get or create singleton SchemaAnalyzer instance.

    Priority:
    1. Explicit jsonl_path parameter
    2. *_dense.jsonl (rich metadata with synonyms/roles)
    3. Any .jsonl file (fallback)

    Args:
        jsonl_path: Path to JSONL file or directory. If None, uses default from config.

    Returns:
        SchemaAnalyzer instance
    """
    global _analyzer_instance

    if _analyzer_instance is None:
        if jsonl_path is None:
            from .config import JSON_INPUT
            jsonl_path = JSON_INPUT

        # Convert to Path object
        jsonl_path_obj = Path(jsonl_path)

        # Handle case where path is a directory (not a file)
        if jsonl_path_obj.is_dir():
            # It's a directory, search for jsonl files inside
            jsonl_dir = jsonl_path_obj
        else:
            # It's a file path (or wildcard pattern), get the parent directory
            jsonl_dir = jsonl_path_obj.parent

        # Priority 1: Look for *_dense.jsonl first (rich metadata with synonyms/roles)
        dense_files = list(jsonl_dir.glob("*_dense.jsonl"))

        if dense_files:
            # Use the dense JSONL file (prioritizes rich metadata)
            jsonl_path = str(dense_files[0])
        else:
            # Priority 2: Use any .jsonl file
            # Handle wildcard patterns or find any jsonl file
            if "*" in jsonl_path or "?" in jsonl_path:
                # It's a wildcard pattern, use it as-is
                jsonl_files = list(jsonl_dir.glob(Path(jsonl_path).name))
                if jsonl_files:
                    jsonl_path = str(jsonl_files[0])
            elif jsonl_path_obj.is_dir() or not jsonl_path_obj.exists():
                # Path is a directory or doesn't exist, find any jsonl file
                jsonl_files = list(jsonl_dir.glob("*.jsonl"))
                if jsonl_files:
                    jsonl_path = str(jsonl_files[0])

        # Validate the final path
        final_path = Path(jsonl_path)
        if not final_path.exists() or final_path.is_dir():
            raise ValueError(
                f"Cannot find valid JSONL file. Path: {jsonl_path}. "
                f"Directory: {jsonl_dir}. "
                f"Please ensure at least one .jsonl file exists."
            )

        _analyzer_instance = SchemaAnalyzer(str(jsonl_path))

    return _analyzer_instance


# ======================================================================
# ==================== CONVENIENCE FUNCTIONS ============================
# ======================================================================

def get_all_tables() -> Set[str]:
    """Get all table names from schema."""
    return get_schema_analyzer().get_tables()


def get_transaction_tables() -> Set[str]:
    """Get all transaction tables dynamically."""
    return get_schema_analyzer().get_transaction_tables()


def get_dimension_tables() -> Set[str]:
    """Get all dimension/master tables dynamically."""
    return get_schema_analyzer().get_dimension_tables()


def get_table_synonyms(table: str) -> List[str]:
    """Get synonyms for a table from metadata."""
    return get_schema_analyzer().get_table_synonyms(table)


def get_column_synonyms(table: str, column: str) -> List[str]:
    """Get synonyms for a specific column from metadata (schema-agnostic)."""
    return get_schema_analyzer().get_column_synonyms(table, column)


def get_anchor_columns(table: str) -> List[str]:
    """Get anchor/scoring columns for a table."""
    return get_schema_analyzer().get_anchor_columns(table)


def is_transaction_table(table: str) -> bool:
    """Check if table is a transaction table."""
    return get_schema_analyzer().is_transaction_table(table)


def get_fk_relationships() -> Dict[str, Dict[str, str]]:
    """Get all FK relationships from schema."""
    return get_schema_analyzer().get_fk_relationships()


def find_table_by_keyword(keyword: str) -> Optional[str]:
    """Find table by searching through metadata."""
    return get_schema_analyzer().find_table_by_synonym(keyword)


def get_table_type(table: str) -> str:
    """Get inferred table type."""
    return get_schema_analyzer().get_table_type(table)


def get_singular_form(table: str) -> str:
    """Get singular form of table name."""
    return get_schema_analyzer().get_singular_form(table)


# ======================================================================
# ==================== DYNAMIC COLUMN INFERENCE =========================
# ======================================================================

def _get_original_case(table_lower: str, col_lower: str) -> str:
    """Get original column name case from metadata, fallback to lowercase."""
    analyzer = get_schema_analyzer()
    col_key = f"{table_lower}.{col_lower}"
    return analyzer._column_original_case.get(col_key, col_lower)


def get_id_column(table: str) -> Optional[str]:
    """
    Get the primary key/ID column for a table from schema metadata.

    Finds column with:
    1. Synonyms containing 'id', 'code', or 'primary key'
    2. Column name patterns: *Number (customerNumber), *Code (productCode), *id
    3. Column name matching table name (e.g., productlines.productLine)

    Args:
        table: Table name

    Returns:
        Column name (original case) if found, None otherwise
    """
    analyzer = get_schema_analyzer()
    table_lower = table.lower()
    columns = analyzer._table_columns.get(table_lower, [])

    # 1. Check synonyms for 'id', 'code', or 'primary key'
    for col in columns:
        synonyms = analyzer._column_synonyms.get(f"{table_lower}.{col}", [])
        for syn in synonyms:
            syn_lower = syn.lower()
            if 'id' in syn_lower or 'code' in syn_lower or 'primary key' in syn_lower:
                return _get_original_case(table_lower, col)

    # 2. Pattern matching (order matters - matches classicmodels pattern)
    # Priority: *number > *code > *id
    for col in columns:
        if col.endswith("number") and col != "ordernumber":
            return _get_original_case(table_lower, col)

    for col in columns:
        if col.endswith("code") and col not in ("zipcode", "postcode"):
            return _get_original_case(table_lower, col)

    for col in columns:
        if col.endswith("id") and "employeenumber" not in col:
            return _get_original_case(table_lower, col)

    # 3. Fallback: column name matches table singular (e.g., productlines.productLine)
    singular = get_singular_form(table_lower)
    for col in columns:
        if col.lower() == singular or col.lower() == singular + "line":
            return _get_original_case(table_lower, col)

    return None


def get_name_column(table: str) -> Optional[str]:
    """
    Get the main name/display column for a table from schema metadata.

    Finds column with:
    1. Role = 'name' in metadata
    2. Pattern {table_singular}name (e.g., customerName, productName)
    3. Any column ending with 'name'

    Args:
        table: Table name

    Returns:
        Column name (original case) if found, None otherwise
    """
    analyzer = get_schema_analyzer()
    table_lower = table.lower()
    columns = analyzer._table_columns.get(table_lower, [])

    # 1. Check role metadata
    for col in columns:
        role = analyzer._column_roles.get(table_lower, {}).get(col, "")
        if role == "name":
            return _get_original_case(table_lower, col)

    # 2. Pattern matching for {singular}name
    singular = get_singular_form(table_lower)
    # Check for column ending with "name" (case-insensitive)
    for col in columns:
        if col.lower().endswith("name"):
            return _get_original_case(table_lower, col)

    # 3. Check synonyms for "name"
    for col in columns:
        synonyms = analyzer._column_synonyms.get(f"{table_lower}.{col}", [])
        if "name" in synonyms:
            return _get_original_case(table_lower, col)

    return None


def get_all_mentioned_tables(question: str) -> Set[str]:
    """
    Get all tables mentioned in the question using schema metadata.

    Uses table name, singular form, and synonyms from JSONL.

    Args:
        question: User question text

    Returns:
        Set of table names that are mentioned
    """
    if not question:
        return set()

    analyzer = get_schema_analyzer()
    q = question.lower()

    mentioned = set()
    for table in analyzer._tables:
        # Direct match
        if table in q:
            mentioned.add(table)
            continue

        # Singular form
        singular = get_singular_form(table)
        if singular and singular in q:
            mentioned.add(table)
            continue

        # Synonyms from metadata
        synonyms = analyzer.get_table_synonyms(table)
        for syn in synonyms:
            if syn in q:
                mentioned.add(table)
                break

    return mentioned


def is_table_mentioned(question: str, table: str) -> bool:
    """
    Check if a table is mentioned in the question using schema metadata.

    Args:
        question: User question text
        table: Table name to check

    Returns:
        True if table is mentioned, False otherwise
    """
    return table.lower() in get_all_mentioned_tables(question)


# Export convenience
__all__ = [
    'SchemaAnalyzer',
    'get_schema_analyzer',
    'get_all_tables',
    'get_transaction_tables',
    'get_dimension_tables',
    'get_table_synonyms',
    'get_anchor_columns',
    'is_transaction_table',
    'get_fk_relationships',
    'find_table_by_keyword',
    'get_table_type',
    'get_singular_form',
    # New functions for schema-agnostic column inference
    'get_id_column',
    'get_name_column',
    'get_all_mentioned_tables',
    'is_table_mentioned',
]
