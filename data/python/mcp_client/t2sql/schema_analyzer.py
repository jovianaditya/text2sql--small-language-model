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
        jsonl_path: Path to JSONL file. If None, uses default from config.

    Returns:
        SchemaAnalyzer instance
    """
    global _analyzer_instance

    if _analyzer_instance is None:
        if jsonl_path is None:
            from .config import JSON_INPUT
            jsonl_path = JSON_INPUT

        # Priority: Look for *_dense.jsonl first
        jsonl_dir = Path(jsonl_path).parent
        dense_files = list(jsonl_dir.glob("*_dense.jsonl"))

        if dense_files:
            # Use the dense JSONL file (prioritizes rich metadata)
            jsonl_path = str(dense_files[0])
        else:
            # Fallback: use the wildcard pattern as-is
            # (allows explicit path like "classicmodels_dense.jsonl")
            if "*" in jsonl_path or "?" in jsonl_path:
                jsonl_files = list(jsonl_dir.glob(Path(jsonl_path).name))
                if jsonl_files:
                    jsonl_path = str(jsonl_files[0])

        _analyzer_instance = SchemaAnalyzer(jsonl_path)

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
]
