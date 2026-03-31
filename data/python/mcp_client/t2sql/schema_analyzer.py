# ======================================================================
# ========================= schema_analyzer.py =========================
# ======================================================================
# Dynamic Schema Analyzer - Extracts schema metadata from JSONL for agnostic Text-to-SQL.
# This module parses the JSONL metadata file to build a dynamic schema configuration,
# eliminating hardcoded table names, synonyms, and patterns.
# Purpose:
# - Auto-detect table types (transaction, dimension, bridge) from column patterns
# - Extract synonyms from metadata
# - Build FK relationship graph
# - Infer table-specific rules dynamically

# ======================================================================
# ====================== STANDARD LIBRARY IMPORT =======================
# ======================================================================
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict

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

        # Extract FK info.
        # Dense JSONL entries often store FK cues in `text` while sparse entries
        # use `metadata.description`, so parse both sources.
        description = metadata.get("description", "")
        fk_source = f"{description} {text}"
        fk_match = re.search(r'FK to\s+(\w+)\.(\w+)', fk_source, re.IGNORECASE)
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

# ============================ GLOBAL INSTANCE ============================
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

# ==================== CONVENIENCE FUNCTIONS ============================

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

# ==================== DYNAMIC COLUMN INFERENCE =========================

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


def get_all_entity_variants(table: str) -> Set[str]:
    """
    Get all variants for entity detection (schema-agnostic synonym expansion).
    
    Builds comprehensive set of variants from:
    1. Table name (exact)
    2. Singular form
    3. Spaced forms (for underscore names)
    4. Table synonyms from metadata
    5. Column synonyms that reference the table
    
    This is used for robust entity detection in questions.
    
    Args:
        table: Table name
    
    Returns:
        Set of all entity variants (lowercase)
    """
    analyzer = get_schema_analyzer()
    t = (table or "").lower().strip()
    if not t:
        return set()
    
    variants: Set[str] = {t}
    
    # Singular form
    singular = get_singular_form(t)
    if singular:
        variants.add(singular)
    
    # Spaced forms
    if "_" in t:
        variants.add(t.replace("_", " "))
    if singular and "_" in singular:
        variants.add(singular.replace("_", " "))
    
    # Table synonyms from metadata
    synonyms = analyzer.get_table_synonyms(t)
    for syn in synonyms:
        s = (syn or "").lower().strip()
        if s and len(s) >= 3:  # Avoid very short synonyms
            variants.add(s)
    
    return {v for v in variants if v}


def get_column_entity_variants(table: str, column: str) -> Set[str]:
    """
    Get all variants for a column entity detection (schema-agnostic).
    
    Builds comprehensive set from:
    1. Column name (exact)
    2. Spaced camelCase (customerName -> customer name)
    3. Spaced snake_case (order_date -> order date)
    4. Column synonyms from metadata
    
    Args:
        table: Table name
        column: Column name
    
    Returns:
        Set of all column variants (lowercase)
    """
    import re
    
    analyzer = get_schema_analyzer()
    c = (column or "").lower().strip()
    t = (table or "").lower().strip()
    if not c:
        return set()
    
    variants: Set[str] = {c}
    
    # Spaced camelCase
    spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', column).lower()
    if spaced != c:
        variants.add(spaced)
    
    # Spaced snake_case
    if "_" in c:
        variants.add(c.replace("_", " "))
    
    # Column synonyms from metadata
    if t:
        synonyms = analyzer.get_column_synonyms(t, c)
        for syn in synonyms:
            s = (syn or "").lower().strip()
            if s and len(s) >= 2:
                variants.add(s)
    
    return {v for v in variants if v}


def get_column_datatype(table: str, column: str) -> str:
    """
    Get column datatype from schema metadata.
    
    Returns datatype string (e.g., 'VARCHAR', 'INT', 'DECIMAL', 'DATE')
    or empty string if not found.
    
    Schema-agnostic - reads from parsed JSONL metadata.
    
    Args:
        table: Table name
        column: Column name
    
    Returns:
        Datatype string (uppercase) or empty string
    """
    analyzer = get_schema_analyzer()
    t = (table or "").lower().strip()
    c = (column or "").lower().strip()
    
    datatypes = analyzer._column_datatypes.get(t, {})
    dtype = datatypes.get(c, "")
    return str(dtype).upper().strip()


def get_column_role(table: str, column: str) -> str:
    """
    Get column role from schema metadata.
    
    Returns role string (e.g., 'metric', 'dimension', 'id')
    or empty string if not found.
    
    Schema-agnostic - reads from parsed JSONL metadata.
    
    Args:
        table: Table name
        column: Column name
    
    Returns:
        Role string (lowercase) or empty string
    """
    analyzer = get_schema_analyzer()
    t = (table or "").lower().strip()
    c = (column or "").lower().strip()
    
    roles = analyzer._column_roles.get(t, {})
    return str(roles.get(c, "")).lower().strip()


# ==================== ENHANCED COLUMN ROLE DETECTION ===================
# Schema-agnostic column classification based on datatype + naming patterns

def is_numeric_column(table: str, column: str) -> bool:
    """
    Check if column is numeric (for ORDER BY, aggregation, comparison).
    
    Detection methods:
    1. Datatype from metadata (INT, DECIMAL, FLOAT, DOUBLE, NUMERIC)
    2. Column name patterns (amount, price, quantity, total, count, number)
    3. Role from metadata (metric)
    
    Args:
        table: Table name
        column: Column name
    
    Returns:
        True if column is numeric
    """
    t = (table or "").lower()
    c = (column or "").lower()
    
    # 1. Check datatype from metadata
    dtype = get_column_datatype(t, c).upper()
    numeric_types = {'INT', 'INTEGER', 'DECIMAL', 'FLOAT', 'DOUBLE', 'NUMERIC', 
                     'SMALLINT', 'BIGINT', 'TINYINT', 'MEDIUMINT', 'REAL'}
    if any(nt in dtype for nt in numeric_types):
        return True
    
    # 2. Check column name patterns
    numeric_patterns = ('amount', 'price', 'quantity', 'total', 'count', 
                       'sum', 'cost', 'value', 'rate', 'percent', 'ratio',
                       'limit', 'balance', 'fee', 'tax', 'discount', 'msrp',
                       'buyprice', 'creditlimit', 'quantityinstock')
    # Use suffix/whole-word matching to avoid 'country' matching 'count'
    if c in numeric_patterns or any(c.endswith(p) for p in numeric_patterns):
        return True
    
    # 3. Check role from metadata
    role = get_column_role(t, c)
    if role in ('metric', 'measure', 'quantity', 'amount'):
        return True
    
    return False


def is_date_column(table: str, column: str) -> bool:
    """
    Check if column is date/datetime (for temporal filters, ordering).
    
    Detection methods:
    1. Datatype from metadata (DATE, DATETIME, TIMESTAMP, TIME)
    2. Column name patterns (date, time, timestamp, created, updated)
    
    Args:
        table: Table name
        column: Column name
    
    Returns:
        True if column is date/datetime type
    """
    t = (table or "").lower()
    c = (column or "").lower()
    
    # 1. Check datatype from metadata
    dtype = get_column_datatype(t, c).upper()
    date_types = {'DATE', 'DATETIME', 'TIMESTAMP', 'TIME', 'YEAR'}
    if any(dt in dtype for dt in date_types):
        return True
    
    # 2. Check column name patterns
    date_patterns = ('date', 'time', 'timestamp', 'created', 'updated', 'modified',
                    'expired', 'due', 'shipped', 'ordered', 'required', 'paid')
    if any(p in c for p in date_patterns):
        return True
    
    return False


def is_id_column(table: str, column: str) -> bool:
    """
    Check if column is ID/key column (for JOIN conditions).
    
    Detection methods:
    1. Column ends with 'id', 'number', 'code' (excluding zip/postal)
    2. Column is FK (from FK relationships)
    3. Column matches {table_singular} pattern
    
    Args:
        table: Table name
        column: Column name
    
    Returns:
        True if column is an ID/key column
    """
    t = (table or "").lower()
    c = (column or "").lower()
    
    # Exclusions - these are not ID columns
    exclude_patterns = ('zipcode', 'postalcode', 'phonenumber', 'faxnumber')
    if any(c == ep or c.endswith(ep) for ep in exclude_patterns):
        return False
    
    # 1. Check name patterns
    id_patterns = ('id', 'number', 'code', 'key')
    if any(c.endswith(p) for p in id_patterns):
        return True
    
    # 2. Check if it's an FK column
    analyzer = get_schema_analyzer()
    fk_map = analyzer._fk_relationships.get(t, {})
    if c in fk_map:
        return True
    
    # 3. Check if column matches singular table name
    singular = get_singular_form(t)
    if c == singular or c == singular.replace('_', ''):
        return True
    
    return False


def is_name_column(table: str, column: str) -> bool:
    """
    Check if column is a name/display column.
    
    Detection methods:
    1. Column ends with 'name' (not 'codename')
    2. Role from metadata is 'name'
    3. Datatype is VARCHAR and column has 'name' in synonyms
    
    Args:
        table: Table name
        column: Column name
    
    Returns:
        True if column is a name column
    """
    t = (table or "").lower()
    c = (column or "").lower()
    
    # 1. Check name pattern
    if c.endswith('name') and not c.endswith('codename'):
        return True
    
    # 2. Check role from metadata
    role = get_column_role(t, c)
    if role == 'name':
        return True
    
    # 3. Check synonyms
    analyzer = get_schema_analyzer()
    synonyms = analyzer.get_column_synonyms(t, c)
    if 'name' in [s.lower() for s in synonyms]:
        return True
    
    return False


def get_columns_by_role(table: str, role: str) -> List[str]:
    """
    Get all columns of a specific role for a table.
    
    Roles:
    - 'numeric': Numeric columns (amount, price, quantity)
    - 'date': Date/datetime columns
    - 'id': ID/key columns (for JOINs)
    - 'name': Name/display columns
    - 'text': Text/string columns (VARCHAR, TEXT)
    
    Args:
        table: Table name
        role: Role to filter by
    
    Returns:
        List of column names (original case)
    """
    analyzer = get_schema_analyzer()
    t = (table or "").lower()
    columns = analyzer._table_columns.get(t, [])
    result = []
    
    for col in columns:
        match = False
        if role == 'numeric':
            match = is_numeric_column(t, col)
        elif role == 'date':
            match = is_date_column(t, col)
        elif role == 'id':
            match = is_id_column(t, col)
        elif role == 'name':
            match = is_name_column(t, col)
        elif role == 'text':
            dtype = get_column_datatype(t, col).upper()
            match = any(tt in dtype for tt in ('VARCHAR', 'CHAR', 'TEXT', 'STRING'))
        
        if match:
            result.append(_get_original_case(t, col))
    
    return result


def get_orderable_columns(table: str) -> List[str]:
    """
    Get columns suitable for ORDER BY (numeric + date columns).
    
    These columns make semantic sense for sorting results.
    
    Args:
        table: Table name
    
    Returns:
        List of orderable column names (original case)
    """
    numeric_cols = get_columns_by_role(table, 'numeric')
    date_cols = get_columns_by_role(table, 'date')
    return list(set(numeric_cols + date_cols))


def get_aggregatable_columns(table: str) -> List[str]:
    """
    Get columns suitable for aggregation (SUM, AVG, COUNT).
    
    Returns numeric columns that make sense to aggregate.
    
    Args:
        table: Table name
    
    Returns:
        List of aggregatable column names (original case)
    """
    return get_columns_by_role(table, 'numeric')


def get_joinable_columns(table: str) -> List[str]:
    """
    Get columns suitable for JOIN conditions (ID columns + FK columns).
    
    Args:
        table: Table name
    
    Returns:
        List of joinable column names (original case)
    """
    return get_columns_by_role(table, 'id')


def get_filterable_date_columns(table: str) -> List[str]:
    """
    Get date columns suitable for temporal filters.
    
    Args:
        table: Table name
    
    Returns:
        List of date column names (original case)
    """
    return get_columns_by_role(table, 'date')


def _question_semantic_tokens(question: str) -> Set[str]:
    q = (question or "").lower()
    return {
        tok
        for tok in re.findall(r"[a-z0-9]+", q)
        if tok and tok not in {
            "the", "a", "an", "of", "for", "to", "by", "in", "on", "at",
            "with", "without", "and", "or", "show", "list", "find", "give",
            "return", "display", "each", "every", "per", "all",
        }
    }


def _column_semantic_tokens(table: str, column: str) -> Set[str]:
    t = (table or "").lower()
    c = (column or "").strip()
    if not t or not c:
        return set()

    phrases: Set[str] = {c, c.replace("_", " ")}
    for syn in get_column_synonyms(t, c):
        s = str(syn or "").strip()
        if s:
            phrases.add(s)
            phrases.add(s.replace("_", " "))
            phrases.add(s.replace("-", " "))

    role = str(get_column_role(t, c) or "").strip().lower()
    if role:
        phrases.add(role)

    tokens: Set[str] = set()
    for text in phrases:
        normalized = re.sub(
            r"([a-z0-9])([A-Z])",
            r"\1 \2",
            str(text or "").replace("_", " ").lower(),
        )
        tokens.update(tok for tok in re.findall(r"[a-z0-9]+", normalized) if tok and not tok.isdigit())
    return tokens


def _best_semantic_numeric_column(table: str, question: str, columns: List[str]) -> Optional[str]:
    q_tokens = _question_semantic_tokens(question)
    if not q_tokens or not columns:
        return None

    best_col: Optional[str] = None
    best_score = 0.0
    for col in columns:
        col_tokens = _column_semantic_tokens(table, col)
        if not col_tokens:
            continue
        overlap = q_tokens.intersection(col_tokens)
        if not overlap:
            continue
        score = float(len(overlap))
        role = str(get_column_role(table, col) or "").strip().lower()
        if role in {"metric", "measure", "amount", "quantity"}:
            score += 0.5
        if score > best_score:
            best_col = col
            best_score = score
    return best_col


def infer_best_order_column(table: str, question: str) -> Optional[str]:
    """
    Infer the best column to ORDER BY based on question context.
    
    Strategy:
    1. If question mentions 'date', 'time', 'recent', 'latest' → date column
    2. If question mentions 'amount', 'price', 'revenue', 'sales' → numeric column
    3. If question asks for 'highest', 'lowest', 'top' → first numeric column
    4. Fallback: first orderable column
    
    Args:
        table: Table name
        question: Question text
    
    Returns:
        Best column name for ordering, or None
    """
    q = (question or "").lower()
    
    # Check for temporal ordering
    temporal_keywords = ('date', 'time', 'recent', 'latest', 'newest', 'oldest', 
                        'first', 'last', 'before', 'after')
    if any(kw in q for kw in temporal_keywords):
        date_cols = get_filterable_date_columns(table)
        if date_cols:
            return date_cols[0]
    
    numeric_cols = get_aggregatable_columns(table)
    semantic_numeric = _best_semantic_numeric_column(table, question, numeric_cols)
    if semantic_numeric:
        return semantic_numeric
    
    # Generic highest/lowest/top → first numeric column
    rank_keywords = ('highest', 'lowest', 'top', 'best', 'worst', 'most', 'least')
    if any(kw in q for kw in rank_keywords):
        if numeric_cols:
            return numeric_cols[0]
    
    # Fallback
    orderable = get_orderable_columns(table)
    return orderable[0] if orderable else None


def infer_best_aggregate_column(table: str, question: str) -> Optional[str]:
    """
    Infer the best column for aggregation based on question context.
    
    Strategy:
    1. If question mentions 'total', 'sum', 'revenue' → amount/sales column
    2. If question mentions 'average', 'avg' → price/amount column
    3. If question mentions 'count', 'how many' → ID column (for COUNT)
    4. Fallback: first numeric column
    
    Args:
        table: Table name
        question: Question text
    
    Returns:
        Best column name for aggregation, or None
    """
    q = (question or "").lower()
    numeric_cols = get_aggregatable_columns(table)
    
    if not numeric_cols:
        return None
    
    semantic_numeric = _best_semantic_numeric_column(table, question, numeric_cols)
    if semantic_numeric:
        return semantic_numeric
    
    # Return first numeric column as fallback
    return numeric_cols[0]


def get_column_semantic_type(table: str, column: str) -> str:
    """
    Get comprehensive semantic type for a column.
    
    Returns one of:
    - 'id': Primary key or foreign key
    - 'metric': Numeric measure (amount, quantity, price)
    - 'date': Date/datetime/timestamp
    - 'name': Name/descriptive text field
    - 'code': Classification code (not PK/FK)
    - 'text': Other text field
    - 'unknown': Cannot determine
    
    Args:
        table: Table name
        column: Column name
    
    Returns:
        Semantic type string
    """
    t = (table or "").lower()
    c = (column or "").lower()
    
    # Check in order of specificity
    if is_id_column(t, c):
        return 'id'
    if is_date_column(t, c):
        return 'date'
    if is_numeric_column(t, c):
        return 'metric'
    if is_name_column(t, c):
        return 'name'
    
    # Check for code columns (not ID but classification)
    if 'code' in c or 'type' in c or 'status' in c or 'category' in c:
        return 'code'
    
    # Check for text columns
    dtype = get_column_datatype(t, c).upper()
    if any(tt in dtype for tt in ('VARCHAR', 'CHAR', 'TEXT', 'STRING')):
        return 'text'
    
    return 'unknown'


def get_table_column_profile(table: str) -> Dict[str, List[str]]:
    """
    Get complete column profile for a table, organized by semantic type.
    
    Returns dict with:
    - 'id': ID/key columns
    - 'metric': Numeric columns
    - 'date': Date columns
    - 'name': Name columns
    - 'code': Code/category columns
    - 'text': Other text columns
    
    Args:
        table: Table name
    
    Returns:
        Dict mapping semantic type to list of column names
    """
    analyzer = get_schema_analyzer()
    t = (table or "").lower()
    columns = analyzer._table_columns.get(t, [])
    
    profile: Dict[str, List[str]] = defaultdict(list)
    
    for col in columns:
        sem_type = get_column_semantic_type(t, col)
        original_col = _get_original_case(t, col)
        profile[sem_type].append(original_col)
    
    return dict(profile)


# Export convenience
__all__ = [
    'SchemaAnalyzer',
    'get_schema_analyzer',
    'get_all_tables',
    'get_transaction_tables',
    'get_dimension_tables',
    'get_table_synonyms',
    'get_column_synonyms',
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
    # Synonym expansion utilities
    'get_all_entity_variants',
    'get_column_entity_variants',
    'get_column_datatype',
    'get_column_role',
    # Enhanced column role detection
    'is_numeric_column',
    'is_date_column',
    'is_id_column',
    'is_name_column',
    'get_columns_by_role',
    'get_orderable_columns',
    'get_aggregatable_columns',
    'get_joinable_columns',
    'get_filterable_date_columns',
    'infer_best_order_column',
    'infer_best_aggregate_column',
    'get_column_semantic_type',
    'get_table_column_profile',
]
