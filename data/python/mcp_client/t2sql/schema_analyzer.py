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
import warnings
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict
from .config import (
    LINEAGE_METADATA_JSONL,
    SCHEMA_PATH_STRICT,
    SCHEMA_ANALYZER_INVALID_JSONL_PREVIEW,
    SCHEMA_ANALYZER_STRICT_JSONL,
)

class SchemaAnalyzer:
    """
    Analyzes JSONL schema metadata to build dynamic configuration.

    Replaces hardcoded table names, synonyms, and patterns with schema-driven inference.
    """

    def __init__(self, jsonl_path: str, lineage_jsonl_path: Optional[str] = None):
        """
        Initialize with path to JSONL metadata file.

        Args:
            jsonl_path: Path to dense JSONL file (e.g., dataset_dense.jsonl)
        """
        self.jsonl_path = jsonl_path
        self.lineage_jsonl_path = lineage_jsonl_path
        self._tables: Set[str] = set()
        self._table_columns: Dict[str, List[str]] = defaultdict(list)
        self._table_synonyms: Dict[str, List[str]] = defaultdict(list)
        self._column_synonyms: Dict[str, List[str]] = defaultdict(list)
        self._table_types: Dict[str, str] = {}  # table -> type
        self._fk_relationships: Dict[str, Dict[str, str]] = defaultdict(dict)  # {table: {fk_col: target_table}}
        self._fk_target_columns: Dict[str, Dict[str, str]] = defaultdict(dict)  # {table: {fk_col: target_col}}
        self._text_fk_relationships: Dict[str, Dict[str, str]] = defaultdict(dict)
        self._text_fk_target_columns: Dict[str, Dict[str, str]] = defaultdict(dict)
        self._lineage_fk_relationships: Dict[str, Dict[str, str]] = defaultdict(dict)
        self._lineage_fk_target_columns: Dict[str, Dict[str, str]] = defaultdict(dict)
        self._column_roles: Dict[str, Dict[str, str]] = defaultdict(dict)  # {table: {col: role}}
        self._column_datatypes: Dict[str, Dict[str, str]] = defaultdict(dict)  # {table: {col: datatype}}
        self._column_descriptions: Dict[str, Dict[str, str]] = defaultdict(dict)  # {table: {col: description}}
        self._column_primary_key_hints: Dict[str, bool] = {}
        self._column_foreign_key_hints: Dict[str, bool] = {}
        # Store original case for column names (lowercase -> original)
        self._column_original_case: Dict[str, str] = {}

        self._parse_jsonl()

    def _parse_jsonl(self):
        """Parse JSONL file and extract all metadata."""
        invalid_lines: List[Tuple[int, str]] = []
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    self._process_entry(data)
                except json.JSONDecodeError as e:
                    invalid_lines.append((line_no, str(e)))

        if invalid_lines:
            preview_count = max(0, int(SCHEMA_ANALYZER_INVALID_JSONL_PREVIEW))
            preview_items = invalid_lines[:preview_count]
            preview_text = "; ".join(
                f"line {line_no}: {err}"
                for line_no, err in preview_items
            )
            more_count = max(0, len(invalid_lines) - len(preview_items))
            if more_count:
                preview_text = f"{preview_text}; ... and {more_count} more" if preview_text else f"... and {more_count} more"

            message = (
                f"Invalid JSONL entries found in schema metadata '{self.jsonl_path}': "
                f"{len(invalid_lines)} invalid line(s)"
            )
            if preview_text:
                message = f"{message}. {preview_text}"

            if SCHEMA_ANALYZER_STRICT_JSONL:
                raise ValueError(message)
            warnings.warn(message, RuntimeWarning)

        self._merge_fk_relationships()
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
        description_from_text = self._next_field_after(text, "Description:")
        merged_description = str(description or "").strip()
        if description_from_text:
            if merged_description:
                if description_from_text.lower() not in merged_description.lower():
                    merged_description = f"{merged_description}. {description_from_text}".strip(" .")
            else:
                merged_description = str(description_from_text).strip()
        self._column_descriptions[table.lower()][column.lower()] = merged_description

        fk_source = f"{description} {text}"
        fk_match = re.search(r'FK to\s+(\w+)\.(\w+)', fk_source, re.IGNORECASE)
        if fk_match:
            self._record_fk_edge(
                self._text_fk_relationships,
                self._text_fk_target_columns,
                source_table=table,
                source_column=column,
                target_table=fk_match.group(1),
                target_column=fk_match.group(2),
            )

        combined_key_text = f"{description} {text}".lower()
        col_key = f"{table.lower()}.{column.lower()}"
        self._column_primary_key_hints[col_key] = bool(
            re.search(r"\bprimary\s+key\b", combined_key_text)
            or re.search(r"\b(?:part\s+of\s+)?composite\s+pk\b", combined_key_text)
        )
        self._column_foreign_key_hints[col_key] = bool(
            fk_match or re.search(r"\bforeign\s+key\b", combined_key_text)
        )

        # Extract column role
        role = self._next_field_after(text, "Role:")
        if role:
            self._column_roles[table.lower()][column.lower()] = role.strip().lower()

        # Extract datatype
        datatype = metadata.get("datatype", "")
        self._column_datatypes[table.lower()][column.lower()] = datatype

    def _record_fk_edge(
        self,
        table_map: Dict[str, Dict[str, str]],
        target_col_map: Dict[str, Dict[str, str]],
        *,
        source_table: str,
        source_column: str,
        target_table: str,
        target_column: str = "",
    ) -> None:
        src_table = str(source_table or "").strip().lower()
        src_col = str(source_column or "").strip().lower()
        tgt_table = str(target_table or "").strip().lower()
        tgt_col = str(target_column or "").strip()
        if not src_table or not src_col or not tgt_table:
            return
        table_map[src_table][src_col] = tgt_table
        target_col_map[src_table][src_col] = tgt_col

    def _known_schema_column(self, table: str, column: str) -> bool:
        table_l = str(table or "").strip().lower()
        col_l = str(column or "").strip().lower()
        if not table_l or not col_l:
            return False
        return col_l in set(self._table_columns.get(table_l, []))

    def _known_schema_table(self, table: str) -> bool:
        return str(table or "").strip().lower() in self._tables

    def _parse_optional_jsonl(
        self,
        path: Optional[str],
        *,
        label: str,
    ) -> List[Dict[str, Any]]:
        resolved_path = self._resolve_optional_jsonl_path(path)
        if resolved_path is None:
            return []

        invalid_lines: List[Tuple[int, str]] = []
        rows: List[Dict[str, Any]] = []
        with open(resolved_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    invalid_lines.append((line_no, str(e)))

        if invalid_lines:
            preview_count = max(0, int(SCHEMA_ANALYZER_INVALID_JSONL_PREVIEW))
            preview_items = invalid_lines[:preview_count]
            preview_text = "; ".join(
                f"line {line_no}: {err}"
                for line_no, err in preview_items
            )
            more_count = max(0, len(invalid_lines) - len(preview_items))
            if more_count:
                preview_text = f"{preview_text}; ... and {more_count} more" if preview_text else f"... and {more_count} more"

            message = (
                f"Invalid JSONL entries found in {label} metadata '{resolved_path}': "
                f"{len(invalid_lines)} invalid line(s)"
            )
            if preview_text:
                message = f"{message}. {preview_text}"

            if SCHEMA_ANALYZER_STRICT_JSONL:
                raise ValueError(message)
            warnings.warn(message, RuntimeWarning)

        return rows

    def _resolve_optional_jsonl_path(self, path: Optional[str]) -> Optional[Path]:
        raw = str(path or "").strip()
        if not raw:
            return None

        path_obj = Path(raw)
        if (
            "*" not in raw
            and "?" not in raw
            and "[" not in raw
            and path_obj.exists()
            and path_obj.is_file()
        ):
            return path_obj.resolve()

        candidates: List[Path] = []
        if path_obj.is_dir():
            candidates = sorted(path_obj.glob("*_lineage.jsonl"))
            if not candidates:
                candidates = sorted(path_obj.glob("*.jsonl"))
        elif "*" in raw or "?" in raw or "[" in raw:
            candidates = sorted(path_obj.parent.glob(path_obj.name))
        elif raw.lower().endswith(".jsonl") and path_obj.exists() and path_obj.is_file():
            candidates = [path_obj]

        if not candidates:
            return None
        return candidates[0].resolve()

    def _load_lineage_fk_relationships(self) -> None:
        rows = self._parse_optional_jsonl(self.lineage_jsonl_path, label="lineage")
        for row in rows:
            source_table = row.get("from_table")
            source_column = row.get("from_column")
            target_table = row.get("to_table")
            target_column = row.get("to_column")

            src_table_l = str(source_table or "").strip().lower()
            src_col_l = str(source_column or "").strip().lower()
            tgt_table_l = str(target_table or "").strip().lower()
            if not src_table_l or not src_col_l or not tgt_table_l:
                continue

            if not self._known_schema_column(src_table_l, src_col_l):
                continue
            if not self._known_schema_table(tgt_table_l):
                continue

            self._record_fk_edge(
                self._lineage_fk_relationships,
                self._lineage_fk_target_columns,
                source_table=src_table_l,
                source_column=src_col_l,
                target_table=tgt_table_l,
                target_column=str(target_column or "").strip(),
            )
            self._column_foreign_key_hints[f"{src_table_l}.{src_col_l}"] = True

    def _merge_fk_relationships(self) -> None:
        self._load_lineage_fk_relationships()

        conflicts: List[str] = []

        for src_table, fk_map in self._lineage_fk_relationships.items():
            for src_col, tgt_table in (fk_map or {}).items():
                self._fk_relationships[src_table][src_col] = tgt_table
                tgt_col = self._lineage_fk_target_columns.get(src_table, {}).get(src_col, "")
                if tgt_col:
                    self._fk_target_columns[src_table][src_col] = tgt_col

        for src_table, fk_map in self._text_fk_relationships.items():
            for src_col, tgt_table in (fk_map or {}).items():
                existing_target = self._fk_relationships.get(src_table, {}).get(src_col)
                if existing_target:
                    if existing_target != tgt_table:
                        conflicts.append(
                            f"{src_table}.{src_col}: lineage -> {existing_target}, text -> {tgt_table}"
                        )
                    continue
                self._fk_relationships[src_table][src_col] = tgt_table
                tgt_col = self._text_fk_target_columns.get(src_table, {}).get(src_col, "")
                if tgt_col:
                    self._fk_target_columns[src_table][src_col] = tgt_col

        if conflicts:
            preview_count = max(0, int(SCHEMA_ANALYZER_INVALID_JSONL_PREVIEW))
            preview_items = conflicts[:preview_count]
            preview_text = "; ".join(preview_items)
            more_count = max(0, len(conflicts) - len(preview_items))
            if more_count:
                preview_text = f"{preview_text}; ... and {more_count} more" if preview_text else f"... and {more_count} more"
            warnings.warn(
                "Schema metadata FK conflict detected; using lineage edge and retaining text FK as fallback only. "
                + preview_text,
                RuntimeWarning,
            )

    def _next_field_after(self, text: str, field: str) -> Optional[str]:
        """Extract the value of a field after a given marker."""
        src = str(text or "")
        if not src or not field:
            return None

        lowered = src.lower()
        marker = field.lower()
        start = lowered.find(marker)
        if start < 0:
            return None

        value_start = start + len(field)
        tail = src[value_start:].lstrip()
        if not tail:
            return None

        # Dense metadata uses labeled sections such as:
        # Table: ... Column: ... Description: ... Synonyms: ... Example value: ...
        # Stop at the next known field marker instead of the first period so
        # abbreviations like "e.g." do not truncate the description.
        next_markers = [
            "Table:",
            "Column:",
            "Description:",
            "Role:",
            "Synonyms:",
            "Example value:",
            "Context:",
        ]
        end = len(tail)
        tail_lower = tail.lower()
        for candidate in next_markers:
            candidate_lower = candidate.lower()
            if candidate_lower == marker:
                continue
            idx = tail_lower.find(candidate_lower)
            if idx >= 0 and idx < end:
                end = idx

        value = tail[:end].strip()
        if not value:
            return None

        if field.strip().lower() == "role:":
            value = re.split(r"\.(?:\s|$)", value, maxsplit=1)[0].strip()

        return value.strip(" .")
        

    def _infer_table_types(self):
        """
        Infer table types from column patterns and metadata.

        Table Types:
        - transaction: Tables with metric-like numeric signals plus relational activity
        - dimension: Master tables (customers, products, employees)
        - bridge: Junction tables with composite FKs (orderdetails)
        - reference: Lookup tables (productlines, offices)
        """
        for table in self._tables:
            columns = self._table_columns[table]

            # Check for bridge table (multiple FKs)
            fk_count = len(self._fk_relationships.get(table, {}))

            roles = self._column_roles.get(table, {})
            datatypes = self._column_datatypes.get(table, {})

            metric_role_cols = {
                col for col in columns
                if str(roles.get(col, "")).strip().lower() in {"metric", "measure", "amount", "quantity"}
            }

            numeric_dtype_cols = {
                col for col in columns
                if any(
                    nt in str(datatypes.get(col, "")).upper()
                    for nt in {
                        "INT", "INTEGER", "DECIMAL", "FLOAT", "DOUBLE", "NUMERIC",
                        "SMALLINT", "BIGINT", "TINYINT", "MEDIUMINT", "REAL",
                    }
                )
            }
            non_key_numeric_cols = {
                col for col in numeric_dtype_cols
                if not self._column_primary_key_hints.get(f"{table}.{col}", False)
                and not self._column_foreign_key_hints.get(f"{table}.{col}", False)
                and not (
                    col.endswith("id")
                    or col.endswith("number")
                    or col.endswith("code")
                    or "postal" in col
                    or "zip" in col
                )
            }

            has_temporal_signal = any(
                any(k in col for k in ("date", "time", "timestamp"))
                or "DATE" in str(datatypes.get(col, "")).upper()
                or "TIME" in str(datatypes.get(col, "")).upper()
                for col in columns
            )

            has_metric_signal = bool(metric_role_cols) or bool(non_key_numeric_cols)
            has_activity_signal = has_metric_signal and (has_temporal_signal or fk_count >= 2)

            # Check for dimension indicators
            has_name = any('name' in col for col in columns)
            is_master = table.endswith('s') and fk_count == 0 and has_name

            if has_activity_signal and fk_count >= 1:
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

    def get_fk_target_column(self, table: str, column: str) -> str:
        """
        Return FK target column name for a given source table.column when known.
        """
        t = str(table or "").strip().lower()
        c = str(column or "").strip().lower()
        if not t or not c:
            return ""
        return str(self._fk_target_columns.get(t, {}).get(c, "")).strip()

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

    def get_column_description(self, table: str, column: str) -> str:
        """
        Get description text for a specific column from metadata.

        Returns merged metadata.description and parsed "Description:" text
        when available.
        """
        t = (table or "").lower().strip()
        c = (column or "").lower().strip()
        return str(self._column_descriptions.get(t, {}).get(c, "")).strip()

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
_analyzer_instances: Dict[str, SchemaAnalyzer] = {}


def _resolve_lineage_metadata_path(spec: Optional[str]) -> Optional[str]:
    raw = str(spec or "").strip()
    if not raw:
        return None

    path_obj = Path(raw)
    if (
        "*" not in raw
        and "?" not in raw
        and "[" not in raw
        and path_obj.exists()
        and path_obj.is_file()
    ):
        return str(path_obj.resolve())

    candidates: List[Path] = []
    if path_obj.is_dir():
        candidates = sorted(path_obj.glob("*_lineage.jsonl"))
        if not candidates:
            candidates = sorted(path_obj.glob("*.jsonl"))
    elif "*" in raw or "?" in raw or "[" in raw:
        candidates = sorted(path_obj.parent.glob(path_obj.name))
    elif raw.lower().endswith(".jsonl") and path_obj.exists() and path_obj.is_file():
        candidates = [path_obj]

    if not candidates:
        return None
    return str(candidates[0].resolve())

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
    global _analyzer_instances

    strict_path_mode = bool(SCHEMA_PATH_STRICT)

    if jsonl_path is None:
        from .config import SCHEMA_METADATA_JSONL
        jsonl_path = SCHEMA_METADATA_JSONL

    # Convert to Path object
    jsonl_path_obj = Path(jsonl_path)

    # If caller provides an explicit file path, keep it as source of truth.
    if (
        "*" not in str(jsonl_path)
        and "?" not in str(jsonl_path)
        and jsonl_path_obj.exists()
        and jsonl_path_obj.is_file()
    ):
        final_path = jsonl_path_obj
        cache_key = str(final_path.resolve())
        lineage_path = _resolve_lineage_metadata_path(LINEAGE_METADATA_JSONL)
        if lineage_path:
            cache_key = f"{cache_key}::lineage={lineage_path}"
        if cache_key not in _analyzer_instances:
            _analyzer_instances[cache_key] = SchemaAnalyzer(
                str(final_path.resolve()),
                lineage_jsonl_path=lineage_path,
            )
        return _analyzer_instances[cache_key]

    # Handle case where path is a directory (not a file)
    if jsonl_path_obj.is_dir():
        # It's a directory, search for jsonl files inside
        jsonl_dir = jsonl_path_obj
    else:
        # It's a file path (or wildcard pattern), get the parent directory
        jsonl_dir = jsonl_path_obj.parent

    # Priority 1: Look for *_dense.jsonl first (rich metadata with synonyms/roles)
    dense_files = sorted(jsonl_dir.glob("*_dense.jsonl"))

    if dense_files:
        if strict_path_mode and len(dense_files) > 1:
            preview = ", ".join(str(p.name) for p in dense_files[:5])
            raise ValueError(
                f"Ambiguous dense schema metadata in '{jsonl_dir}': found {len(dense_files)} files ({preview}). "
                "Set SCHEMA_METADATA_JSONL to an explicit file path or disable SCHEMA_PATH_STRICT."
            )
        # Use the first dense JSONL file in sorted order (deterministic)
        jsonl_path = str(dense_files[0])
    else:
        # Priority 2: Use any .jsonl file
        # Handle wildcard patterns or find any jsonl file
        if "*" in jsonl_path or "?" in jsonl_path:
            # It's a wildcard pattern, use it as-is
            jsonl_files = sorted(jsonl_dir.glob(Path(jsonl_path).name))
            if jsonl_files:
                if strict_path_mode and len(jsonl_files) > 1:
                    preview = ", ".join(str(p.name) for p in jsonl_files[:5])
                    raise ValueError(
                        f"Ambiguous schema metadata wildcard '{jsonl_path}': found {len(jsonl_files)} files ({preview}). "
                        "Set SCHEMA_METADATA_JSONL to an explicit file path or disable SCHEMA_PATH_STRICT."
                    )
                jsonl_path = str(jsonl_files[0])
        elif jsonl_path_obj.is_dir() or not jsonl_path_obj.exists():
            # Path is a directory or doesn't exist, find any jsonl file
            jsonl_files = sorted(jsonl_dir.glob("*.jsonl"))
            if jsonl_files:
                if strict_path_mode and len(jsonl_files) > 1:
                    preview = ", ".join(str(p.name) for p in jsonl_files[:5])
                    raise ValueError(
                        f"Ambiguous schema metadata in '{jsonl_dir}': found {len(jsonl_files)} jsonl files ({preview}). "
                        "Set SCHEMA_METADATA_JSONL to an explicit file path or disable SCHEMA_PATH_STRICT."
                    )
                jsonl_path = str(jsonl_files[0])

    # Validate the final path
    final_path = Path(jsonl_path)
    if not final_path.exists() or final_path.is_dir():
        raise ValueError(
            f"Cannot find valid JSONL file. Path: {jsonl_path}. "
            f"Directory: {jsonl_dir}. "
            f"Please ensure at least one .jsonl file exists."
        )

    cache_key = str(final_path.resolve())
    lineage_path = _resolve_lineage_metadata_path(LINEAGE_METADATA_JSONL)
    if lineage_path:
        cache_key = f"{cache_key}::lineage={lineage_path}"
    if cache_key not in _analyzer_instances:
        _analyzer_instances[cache_key] = SchemaAnalyzer(
            str(final_path.resolve()),
            lineage_jsonl_path=lineage_path,
        )

    return _analyzer_instances[cache_key]

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


def get_column_description(table: str, column: str) -> str:
    """Get description text for a specific column from metadata (schema-agnostic)."""
    return get_schema_analyzer().get_column_description(table, column)


def get_anchor_columns(table: str) -> List[str]:
    """Get anchor/scoring columns for a table."""
    return get_schema_analyzer().get_anchor_columns(table)


def is_transaction_table(table: str) -> bool:
    """Check if table is a transaction table."""
    return get_schema_analyzer().is_transaction_table(table)


def get_fk_relationships() -> Dict[str, Dict[str, str]]:
    """Get all FK relationships from schema."""
    return get_schema_analyzer().get_fk_relationships()


def get_fk_target_column(table: str, column: str) -> str:
    """Get FK target column for a given source table.column when known."""
    return get_schema_analyzer().get_fk_target_column(table, column)


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


def _identifier_phrase_hints(table_lower: str, col_lower: str) -> Set[str]:
    analyzer = get_schema_analyzer()
    out: Set[str] = set()
    for syn in analyzer._column_synonyms.get(f"{table_lower}.{col_lower}", []):
        syn_l = str(syn or "").strip().lower()
        if not syn_l:
            continue
        out.add(syn_l)
        out.add(syn_l.replace(" ", ""))
    return out


def _column_has_identifier_hint(table_lower: str, col_lower: str) -> bool:
    hints = _identifier_phrase_hints(table_lower, col_lower)
    for hint in hints:
        if any(tok in hint for tok in ("primarykey", "primary key", "identifier", "uniqueid", "unique id")):
            return True
        if re.search(r"\b(?:id|key)\b", hint):
            return True
    return False


def _column_matches_entity_key_shape(table_lower: str, col_lower: str) -> bool:
    singular = (get_singular_form(table_lower) or "").replace("_", "").lower()
    col_compact = str(col_lower or "").replace("_", "").lower()
    if not singular or not col_compact:
        return col_compact == "id"

    if col_compact == "id":
        return True

    direct_forms = {
        singular,
        f"{singular}id",
        f"{singular}key",
        f"{singular}code",
        f"{singular}number",
    }
    if col_compact in direct_forms:
        return True

    return col_compact.startswith(singular) and any(
        col_compact.endswith(suffix) for suffix in ("id", "key", "code", "number")
    )


def _column_identifier_score(table_lower: str, col_lower: str) -> int:
    analyzer = get_schema_analyzer()
    key = f"{table_lower}.{col_lower}"
    is_pk = bool(analyzer._column_primary_key_hints.get(key))
    is_fk = bool(
        analyzer._column_foreign_key_hints.get(key)
        or col_lower in analyzer._fk_relationships.get(table_lower, {})
    )
    score = 0

    base_identifier_evidence = False

    if is_pk:
        score += 100
        base_identifier_evidence = True
    if _column_has_identifier_hint(table_lower, col_lower):
        score += 30
        base_identifier_evidence = True
    if _column_matches_entity_key_shape(table_lower, col_lower):
        score += 30
        base_identifier_evidence = True

    if is_fk:
        base_identifier_evidence = True

    # Suffix patterns are useful, but only after we already have some other
    # evidence that the column behaves like an entity key. This avoids false
    # positives such as postalCode or faxNumber.
    if base_identifier_evidence:
        if col_lower.endswith("id"):
            score += 18
        elif col_lower.endswith("number"):
            score += 16
        elif col_lower.endswith("code"):
            score += 14
        elif col_lower.endswith("key"):
            score += 10

    if is_fk and not is_pk:
        score -= 15

    return score


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

    scored_cols: List[Tuple[int, str]] = []
    for col in columns:
        score = _column_identifier_score(table_lower, col)
        if score > 0:
            scored_cols.append((score, col))

    if scored_cols:
        scored_cols.sort(key=lambda item: (-item[0], item[1]))
        return _get_original_case(table_lower, scored_cols[0][1])

    # Fallback: column name matches table singular (e.g., productlines.productLine)
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
    1. Role from metadata (metric/measure)
    2. Datatype from metadata (INT, DECIMAL, FLOAT, DOUBLE, NUMERIC)
    3. Generic cross-domain metric naming cues
    
    Args:
        table: Table name
        column: Column name
    
    Returns:
        True if column is numeric
    """
    t = (table or "").lower()
    c = (column or "").lower()

    # 1. Check role from metadata first
    role = get_column_role(t, c)
    if role in ('metric', 'measure', 'quantity', 'amount'):
        return True
    
    # 2. Check datatype from metadata
    dtype = get_column_datatype(t, c).upper()
    numeric_types = {'INT', 'INTEGER', 'DECIMAL', 'FLOAT', 'DOUBLE', 'NUMERIC', 
                     'SMALLINT', 'BIGINT', 'TINYINT', 'MEDIUMINT', 'REAL'}
    if any(nt in dtype for nt in numeric_types):
        return True
    
    # 3. Check generic cross-domain name patterns.
    # Keep this intentionally narrow and avoid domain-specific tokens such as
    # MSRP / buyPrice / creditLimit / quantityInStock.
    numeric_patterns = (
        'amount', 'price', 'quantity', 'total', 'count',
        'sum', 'cost', 'value', 'rate', 'percent', 'ratio',
    )
    # Use suffix/whole-word matching to avoid 'country' matching 'count'
    if c in numeric_patterns or any(c.endswith(p) for p in numeric_patterns):
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
    
    # 2. Check narrow generic temporal name patterns as a last fallback.
    # Keep this metadata-first and avoid business-process verbs such as
    # shipped / ordered / paid, which are more domain-lexical than temporal.
    date_patterns = (
        'date', 'time', 'timestamp', 'created', 'updated', 'modified',
        'expired', 'due',
    )
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
    
    # 1. Prefer metadata-aware identifier score over raw suffix matching.
    if _column_identifier_score(t, c) > 0:
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
    2. If question semantically overlaps with a numeric column → numeric column
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
    1. If question semantically overlaps with a numeric column → that column
    2. Fallback: first numeric column
    
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
    - 'metric': Numeric measure
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
    'get_column_description',
    'get_anchor_columns',
    'is_transaction_table',
    'get_fk_relationships',
    'get_fk_target_column',
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
