# ======================================================================
# ========================= testing_evaluation .py =====================
# ======================================================================
# Purpose:
# - Batch evaluation script for Text-to-SQL pipeline using MCP client.
# - Loads test cases from YAML file.
# - Executes predicted SQL and groundtruth SQL against MCP database.
# - Compares results for Valid SQL (VA), Execution Accuracy (EX), and Test Suite (TS)
#   across multiple database distributions.

# ======================================================================
# ====================== STANDARD LIBRARY IMPORT =======================
# ======================================================================
import asyncio
import argparse
import csv
import os
import re
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
import yaml
from dotenv import load_dotenv, find_dotenv
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
load_dotenv(find_dotenv(), override=False)
try:
    import openpyxl
    from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
    from openpyxl.formatting.rule import CellIsRule
    from openpyxl.utils.dataframe import dataframe_to_rows
    HAS_EXCEL = True
except ImportError:
    HAS_EXCEL = False
os.environ.setdefault("OLLAMA_URL", "http://ollama:11434")
import httpx
# ======================================================================
# ======================== LOCAL MODULE IMPORT =========================
# ======================================================================
from .config import (
    MCP_BASE,
    CHROMA_DIR,
    COLLECTION_NAME,
    SPARSE_METADATA_JSONL,
    OLLAMA_URL,
    ROUTING_MODEL,
    SMALL_MODEL,
    LARGE_MODEL,
    ROUTING_MODEL_P2,
    SMALL_MODEL_P2,
    LARGE_MODEL_P2,
    ROUTING_MODEL_P3,
    SMALL_MODEL_P3,
    LARGE_MODEL_P3,
    OLLAMA_KEEP_ALIVE,
    EXECUTION_EMPTY_RESULT_CLASSIFY,
    format_heuristic_observability,
)
from .state import make_state
from .graph import build_app
from .mcp_execution import query_db_full, _mcp_call  # Pakai FULL query tanpa LIMIT dan _mcp_call helper
from .helpers_core import _preexecute_sql_safety_pass
from .utils import resolve_jsonl_paths

JAKARTA_TZ = timezone(timedelta(hours=7), name="UTC+07:00")


def _jakarta_now() -> datetime:
    return datetime.now(JAKARTA_TZ)


def _format_jakarta_timestamp(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC+07:00")


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _print_run_timestamp_block(label: str, dt: datetime, *, status: str | None = None, elapsed_seconds: float | None = None) -> None:
    print("\n" + "=" * 120)
    print(f"{label}: {_format_jakarta_timestamp(dt)}")
    print("Timezone: Asia/Jakarta (UTC+07:00)")
    if status:
        print(f"Status: {status}")
    if elapsed_seconds is not None:
        print(f"Elapsed: {_format_duration(elapsed_seconds)} ({elapsed_seconds:.2f}s)")
    print("=" * 120)

def load_testset(yaml_path: str) -> list[dict]:
    """Load test cases dari file YAML."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("cases", [])


def resolve_scenario_config(scenario: str | None) -> tuple[str, dict]:
    """
    Resolve thesis evaluation scenario presets.

    P1/P2/P3:
      - current full system
      - differs only by router/small/large model family
    """
    name = str(scenario or "P1").strip().upper() or "P1"
    base_flags = {
        "rag_enabled": True,
        "intent_ir_enabled": True,
        "multi_agent_enabled": True,
        "adaptive_model_enabled": True,
        "forced_model": "",
    }
    presets = {
        "P1": {
            **base_flags,
            "routing_model": ROUTING_MODEL,
            "small_model": SMALL_MODEL,
            "large_model": LARGE_MODEL,
        },
        "P2": {
            **base_flags,
            "routing_model": ROUTING_MODEL_P2,
            "small_model": SMALL_MODEL_P2,
            "large_model": LARGE_MODEL_P2,
        },
        "P3": {
            **base_flags,
            "routing_model": ROUTING_MODEL_P3,
            "small_model": SMALL_MODEL_P3,
            "large_model": LARGE_MODEL_P3,
        },
    }
    if name not in presets:
        valid = ", ".join(sorted(presets))
        raise ValueError(f"Unknown scenario '{name}'. Valid scenarios: {valid}")
    return name, dict(presets[name])


def apply_scenario_models(scenario_config: dict) -> dict:
    """
    Activate scenario-specific model names for evaluation runtime.

    agents.py imports model constants at module load time, so evaluation must
    update both this module's warmup globals and the agents module globals.
    """
    global ROUTING_MODEL, SMALL_MODEL, LARGE_MODEL

    routing_model = str(scenario_config.get("routing_model") or ROUTING_MODEL).strip()
    small_model = str(scenario_config.get("small_model") or SMALL_MODEL).strip()
    large_model = str(scenario_config.get("large_model") or LARGE_MODEL).strip()

    ROUTING_MODEL = routing_model
    SMALL_MODEL = small_model
    LARGE_MODEL = large_model

    from . import agents

    agents.ROUTING_MODEL = routing_model
    agents.SMALL_MODEL = small_model
    agents.LARGE_MODEL = large_model

    return {
        "routing_model": routing_model,
        "small_model": small_model,
        "large_model": large_model,
    }

def parse_difficulty_filter(filter_spec: str | None) -> tuple[str, set[int] | None]:
    """
    Parse difficulty filter specification.

    Returns:
        (difficulty, ids_set) where:
        - difficulty: "easy", "medium", "hard"
        - ids_set: None for all IDs, or set of specific IDs to include

    Examples:
        "" -> ("easy", None)
        "4" -> ("easy", {4})
        "4-7" -> ("easy", {4, 5, 6, 7})
        "1,3,5" -> ("easy", {1, 3, 5})
    """
    if not filter_spec:
        return "", None

    filter_spec = filter_spec.strip()

    # Parse range: "4-7" -> {4, 5, 6, 7}
    if "-" in filter_spec:
        try:
            start, end = filter_spec.split("-", 1)
            return "", set(range(int(start), int(end) + 1))
        except ValueError:
            return "", None

    # Parse comma-separated: "1,3,5" -> {1, 3, 5}
    if "," in filter_spec:
        try:
            return "", set(int(x.strip()) for x in filter_spec.split(","))
        except ValueError:
            return "", None

    # Single ID: "4" -> {4}
    try:
        return "", {int(filter_spec)}
    except ValueError:
        return "", None


def infer_base_database(
    base_database: str | None,
    ts_databases: list[str] | None = None,
) -> str:
    """
    Resolve the VA/EX base database dynamically.

    Priority:
    1. explicit argument
    2. T2SQL_BASE_DATABASE environment variable
    3. inferred base from TS names like foo_2 -> foo
    4. first TS database as a last fallback
    """
    explicit = str(base_database or "").strip()
    if explicit:
        return explicit

    env_default = str(os.getenv("T2SQL_BASE_DATABASE") or "").strip()
    if env_default:
        return env_default

    ordered_ts = [str(db or "").strip() for db in (ts_databases or []) if str(db or "").strip()]
    for db_name in ordered_ts:
        m = re.match(r"^(.*?)(?:[_-]\d+)$", db_name)
        if m and str(m.group(1) or "").strip():
            return str(m.group(1)).strip()

    if ordered_ts:
        return ordered_ts[0]

    raise ValueError(
        "base_database is not set and could not be inferred. "
        "Provide --base-database or set T2SQL_BASE_DATABASE."
    )


def infer_dataset_name_from_testset(testset_path: str | None) -> str | None:
    """
    Infer dataset/base name from common test-set filenames.
    Example: qa_sales_devset_ori.yml -> sales
    """
    raw = str(testset_path or "").strip()
    if not raw:
        return None

    tokens = [tok for tok in Path(raw).stem.lower().split("_") if tok]
    if tokens and tokens[0] in {"qa", "test", "eval", "evaluation"}:
        tokens = tokens[1:]

    trailing_tags = {
        "devset", "testset", "easy", "medium", "hard",
        "evaluation", "eval",
        "ori", "original", "formal",
        "mix", "mixed",
        "stress", "stresstest", "paraphrase",
        "validation", "val",
    }
    while tokens and tokens[-1] in trailing_tags:
        tokens.pop()

    dataset_name = "_".join(tokens).strip("_")
    return dataset_name or None


def infer_ts_databases(
    ts_databases: list[str] | None,
    *,
    base_database: str | None = None,
    testset_path: str | None = None,
) -> list[str]:
    """
    Resolve TS databases without dataset-specific hardcoded defaults.

    Priority:
    1. explicit CLI list
    2. T2SQL_TS_DATABASES environment variable
    3. base database hint -> {base}_2, {base}_3
    4. dataset name inferred from test-set filename -> {dataset}_2, {dataset}_3
    """
    ordered: list[str] = []
    for db in (ts_databases or []):
        db_name = str(db or "").strip()
        if db_name and db_name not in ordered:
            ordered.append(db_name)
    if ordered:
        return ordered

    env_ts = [x.strip() for x in str(os.getenv("T2SQL_TS_DATABASES") or "").split(",") if x.strip()]
    for db_name in env_ts:
        if db_name not in ordered:
            ordered.append(db_name)
    if ordered:
        return ordered

    base_hint = str(base_database or "").strip()
    if not base_hint:
        base_hint = str(os.getenv("T2SQL_BASE_DATABASE") or "").strip()
    if not base_hint:
        base_hint = str(infer_dataset_name_from_testset(testset_path) or "").strip()
    if base_hint:
        return [f"{base_hint}_2", f"{base_hint}_3"]

    return []

def normalize_value(val, tolerance: float = 0.0001) -> str:
    """
    Normalize a single value for comparison.
    
    For numeric values, rounds to handle floating-point precision issues.
    tolerance: relative tolerance for numeric comparison (default 0.01% = 0.0001)
    """
    if val is None:
        return "NULL"
    
    # Handle numeric types
    if isinstance(val, (int, float)):
        # Convert to float for uniform handling
        num = float(val)
        # Round to 2 decimal places for percentage-like values
        # This handles cases like 0 vs 0.0027 (integer vs float division)
        return f"{num:.2f}"
    
    return str(val).strip()

def _is_null_token(val) -> bool:
    """Return True if value represents SQL NULL in comparison layer."""
    if val is None:
        return True
    if isinstance(val, str) and val.strip().upper() == "NULL":
        return True
    return False


def _question_requires_null_to_zero(question: str) -> bool:
    """
    Return True when question explicitly asks NULL->0 style behavior.

    This is used to avoid forgiving NULL vs 0 mismatches when the prompt
    explicitly requires null handling to be shown in SQL.
    """
    q = (question or "").lower()
    patterns = (
        r"\bcoalesce\b",
        r"\bifnull\b",
        r"\bnvl\b",
        r"\bnull\s+as\s+(?:0|zero)\b",
        r"\b(?:0|zero)\s+instead\s+of\s+null\b",
        r"\breplace\s+null\s+(?:with|by)\s+(?:0|zero)\b",
        r"\bconvert\s+null\s+(?:to|into)\s+(?:0|zero)\b",
        r"\btreat\s+null\s+as\s+(?:0|zero)\b",
        r"\bdefault(?:ing)?\s+(?:null\s+)?(?:to|as)\s+(?:0|zero)\b",
    )
    return any(re.search(p, q) for p in patterns)


def _sql_has_null_to_zero_handling(sql: str) -> bool:
    """
    Detect explicit NULL->0 handling in SQL text.
    """
    s = (sql or "")

    # Direct function-based null handling
    if re.search(r"\b(coalesce|ifnull|nvl)\s*\(", s, flags=re.IGNORECASE):
        return True

    # CASE WHEN ... IS NULL THEN 0 ...
    if re.search(
        r"\bcase\b[\s\S]*?\bwhen\b[\s\S]*?\bis\s+null\b[\s\S]*?\bthen\s+0(?:\.0+)?\b",
        s,
        flags=re.IGNORECASE,
    ):
        return True

    return False


def _resolve_null_zero_equivalent(
    question: str,
    null_zero_mode: str,
) -> tuple[bool, bool, str]:
    """
    Resolve comparison behavior for NULL vs 0.

    Returns:
      - allow_null_zero_equivalent: enable tolerant value comparison
      - explicit_null_to_zero_required: whether question explicitly asks it
      - reason: short mode label for logging
    """
    mode = (null_zero_mode or "auto").strip().lower()
    if mode not in {"off", "on", "auto"}:
        mode = "auto"

    explicit_required = _question_requires_null_to_zero(question)

    if mode == "off":
        return False, explicit_required, "off"
    if mode == "on":
        return True, explicit_required, "on"

    # auto: allow NULL~0 only when question did NOT explicitly ask for null handling
    if explicit_required:
        return False, True, "auto(explicit-null-handling)"
    return True, False, "auto(default)"


def values_match(
    val1,
    val2,
    tolerance: float = 0.01,
    null_zero_equivalent: bool = False,
) -> bool:
    """
    Compare two values with numeric tolerance.
    
    For numeric values: match if within tolerance (default 1%)
    For strings: exact match (case-insensitive)
    """
    # Optional mode: treat NULL and numeric zero as equivalent.
    # Useful when gold SQL may use SUM(...) while prediction uses COALESCE(SUM(...), 0).
    if null_zero_equivalent:
        v1_null = _is_null_token(val1)
        v2_null = _is_null_token(val2)
        if v1_null and v2_null:
            return True
        if v1_null ^ v2_null:
            other = val2 if v1_null else val1
            try:
                return abs(float(other)) < 1e-9
            except (ValueError, TypeError):
                return False

    # Handle None
    if val1 is None and val2 is None:
        return True
    if val1 is None or val2 is None:
        return False
    
    # Try numeric comparison
    try:
        num1 = float(val1)
        num2 = float(val2)
        
        # If both are zero or very close to zero
        if abs(num1) < 1e-9 and abs(num2) < 1e-9:
            return True
        
        # Handle case where one is zero
        if abs(num1) < 1e-9 or abs(num2) < 1e-9:
            return abs(num1 - num2) < tolerance
        
        # Relative difference
        rel_diff = abs(num1 - num2) / max(abs(num1), abs(num2))
        return rel_diff <= tolerance
        
    except (ValueError, TypeError):
        # Fall back to string comparison
        return str(val1).strip().lower() == str(val2).strip().lower()

def compare_results(
    predicted_rows: list,
    groundtruth_rows: list,
    order_sensitive: bool = False,
    numeric_tolerance: float = 0.01,
    null_zero_equivalent: bool = False,
) -> tuple[bool, str]:
    """
    Compare execution results (Execution Accuracy).
    Returns (is_match, reason).
    
    - order_sensitive=False: Treat as SET comparison (row order doesn't matter)
    - order_sensitive=True: Row order must match exactly
    - numeric_tolerance: Relative tolerance for numeric values (default 1%)
    
    Note: Column names/aliases are IGNORED - only values matter.
    """
    if predicted_rows is None and groundtruth_rows is None:
        return True, "Both empty"
    
    if predicted_rows is None or groundtruth_rows is None:
        return False, "One result is None"
    
    # Check row count first
    if len(predicted_rows) != len(groundtruth_rows):
        return False, f"Row count mismatch: pred={len(predicted_rows)}, gt={len(groundtruth_rows)}"
    
    if len(predicted_rows) == 0:
        return True, "Both empty"
    
    # Check column count
    if predicted_rows and groundtruth_rows:
        pred_col_count = len(predicted_rows[0])
        gt_col_count = len(groundtruth_rows[0])
        if pred_col_count != gt_col_count:
            return False, f"Column count mismatch: pred={pred_col_count}, gt={gt_col_count}"
    
    try:
        from collections import Counter

        # Normalize row values and ignore column ORDER.
        pred_rows_norm = [tuple(normalize_value(v) for v in row.values()) for row in predicted_rows]
        gt_rows_norm = [tuple(normalize_value(v) for v in row.values()) for row in groundtruth_rows]

        def rows_match_ignore_col_order(pred_row: tuple, gt_row: tuple) -> bool:
            """Compare one row while ignoring column order."""
            if len(pred_row) != len(gt_row):
                return False

            # Fast exact path (column order already ignored)
            if Counter(pred_row) == Counter(gt_row):
                return True

            used = [False] * len(gt_row)

            def backtrack(i: int) -> bool:
                if i == len(pred_row):
                    return True
                p_val = pred_row[i]
                for j, g_val in enumerate(gt_row):
                    if used[j]:
                        continue
                    if values_match(
                        p_val,
                        g_val,
                        numeric_tolerance,
                        null_zero_equivalent=null_zero_equivalent,
                    ):
                        used[j] = True
                        if backtrack(i + 1):
                            return True
                        used[j] = False
                return False

            return backtrack(0)

        # ORDER SENSITIVE: row order must match exactly, but column order can differ.
        if order_sensitive:
            for i, (p_row, g_row) in enumerate(zip(pred_rows_norm, gt_rows_norm)):
                if not rows_match_ignore_col_order(p_row, g_row):
                    return False, f"Row order/content mismatch at row {i+1}"
            return True, "Exact match (ordered rows, unordered columns)"

        # ORDER INSENSITIVE: row order and column order can differ.
        pred_counter = Counter(tuple(sorted(r)) for r in pred_rows_norm)
        gt_counter = Counter(tuple(sorted(r)) for r in gt_rows_norm)
        if pred_counter == gt_counter:
            return True, "Exact match (unordered rows/columns)"

        # Build bipartite matching between predicted and groundtruth rows.
        n = len(pred_rows_norm)
        edges = [[] for _ in range(n)]
        for i, p_row in enumerate(pred_rows_norm):
            for j, g_row in enumerate(gt_rows_norm):
                if rows_match_ignore_col_order(p_row, g_row):
                    edges[i].append(j)

        # DFS-based maximum bipartite matching
        match_to_pred = [-1] * n

        def dfs(i: int, seen: list[bool]) -> bool:
            for j in edges[i]:
                if seen[j]:
                    continue
                seen[j] = True
                if match_to_pred[j] == -1 or dfs(match_to_pred[j], seen):
                    match_to_pred[j] = i
                    return True
            return False

        fully_matched = True
        for i in range(n):
            seen = [False] * n
            if not dfs(i, seen):
                fully_matched = False
                break

        if fully_matched:
            mode_suffix = " + NULL~0" if null_zero_equivalent else ""
            return True, f"Match with tolerance ({numeric_tolerance*100:.1f}%){mode_suffix}"

        # No match - count unmatched rows (column-order-insensitive exact signatures)
        missing_count = sum((gt_counter - pred_counter).values())
        extra_count = sum((pred_counter - gt_counter).values())
        reason = []
        if missing_count:
            reason.append(f"missing {missing_count} rows")
        if extra_count:
            reason.append(f"extra {extra_count} rows")
        return False, "; ".join(reason) if reason else "Content mismatch"
            
    except Exception as e:
        return False, f"Comparison error: {e}"


def save_to_excel(
    results: list,
    fieldnames: list,
    output_path: str,
    by_difficulty: dict,
    total: int,
    exec_success: int,
    result_match: int,
    latencies: list,
    ts_total_cases: int = 0,
    ts_success_cases: int = 0,
):
    """Save evaluation results to formatted Excel file."""
    
    # Calculate avg latency per difficulty (in seconds)
    latency_by_diff = {}
    for r in results:
        d = r["difficulty"]
        lat = r.get("latency_s", 0)
        if lat > 0:
            if d not in latency_by_diff:
                latency_by_diff[d] = []
            latency_by_diff[d].append(lat)
    wb = openpyxl.Workbook()
    
    # ============================================================
    # Sheet 1: Summary
    # ============================================================
    ws_summary = wb.active
    ws_summary.title = "Summary"
    
    # Styles
    header_font = Font(bold=True, color="000000")
    header_fill = PatternFill(start_color="D9E2F3", end_color="D9E2F3", fill_type="solid")
    match_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    mismatch_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    match_font = Font(bold=True, color="000000")
    mismatch_font = Font(bold=True, color="000000")
    border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )
    
    # Summary metrics
    ws_summary["A1"] = "EVALUATION SUMMARY"
    ws_summary["A1"].font = Font(bold=True, size=14)
    ws_summary.merge_cells("A1:C1")
    
    # Latencies already in seconds
    avg_latency_sec = sum(latencies)/len(latencies) if latencies else 0
    
    summary_data = [
        ("Metric", "Value", "Percentage"),
        ("Total Cases", total, ""),
        ("Result Match (VA)", sum(r.get("va_success", 0) for r in results), f"{100*sum(r.get('va_success', 0) for r in results)/total:.1f}%"),
        ("Result Match (EX)", result_match, f"{100*result_match/total:.1f}%"),
        (
            "Test Suite (TS)",
            ts_success_cases if ts_total_cases else "N/A",
            f"{100*ts_success_cases/ts_total_cases:.1f}%" if ts_total_cases else "N/A",
        ),
        ("Average Latency (s)", f"{avg_latency_sec:.2f}" if latencies else "N/A", ""),
    ]
    
    for row_idx, row_data in enumerate(summary_data, start=3):
        for col_idx, value in enumerate(row_data, start=1):
            cell = ws_summary.cell(row=row_idx, column=col_idx, value=value)
            cell.border = border
            cell.alignment = Alignment(vertical='center')
            if row_idx == 3:
                cell.font = header_font
                cell.fill = header_fill
    
    # By difficulty
    ws_summary["A10"] = "BY DIFFICULTY"
    ws_summary["A10"].font = Font(bold=True, size=12)
    
    diff_header = [
        "Difficulty",
        "Total",
        "EX Match",
        "Percentage VA Success (%)",
        "VA Success",
        "Percentage EX Match (%)",
        "TS Success",
        "Percentage TS Success (%)",
        "Avg Latency (s)"
    ]
    for col_idx, h in enumerate(diff_header, start=1):
        cell = ws_summary.cell(row=11, column=col_idx, value=h)
        cell.font = header_font
        cell.fill = header_fill
        cell.border = border
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)

    for row_idx, (d, stats) in enumerate(sorted(by_difficulty.items()), start=12):
        t = stats["total"]
        ex_match = stats["result_match"]
        va = stats.get("va_success", 0)
        ts_cases = stats.get("ts_cases", 0)
        ts_success = stats.get("ts_success", 0)
        ts_pct = f"{100*ts_success/ts_cases:.1f}%" if ts_cases else "N/A"
        # Calculate avg latency for this difficulty
        diff_latencies = latency_by_diff.get(d, [])
        avg_lat = f"{sum(diff_latencies)/len(diff_latencies):.2f}" if diff_latencies else "N/A"
        pct_va = f"{100*va/t:.1f}%" if t else "N/A"
        pct_ex = f"{100*ex_match/t:.1f}%" if t else "N/A"
        row_data = [d, t, ex_match, pct_va, va, pct_ex, ts_success, ts_pct, avg_lat]
        for col_idx, value in enumerate(row_data, start=1):
            cell = ws_summary.cell(row=row_idx, column=col_idx, value=value)
            cell.border = border
    
    # Adjust column widths
    ws_summary.column_dimensions['A'].width = 20
    ws_summary.column_dimensions['B'].width = 15
    ws_summary.column_dimensions['C'].width = 15
    ws_summary.column_dimensions['D'].width = 15
    ws_summary.column_dimensions['E'].width = 12
    ws_summary.column_dimensions['F'].width = 15
    ws_summary.column_dimensions['G'].width = 12
    ws_summary.column_dimensions['H'].width = 18
    ws_summary.column_dimensions['I'].width = 15
    ws_summary.freeze_panes = "A12"
    
    # ============================================================
    # Sheet 2: Detail Results
    # ============================================================
    ws_detail = wb.create_sheet("Results")
    
    # Header row
    for col_idx, header in enumerate(fieldnames, start=1):
        cell = ws_detail.cell(row=1, column=col_idx, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.border = border
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        cell.number_format = "@"
    ws_detail.row_dimensions[1].height = 24
    ws_detail.row_dimensions[1].hidden = False
    
    # Data rows
    for row_idx, result in enumerate(results, start=2):
        for col_idx, field in enumerate(fieldnames, start=1):
            value = result.get(field, "")
            cell = ws_detail.cell(row=row_idx, column=col_idx, value=value)
            cell.border = border
            cell.alignment = Alignment(wrap_text=True, vertical='top')
            
            # Highlight success flags with the same green/red behavior.
            if field in {"va_success", "exec_success", "ts_success"}:
                cell.value = "TRUE" if bool(value) else "FALSE"
                cell.font = match_font if bool(value) else mismatch_font
                if bool(value):
                    cell.fill = match_fill
                else:
                    cell.fill = mismatch_fill
                cell.alignment = Alignment(horizontal='center', vertical='center')
    
    # Adjust column widths
    col_widths = {
        "id": 8, "scenario": 10, "difficulty": 10, "expected_behavior": 18,
        "question": 50,
        "groundtruth_sql": 40, "predicted_sql": 40,
        "draft_source": 14, "ir_confidence": 12, "ir_gate_reasons": 28,
        "ambiguity_signals": 24, "cannot_answer": 12, "cannot_answer_reason": 34,
        "final_model_used": 12,  # Model tracking
        "va_success": 12, "va_notes": 30,
        "exec_success": 12, "exec_notes": 34, "pred_empty_result": 14, "gt_empty_result": 14,
        "ts_success": 12, "ts_failed_databases": 24,
        "pred_row_count": 28, "gt_row_count": 28,
        "latency_s": 12
    }
    for col_idx, field in enumerate(fieldnames, start=1):
        ws_detail.column_dimensions[openpyxl.utils.get_column_letter(col_idx)].width = col_widths.get(field, 15)

    # Reinforce boolean success cells with conditional formatting so common
    # spreadsheet viewers render the green/red cell background consistently.
    for field in ("va_success", "exec_success", "ts_success"):
        if field not in fieldnames:
            continue
        col_idx = fieldnames.index(field) + 1
        col_letter = openpyxl.utils.get_column_letter(col_idx)
        data_range = f"{col_letter}2:{col_letter}{max(len(results) + 1, 2)}"
        ws_detail.conditional_formatting.add(
            data_range,
            CellIsRule(
                operator="equal",
                formula=['"TRUE"'],
                fill=match_fill,
                font=match_font,
            ),
        )
        ws_detail.conditional_formatting.add(
            data_range,
            CellIsRule(
                operator="equal",
                formula=['"FALSE"'],
                fill=mismatch_fill,
                font=mismatch_font,
            ),
        )
    
    # Freeze header row
    ws_detail.freeze_panes = "A2"
    ws_detail.auto_filter.ref = f"A1:{openpyxl.utils.get_column_letter(len(fieldnames))}{max(len(results) + 1, 1)}"
    ws_detail.sheet_view.showGridLines = True
    ws_detail.sheet_view.tabSelected = True
    wb.active = wb.sheetnames.index("Results")
    
    # Save
    wb.save(output_path)

async def execute_sql(
    mcp: Client,
    sql: str,
    database: str | None = None,
) -> tuple[bool, list | None, str | None]:
    """
    Execute SQL via MCP (FULL mode - tanpa LIMIT) and return (success, rows, error).
    """
    sql_clean = sql.rstrip(";").strip()
    try:
        payload = await query_db_full(sql_clean, mcp=mcp, database=database)
        if isinstance(payload, dict) and payload.get("success"):
            # query_db_full returns rows as list of lists, convert to list of dicts
            columns = payload.get("columns", [])
            raw_rows = payload.get("rows", [])
            rows = [dict(zip(columns, row)) for row in raw_rows]
            return True, rows, None
        err = payload.get("error") if isinstance(payload, dict) else str(payload)
        return False, None, err
    except Exception as e:
        return False, None, str(e)

# --- VA: Valid SQL check using EXPLAIN ---
async def check_valid_sql(
    mcp: Client,
    sql: str,
    database: str | None = None,
) -> tuple[bool, str | None]:
    """
    Run EXPLAIN <sql> via MCP tool to check validity. Return (is_valid, error_message_if_any)
    """
    sql_clean = sql.rstrip(";").strip()
    try:
        req_payload = {"sql": sql_clean}
        if database:
            req_payload["database"] = str(database).strip()
        payload = await _mcp_call(mcp, "query_explain", req_payload)
        # Ambil dict dari CallToolResult jika perlu
        if not isinstance(payload, dict):
            if hasattr(payload, "structured_content") and isinstance(payload.structured_content, dict):
                payload = payload.structured_content
            elif hasattr(payload, "data") and isinstance(payload.data, dict):
                payload = payload.data
        if isinstance(payload, dict) and payload.get("success"):
            return True, None
        err = payload.get("error") if isinstance(payload, dict) else str(payload)
        return False, err
    except Exception as e:
        return False, str(e)

async def run_single_case(
    app,
    mcp: Client,
    paths: list[str],
    case: dict,
    scenario: str = "P1",
    scenario_config: dict | None = None,
    execute_groundtruth: bool = True,
    null_zero_mode: str = "auto",
    ts_databases: list[str] | None = None,
    base_database: str | None = None,
) -> dict:
    """
    Run a single test case and return evaluation result.
    """
    case_id = case.get("id", "unknown")
    question = case.get("question", "")
    groundtruth_sql = case.get("groundtruth_sql", "").strip()
    difficulty = case.get("difficulty", "unknown")
    expected_behavior = str(case.get("expected_behavior", "answerable") or "answerable").strip().lower()
    expects_cannot_answer = expected_behavior in {
        "cannot_answer",
        "abstain",
        "unanswerable",
        "out_of_schema",
    }
    allow_null_zero_equivalent, explicit_null_to_zero_required, null_zero_reason = (
        _resolve_null_zero_equivalent(question, null_zero_mode)
    )
    
    print(f"\n{'='*60}")
    print(f"[{case_id}] ({difficulty}) {question[:80]}...")
    print(f"{'='*60}")
    
    ts_db_list: list[str] = []
    for db in (ts_databases or []):
        db_name = str(db or "").strip()
        if db_name and db_name not in ts_db_list:
            ts_db_list.append(db_name)
    primary_db = infer_base_database(base_database, ts_db_list)
    # TS targets exclude base DB because base is already covered by EX.
    ts_db_targets = [db for db in ts_db_list if db != primary_db]
    eval_db_order: list[str] = [primary_db]
    for db in ts_db_targets:
        if db not in eval_db_order:
            eval_db_order.append(db)

    result = {
        "id": case_id,
        "scenario": str(scenario or "P1").strip().upper(),
        "difficulty": difficulty,
        "expected_behavior": expected_behavior,
        "question": question,
        "base_database": primary_db,
        "groundtruth_sql": groundtruth_sql,
        "predicted_sql": "",
        "draft_source": "",
        "ir_used": False,
        "ir_confidence": 0.0,
        "ir_plannable": False,
        "ir_gate_reasons": "",
        "ambiguity_signals": "",
        "cannot_answer": False,
        "cannot_answer_reason": "",
        "ir_hint_passed": False,
        "initial_model_used": "",  # Initial model chosen by router (3b/7b)
        "final_model_used": "",    # Standardized route label: ir_planner | llm_small | llm_large
        "va_success": False,       # Valid SQL (EXPLAIN)
        "va_notes": None,          # Notes/message from VA check
        "exec_success": False,
        "exec_notes": "",          # EX notes (success/mismatch/error detail)
        "result_match": False,     # Execution Accuracy (hasil sama)
        "match_reason": "",        # Alasan match/mismatch
        "pred_row_count": "",
        "gt_row_count": "",
        "ts_success": False,            # Test Suite: match across all configured DBs
        "ts_match_count": 0,            # Number of DBs that match for this case
        "ts_total_databases": len(ts_db_targets),  # Configured TS DB count for this case
        "ts_failed_databases": "",      # Newline-separated DBs that mismatched/failed
        "pred_empty_result": False,
        "gt_empty_result": False,
        "latency_s": 0,            # Primary: warm prompt->final SQL latency (seconds)
        "error": None,
    }
    pred_counts_by_db = {db: "N/A" for db in eval_db_order}
    gt_counts_by_db = {db: "N/A" for db in eval_db_order}

    def _format_row_counts(counts: dict[str, str]) -> str:
        return "\n".join(f"{db} = {counts.get(db, 'N/A')}" for db in eval_db_order)

    def _refresh_row_count_fields() -> None:
        result["pred_row_count"] = _format_row_counts(pred_counts_by_db)
        result["gt_row_count"] = _format_row_counts(gt_counts_by_db)

    _refresh_row_count_fields()
    
    # 1. Run pipeline
    t0 = time.perf_counter()
    try:
        init_state = make_state(
            paths,
            question,
            scenario=scenario,
            scenario_config=scenario_config,
        )
        state = app.invoke(init_state)
        
        # Get predicted SQL
        predicted_sql = state.get("final_sql") or state.get("answer", "")
        # Remove notes if present
        if "\n\n-- notes:" in predicted_sql:
            predicted_sql = predicted_sql.split("\n\n-- notes:")[0].strip()

        # Final safety hardening before VA/EX execution.
        predicted_sql = _preexecute_sql_safety_pass(predicted_sql, question)
        
        result["predicted_sql"] = predicted_sql
        result["draft_source"] = str(state.get("draft_source", "") or "").strip().lower()
        result["ir_used"] = bool(state.get("ir_used", False))
        result["ir_confidence"] = round(float(state.get("ir_confidence", 0.0) or 0.0), 3)
        result["ir_plannable"] = bool(state.get("ir_plannable", False))
        gate_reasons = state.get("ir_gate_reasons", []) or []
        result["ir_gate_reasons"] = "|".join(str(x) for x in gate_reasons if str(x).strip())
        ambiguity_signals = state.get("ambiguity_signals", []) or []
        result["ambiguity_signals"] = "|".join(str(x) for x in ambiguity_signals if str(x).strip())
        result["cannot_answer"] = bool(state.get("cannot_answer", False))
        result["cannot_answer_reason"] = str(state.get("cannot_answer_reason") or "").strip()
        result["ir_hint_passed"] = bool(state.get("ir_hint_passed", False))
        
        # Primary latency metric for thesis:
        # Warm prompt->final SQL (model receives prompt -> final SQL ready).
        # Fallback to warm inference-only if processing_time is unavailable.
        processing_time = float(state.get("processing_time", 0.0) or 0.0)
        inference_time = float(state.get("draft_inference_time", 0.0) or 0.0)
        latency_source = "unavailable"
        if processing_time > 0:
            result["latency_s"] = round(processing_time, 2)
            latency_source = "warm_prompt_to_final_sql"
        elif inference_time > 0:
            result["latency_s"] = round(inference_time, 2)
            latency_source = "warm_inference_only"
        else:
            # IR path (no LLM call) or missing telemetry -> keep 0.0
            result["latency_s"] = 0.0
            if result["ir_used"]:
                latency_source = "no_llm_ir_path"
        print(f"[LATENCY] source={latency_source}, latency_s={result['latency_s']:.2f}")

        # Track router-computed LLM fallback candidate separately from the
        # actual model used after the IR gate.
        router_model_candidate = (
            str(state.get("router_model_candidate", "") or "").strip()
            or str(state.get("target_model", "") or "").strip()
        )
        target_model = str(state.get("target_model", "") or "").strip()

        active_small_model = str((scenario_config or {}).get("small_model") or SMALL_MODEL).strip()
        active_large_model = str((scenario_config or {}).get("large_model") or LARGE_MODEL).strip()

        # Get router LLM fallback candidate by active scenario role.
        if router_model_candidate == active_small_model:
            result["initial_model_used"] = "small"
        elif router_model_candidate == active_large_model:
            result["initial_model_used"] = "large"
        elif "3b" in router_model_candidate.lower():
            result["initial_model_used"] = "small"
        elif "7b" in router_model_candidate.lower() or "8b" in router_model_candidate.lower():
            result["initial_model_used"] = "large"
        else:
            result["initial_model_used"] = router_model_candidate or "unknown"

        # Standardize final model/route label to match draft_source format.
        final_model = str(state.get("final_model", target_model) or "").strip()
        final_model_lower = final_model.lower()
        if final_model == active_small_model:
            result["final_model_used"] = "llm_small"
        elif final_model == active_large_model:
            result["final_model_used"] = "llm_large"
        elif "3b" in final_model_lower:
            result["final_model_used"] = "llm_small"
        elif "7b" in final_model_lower or "8b" in final_model_lower:
            result["final_model_used"] = "llm_large"
        elif final_model_lower == "ir_planner":
            result["final_model_used"] = "ir_planner"
        elif final_model_lower == "gate_blocked":
            result["final_model_used"] = "gate_blocked"
        elif result["initial_model_used"] == "small":
            result["final_model_used"] = "llm_small"
        elif result["initial_model_used"] == "large":
            result["final_model_used"] = "llm_large"
        else:
            result["final_model_used"] = final_model or (result["draft_source"] or "unknown")

        # Make draft_source explicit for LLM path (llm_small / llm_large),
        # with the same label standard as final_model_used.
        if result["draft_source"] == "llm":
            if result["final_model_used"] in {"llm_small", "llm_large"}:
                result["draft_source"] = result["final_model_used"]
            elif result["initial_model_used"] in {"small", "large"}:
                result["draft_source"] = f"llm_{result['initial_model_used']}"
            else:
                result["draft_source"] = "llm"

        print(f"[MODEL] LLM Candidate: {result['initial_model_used']} → Final: {result['final_model_used']}")
        print(
            f"[ROUTE] draft_source={result['draft_source'] or 'unknown'}, "
            f"ir_used={result['ir_used']}, ir_conf={result['ir_confidence']:.3f}"
        )
        
        print(f"[PREDICTED] {predicted_sql}")

    except Exception as e:
        result["error"] = f"Pipeline error: {e}"
        print(f"[ERROR] Pipeline failed: {e}")
        return result

    if result["cannot_answer"]:
        reason = result["cannot_answer_reason"] or "Cannot answer from available schema evidence"
        if expects_cannot_answer:
            result["va_success"] = True
            result["va_notes"] = "cannot_answer_expected"
            result["exec_success"] = True
            result["exec_notes"] = reason
            result["result_match"] = True
            result["ts_success"] = True
            result["ts_total_databases"] = 0
            result["ts_match_count"] = 0
            result["ts_failed_databases"] = ""
            result["match_reason"] = reason
            result["error"] = None
        else:
            result["va_success"] = False
            result["va_notes"] = "cannot_answer"
            result["exec_success"] = False
            result["exec_notes"] = reason
            result["result_match"] = False
            result["ts_match_count"] = 0
            result["ts_failed_databases"] = "\n".join(ts_db_targets)
            result["match_reason"] = reason
            result["error"] = reason
        _refresh_row_count_fields()
        print(f"[ABSTAIN] {reason}")
        return result

    if expects_cannot_answer:
        result["va_success"] = False
        result["va_notes"] = "expected_cannot_answer_but_sql_generated"
        result["exec_success"] = False
        result["exec_notes"] = "Expected cannot_answer for this benchmark case"
        result["result_match"] = False
        result["ts_success"] = False
        result["ts_total_databases"] = 0
        result["ts_match_count"] = 0
        result["ts_failed_databases"] = ""
        result["match_reason"] = "Expected cannot_answer but model generated SQL"
        result["error"] = "Expected cannot_answer but model generated SQL"
        _refresh_row_count_fields()
        print("[ABSTAIN] Expected cannot_answer, but model generated SQL")
        return result

    # Fail-closed outputs must be treated as failed evaluation cases.
    if "FINAL_GATE_BLOCKED" in (predicted_sql or "").upper():
        result["va_success"] = False
        result["va_notes"] = "FINAL_GATE_BLOCKED"
        result["exec_success"] = False
        result["exec_notes"] = "Skipped: blocked by acceptance gate"
        result["result_match"] = False
        result["ts_match_count"] = 0
        result["ts_failed_databases"] = "\n".join(ts_db_targets)
        result["match_reason"] = "Predicted SQL blocked by deterministic acceptance gate"
        result["error"] = "Predicted SQL blocked by deterministic acceptance gate"
        _refresh_row_count_fields()
        print("[VA] Invalid SQL: FINAL_GATE_BLOCKED")
        print("[EX] Skipped execution: blocked by acceptance gate")
        if ts_db_targets:
            print("[TS] Skipped: blocked by acceptance gate (counted as FAIL)")
        return result
    

    # 2a. Valid SQL (VA) check using EXPLAIN
    va_success, va_error = await check_valid_sql(mcp, predicted_sql, database=primary_db)
    result["va_success"] = va_success
    result["va_notes"] = va_error
    if va_success:
        print("[VA] Valid SQL (EXPLAIN) ✓")
    else:
        print(f"[VA] Invalid SQL (EXPLAIN): {va_error}")
        result["exec_success"] = False
        result["exec_notes"] = f"Skipped: invalid SQL (EXPLAIN): {va_error}"
        result["result_match"] = False
        result["ts_match_count"] = 0
        result["ts_failed_databases"] = "\n".join(ts_db_targets)
        result["match_reason"] = f"Invalid SQL (EXPLAIN): {va_error}"
        result["error"] = f"Predicted SQL invalid (EXPLAIN): {va_error}"
        _refresh_row_count_fields()
        print("[EX] Skipped execution: VA failed")
        if ts_db_targets:
            print("[TS] Skipped: VA failed (counted as FAIL)")
        return result

    # 2b. Execute predicted SQL (base DB)
    pred_success, pred_rows, pred_error = await execute_sql(mcp, predicted_sql, database=primary_db)
    result["exec_success"] = pred_success
    pred_base_rows_count: int | None = None
    if pred_success:
        result["exec_notes"] = "Predicted SQL executed successfully"
        pred_base_rows_count = len(pred_rows or [])
        if EXECUTION_EMPTY_RESULT_CLASSIFY:
            result["pred_empty_result"] = pred_base_rows_count == 0
            if result["pred_empty_result"]:
                result["exec_notes"] = "Predicted SQL executed successfully (empty result)"
        pred_counts_by_db[primary_db] = f"{pred_base_rows_count} rows"
        _refresh_row_count_fields()
        print(f"[EX] Predicted SQL returned {pred_base_rows_count} rows")
    else:
        result["exec_notes"] = f"Predicted SQL execution failed: {pred_error}"
        result["error"] = f"Predicted SQL execution failed: {pred_error}"
        result["match_reason"] = f"Predicted SQL execution failed: {pred_error}"
        pred_counts_by_db[primary_db] = "ERROR"
        _refresh_row_count_fields()
        print(f"[EX] Predicted SQL failed: {pred_error}")

    explicit_null_violation = False
    explicit_null_reason = ""
    if explicit_null_to_zero_required and not _sql_has_null_to_zero_handling(predicted_sql):
        explicit_null_violation = True
        explicit_null_reason = (
            "Question explicitly requires NULL->0 handling, "
            "but predicted SQL has no COALESCE/IFNULL/NVL/CASE null handling"
        )

    # 3. Execute groundtruth SQL for comparison (Execution Accuracy)
    gt_success = False
    gt_rows = None
    gt_error = None
    gt_base_rows_count: int | None = None
    if execute_groundtruth and groundtruth_sql:
        gt_success, gt_rows, gt_error = await execute_sql(mcp, groundtruth_sql, database=primary_db)
        if gt_success:
            gt_base_rows_count = len(gt_rows or [])
            if EXECUTION_EMPTY_RESULT_CLASSIFY:
                result["gt_empty_result"] = gt_base_rows_count == 0
            gt_counts_by_db[primary_db] = f"{gt_base_rows_count} rows"
            _refresh_row_count_fields()
        else:
            gt_counts_by_db[primary_db] = "ERROR"
            _refresh_row_count_fields()
            print(f"[WARN] Groundtruth SQL failed: {gt_error}")

        if pred_success and gt_success:
            # Check if either SQL has ORDER BY - if so, row order matters
            has_order_by = (
                "ORDER BY" in predicted_sql.upper() or
                "ORDER BY" in groundtruth_sql.upper()
            )

            # Debug: show ORDER BY detection
            order_mode = "ORDERED" if has_order_by else "UNORDERED"
            print(f"[EVALUATION] Comparison mode: {order_mode} (ORDER BY detected: {has_order_by})")
            print(
                f"[EVALUATION] NULL~0 mode: {null_zero_reason} "
                f"(enabled: {allow_null_zero_equivalent})"
            )

            # Compare results on base DB for EX
            is_match, reason = compare_results(
                pred_rows,
                gt_rows,
                order_sensitive=has_order_by,
                null_zero_equivalent=allow_null_zero_equivalent,
            )
            result["result_match"] = is_match and not explicit_null_violation
            if explicit_null_violation:
                result["match_reason"] = f"{reason}; {explicit_null_reason}" if reason else explicit_null_reason
            else:
                result["match_reason"] = reason
            base_count_notes = (
                f"base={primary_db}; pred: {pred_base_rows_count} rows, "
                f"groundtruth: {gt_base_rows_count} rows"
            )
            if result["result_match"]:
                result["exec_notes"] = result["match_reason"] or f"MATCH ({base_count_notes})"
            else:
                if result["match_reason"]:
                    result["exec_notes"] = f"{result['match_reason']} | {base_count_notes}"
                else:
                    result["exec_notes"] = f"MISMATCH ({base_count_notes})"

            status = "MATCH" if result["result_match"] else "MISMATCH"
            # exec_success reflects execution accuracy: SQL must run AND match
            if not result["result_match"]:
                result["exec_success"] = False
            print(f"[EVALUATION] {status} - {result['match_reason']}")
            print(f"       (pred: {pred_base_rows_count} rows, gt: {gt_base_rows_count} rows)")
        else:
            if not pred_success and gt_success:
                result["match_reason"] = f"Predicted SQL execution failed: {pred_error}"
                result["exec_notes"] = f"Predicted SQL execution failed: {pred_error}"
            elif pred_success and not gt_success:
                result["match_reason"] = f"Groundtruth SQL failed: {gt_error}"
                result["exec_notes"] = f"Groundtruth SQL failed: {gt_error}"
            else:
                result["match_reason"] = (
                    f"Predicted SQL failed: {pred_error}; Groundtruth SQL failed: {gt_error}"
                )
                result["exec_notes"] = result["match_reason"]

        # TS is evaluated hierarchically: the prediction must first match on the
        # base database. If EX fails, TS is counted as failed for this case.
        if not result["result_match"]:
            result["ts_match_count"] = 0
            result["ts_failed_databases"] = "\n".join(ts_db_targets)
            _refresh_row_count_fields()
            if ts_db_targets:
                print("[TS] Skipped: EX failed on base database (counted as FAIL)")
            else:
                print("[TS] No target TS databases configured; TS will be marked as N/A for this case.")
            return result

        # 4. Test Suite (TS): evaluate on distilled databases only
        ts_rows = []
        if ts_db_targets:
            print(f"[TS] Running cross-database checks on: {', '.join(ts_db_targets)}")
        else:
            print("[TS] No target TS databases configured; TS will be marked as N/A for this case.")
        for db_name in ts_db_targets:
            db_pred_success, db_pred_rows, db_pred_error = await execute_sql(
                mcp, predicted_sql, database=db_name
            )
            if not db_pred_success:
                pred_counts_by_db[db_name] = "ERROR"
                ts_rows.append({
                    "database": db_name,
                    "match": False,
                    "reason": f"Predicted SQL failed: {db_pred_error}",
                })
                continue
            pred_counts_by_db[db_name] = f"{len(db_pred_rows or [])} rows"

            db_gt_success, db_gt_rows, db_gt_error = await execute_sql(
                mcp, groundtruth_sql, database=db_name
            )
            if not db_gt_success:
                gt_counts_by_db[db_name] = "ERROR"
                ts_rows.append({
                    "database": db_name,
                    "match": False,
                    "reason": f"Groundtruth SQL failed: {db_gt_error}",
                })
                continue
            gt_counts_by_db[db_name] = f"{len(db_gt_rows or [])} rows"

            db_has_order_by = (
                "ORDER BY" in predicted_sql.upper() or
                "ORDER BY" in groundtruth_sql.upper()
            )
            db_is_match, db_reason = compare_results(
                db_pred_rows,
                db_gt_rows,
                order_sensitive=db_has_order_by,
                null_zero_equivalent=allow_null_zero_equivalent,
            )
            if explicit_null_violation:
                db_is_match = False
                db_reason = (
                    f"{db_reason}; explicit NULL->0 handling required"
                    if db_reason else
                    "explicit NULL->0 handling required"
                )

            ts_rows.append({
                "database": db_name,
                "match": bool(db_is_match),
                "reason": str(db_reason),
            })
        _refresh_row_count_fields()

        # TS validity uses distilled DBs only (base DB is covered by EX metric).
        result["ts_total_databases"] = len(ts_rows)
        result["ts_match_count"] = sum(1 for x in ts_rows if x.get("match"))
        result["ts_success"] = (
            result["ts_total_databases"] > 0
            and result["ts_match_count"] == result["ts_total_databases"]
        )
        result["ts_failed_databases"] = "\n".join(
            x["database"] for x in ts_rows if not x.get("match")
        )
        ts_status = "PASS" if result["ts_success"] else "FAIL"
        print(
            f"[TS] {ts_status} - {result['ts_match_count']}/{result['ts_total_databases']} "
            f"databases matched"
        )
    _refresh_row_count_fields()
    return result


async def warmup_models(app, paths: list[str]):
    """
    Warmup models by calling Ollama API directly.
    This guarantees model loading regardless of pipeline success.
    Eliminates cold start bias in latency measurements.
    """
    await warmup_models_selected(app, paths, resolve_warmup_targets("all"))


def resolve_warmup_targets(warmup_models_spec: str | None) -> list[tuple[str, str]]:
    """
    Resolve CLI warmup target spec into ordered `(model_name, description)` tuples.

    Supported tokens:
      - all
      - router
      - small / 3b / simple
      - large / 7b / complex

    Unknown tokens raise ValueError so the caller gets a clear CLI failure.
    """
    available_targets = {
        "router": (ROUTING_MODEL, "Router"),
        "small": (SMALL_MODEL, "Simple SQL"),
        "3b": (SMALL_MODEL, "Simple SQL"),
        "simple": (SMALL_MODEL, "Simple SQL"),
        "large": (LARGE_MODEL, "Complex SQL"),
        "7b": (LARGE_MODEL, "Complex SQL"),
        "complex": (LARGE_MODEL, "Complex SQL"),
    }
    default_order = ["router", "small", "large"]

    spec = str(warmup_models_spec or "all").strip().lower()
    if not spec or spec == "all":
        return [available_targets[key] for key in default_order]

    requested_tokens = [tok.strip().lower() for tok in spec.split(",") if tok.strip()]
    if not requested_tokens:
        return [available_targets[key] for key in default_order]

    resolved: list[tuple[str, str]] = []
    seen_models: set[str] = set()
    for token in requested_tokens:
        if token == "all":
            for key in default_order:
                model_name, desc = available_targets[key]
                if model_name not in seen_models:
                    resolved.append((model_name, desc))
                    seen_models.add(model_name)
            continue
        if token not in available_targets:
            valid_tokens = "all, router, small, 3b, simple, large, 7b, complex"
            raise ValueError(
                f"Unknown warmup target '{token}'. Valid values: {valid_tokens}"
            )
        model_name, desc = available_targets[token]
        if model_name in seen_models:
            continue
        resolved.append((model_name, desc))
        seen_models.add(model_name)

    return resolved


async def warmup_models_selected(
    app,
    paths: list[str],
    warmup_targets: list[tuple[str, str]],
):
    """
    Warmup selected models by calling Ollama API directly.
    """
    if not warmup_targets:
        print("[STATUS] Warmup skipped - no warmup targets selected.")
        return

    print("\n" + "=" * 60)
    print("WARMUP PHASE - Loading models into memory...")
    print("=" * 60)
    print(
        "[STATUS] Warmup targets: "
        + ", ".join(f"{desc}: {model}" for model, desc in warmup_targets)
    )

    warmup_prompt = "Say 'ready' in one word."

    try:
        from . import agents
    except ImportError:
        agents = None
    async with httpx.AsyncClient(timeout=120.0) as client:
        now = time.time()
        for model, desc in warmup_targets:
            print(f"[STATUS] Loading {desc}: {model}")
            try:
                response = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": model,
                        "prompt": warmup_prompt,
                        "stream": False,
                        "options": {"num_predict": 5},
                        "keep_alive": OLLAMA_KEEP_ALIVE,
                    }
                )
                if response.status_code == 200:
                    print(f"  ✓ {desc} loaded successfully")
                    if agents is not None:
                        agents._last_model_warmup[model] = now
                else:
                    print(f"  ⚠ {desc} returned status {response.status_code}")
            except Exception as e:
                print(f"  ⚠ Failed to load {desc}: {e}")

    print("=" * 60)
    print("Warmup complete - selected models are now in memory")
    print("=" * 60 + "\n")


async def run_batch_eval(
    testset_path: str,
    output_path: str | None = None,
    paths: list[str] | None = None,
    scenario: str = "P1",
    execute_groundtruth: bool = True,
    skip_warmup: bool = False,
    warmup_models_spec: str | None = None,
    difficulty_filters: dict[str, set[int] | None] | None = None,
    null_zero_mode: str = "auto",
    ts_databases: list[str] | None = None,
    base_database: str | None = None,
):
    """
    Main batch evaluation function.

    Args:
        testset_path: Path to test set YAML file
        output_path: Path to output file (.xlsx for Excel, .csv for CSV)
        paths: JSONL paths for context
        execute_groundtruth: Whether to execute groundtruth SQL for comparison
        skip_warmup: Skip warmup phase (not recommended)
        warmup_models_spec: Comma-separated warmup targets. Default warms all.
        difficulty_filters: Dict mapping difficulty to ID filter (e.g., {'easy': {4, 5, 6, 7}})
                          None value means all IDs for that difficulty.
                          Example: {'easy': {4, 5, 6, 7}, 'hard': None}
        ts_databases: Ordered list of DB names for TS evaluation.
        base_database: Database used for VA/EX only.
    """
    scenario_name, scenario_config = resolve_scenario_config(scenario)
    active_models = apply_scenario_models(scenario_config)
    scenario_config.update(active_models)
    print(f"Scenario: {scenario_name} -> {scenario_config}")

    # Load test cases
    all_cases = load_testset(testset_path)
    print(f"Loaded {len(all_cases)} test cases from {testset_path}")

    # Filter by difficulty and optional ID range if specified
    if difficulty_filters:
        filtered_cases = []
        for case in all_cases:
            difficulty = case.get("difficulty", "").lower()
            if difficulty in difficulty_filters:
                id_filter = difficulty_filters[difficulty]
                # No ID filter specified - include all cases for this difficulty
                if id_filter is None:
                    filtered_cases.append(case)
                else:
                    # Check if case ID is in the filter set
                    case_id = case.get("id", "")
                    # Extract numeric ID from strings like "easy1", "E1", etc.
                    import re
                    id_match = re.search(r'\d+', str(case_id))
                    if id_match:
                        numeric_id = int(id_match.group())
                        if numeric_id in id_filter:
                            filtered_cases.append(case)

        # Build description for logging
        filter_desc = []
        for diff, id_set in difficulty_filters.items():
            if id_set is None:
                filter_desc.append(diff)
            else:
                # Format the ID set as ranges for cleaner display
                sorted_ids = sorted(id_set)
                ranges = []
                start = sorted_ids[0]
                prev = start
                for curr in sorted_ids[1:]:
                    if curr == prev + 1:
                        prev = curr
                    else:
                        if start == prev:
                            ranges.append(f"{start}")
                        else:
                            ranges.append(f"{start}-{prev}")
                        start = curr
                        prev = curr
                        # Add final range
                        if start == prev:
                            ranges.append(f"{start}")
                        else:
                            ranges.append(f"{start}-{prev}")
                filter_desc.append(f"{diff}({','.join(ranges)})")

        cases = filtered_cases
        print(f"Filtered to {len(cases)} test cases with difficulty: {', '.join(filter_desc)}")
    else:
        cases = all_cases
        print(f"No difficulty filter - running all {len(cases)} test cases")
    
    # Default paths for JSONL - resolve from config like CLI does
    if paths is None or len(paths) == 0:
        paths = resolve_jsonl_paths(SPARSE_METADATA_JSONL)
        if not paths:
            print("[WARN] No sparse JSONL paths found - BM25 retrieval may not work!")
    
    ts_db_list: list[str] = []
    for db in (ts_databases or []):
        db_name = str(db or "").strip()
        if db_name and db_name not in ts_db_list:
            ts_db_list.append(db_name)
    if not ts_db_list:
        ts_db_list = infer_ts_databases(
            None,
            base_database=base_database,
            testset_path=testset_path,
        )
    if not ts_db_list:
        raise ValueError(
            "TS databases are not set and could not be inferred. "
            "Provide --ts-databases, --base-database, set T2SQL_TS_DATABASES, "
            "or use a testset filename that contains the base dataset name."
        )

    base_db = infer_base_database(base_database, ts_db_list)
    ts_db_targets = [db for db in ts_db_list if db != base_db]
    print(f"Base database (VA/EX): {base_db}")
    print(f"TS databases (excluding base): {ts_db_targets}")

    # Build app
    app = build_app()
    
    # Warmup phase to eliminate cold start bias
    if not skip_warmup:
        warmup_targets = resolve_warmup_targets(warmup_models_spec)
        await warmup_models_selected(app, paths, warmup_targets)
    
    # Setup MCP connection
    transport = StreamableHttpTransport(MCP_BASE)
    
    results = []
    
    async with Client(transport=transport) as mcp:
        for i, case in enumerate(cases):
            print(f"\n[Progress: {i+1}/{len(cases)}]")
            result = await run_single_case(
                app, mcp, paths, case,
                scenario=scenario_name,
                scenario_config=scenario_config,
                execute_groundtruth=execute_groundtruth,
                null_zero_mode=null_zero_mode,
                ts_databases=ts_db_targets,
                base_database=base_db,
            )
            results.append(result)
    

    # Calculate metrics
    total = len(results)
    va_success = sum(1 for r in results if r["va_success"])
    exec_success = sum(1 for r in results if r["exec_success"])
    result_match = sum(1 for r in results if r["result_match"])
    ts_evaluable_cases = [r for r in results if int(r.get("ts_total_databases", 0) or 0) > 0]
    ts_total_cases = len(ts_evaluable_cases)
    ts_success_cases = sum(1 for r in ts_evaluable_cases if bool(r.get("ts_success", False)))
    ir_used_count = sum(1 for r in results if bool(r.get("ir_used", False)))
    ir_hint_passed_count = sum(1 for r in results if bool(r.get("ir_hint_passed", False)))
    llm_fallback_count = total - ir_used_count
    final_small_count = sum(
        1 for r in results if str(r.get("final_model_used", "")).strip().lower() == "llm_small"
    )
    final_large_count = sum(
        1 for r in results if str(r.get("final_model_used", "")).strip().lower() == "llm_large"
    )

    # Metrics by difficulty
    by_difficulty = {}
    for r in results:
        d = r["difficulty"]
        if d not in by_difficulty:
            by_difficulty[d] = {
                "total": 0,
                "va_success": 0,
                "exec_success": 0,
                "result_match": 0,
                "ts_cases": 0,
                "ts_success": 0,
            }
        by_difficulty[d]["total"] += 1
        if r["va_success"]:
            by_difficulty[d]["va_success"] += 1
        if r["exec_success"]:
            by_difficulty[d]["exec_success"] += 1
        if r["result_match"]:
            by_difficulty[d]["result_match"] += 1
        if int(r.get("ts_total_databases", 0) or 0) > 0:
            by_difficulty[d]["ts_cases"] += 1
            if bool(r.get("ts_success", False)):
                by_difficulty[d]["ts_success"] += 1


    # Print summary (custom format)
    print("\n" + "="*120)
    print("EVALUATION SUMMARY")
    print("="*120)
    print(f"Total cases: {total}")
    print(f"Valid SQL (VA): {va_success}/{total} ({100*va_success/total:.1f}%)")
    print(f"Execution Summary (EX): {result_match}/{total} ({100*result_match/total:.1f}%)")
    if ts_total_cases > 0:
        print(f"Test Suite (TS): {ts_success_cases}/{ts_total_cases} ({100*ts_success_cases/ts_total_cases:.1f}%)")
    else:
        print("Test Suite (TS): N/A")
    print(
        f"Route Split: LLM fallback={llm_fallback_count}/{total} ({100*llm_fallback_count/total:.1f}%), "
        f"IR hint passed={ir_hint_passed_count}/{llm_fallback_count}"
    )
    print(
        f"Model Usage (Final): small={final_small_count}/{total} ({100*final_small_count/total:.1f}%), "
        f"large={final_large_count}/{total} ({100*final_large_count/total:.1f}%)"
    )

    print("\nBy Difficulty:")
    for d, stats in sorted(by_difficulty.items()):
        t = stats["total"]
        va = stats["va_success"]
        rm = stats["result_match"]
        ts_cases = stats.get("ts_cases", 0)
        ts_ok = stats.get("ts_success", 0)
        ts_str = f", TS={ts_ok}/{ts_cases} ({100*ts_ok/ts_cases:.1f}%)" if ts_cases else ", TS=N/A"
        print(f"  {d}: VA={va}/{t} ({100*va/t:.1f}%), Execution Summary ={rm}/{t} ({100*rm/t:.1f}%){ts_str}")

    print("\n\n")

    # Average latency (in seconds)
    latencies = [r["latency_s"] for r in results if r["latency_s"] > 0]
    if latencies:
        print(f"Average Latency: {sum(latencies)/len(latencies):.2f}s\n")
    
    # Save to Excel or CSV

    if output_path:
        fieldnames = [
            # Keep key identifiers first to reduce horizontal scrolling.
            "id", "scenario", "difficulty", "expected_behavior",
            "question",
            "groundtruth_sql", "predicted_sql",
            "draft_source", "final_model_used", "ir_confidence", "ir_gate_reasons",
            "ambiguity_signals", "cannot_answer", "cannot_answer_reason",
            "va_success", "va_notes",
            "exec_success", "exec_notes", "pred_empty_result", "gt_empty_result",
            "ts_success", "ts_failed_databases",
            "pred_row_count", "gt_row_count",
            "latency_s"
        ]
        
        if output_path.endswith('.xlsx') and HAS_EXCEL:
            # Save as Excel with formatting
            save_to_excel(
                results,
                fieldnames,
                output_path,
                by_difficulty,
                total,
                exec_success,
                result_match,
                latencies,
                ts_total_cases=ts_total_cases,
                ts_success_cases=ts_success_cases,
            )
        else:
            # Fallback to CSV
            if output_path.endswith('.xlsx'):
                print("[WARN] openpyxl not installed, falling back to CSV")
                output_path = output_path.replace('.xlsx', '.csv')
            
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
        print(f"\nResults saved to: {output_path}")
    
    return results


def preprocess_argv(argv: list[str]) -> list[str]:
    """
    Preprocess argv to convert patterns like --easy4, --easy4-7, --hard1,3,5
    into --easy 4, --easy 4-7, --hard 1,3,5
    """
    import re
    processed = []
    for arg in argv:
        # Match patterns: --easy<number>, --medium<number>, --hard<number>
        # where number can be: 4, 4-7, 1,3,5
        match = re.match(r'--(easy|medium|hard)(.+)', arg)
        if match:
            difficulty = match.group(1)
            value = match.group(2)
            processed.extend([f"--{difficulty}", value])
        else:
            processed.append(arg)
    return processed

def main():
    # Preprocess argv to handle --easy4, --easy4-7 patterns
    import sys
    sys.argv = preprocess_argv(sys.argv)

    parser = argparse.ArgumentParser(
        description="Batch evaluation for Text-to-SQL pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all questions
  python -m t2sql.testing_evaluation --testset qa_dataset_devset.yml --output results.xlsx

  # Test only hard questions
  python -m t2sql.testing_evaluation --testset qa_dataset_devset.yml --output results.xlsx --hard

  # Test easy and medium questions
  python -m t2sql.testing_evaluation --testset qa_dataset_devset.yml --output results.xlsx --easy --medium

  # Test only easy question #4
  python -m t2sql.testing_evaluation --testset qa_dataset_devset.yml --output results.xlsx --easy4

  # Test easy questions #4-7
  python -m t2sql.testing_evaluation --testset qa_dataset_devset.yml --output results.xlsx --easy4-7

  # Test specific questions: easy #1, #3, #5
  python -m t2sql.testing_evaluation --testset qa_dataset_devset.yml --output results.xlsx --easy1,3,5

  # Run EX/VA on base DB and TS on distilled DBs
  python -m t2sql.testing_evaluation --testset qa_dataset_devset.yml --output results.xlsx --base-database mydb --ts-databases mydb,mydb_2,mydb_3

  # Recommended (explicit): EX/VA on base DB, TS only on shifted DBs
  python -m t2sql.testing_evaluation --testset qa_dataset_devset.yml --output results.xlsx --base-database mydb --ts-databases mydb_2,mydb_3
        """
    )
    parser.add_argument(
        "--testset", "-t",
        required=True,
        help="Path to test set YAML file"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to output file (.xlsx for Excel, .csv for CSV)"
    )
    parser.add_argument(
        "--scenario",
        choices=["P1", "P2", "P3", "p1", "p2", "p3"],
        default="P1",
        help=(
            "Evaluation scenario preset: "
            "P1=qwen2.5, P2=qwen2.5-coder, P3=llama."
        ),
    )
    parser.add_argument(
        "--no-groundtruth-exec",
        action="store_true",
        help="Skip executing groundtruth SQL (only check if predicted SQL runs)"
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip warmup phase (not recommended - will cause cold start bias)"
    )
    parser.add_argument(
        "--warmup-models",
        default="all",
        help=(
            "Comma-separated warmup targets. "
            "Supported: all, router, small, 3b, simple, large, 7b, complex. "
            "Examples: --warmup-models small or --warmup-models router,small"
        ),
    )
    parser.add_argument(
        "--null-zero-equivalent",
        action="store_true",
        help="Backward-compatible alias for --null-zero-mode on"
    )
    parser.add_argument(
        "--null-zero-mode",
        choices=["off", "on", "auto"],
        default="auto",
        help=(
            "NULL vs 0 comparison mode: off=strict, on=always treat NULL~0 as equal, "
            "auto=allow NULL~0 unless question explicitly asks NULL->0 handling"
        )
    )
    parser.add_argument(
        "--ts-databases",
        default=None,
        help=(
            "Comma-separated database list for Test Suite evaluation. "
            "Base DB is excluded automatically from TS metrics. "
            "If omitted, uses T2SQL_TS_DATABASES or infers from --base-database / testset name."
        ),
    )
    parser.add_argument(
        "--base-database",
        default=None,
        help=(
            "Database for VA/EX checks. Defaults to T2SQL_BASE_DATABASE or "
            "is inferred from the TS database list."
        ),
    )
    # Difficulty filter arguments (supports: --easy, --easy4, --easy4-7, etc.)
    parser.add_argument(
        "--easy",
        nargs="?",
        const="",
        help="Only test easy questions. Optionally specify ID/range: --easy4 or --easy4-7"
    )
    parser.add_argument(
        "--medium",
        nargs="?",
        const="",
        help="Only test medium questions. Optionally specify ID/range: --medium4 or --medium4-7"
    )
    parser.add_argument(
        "--hard",
        nargs="?",
        const="",
        help="Only test hard questions. Optionally specify ID/range: --hard4 or --hard4-7"
    )

    args = parser.parse_args()
    # Backward-compatible CLI alias for existing commands
    null_zero_mode = "on" if args.null_zero_equivalent else args.null_zero_mode
    base_database = str(args.base_database or "").strip() or None
    ts_databases = infer_ts_databases(
        [x.strip() for x in str(args.ts_databases or "").split(",") if x.strip()],
        base_database=base_database,
        testset_path=args.testset,
    )

    # Build difficulty filter dict with optional ID ranges
    difficulty_filters = {}
    for difficulty_name, arg_value in [("easy", args.easy), ("medium", args.medium), ("hard", args.hard)]:
        if arg_value is not None:  # Flag was used (with or without value)
            _, id_filter = parse_difficulty_filter(arg_value)
            difficulty_filters[difficulty_name] = id_filter

    # Generate default output filename if not specified
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Default to Excel if available
        ext = ".xlsx" if HAS_EXCEL else ".csv"
        # Add difficulty suffix if filtering
        if difficulty_filters:
            parts = []
            for diff, id_filter in difficulty_filters.items():
                if id_filter is None:
                    parts.append(diff)
                else:
                    sorted_ids = sorted(id_filter)
                    if len(sorted_ids) == 1:
                        parts.append(f"{diff}{sorted_ids[0]}")
                    else:
                        parts.append(f"{diff}{sorted_ids[0]}-{sorted_ids[-1]}")
            suffix = f"_{'-'.join(parts)}"
        else:
            suffix = ""
        args.output = f"eval_results{suffix}_{timestamp}{ext}"

    started_at = _jakarta_now()
    started_perf = time.perf_counter()
    run_status = "SUCCESS"
    _print_run_timestamp_block("RUN START", started_at)

    try:
        asyncio.run(run_batch_eval(
            testset_path=args.testset,
            output_path=args.output,
            execute_groundtruth=not args.no_groundtruth_exec,
            scenario=str(args.scenario or "P1").upper(),
            skip_warmup=args.skip_warmup,
            warmup_models_spec=args.warmup_models,
            difficulty_filters=difficulty_filters if difficulty_filters else None,
            null_zero_mode=null_zero_mode,
            ts_databases=ts_databases,
            base_database=base_database,
        ))
    except KeyboardInterrupt:
        run_status = "INTERRUPTED"
        raise
    except Exception:
        run_status = "FAILED"
        raise
    finally:
        finished_at = _jakarta_now()
        elapsed_seconds = time.perf_counter() - started_perf
        _print_run_timestamp_block(
            "RUN END",
            finished_at,
            status=run_status,
            elapsed_seconds=elapsed_seconds,
        )

if __name__ == "__main__":
    main()
