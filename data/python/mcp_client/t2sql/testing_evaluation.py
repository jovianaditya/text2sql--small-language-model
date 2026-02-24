# ======================================================================
# ========================= testing_evaluation .py =====================
# ======================================================================
# Purpose:
# - Batch evaluation script for Text-to-SQL pipeline using MCP client.
# - Loads test cases from YAML file.
# - Executes predicted SQL and groundtruth SQL against MCP database.
# - Currently only compares results for Execution Accuracy (EX) and Valid SQL (VA) checks.

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
from datetime import datetime
import yaml
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
try:
    import openpyxl
    from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
    from openpyxl.utils.dataframe import dataframe_to_rows
    HAS_EXCEL = True
except ImportError:
    HAS_EXCEL = False
os.environ.setdefault("OLLAMA_URL", "http://ollama:11434")
import httpx
# ======================================================================
# ======================== LOCAL MODULE IMPORT =========================
# ======================================================================
from .config import MCP_BASE, CHROMA_DIR, COLLECTION_NAME, JSON_INPUT, OLLAMA_URL, ROUTING_MODEL, SMALL_MODEL, LARGE_MODEL, OLLAMA_KEEP_ALIVE
from .state import make_state
from .graph import build_app
from .mcp_execution import query_db_full, _mcp_call  # Pakai FULL query tanpa LIMIT dan _mcp_call helper
from .helpers_core import _preexecute_sql_safety_pass
from .utils import resolve_jsonl_paths

def load_testset(yaml_path: str) -> list[dict]:
    """Load test cases dari file YAML."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("cases", [])

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


def save_to_excel(results: list, fieldnames: list, output_path: str, 
                  by_difficulty: dict, total: int, exec_success: int, 
                  result_match: int, latencies: list):
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
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    match_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    mismatch_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
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
        ("Average Latency (s)", f"{avg_latency_sec:.2f}" if latencies else "N/A", ""),
    ]
    
    for row_idx, row_data in enumerate(summary_data, start=3):
        for col_idx, value in enumerate(row_data, start=1):
            cell = ws_summary.cell(row=row_idx, column=col_idx, value=value)
            cell.border = border
            if row_idx == 3:
                cell.font = header_font
                cell.fill = header_fill
    
    # By difficulty
    ws_summary["A10"] = "BY DIFFICULTY"
    ws_summary["A10"].font = Font(bold=True, size=12)
    
    diff_header = [
        "Difficulty",
        "Total",
        "EX Success",
        "Percentage VA Success (%)",
        "VA Success",
        "Percentage EX Success (%)",
        "Avg Latency (s)"
    ]
    for col_idx, h in enumerate(diff_header, start=1):
        cell = ws_summary.cell(row=11, column=col_idx, value=h)
        cell.font = header_font
        cell.fill = header_fill
        cell.border = border

    for row_idx, (d, stats) in enumerate(sorted(by_difficulty.items()), start=12):
        t = stats["total"]
        es = stats["exec_success"]
        va = stats.get("va_success", 0)
        # Calculate avg latency for this difficulty
        diff_latencies = latency_by_diff.get(d, [])
        avg_lat = f"{sum(diff_latencies)/len(diff_latencies):.2f}" if diff_latencies else "N/A"
        pct_va = f"{100*va/t:.1f}%" if t else "N/A"
        pct_ex = f"{100*es/t:.1f}%" if t else "N/A"
        row_data = [d, t, es, pct_va, va, pct_ex, avg_lat]
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
    
    # Data rows
    for row_idx, result in enumerate(results, start=2):
        for col_idx, field in enumerate(fieldnames, start=1):
            value = result.get(field, "")
            cell = ws_detail.cell(row=row_idx, column=col_idx, value=value)
            cell.border = border
            cell.alignment = Alignment(wrap_text=True, vertical='top')
            
            # Highlight result_match column
            if field == "result_match":
                if value:
                    cell.fill = match_fill
                else:
                    cell.fill = mismatch_fill
    
    # Adjust column widths
    col_widths = {
        "id": 8, "difficulty": 10,
        "question": 50,
        "groundtruth_sql": 40, "predicted_sql": 40,
        "draft_source": 14, "ir_used": 10, "ir_confidence": 12, "ir_plannable": 12, "ir_gate_reasons": 28,
        "initial_model_used": 12, "final_model_used": 12,  # Model tracking
        "va_success": 12, "va_error": 30,
        "exec_success": 12, "result_match": 12, "match_reason": 25,
        "pred_row_count": 12, "gt_row_count": 12,
        "latency_s": 12, "error": 30
    }
    for col_idx, field in enumerate(fieldnames, start=1):
        ws_detail.column_dimensions[openpyxl.utils.get_column_letter(col_idx)].width = col_widths.get(field, 15)
    
    # Freeze header row
    ws_detail.freeze_panes = "A2"
    
    # Save
    wb.save(output_path)
    print(f"\nResults saved to Excel: {output_path}")

async def execute_sql(mcp: Client, sql: str) -> tuple[bool, list | None, str | None]:
    """
    Execute SQL via MCP (FULL mode - tanpa LIMIT) and return (success, rows, error).
    """
    sql_clean = sql.rstrip(";").strip()
    try:
        payload = await query_db_full(sql_clean, mcp=mcp)
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
async def check_valid_sql(mcp: Client, sql: str) -> tuple[bool, str | None]:
    """
    Run EXPLAIN <sql> via MCP tool to check validity. Return (is_valid, error_message_if_any)
    """
    sql_clean = sql.rstrip(";").strip()
    try:
        payload = await _mcp_call(mcp, "query_explain", {"sql": sql_clean})
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
    execute_groundtruth: bool = True,
    null_zero_mode: str = "auto",
) -> dict:
    """
    Run a single test case and return evaluation result.
    """
    case_id = case.get("id", "unknown")
    question = case.get("question", "")
    groundtruth_sql = case.get("groundtruth_sql", "").strip()
    difficulty = case.get("difficulty", "unknown")
    allow_null_zero_equivalent, explicit_null_to_zero_required, null_zero_reason = (
        _resolve_null_zero_equivalent(question, null_zero_mode)
    )
    
    print(f"\n{'='*60}")
    print(f"[{case_id}] ({difficulty}) {question[:80]}...")
    print(f"{'='*60}")
    
    result = {
        "id": case_id,
        "difficulty": difficulty,
        "question": question,
        "groundtruth_sql": groundtruth_sql,
        "predicted_sql": "",
        "draft_source": "",
        "ir_used": False,
        "ir_confidence": 0.0,
        "ir_plannable": False,
        "ir_gate_reasons": "",
        "initial_model_used": "",  # Initial model chosen by router (3b/7b)
        "final_model_used": "",    # Final model used (may differ if regeneration occurred)
        "va_success": False,       # Valid SQL (EXPLAIN)
        "va_error": None,          # Error message if not valid
        "exec_success": False,
        "result_match": False,     # Execution Accuracy (hasil sama)
        "match_reason": "",        # Alasan match/mismatch
        "pred_row_count": 0,
        "gt_row_count": 0,
        "latency_s": 0,            # Latency dalam detik
        "error": None,
    }
    
    # 1. Run pipeline
    t0 = time.perf_counter()
    try:
        init_state = make_state(paths, question)
        state = app.invoke(init_state)
        
        # Get predicted SQL
        predicted_sql = state.get("final_sql") or state.get("answer", "")
        # Remove notes if present
        if "\n\n-- notes:" in predicted_sql:
            predicted_sql = predicted_sql.split("\n\n-- notes:")[0].strip()

        # Final safety hardening before VA/EX execution.
        predicted_sql = _preexecute_sql_safety_pass(predicted_sql, question)
        
        result["predicted_sql"] = predicted_sql
        result["draft_source"] = str(state.get("draft_source", "") or "")
        result["ir_used"] = bool(state.get("ir_used", False))
        result["ir_confidence"] = round(float(state.get("ir_confidence", 0.0) or 0.0), 3)
        result["ir_plannable"] = bool(state.get("ir_plannable", False))
        gate_reasons = state.get("ir_gate_reasons", []) or []
        result["ir_gate_reasons"] = "|".join(str(x) for x in gate_reasons if str(x).strip())
        
        # Use inference time from agent (excludes cold start/model loading)
        # Fallback to total time if not available
        inference_time = state.get("draft_inference_time", 0)
        if inference_time > 0:
            result["latency_s"] = round(inference_time, 2)
        else:
            result["latency_s"] = round((time.perf_counter() - t0), 2)

        # Get initial model used (extract short name: 3b/7b)
        target_model = state.get("target_model", "")
        if "3b" in target_model.lower():
            result["initial_model_used"] = "3b"
        elif "7b" in target_model.lower():
            result["initial_model_used"] = "7b"
        else:
            result["initial_model_used"] = target_model or "unknown"

        # Get final model used (may differ if regeneration occurred)
        final_model = state.get("final_model", target_model)
        if "3b" in final_model.lower():
            result["final_model_used"] = "3b"
        elif "7b" in final_model.lower():
            result["final_model_used"] = "7b"
        else:
            result["final_model_used"] = final_model or result["initial_model_used"]

        print(f"[MODEL] Initial: {result['initial_model_used']} → Final: {result['final_model_used']}")
        print(
            f"[ROUTE] draft_source={result['draft_source'] or 'unknown'}, "
            f"ir_used={result['ir_used']}, ir_conf={result['ir_confidence']:.3f}"
        )
        
        print(f"[PREDICTED] {predicted_sql}")
        
    except Exception as e:
        result["error"] = f"Pipeline error: {e}"
        print(f"[ERROR] Pipeline failed: {e}")
        return result
    

    # 2a. Valid SQL (VA) check using EXPLAIN
    va_success, va_error = await check_valid_sql(mcp, predicted_sql)
    result["va_success"] = va_success
    result["va_error"] = va_error
    if va_success:
        print("[VA] Valid SQL (EXPLAIN) ✓")
    else:
        print(f"[VA] Invalid SQL (EXPLAIN): {va_error}")

    # 2b. Execute predicted SQL
    pred_success, pred_rows, pred_error = await execute_sql(mcp, predicted_sql)
    result["exec_success"] = pred_success
    
    if not pred_success:
        result["error"] = f"Predicted SQL execution failed: {pred_error}"
        print(f"[EX] Predicted SQL failed: {pred_error}")
        return result
    
    result["pred_row_count"] = len(pred_rows or [])
    print(f"[EX] Predicted SQL returned {result['pred_row_count']} rows")
    
    # 3. Execute groundtruth SQL for comparison (Execution Accuracy)
    if execute_groundtruth and groundtruth_sql:
        gt_success, gt_rows, gt_error = await execute_sql(mcp, groundtruth_sql)
        
        if gt_success:
            result["gt_row_count"] = len(gt_rows or [])

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

            explicit_null_violation = False
            explicit_null_reason = ""
            if explicit_null_to_zero_required and not _sql_has_null_to_zero_handling(predicted_sql):
                explicit_null_violation = True
                explicit_null_reason = (
                    "Question explicitly requires NULL->0 handling, "
                    "but predicted SQL has no COALESCE/IFNULL/NVL/CASE null handling"
                )

            # Compare results
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
            
            status = "MATCH" if result["result_match"] else "MISMATCH"
            print(f"[EVALUATION] {status} - {result['match_reason']}")
            print(f"       (pred: {result['pred_row_count']} rows, gt: {result['gt_row_count']} rows)")
        else:
            result["match_reason"] = f"Groundtruth SQL failed: {gt_error}"
            print(f"[WARN] Groundtruth SQL failed: {gt_error}")
    
    return result


async def warmup_models(app, paths: list[str]):
    """
    Warmup models by calling Ollama API directly.
    This guarantees model loading regardless of pipeline success.
    Eliminates cold start bias in latency measurements.
    """
    print("\n" + "=" * 60)
    print("WARMUP PHASE - Loading models into memory...")
    print("=" * 60)
    
    # Models to warm up (directly call Ollama, bypass pipeline)
    models_to_warmup = [
        (ROUTING_MODEL, "Router (1.5b)"),
        (SMALL_MODEL, "Simple SQL (3b)"),
        (LARGE_MODEL, "Complex SQL (7b)"),
    ]
    
    # Simple prompt that any model can answer
    warmup_prompt = "Say 'ready' in one word."
    
    import sys
    # Set agents._last_model_warmup after warmup
    try:
        from . import agents
    except ImportError:
        agents = None
    async with httpx.AsyncClient(timeout=120.0) as client:
        now = time.time()
        for model, desc in models_to_warmup:
            print(f"[Warmup] Loading {desc}: {model}")
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
                    # Update agents._last_model_warmup so agent logic can skip warmup
                    if agents is not None:
                        agents._last_model_warmup[model] = now
                else:
                    print(f"  ⚠ {desc} returned status {response.status_code}")
            except Exception as e:
                print(f"  ⚠ Failed to load {desc}: {e}")
    
    print("=" * 60)
    print("Warmup complete - all models are now in memory")
    print("=" * 60 + "\n")


async def run_batch_eval(
    testset_path: str,
    output_path: str | None = None,
    paths: list[str] | None = None,
    execute_groundtruth: bool = True,
    skip_warmup: bool = False,
    difficulty_filters: dict[str, set[int] | None] | None = None,
    null_zero_mode: str = "auto",
):
    """
    Main batch evaluation function.

    Args:
        testset_path: Path to test set YAML file
        output_path: Path to output file (.xlsx for Excel, .csv for CSV)
        paths: JSONL paths for context
        execute_groundtruth: Whether to execute groundtruth SQL for comparison
        skip_warmup: Skip warmup phase (not recommended)
        difficulty_filters: Dict mapping difficulty to ID filter (e.g., {'easy': {4, 5, 6, 7}})
                          None value means all IDs for that difficulty.
                          Example: {'easy': {4, 5, 6, 7}, 'hard': None}
    """
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
        paths = resolve_jsonl_paths(JSON_INPUT)
        if paths:
            print(f"Using JSONL paths: {paths}")
        else:
            print("[WARN] No JSONL paths found - sparse search may not work!")
    
    # Build app
    app = build_app()
    
    # Warmup phase to eliminate cold start bias
    if not skip_warmup:
        await warmup_models(app, paths)
    
    # Setup MCP connection
    transport = StreamableHttpTransport(MCP_BASE)
    
    results = []
    
    async with Client(transport=transport) as mcp:
        for i, case in enumerate(cases):
            print(f"\n[Progress: {i+1}/{len(cases)}]")
            result = await run_single_case(
                app, mcp, paths, case, 
                execute_groundtruth=execute_groundtruth,
                null_zero_mode=null_zero_mode,
            )
            results.append(result)
    

    # Calculate metrics
    total = len(results)
    va_success = sum(1 for r in results if r["va_success"])
    exec_success = sum(1 for r in results if r["exec_success"])
    result_match = sum(1 for r in results if r["result_match"])
    ir_used_count = sum(1 for r in results if bool(r.get("ir_used", False)))
    llm_fallback_count = total - ir_used_count

    # Metrics by difficulty
    by_difficulty = {}
    for r in results:
        d = r["difficulty"]
        if d not in by_difficulty:
            by_difficulty[d] = {"total": 0, "va_success": 0, "exec_success": 0, "result_match": 0}
        by_difficulty[d]["total"] += 1
        if r["va_success"]:
            by_difficulty[d]["va_success"] += 1
        if r["exec_success"]:
            by_difficulty[d]["exec_success"] += 1
        if r["result_match"]:
            by_difficulty[d]["result_match"] += 1


    # Print summary (custom format)
    print("\n" + "="*120)
    print("EVALUATION SUMMARY")
    print("="*120)
    print(f"Total cases: {total}")
    print(f"Valid SQL (VA): {va_success}/{total} ({100*va_success/total:.1f}%)")
    print(f"Execution Summary (EX): {result_match}/{total} ({100*result_match/total:.1f}%)")
    print(
        f"Route Split: IR={ir_used_count}/{total} ({100*ir_used_count/total:.1f}%), "
        f"LLM fallback={llm_fallback_count}/{total} ({100*llm_fallback_count/total:.1f}%)"
    )

    print("\nBy Difficulty:")
    for d, stats in sorted(by_difficulty.items()):
        t = stats["total"]
        va = stats["va_success"]
        rm = stats["result_match"]
        print(f"  {d}: VA={va}/{t} ({100*va/t:.1f}%), Execution Summary ={rm}/{t} ({100*rm/t:.1f}%)")

    print("\n\n")

    # Average latency (in seconds)
    latencies = [r["latency_s"] for r in results if r["latency_s"] > 0]
    if latencies:
        print(f"Average Latency: {sum(latencies)/len(latencies):.2f}s\n")
    
    # Save to Excel or CSV

    if output_path:
        fieldnames = [
            # Put final result verdict early to avoid horizontal scrolling.
            "id", "difficulty", "result_match", "match_reason",
            "question",
            "groundtruth_sql", "predicted_sql",
            "draft_source", "ir_used", "ir_confidence", "ir_plannable", "ir_gate_reasons",
            "initial_model_used", "final_model_used",  # Model tracking
            "va_success", "va_error",
            "exec_success",
            "pred_row_count", "gt_row_count",
            "latency_s", "error"
        ]
        
        if output_path.endswith('.xlsx') and HAS_EXCEL:
            # Save as Excel with formatting
            save_to_excel(results, fieldnames, output_path, by_difficulty, total, exec_success, result_match, latencies)
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
  python -m t2sql.testing_evaluation --testset qa_classicmodels_devset.yml --output results.xlsx

  # Test only hard questions
  python -m t2sql.testing_evaluation --testset qa_classicmodels_devset.yml --output results.xlsx --hard

  # Test easy and medium questions
  python -m t2sql.testing_evaluation --testset qa_classicmodels_devset.yml --output results.xlsx --easy --medium

  # Test only easy question #4
  python -m t2sql.testing_evaluation --testset qa_classicmodels_devset.yml --output results.xlsx --easy4

  # Test easy questions #4-7
  python -m t2sql.testing_evaluation --testset qa_classicmodels_devset.yml --output results.xlsx --easy4-7

  # Test specific questions: easy #1, #3, #5
  python -m t2sql.testing_evaluation --testset qa_classicmodels_devset.yml --output results.xlsx --easy1,3,5
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
    # Backward compatibility for existing commands
    null_zero_mode = "on" if args.null_zero_equivalent else args.null_zero_mode

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

    asyncio.run(run_batch_eval(
        testset_path=args.testset,
        output_path=args.output,
        execute_groundtruth=not args.no_groundtruth_exec,
        skip_warmup=args.skip_warmup,
        difficulty_filters=difficulty_filters if difficulty_filters else None,
        null_zero_mode=null_zero_mode,
    ))

if __name__ == "__main__":
    main()
