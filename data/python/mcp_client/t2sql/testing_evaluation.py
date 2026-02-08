"""
Batch Evaluation Script untuk Text-to-SQL Pipeline.
Membaca test set dari YAML dan menjalankan semua pertanyaan secara otomatis.

Usage:
    python -m t2sql.batch_eval --testset path/to/testset.yml
"""

import asyncio
import argparse
import csv
import os
import time
from pathlib import Path
from datetime import datetime

import yaml
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

# Excel support
try:
    import openpyxl
    from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
    from openpyxl.utils.dataframe import dataframe_to_rows
    HAS_EXCEL = True
except ImportError:
    HAS_EXCEL = False

# ============================================================
# Set default OLLAMA_URL untuk Docker environment
# ============================================================
os.environ.setdefault("OLLAMA_URL", "http://ollama:11434")

from .config import MCP_BASE, CHROMA_DIR, COLLECTION_NAME, JSON_INPUT, OLLAMA_URL, ROUTING_MODEL, SMALL_MODEL, LARGE_MODEL, OLLAMA_KEEP_ALIVE
from .state import make_state
from .graph import build_app
from .mcp_execution import query_db_full, _mcp_call  # Pakai FULL query tanpa LIMIT dan _mcp_call helper
from .utils import resolve_jsonl_paths

# For direct Ollama warmup
import httpx


def load_testset(yaml_path: str) -> list[dict]:
    """Load test cases dari file YAML."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("cases", [])


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


def values_match(val1, val2, tolerance: float = 0.01) -> bool:
    """
    Compare two values with numeric tolerance.
    
    For numeric values: match if within tolerance (default 1%)
    For strings: exact match (case-insensitive)
    """
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


def compare_results(predicted_rows: list, groundtruth_rows: list, order_sensitive: bool = False, numeric_tolerance: float = 0.01) -> tuple[bool, str]:
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
        # Normalize rows - convert each row to tuple of SORTED VALUES
        def row_to_tuple(row: dict) -> tuple:
            return tuple(sorted(normalize_value(v) for v in row.values()))
        
        pred_tuples = [row_to_tuple(r) for r in predicted_rows]
        gt_tuples = [row_to_tuple(r) for r in groundtruth_rows]
        
        # First try exact match (fast path)
        pred_set = set(pred_tuples)
        gt_set = set(gt_tuples)
        
        if pred_set == gt_set:
            return True, "Exact match (unordered)"
        
        # If no exact match, try with numeric tolerance
        # Compare row by row with tolerance
        def rows_match_with_tolerance(pred_list, gt_list) -> bool:
            """Check if all rows match with numeric tolerance."""
            if len(pred_list) != len(gt_list):
                return False
            
            # Sort both lists for comparison
            pred_sorted = sorted(pred_list)
            gt_sorted = sorted(gt_list)
            
            for p_row, g_row in zip(pred_sorted, gt_sorted):
                if len(p_row) != len(g_row):
                    return False
                for p_val, g_val in zip(p_row, g_row):
                    if not values_match(p_val, g_val, numeric_tolerance):
                        return False
            return True
        
        if rows_match_with_tolerance(pred_tuples, gt_tuples):
            return True, f"Match with tolerance ({numeric_tolerance*100:.1f}%)"
        
        # No match - find differences for debugging
        missing = gt_set - pred_set
        extra = pred_set - gt_set
        
        reason = []
        if missing:
            reason.append(f"missing {len(missing)} rows")
        if extra:
            reason.append(f"extra {len(extra)} rows")
        
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
        "id": 8, "difficulty": 10, "question": 50,
        "groundtruth_sql": 40, "predicted_sql": 40,
        "model_used": 10,  # Model 3b/7b
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
) -> dict:
    """
    Run a single test case and return evaluation result.
    """
    case_id = case.get("id", "unknown")
    question = case.get("question", "")
    groundtruth_sql = case.get("groundtruth_sql", "").strip()
    difficulty = case.get("difficulty", "unknown")
    
    print(f"\n{'='*60}")
    print(f"[{case_id}] ({difficulty}) {question[:80]}...")
    print(f"{'='*60}")
    
    result = {
        "id": case_id,
        "difficulty": difficulty,
        "question": question,
        "groundtruth_sql": groundtruth_sql,
        "predicted_sql": "",
        "model_used": "",       # Model yang digunakan (3b/7b)
        "va_success": False,     # Valid SQL (EXPLAIN)
        "va_error": None,        # Error message if not valid
        "exec_success": False,
        "result_match": False,  # Execution Accuracy (hasil sama)
        "match_reason": "",     # Alasan match/mismatch
        "pred_row_count": 0,
        "gt_row_count": 0,
        "latency_s": 0,  # Latency dalam detik
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
        
        result["predicted_sql"] = predicted_sql
        
        # Use inference time from agent (excludes cold start/model loading)
        # Fallback to total time if not available
        inference_time = state.get("draft_inference_time", 0)
        if inference_time > 0:
            result["latency_s"] = round(inference_time, 2)
        else:
            result["latency_s"] = round((time.perf_counter() - t0), 2)
        
        # Get model used (extract short name: 3b/7b)
        target_model = state.get("target_model", "")
        if "3b" in target_model.lower():
            result["model_used"] = "3b"
        elif "7b" in target_model.lower():
            result["model_used"] = "7b"
        else:
            result["model_used"] = target_model or "unknown"
        
        print(f"[MODEL] {result['model_used']}")
        
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
            
            # Compare results
            is_match, reason = compare_results(pred_rows, gt_rows, order_sensitive=False)
            result["result_match"] = is_match
            result["match_reason"] = reason
            
            status = "✅ MATCH" if is_match else "❌ MISMATCH"
            print(f"[EVALUATION] {status} - {reason}")
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
    print("🔥 WARMUP PHASE - Loading models into memory...")
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
    print("✅ Warmup complete - all models are now in memory")
    print("=" * 60 + "\n")


async def run_batch_eval(
    testset_path: str,
    output_path: str | None = None,
    paths: list[str] | None = None,
    execute_groundtruth: bool = True,
    skip_warmup: bool = False,
    difficulties: list[str] | None = None,
):
    """
    Main batch evaluation function.

    Args:
        testset_path: Path to test set YAML file
        output_path: Path to output file (.xlsx for Excel, .csv for CSV)
        paths: JSONL paths for context
        execute_groundtruth: Whether to execute groundtruth SQL for comparison
        skip_warmup: Skip warmup phase (not recommended)
        difficulties: List of difficulties to filter (e.g., ['easy', 'hard']). None = all
    """
    # Load test cases
    all_cases = load_testset(testset_path)
    print(f"Loaded {len(all_cases)} test cases from {testset_path}")

    # Filter by difficulty if specified
    if difficulties:
        difficulties_lower = [d.lower() for d in difficulties]
        cases = [c for c in all_cases if c.get("difficulty", "").lower() in difficulties_lower]
        print(f"Filtered to {len(cases)} test cases with difficulty: {', '.join(difficulties)}")
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
                execute_groundtruth=execute_groundtruth
            )
            results.append(result)
    

    # Calculate metrics
    total = len(results)
    va_success = sum(1 for r in results if r["va_success"])
    exec_success = sum(1 for r in results if r["exec_success"])
    result_match = sum(1 for r in results if r["result_match"])

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
            "id", "difficulty", "question", 
            "groundtruth_sql", "predicted_sql",
            "model_used",  # Model yang digunakan (3b/7b)
            "va_success", "va_error",
            "exec_success", "result_match", "match_reason",
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


def main():
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

  # Test only medium questions (single difficulty)
  python -m t2sql.testing_evaluation --testset qa_classicmodels_devset.yml --output results.xlsx --medium
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
    # Difficulty filter arguments
    parser.add_argument(
        "--easy",
        action="store_true",
        help="Only test easy questions"
    )
    parser.add_argument(
        "--medium",
        action="store_true",
        help="Only test medium questions"
    )
    parser.add_argument(
        "--hard",
        action="store_true",
        help="Only test hard questions"
    )

    args = parser.parse_args()

    # Build difficulty filter list
    difficulties = []
    if args.easy:
        difficulties.append("easy")
    if args.medium:
        difficulties.append("medium")
    if args.hard:
        difficulties.append("hard")

    # Generate default output filename if not specified
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Default to Excel if available
        ext = ".xlsx" if HAS_EXCEL else ".csv"
        # Add difficulty suffix if filtering
        suffix = f"_{'-'.join(difficulties)}" if difficulties else ""
        args.output = f"eval_results{suffix}_{timestamp}{ext}"

    asyncio.run(run_batch_eval(
        testset_path=args.testset,
        output_path=args.output,
        execute_groundtruth=not args.no_groundtruth_exec,
        skip_warmup=args.skip_warmup,
        difficulties=difficulties if difficulties else None,
    ))


if __name__ == "__main__":
    main()
