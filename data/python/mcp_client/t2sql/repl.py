# ======================================================================
# ============================== repl.py ===============================
# ======================================================================
# Purpose:
# - REPL for interacting with the MCP client to ask questions about data.
# - Handles user input, invokes the application logic, and displays results.

# ======================================================================
# ====================== STANDARD LIBRARY IMPORT =======================
# ======================================================================
import asyncio
import csv
import json
import re
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
# ======================================================================
# ======================== LOCAL MODULE IMPORT =========================
# ======================================================================
from .config import (
    MCP_BASE,
    CHROMA_DIR,
    COLLECTION_NAME,
    CSV_EXPORT_DIR,
    CSV_EXPORT_FILENAME_PREFIX,
    HITL_ENABLED,
    HITL_GENERATED_ERROR_SQL_PATTERN,
    HITL_LOG_FILENAME_PREFIX,
    HITL_LOG_PATH,
    HITL_SUCCESS_REVIEW_ENABLED,
    HITL_TEST_FORCE_FAILURE,
)
from .state import make_state
from .graph import build_app
from .chroma_embedding import build_vector_store
from .mcp_execution import query_db, query_db_full
from .helpers_core import _preexecute_sql_safety_pass
from .utils import _now, _log_total, _print_banner, display_table_from_json

def _print_latency_debug(state: dict):
    """
    Unified latency debug for CLI/REPL runs.
    Primary metric:
    - processing_time -> warm prompt->final SQL
      (model receives prompt -> final SQL ready)
    """
    processing_time = float(state.get("processing_time", 0.0) or 0.0)
    draft_inference_time = float(state.get("draft_inference_time", 0.0) or 0.0)

    latency_source = "unavailable"
    latency_s = 0.0
    if processing_time > 0:
        latency_source = "warm_prompt_to_final_sql"
        latency_s = processing_time
    elif draft_inference_time > 0:
        latency_source = "warm_inference_only"
        latency_s = draft_inference_time
    elif bool(state.get("ir_used", False)):
        latency_source = "no_llm_ir_path"

    print(f"[LATENCY] source={latency_source}, latency_s={latency_s:.3f}")
    if processing_time > 0:
        postproc_time = max(0.0, processing_time - draft_inference_time)
        print(
            f"[LATENCY] breakdown: draft={draft_inference_time:.3f}s, "
            f"postprocess={postproc_time:.3f}s"
        )

async def _exec_via_mcp(
    mcp: Client,
    sql_block: str,
    final_sql: str | None = None,
    question: str = "",
) -> dict:
    sql_only = (sql_block or "").split("\n\n-- notes:")[0].strip()
    sql_clean = (final_sql or sql_only).rstrip(";").strip()
    sql_clean = _preexecute_sql_safety_pass(sql_clean, question).rstrip(";").strip()

    print("\n[SQL EXEC] Final query sent to MCP:")
    print(sql_clean + ";")

    try:
        generated_error_match = bool(
            re.search(HITL_GENERATED_ERROR_SQL_PATTERN, sql_clean, flags=re.IGNORECASE)
        )
    except re.error as exc:
        print(f"[HITL] Invalid generated-error SQL regex: {exc}")
        generated_error_match = False

    if generated_error_match:
        print("query_db skipped: generated SQL indicates system failure")
        return {
            "success": False,
            "sql": sql_clean,
            "error_type": "generation_failure_sql",
            "error_message": sql_clean,
        }

    if HITL_TEST_FORCE_FAILURE:
        print("query_db failed: forced HITL test failure")
        return {
            "success": False,
            "sql": sql_clean,
            "error_type": "forced_hitl_test",
            "error_message": "Forced HITL test failure",
        }

    try:
        payload = await query_db(sql_clean, mcp=mcp)

        if isinstance(payload, dict) and payload.get("success"):
            display_table_from_json(payload)
            return {
                "success": True,
                "sql": sql_clean,
                "payload": payload,
            }

        err = payload.get("error") if isinstance(payload, dict) else str(payload)
        print(f"query_db failed: {err}")
        return {
            "success": False,
            "sql": sql_clean,
            "error_type": "execution_error",
            "error_message": str(err or "Unknown query_db failure"),
            "payload": payload,
        }

    except asyncio.TimeoutError:
        print("MCP query_db timeout")
        return {
            "success": False,
            "sql": sql_clean,
            "error_type": "timeout",
            "error_message": "MCP query_db timeout",
        }
    except Exception as e:
        print(f"MCP query_db error: {e}")
        return {
            "success": False,
            "sql": sql_clean,
            "error_type": "exception",
            "error_message": str(e),
        }

def _append_hitl_review_log(
    state: dict,
    execution_result: dict,
    action: str,
    feedback: str = "",
) -> None:
    base_path = Path(HITL_LOG_PATH or "reports")
    if base_path.suffix:
        log_path = base_path
    else:
        jakarta_tz = timezone(timedelta(hours=7))
        timestamp = datetime.now(jakarta_tz).strftime("%Y%m%d_%H%M%S")
        prefix = re.sub(
            r"[^A-Za-z0-9_-]+",
            "_",
            str(HITL_LOG_FILENAME_PREFIX or "hitl_review_log"),
        )
        prefix = prefix.strip("_") or "hitl_review_log"
        log_path = base_path / f"{prefix}_{timestamp}.json"
        if log_path.exists():
            for i in range(1, 1000):
                numbered = base_path / f"{prefix}_{timestamp}_{i:03d}.json"
                if not numbered.exists():
                    log_path = numbered
                    break
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": state.get("question", ""),
            "draft_sql": state.get("draft_sql", ""),
            "final_sql": state.get("final_sql", ""),
            "execution_sql": execution_result.get("sql", ""),
            "error_type": execution_result.get("error_type", ""),
            "error_message": execution_result.get("error_message", ""),
            "hitl_action": action,
            "hitl_feedback": feedback,
            "router_decision": state.get("router_decision", {}),
            "router_tables": state.get("router_tables", []),
            "target_model": state.get("target_model", ""),
            "final_model": state.get("final_model", ""),
            "qtype": state.get("qtype", ""),
            "sql_confidence": state.get("sql_confidence", 0.0),
            "validation_errors": state.get("validation_errors", []),
            "critic_feedback": state.get("critic_feedback", ""),
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
        print(f"[HITL] Review logged to {log_path}")
    except Exception as e:
        print(f"[HITL] Failed to write review log: {e}")

def _print_failure_summary(state: dict, execution_result: dict) -> None:
    error_type = str(execution_result.get("error_type") or "unknown_error")
    error_message = str(execution_result.get("error_message") or "").strip()

    print("\n[FAILURE SUMMARY]")
    if error_type == "generation_failure_sql":
        print("SQL query was not produced successfully.")
        print("The system generated an internal error marker instead of a valid data query.")
    elif error_type == "pipeline_exception":
        print("The text-to-SQL pipeline failed before SQL execution.")
    elif error_type in {"execution_error", "timeout", "exception"}:
        print("SQL was generated, but execution failed.")
    elif error_type == "forced_hitl_test":
        print("HITL test mode forced this query to fail.")
    else:
        print("The system could not complete this query.")

    if error_message:
        print(f"Reason: {error_message}")

    final_sql = str(state.get("final_sql") or execution_result.get("sql") or "").strip()
    if final_sql:
        print(f"Last SQL candidate: {final_sql}")

def _prompt_hitl_review(state: dict, execution_result: dict) -> tuple[str, str]:
    if not HITL_ENABLED or execution_result.get("success", False):
        return "continue", ""

    error_info = {
        "type": execution_result.get("error_type", ""),
        "message": execution_result.get("error_message", ""),
    }
    state["execution_error"] = error_info

    _print_failure_summary(state, execution_result)

    print("\n[HITL] Optional post-failure review")
    print("1. Accept system decision and continue")
    print("2. Report system issue")
    print("3. Rephrase and submit a new question")
    print("4. End process")

    choice = input("Choose action [1-4, Enter=1]: ").strip().lower()
    if choice in {"", "1", "accept", "a"}:
        action = "accept_decision"
        feedback = ""
        next_question = ""
    elif choice in {"2", "report", "r"}:
        action = "report_issue"
        feedback = input("Describe the issue briefly: ").strip()
        next_question = ""
    elif choice in {"3", "rephrase", "retry"}:
        action = "rephrase_question"
        feedback = ""
        next_question = input("Rephrased question> ").strip()
        if not next_question:
            action = "rephrase_empty"
    elif choice in {"4", "exit", "quit", "q"}:
        action = "end_process"
        feedback = ""
        next_question = ""
    else:
        action = "invalid_choice_accept_decision"
        feedback = f"raw_choice={choice}"
        next_question = ""

    state["hitl_used"] = True
    state["hitl_action"] = action
    state["hitl_feedback"] = feedback
    _append_hitl_review_log(state, execution_result, action, feedback)

    if action == "end_process":
        return "exit", ""
    if action == "rephrase_question" and next_question:
        return "rephrase", next_question
    return "continue", ""

def _csv_export_path() -> Path:
    export_dir = Path(CSV_EXPORT_DIR or "hitl_exports")
    jakarta_tz = timezone(timedelta(hours=7))
    timestamp = datetime.now(jakarta_tz).strftime("%Y%m%d_%H%M%S")
    prefix = re.sub(r"[^A-Za-z0-9_-]+", "_", str(CSV_EXPORT_FILENAME_PREFIX or "t2sql_result"))
    prefix = prefix.strip("_") or "t2sql_result"
    candidate = export_dir / f"{prefix}_{timestamp}.csv"
    if not candidate.exists():
        return candidate

    for i in range(1, 1000):
        numbered = export_dir / f"{prefix}_{timestamp}_{i:03d}.csv"
        if not numbered.exists():
            return numbered
    return export_dir / f"{prefix}_{timestamp}_{datetime.now(timezone.utc).strftime('%f')}.csv"

async def _save_full_result_csv(
    mcp: Client,
    state: dict,
    execution_result: dict,
) -> tuple[bool, str]:
    sql = str(execution_result.get("sql") or state.get("final_sql") or "").strip()
    if not sql:
        return False, "No SQL available to export."

    try:
        payload = await query_db_full(sql, mcp=mcp)
    except Exception as exc:
        return False, f"query_db_full error: {exc}"

    if not isinstance(payload, dict) or not payload.get("success"):
        err = payload.get("error") if isinstance(payload, dict) else str(payload)
        return False, f"query_db_full failed: {err}"

    columns = list(payload.get("columns") or [])
    rows = list(payload.get("rows") or [])
    csv_path = _csv_export_path()

    try:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if columns:
                writer.writerow(columns)
            writer.writerows(rows)
    except Exception as exc:
        return False, f"CSV write failed: {exc}"

    exported_rows = len(rows)
    total_rows = payload.get("total_rows", exported_rows)
    truncated = bool(payload.get("truncated", False))
    note = f"Saved {exported_rows} rows to {csv_path}"
    if total_rows != exported_rows:
        note += f" (full result has {total_rows} rows)"
    if truncated:
        note += " [truncated by server export cap]"
    return True, note

async def _prompt_success_review(
    mcp: Client,
    state: dict,
    execution_result: dict,
) -> None:
    if not HITL_ENABLED or not HITL_SUCCESS_REVIEW_ENABLED:
        return
    if not execution_result.get("success", False):
        return

    choice = input(
        "\n[HITL] Press Enter to ask a new question, type 'report' if incorrect, or 'save' to export full CSV: "
    ).strip().lower()
    if choice in {"save", "s", "csv"}:
        ok, message = await _save_full_result_csv(mcp, state, execution_result)
        print(f"[HITL] {message}")
        if ok:
            state["hitl_used"] = True
            state["hitl_action"] = "save_success_csv"
            state["hitl_feedback"] = message
        return

    if choice not in {"report", "r"}:
        return

    feedback = input("Describe the issue briefly: ").strip()
    review_result = {
        **execution_result,
        "error_type": "semantic_review",
        "error_message": "User reported successful execution as semantically incorrect",
    }
    state["hitl_used"] = True
    state["hitl_action"] = "report_success_semantic_issue"
    state["hitl_feedback"] = feedback
    _append_hitl_review_log(
        state,
        review_result,
        "report_success_semantic_issue",
        feedback,
    )

async def _run_question_with_optional_review(
    app,
    mcp: Client,
    paths,
    question: str,
    total_label: str,
) -> str:
    current_question = question
    current_label = total_label

    while current_question:
        t0_total = _now()

        state = make_state(paths, current_question)
        try:
            state = app.invoke(state)
        except Exception as exc:
            print(f"\n[PIPELINE ERROR] {type(exc).__name__}: {exc}")
            execution_result = {
                "success": False,
                "sql": state.get("final_sql", ""),
                "error_type": "pipeline_exception",
                "error_message": str(exc),
                "traceback": traceback.format_exc(limit=12),
            }
            _log_total(current_label, t0_total)
            action, next_question = _prompt_hitl_review(state, execution_result)
            if action == "exit":
                print("Bye.")
                return "exit"
            if action == "rephrase" and next_question:
                current_question = next_question
                current_label = "End-to-end total time (rephrased question)"
                continue
            return "continue"

        execution_result = await _exec_via_mcp(
            mcp,
            state.get("answer", ""),
            state.get("final_sql"),
            current_question,
        )

        _log_total(current_label, t0_total)
        _print_latency_debug(state)

        action, next_question = _prompt_hitl_review(state, execution_result)
        if action == "exit":
            print("Bye.")
            return "exit"
        if action == "rephrase" and next_question:
            current_question = next_question
            current_label = "End-to-end total time (rephrased question)"
            continue
        await _prompt_success_review(mcp, state, execution_result)
        return "continue"

    return "continue"

async def repl(paths):
    app = build_app()
    transport = StreamableHttpTransport(MCP_BASE)

    async with Client(transport=transport) as mcp:
        first_q = input("Ask me anything about your data: ").strip()
        if not first_q:
            print("No question entered. Exiting.")
            return

        status = await _run_question_with_optional_review(
            app,
            mcp,
            paths,
            first_q,
            "End-to-end total time (first run)",
        )
        if status == "exit":
            return

        print("\nAsk me more questions! (type 'exit' to quit')")

        # ---------- LOOP (end-to-end only) ----------
        while True:
            q = input("\nQuestion> ").strip()
            if not q or q.lower() in {"exit", "quit", "q"}:
                print("Bye.")
                break

            status = await _run_question_with_optional_review(
                app,
                mcp,
                paths,
                q,
                "End-to-end total time (this question)",
            )
            if status == "exit":
                return
