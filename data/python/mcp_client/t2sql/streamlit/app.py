from __future__ import annotations

import asyncio
import html
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import find_dotenv, load_dotenv

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

load_dotenv(find_dotenv(), override=False)

from t2sql.config import COLLECTION_NAME, HITL_LOG_FILENAME_PREFIX, HITL_LOG_PATH, SPARSE_METADATA_JSONL
from t2sql.graph import build_app
from t2sql.helpers_core import _preexecute_sql_safety_pass
from t2sql.mcp_execution import query_db, query_db_full
from t2sql.repl import _is_executable_sql_candidate
from t2sql.state import make_state
from t2sql.utils import resolve_jsonl_paths


st.set_page_config(
    page_title="Text to SQL",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner=False)
def _compiled_graph():
    return build_app()


@st.cache_data(show_spinner=False)
def _metadata_paths(spec: str) -> list[str]:
    return resolve_jsonl_paths(spec)


def _load_css() -> None:
    css_path = Path(__file__).with_name("styles.css")
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _ensure_ui_state() -> None:
    st.session_state.setdefault("page_nav", "Home")
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("feedback_open", False)
    st.session_state.setdefault("feedback_message", "")


def _database_label() -> str:
    for key in ("MYSQL_DATABASE", "DATABASE_NAME", "DB_NAME", "T2SQL_DATABASE"):
        value = str(os.getenv(key) or "").strip()
        if value:
            return value
    return COLLECTION_NAME or "database"


def _database_options() -> list[str]:
    raw = str(os.getenv("T2SQL_DATABASE_OPTIONS") or os.getenv("DATABASE_OPTIONS") or "").strip()
    values = [item.strip() for item in raw.split(",") if item.strip()]
    active = _database_label()
    out: list[str] = []
    for item in [active, *values]:
        if item and item not in out:
            out.append(item)
    return out or [_database_label()]


def _selected_database() -> str:
    options = _database_options()
    selected = st.selectbox(
        "Database",
        options,
        index=0,
        key="database_choice",
        label_visibility="collapsed",
        help="Choose an available database.",
    )
    return str(selected or "").strip()


def _payload_frame(payload: dict[str, Any] | None) -> pd.DataFrame:
    if not isinstance(payload, dict):
        return pd.DataFrame()
    columns = payload.get("preview_columns") or payload.get("columns") or []
    rows = payload.get("preview_rows") or payload.get("rows") or []
    if not isinstance(columns, list) or not isinstance(rows, list):
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=columns)


def _styled_result_frame(df: pd.DataFrame) -> Any:
    return (
        df.style
        .set_properties(
            **{
                "background-color": "#11141a",
                "color": "#eef6ff",
                "border-color": "rgba(255, 255, 255, 0.10)",
            }
        )
        .set_table_styles(
            [
                {"selector": "th", "props": [("background-color", "#1c2028"), ("color", "#cbd5e1")]},
                {"selector": "td", "props": [("background-color", "#11141a"), ("color", "#eef6ff")]},
            ]
        )
    )


async def _execute_query(sql: str, database: str | None = None) -> dict[str, Any]:
    return await query_db(sql, database=database)


async def _execute_full_query(sql: str, database: str | None = None) -> dict[str, Any]:
    return await query_db_full(sql, database=database)


def _rows_frame(payload: dict[str, Any] | None) -> pd.DataFrame:
    if not isinstance(payload, dict):
        return pd.DataFrame()
    columns = payload.get("columns") or []
    rows = payload.get("rows") or []
    if not isinstance(columns, list) or not isinstance(rows, list):
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=columns)


@st.cache_data(show_spinner=False)
def _full_csv_payload(sql: str, database: str) -> tuple[bytes, str]:
    sql_clean = str(sql or "").strip()
    if not sql_clean:
        return b"", "No SQL available for export."
    db_name = str(database or "").strip() or None
    payload = asyncio.run(_execute_full_query(sql_clean, db_name))
    if not isinstance(payload, dict) or not payload.get("success"):
        return b"", str((payload or {}).get("error") or "Full CSV export failed.")
    df = _rows_frame(payload)
    return df.to_csv(index=False).encode("utf-8"), ""


def _feedback_log_path() -> Path:
    raw_path = str(os.getenv("T2SQL_STREAMLIT_FEEDBACK_LOG") or "").strip()
    if raw_path:
        return Path(raw_path)

    base_path = Path(HITL_LOG_PATH or "reports")
    if base_path.suffix:
        return base_path

    prefix = "_".join(
        part
        for part in str(HITL_LOG_FILENAME_PREFIX or "hitl_review_log").replace("-", "_").split()
        if part
    ).strip("_")
    prefix = prefix or "hitl_review_log"
    return base_path / f"{prefix}_streamlit.jsonl"


def _append_feedback_log(result: dict[str, Any], review_type: str, feedback: str) -> Path:
    state = result.get("state") or {}
    execution = result.get("execution") or {}
    if not isinstance(execution, dict):
        execution = {"raw": execution}

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "streamlit",
        "question": state.get("question") or st.session_state.get("question_input", ""),
        "draft_sql": state.get("draft_sql", ""),
        "final_sql": state.get("final_sql", ""),
        "execution_sql": result.get("sql", ""),
        "database": result.get("database", ""),
        "status": result.get("status", ""),
        "success": bool(result.get("success", False)),
        "error_type": result.get("status", ""),
        "error_message": result.get("error", ""),
        "hitl_action": "streamlit_review",
        "hitl_feedback_type": review_type,
        "hitl_feedback": feedback,
        "router_decision": state.get("router_decision", {}),
        "router_tables": state.get("router_tables", []),
        "target_model": state.get("target_model", ""),
        "final_model": state.get("final_model", ""),
        "qtype": state.get("qtype", ""),
        "sql_confidence": state.get("sql_confidence", 0.0),
        "validation_errors": state.get("validation_errors", []),
        "critic_feedback": state.get("critic_feedback", ""),
        "elapsed": result.get("elapsed", 0.0),
        "total_rows": execution.get("total_rows"),
    }
    log_path = _feedback_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")
    try:
        log_path.parent.chmod(0o777)
        log_path.chmod(0o666)
    except OSError:
        pass
    return log_path


def _run_question(question: str, database: str | None = None) -> dict[str, Any]:
    paths = _metadata_paths(SPARSE_METADATA_JSONL)
    if not paths:
        return {
            "success": False,
            "status": "Metadata missing",
            "error": f"No sparse JSONL input found for SPARSE_METADATA_JSONL={SPARSE_METADATA_JSONL!r}.",
            "state": {},
            "execution": {},
            "elapsed": 0.0,
        }

    app = _compiled_graph()
    state = make_state(paths, question)
    started = time.perf_counter()

    try:
        state = app.invoke(state)
    except Exception as exc:
        return {
            "success": False,
            "status": "Pipeline error",
            "error": str(exc),
            "traceback": traceback.format_exc(limit=8),
            "state": state,
            "execution": {},
            "elapsed": time.perf_counter() - started,
        }

    decision = state.get("router_decision", {}) or {}
    if decision.get("is_valid_question") is False:
        reason = str(decision.get("rejection_reason") or state.get("answer") or "Question rejected.").strip()
        return {
            "success": False,
            "status": "Rejected",
            "error": reason,
            "state": state,
            "execution": {},
            "elapsed": time.perf_counter() - started,
        }

    sql_candidate = str(state.get("final_sql") or state.get("answer") or "").strip()
    final_model = str(state.get("final_model") or "").strip().lower()
    if final_model == "gate_blocked" or "FINAL_GATE_BLOCKED" in sql_candidate.upper():
        return {
            "success": False,
            "status": "Needs review",
            "error": "The system blocked this result because the generated SQL did not pass validation or safety checks.",
            "sql": sql_candidate,
            "state": state,
            "execution": {},
            "database": str(database or "").strip(),
            "elapsed": time.perf_counter() - started,
        }

    sql_clean = _preexecute_sql_safety_pass(sql_candidate, question).rstrip(";").strip()
    executable, reason = _is_executable_sql_candidate(sql_clean)
    if not executable:
        return {
            "success": False,
            "status": "Not executable",
            "error": reason,
            "sql": sql_clean,
            "state": state,
            "execution": {},
            "elapsed": time.perf_counter() - started,
        }

    db_name = str(database or "").strip() or None
    execution = asyncio.run(_execute_query(sql_clean, db_name))
    ok = bool(isinstance(execution, dict) and execution.get("success"))
    return {
        "success": ok,
        "status": "Success" if ok else "Execution failed",
        "error": "" if ok else str((execution or {}).get("error") or "Unknown execution error"),
        "sql": sql_clean,
        "state": state,
        "execution": execution if isinstance(execution, dict) else {"raw": execution},
        "database": db_name or "",
        "elapsed": time.perf_counter() - started,
    }


def _metric_card(label: str, value: str, tone: str = "") -> None:
    safe_label = html.escape(str(label))
    safe_value = html.escape(str(value))
    safe_tone = "success" if tone == "success" else "failed" if tone == "failed" else ""
    st.markdown(
        f"""
        <div class="metric-card {safe_tone}">
            <div class="metric-label">{safe_label}</div>
            <div class="metric-value">{safe_value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_nav_panel() -> str:
    current_page = str(st.session_state.get("page_nav") or "Home")
    st.markdown(
        """
        <div class="sidebar-brand-card">
            <div class="sidebar-brand-row">
                <div class="sidebar-brand-icon" aria-hidden="true">
                    <svg viewBox="0 0 28 28" role="img" focusable="false">
                        <ellipse cx="14" cy="7" rx="8" ry="4" />
                        <path d="M6 7v9c0 2.2 3.6 4 8 4s8-1.8 8-4V7" />
                        <path d="M6 11c0 2.2 3.6 4 8 4s8-1.8 8-4" />
                        <path d="M10 20l-2.2 2.2L10 24.4" />
                        <path d="M18 20l2.2 2.2L18 24.4" />
                    </svg>
                </div>
                <div>
                    <div class="sidebar-brand-name">Text-to-SQL</div>
                    <div class="sidebar-brand-note">Prototype</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(key="sidebar_nav"):
        nav_home_col, nav_about_col = st.columns(2, gap="small")
        with nav_home_col:
            if st.button(
                "Home",
                key="nav_home",
                use_container_width=True,
                type="primary" if current_page == "Home" else "secondary",
            ):
                st.session_state["page_nav"] = "Home"
                current_page = "Home"
        with nav_about_col:
            if st.button(
                "About",
                key="nav_about",
                use_container_width=True,
                type="primary" if current_page == "About" else "secondary",
            ):
                st.session_state["page_nav"] = "About"
                current_page = "About"

    with st.expander("Tips", expanded=True):
        st.markdown(
            """
            <div class="tips-accordion-body">
                <ul>
                    <li>Ask a clear question in natural language.</li>
                    <li>Add details such as filters, date ranges, sorting order, or result limits when needed.</li>
                    <li>Specify output columns only when you need particular fields.</li>
                    <li>Write casually or formally. The system can interpret both styles.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    return current_page


def _render_about() -> None:
    st.markdown(
        """
        <div class="about-page">
            <div class="workspace-kicker">Query Workspace</div>
            <div class="workspace-title">About this workspace</div>
            <div class="workspace-subtitle">A Text-to-SQL workspace for turning natural language questions into executable database queries.</div>
            <div class="content-panel about-panel">
                <div class="panel-title">How this project works</div>
                <div class="panel-copy">This project uses centralized metadata from DataHub as the source of truth for schema context, table relationships, and business descriptions.</div>
                <div class="panel-copy">A retrieval and multi-agent planning flow uses that metadata to generate SQL, validate identifiers, and keep query generation grounded in the available database structure.</div>
                <div class="panel-copy">SQL execution runs through the Model Context Protocol (MCP) layer, which handles database calls and returns preview or full-result payloads to the app.</div>
                <div class="panel-copy">The project is built with open-source tools and containerized with Docker, making the environment easier to reproduce across machines.</div>
                <div class="panel-copy">The Streamlit interface lets users ask questions, inspect preview results, export full query output, and provide feedback on the system result.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def _section_heading_html(text: str, extra_classes: str = "") -> str:
    classes = "section-heading"
    if extra_classes.strip():
        classes = f"{classes} {extra_classes.strip()}"
    return f"<div class=\"{classes}\" style=\"font-size: 18px; line-height: 1.25; font-weight: 850; letter-spacing: 0;\">{html.escape(text)}</div>"


def _render_home() -> None:
    header_left, header_right = st.columns([0.74, 0.26], vertical_alignment="top")
    with header_left:
        st.markdown(
            """
            <div class="workspace-kicker">Query Workspace</div>
            <div class="workspace-title">Ask your database a question</div>
            <div class="workspace-subtitle">Type your request in English and preview up to 5 rows. You can download the full output when needed.</div>
            """,
            unsafe_allow_html=True,
        )
    with header_right:
        with st.container(key="database_picker"):
            st.markdown('<div class="field-label">Database</div>', unsafe_allow_html=True)
            selected_database = _selected_database()

    st.markdown(_section_heading_html("Question"), unsafe_allow_html=True)
    prompt_col, button_col = st.columns([0.78, 0.22], vertical_alignment="center")
    with prompt_col:
        st.text_area(
            "Question",
            placeholder="Example: Show the top 10 customers by total payment amount.",
            label_visibility="collapsed",
            key="question_input",
            height=72,
        )
    with button_col:
        submitted = st.button("Generate SQL", use_container_width=True, type="primary", key="generate_sql")

    question = str(st.session_state.get("question_input") or "")
    if submitted and question.strip():
        st.session_state["feedback_open"] = False
        st.session_state["feedback_message"] = ""
        loading_slot = st.empty()
        loading_slot.markdown(
            """
            <div class="loading-row">
                <span class="loading-spinner"></span>
                <span>Generating and executing SQL...</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        try:
            result = _run_question(question.strip(), selected_database)
        finally:
            loading_slot.empty()
        st.session_state["last_result"] = result
        st.session_state["history"].insert(
            0,
            {
                "question": question.strip(),
                "status": result.get("status", ""),
                "elapsed": result.get("elapsed", 0.0),
                "database": selected_database,
                "submitted_at": datetime.now(ZoneInfo("Asia/Jakarta")).strftime("%d %b %Y, %H:%M WIB"),
            },
        )
        st.session_state["history"] = st.session_state["history"][:8]

    result = st.session_state.get("last_result")
    if result:
        _render_result(result)
    else:
        _render_empty_result()

    _render_history()


def _render_empty_result() -> None:
    st.markdown(_section_heading_html("Query Result", "result-heading"), unsafe_allow_html=True)
    st.markdown(
        """
        <div class="content-panel result-panel empty-panel">
            <div class="empty-title">No query has been generated yet.</div>
            <div class="empty-subtitle">Your result preview will appear here after you ask a question.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _friendly_result_error_message(result: dict[str, Any]) -> str:
    status = str(result.get("status") or "").strip().lower()
    if status in {"needs review", "not executable", "execution failed", "pipeline error"}:
        return (
            "We couldn't produce a reliable result for this question. "
            "Please try rephrasing it with clearer columns, filters, or sorting details. "
            "You can also use Give Feedback to help review this result."
        )
    return str(result.get("error") or "The query could not be completed.")


def _render_result(result: dict[str, Any]) -> None:
    execution = result.get("execution") or {}
    state = result.get("state") or {}
    df = _payload_frame(execution)
    total_rows = execution.get("total_rows", len(df)) if isinstance(execution, dict) else len(df)

    st.markdown(_section_heading_html("Query Result", "result-heading"), unsafe_allow_html=True)
    sql_text = str(result.get("sql") or state.get("final_sql") or "").strip()
    database = str(result.get("database") or "").strip()
    full_csv = b""
    export_error = ""
    if result.get("success") and sql_text:
        with st.spinner("Preparing full CSV..."):
            full_csv, export_error = _full_csv_payload(sql_text, database)
    if result.get("success") and not df.empty:
        st.dataframe(_styled_result_frame(df), use_container_width=True, hide_index=False)
        preview_rows = len(df.index)
        if total_rows > preview_rows:
            st.markdown(
                f'<div class="preview-note">Showing {preview_rows} preview rows from {int(total_rows)} total rows. Download CSV for the complete result.</div>',
                unsafe_allow_html=True,
            )
    elif result.get("success"):
        st.info("Query executed successfully, but returned no rows.")
    else:
        st.error(_friendly_result_error_message(result))

    action_col, metric_col_1, metric_col_2 = st.columns([0.36, 0.32, 0.32], vertical_alignment="top")
    with action_col:
        st.download_button(
            "Download CSV",
            data=full_csv,
            file_name="t2sql_result.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=not full_csv,
        )
        if st.button("Give Feedback", key="review_result_button", use_container_width=True):
            st.session_state["feedback_open"] = True
            st.session_state["feedback_message"] = ""
        if export_error:
            st.warning(export_error)
    with metric_col_1:
        _metric_card("Execution Time", f"{float(result.get('elapsed') or 0.0):.3f} sec")
    with metric_col_2:
        _metric_card("Rows Returned", f"{total_rows} rows", "success" if result.get("success") else "failed")

    _render_feedback_form(result)


def _render_feedback_form(result: dict[str, Any]) -> None:
    message = str(st.session_state.get("feedback_message") or "")
    if message:
        st.success(message)

    if not st.session_state.get("feedback_open"):
        return

    st.markdown('<div class="feedback-heading">Give Feedback</div>', unsafe_allow_html=True)
    with st.form("result_feedback_form", clear_on_submit=True):
        review_type = st.selectbox(
            "Review type",
            [
                "Semantically incorrect answer",
                "Incomplete answer",
                "Wrong columns or fields",
                "Wrong rows or filters",
                "System blocked or failed",
                "Other",
            ],
            index=0 if result.get("success") else 4,
        )
        feedback = st.text_area(
            "Feedback details",
            placeholder="Describe why the answer is incorrect, what result you expected, or how the question should be interpreted.",
            height=110,
        )
        submitted = st.form_submit_button("Submit Review", use_container_width=True)

    if submitted:
        try:
            _append_feedback_log(result, review_type, feedback.strip())
            st.session_state["feedback_open"] = False
            st.session_state["feedback_message"] = "Review saved. Thank you for the feedback."
            st.success(st.session_state["feedback_message"])
        except Exception as exc:
            st.error(f"Feedback could not be saved: {exc}")


def _render_history() -> None:
    history = st.session_state.get("history", [])
    st.markdown(_section_heading_html("Recent Questions", "history-heading"), unsafe_allow_html=True)
    if not history:
        st.markdown('<div class="history-empty">Recent questions will appear here.</div>', unsafe_allow_html=True)
        return

    for idx, item in enumerate(history[:4]):
        safe_question = html.escape(str(item.get("question", "")))
        safe_status = html.escape(str(item.get("status", "")))
        elapsed = float(item.get("elapsed") or 0.0)
        submitted_at = str(item.get("submitted_at") or "").strip()
        safe_submitted_at = html.escape(submitted_at)

        with st.container(key=f"history_row_{idx}"):
            text_col, time_col, delete_col = st.columns([0.84, 0.12, 0.04], gap="small", vertical_alignment="center")
            with text_col:
                st.markdown(
                    f'''
                    <div class="history-row-text">
                        <div class="history-row-question">{safe_question}</div>
                        <div class="history-row-meta">{safe_status} · {elapsed:.2f}s</div>
                    </div>
                    ''',
                    unsafe_allow_html=True,
                )
            with time_col:
                if safe_submitted_at:
                    st.markdown(f'<div class="history-row-time">{safe_submitted_at}</div>', unsafe_allow_html=True)
            with delete_col:
                if st.button("×", key=f"delete_history_{idx}", help="Remove this recent question"):
                    st.session_state["history"].pop(idx)
                    st.rerun()


def main() -> None:
    _ensure_ui_state()
    _load_css()
    nav_col, content_col = st.columns([0.24, 0.76], gap="large")
    with nav_col:
        with st.container(key="sidebar_panel"):
            page = _render_nav_panel()
    with content_col:
        if page == "About":
            _render_about()
        else:
            _render_home()

    st.markdown('<div class="app-watermark">© 2026 Jovian Aditya Wiranata</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
