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
from pathlib import Path
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
# ======================================================================
# ======================== LOCAL MODULE IMPORT =========================
# ======================================================================
from .config import MCP_BASE, CHROMA_DIR, COLLECTION_NAME
from .state import make_state
from .graph import build_app
from .chroma_embedding import build_vector_store
from .mcp_execution import query_db
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
):
    sql_only = (sql_block or "").split("\n\n-- notes:")[0].strip()
    sql_clean = (final_sql or sql_only).rstrip(";").strip()
    sql_clean = _preexecute_sql_safety_pass(sql_clean, question).rstrip(";").strip()

    print("\n[SQL EXEC] Final query sent to MCP:")
    print(sql_clean + ";")

    try:
        payload = await query_db(sql_clean, mcp=mcp)

        if isinstance(payload, dict) and payload.get("success"):
            display_table_from_json(payload)
            return payload

        err = payload.get("error") if isinstance(payload, dict) else str(payload)
        print(f"query_db failed: {err}")
        return None

    except asyncio.TimeoutError:
        print("MCP query_db timeout")
        return None
    except Exception as e:
        print(f"MCP query_db error: {e}")
        return None

async def repl(paths):
    app = build_app()
    transport = StreamableHttpTransport(MCP_BASE)

    async with Client(transport=transport) as mcp:
        first_q = input("Ask me anything about your data: ").strip()
        if not first_q:
            print("No question entered. Exiting.")
            return

        # ---------- FIRST RUN (end-to-end only) ----------
        t0_total = _now()

        init_state = make_state(paths, first_q)
        state = app.invoke(init_state)

        await _exec_via_mcp(
            mcp,
            state.get("answer", ""),
            state.get("final_sql"),
            first_q,
        )

        _log_total("End-to-end total time (first run)", t0_total)
        _print_latency_debug(state)

        print("\nAsk me more questions! (type 'exit' to quit')")

        # ---------- LOOP (end-to-end only) ----------
        while True:
            q = input("\nQuestion> ").strip()
            if not q or q.lower() in {"exit", "quit", "q"}:
                print("Bye.")
                break

            t0_total = _now()

            s2 = make_state(paths, q)
            out = app.invoke(s2)

            await _exec_via_mcp(
                mcp,
                out.get("answer", ""),
                out.get("final_sql"),
                q,
            )

            _log_total("End-to-end total time (this question)", t0_total)
            _print_latency_debug(out)
