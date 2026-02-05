# import asyncio
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

from .config import MCP_BASE, CHROMA_DIR, COLLECTION_NAME
from .state import make_state
from .graph import build_app
from .chroma_embedding import build_vector_store
from .mcp_execution import query_db
from .utils import _now, _log_total, _print_banner, display_table_from_json


async def _exec_via_mcp(
    mcp: Client,
    sql_block: str,
    final_sql: str | None = None,
):
    sql_only = (sql_block or "").split("\n\n-- notes:")[0].strip()
    sql_clean = (final_sql or sql_only).rstrip(";").strip()

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

        await _exec_via_mcp(mcp, state.get("answer", ""), state.get("final_sql"))

        _log_total("End-to-end total time (first run)", t0_total)

        # 🆕 Log processing time (Model ready → Final SQL) for thesis benchmark
        processing_time = state.get("processing_time")
        if processing_time:
            print(f"[BENCHMARK] Processing latency (Model ready → Final SQL): {processing_time:.3f}s")

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

            await _exec_via_mcp(mcp, out.get("answer", ""), out.get("final_sql"))

            _log_total("End-to-end total time (this question)", t0_total)

            # 🆕 Log processing time (Model ready → Final SQL) for thesis benchmark
            processing_time = out.get("processing_time")
            if processing_time:
                print(f"[BENCHMARK] Processing latency (Model ready → Final SQL): {processing_time:.3f}s")