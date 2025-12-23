# repl.py
import asyncio
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

from .config import MCP_BASE, CHROMA_DIR, COLLECTION_NAME
from .state import RAGState
from .graph import build_app
from .chroma_embedding import build_vector_store
from .mcp_execution import _mcp_call, _extract_payload, _ensure_json, query_db
from .utils import _now, _log_total, _print_banner
from tabulate import tabulate

def _make_state(paths, question: str) -> RAGState:
    # Kalau kamu set state.py menjadi total=False, ini bisa diperkecil
    return {
        "jsonl_paths": paths,
        "question": question,
        "added": 0,
        "updated": 0,
        "skipped": 0,
        "context": "",
        "answer": "",
        "draft_sql": "",
        "final_sql": "",
        "qtype": "",
        "requires_regex": False,
        "notes": "",
        "validation_errors": [],
        "critic_feedback": "",
    }

async def _exec_via_mcp(mcp: Client, sql_block: str, final_sql: str | None = None):
    # Ambil SQL saja (kalau answer berisi SQL + notes)
    sql_only = sql_block.split("\n\n-- notes:")[0].strip()
    sql_clean = (final_sql or sql_only).rstrip(";").strip()

    print("\n[SQL EXEC] Final query sent to MCP:")
    print(sql_clean + ";")

    try:
        res = await query_db(sql_clean, mcp=mcp)
        payload = res

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

        t0_total = _now()
        init_state = _make_state(paths, first_q)
        state = app.invoke(init_state)

        # Reporting ingestion status
        col = build_vector_store()
        total = col.count()
        line = _print_banner("Collection Ingestion Report")
        print(f"Added (first run): {state.get('added', 0)}")
        print(f"Updated (upserted): {state.get('updated', 0)}")
        print(f"Skipped (unchanged): {state.get('skipped', 0)}")
        print(f"Total docs now in collection: {total}")
        print(f"Chroma path: {Path(CHROMA_DIR).resolve()} | Collection: {COLLECTION_NAME}")
        print(f"{line}\n")

        await _exec_via_mcp(mcp, state.get("answer", ""), state.get("final_sql"))
        _log_total("End-to-end total time (first run)", t0_total)

        print("\nAsk me more questions! (type 'exit' to quit')")

        while True:
            q = input("\nQuestion> ").strip()
            if not q or q.lower() in {"exit", "quit", "q"}:
                print("Bye.")
                break

            t0_total = _now()
            s2 = _make_state(paths, q)
            out = app.invoke(s2)
            await _exec_via_mcp(mcp, out.get("answer", ""), out.get("final_sql"))
            _log_total("End-to-end total time (this question)", t0_total)

def display_table_from_json(data, max_cell_len: int = 80):
    preview_rows = data.get("preview_rows", [])
    preview_cols = data.get("preview_columns", [])
    total_rows = data.get("total_rows", len(preview_rows))
    total_cols = data.get("total_columns", len(preview_cols))

    def _shorten(val, max_len: int) -> str:
        if not isinstance(val, str):
            return val
        s = val.replace("\n", " ").strip()
        if len(s) <= max_len:
            return s
        return s[: max_len - 3] + "..."

    truncated_rows = [[_shorten(cell, max_cell_len) for cell in row] for row in preview_rows]

    if truncated_rows and preview_cols:
        print("\nQuery Result (preview):")
        print(
            tabulate(
                truncated_rows,
                headers=preview_cols,
                tablefmt="github",
                showindex=range(1, len(truncated_rows) + 1),
            )
        )
        print(
            f"\nShowing preview: {len(preview_rows)} rows × {len(preview_cols)} columns "
            f"(full result: {total_rows} rows × {total_cols} columns)"
        )
    else:
        print("Query executed but returned no rows.")