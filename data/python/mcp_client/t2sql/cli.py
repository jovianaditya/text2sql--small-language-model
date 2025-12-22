import asyncio
from pathlib import Path
from time import perf_counter
import json
from langgraph.graph import StateGraph, END

# MCP client transport
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

# Modular function in file import
from .config import (
    JSON_INPUT,
    CHROMA_DIR,
    COLLECTION_NAME,
    MCP_BASE,
)
from .state import RAGState
from .chroma_embedding import build_vector_store
from .ingestion import resolve_jsonl_paths, ingest_node
from .retrieval import retrieve_node
from .agents import draft_sql_agent, refine_sql_agent
from .mcp_execution import _mcp_call, _extract_payload, _ensure_json, display_table_from_json
from .utils import _now, _log_total, _print_banner

def build_app():
    g = StateGraph(RAGState)

    g.add_node("ingest", ingest_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("draft_sql", draft_sql_agent)
    g.add_node("refine_sql", refine_sql_agent)

    g.set_entry_point("ingest")
    g.add_edge("ingest", "retrieve")
    g.add_edge("retrieve", "draft_sql")
    g.add_edge("draft_sql", "refine_sql")
    g.add_edge("refine_sql", END)

    return g.compile()


async def repl(paths):
    app = build_app()

    transport = StreamableHttpTransport(MCP_BASE)
    async with Client(transport=transport) as mcp:
        first_q = input("Ask me anything about your data: ").strip()
        if not first_q:
            print("No question entered. Exiting.")
            return

        t0_total = _now()

        init: RAGState = {
            "jsonl_paths": paths,
            "question": first_q,
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

        state = app.invoke(init)

        col = build_vector_store()
        total = col.count()
        line = _print_banner("Collection Ingestion Report")
        print(f"Added (first run): {state.get('added', 0)}")
        print(f"Updated (upserted): {state.get('updated', 0)}")
        print(f"Skipped (unchanged): {state.get('skipped', 0)}")
        print(f"Total docs now in collection: {total}")
        print(f"Chroma path: {Path(CHROMA_DIR).resolve()} | Collection: {COLLECTION_NAME}")
        print(f"{line}\n")

        async def exec_via_mcp(sql_block: str, final_sql: str | None = None):
            sql_only = sql_block.split("\n\n-- notes:")[0].strip()
            sql_clean = (final_sql or sql_only).rstrip(";").strip()

            print("\n[SQL EXEC] Final query sent to MCP:")
            print(sql_clean + ";")

            try:
                res = await _mcp_call(mcp, "query_db", {"sql": sql_clean})
                payload = _ensure_json(_extract_payload(res))
                if isinstance(payload, dict) and payload.get("success"):
                    display_table_from_json(payload)
                    return payload
                else:
                    err = payload.get("error") if isinstance(payload, dict) else str(payload)
                    print(f"query_db failed: {err}")
                    return None
            except asyncio.TimeoutError:
                print("MCP query_db timeout")
                return None
            except Exception as e:
                print(f"MCP query_db error: {e}")
                return None

        await exec_via_mcp(state.get("answer", ""), state.get("final_sql"))

        _log_total("End-to-end total time (first run)", t0_total)

        print("\nAsk me more questions! (type 'exit' to quit')")
        while True:
            q = input("\nQuestion> ").strip()
            if not q or q.lower() in {"exit", "quit", "q"}:
                print("Bye.")
                break

            t0_total = _now()

            state2: RAGState = {
                "jsonl_paths": paths,
                "question": q,
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

            state2 = app.invoke(state2)
            await exec_via_mcp(state2.get("answer", ""), state2.get("final_sql"))
            _log_total("End-to-end total time (this question)", t0_total)


def main():
    paths = resolve_jsonl_paths(JSON_INPUT)
    if not paths:
        raise RuntimeError(
            "No JSONL inputs found.\n"
            "Set EXPORT_JSONL_OUTPUT to one of:\n"
            "  • a single file: ./exports/classicmodels.jsonl\n"
            "  • a directory:   ./datahub_metadata\n"
            "  • a glob:        ./datahub_metadata/*.jsonl\n"
        )

    try:
        asyncio.run(repl(paths))
    except RuntimeError as e:
        if "asyncio.run() cannot be called" in str(e):
            import nest_asyncio
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            loop.run_until_complete(repl(paths))
        else:
            raise


if __name__ == "__main__":
    main()