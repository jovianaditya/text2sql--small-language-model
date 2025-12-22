import asyncio
import json
from typing import Any, Optional
from tabulate import tabulate
from fastmcp import Client
from .config import MCP_TIMEOUT

async def _mcp_call(client: Client, tool: str, payload: dict | None = None, timeout: int = MCP_TIMEOUT):
    coro = client.call_tool(tool, payload or {})
    return await asyncio.wait_for(coro, timeout=timeout)

def _extract_payload(result) -> Any:
    if hasattr(result, 'data') and result.data is not None:
        return result.data
    if hasattr(result, "structuredContent") and result.structuredContent is not None:
        return result.structuredContent
    if hasattr(result, "content") and result.content:
        return result.content[0].text if isinstance(result.content, list) else str(result.content)
    return result

def _ensure_json(d: Any) -> Optional[dict]:
    if isinstance(d, dict):
        return d
    if isinstance(d, str):
        try:
            return json.loads(d)
        except Exception:
            return None
    return None

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