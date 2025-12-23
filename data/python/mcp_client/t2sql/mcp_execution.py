# t2sql/mcp_execution.py
import asyncio
import json
from typing import Any, Optional

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

from .config import MCP_TIMEOUT, MCP_BASE


async def _mcp_call(
    client: Client,
    tool: str,
    payload: dict | None = None,
    timeout: int = MCP_TIMEOUT,
):
    coro = client.call_tool(tool, payload or {})
    return await asyncio.wait_for(coro, timeout=timeout)


def _extract_payload(result) -> Any:
    if hasattr(result, "data") and result.data is not None:
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


async def query_db(sql: str, mcp: Client | None = None) -> dict:
    """
    Panggil tool `query_db` pada MCP server.

    - Jika `mcp` diberikan (Client yang sudah dibuat di repl.py), fungsi tidak membuat koneksi baru.
    - Jika `mcp` None, fungsi akan membuat Client sendiri memakai MCP_BASE.
    - Server kamu menerima key payload: {"sql": "..."} (bukan "query").
    """
    sql_clean = (sql or "").strip().rstrip(";")

    async def _do_call(client: Client) -> dict:
        result = await _mcp_call(client, "query_db", {"sql": sql_clean})
        payload = _extract_payload(result)
        as_json = _ensure_json(payload)
        return as_json if as_json is not None else {"raw": payload}

    if mcp is not None:
        return await _do_call(mcp)

    transport = StreamableHttpTransport(MCP_BASE)
    async with Client(transport=transport) as client:
        return await _do_call(client)