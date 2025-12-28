import asyncio
import json
from typing import Any, Optional

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

from .config import MCP_TIMEOUT, MCP_BASE
from .utils import _extract_payload, _ensure_json

async def _mcp_call(
    client: Client,
    tool: str,
    payload: dict | None = None,
    timeout: int = MCP_TIMEOUT,
):
    coro = client.call_tool(tool, payload or {})
    return await asyncio.wait_for(coro, timeout=timeout)

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