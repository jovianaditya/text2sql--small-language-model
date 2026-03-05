# ======================================================================
# =========================== mcp_execution.py =========================
# ======================================================================
# Purpose:
# - MCP client functions to execute SQL queries via MCP tools.

# ======================================================================
# ====================== STANDARD LIBRARY IMPORT =======================
# ======================================================================
import asyncio
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
# ======================================================================
# ======================== LOCAL MODULE IMPORT =========================
# ======================================================================
from .utils import _extract_payload, _ensure_json
from .config import MCP_TIMEOUT, MCP_BASE

async def _mcp_call(
    client: Client,
    tool: str,
    payload: dict | None = None,
    timeout: int = MCP_TIMEOUT,
):
    coro = client.call_tool(tool, payload or {})
    return await asyncio.wait_for(coro, timeout=timeout)

async def query_db(
    sql: str,
    mcp: Client | None = None,
    database: str | None = None,
) -> dict:
    """
    Panggil tool `query_db` pada MCP server.
    (Tanpa stage timing; end-to-end diukur di repl.py)
    NOTE: Ini pakai LIMIT otomatis dari server (preview mode).
    """
    sql_clean = (sql or "").strip().rstrip(";")

    async def _do_call(client: Client) -> dict:
        payload = {"sql": sql_clean}
        if database:
            payload["database"] = str(database).strip()
        result = await _mcp_call(client, "query_db", payload)

        payload = _extract_payload(result)
        as_json = _ensure_json(payload)

        return as_json if as_json is not None else {"raw": payload}

    if mcp is not None:
        return await _do_call(mcp)

    transport = StreamableHttpTransport(MCP_BASE)
    async with Client(transport=transport) as client:
        return await _do_call(client)

async def query_db_full(
    sql: str,
    mcp: Client | None = None,
    database: str | None = None,
) -> dict:
    """
    Panggil tool `query_db_full` pada MCP server.
    Mengembalikan SEMUA rows tanpa LIMIT (untuk evaluation/export).
    """
    sql_clean = (sql or "").strip().rstrip(";")

    async def _do_call(client: Client) -> dict:
        payload = {"sql": sql_clean}
        if database:
            payload["database"] = str(database).strip()
        result = await _mcp_call(client, "query_db_full", payload)

        payload = _extract_payload(result)
        as_json = _ensure_json(payload)

        return as_json if as_json is not None else {"raw": payload}

    if mcp is not None:
        return await _do_call(mcp)

    transport = StreamableHttpTransport(MCP_BASE)
    async with Client(transport=transport) as client:
        return await _do_call(client)
