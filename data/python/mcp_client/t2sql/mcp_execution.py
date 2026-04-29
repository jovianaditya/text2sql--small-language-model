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


def _error_text_from_payload(payload: dict | str | object) -> str:
    if isinstance(payload, dict):
        for key in ("error", "message", "detail", "reason"):
            val = str(payload.get(key) or "").strip()
            if val:
                return val
        raw_val = payload.get("raw")
        if raw_val is not None:
            raw_txt = str(raw_val).strip()
            if raw_txt:
                return raw_txt
    return str(payload or "").strip()


def _classify_execution_error(message: str) -> str:
    msg = str(message or "").lower()
    if not msg:
        return "unknown"
    if any(tok in msg for tok in ("timeout", "timed out", "deadline exceeded")):
        return "timeout"
    if any(tok in msg for tok in ("429", "rate limit", "too many requests")):
        return "rate_limit"
    if any(
        tok in msg
        for tok in (
            "connection refused",
            "connection reset",
            "connection error",
            "failed to connect",
            "name or service not known",
            "no route to host",
            "network is unreachable",
            "transport error",
        )
    ):
        return "connection"
    if any(tok in msg for tok in ("unknown database", "database does not exist", "no such database")):
        return "database_not_found"
    if any(tok in msg for tok in ("access denied", "permission denied", "not authorized", "insufficient privilege")):
        return "permission"
    if any(
        tok in msg
        for tok in (
            "syntax error",
            "parse error",
            "unknown column",
            "unknown table",
            "doesn't exist",
            "does not exist",
            "invalid identifier",
            "column not found",
        )
    ):
        return "syntax_or_validation"
    return "execution_runtime"


def _fallback_hint_for_error(error_type: str) -> str:
    et = str(error_type or "").strip().lower()
    if et == "timeout":
        return "retry_with_simpler_query_or_higher_timeout"
    if et == "rate_limit":
        return "retry_with_backoff"
    if et == "connection":
        return "check_mcp_server_connectivity"
    if et == "database_not_found":
        return "verify_target_database_name"
    if et == "permission":
        return "verify_database_permissions"
    if et == "syntax_or_validation":
        return "regenerate_sql_with_schema_constraints"
    if et == "execution_runtime":
        return "inspect_query_logic_and_data_types"
    return "inspect_mcp_payload"


def _normalize_failure_payload(payload: dict | str | object) -> dict:
    base: dict
    if isinstance(payload, dict):
        base = dict(payload)
    else:
        base = {"raw": payload}

    message = _error_text_from_payload(base)
    error_type = str(base.get("error_type") or "").strip().lower()
    if not error_type:
        error_type = _classify_execution_error(message)

    base["success"] = False
    base["error"] = message or "Unknown MCP execution error"
    base["error_type"] = error_type
    base["error_category"] = error_type
    base["retryable"] = error_type in {"timeout", "rate_limit", "connection"}
    base["fallback_hint"] = _fallback_hint_for_error(error_type)
    return base

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
        try:
            result = await _mcp_call(client, "query_db", payload)
            payload = _extract_payload(result)
            as_json = _ensure_json(payload)
            out = as_json if as_json is not None else {"raw": payload}
            if isinstance(out, dict) and out.get("success"):
                return out
            return _normalize_failure_payload(out)
        except asyncio.TimeoutError:
            return _normalize_failure_payload({"error": "MCP query_db timeout", "error_type": "timeout"})
        except Exception as exc:
            return _normalize_failure_payload({"error": str(exc)})

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
        try:
            result = await _mcp_call(client, "query_db_full", payload)
            payload = _extract_payload(result)
            as_json = _ensure_json(payload)
            out = as_json if as_json is not None else {"raw": payload}
            if isinstance(out, dict) and out.get("success"):
                return out
            return _normalize_failure_payload(out)
        except asyncio.TimeoutError:
            return _normalize_failure_payload({"error": "MCP query_db_full timeout", "error_type": "timeout"})
        except Exception as exc:
            return _normalize_failure_payload({"error": str(exc)})

    if mcp is not None:
        return await _do_call(mcp)

    transport = StreamableHttpTransport(MCP_BASE)
    async with Client(transport=transport) as client:
        return await _do_call(client)
