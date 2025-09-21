import os
from dotenv import load_dotenv
import pymysql
from fastmcp import FastMCP
import logging
from datetime import datetime
import time
import uuid
import re
import sqlparse
from sqlparse.sql import Identifier, IdentifierList
from sqlparse.tokens import Keyword, DML
from difflib import get_close_matches

# -----------------------------
# Env & logging (minimal tweaks)
# -----------------------------
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("MCPServer")
logging.getLogger("pymysql").setLevel(logging.WARNING)  # keep driver noise down

def _req_id() -> str:
    return uuid.uuid4().hex[:8]

def _elapsed_ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0

def _log_sql(req: str, **parts):
    for k, v in parts.items():
        if v:
            logger.debug(f"[{req}] {k}: {v}")

# -----------------------------
# MCP & DB config (MySQL)
# -----------------------------
mcp = FastMCP("mysql-mcp")

DB_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "port": int(os.getenv("MYSQL_PORT", 3306)),
    "db": os.getenv("MYSQL_DB", "classicmodels"),
    "user": os.getenv("MYSQL_USER", "datahub"),
    "password": os.getenv("MYSQL_PASSWORD", "datahub"),
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.Cursor,  # plain tuples
    "autocommit": True,
}

# Preview sizing (server-side)
PREVIEW_ROW_LIMIT = int(os.getenv("PREVIEW_ROW_LIMIT", "5"))
PREVIEW_COL_LIMIT = int(os.getenv("PREVIEW_COL_LIMIT", "5"))
STATEMENT_TIMEOUT_MS = int(os.getenv("STATEMENT_TIMEOUT_MS", "5000"))  # 5s (MySQL: max_execution_time)
MAX_EXPORT_ROWS = int(os.getenv("MAX_EXPORT_ROWS", "200000"))

# -----------------------------
# SQL helpers (MySQL-flavored)
# -----------------------------
def quote_identifiers(sql: str) -> str:
    """
    Best-effort quoting of identifiers using sqlparse.
    Convert a.b -> `a`.`b`. Leaves literals/keywords untouched.
    """
    parsed = sqlparse.parse(sql)
    if not parsed:
        return sql
    stmt = parsed[0]

    def q(name: str) -> str:
        # strip existing quotes/backticks then wrap with backticks
        clean = name.strip().strip('`').strip('"')
        return f'`{clean}`' if clean else clean

    def process_identifier(identifier: Identifier) -> str:
        parts = [p.strip().strip('`').strip('"') for p in identifier.value.split(".")]
        parts = [q(p) for p in parts if p]
        return ".".join(parts) if parts else str(identifier)

    def recurse(token_list):
        parts = []
        for token in token_list.tokens:
            if token.is_group:
                if isinstance(token, IdentifierList):
                    parts.append(", ".join(process_identifier(i) for i in token.get_identifiers()))
                elif isinstance(token, Identifier):
                    parts.append(process_identifier(token))
                else:
                    parts.append(recurse(token))
            else:
                parts.append(token.value)
        return " ".join(p.strip() for p in parts if p and p.strip())

    return recurse(stmt)

def is_single_select(sql: str) -> bool:
    """True if exactly one statement and it starts with SELECT."""
    statements = [s for s in sqlparse.parse(sql) if str(s).strip()]
    if len(statements) != 1:
        return False
    s = statements[0]
    for tok in s.tokens:
        if tok.is_whitespace:
            continue
        if tok.ttype in (Keyword, DML) and tok.value.upper().startswith("SELECT"):
            return True
        return False
    return False

def extract_base_query_without_limit(sql: str) -> str:
    """Strip trailing LIMIT/OFFSET from SQL."""
    sql_clean = sql.strip().rstrip(";").strip()
    sql_no_limit = re.sub(
        r'\s+(LIMIT\s+\d+(\s+OFFSET\s+\d+)?|OFFSET\s+\d+(\s+LIMIT\s+\d+)?)\s*$',
        "",
        sql_clean,
        flags=re.IGNORECASE
    )
    return sql_no_limit.strip()

def wrap_with_limit(sql: str, limit: int) -> str:
    # MySQL requires alias (AS optional). This works fine:
    return f'SELECT * FROM ({sql}) subq LIMIT {int(limit)}'

def list_all_tables(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_type='BASE TABLE'
              AND table_schema NOT IN ('mysql','information_schema','performance_schema','sys')
            ORDER BY table_schema, table_name
        """)
        return [f"{s}.{t}" for s, t in cur.fetchall()]

def suggest_tables(conn, missing_name: str, max_suggestions: int = 3):
    tables = list_all_tables(conn)
    bare = [t.split(".", 1)[1] if "." in t else t for t in tables]
    suggestions = get_close_matches(missing_name, tables, n=max_suggestions, cutoff=0.6)
    if not suggestions:
        suggestions = [f"{DB_CONFIG.get('db','classicmodels')}.{m}" for m in get_close_matches(missing_name, bare, n=max_suggestions, cutoff=0.6)]
    return tables, suggestions

def set_statement_timeout(conn, timeout_ms: int):
    """
    MySQL 8 has session variable max_execution_time (ms).
    We'll try to set it; if not supported, just continue silently.
    """
    try:
        with conn.cursor() as cur:
            cur.execute(f"SET SESSION max_execution_time = {int(timeout_ms)};")
        logger.debug(f"max_execution_time set to {timeout_ms} ms")
    except Exception as e:
        logger.warning(f"max_execution_time not settable: {e}")

# -----------------------------
# Tools
# -----------------------------
@mcp.tool()
def list_tables() -> dict:
    """List all table names in the connected MySQL database."""
    req = _req_id()
    t0 = time.perf_counter()
    logger.info(f"[{req}] list_tables CALLED host={DB_CONFIG.get('host')} db={DB_CONFIG.get('db')}")
    conn = None
    try:
        conn = pymysql.connect(**DB_CONFIG)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_schema, table_name
                FROM information_schema.tables
                WHERE table_type='BASE TABLE'
                  AND table_schema NOT IN ('mysql','information_schema','performance_schema','sys')
                ORDER BY table_schema, table_name
            """)
            tables = cur.fetchall()
            table_list = [f"{schema}.{name}" for schema, name in tables]
        logger.info(f"[{req}] list_tables DONE total={len(table_list)} elapsed_ms={_elapsed_ms(t0):.1f}")
        return {"success": True, "total_tables": len(table_list), "tables": table_list}
    except Exception as e:
        logger.exception(f"[{req}] ❌ list_tables ERROR elapsed_ms={_elapsed_ms(t0):.1f}")
        return {"success": False, "error": str(e)}
    finally:
        if conn:
            conn.close()

@mcp.tool()
def query_db(sql: str) -> dict:
    """Return a safe server-side 5x5 preview + counts for a SELECT query (MySQL)."""
    req = _req_id()
    t0 = time.perf_counter()
    logger.info(f"[{req}] query_db CALLED host={DB_CONFIG.get('host')} db={DB_CONFIG.get('db')}")
    logger.info(f"[{req}] 📥 Input SQL: {sql}")

    if not is_single_select(sql):
        logger.warning(f"[{req}] rejected: not a single SELECT")
        return {"success": False, "error": "Only a single SELECT statement is allowed."}

    conn = None
    try:
        conn = pymysql.connect(**DB_CONFIG)
        set_statement_timeout(conn, STATEMENT_TIMEOUT_MS)

        quoted_sql = quote_identifiers(sql)
        base_query = extract_base_query_without_limit(quoted_sql)
        count_sql = f"SELECT COUNT(*) FROM ({base_query}) count_subquery"
        _log_sql(req, quoted_query=quoted_sql, base_query=base_query, count_sql=count_sql)

        # true count
        total_rows = 0
        try:
            with conn.cursor() as count_cur:
                count_cur.execute(count_sql)
                total_rows = count_cur.fetchone()[0]
        except pymysql.err.ProgrammingError as e:
            code = e.args[0] if e.args else None
            msg = e.args[1] if len(e.args) > 1 else str(e)
            logger.warning(f"[{req}] count failed code={code} msg={msg}")
            if code == 1146 or "doesn't exist" in msg:
                m = re.search(r"Table '([^']+)' doesn't exist", msg)
                missing = m.group(1) if m else ""
                tables, suggestions = suggest_tables(conn, missing.split('.')[-1] if missing else "", 3)
                logger.info(f"[{req}] suggestions={suggestions}")
                return {
                    "success": False,
                    "error": f"Table does not exist: {missing or 'unknown'}",
                    "suggestions": suggestions,
                    "available_tables": tables
                }
            return {"success": False, "error": f"Count failed: {msg}"}
        except Exception as e:
            logger.warning(f"[{req}] count failed generic: {e}")
            return {"success": False, "error": f"Count failed: {str(e)}"}

        # preview with limit
        preview_sql = wrap_with_limit(base_query, PREVIEW_ROW_LIMIT)
        _log_sql(req, preview_sql=preview_sql)

        with conn.cursor() as data_cur:
            data_cur.execute(preview_sql)
            columns = [d[0] for d in data_cur.description] if data_cur.description else []
            total_cols = len(columns)
            preview_cols = columns[:PREVIEW_COL_LIMIT]
            preview_rows = data_cur.fetchall()
            preview_rows_limited = [list(row[:len(preview_cols)]) for row in preview_rows]

        logger.info(
            f"[{req}] query_db DONE rows_preview={len(preview_rows_limited)} "
            f"cols_total={total_cols} rows_total={total_rows} elapsed_ms={_elapsed_ms(t0):.1f}"
        )

        duration = (datetime.now() - datetime.now()).total_seconds()  # kept for compatibility (unused)
        return {
            "success": True,
            "preview_rows": preview_rows_limited,
            "preview_columns": preview_cols,
            "total_rows": total_rows,
            "total_columns": total_cols,
            "duration_seconds": _elapsed_ms(t0) / 1000.0,
            "_debug_info": {
                "original_query": sql,
                "quoted_query": quoted_sql,
                "base_query": base_query,
                "preview_sql": preview_sql,
                "query_returned_rows": len(preview_rows_limited),
                "true_total_rows": total_rows,
            },
        }
    except Exception as e:
        logger.exception(f"[{req}] ❌ query_db ERROR elapsed_ms={_elapsed_ms(t0):.1f}")
        return {"success": False, "error": str(e)}
    finally:
        if conn:
            conn.close()

@mcp.tool()
def query_db_full(sql: str) -> dict:
    """
    Execute the SELECT (without LIMIT) and return ALL rows (capped by MAX_EXPORT_ROWS).
    Intended for file export only—do not print these rows on client. (MySQL)
    """
    req = _req_id()
    t0 = time.perf_counter()
    logger.info(f"[{req}] query_db_full CALLED host={DB_CONFIG.get('host')} db={DB_CONFIG.get('db')}")
    logger.info(f"[{req}] 📥 Input SQL: {sql}")

    if not is_single_select(sql):
        logger.warning(f"[{req}] rejected: not a single SELECT")
        return {"success": False, "error": "Only a single SELECT statement is allowed."}

    conn = None
    try:
        conn = pymysql.connect(**DB_CONFIG)
        set_statement_timeout(conn, STATEMENT_TIMEOUT_MS)

        quoted_sql = quote_identifiers(sql)
        base_query = extract_base_query_without_limit(quoted_sql)
        _log_sql(req, quoted_query=quoted_sql, base_query=base_query)

        with conn.cursor() as data_cur:
            data_cur.execute(base_query)
            columns = [d[0] for d in data_cur.description] if data_cur.description else []
            rows = data_cur.fetchall()

        total_rows = len(rows)
        truncated = False
        if total_rows > MAX_EXPORT_ROWS:
            rows = rows[:MAX_EXPORT_ROWS]
            truncated = True

        logger.info(
            f"[{req}] query_db_full DONE rows_total={total_rows} truncated={truncated} "
            f"elapsed_ms={_elapsed_ms(t0):.1f}"
        )

        return {
            "success": True,
            "columns": columns,
            "rows": [list(r) for r in rows],
            "total_rows": total_rows,
            "truncated": truncated,
            "_debug_info": {
                "original_query": sql,
                "quoted_query": quoted_sql,
                "base_query": base_query,
                "max_export_rows": MAX_EXPORT_ROWS,
                "duration_seconds": _elapsed_ms(t0) / 1000.0
            }
        }
    except pymysql.err.ProgrammingError as e:
        code = e.args[0] if e.args else None
        msg = e.args[1] if len(e.args) > 1 else str(e)
        logger.warning(f"[{req}] ProgrammingError code={code} msg={msg}")
        if code == 1146 or "doesn't exist" in msg:
            m = re.search(r"Table '([^']+)' doesn't exist", msg)
            missing = m.group(1) if m else ""
            tables, suggestions = ([], [])
            try:
                tables, suggestions = suggest_tables(conn, missing.split('.')[-1] if missing else "", 3)
            except Exception:
                pass
            return {
                "success": False,
                "error": f"Table does not exist: {missing or 'unknown'}",
                "suggestions": suggestions,
                "available_tables": tables
            }
        logger.error("❌ Error exporting full query", exc_info=True)
        return {"success": False, "error": msg}
    except Exception as e:
        logger.exception(f"[{req}] ❌ query_db_full ERROR elapsed_ms={_elapsed_ms(t0):.1f}")
        return {"success": False, "error": str(e)}
    finally:
        if conn:
            conn.close()

# -----------------------------
# Run MCP Server (unchanged)
# -----------------------------
if __name__ == "__main__":
    logger.info("🚀 Starting MySQL MCP Server...")
    mcp.run(transport="http", host="0.0.0.0", port=8000, log_level="info")