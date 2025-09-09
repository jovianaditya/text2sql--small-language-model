import os
from dotenv import load_dotenv
import psycopg2
from fastmcp import FastMCP
import logging
from datetime import datetime
import re
import sqlparse
from sqlparse.sql import Identifier, IdentifierList
from sqlparse.tokens import Keyword, DML
from difflib import get_close_matches

# -----------------------------
# Env & logging
# -----------------------------
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("MCPServer")

# -----------------------------
# MCP & DB config
# -----------------------------
mcp = FastMCP("postgres-mcp")

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", 5432)),
    "database": os.getenv("POSTGRES_DB", "datahub"),
    "user": os.getenv("POSTGRES_USER", "datahub"),
    "password": os.getenv("POSTGRES_PASSWORD", "datahub"),
}

# Preview sizing (server-side)
PREVIEW_ROW_LIMIT = int(os.getenv("PREVIEW_ROW_LIMIT", "5"))
PREVIEW_COL_LIMIT = int(os.getenv("PREVIEW_COL_LIMIT", "5"))
STATEMENT_TIMEOUT_MS = int(os.getenv("STATEMENT_TIMEOUT_MS", "5000"))  # 5s
MAX_EXPORT_ROWS = int(os.getenv("MAX_EXPORT_ROWS", "200000"))

# -----------------------------
# SQL helpers
# -----------------------------
def quote_identifiers(sql: str) -> str:
    """Best-effort quoting of identifiers using sqlparse."""
    parsed = sqlparse.parse(sql)
    if not parsed:
        return sql
    stmt = parsed[0]

    def process_identifier(identifier: Identifier) -> str:
        parts = [p.strip().strip('"') for p in identifier.value.split(".")]
        parts = [f'"{p}"' for p in parts if p]
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
    return f'SELECT * FROM ({sql}) AS subq LIMIT {int(limit)}'

def list_all_tables(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_type='BASE TABLE' AND table_schema NOT IN ('pg_catalog', 'information_schema')
            ORDER BY table_schema, table_name
        """)
        return [f"{s}.{t}" for s, t in cur.fetchall()]

def suggest_tables(conn, missing_name: str, max_suggestions: int = 3):
    tables = list_all_tables(conn)
    bare = [t.split(".", 1)[1] if "." in t else t for t in tables]
    suggestions = get_close_matches(missing_name, tables, n=max_suggestions, cutoff=0.6)
    if not suggestions:
        suggestions = [f"public.{m}" for m in get_close_matches(missing_name, bare, n=max_suggestions, cutoff=0.6)]
    return tables, suggestions

# -----------------------------
# Tools
# -----------------------------
@mcp.tool()
def list_tables() -> dict:
    """List all table names in the connected PostgreSQL database."""
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_schema, table_name
                FROM information_schema.tables
                WHERE table_type='BASE TABLE' AND table_schema NOT IN ('pg_catalog', 'information_schema')
                ORDER BY table_schema, table_name
            """)
            tables = cur.fetchall()
            table_list = [f"{schema}.{name}" for schema, name in tables]
        return {"success": True, "total_tables": len(table_list), "tables": table_list}
    except Exception as e:
        logger.error(f"❌ Error listing tables: {e}", exc_info=True)
        return {"success": False, "error": str(e)}
    finally:
        if conn:
            conn.close()

@mcp.tool()
def query_db(sql: str) -> dict:
    """Return a safe server-side 5x5 preview + counts for a SELECT query."""
    from psycopg2 import errors
    logger.info("=== query_db CALLED ===")
    logger.info(f"📥 Input SQL: {sql}")

    if not is_single_select(sql):
        return {"success": False, "error": "Only a single SELECT statement is allowed."}

    start_time = datetime.now()
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.set_session(readonly=True, autocommit=True)
        with conn.cursor() as setup_cur:
            setup_cur.execute(f"SET statement_timeout = {STATEMENT_TIMEOUT_MS};")

        quoted_sql = quote_identifiers(sql)
        base_query = extract_base_query_without_limit(quoted_sql)

        # true count
        total_rows = 0
        try:
            with conn.cursor() as count_cur:
                count_cur.execute(f'SELECT COUNT(*) FROM ({base_query}) AS count_subquery')
                total_rows = count_cur.fetchone()[0]
        except errors.UndefinedTable as e:
            m = re.search(r'relation "(.*?)" does not exist', str(e))
            missing = m.group(1) if m else ""
            tables, suggestions = suggest_tables(conn, missing, 3)
            return {
                "success": False,
                "error": f"Table does not exist: {missing or 'unknown'}",
                "suggestions": suggestions,
                "available_tables": tables
            }
        except Exception as e:
            return {"success": False, "error": f"Count failed: {str(e)}"}

        # preview with limit
        preview_sql = wrap_with_limit(base_query, PREVIEW_ROW_LIMIT)
        with conn.cursor() as data_cur:
            data_cur.execute(preview_sql)
            columns = [d[0] for d in data_cur.description] if data_cur.description else []
            total_cols = len(columns)
            preview_cols = columns[:PREVIEW_COL_LIMIT]
            preview_rows = data_cur.fetchall()
            preview_rows_limited = [list(row[:len(preview_cols)]) for row in preview_rows]

        duration = (datetime.now() - start_time).total_seconds()
        return {
            "success": True,
            "preview_rows": preview_rows_limited,
            "preview_columns": preview_cols,
            "total_rows": total_rows,
            "total_columns": total_cols,
            "duration_seconds": duration,
            "_debug_info": {
                "original_query": sql,
                "quoted_query": quoted_sql,
                "base_query": base_query,
                "preview_sql": preview_sql,
                "query_returned_rows": len(preview_rows_limited),
                "true_total_rows": total_rows,
            },
        }
    finally:
        if conn:
            conn.close()

@mcp.tool()
def query_db_full(sql: str) -> dict:
    """
    Execute the SELECT (without LIMIT) and return ALL rows (capped by MAX_EXPORT_ROWS).
    Intended for file export only—do not print these rows on client.
    """
    from psycopg2 import errors
    logger.info("=== query_db_full CALLED ===")
    logger.info(f"📥 Input SQL: {sql}")

    if not is_single_select(sql):
        return {"success": False, "error": "Only a single SELECT statement is allowed."}

    start_time = datetime.now()
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.set_session(readonly=True, autocommit=True)
        with conn.cursor() as setup_cur:
            setup_cur.execute(f"SET statement_timeout = {STATEMENT_TIMEOUT_MS};")

        quoted_sql = quote_identifiers(sql)
        base_query = extract_base_query_without_limit(quoted_sql)

        with conn.cursor() as data_cur:
            data_cur.execute(base_query)
            columns = [d[0] for d in data_cur.description] if data_cur.description else []
            rows = data_cur.fetchall()

        total_rows = len(rows)
        if total_rows > MAX_EXPORT_ROWS:
            rows = rows[:MAX_EXPORT_ROWS]
            truncated = True
        else:
            truncated = False

        duration = (datetime.now() - start_time).total_seconds()
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
                "duration_seconds": duration
            }
        }
    except errors.UndefinedTable as e:
        m = re.search(r'relation "(.*?)" does not exist', str(e))
        missing = m.group(1) if m else ""
        tables, suggestions = suggest_tables(conn, missing, 3) if conn else ([], [])
        return {
            "success": False,
            "error": f"Table does not exist: {missing or 'unknown'}",
            "suggestions": suggestions,
            "available_tables": tables
        }
    except Exception as e:
        logger.error("❌ Error exporting full query", exc_info=True)
        return {"success": False, "error": str(e)}
    finally:
        if conn:
            conn.close()

# -----------------------------
# Run MCP Server
# -----------------------------
if __name__ == "__main__":
    logger.info("🚀 Starting PostgreSQL MCP Server...")
    mcp.run(transport="http", host="0.0.0.0", port=8000, log_level="info")