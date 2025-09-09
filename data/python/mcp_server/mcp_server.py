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
# Load environment variables
# -----------------------------
load_dotenv()

# -----------------------------
# Setup logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("MCPServer")

# -----------------------------
# Initialize MCP
# -----------------------------
mcp = FastMCP("postgres-mcp")

# -----------------------------
# Database connection config
# -----------------------------
DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", 5432)),
    "database": os.getenv("POSTGRES_DB", "datahub"),
    "user": os.getenv("POSTGRES_USER", "datahub"),
    "password": os.getenv("POSTGRES_PASSWORD", "datahub"),
}

# Preview sizing (server-side guardrails)
PREVIEW_ROW_LIMIT = int(os.getenv("PREVIEW_ROW_LIMIT", "5"))
PREVIEW_COL_LIMIT = int(os.getenv("PREVIEW_COL_LIMIT", "5"))
STATEMENT_TIMEOUT_MS = int(os.getenv("STATEMENT_TIMEOUT_MS", "5000"))  # 5s

# -----------------------------
# Helper: Quote all identifiers (basic)
# -----------------------------
def quote_identifiers(sql: str) -> str:
    """
    Wrap table/column identifiers with double quotes.
    NOTE: This is a best-effort pass using sqlparse. It avoids touching string literals,
    but is still heuristic. If you use case-sensitive identifiers in Postgres, this helps.
    """
    parsed = sqlparse.parse(sql)
    if not parsed:
        return sql
    stmt = parsed[0]

    def process_identifier(identifier: Identifier) -> str:
        # Preserve dotted names: schema.table -> "schema"."table"
        names = [t.value for t in identifier.tokens if t.ttype != Keyword]
        real = identifier.get_real_name()  # last part
        if real:
            parts = identifier.value.split(".")
            quoted_parts = []
            for part in parts:
                p = part.strip().strip('"')
                if p:
                    quoted_parts.append(f'"{p}"')
            return ".".join(quoted_parts)
        return str(identifier)

    def recurse(token_list):
        parts = []
        for token in token_list.tokens:
            if token.is_group:
                if isinstance(token, IdentifierList):
                    sub_parts = [process_identifier(id_) for id_ in token.get_identifiers()]
                    parts.append(", ".join(sub_parts))
                elif isinstance(token, Identifier):
                    parts.append(process_identifier(token))
                else:
                    parts.append(recurse(token))
            else:
                # Keep keywords as-is, avoid uppercasing everything
                parts.append(token.value)
        # collapse whitespace
        return " ".join(p.strip() for p in parts if p and p.strip())

    return recurse(stmt)

# -----------------------------
# Helpers: Query shape / safety
# -----------------------------
def is_single_select(sql: str) -> bool:
    """True if the SQL is a single SELECT statement (no multiple statements)."""
    statements = [s for s in sqlparse.parse(sql) if str(s).strip()]
    if len(statements) != 1:
        return False
    s = statements[0]
    # First meaningful token must be SELECT
    for tok in s.tokens:
        if tok.is_whitespace:
            continue
        if tok.ttype in (Keyword, DML) and tok.value.upper().startswith("SELECT"):
            return True
        # Anything else first = reject
        return False
    return False

def extract_base_query_without_limit(sql: str) -> str:
    """
    Strip trailing LIMIT/OFFSET (if present). Used for true row count.
    """
    sql_clean = sql.strip().rstrip(";").strip()
    sql_no_limit = re.sub(
        r'\s+(LIMIT\s+\d+(\s+OFFSET\s+\d+)?|OFFSET\s+\d+(\s+LIMIT\s+\d+)?)\s*$',
        "",
        sql_clean,
        flags=re.IGNORECASE
    )
    return sql_no_limit.strip()

def wrap_with_limit(sql: str, limit: int) -> str:
    """
    Wrap an arbitrary SELECT as a subquery and apply LIMIT safely.
    """
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
    """
    Suggest close matches for a missing table. Compares against:
    - fully qualified names (schema.table)
    - bare table names
    """
    tables = list_all_tables(conn)
    bare = [t.split(".", 1)[1] if "." in t else t for t in tables]
    # Try both fully-qualified and bare matches
    suggestions = get_close_matches(missing_name, tables, n=max_suggestions, cutoff=0.6)
    if not suggestions:
        suggestions = [f"public.{m}" for m in get_close_matches(missing_name, bare, n=max_suggestions, cutoff=0.6)]
    return tables, suggestions

# -----------------------------
# MCP Tool: List available tables
# -----------------------------
@mcp.tool()
def list_tables() -> dict:
    """
    List all table names in the connected PostgreSQL database.
    """
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

# -----------------------------
# MCP Tool: Execute SQL query
# -----------------------------
@mcp.tool()
def query_db(sql: str) -> dict:
    """
    Execute a SELECT SQL query against PostgreSQL.

    Returns:
      - preview_rows / preview_columns (server-limited to PREVIEW_* constants)
      - total_rows (true count ignoring LIMIT)
      - total_columns
      - _debug_info
    """
    from psycopg2 import errors  # for specific exception types

    logger.info("=== query_db CALLED ===")
    logger.info(f"📥 Input SQL: {sql}")

    if not is_single_select(sql):
        return {"success": False, "error": "Only a single SELECT statement is allowed."}

    start_time = datetime.now()
    conn = None

    try:
        logger.info("🔗 Connecting to database...")
        conn = psycopg2.connect(**DB_CONFIG)
        # Enforce read-only & autocommit (safe for SELECT)
        conn.set_session(readonly=True, autocommit=True)
        logger.info(f"✅ Connected successfully to {DB_CONFIG['database']}")

        with conn.cursor() as setup_cur:
            # Safety timeout to avoid hanging
            setup_cur.execute(f"SET statement_timeout = {STATEMENT_TIMEOUT_MS};")

        # Quote identifiers (best effort)
        quoted_sql = quote_identifiers(sql)
        logger.info(f"🛠 Quoted SQL: {quoted_sql}")

        # Extract base query (for true row count)
        base_query = extract_base_query_without_limit(quoted_sql)
        logger.info(f"📊 Base query (no LIMIT): {base_query}")

        # --- TRUE total rows ---
        total_rows = 0
        try:
            with conn.cursor() as count_cur:
                count_query = f'SELECT COUNT(*) FROM ({base_query}) AS count_subquery'
                logger.info(f"▶ Executing count query: {count_query}")
                count_cur.execute(count_query)
                total_rows = count_cur.fetchone()[0]
                logger.info(f"✅ TRUE total rows: {total_rows}")
        except errors.UndefinedTable as e:
            # Suggest closest tables
            missing = None
            # Try to extract the missing table name from the error message if possible
            m = re.search(r'relation "(.*?)" does not exist', str(e))
            if m:
                missing = m.group(1)
            tables, suggestions = suggest_tables(conn, missing or "", max_suggestions=3)
            logger.warning(f"❌ Table not found for count query: {e}")
            return {
                "success": False,
                "error": f"Table does not exist: {missing or 'unknown'}",
                "suggestions": suggestions,
                "available_tables": tables
            }
        except Exception as count_e:
            logger.error(f"❌ Count query failed: {count_e}")
            return {"success": False, "error": f"Count failed: {str(count_e)}"}

        # --- Main preview query (server-side LIMIT) ---
        try:
            preview_sql = wrap_with_limit(base_query, PREVIEW_ROW_LIMIT)
            with conn.cursor() as data_cur:
                logger.info(f"▶ Executing preview query: {preview_sql}")
                data_cur.execute(preview_sql)
                columns = [d[0] for d in data_cur.description] if data_cur.description else []
                total_cols = len(columns)
                # trim columns for preview
                preview_cols = columns[:PREVIEW_COL_LIMIT]
                preview_rows = data_cur.fetchall()

                # enforce preview col limit
                preview_rows_limited = [
                    list(row[:len(preview_cols)])
                    for row in preview_rows
                ]

        except errors.UndefinedTable as e:
            missing = None
            m = re.search(r'relation "(.*?)" does not exist', str(e))
            if m:
                missing = m.group(1)
            tables, suggestions = suggest_tables(conn, missing or "", max_suggestions=3)
            logger.warning(f"❌ Table not found during main query: {e}")
            return {
                "success": False,
                "error": f"Table does not exist: {missing or 'unknown'}",
                "suggestions": suggestions,
                "available_tables": tables
            }
        except Exception as e:
            logger.error(f"❌ Error executing main query: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

        duration = (datetime.now() - start_time).total_seconds()

        result = {
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

        logger.info("=== FINAL RESULT ===")
        logger.info(f"true_total_rows: {total_rows}")
        logger.info(f"total_columns: {total_cols}")
        logger.info(f"preview_rows: {len(preview_rows_limited)}")
        logger.info("===================")

        return result

    finally:
        if conn:
            conn.close()
            logger.info("🔌 Database connection closed")

# -----------------------------
# Run MCP Server
# -----------------------------
if __name__ == "__main__":
    logger.info("🚀 Starting PostgreSQL MCP Server...")
    mcp.run(transport="http", host="0.0.0.0", port=8000, log_level="info")