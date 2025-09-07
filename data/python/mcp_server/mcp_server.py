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

# -----------------------------
# Helper: Quote all identifiers
# -----------------------------
def quote_identifiers(sql: str) -> str:
    """
    Wrap all table and column names in double quotes.
    Example:
      SELECT emp_id, first_name FROM employees
      -> SELECT "emp_id", "first_name" FROM "employees"
    """
    parsed = sqlparse.parse(sql)
    if not parsed:
        return sql

    stmt = parsed[0]

    def process_identifier(identifier: Identifier) -> str:
        name = identifier.get_real_name()
        if name:
            return f'"{name}"'
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
                if token.ttype in (Keyword, DML):
                    parts.append(token.value.upper())
                else:
                    parts.append(token.value)
        return " ".join(p.strip() for p in parts if p.strip())

    return recurse(stmt)

# -----------------------------
# Helper: Strip LIMIT/OFFSET
# -----------------------------
def extract_base_query_without_limit(sql: str) -> str:
    """
    Extract the base query without LIMIT/OFFSET clauses
    for computing true row counts.
    """
    sql_clean = sql.strip().rstrip(";").strip()
    sql_no_limit = re.sub(
        r'\s+(LIMIT\s+\d+(\s+OFFSET\s+\d+)?|OFFSET\s+\d+(\s+LIMIT\s+\d+)?)\s*$',
        "",
        sql_clean,
        flags=re.IGNORECASE
    )
    return sql_no_limit.strip()

# -----------------------------
# MCP Tool: Execute SQL query
# -----------------------------
@mcp.tool()
def query_db(sql: str) -> dict:
    """
    Execute a SELECT SQL query against PostgreSQL.
    Returns:
      - preview (5 rows x 5 cols)
      - true total rows (ignoring LIMIT)
      - total columns
    """
    logger.info("=== query_db CALLED ===")
    logger.info(f"📥 Input SQL: {sql}")

    if not sql.strip().upper().startswith("SELECT"):
        return {"success": False, "error": "Only SELECT queries are allowed."}

    start_time = datetime.now()
    conn = None

    try:
        logger.info("🔗 Connecting to database...")
        conn = psycopg2.connect(**DB_CONFIG)
        logger.info(f"✅ Connected successfully to {DB_CONFIG['database']}")

        # Quote identifiers
        quoted_sql = quote_identifiers(sql)
        logger.info(f"🛠 Quoted SQL: {quoted_sql}")

        # Extract base query (for true row count)
        base_query = extract_base_query_without_limit(quoted_sql)
        logger.info(f"📊 Base query (no LIMIT): {base_query}")

        # --- TRUE total rows ---
        total_rows = 0
        with conn.cursor() as count_cur:
            count_query = f'SELECT COUNT(*) FROM ({base_query}) AS count_subquery'
            logger.info(f"▶ Executing count query: {count_query}")
            try:
                count_cur.execute(count_query)
                count_result = count_cur.fetchone()
                total_rows = count_result[0] if count_result else 0
                logger.info(f"✅ TRUE total rows: {total_rows}")
            except Exception as count_e:
                logger.error(f"❌ Count query failed: {count_e}")

        # --- Main query ---
        with conn.cursor() as data_cur:
            logger.info(f"▶ Executing quoted query: {quoted_sql}")
            data_cur.execute(quoted_sql)

            all_columns = [desc[0] for desc in data_cur.description]
            total_cols = len(all_columns)
            logger.info(f"📑 Columns ({total_cols}): {all_columns}")

            all_result_rows = data_cur.fetchall()
            logger.info(f"📦 Query returned {len(all_result_rows)} rows")

            # Preview (5x5)
            preview_rows = all_result_rows[:5]
            preview_cols = all_columns[:5]
            preview_rows_limited = [list(row[:5]) for row in preview_rows]

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
                "query_returned_rows": len(all_result_rows),
                "true_total_rows": total_rows,
            },
        }

        logger.info("=== FINAL RESULT ===")
        logger.info(f"true_total_rows: {total_rows}")
        logger.info(f"total_columns: {total_cols}")
        logger.info(f"preview_rows: {len(preview_rows_limited)}")
        logger.info("===================")

        return result

    except Exception as e:
        logger.error(f"❌ Error executing query: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

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