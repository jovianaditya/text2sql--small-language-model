import os
from dotenv import load_dotenv
import psycopg2
from fastmcp import FastMCP
import logging
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# -----------------------------
# Setup logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------
# Create MCP server
# -----------------------------
mcp = FastMCP("postgres-mcp")

# -----------------------------
# Database connection config
# -----------------------------
DB_CONFIG = {
    "host": os.environ.get("POSTGRES_HOST", "localhost"),
    "port": int(os.environ.get("POSTGRES_PORT", 5432)),
    "database": os.environ.get("POSTGRES_DB", "datahub"),
    "user": os.environ.get("POSTGRES_USER", "datahub"),
    "password": os.environ.get("POSTGRES_PASSWORD", "datahub"),
}

# -----------------------------
# MCP Tool: Execute SQL query
# -----------------------------
@mcp.tool()
def query_db(sql: str) -> dict:
    """
    Execute a SELECT SQL query.
    Return a limited preview (5 rows x 5 columns) but include total rows/columns info.
    """
    sql_stripped = sql.strip().upper()
    if not sql_stripped.startswith("SELECT"):
        return {"success": False, "error": "Only SELECT queries are allowed."}

    start_time = datetime.now()
    conn = None

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Get total row count
        cursor.execute(f"SELECT COUNT(*) FROM ({sql}) AS t")
        total_rows = cursor.fetchone()[0]

        # Execute the actual query
        cursor.execute(sql)

        # Fetch preview rows (5 max)
        preview_rows = cursor.fetchmany(5)
        # Limit preview columns (5 max)
        preview_cols = [desc[0] for desc in cursor.description[:5]]
        preview_rows = [row[:5] for row in preview_rows]

        # Total columns
        total_cols = len(cursor.description)

        duration = (datetime.now() - start_time).total_seconds()

        return {
            "success": True,
            "preview_rows": preview_rows,
            "preview_columns": preview_cols,
            "total_rows": total_rows,
            "total_columns": total_cols,
            "duration_seconds": duration
        }

    except Exception as e:
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