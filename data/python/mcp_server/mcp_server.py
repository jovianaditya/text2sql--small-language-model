# mcp_server.py
import os
from dotenv import load_dotenv
import pymysql
from fastmcp import FastMCP
import logging
from datetime import datetime
import time
import uuid
import re
from difflib import get_close_matches

# -----------------------------
# Env & logging
# -----------------------------
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("MCPServer")
logging.getLogger("pymysql").setLevel(logging.WARNING)

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
    # Optional: stabilitas sesi awal
    "init_command": "SET time_zone = '+00:00'",
}

# Preview sizing (server-side)
PREVIEW_ROW_LIMIT = int(os.getenv("PREVIEW_ROW_LIMIT", "5"))
PREVIEW_COL_LIMIT = int(os.getenv("PREVIEW_COL_LIMIT", "5"))
STATEMENT_TIMEOUT_MS = int(os.getenv("STATEMENT_TIMEOUT_MS", "5000"))  # 5s
MAX_EXPORT_ROWS = int(os.getenv("MAX_EXPORT_ROWS", "200000"))

# -----------------------------
# SQL normalization helpers
# -----------------------------
# Keyword set (for spacing detection)
_SQL_KW = r"(?:SELECT|FROM|WHERE|GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT|OFFSET|JOIN|LEFT\s+JOIN|RIGHT\s+JOIN|INNER\s+JOIN|FULL\s+JOIN|ON|VALUES|SET)"

# Segment regex to skip quotes/backticks while normalizing
_SEG_LITERALS = re.compile(r"(`[^`]*`|'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")", re.DOTALL)

def _strip_sql_comments(sql: str) -> str:
    s = re.sub(r'--.*?$', '', sql, flags=re.M)
    s = re.sub(r'/\*.*?\*/', '', s, flags=re.S)
    return s

def _insert_space_before_sql_keywords(sql: str) -> str:
    """
    Insert one space immediately before a SQL clause keyword ONLY when:
      - we're NOT inside quotes/backticks, and
      - there is no whitespace already, and
      - the previous character is NOT an identifier char ([A-Za-z0-9_])
    Prevents splitting identifiers like 'creditLimit' -> 'credit Limit'.
    """
    def _fix(segment: str) -> str:
        pattern = rf"(?i)(?<!\s)(?<![A-Za-z0-9_])(?={_SQL_KW}\b)"
        return re.sub(pattern, " ", segment)

    out, i = [], 0
    for m in _SEG_LITERALS.finditer(sql):
        out.append(_fix(sql[i:m.start()]))   # process code
        out.append(m.group(0))               # keep literal/backtick intact
        i = m.end()
    out.append(_fix(sql[i:]))
    return "".join(out)

def _fix_missing_space_before_clauses(sql: str) -> str:
    """
    Ensure a space exists before a clause keyword, with the same safeguards as above.
    """
    CLAUSES = r"(SELECT|FROM|WHERE|GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT|OFFSET|JOIN|LEFT\s+JOIN|RIGHT\s+JOIN|INNER\s+JOIN|FULL\s+JOIN|ON|VALUES|SET)"

    def _fix(segment: str) -> str:
        pattern = rf"(?i)(?<!\s)(?<![A-Za-z0-9_])\b{CLAUSES}\b"
        return re.sub(pattern, r" \1", segment)

    out, i = [], 0
    for m in _SEG_LITERALS.finditer(sql):
        out.append(_fix(sql[i:m.start()]))
        out.append(m.group(0))
        i = m.end()
    out.append(_fix(sql[i:]))
    return "".join(out)

def _collapse_spaces(sql: str) -> str:
    return re.sub(r"[ \t]{2,}", " ", sql)

def _normalize_sql_for_wrapping(sql: str) -> str:
    """
    Normalisasi ringan supaya query aman:
    - hapus komentar
    - rapikan spasi, sisipkan spasi sebelum keyword (safely)
    - hapus ';' di akhir
    """
    s = _strip_sql_comments(sql or "")
    s = s.strip().rstrip(";").strip()
    s = _insert_space_before_sql_keywords(s)
    s = _fix_missing_space_before_clauses(s)
    s = _collapse_spaces(s)
    return s

# -----------------------------
# SQL helpers (no rewriting beyond normalization)
# -----------------------------
def is_single_select(sql: str) -> bool:
    """True if the SQL is exactly one SELECT statement."""
    if not isinstance(sql, str):
        return False
    s = _strip_sql_comments(sql).strip()
    if not s:
        return False
    parts = [p.strip() for p in re.split(r';\s*', s) if p.strip()]
    if len(parts) != 1:
        return False
    return bool(re.match(r'^\s*select\b', parts[0], flags=re.I))

def extract_base_query_without_limit(sql: str) -> str:
    """
    Strip trailing LIMIT/OFFSET dari SQL.
    Lakukan setelah normalisasi supaya pola '...productCodeLIMIT 5' sudah benar.
    """
    normalized = _normalize_sql_for_wrapping(sql)
    # Hapus LIMIT/OFFSET paling akhir jika ada
    sql_no_limit = re.sub(
        r'(?is)\s+(LIMIT\s+\d+(\s+OFFSET\s+\d+)?|OFFSET\s+\d+(\s+LIMIT\s+\d+)?)\s*$',
        "",
        normalized
    )
    return sql_no_limit.strip()

def wrap_with_limit(sql: str, limit: int) -> str:
    """Legacy helper (unused in preview now) — kept for reference."""
    core = _normalize_sql_for_wrapping(sql)
    return f"SELECT * FROM ({core}) AS subq LIMIT {int(limit)}"

def _build_preview_sql(normalized_sql: str, limit: int) -> str:
    """
    Build a simple preview statement: NEVER wrap in a subquery.
    - If the incoming normalized SQL already ends with LIMIT (optionally OFFSET),
      keep it if <= limit; cap it if it's larger; otherwise append LIMIT {limit}.
    """
    q = normalized_sql.strip()
    m = re.search(r'(?is)\blimit\s+(\d+)(\s+offset\s+\d+)?\s*$', q)
    if m:
        try:
            existing = int(m.group(1))
        except Exception:
            existing = None
        if existing is not None and existing > limit:
            # cap the limit but keep any trailing OFFSET
            # rebuild the tail
            capped_tail = f"LIMIT {limit}{m.group(2) or ''}"
            return q[:m.start()] + capped_tail
        return q
    return f"{q} LIMIT {int(limit)}"

def _conn():
    """Open connection + ping to ensure healthy session."""
    c = pymysql.connect(**DB_CONFIG)
    try:
        c.ping(reconnect=True)
    except Exception:
        pass
    return c

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

def suggest_columns(conn, table_fqn: str, missing_col: str, max_suggestions: int = 5):
    """Given a table and a missing column name, return closest columns."""
    schema = DB_CONFIG.get("db")
    table = table_fqn
    if "." in table_fqn:
        schema, table = table_fqn.split(".", 1)
    schema = schema.strip("`")
    table = table.strip("`")
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema=%s AND table_name=%s
            """, (schema, table))
            cols = [r[0] for r in cur.fetchall()]
    except Exception:
        cols = []
    return get_close_matches(missing_col, cols, n=max_suggestions, cutoff=0.5)

def set_statement_timeout(conn, timeout_ms: int):
    """Try to set MySQL 8 session max_execution_time (ms)."""
    try:
        with conn.cursor() as cur:
            cur.execute(f"SET SESSION max_execution_time = {int(timeout_ms)};")
        logger.debug(f"max_execution_time set to {timeout_ms} ms")
    except Exception as e:
        logger.warning(f"max_execution_time not settable: {e}")

def _preflight_explain(conn, sql: str):
    """Optional: EXPLAIN to fail-fast on schema errors."""
    try:
        with conn.cursor() as cur:
            cur.execute("EXPLAIN " + sql)
    except pymysql.err.ProgrammingError:
        # Re-raise for the normal handler to categorize
        raise
    except Exception:
        # Ignore non-programming errors
        pass

def _build_count_sql(base_sql: str) -> str:
    """Count total rows by wrapping the LIMIT-less, normalized base query."""
    core = _normalize_sql_for_wrapping(base_sql)
    return f"SELECT COUNT(*) AS total_rows FROM ({core}) AS count_subquery"

# (Optional) Guardrail: warn if an identifier/keyword split is detected
def _warn_identifier_keyword_split(tag: str, sql: str):
    m = re.search(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\.\s*([A-Za-z_][A-Za-z0-9_]*)\s+"
        r"(SELECT|FROM|WHERE|GROUP|BY|HAVING|ORDER|LIMIT|JOIN|LEFT|RIGHT|INNER|FULL|ON)\b",
        sql, flags=re.I
    )
    if m:
        logger.warning("[%s] Possible identifier/keyword split near: ...%s...",
                       tag, sql[max(0, m.start()-30):m.end()+30])

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
        conn = _conn()
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
    except Exception:
        logger.exception(f"[{req}] ❌ list_tables ERROR elapsed_ms={_elapsed_ms(t0):.1f}")
        return {"success": False, "error": "Failed to list tables."}
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
        conn = _conn()
        set_statement_timeout(conn, STATEMENT_TIMEOUT_MS)

        # Normalize terlebih dahulu (safe)
        normalized_sql = _normalize_sql_for_wrapping(sql)
        base_query = extract_base_query_without_limit(normalized_sql)

        # Guardrail log (optional)
        _warn_identifier_keyword_split(req, base_query)

        # true count (wrapped, LIMIT-less)
        count_sql = _build_count_sql(base_query)
        _log_sql(req, normalized_sql=normalized_sql, base_query=base_query, count_sql=count_sql)
        total_rows = 0
        try:
            with conn.cursor() as count_cur:
                count_cur.execute(count_sql)
                total_rows = count_cur.fetchone()[0]
        except pymysql.err.ProgrammingError as e:
            code = e.args[0] if e.args else None
            msg = e.args[1] if len(e.args) > 1 else str(e)
            logger.warning(f"[{req}] count failed code={code} msg={msg}")

            # Table missing
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

            # Unknown column → column suggestions
            if code == 1054 or "Unknown column" in msg:
                col_m = re.search(r"Unknown column '([^']+)' in '([^']+)'", msg)
                missing_col = col_m.group(1) if col_m else ""
                tbl_m = re.search(r"\bfrom\s+([`\"\w\.]+)", base_query, flags=re.I)
                table_fqn = tbl_m.group(1).strip('`"') if tbl_m else ""
                col_sug = suggest_columns(conn, table_fqn, missing_col, 5) if (table_fqn and missing_col) else []
                return {
                    "success": False,
                    "error": f"Unknown column: {missing_col or 'unknown'}",
                    "table": table_fqn,
                    "column_suggestions": col_sug
                }

            logger.warning(f"[{req}] Count failed; continuing without total. msg={msg}")
            total_rows = None
        except Exception as e:
            logger.warning(f"[{req}] count failed generic: {e}")
            total_rows = None

        # PREVIEW (NO WRAP) — keeps the query simple
        preview_sql = _build_preview_sql(normalized_sql, PREVIEW_ROW_LIMIT)
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
            f"cols_total={total_cols} rows_total={total_rows if total_rows is not None else 'n/a'} "
            f"elapsed_ms={_elapsed_ms(t0):.1f}"
        )

        return {
            "success": True,
            "preview_rows": preview_rows_limited,
            "preview_columns": preview_cols,
            "total_rows": total_rows,
            "total_columns": total_cols,
            "duration_seconds": _elapsed_ms(t0) / 1000.0,
            "_debug_info": {
                "original_query": sql,
                "normalized_query": normalized_sql,
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

    if not is_single_select(sql := str(sql)):
        logger.warning(f"[{req}] rejected: not a single SELECT")
        return {"success": False, "error": "Only a single SELECT statement is allowed."}

    conn = None
    try:
        conn = _conn()
        set_statement_timeout(conn, STATEMENT_TIMEOUT_MS)

        normalized_sql = _normalize_sql_for_wrapping(sql)
        base_query = extract_base_query_without_limit(normalized_sql)

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
                "normalized_query": normalized_sql,
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

        if code == 1054 or "Unknown column" in msg:
            col_m = re.search(r"Unknown column '([^']+)' in '([^']+)'", msg)
            missing_col = col_m.group(1) if col_m else ""
            tbl_m = re.search(r"\bfrom\s+([`\"\w\.]+)", base_query, flags=re.I)
            table_fqn = tbl_m.group(1).strip('`"') if tbl_m else ""
            col_sug = suggest_columns(conn, table_fqn, missing_col, 5) if (table_fqn and missing_col) else []
            return {
                "success": False,
                "error": f"Unknown column: {missing_col or 'unknown'}",
                "table": table_fqn,
                "column_suggestions": col_sug
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
# Run MCP Server
# -----------------------------
if __name__ == "__main__":
    logger.info("🚀 Starting MySQL MCP Server...")
    mcp.run(transport="http", host="0.0.0.0", port=8000, log_level="info")