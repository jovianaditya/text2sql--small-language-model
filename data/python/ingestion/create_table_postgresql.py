# create_table_postgresql.py
import sqlite3
import pandas as pd
import io, csv, re
import psycopg2
from psycopg2 import sql
import numpy as np

# ---------- helpers ----------
def df_to_csv_buffer(df: pd.DataFrame) -> io.StringIO:
    """Convert a DataFrame chunk to an in-memory CSV buffer for COPY."""
    df = df.astype(object).where(pd.notnull(df), '')
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    buf.seek(0)
    return buf

def normalize_integer_columns(df: pd.DataFrame, int_cols: list[str]) -> pd.DataFrame:
    """Convert values like '1.0', '418.000' → '1', '418' for integer columns."""
    if not int_cols:
        return df

    def fix_cell(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ''
        s = str(v).strip()
        if s == '':
            return ''
        # integer-valued floats -> int string
        try:
            f = float(s)
            if f.is_integer():
                return str(int(f))
        except Exception:
            pass
        # already an integer string
        if re.fullmatch(r"-?\d+", s):
            return s
        # strip trailing .0... if present
        m = re.fullmatch(r"\s*(-?\d+)\.0+\s*", s)
        if m:
            return m.group(1)
        return s

    for c in int_cols:
        if c in df.columns:
            df[c] = df[c].map(fix_cell)
    return df

# --- type mapping: force strings to VARCHAR (keep length if present) ---
_type_re_cache = {
    "varchar": re.compile(r"(?i)^n?varchar\((\d+)\)$"),
    "char": re.compile(r"(?i)^n?char\((\d+)\)$"),
    "decimal": re.compile(r"(?i)^(?:decimal|numeric)\s*\(\s*(\d+)\s*(?:,\s*(\d+)\s*)?\)$"),
    "float": re.compile(r"(?i)^float(?:\s*\(\s*\d+\s*\))?$"),
}

def map_sqlite_decl_to_pg_type(decl: str) -> str:
    """
    Map a SQLite declared type to a PostgreSQL type.
    REQUIREMENT: all string types -> VARCHAR (with length if provided).
    """
    if not decl:
        return "VARCHAR"  # default to VARCHAR (unbounded)

    t = decl.strip().upper()

    # VARCHAR(n) / NVARCHAR(n) -> VARCHAR(n)
    m = _type_re_cache["varchar"].match(t)
    if m:
        return f"VARCHAR({m.group(1)})"

    # CHAR(n) / NCHAR(n) -> VARCHAR(n)
    m = _type_re_cache["char"].match(t)
    if m:
        return f"VARCHAR({m.group(1)})"

    # DECIMAL/NUMERIC(p[,s])
    m = _type_re_cache["decimal"].match(t)
    if m:
        p = m.group(1); s = m.group(2)
        return f"NUMERIC({p}{',' + s if s else ''})"

    # FLOAT -> DOUBLE PRECISION
    if _type_re_cache["float"].match(t):
        return "DOUBLE PRECISION"

    # Integer family
    if "INT" in t:
        if "TINY" in t or t.endswith("INT2"):
            return "SMALLINT"
        if "BIG" in t or t.endswith("INT8"):
            return "BIGINT"
        if "SMALL" in t:
            return "SMALLINT"
        return "INTEGER"

    # Any string-ish types without explicit length -> VARCHAR
    if any(x in t for x in ("CHAR", "CLOB", "TEXT", "STRING")):
        return "VARCHAR"

    # Binary / blob
    if "BLOB" in t or "BINARY" in t:
        return "BYTEA"

    # Floating/real
    if any(x in t for x in ("DOUBLE", "REAL", "FLOA")):
        return "DOUBLE PRECISION"

    # Generic numeric
    if "NUMERIC" in t or "DECIMAL" in t:
        return "NUMERIC"

    # Booleans
    if "BOOLEAN" in t or "BOOL" in t:
        return "BOOLEAN"

    # Date/time
    if "DATETIME" in t or "TIMESTAMP" in t or ("DATE" in t and "TIME" in t):
        return "TIMESTAMP"
    if "DATE" in t:
        return "DATE"
    if "TIME" in t:
        return "TIME"

    if t.startswith("UUID"):
        return "UUID"

    # Otherwise, default to VARCHAR
    return "VARCHAR"

def sql_identifier_quote(name: str) -> str:
    """Quote for PRAGMA usage (SQLite side)."""
    return '"' + name.replace('"', '""') + '"'

def get_sqlite_columns(sqlite_conn, table_name: str):
    """
    Return list of (name, sqlite_declared_type, pg_type, notnull, default, pk)
    Ordered by the SQLite schema order.
    """
    cur = sqlite_conn.execute(f"PRAGMA table_info({sql_identifier_quote(table_name)});")
    rows = cur.fetchall()
    cols = []
    for cid, name, decl_type, notnull, dflt_value, pk in rows:
        pg_type = map_sqlite_decl_to_pg_type(decl_type or "")
        cols.append((name, decl_type or "", pg_type, notnull, dflt_value, pk))
    return cols

# ---------- connections ----------
sqlite_conn = sqlite3.connect("/app/sqlite/california_schools.sqlite")

tables_df = pd.read_sql(
    "SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence';",
    sqlite_conn
)
table_names = tables_df['name'].tolist()
print(f"Found tables: {table_names}")

conn = psycopg2.connect(
    host="188.166.245.16",
    database="california_schools",
    user="datahub",
    password="datahub",
)

with conn:
    with conn.cursor() as cur:
        for table in table_names:
            try:
                cols_info = get_sqlite_columns(sqlite_conn, table)
                if not cols_info:
                    print(f"Skipping {table}: no columns found")
                    continue

                # DROP + CREATE with mapped types (preserve order)
                cur.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(table)))

                col_defs = []
                pk_cols = []
                for name, decl, pg_type, notnull, dflt, pk in cols_info:
                    col_sql = sql.SQL("{} {}").format(sql.Identifier(name), sql.SQL(pg_type))
                    if notnull:
                        col_sql += sql.SQL(" NOT NULL")
                    if dflt is not None:
                        # dflt is SQLite literal (often already quoted); pass through as SQL
                        col_sql += sql.SQL(" DEFAULT ") + sql.SQL(str(dflt))
                    col_defs.append(col_sql)
                    if pk:
                        pk_cols.append(sql.Identifier(name))

                create_sql = sql.SQL("CREATE TABLE {} (").format(sql.Identifier(table)) + sql.SQL(", ").join(col_defs)
                if pk_cols:
                    create_sql += sql.SQL(", PRIMARY KEY (") + sql.SQL(", ").join(pk_cols) + sql.SQL(")")
                create_sql += sql.SQL(")")
                cur.execute(create_sql)

                # Identify integer columns in PG to sanitize input ("1.0" -> "1")
                int_types = {"SMALLINT", "INTEGER", "BIGINT"}
                int_cols = [name for (name, _, pg_type, _, _, _) in cols_info if pg_type.upper() in int_types]

                # COPY with exact column order
                col_idents = [sql.Identifier(c[0]) for c in cols_info]
                copy_stmt = sql.SQL(
                    "COPY {} ({}) FROM STDIN WITH (FORMAT CSV, DELIMITER ',', QUOTE '\"', NULL '')"
                ).format(
                    sql.Identifier(table),
                    sql.SQL(", ").join(col_idents)
                ).as_string(cur)

                total_rows = 0
                for chunk in pd.read_sql_query(f'SELECT * FROM {sql_identifier_quote(table)}', sqlite_conn, chunksize=10_000):
                    chunk = normalize_integer_columns(chunk, int_cols)
                    buf = df_to_csv_buffer(chunk)
                    cur.copy_expert(copy_stmt, buf)
                    total_rows += len(chunk)
                    print(f"  {table}: inserted {total_rows} rows...", end="\r")

                print(f"\n✓ Done: {table} ({total_rows} rows)")

            except Exception as e:
                print(f"✗ Failed {table}: {e}")
                conn.rollback()

sqlite_conn.close()
print("Migration complete.")