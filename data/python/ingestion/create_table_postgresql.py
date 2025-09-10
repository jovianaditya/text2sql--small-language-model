import sqlite3
import pandas as pd
import numpy as np
import io
import csv
import psycopg2
from psycopg2 import sql

# ---------- helpers ----------
def df_to_csv_buffer(df: pd.DataFrame) -> io.StringIO:
    """
    Convert a DataFrame chunk to an in-memory CSV buffer for COPY.
    - Convert NaN/None to empty string, and use COPY ... NULL '' to load as NULL.
    """
    # Ensure object dtype and empty strings for nulls (COPY will treat '' as NULL)
    df = df.astype(object)
    df = df.where(pd.notnull(df), '')

    buf = io.StringIO()
    # No header; CSV quoting handles commas/quotes safely
    df.to_csv(
        buf,
        index=False,
        header=False,
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
    )
    buf.seek(0)
    return buf

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
                # Get column names only
                cols_df = pd.read_sql_query(f'SELECT * FROM "{table}" LIMIT 0', sqlite_conn)
                cols = list(cols_df.columns)
                print(f"Migrating {table}... ({len(cols)} columns)")

                # DROP + CREATE (all TEXT as per your approach)
                cur.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(table)))
                cols_def = sql.SQL(", ").join(
                    sql.SQL("{} TEXT").format(sql.Identifier(c)) for c in cols
                )
                cur.execute(
                    sql.SQL("CREATE TABLE {} ({})").format(
                        sql.Identifier(table),
                        cols_def
                    )
                )

                # COPY statement (note: we pass a string to copy_expert; no params)
                copy_stmt = sql.SQL(
                    "COPY {} ({}) FROM STDIN WITH (FORMAT CSV, DELIMITER ',', QUOTE '\"', NULL '')"
                ).format(
                    sql.Identifier(table),
                    sql.SQL(", ").join(map(sql.Identifier, cols))
                ).as_string(cur)

                chunksize = 10_000
                total_rows = 0

                for chunk in pd.read_sql_query(f'SELECT * FROM "{table}"', sqlite_conn, chunksize=chunksize):
                    buf = df_to_csv_buffer(chunk)
                    cur.copy_expert(copy_stmt, buf)
                    total_rows += len(chunk)
                    print(f"  Inserted {total_rows} rows...", end="\r")

                print(f"\n✓ Done: {table} ({total_rows} rows)")

            except Exception as e:
                print(f"✗ Failed {table}: {e}")
                conn.rollback()

sqlite_conn.close()
print("Migration complete.")