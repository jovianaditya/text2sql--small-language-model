from collections import defaultdict
import os

from datahub.metadata.urns import DatasetUrn
from datahub.sdk import DataHubClient
from dotenv import find_dotenv, load_dotenv
import pymysql


load_dotenv(find_dotenv(usecwd=True), override=False)


def _env(name: str, default: str = "") -> str:
    return str(os.getenv(name, default) or "").strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for {name}: {raw}") from exc


def _datahub_server() -> str:
    explicit = _env("DATAHUB_GMS_URL")
    if explicit:
        return explicit

    host = _env("DATAHUB_GMS_HOST", "datahub-gms")
    port = _env_int("DATAHUB_GMS_PORT", 8080)
    return f"http://{host}:{port}"


def _mysql_config() -> dict:
    return {
        "host": _env("MYSQL_HOST", "mysql"),
        "port": _env_int("MYSQL_PORT", 3306),
        "user": _env("MYSQL_USER", "datahub"),
        "password": _env("MYSQL_PASSWORD", "datahub"),
        "database": _env("MYSQL_DB") or _env("MYSQL_DATABASE", "classicmodels"),
        "charset": "utf8mb4",
        "cursorclass": pymysql.cursors.DictCursor,
        "autocommit": True,
    }


def _dataset_urn(platform: str, database: str, table: str, env: str) -> DatasetUrn:
    return DatasetUrn(platform=platform, name=f"{database}.{table}", env=env)


def fetch_foreign_keys(conn, database: str) -> list[dict]:
    query = """
    SELECT
      kcu.TABLE_NAME AS child_table,
      kcu.COLUMN_NAME AS child_column,
      kcu.REFERENCED_TABLE_NAME AS parent_table,
      kcu.REFERENCED_COLUMN_NAME AS parent_column
    FROM information_schema.KEY_COLUMN_USAGE kcu
    WHERE kcu.TABLE_SCHEMA = %s
      AND kcu.REFERENCED_TABLE_NAME IS NOT NULL
    ORDER BY child_table, child_column
    """

    with conn.cursor() as cur:
        cur.execute(query, (database,))
        return list(cur.fetchall())


def group_column_lineage(rows: list[dict], database: str, platform: str, env: str) -> dict:
    grouped: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(dict)

    for row in rows:
        upstream = _dataset_urn(platform, database, row["parent_table"], env)
        downstream = _dataset_urn(platform, database, row["child_table"], env)
        grouped[(str(upstream), str(downstream))][row["child_column"]] = [row["parent_column"]]

    return grouped


def emit_lineage(client: DataHubClient, grouped_lineage: dict) -> int:
    emitted = 0

    for (upstream, downstream), column_lineage in grouped_lineage.items():
        client.lineage.add_lineage(
            upstream=upstream,
            downstream=downstream,
            column_lineage=column_lineage,
            transformation_text="MySQL foreign key lineage from information_schema.KEY_COLUMN_USAGE",
        )
        emitted += 1
        print(
            f"[OK] {downstream} -> {upstream} "
            f"({len(column_lineage)} column mapping{'s' if len(column_lineage) != 1 else ''})"
        )

    return emitted


def main() -> None:
    mysql_cfg = _mysql_config()
    database = mysql_cfg["database"]
    platform = _env("DATAHUB_DATASET_PLATFORM", "mysql")
    env = _env("DATAHUB_ENV", "PROD")
    server = _datahub_server()
    token = _env("DATAHUB_TOKEN")

    if not token:
        raise RuntimeError("DATAHUB_TOKEN is required.")
    if not database:
        raise RuntimeError("MYSQL_DB or MYSQL_DATABASE is required.")

    print(f"[INFO] DataHub server: {server}")
    print(f"[INFO] MySQL source: {mysql_cfg['host']}:{mysql_cfg['port']}/{database}")
    print(f"[INFO] Dataset platform/env: {platform}/{env}")

    client = DataHubClient(server=server, token=token)

    with pymysql.connect(**mysql_cfg) as conn:
        fk_rows = fetch_foreign_keys(conn, database)

    print(f"[INFO] Foreign keys discovered: {len(fk_rows)}")

    grouped_lineage = group_column_lineage(fk_rows, database, platform, env)
    emitted = emit_lineage(client, grouped_lineage)

    print(
        f"[DONE] Emitted {emitted} dataset lineage edge"
        f"{'s' if emitted != 1 else ''} from {len(fk_rows)} foreign key row"
        f"{'s' if len(fk_rows) != 1 else ''}."
    )


if __name__ == "__main__":
    main()
