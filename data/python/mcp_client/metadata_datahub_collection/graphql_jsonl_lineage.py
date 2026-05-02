#!/usr/bin/env python3
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import quote

import requests
from dotenv import find_dotenv, load_dotenv


load_dotenv(find_dotenv(), override=True)


GMS_BASE = os.getenv("DATAHUB_GMS_URL", "http://datahub-gms:8080")
GRAPHQL_URL = os.getenv("DATAHUB_GRAPHQL_URL", f"{GMS_BASE}/api/graphql")
TOKEN = os.getenv("DATAHUB_TOKEN")
PAGE_SIZE = int(os.getenv("DATAHUB_PAGE_SIZE", "100"))
TIMEOUT = 60
ACTOR = os.getenv("DATAHUB_ACTOR")
OUTPUT_DIR = os.getenv("EXPORT_LINEAGE_JSONL_OUTPUT", os.getenv("EXPORT_JSONL_OUTPUT", "./exports_lineage_final"))
ONLY_RELATIONAL = os.getenv("DATAHUB_LINEAGE_ONLY_RELATIONAL", "1").strip().lower() in {"1", "true", "yes", "on"}

if not TOKEN:
    raise RuntimeError("DATAHUB_TOKEN is not set.")

RELATIONAL_3LEVEL = {"postgres", "snowflake", "redshift", "bigquery", "mssql", "mysql"}
UPSTREAM_LINEAGE_ASPECT = "com.linkedin.dataset.UpstreamLineage"
SCHEMA_METADATA_ASPECT = "com.linkedin.schema.SchemaMetadata"


SCROLL_QUERY = """
query scrollPG($types: [EntityType!], $query: String!, $count: Int!, $scrollId: String) {
  scrollAcrossEntities(input: { types: $types, query: $query, count: $count, scrollId: $scrollId }) {
    nextScrollId
    searchResults {
      entity {
        urn
        type
        ... on Dataset { urn }
      }
    }
  }
}
""".strip()


def _headers_json() -> Dict[str, str]:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"}
    if ACTOR:
        headers["X-DataHub-Actor"] = ACTOR
    return headers


def _graphql(query: str, variables: dict) -> dict:
    try:
        resp = requests.post(
            GRAPHQL_URL,
            json={"query": query, "variables": variables},
            headers=_headers_json(),
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("errors"):
            msg = data.get("errors", [{}])[0].get("message", "unknown GraphQL error")
            print(f"[WARN] GraphQL returned errors: {msg}")
            return {}
        return data.get("data", {})
    except Exception as exc:
        print(f"[WARN] GraphQL request failed: {exc}")
        return {}


def iter_dataset_urns() -> Iterable[str]:
    scroll_id = None
    while True:
        data = _graphql(
            SCROLL_QUERY,
            {"types": ["DATASET"], "query": "*", "count": PAGE_SIZE, "scrollId": scroll_id},
        ).get("scrollAcrossEntities", {})
        results = data.get("searchResults") or []
        if not results:
            break
        for item in results:
            urn = ((item.get("entity") or {}).get("urn") or "").strip()
            if urn:
                yield urn
        scroll_id = data.get("nextScrollId")
        if not scroll_id:
            break


def _parse_urn_name(urn: str) -> Optional[str]:
    try:
        inner = urn[urn.index("(") + 1 : urn.rindex(")")]
        parts = inner.split(",", 2)
        return parts[1] if len(parts) > 1 else None
    except Exception:
        return None


def _parse_dataset_urn(urn: str, delim: str = ".") -> Dict[str, Optional[str]]:
    name_part = _parse_urn_name(urn)
    platform_match = re.search(r"urn:li:dataPlatform:([^,)+]+)", urn)
    env_match = re.search(r",([^,()]+)\)$", urn)
    platform = (platform_match.group(1).lower() if platform_match else None)
    env = env_match.group(1) if env_match else None

    database = None
    schema = None
    table = None

    if name_part:
        if (platform or "").lower() in RELATIONAL_3LEVEL:
            parts = name_part.split(delim)
            if len(parts) >= 3:
                database, schema, table = parts[0], parts[1], delim.join(parts[2:])
            elif len(parts) == 2:
                database, table = parts[0], parts[1]
            elif len(parts) == 1:
                table = parts[0]
        else:
            table = name_part

    return {
        "urn": urn,
        "platform": platform,
        "env": env,
        "name": name_part,
        "database": database,
        "schema": schema,
        "table": table,
    }


def _parse_schema_field_urn(urn: str) -> Dict[str, Optional[str]]:
    match = re.match(r"urn:li:schemaField:\((urn:li:dataset:\(.+?\)),(.+)\)$", urn)
    if not match:
        return {"urn": urn, "dataset_urn": None, "column": None}
    dataset_urn, column = match.group(1), match.group(2)
    info = _parse_dataset_urn(dataset_urn)
    info.update({"dataset_urn": dataset_urn, "column": column, "urn": urn})
    return info


def fetch_dataset_snapshot(urn: str) -> Optional[dict]:
    url = f"{GMS_BASE}/entities/{quote(urn, safe='')}"
    try:
        resp = requests.get(url, headers=_headers_json(), timeout=TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()
        return (payload.get("value") or {}).get("com.linkedin.metadata.snapshot.DatasetSnapshot")
    except Exception as exc:
        print(f"[WARN] Entity fetch failed for {urn}: {exc}")
        return None


def _snapshot_aspect_map(snapshot: dict) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for aspect in snapshot.get("aspects") or []:
        if not isinstance(aspect, dict):
            continue
        for key, value in aspect.items():
            out[key] = value
    return out


def _sanitize_filename(name: str) -> str:
    if not name:
        return "unknown"
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _normalize_table_name(info: Dict[str, Optional[str]]) -> Optional[str]:
    return (info.get("table") or info.get("name") or None)


def _make_row(
    downstream: Dict[str, Optional[str]],
    upstream: Dict[str, Optional[str]],
    downstream_column: Optional[str],
    upstream_column: Optional[str],
) -> Dict[str, Any]:
    return {
        "id": (
            f"{_normalize_table_name(downstream) or 'unknown'}.{downstream_column or '*'}"
            f"->{_normalize_table_name(upstream) or 'unknown'}.{upstream_column or '*'}"
        ),
        "from_table": _normalize_table_name(downstream),
        "from_column": downstream_column,
        "to_table": _normalize_table_name(upstream),
        "to_column": upstream_column,
    }


def _extract_lineage_rows_from_upstream_aspect(
    dataset_info: Dict[str, Optional[str]],
    upstream_aspect: dict,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for fine in upstream_aspect.get("fineGrainedLineages") or []:
        upstream_fields = [_parse_schema_field_urn(x) for x in (fine.get("upstreams") or [])]
        downstream_fields = [_parse_schema_field_urn(x) for x in (fine.get("downstreams") or [])]

        for downstream_field in downstream_fields:
            for upstream_field in upstream_fields:
                rows.append(
                    _make_row(
                        downstream=downstream_field,
                        upstream=upstream_field,
                        downstream_column=downstream_field.get("column"),
                        upstream_column=upstream_field.get("column"),
                    )
                )

    if rows:
        return rows

    for upstream in upstream_aspect.get("upstreams") or []:
        upstream_info = _parse_dataset_urn(upstream.get("dataset") or "")
        rows.append(
            _make_row(
                downstream=dataset_info,
                upstream=upstream_info,
                downstream_column=None,
                upstream_column=None,
            )
        )

    return rows


def _extract_lineage_rows_from_foreign_keys(schema_aspect: dict) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for foreign_key in schema_aspect.get("foreignKeys") or []:
        downstream_fields = [_parse_schema_field_urn(x) for x in (foreign_key.get("sourceFields") or [])]
        upstream_fields = [_parse_schema_field_urn(x) for x in (foreign_key.get("foreignFields") or [])]

        if downstream_fields and upstream_fields:
            pair_count = min(len(downstream_fields), len(upstream_fields))
            for idx in range(pair_count):
                downstream_field = downstream_fields[idx]
                upstream_field = upstream_fields[idx]
                rows.append(
                    _make_row(
                        downstream=downstream_field,
                        upstream=upstream_field,
                        downstream_column=downstream_field.get("column"),
                        upstream_column=upstream_field.get("column"),
                    )
                )
        else:
            downstream_info = _parse_dataset_urn(
                ((downstream_fields[0] if downstream_fields else {}).get("dataset_urn")) or ""
            )
            upstream_info = _parse_dataset_urn(foreign_key.get("foreignDataset") or "")
            rows.append(
                _make_row(
                    downstream=downstream_info,
                    upstream=upstream_info,
                    downstream_column=None,
                    upstream_column=None,
                )
            )

    return rows


def _dedupe_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()

    for row in rows:
        key = (
            row.get("from_table"),
            row.get("from_column"),
            row.get("to_table"),
            row.get("to_column"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)

    return out


def fetch_dataset_lineage_rows(urn: str) -> List[Dict[str, Any]]:
    snapshot = fetch_dataset_snapshot(urn)
    if not snapshot:
        return []

    dataset_info = _parse_dataset_urn(snapshot.get("urn") or urn)
    if ONLY_RELATIONAL and (dataset_info.get("platform") or "").lower() not in RELATIONAL_3LEVEL:
        return []

    aspects = _snapshot_aspect_map(snapshot)
    rows: List[Dict[str, Any]] = []

    upstream_aspect = aspects.get(UPSTREAM_LINEAGE_ASPECT) or {}
    if upstream_aspect:
        rows.extend(_extract_lineage_rows_from_upstream_aspect(dataset_info, upstream_aspect))

    if not rows:
        schema_aspect = aspects.get(SCHEMA_METADATA_ASPECT) or {}
        if schema_aspect:
            rows.extend(_extract_lineage_rows_from_foreign_keys(schema_aspect))

    return _dedupe_rows(rows)


def main() -> None:
    output_dir = OUTPUT_DIR
    if output_dir.lower().endswith(".jsonl"):
        output_dir = os.path.dirname(output_dir) or "."
        print(f"[WARN] OUTPUT path looks like a file. Using parent directory for lineage export: {output_dir}")
    elif os.path.exists(output_dir) and os.path.isfile(output_dir):
        output_dir = os.path.dirname(output_dir) or "."
        print(f"[WARN] OUTPUT path is a file. Using parent directory for lineage export: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)
    writers: Dict[str, Any] = {}
    count = 0

    print(f"[INFO] Exporting LINEAGE JSONL to: {os.path.abspath(output_dir)}")

    for urn in iter_dataset_urns():
        try:
            rows = fetch_dataset_lineage_rows(urn)
            if not rows:
                continue

            print(f"[proc] {urn} -> {len(rows)} relation rows")
            db_name = _parse_dataset_urn(urn).get("database") or "unknown"
            for row in rows:
                fname = f"{_sanitize_filename(db_name)}_lineage.jsonl"
                if fname not in writers:
                    writers[fname] = open(os.path.join(output_dir, fname), "w", encoding="utf-8")
                writers[fname].write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
        except Exception as exc:
            print(f"[ERROR] {urn}: {exc}")

    for handle in writers.values():
        handle.close()

    print(f"[DONE] Total {count} lineage rows exported.")


if __name__ == "__main__":
    main()
