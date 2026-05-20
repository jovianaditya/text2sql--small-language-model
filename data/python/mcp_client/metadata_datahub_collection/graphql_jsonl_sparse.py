#!/usr/bin/env python3
import os
import json
import requests
import re
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv, find_dotenv
from urllib.parse import quote

# Load .env without overriding runtime/container env vars.
load_dotenv(find_dotenv(), override=False)

# =========================
# 1. CONFIGURATION
# =========================
GMS_BASE = os.getenv("DATAHUB_GMS_URL", "http://datahub-gms:8080")
GRAPHQL_URL = os.getenv("DATAHUB_GRAPHQL_URL", f"{GMS_BASE}/api/graphql")
TOKEN = os.getenv("DATAHUB_TOKEN")
PAGE_SIZE = int(os.getenv("DATAHUB_PAGE_SIZE", "100"))
TIMEOUT = 60
ACTOR = os.getenv("DATAHUB_ACTOR", "urn:li:corpuser:admin")

# OUTPUT KHUSUS SPARSE
# Prefer dedicated sparse output env var, then fallback to generic export env.
OUTPUT_DIR = os.getenv("EXPORT_SPARSE_JSONL_OUTPUT", os.getenv("EXPORT_JSONL_OUTPUT", "./exports_sparse_final"))
SPARSE_STRING_SAMPLE_LIMIT = int(os.getenv("SPARSE_STRING_SAMPLE_LIMIT", "8"))
SPARSE_NUMERIC_SAMPLE_LIMIT = int(os.getenv("SPARSE_NUMERIC_SAMPLE_LIMIT", "3"))
SPARSE_SAMPLE_VALUE_MAX_CHARS = int(os.getenv("SPARSE_SAMPLE_VALUE_MAX_CHARS", "64"))
SPARSE_STRING_SAMPLE_DISTINCT_MAX = int(os.getenv("SPARSE_STRING_SAMPLE_DISTINCT_MAX", "200"))
SPARSE_METADATA_SAMPLE_MAX_CHARS = int(os.getenv("SPARSE_METADATA_SAMPLE_MAX_CHARS", "220"))
SPARSE_INCLUDE_SAMPLES_IN_KEYWORDS = os.getenv("SPARSE_INCLUDE_SAMPLES_IN_KEYWORDS", "0").strip().lower() in {"1", "true", "yes", "on"}

RELATIONAL_3LEVEL = {"postgres", "snowflake", "redshift", "bigquery", "mssql", "mysql"}

# =========================
# 2. GRAPHQL QUERIES
# =========================
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

DATASET_META_QUERY = """
query datasetMeta($urn: String!) {
  dataset(urn: $urn) {
    urn
    properties { description } 
    platform { name properties { datasetNameDelimiter } }
    schemaMetadata(version: 0) {
      fields {
        fieldPath
        description
        nativeDataType
        type
        globalTags {
          tags {
            tag {
              urn
              properties { name }
            }
          }
        }
      }
    }
    editableSchemaMetadata {
      editableSchemaFieldInfo {
        fieldPath
        globalTags {
          tags {
            tag {
              urn
              properties { name }
            }
          }
        }
      }
    }
  }
}
""".strip()

DATASET_PROFILE_QUERY = """
query datasetProfile($urn: String!) {
  dataset(urn: $urn) {
    datasetProfiles(limit: 1) {
      rowCount
      fieldProfiles {
        fieldPath
        uniqueCount
        min
        max
        mean
        stdev
        sampleValues
      }
    }
  }
}
""".strip()

# =========================
# 3. HELPER FUNCTIONS
# =========================
def _headers_json() -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if TOKEN:
        h["Authorization"] = f"Bearer {TOKEN}"
    elif ACTOR:
        h["X-DataHub-Actor"] = ACTOR
    return h

def _graphql(query: str, variables: dict) -> dict:
    try:
        resp = requests.post(GRAPHQL_URL, json={"query": query, "variables": variables}, headers=_headers_json(), timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if data.get("errors"):
            msg = data.get("errors", [{}])[0].get("message", "unknown GraphQL error")
            print(f"[WARN] GraphQL returned errors: {msg}")
            return {}
        return data.get("data", {})
    except Exception as e:
        print(f"[WARN] GraphQL request failed: {e}")
        return {}

def iter_dataset_urns():
    scroll_id = None
    while True:
        data = _graphql(SCROLL_QUERY, {"types": ["DATASET"], "query": "*", "count": PAGE_SIZE, "scrollId": scroll_id}).get("scrollAcrossEntities", {})
        results = data.get("searchResults") or []
        if not results: break
        for item in results:
            ent = item.get("entity") or {}
            urn = ent.get("urn")
            if urn: yield urn
        scroll_id = data.get("nextScrollId")
        if not scroll_id: break

def _parse_urn_name(urn: str) -> Optional[str]:
    try:
        inner = urn[urn.index("(") + 1 : urn.rindex(")")]
        parts = inner.split(",", 2)
        return parts[1] if len(parts) > 1 else None
    except: return None

def _split_from_urn(urn: str, platform_name: Optional[str], delim: str = ".") -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    name_part = _parse_urn_name(urn)
    if not name_part: return None, None, None, None
    plat = (platform_name or "").lower()
    if plat in RELATIONAL_3LEVEL:
        parts = name_part.split(delim)
        if len(parts) >= 3: return name_part, parts[0], parts[1], delim.join(parts[2:])
        elif len(parts) == 2: return name_part, parts[0], None, parts[1]
        else: return name_part, None, None, parts[0]
    return name_part, None, None, name_part

def _norm_field(s: Optional[str]) -> Optional[str]:
    if s is None: return None
    return s.strip().strip('`"[]').lower()

def _as_float(x):
    if x is None: return None
    try: return float(x)
    except: return None

# Ambil SEMUA sample values untuk di-slice nanti
def _get_all_sample_values(fp: dict) -> List[str]:
    sv = fp.get("sampleValues") or []
    return [str(v) for v in sv if v is not None]

def _pick_sample_value(fp: dict) -> str:
    sv = fp.get("sampleValues") or []
    if isinstance(sv, list) and sv: return str(sv[0])
    return ""

def _profile_map_by_field(urn: str) -> dict:
    d = _graphql(DATASET_PROFILE_QUERY, {"urn": urn})
    if not d: return {}
    ds = d.get("dataset") or {}
    prof = (ds.get("datasetProfiles") or [None])[0]
    if not prof: return {}
    out = {}
    for fp in (prof.get("fieldProfiles") or []):
        key = _norm_field(fp.get("fieldPath"))
        if not key: continue
        out[key] = {
            "min": _as_float(fp.get("min")),
            "max": _as_float(fp.get("max")),
            "mean": _as_float(fp.get("mean")),
            "stddev": _as_float(fp.get("stdev")),
            "distinct_count": fp.get("uniqueCount"),
            "sample_value": _pick_sample_value(fp),     # Untuk Metadata (single)
            "all_sample_values": _get_all_sample_values(fp) # Untuk Keyword Search (list)
        }
    return out

def fetch_schema_via_rest(urn: str) -> Optional[dict]:
    url = f"{GMS_BASE}/entities/{quote(urn, safe='')}"
    params = {"aspect": "schemaMetadata", "version": "0"}
    try:
        resp = requests.get(url, headers=_headers_json(), params=params, timeout=TIMEOUT)
        if resp.status_code != 200: return None
        sm = (resp.json().get("entityResponse") or resp.json()).get("aspects", {}).get("schemaMetadata")
        return sm.get("value") if (sm and "value" in sm) else sm
    except: return None

def build_rest_tag_map(rest_schema: dict) -> Dict[str, List[str]]:
    out = {}
    if not rest_schema: return out
    for f in rest_schema.get("fields") or []:
        fp = f.get("fieldPath")
        if not fp: continue
        gtags = ((f.get("globalTags") or {}).get("tags") or [])
        names = []
        for gt in gtags:
            val = gt.get("tag")
            if isinstance(val, str): names.append(val)
            elif isinstance(val, dict):
                n = (val.get("properties") or {}).get("name") or val.get("urn")
                if n: names.append(n)
        out[fp] = names
    return out

def split_alias_and_role(raw_tags: List[str]) -> Tuple[List[str], List[str]]:
    alias_out, role_out = [], []
    for t in raw_tags:
        clean_urn = t.replace("urn:li:tag:", "")
        lowered = clean_urn.lower()
        if lowered.startswith("role:") or lowered.startswith("role."):
            clean = re.sub(r"^role[:.]", "", clean_urn, flags=re.IGNORECASE).strip()
            if clean: role_out.append(clean)
        elif lowered.startswith("alias:") or lowered.startswith("alias."):
            clean = re.sub(r"^alias[:.]", "", clean_urn, flags=re.IGNORECASE).strip()
            if clean: alias_out.append(clean)
        # ignore non alias/role tags for sparse keyword hygiene
    return alias_out, role_out


ROLE_CANONICAL = {
    "pk": "pk",
    "primary_key": "primary_key",
    "primary key": "primary_key",
    "fk": "fk",
    "foreign_key": "foreign_key",
    "foreign key": "foreign_key",
    "id": "id",
    "identifier": "identifier",
    "measure": "measure",
    "metric": "measure",
    "dimension": "dimension",
    "date": "date",
    "datetime": "date",
    "time": "date",
    "temporal": "temporal",
    "geo": "geo",
    "contact": "contact",
    "address": "address",
    "text": "text",
    "binary": "binary",
}


def normalize_role(role: str) -> str:
    key = str(role or "").strip().lower().replace("-", "_").replace(" ", "_")
    return ROLE_CANONICAL.get(key, key)


def _sanitize_fragment(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip().strip(".")


def _dedupe_keep_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for it in items:
        norm = it.lower()
        if norm and norm not in seen:
            seen.add(norm)
            out.append(it)
    return out


def _truncate_metadata_sample(value: Any, is_string: bool) -> str:
    sample = str(value or "")
    if not is_string:
        return sample
    if len(sample) <= SPARSE_METADATA_SAMPLE_MAX_CHARS:
        return sample
    return sample[:SPARSE_METADATA_SAMPLE_MAX_CHARS].rstrip() + "..."

# ==============================================================================
# 4. TRANSFORMATION LOGIC (SPARSE RETRIEVAL - STRICT LIMIT)
# ==============================================================================

def transform_to_sparse_schema(raw_row: Dict[str, Any], table_desc: str = "") -> Dict[str, Any]:
    """
    Sparse/Keyword Schema dengan STRICT LIMIT pada Sample Values.
    Updated: Menyimpan 'description' secara eksplisit di metadata.
    """
    dt = str(raw_row.get("datatype", "")).lower()
    is_numeric = any(k in dt for k in ('int', 'double', 'float', 'decimal', 'numeric', 'real', 'number', 'money', 'bit'))
    is_temporal = any(k in dt for k in ('date', 'time', 'datetime', 'timestamp', 'year'))
    is_string = any(k in dt for k in ('char', 'text', 'string', 'clob'))

    tbl = raw_row.get("table") or ""
    col = raw_row.get("column") or ""
    desc = _sanitize_fragment(raw_row.get("description"))
    aliases = _dedupe_keep_order([_sanitize_fragment(a) for a in raw_row.get("tags_alias", []) if _sanitize_fragment(a)])
    roles = _dedupe_keep_order([normalize_role(r) for r in raw_row.get("tags_role", []) if str(r or "").strip()])
    
    # --- KEYWORD CONSTRUCTION (Untuk Search BM25) ---
    keywords_list: List[str] = []
    
    # 1. Core Metadata
    keywords_list.append(_sanitize_fragment(tbl))
    keywords_list.append(_sanitize_fragment(col))
    if aliases: keywords_list.extend(aliases)
    if desc: keywords_list.append(desc) # Desc tetap masuk keyword agar bisa disearch
    if roles: keywords_list.extend(roles)

    # 2. Sample Values Logic (STRICT LIMIT)
    all_samples = raw_row.get("all_sample_values", [])
    
    if SPARSE_INCLUDE_SAMPLES_IN_KEYWORDS and all_samples:
        limit = SPARSE_NUMERIC_SAMPLE_LIMIT  # Default limit (Numeric/Date)

        if is_string:
            limit = SPARSE_STRING_SAMPLE_LIMIT
            distinct_raw = raw_row.get("distinct_count")
            if isinstance(distinct_raw, int) and distinct_raw > SPARSE_STRING_SAMPLE_DISTINCT_MAX:
                limit = min(limit, 3)
        elif is_numeric:
            limit = SPARSE_NUMERIC_SAMPLE_LIMIT

        filtered_samples: List[str] = []
        for sv in all_samples:
            s = _sanitize_fragment(sv)
            if not s:
                continue
            if len(s) > SPARSE_SAMPLE_VALUE_MAX_CHARS:
                continue
            if "\n" in s:
                continue
            filtered_samples.append(s)

        # Ambil Top-N samples
        limited_samples = filtered_samples[:limit]
        keywords_list.extend(limited_samples)

    # Gabungkan & Bersihkan
    keywords_list = [k for k in (_sanitize_fragment(k) for k in keywords_list) if k]
    keywords_list = _dedupe_keep_order(keywords_list)
    keywords_string = " ".join(keywords_list)
    keywords_string = re.sub(r'\s+', ' ', keywords_string).strip()

    # --- METADATA (PAYLOAD UNTUK LLM/RETRIEVAL) ---
    stats = {}
    stats["sample_value"] = _truncate_metadata_sample(raw_row.get("sample_value"), is_string)
    
    if is_numeric or is_temporal:
        stats["min"] = raw_row.get("min")
        stats["max"] = raw_row.get("max")
    
    if is_string:
        stats["distinct_count"] = raw_row.get("distinct_count")

    metadata = {
        "database": raw_row.get("database"),
        "table": tbl,
        "column": col,
        "datatype": raw_row.get("datatype"),
        "tags_alias": aliases,
        "tags_role": roles,
        # >>> INI PERUBAHANNYA: Masukkan description eksplisit <<<
        "description": desc,
        "stats": stats
    }

    return {
        "id": f"{tbl}.{col}",
        "keywords": keywords_string,
        "metadata": metadata
    }

# ==============================================================================
# 5. FETCH & PROCESS
# ==============================================================================
def fetch_dataset_columns_sparse(urn: str) -> List[Dict[str, Any]]:
    # A. Fetch Metadata
    d = _graphql(DATASET_META_QUERY, {"urn": urn}).get("dataset")
    if not d: return []

    table_desc = (d.get("properties") or {}).get("description") or ""
    platform = d.get("platform") or {}
    delim = (platform.get("properties") or {}).get("datasetNameDelimiter") or "."
    _, db, _, table = _split_from_urn(urn, (platform.get("name") or "").lower(), delim)

    # B. Profiles & Tags
    profile_map = _profile_map_by_field(urn)
    rest_schema = fetch_schema_via_rest(urn)
    rest_tags = build_rest_tag_map(rest_schema)
    
    # C. Editable Tags
    editable_tags = {}
    for e in (d.get("editableSchemaMetadata") or {}).get("editableSchemaFieldInfo") or []:
        fp = e.get("fieldPath")
        if fp:
            gtags = ((e.get("globalTags") or {}).get("tags") or [])
            names = [((gt.get("tag") or {}).get("properties") or {}).get("name") or (gt.get("tag") or {}).get("urn") for gt in gtags]
            editable_tags[fp] = [n for n in names if n]

    # D. Loop Columns
    sparse_rows = []
    sm = d.get("schemaMetadata") or {}
    for f in (sm.get("fields") or []):
        fp = f.get("fieldPath")
        prof = profile_map.get(_norm_field(fp), {})
        
        schema_tags = [((gt.get("tag") or {}).get("properties") or {}).get("name") or (gt.get("tag") or {}).get("urn") for gt in ((f.get("globalTags") or {}).get("tags") or [])]
        schema_tags = [t for t in schema_tags if t]
        
        all_tags = list(dict.fromkeys(schema_tags + editable_tags.get(fp, []) + rest_tags.get(fp, [])))
        t_alias, t_role = split_alias_and_role(all_tags)
        
        raw_row = {
            "database": db,
            "table": table,
            "column": fp,
            "datatype": f.get("nativeDataType") or f.get("type"),
            "description": f.get("description"), # Ini sumber description
            "tags_alias": t_alias,
            "tags_role": t_role,
            "min": prof.get("min"),
            "max": prof.get("max"),
            "mean": prof.get("mean"),
            "stddev": prof.get("stddev"),
            "sample_value": prof.get("sample_value"),       
            "all_sample_values": prof.get("all_sample_values", []), 
            "distinct_count": prof.get("distinct_count")
        }
        
        sparse_rows.append(transform_to_sparse_schema(raw_row, table_desc))
        
    return sparse_rows

def _sanitize_filename(name: str) -> str:
    if not name: return "unknown"
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

def main():
    output_dir = OUTPUT_DIR
    # If env points to a file path (e.g., /app/metadata_jsonl/classicmodels_dense.jsonl),
    # write sparse output next to that file instead of failing on makedirs.
    if output_dir.lower().endswith(".jsonl"):
        output_dir = os.path.dirname(output_dir) or "."
        print(f"[WARN] OUTPUT path looks like a file. Using parent directory for sparse export: {output_dir}")
    elif os.path.exists(output_dir) and os.path.isfile(output_dir):
        output_dir = os.path.dirname(output_dir) or "."
        print(f"[WARN] OUTPUT path is a file. Using parent directory for sparse export: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)
    writers = {}
    print(f"[INFO] Exporting SPARSE JSONL (Keywords) to: {os.path.abspath(output_dir)}")
    print(f"[INFO] Config: include samples in keywords = {SPARSE_INCLUDE_SAMPLES_IN_KEYWORDS}")
    print(f"[INFO] Config: sample limits (string={SPARSE_STRING_SAMPLE_LIMIT}, numeric={SPARSE_NUMERIC_SAMPLE_LIMIT}).")
    print("[INFO] Metadata includes explicit 'description'.")
    
    count = 0
    for urn in iter_dataset_urns():
        try:
            rows = fetch_dataset_columns_sparse(urn)
            if not rows: continue
            print(f"[proc] {urn} -> {len(rows)} cols")
            
            for row in rows:
                db_name = row["metadata"].get("database") or "unknown"
                fname = f"{_sanitize_filename(db_name)}_sparse.jsonl"
                
                if fname not in writers:
                    writers[fname] = open(os.path.join(output_dir, fname), "w", encoding="utf-8")
                
                writers[fname].write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
        except Exception as e:
            print(f"[ERROR] {urn}: {e}")

    for f in writers.values(): f.close()
    print(f"[DONE] Total {count} columns exported.")

if __name__ == "__main__":
    main()
