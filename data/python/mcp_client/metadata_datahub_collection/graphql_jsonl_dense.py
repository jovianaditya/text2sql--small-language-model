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
# Prefer dedicated dense output env var, then fallback to generic export env.
OUTPUT_DIR = os.getenv("EXPORT_DENSE_JSONL_OUTPUT", os.getenv("EXPORT_JSONL_OUTPUT", "./exports_dense_final"))
TABLE_CONTEXT_MAX_CHARS = int(os.getenv("DATAHUB_TABLE_CONTEXT_MAX_CHARS", "280"))
SAMPLE_TEXT_MAX_CHARS = int(os.getenv("DATAHUB_SAMPLE_TEXT_MAX_CHARS", "64"))
SAMPLE_DISTINCT_MAX = int(os.getenv("DATAHUB_SAMPLE_DISTINCT_MAX", "10"))

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

# ✨ ADDED: properties { description } untuk mengambil deskripsi tabel
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
            "sample_value": _pick_sample_value(fp),
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
        # FIX: Jangan split(':')[-1], karena itu membuang prefix 'alias' jika tag-nya 'alias:cost'
        # Cukup buang header urn-nya saja.
        clean_urn = t.replace("urn:li:tag:", "")
        
        lowered = clean_urn.lower()
        
        # Cek Role
        if lowered.startswith("role:") or lowered.startswith("role."):
            val = re.sub(r"^role[:.]", "", clean_urn, flags=re.IGNORECASE).strip()
            if val: role_out.append(val)
            
        # Cek Alias
        elif lowered.startswith("alias:") or lowered.startswith("alias."):
            val = re.sub(r"^alias[:.]", "", clean_urn, flags=re.IGNORECASE).strip()
            if val: alias_out.append(val)
            
        # Ignore non alias/role tags to avoid noisy synonym injection.
            
    return alias_out, role_out
# ==============================================================================
# 4. TRANSFORMATION LOGIC (GOLD STANDARD FOR DENSE RETRIEVAL)
# ==============================================================================

ROLE_CANONICAL = {
    "pk": "pk",
    "primary_key": "primary_key",
    "primary key": "pk",
    "fk": "fk",
    "foreign_key": "foreign_key",
    "foreign key": "fk",
    "id": "id",
    "identifier": "identifier",
    "measure": "measure",
    "metric": "measure",
    "dimension": "dimension",
    "date": "date",
    "time": "date",
    "datetime": "date",
    "temporal": "temporal",
    "geo": "geo",
    "location": "geo",
    "contact": "contact",
    "address": "address",
    "text": "text",
    "binary": "binary",
}

PRIMARY_ROLE_CANONICAL = {
    "pk": "pk",
    "primary_key": "pk",
    "fk": "fk",
    "foreign_key": "fk",
    "id": "id",
    "identifier": "id",
    "measure": "measure",
    "metric": "measure",
    "amount": "measure",
    "quantity": "measure",
    "inventory": "measure",
    "price": "measure",
    "monetary": "measure",
    "threshold": "measure",
    "transaction_amount": "measure",
    "transaction_quantity": "measure",
    "date": "date",
    "time": "date",
    "datetime": "date",
    "temporal": "date",
    "transaction_date": "date",
    "due_date": "date",
    "fulfillment_date": "date",
    "geo": "geo",
    "contact": "contact",
    "address": "address",
    "text": "text",
    "binary": "binary",
    "dimension": "dimension",
    "name": "dimension",
    "entity_name": "dimension",
    "status": "dimension",
    "classification": "dimension",
}

ROLE_PRIORITY = [
    "pk",
    "fk",
    "id",
    "measure",
    "date",
    "geo",
    "contact",
    "address",
    "text",
    "dimension",
    "binary",
]


def _normalize_role_label(role: str) -> str:
    raw = str(role or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not raw:
        return ""
    base = raw.split(":", 1)[0].replace(".", "_").strip("_")
    return ROLE_CANONICAL.get(base, ROLE_CANONICAL.get(raw, raw))


def _normalize_roles(roles: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for role in roles or []:
        norm = _normalize_role_label(role)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def _normalize_aliases(aliases: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for alias in aliases or []:
        clean = str(alias or "").strip()
        key = clean.lower()
        if not clean or key in seen:
            continue
        seen.add(key)
        out.append(clean)
    return out


def _primary_role_bucket(role: str) -> str:
    return PRIMARY_ROLE_CANONICAL.get(str(role or "").strip().lower(), "")


def _infer_fallback_role(raw_row: Dict[str, Any], fk_target: Optional[str]) -> str:
    col = str(raw_row.get("column") or "").strip().lower()
    desc = str(raw_row.get("description") or "").strip().lower()
    dt = str(raw_row.get("datatype") or "").strip().lower()
    joined = f"{col} {desc}"

    if re.search(r"\bprimary\s+key\b", desc):
        return "pk"
    if fk_target or re.search(r"\bforeign\s+key\b|\breferences\b", desc):
        return "fk"
    if re.search(r"(date|time|year|month|day)", dt) or re.search(r"\b(date|time|year|month|day)\b", col):
        return "date"
    if re.search(r"\b(email|phone|fax|contact)\b", joined):
        return "contact"
    if re.search(r"\b(address|city|state|postal|zip|country)\b", joined):
        return "address"
    if re.search(r"\b(id|code|number)\b", col):
        return "id"
    if re.search(r"\b(char|text|clob|json|xml)\b", dt):
        return "text"

    is_numeric = any(k in dt for k in ("int", "double", "float", "decimal", "numeric", "real", "number", "money"))
    if is_numeric and re.search(
        r"\b(amount|price|cost|total|quantity|qty|stock|inventory|volume|balance|rate|ratio|score|limit|percent)\b",
        joined,
    ):
        return "measure"

    return "dimension"


def _select_primary_role(raw_row: Dict[str, Any], roles: List[str], fk_target: Optional[str]) -> str:
    normalized = _normalize_roles(roles)

    primary_candidates = []
    seen_primary = set()
    for role in normalized:
        primary = _primary_role_bucket(role)
        if not primary or primary in seen_primary:
            continue
        seen_primary.add(primary)
        primary_candidates.append(primary)

    for preferred in ROLE_PRIORITY:
        if preferred in primary_candidates:
            return preferred
    if normalized:
        return _primary_role_bucket(normalized[0]) or normalized[0]
    return _infer_fallback_role(raw_row, fk_target)


def _extract_fk_target(text: str) -> Optional[str]:
    """
    Extract FK target in normalized form 'table.column' from free-form text.
    Supports variants such as:
    - 'FK to table.column'
    - 'foreign key to table.column'
    - 'references table.column'
    """
    src = str(text or "")
    patterns = [
        r"\bfk\s+to\s+([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b",
        r"\bforeign\s+key\s+to\s+([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b",
        r"\breferences\s+([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b",
    ]
    for pat in patterns:
        m = re.search(pat, src, flags=re.IGNORECASE)
        if m:
            return f"{m.group(1)}.{m.group(2)}"
    return None


def _has_fk_literal(text: str, fk_target: str) -> bool:
    """Check whether text already contains an explicit 'FK to table.column' literal."""
    if not text or not fk_target:
        return False
    escaped = re.escape(fk_target)
    return bool(re.search(rf"\bfk\s+to\s+{escaped}\b", text, flags=re.IGNORECASE))

def transform_to_dense_schema(raw_row: Dict[str, Any], table_desc: str = "") -> Dict[str, Any]:
    """
    Gold Standard: Context Injection + Metadata Separation
    """
    # 1. Type Detection
    dt = str(raw_row.get("datatype", "")).lower()
    is_numeric = any(k in dt for k in ('int', 'double', 'float', 'decimal', 'numeric', 'real', 'number', 'money'))
    is_string = any(k in dt for k in ('char', 'text', 'string', 'clob'))

    # 2. Prepare Text Components
    tbl = raw_row.get("table")
    col = raw_row.get("column")
    desc = str(raw_row.get("description") or "")
    desc_clean = re.sub(r"\s+", " ", desc).strip().rstrip(" .")
    aliases = _normalize_aliases(raw_row.get("tags_alias", []))
    roles = raw_row.get("tags_role", [])
    normalized_roles = _normalize_roles(roles)
    fk_target = _extract_fk_target(desc_clean)
    primary_role = _select_primary_role(raw_row, normalized_roles, fk_target)
    
    # 3. Sample Value Logic (For Text Embedding)
    # Only inject sample into text if it's categorical (distinct < 20)
    distinct_raw = raw_row.get("distinct_count")
    sample_raw = str(raw_row.get("sample_value", ""))
    sample_sentence = ""
    if (
        isinstance(distinct_raw, int)
        and 0 < distinct_raw <= SAMPLE_DISTINCT_MAX
        and len(sample_raw) <= SAMPLE_TEXT_MAX_CHARS
        and "\n" not in sample_raw
        and sample_raw.lower() not in ["", "unknown", "none", "null"]
    ):
        sample_sentence = f"Example value: {sample_raw}."

    # 4. Context Injection (Table Description)
    # Jika DataHub punya deskripsi tabel, masukkan ke embedding kolom
    table_part = f"Table: {tbl}."
    if table_desc:
        clean_tbl_desc = table_desc.replace("\n", " ").strip()
        if len(clean_tbl_desc) > 3: # Validasi panjang minimal
            if len(clean_tbl_desc) > TABLE_CONTEXT_MAX_CHARS:
                clean_tbl_desc = clean_tbl_desc[:TABLE_CONTEXT_MAX_CHARS].rstrip() + "..."
            table_part = f"Table: {tbl} (Context: {clean_tbl_desc})."

    # 5. Build Embedding Sentence
    alias_phrases = [f"{alias}." for alias in aliases if str(alias or "").strip()]
    text_parts = [
        table_part,
        f"Column: {col}.",
        f"{desc_clean}." if desc_clean else "",
        f"FK to {fk_target}." if (fk_target and not _has_fk_literal(desc_clean, fk_target)) else "",
        *alias_phrases,
        sample_sentence,
    ]
    full_text = " ".join([p for p in text_parts if p]).strip()

    # 6. Build Metadata Payload (Stats separation)
    stats = {}
    stats["sample_value"] = sample_raw # Always keep raw sample in metadata
    
    if is_numeric:
        stats["min"] = raw_row.get("min")
        stats["max"] = raw_row.get("max")
        stats["mean"] = raw_row.get("mean")
        stats["stddev"] = raw_row.get("stddev")
    
    if is_string:
        stats["distinct_count"] = raw_row.get("distinct_count")

    metadata = {
        "database": raw_row.get("database"),
        "table": tbl,
        "column": col,
        "datatype": raw_row.get("datatype"),
        "description": desc_clean,
        "primary_role": primary_role,
        "tags_role": normalized_roles,
        "tags_alias": aliases,
        "stats": stats
    }

    return {
        "id": f"{tbl}.{col}",
        "text": full_text,
        "metadata": metadata
    }

# ==============================================================================
# 5. FETCH & PROCESS
# ==============================================================================
def fetch_dataset_columns_dense(urn: str) -> List[Dict[str, Any]]:
    # A. Fetch Metadata
    d = _graphql(DATASET_META_QUERY, {"urn": urn}).get("dataset")
    if not d: return []

    # Get Table Description (Global Context)
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
    dense_rows = []
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
            "description": f.get("description"),
            "tags_alias": t_alias,
            "tags_role": t_role,
            "min": prof.get("min"),
            "max": prof.get("max"),
            "mean": prof.get("mean"),
            "stddev": prof.get("stddev"),
            "sample_value": prof.get("sample_value"),
            "distinct_count": prof.get("distinct_count")
        }
        
        # Pass table_desc down for context injection
        dense_rows.append(transform_to_dense_schema(raw_row, table_desc))
        
    return dense_rows

def _sanitize_filename(name: str) -> str:
    if not name: return "unknown"
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

def main():
    output_dir = OUTPUT_DIR
    # If env points to a file path (e.g., /app/metadata_jsonl/classicmodels_dense.jsonl),
    # write dense output next to that file instead of creating a directory with that name.
    if output_dir.lower().endswith(".jsonl"):
        output_dir = os.path.dirname(output_dir) or "."
        print(f"[WARN] OUTPUT path looks like a file. Using parent directory for dense export: {output_dir}")
    elif os.path.exists(output_dir) and os.path.isfile(output_dir):
        output_dir = os.path.dirname(output_dir) or "."
        print(f"[WARN] OUTPUT path is a file. Using parent directory for dense export: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)
    writers = {}
    print(f"[INFO] Exporting Gold Standard Dense JSONL to: {os.path.abspath(output_dir)}")
    
    count = 0
    for urn in iter_dataset_urns():
        try:
            rows = fetch_dataset_columns_dense(urn)
            if not rows: continue
            print(f"[proc] {urn} -> {len(rows)} cols")
            
            for row in rows:
                db_name = row["metadata"].get("database") or "unknown"
                fname = f"{_sanitize_filename(db_name)}_dense.jsonl"
                
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
