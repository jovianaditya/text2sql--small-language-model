# ============================================================
# ========================= utils.py =========================
# ============================================================
# Purpose:
# - General runtime utilities (timing, debug print, banners)
# - JSONL path resolution + JSONL reader
# - Metadata helpers (table/column extraction)
# - MCP payload parsing helpers
# - Ingest canonicalization helpers (stable hashing, sanitize)
# - CLI display helpers (tabulate preview)

# ======================================================================
# ====================== STANDARD LIBRARY IMPORT =======================
# ======================================================================
from __future__ import annotations
import re
import hashlib
import glob
import json
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional
from langchain_core.messages import BaseMessage
from collections import defaultdict
from tabulate import tabulate
# ======================================================================
# ======================== LOCAL MODULE IMPORT =========================
# ======================================================================
from .config import SQL_DEBUG

if TYPE_CHECKING:
    from .state import RAGState
# ======================================================================
# ======================= RUNTIME / DEBUG UTILITIES ====================
# ======================================================================
def _now() -> float:
    return perf_counter()

def _elapsed_ms(t0: float) -> int:
    return int((perf_counter() - t0) * 1000)

def _log_total(label: str, t0: float):
    ms = _elapsed_ms(t0)
    print(f"\n[TOTAL] {label}: {ms} ms (~{ms/1000:.3f}s)")

def _print_banner(title: str, width: int = 150) -> str:
    line = "─" * width
    print(f"\n{line}")
    print(title.center(width))
    print(line)
    return line

def _dbg_print_two(label: str, sql: str):
    if SQL_DEBUG:
        print(f"\n{label}")
        print(sql)
# ======================================================================
# ==================== JSONL PATH RESOLUTION + READER ==================
# ======================================================================
def resolve_jsonl_paths(spec: str) -> List[str]:
    paths: List[str] = []
    if not spec:
        return paths
    p = Path(spec)
    if any(ch in spec for ch in "*?[]"):
        paths.extend(glob.glob(spec))
    elif p.is_dir():
        paths.extend(str(x) for x in p.glob("*.jsonl"))
    elif p.is_file():
        paths.append(str(p))
    else:
        paths.extend(glob.glob(spec))
    out, seen = [], set()
    for x in paths:
        xp = str(Path(x))
        if xp.lower().endswith(".jsonl") and xp not in seen:
            seen.add(xp)
            out.append(xp)
    return sorted(out)

def read_jsonl(path: str) -> Iterable[tuple[str, Optional[Dict[str, Any]]]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                obj = None
            yield s, obj
# ======================================================================
# =================== METADATA ACCESSORS (TABLE/COLUMN) ================
# ======================================================================
def _meta_table(md: Dict[str, Any]) -> str:
    return str(md.get("table") or md.get("tableName") or "").strip()

def _meta_column(md: Dict[str, Any]) -> str:
    return str(md.get("column") or md.get("columnName") or "").strip()
# ======================================================================
# =================== MCP RESULT NORMALIZATION HELPERS =================
# ======================================================================
def _extract_payload(result) -> Any:
    if hasattr(result, "data") and result.data is not None:
        return result.data
    if hasattr(result, "structuredContent") and result.structuredContent is not None:
        return result.structuredContent
    if hasattr(result, "content") and result.content:
        return result.content[0].text if isinstance(result.content, list) else str(result.content)
    return result

def _ensure_json(d: Any) -> Optional[dict]:
    if isinstance(d, dict):
        return d
    if isinstance(d, str):
        try:
            return json.loads(d)
        except Exception:
            return None
    return None
# ======================================================================
# ======================== INGEST CANONICALIZATION =====================
# ======================================================================
# -------------------------------------------------------------
# 5A) Metadata parsing & normalization helpers
# -------------------------------------------------------------
def _parse_str_or_list_lower(x: Any) -> List[str]:
    """
    Normalisasi input yang bisa berupa:
    - list → jadikan list string lower-case
    - string JSON list → parse, lalu lower
    - string biasa → jadikan [string.lower()]
    - lainnya → []
    """
    if isinstance(x, str):
        s = x.strip()
        # coba parse JSON list kalau cocok
        try:
            j = json.loads(s)
            if isinstance(j, list):
                return [str(v).lower() for v in j]
            # kalau bukan list, fallback ke string itu sendiri
            return [s.lower()]
        except Exception:
            return [s.lower()]
    elif isinstance(x, list):
        return [str(v).lower() for v in x]
    elif x is None:
        return []
    else:
        return [str(x).lower()]

def _extract_roles_aliases(md: Dict[str, Any]) -> tuple[List[str], List[str]]:
    """
    Helper untuk ambil roles & aliases dalam bentuk list lowercase.
    Dipakai supaya nggak duplikasi pola get(tags_role/roles) & get(tags_alias/aliases).
    """
    roles = _parse_str_or_list_lower(md.get("tags_role") or md.get("roles") or [])
    aliases = _parse_str_or_list_lower(md.get("tags_alias") or md.get("aliases") or [])
    return roles, aliases

def _extract_aliases_for_map(md: Dict[str, Any]) -> List[str]:
    """
    Ambil aliases dari metadata (aliases/tags_alias/synonyms),
    pastikan bentuknya List[str] lowercase.
    """
    return _parse_str_or_list_lower(
        md.get("aliases") or md.get("tags_alias") or md.get("synonyms") or []
    )
# -------------------------------------------------------------
# 5B) Canonicalization & hashing helpers
# -------------------------------------------------------------
# kunci yang sering berubah & memicu upsert palsu → kita buang
VOLATILE_KEYS = {
    "_source", "updated_at", "lastModified", "timestamp", "ts"
}

def _norm_str(s: Any) -> str:
    s = "" if s is None else str(s)
    # rapikan whitespace tanpa mengubah isi
    s = "\n".join(re.sub(r"[ \t]{2,}", " ", line.rstrip()) for line in s.splitlines())
    return s.strip()

def _norm_list(x: Any) -> List[str]:
    # list stabil: json->list bila string JSON, trim, dedup casefold, lalu sort
    if isinstance(x, str):
        try:
            j = json.loads(x)
            x = j if isinstance(j, list) else [x]
        except Exception:
            x = [x]
    elif not isinstance(x, list):
        x = [x]
    cleaned, seen = [], set()
    for v in x:
        s = _norm_str(v)
        k = s.casefold()
        if s and k not in seen:
            seen.add(k)
            cleaned.append(s)
    return sorted(cleaned, key=str.casefold)

def _stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def _coerce_json_like(v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                v = json.loads(s)
            except Exception:
                return v
        else:
            return v
    if isinstance(v, list):
        return _norm_list(v)
    return v

def _sanitize_for_compare(meta: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (meta or {}).items():
        if k in VOLATILE_KEYS or k == "_canon_hash":
            continue
        vv = _coerce_json_like(v)
        out[k] = vv
    return out

def _sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (meta or {}).items():
        if k in VOLATILE_KEYS:
            continue
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = _norm_str(v) if isinstance(v, str) else v
        elif isinstance(v, list):
            out[k] = _stable_json(_norm_list(v))
        elif isinstance(v, dict):
            out[k] = _stable_json(v)
        else:
            out[k] = _norm_str(v)
    return out

def _canon_doc(s: str) -> str:
    return "\n".join(line.rstrip() for line in (s or "").splitlines()).strip()

def _canon_meta(d: Dict[str, Any]) -> str:
    try:
        normalized = _sanitize_for_compare(d or {})
        return _stable_json(normalized)
    except Exception:
        return str(d or {})

def _compute_canon_hash(doc: str, meta: Dict[str, Any]) -> str:
    a = _canon_doc(doc or "")
    b = _canon_meta(meta or {})
    payload = f"{a}\n{b}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()

# ======================================================================
# ================== CLI OUTPUT HELPERS (TABLE PREVIEW) ================
# ======================================================================
def display_table_from_json(data, max_cell_len: int = 80):
    preview_rows = data.get("preview_rows", [])
    preview_cols = data.get("preview_columns", [])
    total_rows = data.get("total_rows", len(preview_rows))
    total_cols = data.get("total_columns", len(preview_cols))

    def _shorten(val, max_len: int) -> str:
        if not isinstance(val, str):
            return val
        s = val.replace("\n", " ").strip()
        if len(s) <= max_len:
            return s
        return s[: max_len - 3] + "..."

    truncated_rows = [[_shorten(cell, max_cell_len) for cell in row] for row in preview_rows]

    if truncated_rows and preview_cols:
        print("\nQuery Result (preview):")
        print(
            tabulate(
                truncated_rows,
                headers=preview_cols,
                tablefmt="github",
                showindex=range(1, len(truncated_rows) + 1),
            )
        )
        print(
            f"\nShowing preview: {len(preview_rows)} rows × {len(preview_cols)} columns "
            f"(full result: {total_rows} rows × {total_cols} columns)"
        )
    else:
        print("Query executed but returned no rows.")
    
# Helper wajib untuk extract JSON dari teks LLM yang kotor
def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    try:
        # Cari pattern {...} termasuk nested braces
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
    except json.JSONDecodeError:
        return None
    return None

def try_llm_json(
    llm: Any,  # ChatOllama atau tipe LLM lain
    prompt: str,
    max_attempts: int = 1,
    oom_retry_wait: int = 0  # Seconds to wait if OOM detected (0 = no retry)
) -> Optional[Dict[str, Any]]:
    """
    Generic function untuk memanggil LLM dan mengharapkan return JSON.
    Tidak ada validasi key spesifik di sini (supaya bisa dipakai Router & SQL Generator).
    
    If oom_retry_wait > 0: Will wait and retry when OOM error is detected.
    """
    import time
    
    # Debug: Cek apakah model LLM valid
    if not llm:
        print("[LLM Error] LLM Object is None.")
        return None

    for attempt in range(max_attempts):
        try:
            # 1. Invoke LLM
            # print(f"[LLM Debug] Attempt {attempt+1}: Invoking LLM...") # Uncomment for verbose
            resp = llm.invoke(prompt)
            
            # Handle response types (String or Object)
            content = getattr(resp, "content", None) or str(resp)
            
            # 2. Extract JSON (Gunakan regex yang kuat)
            obj = _extract_json_object(content)
            
            if obj:
                return obj
            else:
                print(f"[LLM Warning] Gagal extract JSON dari response:\n{content[:100]}...")
                
        except Exception as e:
            error_msg = str(e).lower()
            # INI PENTING: Print error asli supaya tidak 'silent fail'
            print(f"[LLM CRITICAL ERROR] Attempt {attempt+1} failed: {e}")
            
            # Check for OOM error
            if "memory" in error_msg and oom_retry_wait > 0:
                print(f"[LLM OOM] Detected memory issue. Waiting {oom_retry_wait}s for model to unload...")
                time.sleep(oom_retry_wait)
                print(f"[LLM OOM] Retrying...")
                # Don't count this as an attempt - let outer loop handle retry
                continue

    return None

def build_router_schema(context_input) -> str:
    """
    Versi Hybrid:
    1. Support JSON (jika retrieval mengembalikan raw JSONL).
    2. Support Text Format: "- table: X | column: Y | desc: Z".
    """
    table_columns = defaultdict(list)
    
    # Normalisasi Input ke List of Strings
    lines = []
    if isinstance(context_input, str):
        lines = context_input.split("\n")
    elif isinstance(context_input, list):
        # Handle jika list of objects/dicts
        for item in context_input:
            if isinstance(item, str):
                lines.append(item)
            elif hasattr(item, 'page_content'): # Langchain Document
                lines.append(item.page_content)
            elif isinstance(item, dict):
                # Kalau sudah dict, convert balik ke string json biar seragam diproses loop bawah
                lines.append(json.dumps(item))
    
    # Regex untuk menangkap format teks kamu:
    # "- table: products | column: productCode | desc: Unique product code..."
    text_pattern = re.compile(r"table:\s*(?P<table>\w+)\s*\|\s*column:\s*(?P<col>\w+)\s*\|\s*desc:\s*(?P<desc>.*)", re.IGNORECASE)

    for line in lines:
        line = line.strip()
        if not line: continue
        
        # Variabel penampung hasil ekstraksi
        t_name, c_name, tags, description = None, None, [], ""
        
        # --- CARA 1: COBA PARSE JSON ---
        if line.startswith("{"):
            try:
                data = json.loads(line)
                t_name = data.get("table")
                c_name = data.get("column")
                tags = data.get("tags_role") or []
                description = data.get("description", "")
            except:
                pass # Lanjut ke cara 2
        
        # --- CARA 2: COBA PARSE TEXT REGEX (Sesuai format kamu) ---
        if not t_name: 
            match = text_pattern.search(line)
            if match:
                t_name = match.group("table")
                c_name = match.group("col")
                description = match.group("desc")
                # Di format teks, tags_role hilang, jadi kita andalkan description
                tags = [] 

        # --- LOGIC SELEKSI (AMBIL JIKA PENTING) ---
        if t_name and c_name:
            is_important = False
            
            # Normalisasi string untuk pengecekan
            desc_lower = description.lower() if description else ""
            col_lower = c_name.lower()
            
            # Cek 1: Primary Key
            if "id" in tags or "primary key" in desc_lower or "(pk)" in desc_lower:
                is_important = True
                
            # Cek 2: Foreign Key
            elif any(str(t).startswith("fk:") for t in tags) or "fk" in desc_lower or "foreign key" in desc_lower:
                is_important = True
            
            # Cek 3: Fallback Nama Kolom (Code, ID, Number)
            elif "code" in col_lower or "id" in col_lower or "number" in col_lower or "key" in col_lower:
                is_important = True

            if is_important:
                table_columns[t_name].append(c_name)

    # Format Output
    schema_parts = []
    for table, cols in table_columns.items():
        unique_cols = sorted(list(set(cols)))
        if unique_cols:
            schema_parts.append(f"{table}({', '.join(unique_cols)})")
            
    result = ", ".join(schema_parts)
    
    # Fallback jika kosong sama sekali (misal format context berubah total)
    if not result and len(lines) > 0:
        print("[Router Warning] Schema parsing return empty. Using simplified table listing.")
        # Coba ambil nama tabel aja dari baris "table: ..."
        tables_found = set(re.findall(r"table:\s*(\w+)", "\n".join(lines)))
        if tables_found:
            return ", ".join(list(tables_found))
            
    return result