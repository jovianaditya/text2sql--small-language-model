# ======================================================================
# ============================= config.py ==============================
# ======================================================================
# Purpose:
# - Configuration file for MCP client application.
# - Loads settings from environment variables with default fallbacks.
# - Defines constants for application behavior.
# - Organized into sections for clarity.
# - Handles settings for general app behavior, retrieval parameters, MCP server interaction, and debugging options.

# ======================================================================
# ====================== STANDARD LIBRARY IMPORT =======================
# ======================================================================
import os
import json


def _env_json_list(name: str, default: list[str]) -> list[str]:
    """
    Read list config from env.

    Priority:
    1) JSON array string
    2) Comma-separated string
    3) default
    """
    raw = os.getenv(name)
    if not raw:
        return default

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            items = [str(x).strip() for x in parsed if str(x).strip()]
            return items or default
    except Exception:
        pass

    items = [x.strip() for x in raw.split(",") if x.strip()]
    return items or default


def _env_json_dict(name: str, default: dict[str, str]) -> dict[str, str]:
    """
    Read dict config from env as JSON object; fallback to default.
    """
    raw = os.getenv(name)
    if not raw:
        return default

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            out: dict[str, str] = {}
            for k, v in parsed.items():
                key = str(k).strip()
                val = str(v).strip()
                if key and val:
                    out[key] = val
            return out or default
    except Exception:
        pass

    return default
# ======================================================================
# ============================ APP/GENERAL =============================
# ======================================================================
JSON_INPUT        = os.getenv("EXPORT_JSONL_OUTPUT", "./metadata_jsonl/*.jsonl").strip()
CHROMA_DIR        = os.getenv("CHROMA_DIR", "./chroma_db").strip()
COLLECTION_NAME   = os.getenv("COLLECTION_NAME", "classicmodels").strip()
OLLAMA_URL        = os.getenv("OLLAMA_URL", "http://localhost:11434").strip()
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "180s")  # keep_alive 90s, model stays loaded for batch eval
ROUTING_MODEL     = os.getenv("ROUTING_MODEL", "qwen2.5:1.5b-instruct").strip()
SMALL_MODEL       = os.getenv("SMALL_MODEL", "qwen2.5:3b-instruct").strip()
LARGE_MODEL       = os.getenv("LARGE_MODEL", "qwen2.5:7b-instruct").strip()
# ======================================================================
# ============================ RETRIEVAL ===============================
# ======================================================================
TOP_K            = int(os.getenv("TOP_K", "5"))
SPARSE_BONUS     = float(os.getenv("SPARSE_BONUS", "0.75"))
NAME_QUERY_BONUS = float(os.getenv("NAME_QUERY_BONUS", "0.3"))
ID_QUERY_BONUS   = float(os.getenv("ID_QUERY_BONUS", "0.4"))
# >>> Retrieval scoring tunables (externalized from retrieval.py)
COL_BOOST_DIM_NAME_EXPLICIT = float(os.getenv("COL_BOOST_DIM_NAME_EXPLICIT", "0.9"))
COL_BOOST_DIM_CONTACT_PENALTY = float(os.getenv("COL_BOOST_DIM_CONTACT_PENALTY", "-0.8"))
COL_BOOST_DIM_NAME_DEFAULT = float(os.getenv("COL_BOOST_DIM_NAME_DEFAULT", "0.6"))
COL_BOOST_DIM_ID_PENALTY = float(os.getenv("COL_BOOST_DIM_ID_PENALTY", "-0.35"))
COL_BOOST_OTHER_NAME_DEFAULT = float(os.getenv("COL_BOOST_OTHER_NAME_DEFAULT", "0.5"))
COL_BOOST_OTHER_ID_PENALTY = float(os.getenv("COL_BOOST_OTHER_ID_PENALTY", "-0.3"))
TABLE_SCORE_SPARSE_HIT = float(os.getenv("TABLE_SCORE_SPARSE_HIT", "0.5"))
TABLE_SCORE_TABLE_MENTION = float(os.getenv("TABLE_SCORE_TABLE_MENTION", "1.0"))
TABLE_SCORE_DESC_TOKEN = float(os.getenv("TABLE_SCORE_DESC_TOKEN", "0.3"))
TABLE_SCORE_MENTION_DIM = float(os.getenv("TABLE_SCORE_MENTION_DIM", "0.9"))
TABLE_SCORE_MENTION_TXN = float(os.getenv("TABLE_SCORE_MENTION_TXN", "0.7"))
TABLE_SCORE_MENTION_OTHER = float(os.getenv("TABLE_SCORE_MENTION_OTHER", "0.8"))
TABLE_SCORE_BRIDGE_PENALTY = float(os.getenv("TABLE_SCORE_BRIDGE_PENALTY", "-0.4"))
TABLE_SCORE_BRIDGE_NUMERIC_BOOST = float(os.getenv("TABLE_SCORE_BRIDGE_NUMERIC_BOOST", "0.6"))
MAX_TOP_TABLES = int(os.getenv("MAX_TOP_TABLES", "3"))
# >>> Batasi berapa banyak dokumen yang masuk ke context LLM
MAX_CTX_DOCS = int(os.getenv("MAX_CTX_DOCS", "12"))
SPARSE_MAX_HITS  = int(os.getenv("SPARSE_MAX_HITS", "20"))
# >>> Bisa dimatiin kalau mau lebih silent
SPARSE_DEBUG   = bool(int(os.getenv("SPARSE_DEBUG", "0")))
RETRIEVE_DEBUG = bool(int(os.getenv("RETRIEVE_DEBUG", "1")))
EXPORT_JSONL_OUTPUT = os.getenv("EXPORT_JSONL_OUTPUT", "classicmodels_dense.jsonl").strip()
# ======================================================================
# =============================== MCP ==================================
# ======================================================================
MCP_BASE    = os.getenv("MCP_BASE", "http://mcp_server:8000/mcp")
MCP_TIMEOUT = int(os.getenv("MCP_CLIENT_TIMEOUT", "60"))
# ======================================================================
# ========================= DRAFT/RETRY ================================
# ======================================================================
OOM_WAIT_TIME = int(os.getenv("OOM_WAIT_TIME", "35"))
OOM_MAX_RETRIES = int(os.getenv("OOM_MAX_RETRIES", "3"))
# ======================================================================
# ============================= DEBUG ==================================
# ======================================================================
SQL_DEBUG = bool(int(os.getenv("SQL_DEBUG", "1")))
COMPARE_DEBUG = bool(int(os.getenv("COMPARE_DEBUG", "0")))

# ======================================================================
# ========================= AGENT HEURISTICS ============================
# ======================================================================
AGENT_INSTANT_REJECT_PATTERNS = _env_json_list(
    "AGENT_INSTANT_REJECT_PATTERNS",
    [
        r"^\s*(hi|hello|hey|halo|hai|hallo)\s*[!.,]?\s*$",
        r"^\s*(good\s*)?(morning|afternoon|evening|night)\s*[!.,]?\s*$",
        r"^\s*(selamat\s+)?(pagi|siang|sore|malam)\s*[!.,]?\s*$",
        r"^\s*(apa\s+kabar|how\s+are\s+you|what'?s\s+up)\s*[?!.,]?\s*$",
        r"^\s*(thanks?|thank\s+you|terima\s*kasih|makasih)\s*[!.,]?\s*$",
        r"^\s*(bye|goodbye|see\s+you|sampai\s+jumpa)\s*[!.,]?\s*$",
        r"^\s*(yes|no|ya|tidak|ok|okay|oke|sip)\s*[!.,]?\s*$",
        r"^\s*(test|testing|tes|coba)\s*[!.,]?\s*$",
        r"^\s*[!?.,:;\-_=+*#@&]+\s*$",
        r"^\s*\d+\s*$",
        r"^\s*.{0,2}\s*$",
    ],
)

AGENT_DEFINITE_SQL_PATTERNS = _env_json_list(
    "AGENT_DEFINITE_SQL_PATTERNS",
    [
        r"\b(show|list|display|get|find|retrieve|fetch|count|total|sum|average)\b",
        r"\b(tampilkan|tunjukkan|lihat|ambil|cari|hitung|berapa)\b",
        r"\b(select|from|where|join|group by|order by)\b",
        r"\b(highest|lowest|top|most|least|largest|smallest|latest|earliest)\b",
        r"\b(tertinggi|terendah|terbesar|terkecil|terbaru|terlama)\b",
    ],
)

AGENT_GIBBERISH_KEYBOARD_PATTERNS = _env_json_list(
    "AGENT_GIBBERISH_KEYBOARD_PATTERNS",
    ["qwert", "asdfg", "zxcvb", "qazwsx", "yxcvbn", "hjkl", "yuiop"],
)

AGENT_MISSING_COL_SUFFIX_SYNONYMS = _env_json_dict(
    "AGENT_MISSING_COL_SUFFIX_SYNONYMS",
    {
        "number": r"(number|no|id|code|kode|nomor)",
        "code": r"(code|id|no|number|kode|nomor)",
        "id": r"(id|no|code|number|kode|nomor)",
        "date": r"(date|tanggal|tgl|waktu)",
        "name": r"(name|nama)",
        "amount": r"(amount|total|jumlah|nilai|harga)",
        "extension": r"(extension|ext)",
    },
)

AGENT_REGEX_TRIGGERS = _env_json_list(
    "AGENT_REGEX_TRIGGERS",
    [
        r"exactly\s+\w+\s+digits",
        r"followed\s+by\s+\w+\s+digits",
        r"\d+\s+digits",
        r"pattern",
        r"format",
    ],
)

# ======================================================================
# ======================== RETRIEVE HEURISTICS ==========================
# ======================================================================
RETRIEVAL_EXTRA_STOPWORDS = set(
    _env_json_list(
        "RETRIEVAL_EXTRA_STOPWORDS",
        [
            "show", "all", "list", "their", "that", "have", "no",
            "include", "includes", "including", "sort", "sorted", "sorting",
            "result", "results", "record", "records", "row", "rows",
            "display", "return", "get", "give", "find", "fetch", "please", "kindly",
            "alphabetical", "alphabetically", "ascending", "descending", "distinct",
            "yang", "dan", "atau", "ke", "dari", "di", "itu", "ini",
            "tampilkan", "daftar",
        ],
    )
)

RETRIEVAL_GENERIC_MENTION_TERMS = set(
    _env_json_list(
        "RETRIEVAL_GENERIC_MENTION_TERMS",
        [
            "name", "names", "id", "ids", "code", "codes", "number", "numbers",
            "date", "dates", "time", "status", "type", "types", "value", "values",
            "amount", "amounts", "total", "totals", "count", "counts", "description",
            "descriptions", "detail", "details", "record", "records", "row", "rows",
            "address", "city", "country", "state", "phone", "email", "contact",
        ],
    )
)

RETRIEVAL_JOIN_SIGNALS = _env_json_list(
    "RETRIEVAL_JOIN_SIGNALS",
    [
        "join", "gabung", "combine", "combine with", "along with", "together with",
        "and their", "with their", "beserta", "bersama", "dengan",
        "for each", "per ", "setiap",
    ],
)
