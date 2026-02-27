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
TABLE_SCORE_SPARSE_HIT = float(os.getenv("TABLE_SCORE_SPARSE_HIT", "0.35"))
TABLE_SCORE_TABLE_MENTION = float(os.getenv("TABLE_SCORE_TABLE_MENTION", "1.3"))
TABLE_SCORE_DESC_TOKEN = float(os.getenv("TABLE_SCORE_DESC_TOKEN", "0.3"))
TABLE_SCORE_MENTION_DIM = float(os.getenv("TABLE_SCORE_MENTION_DIM", "0.9"))
TABLE_SCORE_MENTION_TXN = float(os.getenv("TABLE_SCORE_MENTION_TXN", "0.7"))
TABLE_SCORE_MENTION_OTHER = float(os.getenv("TABLE_SCORE_MENTION_OTHER", "0.8"))
TABLE_SCORE_BRIDGE_PENALTY = float(os.getenv("TABLE_SCORE_BRIDGE_PENALTY", "-0.5"))
TABLE_SCORE_BRIDGE_NUMERIC_BOOST = float(os.getenv("TABLE_SCORE_BRIDGE_NUMERIC_BOOST", "0.3"))
MAX_TOP_TABLES = int(os.getenv("MAX_TOP_TABLES", "3"))
# >>> Batasi berapa banyak dokumen yang masuk ke context LLM
MAX_CTX_DOCS = int(os.getenv("MAX_CTX_DOCS", "12"))
SPARSE_MAX_HITS  = int(os.getenv("SPARSE_MAX_HITS", "20"))
# Single-table lock confidence (4C)
RETRIEVAL_SINGLE_LOCK_MIN_SCORE_GAP = float(
    os.getenv("RETRIEVAL_SINGLE_LOCK_MIN_SCORE_GAP", "0.6")
)
RETRIEVAL_SINGLE_LOCK_MAX_NON_PRIMARY_SPARSE_HITS = int(
    os.getenv("RETRIEVAL_SINGLE_LOCK_MAX_NON_PRIMARY_SPARSE_HITS", "4")
)
RETRIEVAL_SINGLE_LOCK_FALLBACK_TOPK = int(
    os.getenv("RETRIEVAL_SINGLE_LOCK_FALLBACK_TOPK", "1")
)
# >>> Bisa dimatiin kalau mau lebih silent
SPARSE_DEBUG   = bool(int(os.getenv("SPARSE_DEBUG", "0")))
RETRIEVE_DEBUG = bool(int(os.getenv("RETRIEVE_DEBUG", "1")))
EXPORT_JSONL_OUTPUT = os.getenv("EXPORT_JSONL_OUTPUT", "classicmodels_dense.jsonl").strip()

RETRIEVAL_STATS_MAX_SAMPLES = int(os.getenv("RETRIEVAL_STATS_MAX_SAMPLES", "5"))
RETRIEVAL_DESC_KEYWORD_HEAD_WORDS = int(os.getenv("RETRIEVAL_DESC_KEYWORD_HEAD_WORDS", "5"))
RETRIEVAL_STATS_KEY_MIN = os.getenv("RETRIEVAL_STATS_KEY_MIN", "min").strip()
RETRIEVAL_STATS_KEY_MAX = os.getenv("RETRIEVAL_STATS_KEY_MAX", "max").strip()
RETRIEVAL_STATS_KEY_MEAN = os.getenv("RETRIEVAL_STATS_KEY_MEAN", "mean").strip()
RETRIEVAL_STATS_KEY_DISTINCT_COUNT = os.getenv("RETRIEVAL_STATS_KEY_DISTINCT_COUNT", "distinct_count").strip()
RETRIEVAL_META_DESCRIPTION_KEY = os.getenv("RETRIEVAL_META_DESCRIPTION_KEY", "description").strip()
RETRIEVAL_META_KEYWORDS_KEY = os.getenv("RETRIEVAL_META_KEYWORDS_KEY", "keywords").strip()
RETRIEVAL_DOC_DESCRIPTION_MARKER = os.getenv("RETRIEVAL_DOC_DESCRIPTION_MARKER", "Description:")
RETRIEVAL_DOC_DESC_END_MARKER = os.getenv("RETRIEVAL_DOC_DESC_END_MARKER", "..")
RETRIEVAL_DOC_SYNONYMS_MARKER = os.getenv("RETRIEVAL_DOC_SYNONYMS_MARKER", "Synonyms:")
RETRIEVAL_DOC_PRIMARY_KEY_MARKER = os.getenv("RETRIEVAL_DOC_PRIMARY_KEY_MARKER", "Primary Key")
RETRIEVAL_CARDS_MAX_DOCS = int(os.getenv("RETRIEVAL_CARDS_MAX_DOCS", "15"))
RETRIEVAL_CARDS_MAX_INJECT = int(os.getenv("RETRIEVAL_CARDS_MAX_INJECT", "12"))
RETRIEVAL_CARDS_DESC_MAX_LEN = int(os.getenv("RETRIEVAL_CARDS_DESC_MAX_LEN", "100"))
RETRIEVAL_CARDS_DEBUG_PREVIEW_COUNT = int(os.getenv("RETRIEVAL_CARDS_DEBUG_PREVIEW_COUNT", "5"))
RETRIEVAL_DENSE_DESC_FALLBACK_LEN = int(os.getenv("RETRIEVAL_DENSE_DESC_FALLBACK_LEN", "100"))
RETRIEVAL_SPARSE_DEBUG_PREVIEW_LIMIT = int(os.getenv("RETRIEVAL_SPARSE_DEBUG_PREVIEW_LIMIT", "10"))
RETRIEVAL_RETRIEVE_DEBUG_MAX_SHOW = int(os.getenv("RETRIEVAL_RETRIEVE_DEBUG_MAX_SHOW", "7"))
RETRIEVAL_AUTO_INJECT_BONUS_DELTA = float(os.getenv("RETRIEVAL_AUTO_INJECT_BONUS_DELTA", "0.05"))
RETRIEVAL_TOP_TABLE_MIN_SCORE = float(os.getenv("RETRIEVAL_TOP_TABLE_MIN_SCORE", "0.0"))
RETRIEVAL_BRIDGE_MIN_TABLE_COUNT = int(os.getenv("RETRIEVAL_BRIDGE_MIN_TABLE_COUNT", "2"))
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
# IR-first deterministic planner
AGENT_IR_ENABLED = bool(int(os.getenv("AGENT_IR_ENABLED", "1")))
AGENT_IR_MIN_CONFIDENCE = float(os.getenv("AGENT_IR_MIN_CONFIDENCE", "0.80"))
AGENT_IR_DEBUG = bool(int(os.getenv("AGENT_IR_DEBUG", "1")))
AGENT_IR_ALLOW_MULTI_TABLE = bool(int(os.getenv("AGENT_IR_ALLOW_MULTI_TABLE", "1")))
AGENT_IR_MULTI_TABLE_MIN_CONFIDENCE = float(os.getenv("AGENT_IR_MULTI_TABLE_MIN_CONFIDENCE", "0.90"))
AGENT_IR_MAX_TABLES = int(os.getenv("AGENT_IR_MAX_TABLES", "2"))

# AST guardrails (sqlglot-based safety checks in refine phase)
AGENT_AST_GUARDS_ENABLED = bool(int(os.getenv("AGENT_AST_GUARDS_ENABLED", "1")))
AGENT_AST_GUARDS_DEBUG = bool(int(os.getenv("AGENT_AST_GUARDS_DEBUG", "0")))
AGENT_AST_SEMANTIC_FIRST = bool(int(os.getenv("AGENT_AST_SEMANTIC_FIRST", "1")))
AGENT_NORMALIZE_REWRITE_BUDGET = int(os.getenv("AGENT_NORMALIZE_REWRITE_BUDGET", "1"))
AGENT_NORMALIZE_EARLY_STOP = bool(int(os.getenv("AGENT_NORMALIZE_EARLY_STOP", "1")))
AGENT_REGEN_PRECHECK_ENABLED = bool(int(os.getenv("AGENT_REGEN_PRECHECK_ENABLED", "1")))
# ======================================================================
# ============================= DEBUG ==================================
# ======================================================================
SQL_DEBUG = bool(int(os.getenv("SQL_DEBUG", "1")))
COMPARE_DEBUG = bool(int(os.getenv("COMPARE_DEBUG", "0")))

# Alias output style
ALIAS_MAX_LEN = int(os.getenv("ALIAS_MAX_LEN", "36"))

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

AGENT_GUARD_EXAMPLE_MAX_ITEMS = int(os.getenv("AGENT_GUARD_EXAMPLE_MAX_ITEMS", "3"))
AGENT_GUARD_EXAMPLE_YEAR = int(os.getenv("AGENT_GUARD_EXAMPLE_YEAR", "2023"))
AGENT_GUARD_EXAMPLE_VALUE_THRESHOLD = int(os.getenv("AGENT_GUARD_EXAMPLE_VALUE_THRESHOLD", "100"))
AGENT_GUARD_BANNER_CHAR = os.getenv("AGENT_GUARD_BANNER_CHAR", "\u2500")
AGENT_GUARD_BANNER_WIDTH = int(os.getenv("AGENT_GUARD_BANNER_WIDTH", "150"))
AGENT_GUARD_LLM_TEMPERATURE = float(os.getenv("AGENT_GUARD_LLM_TEMPERATURE", "0.0"))
AGENT_GUARD_LLM_NUM_CTX = int(os.getenv("AGENT_GUARD_LLM_NUM_CTX", "512"))
AGENT_GUARD_LLM_NUM_PREDICT = int(os.getenv("AGENT_GUARD_LLM_NUM_PREDICT", "64"))
AGENT_GUARD_SCHEMA_EXAMPLE_MAX_ITEMS = int(os.getenv("AGENT_GUARD_SCHEMA_EXAMPLE_MAX_ITEMS", "6"))

AGENT_GIBBERISH_MIN_TEXT_LENGTH = int(os.getenv("AGENT_GIBBERISH_MIN_TEXT_LENGTH", "3"))
AGENT_GIBBERISH_SENTENCE_WORD_BYPASS = int(os.getenv("AGENT_GIBBERISH_SENTENCE_WORD_BYPASS", "3"))
AGENT_GIBBERISH_MIN_ALPHA_LENGTH = int(os.getenv("AGENT_GIBBERISH_MIN_ALPHA_LENGTH", "3"))
AGENT_GIBBERISH_VOWEL_RATIO_MIN = float(os.getenv("AGENT_GIBBERISH_VOWEL_RATIO_MIN", "0.12"))
AGENT_GIBBERISH_VOWEL_CHECK_MIN_LEN = int(os.getenv("AGENT_GIBBERISH_VOWEL_CHECK_MIN_LEN", "5"))
AGENT_GIBBERISH_CONSONANT_RUN_MIN = int(os.getenv("AGENT_GIBBERISH_CONSONANT_RUN_MIN", "7"))
AGENT_GIBBERISH_CONSONANT_CHECK_MIN_LEN = int(os.getenv("AGENT_GIBBERISH_CONSONANT_CHECK_MIN_LEN", "6"))
AGENT_GIBBERISH_CONSONANT_CLASS = os.getenv(
    "AGENT_GIBBERISH_CONSONANT_CLASS",
    "bcdfghjklmnpqrstvwxyz",
).strip().lower()
AGENT_GIBBERISH_NO_SPACE_LEN = int(os.getenv("AGENT_GIBBERISH_NO_SPACE_LEN", "15"))
AGENT_GIBBERISH_REPEAT_MIN_LEN = int(os.getenv("AGENT_GIBBERISH_REPEAT_MIN_LEN", "6"))
AGENT_GIBBERISH_REPEAT_PATTERN_LENGTHS = [
    int(x) for x in _env_json_list("AGENT_GIBBERISH_REPEAT_PATTERN_LENGTHS", ["2", "3"])
    if str(x).strip().isdigit()
] or [2, 3]
AGENT_GIBBERISH_VOWELS = os.getenv("AGENT_GIBBERISH_VOWELS", "aiueo").strip().lower()
AGENT_GIBBERISH_ALPHA_ONLY_PATTERN = os.getenv(
    "AGENT_GIBBERISH_ALPHA_ONLY_PATTERN",
    r"[^a-z]",
).strip()
AGENT_GIBBERISH_HAS_UPPER_PATTERN = os.getenv(
    "AGENT_GIBBERISH_HAS_UPPER_PATTERN",
    r"[A-Z]",
).strip()
AGENT_GIBBERISH_HAS_SEPARATOR_PATTERN = os.getenv(
    "AGENT_GIBBERISH_HAS_SEPARATOR_PATTERN",
    r"[_\-]",
).strip()
AGENT_TABLE_VARIANT_MIN_TAIL_LEN = int(os.getenv("AGENT_TABLE_VARIANT_MIN_TAIL_LEN", "3"))
AGENT_COLUMN_ORDERED_TOKEN = os.getenv("AGENT_COLUMN_ORDERED_TOKEN", "ordered").strip().lower()
AGENT_COLUMN_ORDERED_BY_PATTERN = os.getenv(
    "AGENT_COLUMN_ORDERED_BY_PATTERN",
    r"\bordered\s+by\b",
).strip()

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

# Additional low-risk retrieval literals (externalized from retrieval.py)
RETRIEVAL_NUMERIC_SQL_TYPES = set(
    _env_json_list(
        "RETRIEVAL_NUMERIC_SQL_TYPES",
        [
            "int", "integer", "tinyint", "smallint", "mediumint", "bigint",
            "decimal", "numeric", "float", "double", "double precision",
            "real", "money", "smallmoney", "bit",
        ],
    )
)

RETRIEVAL_BM25_STEM_EXCEPTIONS_BASE = set(
    _env_json_list(
        "RETRIEVAL_BM25_STEM_EXCEPTIONS_BASE",
        ["status", "address", "business", "sales"],
    )
)

RETRIEVAL_BM25_STOPWORD_EXCEPTIONS = set(
    _env_json_list(
        "RETRIEVAL_BM25_STOPWORD_EXCEPTIONS",
        ["name", "names"],
    )
)

RETRIEVAL_STEM_SUFFIX_IES = os.getenv("RETRIEVAL_STEM_SUFFIX_IES", "ies").strip().lower()
RETRIEVAL_STEM_SUFFIX_IES_MIN_LEN = int(os.getenv("RETRIEVAL_STEM_SUFFIX_IES_MIN_LEN", "5"))
RETRIEVAL_STEM_SUFFIX_IES_TO = os.getenv("RETRIEVAL_STEM_SUFFIX_IES_TO", "y").strip().lower()
RETRIEVAL_STEM_SUFFIX_S = os.getenv("RETRIEVAL_STEM_SUFFIX_S", "s").strip().lower()
RETRIEVAL_STEM_SUFFIX_S_MIN_LEN = int(os.getenv("RETRIEVAL_STEM_SUFFIX_S_MIN_LEN", "4"))
RETRIEVAL_STEM_SUFFIX_S_EXCLUDE = os.getenv("RETRIEVAL_STEM_SUFFIX_S_EXCLUDE", "ss").strip().lower()
RETRIEVAL_TOKENIZE_ALNUM_PATTERN = os.getenv(
    "RETRIEVAL_TOKENIZE_ALNUM_PATTERN",
    r"[a-z0-9]+",
).strip()
RETRIEVAL_NORMALIZE_THOUSANDS_PATTERN = os.getenv(
    "RETRIEVAL_NORMALIZE_THOUSANDS_PATTERN",
    r"(\d)[,\.](\d{3})(\b|$)",
).strip()
RETRIEVAL_NORMALIZE_POSSESSIVE_PATTERN = os.getenv(
    "RETRIEVAL_NORMALIZE_POSSESSIVE_PATTERN",
    r"(\w)'s\b",
).strip()
RETRIEVAL_NORMALIZE_NON_ALNUM_PATTERN = os.getenv(
    "RETRIEVAL_NORMALIZE_NON_ALNUM_PATTERN",
    r"[^a-z0-9_]+",
).strip()
RETRIEVAL_DESC_CAMEL_SPLIT_PATTERN_1 = os.getenv(
    "RETRIEVAL_DESC_CAMEL_SPLIT_PATTERN_1",
    r"(.)([A-Z][a-z]+)",
).strip()
RETRIEVAL_DESC_CAMEL_SPLIT_PATTERN_2 = os.getenv(
    "RETRIEVAL_DESC_CAMEL_SPLIT_PATTERN_2",
    r"([a-z0-9])([A-Z])",
).strip()

RETRIEVAL_NAME_QUERY_TERMS = _env_json_list(
    "RETRIEVAL_NAME_QUERY_TERMS",
    ["name"],
)

RETRIEVAL_ID_LIKE_QUERY_TERMS = _env_json_list(
    "RETRIEVAL_ID_LIKE_QUERY_TERMS",
    [" id", "id ", " id,", "identifier", "code", "number"],
)

RETRIEVAL_NAME_REQUEST_TERMS = _env_json_list(
    "RETRIEVAL_NAME_REQUEST_TERMS",
    ["name", "names", "nama"],
)

RETRIEVAL_CONTACT_NAME_TERMS = _env_json_list(
    "RETRIEVAL_CONTACT_NAME_TERMS",
    [
        "contact name", "contact first", "contact last",
        "first name", "last name", "person name",
    ],
)

RETRIEVAL_NAME_ROLE_HINT_TERMS = set(
    _env_json_list(
        "RETRIEVAL_NAME_ROLE_HINT_TERMS",
        ["name"],
    )
)

RETRIEVAL_NAME_COLUMN_HINT_TERMS = set(
    _env_json_list(
        "RETRIEVAL_NAME_COLUMN_HINT_TERMS",
        ["name"],
    )
)

RETRIEVAL_ID_ROLE_HINT_TERMS = set(
    _env_json_list(
        "RETRIEVAL_ID_ROLE_HINT_TERMS",
        ["id"],
    )
)

RETRIEVAL_ID_COLUMN_SUFFIX_TERMS = _env_json_list(
    "RETRIEVAL_ID_COLUMN_SUFFIX_TERMS",
    ["id", "code", "number"],
)

RETRIEVAL_ID_COLUMN_SUBSTRING_TERMS = set(
    _env_json_list(
        "RETRIEVAL_ID_COLUMN_SUBSTRING_TERMS",
        ["number"],
    )
)

RETRIEVAL_ID_BOOST_COLUMN_SUFFIX_TERMS = _env_json_list(
    "RETRIEVAL_ID_BOOST_COLUMN_SUFFIX_TERMS",
    ["id"],
)

RETRIEVAL_ID_BOOST_COLUMN_SUBSTRING_TERMS = set(
    _env_json_list(
        "RETRIEVAL_ID_BOOST_COLUMN_SUBSTRING_TERMS",
        ["number"],
    )
)

RETRIEVAL_CONTACT_COLUMN_HINT_TERMS = set(
    _env_json_list(
        "RETRIEVAL_CONTACT_COLUMN_HINT_TERMS",
        ["contact"],
    )
)

RETRIEVAL_DIMENSION_LIKE_TABLE_TYPES = set(
    _env_json_list(
        "RETRIEVAL_DIMENSION_LIKE_TABLE_TYPES",
        ["dimension", "unknown"],
    )
)

RETRIEVAL_ID_BOOST_QUERY_TERMS = _env_json_list(
    "RETRIEVAL_ID_BOOST_QUERY_TERMS",
    [" id ", "id ", " id,", "identifier", "code"],
)

RETRIEVAL_NUMERIC_SIGNAL_KEYWORDS = set(
    _env_json_list(
        "RETRIEVAL_NUMERIC_SIGNAL_KEYWORDS",
        ["quantity", "price", "amount", "total", "cost", "value", "each", "rate"],
    )
)

RETRIEVAL_NUMERIC_PATTERN_KEYWORDS = set(
    _env_json_list(
        "RETRIEVAL_NUMERIC_PATTERN_KEYWORDS",
        ["quantity", "stock", "amount", "price", "cost", "value"],
    )
)

RETRIEVAL_NUMERIC_EXCLUDE_COLUMN_KEYWORDS = set(
    _env_json_list(
        "RETRIEVAL_NUMERIC_EXCLUDE_COLUMN_KEYWORDS",
        ["id", "code", "key"],
    )
)

RETRIEVAL_DESC_GENERIC_TERMS = set(
    _env_json_list(
        "RETRIEVAL_DESC_GENERIC_TERMS",
        ["number", "total", "each", "per", "for", "the", "a", "an"],
    )
)

RETRIEVAL_AGGREGATE_QUERY_TERMS = set(
    _env_json_list(
        "RETRIEVAL_AGGREGATE_QUERY_TERMS",
        ["sum", "avg", "average", "total"],
    )
)
RETRIEVAL_COUNT_QUERY_TERMS = _env_json_list(
    "RETRIEVAL_COUNT_QUERY_TERMS",
    ["count"],
)
RETRIEVAL_MULTI_ENTITY_MIN_TABLES = int(os.getenv("RETRIEVAL_MULTI_ENTITY_MIN_TABLES", "2"))

RETRIEVAL_BASE_K_FLOOR = int(os.getenv("RETRIEVAL_BASE_K_FLOOR", "12"))
RETRIEVAL_DENSE_K_FLOOR = int(os.getenv("RETRIEVAL_DENSE_K_FLOOR", "28"))
RETRIEVAL_SPARSE_FILENAME_MARKER = os.getenv("RETRIEVAL_SPARSE_FILENAME_MARKER", "_sparse").strip()
RETRIEVAL_FK_PATH_MAX_DEPTH = int(os.getenv("RETRIEVAL_FK_PATH_MAX_DEPTH", "3"))

# Additional low-risk agent literals (externalized from agents.py)
AGENT_ROUTER_MATH_KEYWORDS = _env_json_list(
    "AGENT_ROUTER_MATH_KEYWORDS",
    [
        "sum", "count", "total", "avg", "average", "number of", "jumlah", "rata-rata",
        "percentage", "percent", "ratio", "proportion", "persentase", "rasio",
    ],
)

AGENT_ROUTER_GROUP_INDICATORS = _env_json_list(
    "AGENT_ROUTER_GROUP_INDICATORS",
    ["each", "per", "every", "masing-masing", "setiap", "for each", "classify"],
)

AGENT_ROUTER_VALUE_RANK_TERMS = _env_json_list(
    "AGENT_ROUTER_VALUE_RANK_TERMS",
    [
        "highest", "lowest", "top", "bottom", "most", "least", "largest", "smallest",
        "terbesar", "terkecil", "paling", "tertinggi", "terendah",
    ],
)

AGENT_ROUTER_TIME_RANK_TERMS = _env_json_list(
    "AGENT_ROUTER_TIME_RANK_TERMS",
    ["latest", "earliest", "newest", "oldest", "terbaru", "terlama"],
)

AGENT_ROUTER_EXPLICIT_RANK_PATTERNS = _env_json_list(
    "AGENT_ROUTER_EXPLICIT_RANK_PATTERNS",
    [
        r"\ba\s+rank\s+of\b",
        r"\brank\s+\w+\s+by\b",
        r"\branking\s+of\b",
        r"\breturn.*\brank\b.*\bwithin\b",
        r"\brank\s+.*\bbased\s+on\b",
        r"\brank\s+the\s+\w+\b",
        r"\branked\s+by\b",
    ],
)

AGENT_ROUTER_ORDINAL_WORDS = _env_json_list(
    "AGENT_ROUTER_ORDINAL_WORDS",
    ["second", "third", "fourth", "fifth", "2nd", "3rd", "4th", "5th", "kedua", "ketiga", "keempat", "kelima"],
)

AGENT_ROUTER_ORDINAL_TARGET_TERMS = _env_json_list(
    "AGENT_ROUTER_ORDINAL_TARGET_TERMS",
    [
        "highest", "lowest", "largest", "smallest", "biggest", "top", "bottom",
        "best", "worst", "most", "least", "terbesar", "terkecil", "tertinggi", "terendah",
    ],
)

AGENT_ROUTER_WITHIN_SAME_PATTERN = os.getenv(
    "AGENT_ROUTER_WITHIN_SAME_PATTERN",
    r"\b(within|in)\s+the\s+same\s+(\w+)",
).strip()
AGENT_ROUTER_BY_TOKEN = (os.getenv("AGENT_ROUTER_BY_TOKEN", "by ") or "by ").lower()

AGENT_ROUTER_EXPLICIT_SORT_TERMS = _env_json_list(
    "AGENT_ROUTER_EXPLICIT_SORT_TERMS",
    ["sort by", "ordered by", "sorted by", "order by", "sorted from"],
)

AGENT_ROUTER_CROSS_TABLE_EXCLUSION_TERMS = _env_json_list(
    "AGENT_ROUTER_CROSS_TABLE_EXCLUSION_TERMS",
    ["never ordered", "never paid", "without any", "have no", "has no", "tidak pernah", "belum pernah", "tidak ada", "yang tidak punya"],
)

AGENT_SQL_RESERVED_EXTRA_TOKENS = set(
    _env_json_list(
        "AGENT_SQL_RESERVED_EXTRA_TOKENS",
        [
            "year", "month", "day", "date", "date_format", "now", "curdate",
            "concat", "group_concat", "coalesce", "ifnull", "sum", "count", "avg",
            "t", "limit", "offset", "as", "regexp_like", "str_to_date",
            "case", "when", "then", "else", "end", "union", "all", "distinct",
            "values", "exists", "having", "substr",
            # Window function grammar (not schema columns)
            "over", "partition", "range", "rows", "row", "between",
            "unbounded", "preceding", "following", "current",
        ],
    )
)

# Additional retrieval heuristics (low-risk literal centralization)
RETRIEVAL_ID_ROLE_TAGS = set(
    _env_json_list(
        "RETRIEVAL_ID_ROLE_TAGS",
        ["id", "key", "primary", "identifier"],
    )
)

RETRIEVAL_METRIC_ROLE_TAGS = set(
    _env_json_list(
        "RETRIEVAL_METRIC_ROLE_TAGS",
        ["metric", "measure", "amount", "quantity"],
    )
)

RETRIEVAL_NUMERIC_TYPE_HINTS = set(
    _env_json_list(
        "RETRIEVAL_NUMERIC_TYPE_HINTS",
        ["int", "decimal", "numeric", "float", "double"],
    )
)

RETRIEVAL_TABLE_SCORING_MATH_KEYWORDS = _env_json_list(
    "RETRIEVAL_TABLE_SCORING_MATH_KEYWORDS",
    [
        "sum", "count", "total", "avg", "average", "number of", "jumlah", "rata-rata",
        "percentage", "percent", "ratio", "proportion", "persentase", "rasio",
    ],
)

RETRIEVAL_NAME_ONLY_ID_PATTERN = os.getenv(
    "RETRIEVAL_NAME_ONLY_ID_PATTERN",
    r"\b(ids?|numbers?|codes?|identifier|nomor|kode|no)\b",
).strip()

RETRIEVAL_NAME_ONLY_NAME_PATTERN = os.getenv(
    "RETRIEVAL_NAME_ONLY_NAME_PATTERN",
    r"\b(name|names|nama)\b",
).strip()

RETRIEVAL_NAME_ONLY_NUMBER_OF_PATTERN = os.getenv(
    "RETRIEVAL_NAME_ONLY_NUMBER_OF_PATTERN",
    r"\b(total\s+)?number\s+of\b",
).strip()

RETRIEVAL_DEBUG_FOCUS_COLUMNS = set(
    _env_json_list(
        "RETRIEVAL_DEBUG_FOCUS_COLUMNS",
        ["quantityinstock", "buyprice", "msrp"],
    )
)

# Additional agent heuristics (low-risk literal centralization)
AGENT_COLUMN_SEMANTIC_SUFFIX_KEYWORDS = {
    "name": set(_env_json_list("AGENT_COLUMN_SUFFIX_NAME_TERMS", ["name", "names", "nama"])),
    "date": set(_env_json_list("AGENT_COLUMN_SUFFIX_DATE_TERMS", ["date", "dates", "tanggal", "tgl", "time", "waktu"])),
    "number": set(_env_json_list("AGENT_COLUMN_SUFFIX_NUMBER_TERMS", ["number", "numbers", "id", "ids", "code", "codes", "nomor", "kode", "no"])),
    "code": set(_env_json_list("AGENT_COLUMN_SUFFIX_CODE_TERMS", ["code", "codes", "id", "ids", "number", "numbers", "nomor", "kode", "no"])),
    "id": set(_env_json_list("AGENT_COLUMN_SUFFIX_ID_TERMS", ["id", "ids", "number", "numbers", "code", "codes", "nomor", "kode", "no"])),
}

AGENT_COLUMN_STOP_TOKENS = set(
    _env_json_list(
        "AGENT_COLUMN_STOP_TOKENS",
        [
            "id", "ids", "code", "codes", "number", "numbers",
            "name", "names", "date", "dates", "line", "lines",
            "type", "types", "status", "each", "table", "order",
            "quantity", "price", "amount", "total", "average", "value",
        ],
    )
)

AGENT_COLUMN_TOKEN_SUFFIX_Y = os.getenv("AGENT_COLUMN_TOKEN_SUFFIX_Y", "y").strip().lower()
AGENT_COLUMN_TOKEN_SUFFIX_IES = os.getenv("AGENT_COLUMN_TOKEN_SUFFIX_IES", "ies").strip().lower()
AGENT_COLUMN_TOKEN_SUFFIX_S = os.getenv("AGENT_COLUMN_TOKEN_SUFFIX_S", "s").strip().lower()
AGENT_COLUMN_TOKEN_SUFFIX_Y_MIN_LEN = int(os.getenv("AGENT_COLUMN_TOKEN_SUFFIX_Y_MIN_LEN", "4"))
AGENT_COLUMN_TOKEN_SUFFIX_IES_MIN_LEN = int(os.getenv("AGENT_COLUMN_TOKEN_SUFFIX_IES_MIN_LEN", "5"))
AGENT_COLUMN_TOKEN_SUFFIX_S_MIN_LEN = int(os.getenv("AGENT_COLUMN_TOKEN_SUFFIX_S_MIN_LEN", "5"))
AGENT_COLUMN_CAMEL_SPLIT_PATTERN = os.getenv(
    "AGENT_COLUMN_CAMEL_SPLIT_PATTERN",
    r"([a-z0-9])([A-Z])",
).strip()
AGENT_COLUMN_CAMEL_SPLIT_LOWER_PATTERN = os.getenv(
    "AGENT_COLUMN_CAMEL_SPLIT_LOWER_PATTERN",
    r"([a-z])([A-Z])",
).strip()
AGENT_COLUMN_TOKEN_PATTERN = os.getenv(
    "AGENT_COLUMN_TOKEN_PATTERN",
    r"[a-z0-9]+",
).strip()

AGENT_ROUTER_FK_SUFFIX_TERMS = _env_json_list(
    "AGENT_ROUTER_FK_SUFFIX_TERMS",
    ["number", "code", "id"],
)

AGENT_ROUTER_IGNORE_PATTERNS = {
    "order": _env_json_list(
        "AGENT_ROUTER_IGNORE_PATTERNS_ORDER",
        [r"\border(ed|s)?\s+by\b", r"\b(descending|ascending)\s+order\b", r"\bin\s+order\b"],
    ),
    "group": _env_json_list(
        "AGENT_ROUTER_IGNORE_PATTERNS_GROUP",
        [r"\bgroup\s+by\b"],
    ),
    "sort": _env_json_list(
        "AGENT_ROUTER_IGNORE_PATTERNS_SORT",
        [r"\bsort(ed)?\s+by\b"],
    ),
    "rank": _env_json_list(
        "AGENT_ROUTER_IGNORE_PATTERNS_RANK",
        [r"\brank\s+over\b"],
    ),
}

AGENT_ROUTER_EVIDENCE_TAG_FOR_EACH = os.getenv(
    "AGENT_ROUTER_EVIDENCE_TAG_FOR_EACH",
    "'for each' pattern",
).strip()

AGENT_ROUTER_EVIDENCE_TAG_TABLE_ATTRIBUTE = os.getenv(
    "AGENT_ROUTER_EVIDENCE_TAG_TABLE_ATTRIBUTE",
    "table+attribute pattern",
).strip()

AGENT_ROUTER_EVIDENCE_TAG_AGGREGATE_CONTEXT = os.getenv(
    "AGENT_ROUTER_EVIDENCE_TAG_AGGREGATE_CONTEXT",
    "aggregate context",
).strip()

AGENT_ROUTER_FOR_EACH_ENTITY_PATTERNS = _env_json_list(
    "AGENT_ROUTER_FOR_EACH_ENTITY_PATTERNS",
    [
        r"\bfor\s+each\s+(?:\w+\s+)?__ENTITY__\b",
        r"\bper\s+(?:\w+\s+)?__ENTITY__\b",
        r"\beach\s+(?:\w+\s+)?__ENTITY__\b",
    ],
)

AGENT_ROUTER_TABLE_ATTRIBUTE_PATTERNS = _env_json_list(
    "AGENT_ROUTER_TABLE_ATTRIBUTE_PATTERNS",
    [
        r"\b__ENTITY__\b\s+(name|number|id|code)\b",
    ],
)

AGENT_ROUTER_AGGREGATE_CONTEXT_PATTERNS = _env_json_list(
    "AGENT_ROUTER_AGGREGATE_CONTEXT_PATTERNS",
    [
        r"\b(total|sum|count|average|avg)\s+(of\s+)?__ENTITY__\b",
        r"\b(total|sum|count|average|avg)\s+(of\s+)?__ENTITY__\s+\w+",
        r"\bnumber\s+of\s+__ENTITY__\b",
        r"\b__ENTITY__\b\s+(total|sum|count|average)\b",
    ],
)

AGENT_ROUTER_ID_REFERENCE_TERMS = _env_json_list(
    "AGENT_ROUTER_ID_REFERENCE_TERMS",
    ["number", "id", "code", "kode", "nomor"],
)

AGENT_ROUTER_FK_VALUE_LIST_PATTERN = os.getenv(
    "AGENT_ROUTER_FK_VALUE_LIST_PATTERN",
    r"[\d,\sor]+",
).strip()

AGENT_ROUTER_PRIMARY_FROM_IN_PATTERNS = _env_json_list(
    "AGENT_ROUTER_PRIMARY_FROM_IN_PATTERNS",
    [
        r"\b\w+\s+(from|in)\s+__ENTITY__\b",
        r"\b__ENTITY__\b\s+who\s+",
        r"\b__ENTITY__\b\s+that\s+",
    ],
)

AGENT_ROUTER_PRIMARY_ALL_ENTITY_PATTERN = os.getenv(
    "AGENT_ROUTER_PRIMARY_ALL_ENTITY_PATTERN",
    r"\ball\s+__ENTITY__\b",
).strip()

AGENT_ROUTER_PRIMARY_REASON_FROM_IN = os.getenv(
    "AGENT_ROUTER_PRIMARY_REASON_FROM_IN",
    "from/in pattern",
).strip()

AGENT_ROUTER_PRIMARY_REASON_ALL = os.getenv(
    "AGENT_ROUTER_PRIMARY_REASON_ALL",
    "'all X' pattern",
).strip()

AGENT_ROUTER_PRIMARY_REASON_EXPLICIT_SUBJECT = os.getenv(
    "AGENT_ROUTER_PRIMARY_REASON_EXPLICIT_SUBJECT",
    "explicit subject pattern",
).strip()

AGENT_ROUTER_EXPLICIT_SUBJECT_ENTITY_PATTERNS = _env_json_list(
    "AGENT_ROUTER_EXPLICIT_SUBJECT_ENTITY_PATTERNS",
    [
        r"^\s*(?:show|list|display|return|get|find|fetch)\s+(?:only\s+|all\s+)?(?:\w+\s+){0,2}__ENTITY__\b",
        r"^\s*(?:which|what)\s+(?:\w+\s+){0,2}__ENTITY__\b",
    ],
)

AGENT_ROUTER_FK_EVIDENCE_EXTRACT_PATTERN = os.getenv(
    "AGENT_ROUTER_FK_EVIDENCE_EXTRACT_PATTERN",
    r"^FK column '([^']+)'$",
).strip()

AGENT_ROUTER_FK_EVIDENCE_PREFIX = os.getenv(
    "AGENT_ROUTER_FK_EVIDENCE_PREFIX",
    "FK column",
).strip()

AGENT_ROUTER_FK_PATTERN_CARD = os.getenv(
    "AGENT_ROUTER_FK_PATTERN_CARD",
    r"table:\s*(\w+)\s*\|\s*column:\s*(\w+).*?FK to (\w+)\.(\w+)",
).strip()

AGENT_ROUTER_FK_PATTERN_DOT = os.getenv(
    "AGENT_ROUTER_FK_PATTERN_DOT",
    r"(\w+)\.(\w+).*?FK to (\w+)\.(\w+)",
).strip()

AGENT_ROUTER_WEAK_EVIDENCE_TAGS = set(
    _env_json_list(
        "AGENT_ROUTER_WEAK_EVIDENCE_TAGS",
        [AGENT_ROUTER_EVIDENCE_TAG_FOR_EACH, AGENT_ROUTER_EVIDENCE_TAG_TABLE_ATTRIBUTE],
    )
)

# Medium-risk heuristics (externalized with behavior-preserving defaults)
RETRIEVAL_TABLE_NAME_PREFIX_STRIP_CHARS = os.getenv(
    "RETRIEVAL_TABLE_NAME_PREFIX_STRIP_CHARS",
    "s",
)

RETRIEVAL_TABLE_VARIANT_MIN_TAIL_LEN = int(
    os.getenv("RETRIEVAL_TABLE_VARIANT_MIN_TAIL_LEN", "3")
)

RETRIEVAL_COLUMN_VARIANT_REPLACEMENTS = _env_json_dict(
    "RETRIEVAL_COLUMN_VARIANT_REPLACEMENTS",
    {
        "quantity": "",
        "instock": "in stock",
    },
)

AGENT_ROUTER_WINNER_ROW_RECORD_PATTERN = os.getenv(
    "AGENT_ROUTER_WINNER_ROW_RECORD_PATTERN",
    r"\b(record|row|entry|item)\s+with\s+(the\s+)?(largest|highest|lowest|smallest|biggest)",
).strip()

AGENT_ROUTER_WINNER_ROW_FOR_EACH_PATTERN = os.getenv(
    "AGENT_ROUTER_WINNER_ROW_FOR_EACH_PATTERN",
    r"\b(largest|highest|lowest|smallest|biggest)\s+\w+\s+(for\s+each|per)\b",
).strip()

AGENT_ROUTER_TIEBREAKER_PATTERN = os.getenv(
    "AGENT_ROUTER_TIEBREAKER_PATTERN",
    r"\bif\s+(multiple|several|there\s+are\s+ties?|same)\b.*\b(choose|pick|select|take)\s+(the\s+)?(one|latest|earliest|first|last)\b",
).strip()

AGENT_ROUTER_MISSING_EMPTY_PATTERN = os.getenv(
    "AGENT_ROUTER_MISSING_EMPTY_PATTERN",
    r"\b(missing|empty)\b",
).strip()

AGENT_ROUTER_ESCALATE_ON_MULTI_TABLE = bool(int(os.getenv("AGENT_ROUTER_ESCALATE_ON_MULTI_TABLE", "1")))
AGENT_ROUTER_ESCALATE_ON_WINNER_ROW = bool(int(os.getenv("AGENT_ROUTER_ESCALATE_ON_WINNER_ROW", "1")))
AGENT_ROUTER_ESCALATE_ON_EXCLUSION = bool(int(os.getenv("AGENT_ROUTER_ESCALATE_ON_EXCLUSION", "1")))
AGENT_ROUTER_ESCALATE_ON_WINDOW_AGGREGATE = bool(int(os.getenv("AGENT_ROUTER_ESCALATE_ON_WINDOW_AGGREGATE", "1")))
AGENT_ROUTER_ESCALATE_ON_EXPLICIT_RANK = bool(int(os.getenv("AGENT_ROUTER_ESCALATE_ON_EXPLICIT_RANK", "1")))
AGENT_ROUTER_ESCALATE_ON_SUBQUERY_INTENT = bool(int(os.getenv("AGENT_ROUTER_ESCALATE_ON_SUBQUERY_INTENT", "1")))
AGENT_ROUTER_ALLOW_SIMPLE_MULTI_TABLE_3B = bool(int(os.getenv("AGENT_ROUTER_ALLOW_SIMPLE_MULTI_TABLE_3B", "1")))
AGENT_ROUTER_SIMPLE_MULTI_TABLE_MAX_TABLES = int(os.getenv("AGENT_ROUTER_SIMPLE_MULTI_TABLE_MAX_TABLES", "3"))
AGENT_ROUTER_ALLOW_SIMPLE_SINGLE_TABLE_SUBQUERY_3B = bool(
    int(os.getenv("AGENT_ROUTER_ALLOW_SIMPLE_SINGLE_TABLE_SUBQUERY_3B", "1"))
)
AGENT_ROUTER_SUBQUERY_INTENT_PATTERN = os.getenv(
    "AGENT_ROUTER_SUBQUERY_INTENT_PATTERN",
    r"\b(above|below|greater\s+than|less\s+than)\s+the\s+average\b|\bcompared\s+to\s+the\s+average\b|\brelative\s+to\s+the\s+average\b|\bthan\s+the\s+overall\s+average\b|\bexists\b|\bnot\s+exists\b",
).strip()
AGENT_ROUTER_SIMPLE_SINGLE_TABLE_SUBQUERY_PATTERN = os.getenv(
    "AGENT_ROUTER_SIMPLE_SINGLE_TABLE_SUBQUERY_PATTERN",
    r"\b(greater|less|higher|lower|above|below)\s+than\s+the\s+(average|overall\s+average|mean)\b|\b(compared|relative)\s+to\s+the\s+(average|overall\s+average|mean)\b|\bthan\s+the\s+overall\s+average\b",
).strip()
AGENT_ROUTER_COMPLEX_SUBQUERY_CUE_PATTERN = os.getenv(
    "AGENT_ROUTER_COMPLEX_SUBQUERY_CUE_PATTERN",
    r"\b(exists|not\s+exists|latest|most\s+recent|newest|oldest|earliest|for\s+each|per)\b",
).strip()

AGENT_ROUTER_COMPLEXITY_SIMPLE = os.getenv("AGENT_ROUTER_COMPLEXITY_SIMPLE", "simple").strip()
AGENT_ROUTER_COMPLEXITY_COMPLEX = os.getenv("AGENT_ROUTER_COMPLEXITY_COMPLEX", "complex").strip()
AGENT_ROUTER_COMPLEXITY_REJECTED = os.getenv("AGENT_ROUTER_COMPLEXITY_REJECTED", "rejected").strip()
AGENT_ROUTER_QTYPE_LOOKUP = os.getenv("AGENT_ROUTER_QTYPE_LOOKUP", "lookup").strip()
AGENT_ROUTER_QTYPE_AGGREGATE = os.getenv("AGENT_ROUTER_QTYPE_AGGREGATE", "aggregate").strip()
AGENT_ROUTER_QTYPE_RANK = os.getenv("AGENT_ROUTER_QTYPE_RANK", "rank").strip()
AGENT_ROUTER_QTYPE_INVALID = os.getenv("AGENT_ROUTER_QTYPE_INVALID", "invalid").strip()

AGENT_DRAFT_STRIP_DESC_ASC_BEFORE_FROM_PATTERN = os.getenv(
    "AGENT_DRAFT_STRIP_DESC_ASC_BEFORE_FROM_PATTERN",
    r"\s+(DESC|ASC)\s*(?=FROM)",
).strip()

AGENT_DRAFT_SELECT_FROM_PATTERN = os.getenv(
    "AGENT_DRAFT_SELECT_FROM_PATTERN",
    r"\bSELECT\b(.*?)\bFROM\b",
).strip()

AGENT_DRAFT_STRIP_DESC_ASC_BEFORE_COMMA_PATTERN = os.getenv(
    "AGENT_DRAFT_STRIP_DESC_ASC_BEFORE_COMMA_PATTERN",
    r"\s+(DESC|ASC)\s*(?=,)",
).strip()

AGENT_DRAFT_LIMIT_NULL_ZERO_PATTERN = os.getenv(
    "AGENT_DRAFT_LIMIT_NULL_ZERO_PATTERN",
    r"LIMIT\s+(NULL|0)",
).strip()

AGENT_DRAFT_LIMIT_CAPTURE_PATTERN = os.getenv(
    "AGENT_DRAFT_LIMIT_CAPTURE_PATTERN",
    r"LIMIT\s+(\d+)",
).strip()

AGENT_DRAFT_LIMIT_REMOVE_PATTERN = os.getenv(
    "AGENT_DRAFT_LIMIT_REMOVE_PATTERN",
    r"LIMIT\s+\d+\s*;?",
).strip()

AGENT_DRAFT_LIMIT_INTENT_TERMS = _env_json_list(
    "AGENT_DRAFT_LIMIT_INTENT_TERMS",
    ["top", "best", "highest", "lowest", "limit"],
)
AGENT_DRAFT_TRY_JSON_MAX_ATTEMPTS = int(os.getenv("AGENT_DRAFT_TRY_JSON_MAX_ATTEMPTS", "2"))

AGENT_KEEP_ALIVE_PARSE_PATTERN = os.getenv(
    "AGENT_KEEP_ALIVE_PARSE_PATTERN",
    r"(\d+)([smh])",
).strip()
AGENT_PREWARM_MEMORY_ERROR_TOKEN = os.getenv("AGENT_PREWARM_MEMORY_ERROR_TOKEN", "memory").strip().lower()
AGENT_KEEP_ALIVE_DEFAULT_SECONDS = int(os.getenv("AGENT_KEEP_ALIVE_DEFAULT_SECONDS", "60"))
AGENT_KEEP_ALIVE_UNIT_SECONDS = {
    k.lower().strip(): int(v)
    for k, v in _env_json_dict(
        "AGENT_KEEP_ALIVE_UNIT_SECONDS",
        {"s": "1", "m": "60", "h": "3600"},
    ).items()
    if str(k).strip() and str(v).strip()
}

AGENT_REFINE_DEFINED_ALIAS_PATTERN = os.getenv(
    "AGENT_REFINE_DEFINED_ALIAS_PATTERN",
    r"\bAS\s+([a-zA-Z0-9_]+)\b",
).strip()

AGENT_REFINE_QUOTED_ALIAS_PATTERN = os.getenv(
    "AGENT_REFINE_QUOTED_ALIAS_PATTERN",
    r"\bAS\s+['\"`]?([a-zA-Z0-9_]+)['\"`]?",
).strip()

AGENT_REFINE_SUBQUERY_ALIAS_PATTERN = os.getenv(
    "AGENT_REFINE_SUBQUERY_ALIAS_PATTERN",
    r"\)\s*(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\b",
).strip()

AGENT_REFINE_TABLE_ALIAS_PATTERN = os.getenv(
    "AGENT_REFINE_TABLE_ALIAS_PATTERN",
    r"\b(?:FROM|JOIN)\s+[a-zA-Z_][a-zA-Z0-9_]*\s+(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\b",
).strip()

AGENT_REFINE_SUBQUERY_ALIAS_RESERVED_TOKENS = set(
    _env_json_list(
        "AGENT_REFINE_SUBQUERY_ALIAS_RESERVED_TOKENS",
        ["where", "group", "order", "having", "limit", "on", "and", "or"],
    )
)

AGENT_REFINE_TABLE_ALIAS_RESERVED_TOKENS = set(
    _env_json_list(
        "AGENT_REFINE_TABLE_ALIAS_RESERVED_TOKENS",
        ["where", "group", "order", "having", "limit", "on", "and", "or", "select"],
    )
)

AGENT_REFINE_PRIMARY_FROM_PATTERN = os.getenv(
    "AGENT_REFINE_PRIMARY_FROM_PATTERN",
    r"FROM\s+`?([A-Za-z0-9_]+)`?",
).strip()

AGENT_REFINE_HAS_SUBQUERY_PATTERN = os.getenv(
    "AGENT_REFINE_HAS_SUBQUERY_PATTERN",
    r"\(\s*SELECT\b",
).strip()

AGENT_REFINE_MAIN_SELECT_FROM_PATTERN = os.getenv(
    "AGENT_REFINE_MAIN_SELECT_FROM_PATTERN",
    r"^SELECT\s+(.*?)\s+FROM",
).strip()

AGENT_REFINE_SINGLE_QUOTED_STRING_PATTERN = os.getenv(
    "AGENT_REFINE_SINGLE_QUOTED_STRING_PATTERN",
    r"'(?:''|[^'])*'",
).strip()

AGENT_REFINE_DOUBLE_QUOTED_STRING_PATTERN = os.getenv(
    "AGENT_REFINE_DOUBLE_QUOTED_STRING_PATTERN",
    r'"(?:""|[^"])*"',
).strip()

AGENT_REFINE_FUNCTION_CALL_PATTERN = os.getenv(
    "AGENT_REFINE_FUNCTION_CALL_PATTERN",
    r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(",
).strip()

AGENT_REFINE_SQL_TOKEN_PATTERN = os.getenv(
    "AGENT_REFINE_SQL_TOKEN_PATTERN",
    r"\b([a-z][a-z0-9_]*)\b",
).strip()

AGENT_REFINE_TABLE_PLACEHOLDER_TOKEN = os.getenv(
    "AGENT_REFINE_TABLE_PLACEHOLDER_TOKEN",
    "_TBL_",
).strip()

AGENT_REFINE_TABLE_PLACEHOLDER_DISCARD = os.getenv(
    "AGENT_REFINE_TABLE_PLACEHOLDER_DISCARD",
    "_tbl_",
).strip()

AGENT_REFINE_JOIN_KEYWORD = os.getenv(
    "AGENT_REFINE_JOIN_KEYWORD",
    "JOIN",
).strip().upper()

AGENT_REFINE_PRIMARY_FROM_SKIP_TOKENS = set(
    _env_json_list(
        "AGENT_REFINE_PRIMARY_FROM_SKIP_TOKENS",
        ["select", "with"],
    )
)

AGENT_REFINE_TABLE_REF_TOKEN_PATTERN = os.getenv(
    "AGENT_REFINE_TABLE_REF_TOKEN_PATTERN",
    r"\b([a-z0-9_]+)\.",
).strip()

AGENT_REFINE_GROUP_RISKY_COMPARE_PATTERN = os.getenv(
    "AGENT_REFINE_GROUP_RISKY_COMPARE_PATTERN",
    r"\b([a-z0-9_]+)\s*[<>=]",
).strip()

AGENT_REFINE_GROUP_RISKY_PAREN_PATTERN = os.getenv(
    "AGENT_REFINE_GROUP_RISKY_PAREN_PATTERN",
    r"\((?:.*?\s)?([a-z0-9_]+)(?:\s.*?)?\)",
).strip()

AGENT_REFINE_WINNER_ROW_LEAK_PATTERN = os.getenv(
    "AGENT_REFINE_WINNER_ROW_LEAK_PATTERN",
    r"SELECT\s+.*?\brn\b.*?FROM",
).strip()

RETRIEVAL_FK_REFERENCE_PATTERN = os.getenv(
    "RETRIEVAL_FK_REFERENCE_PATTERN",
    r"FK to (\w+)\.(\w+)",
).strip()

AGENT_REFINE_TUPLE_IN_PATTERN = os.getenv(
    "AGENT_REFINE_TUPLE_IN_PATTERN",
    r"WHERE\s*\([^)]+,[^)]+\)\s*IN\s*\(SELECT",
).strip()

AGENT_REFINE_PARTITION_BY_PATTERN = os.getenv(
    "AGENT_REFINE_PARTITION_BY_PATTERN",
    r"PARTITION\s+BY\s+([A-Za-z0-9_]+)",
).strip()

AGENT_REFINE_BROKEN_ALIAS_PATTERN = os.getenv(
    "AGENT_REFINE_BROKEN_ALIAS_PATTERN",
    r"\b[a-z0-9_]+\.\(",
).strip()

AGENT_REFINE_MONTH_NAME_GROUP_PATTERN = os.getenv(
    "AGENT_REFINE_MONTH_NAME_GROUP_PATTERN",
    r"GROUP\s+BY\s+.*?DATE_FORMAT",
).strip()

AGENT_REFINE_SAFE_MONTH_SORT_FUNC_PATTERN = os.getenv(
    "AGENT_REFINE_SAFE_MONTH_SORT_FUNC_PATTERN",
    r"ORDER\s+BY\s+(.*?\.)?(MONTH|YEAR|MIN|MAX|STR_TO_DATE)\(",
).strip()

AGENT_REFINE_SAFE_MONTH_SORT_DATE_PATTERN = os.getenv(
    "AGENT_REFINE_SAFE_MONTH_SORT_DATE_PATTERN",
    r"ORDER\s+BY\s+.*?\w*Date\b",
).strip()
AGENT_REFINE_GROUP_BY_TOKEN = os.getenv("AGENT_REFINE_GROUP_BY_TOKEN", "GROUP BY").strip().upper()
AGENT_REFINE_LEFT_JOIN_TOKEN = os.getenv("AGENT_REFINE_LEFT_JOIN_TOKEN", "LEFT JOIN").strip().upper()
AGENT_REFINE_WILDCARD_TOKEN = os.getenv("AGENT_REFINE_WILDCARD_TOKEN", "*").strip()
AGENT_REFINE_LIKE_TOKEN = os.getenv("AGENT_REFINE_LIKE_TOKEN", "LIKE").strip().upper()
AGENT_REFINE_REGEX_TOKENS = set(
    tok.strip().upper()
    for tok in _env_json_list("AGENT_REFINE_REGEX_TOKENS", ["REGEXP", "RLIKE"])
    if tok and tok.strip()
)
AGENT_REFINE_MONTH_TOKEN = os.getenv("AGENT_REFINE_MONTH_TOKEN", "MONTH").strip().upper()
AGENT_REFINE_ORDER_TOKEN = os.getenv("AGENT_REFINE_ORDER_TOKEN", "ORDER").strip().upper()
AGENT_REFINE_FUNC_LOOKAHEAD_LEN = int(os.getenv("AGENT_REFINE_FUNC_LOOKAHEAD_LEN", "5"))
AGENT_REFINE_ALIAS_WINDOW_CHARS = int(os.getenv("AGENT_REFINE_ALIAS_WINDOW_CHARS", "10"))
AGENT_REFINE_ALIAS_TOKEN_WITH_SPACE = (os.getenv("AGENT_REFINE_ALIAS_TOKEN_WITH_SPACE", "AS ") or "AS ").upper()
AGENT_REFINE_ALIAS_TOKEN = (os.getenv("AGENT_REFINE_ALIAS_TOKEN", "AS") or "AS").upper()

AGENT_REFINE_GROUP_CLAUSE_PATTERN = os.getenv(
    "AGENT_REFINE_GROUP_CLAUSE_PATTERN",
    r"GROUP\s+BY\s+(.*?)($|ORDER|LIMIT|HAVING)",
).strip()

AGENT_REFINE_WHERE_CLAUSE_PATTERN = os.getenv(
    "AGENT_REFINE_WHERE_CLAUSE_PATTERN",
    r"WHERE\s+(.*?)($|GROUP|ORDER|LIMIT)",
).strip()

AGENT_REFINE_SELECT_FROM_CLAUSE_PATTERN = os.getenv(
    "AGENT_REFINE_SELECT_FROM_CLAUSE_PATTERN",
    r"^SELECT\s+(.*?)\s+FROM",
).strip()

AGENT_REFINE_LOWER_WHERE_PATTERN = os.getenv(
    "AGENT_REFINE_LOWER_WHERE_PATTERN",
    r"\bwhere\s+(.*?)(?:\bgroup\b|\border\b|\blimit\b|\bhaving\b|;|$)",
).strip()

AGENT_REFINE_LOWER_HAVING_PATTERN = os.getenv(
    "AGENT_REFINE_LOWER_HAVING_PATTERN",
    r"\bhaving\s+(.*?)(?:\border\b|\blimit\b|;|$)",
).strip()
