import os
# ======================================================================
# ============================ CONFIG (ENV) ============================
# ======================================================================

# ============================ APP/GENERAL =============================
JSON_INPUT       = os.getenv("EXPORT_JSONL_OUTPUT", "./datahub_metadata/*.jsonl").strip()
CHROMA_DIR       = os.getenv("CHROMA_DIR", "./chroma_db").strip()
COLLECTION_NAME  = os.getenv("COLLECTION_NAME", "classicmodels").strip()
OLLAMA_BASE_URL  = os.getenv("OLLAMA_URL", "http://localhost:11434").strip()
CHAT_MODEL       = os.getenv("CHAT_MODEL", "HridaAI/hrida-t2sql:q8_0").strip()

# ============================ RETRIEVAL ===============================
TOP_K            = int(os.getenv("TOP_K", "5"))
MICRO_PROBE = ["id", "date", "status"]
SPARSE_BONUS     = float(os.getenv("SPARSE_BONUS", "0.75"))
NAME_QUERY_BONUS = float(os.getenv("NAME_QUERY_BONUS", "0.3"))
ID_QUERY_BONUS   = float(os.getenv("ID_QUERY_BONUS", "0.4"))
# >>> Batasi berapa banyak dokumen yang masuk ke context LLM
MAX_CTX_DOCS = int(os.getenv("MAX_CTX_DOCS", "12"))
SPARSE_MAX_HITS  = int(os.getenv("SPARSE_MAX_HITS", "20"))
# >>> Bisa dimatiin kalau mau lebih silent
SPARSE_DEBUG   = bool(int(os.getenv("SPARSE_DEBUG", "0")))
RETRIEVE_DEBUG = bool(int(os.getenv("RETRIEVE_DEBUG", "1")))
COLUMN_PRIORITY_CONFIG = [
    {
        "id": "customers.primary_name",
        "tables": ["customers"],
        "role_any": ["name"],
        "alias_any": [
            "customer name",
            "nama customer",
            "nama pelanggan",
            "company name",
            "nama perusahaan",
            "account name",
        ],
        "column_contains": ["customername"],
        "query_terms_any": ["name", "nama"],
        "query_terms_also_any": ["customer", "customers", "pelanggan"],
        "boost": 0.4,
    },
]

# =============================== MCP ==================================
MCP_BASE    = os.getenv("MCP_BASE", "http://mcp_server:8000/mcp")
MCP_TIMEOUT = int(os.getenv("MCP_CLIENT_TIMEOUT", "60"))

# ============================= DEBUG ==================================
SQL_DEBUG = bool(int(os.getenv("SQL_DEBUG", "1")))
COMPARE_DEBUG = bool(int(os.getenv("COMPARE_DEBUG", "0")))