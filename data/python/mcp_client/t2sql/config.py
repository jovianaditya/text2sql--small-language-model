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
