# ======================================================================
# ============================= llm_utils.py ===========================
# ======================================================================
# - Purpose:
# - LLM Builders untuk T2SQL Agents (JSON Mode & Repair Mode).

# ======================================================================
# ====================== STANDARD LIBRARY IMPORT =======================
# ======================================================================
from __future__ import annotations
from langchain_ollama import ChatOllama
# ======================================================================
# ======================== LOCAL MODULE IMPORT =========================
# ======================================================================
from .config import (
    LARGE_MODEL,
    OLLAMA_URL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_SEED,
    LLM_JSON_NUM_PREDICT,
    LLM_REPAIR_NUM_PREDICT,
    LLM_JSON_TIMEOUT,
    LLM_REPAIR_TIMEOUT,
)

def build_llm_json(model: str | None = None) -> ChatOllama:
    """
    Builder untuk Agent yang WAJIB output JSON (Agent 1 Router & Agent 2 Drafter).
    
    Fitur Utama:
    1. format="json": Memaksa model output JSON murni (tanpa teks pembuka/penutup).
    2. keep_alive: Mencegah model unload dari VRAM saat ganti agent.
    3. timeout: Extended timeout untuk model switching (3b→7b).
    """
    return ChatOllama(
        # Gunakan model yang di passing, atau fallback ke LARGE_MODEL
        model=(model or LARGE_MODEL),
        base_url=OLLAMA_URL,
        temperature=0.0,
        seed=OLLAMA_SEED,
        format="json",
        keep_alive=OLLAMA_KEEP_ALIVE,
        num_ctx=4096,
        num_predict=LLM_JSON_NUM_PREDICT,
        stop=None,
        timeout=LLM_JSON_TIMEOUT,
    )

def build_llm_repair(model: str | None = None) -> ChatOllama:
    """
    Builder untuk Agent 3 Refiner (Text Mode).
    Memperbaikki SQL Query hasil Output dari Agent 2 (add heuristics, dll.)
    """
    return ChatOllama(
        # Gunakan model yang di passing, atau fallback ke LARGE_MODEL
        model=(model or LARGE_MODEL),
        base_url=OLLAMA_URL,
        temperature=0.0,
        seed=OLLAMA_SEED,
        keep_alive=OLLAMA_KEEP_ALIVE,
        num_ctx=4096,
        num_predict=LLM_REPAIR_NUM_PREDICT,
        stop=None,
        timeout=LLM_REPAIR_TIMEOUT,
    )
