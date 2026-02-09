# ======================================================================
# ============================== state.py ==============================
# ======================================================================
# Purpose:
# - Define the RAGState TypedDict for tracking state in the RAG process.
# - Provide a function to initialize the RAGState with default values.

# ======================================================================
# ====================== STANDARD LIBRARY IMPORT =======================
# ======================================================================
from typing import TypedDict, List, Dict, Any

class RAGState(TypedDict, total=False):
    jsonl_paths: List[str]
    question: str
    added: int
    updated: int
    skipped: int
    # retrieval output (context untuk LLM)
    context: str
    # ROUTING
    router_decision: Dict[str, Any]  # Output JSON Router
    router_tables: List[str]
    primary_table: str  # Primary table determined by router
    primary_table_reason: str  # Reason for primary table selection
    target_model: str  # Initial model chosen by router
    final_model: str  # Final model used (may differ if regeneration occurred)
    # generation outputs
    answer: str
    draft_sql: str
    ranked_cols: List[tuple[str, str]]
    final_sql: str
    # Jenis query hasil classifier ('lookup', 'aggregate', 'rank', dst.)
    qtype: str
    # Flag apakah pertanyaan ini butuh REGEXP/pattern handling
    requires_regex: bool
    # Catatan dari LLM (Agent 1/2) tentang alasan join, dll.
    agent_notes: str
    # Daftar error whitelist (identifier tidak ada di ALLOWED IDENTIFIERS)
    validation_errors: List[str]
    # Ringkasan apa saja yang dikoreksi Agent 2 (join/regex/whitelist)
    critic_feedback: str
    # Benchmark policy / audit trail
    group_by_policy: str
    group_by_added_exprs: List[str]
    # Benchmark timing
    processing_start_time: float
    processing_time: float

def make_state(paths, question: str) -> RAGState:
    return {
        "jsonl_paths": paths,
        "question": question,
        "added": 0,
        "updated": 0,
        "skipped": 0,
        "context": "",
        "router_decision": {},
        "router_tables": [],
        "primary_table": "",
        "primary_table_reason": "",
        "target_model": "",
        "final_model": "",  # Track final model after any regeneration
        "answer": "",
        "draft_sql": "",
        "ranked_cols": [],
        "final_sql": "",
        "qtype": "",
        "requires_regex": False,
        "agent_notes": "",
        "validation_errors": [],
        "critic_feedback": "",
        # Benchmark policy / audit trail
        "group_by_policy": "explicit_select_cols",
        "group_by_added_exprs": [],
        # Benchmark timing
        "processing_start_time": 0.0,
        "processing_time": 0.0,
    }