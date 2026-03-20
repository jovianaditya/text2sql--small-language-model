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
    router_model_candidate: str  # Router-computed LLM fallback candidate
    target_model: str  # Actual LLM model used if LLM lane is selected
    final_model: str  # Final model used (may differ if regeneration occurred)
    # generation outputs
    answer: str
    draft_sql: str
    draft_source: str  # llm|ir_planner
    draft_inference_time: float
    ranked_cols: List[tuple[str, str]]
    final_sql: str
    # Agnostic pipeline artifacts
    intent_ir: Dict[str, Any]
    join_plan: Dict[str, Any]
    # Route telemetry
    ir_used: bool
    ir_confidence: float
    ir_plannable: bool
    ir_gate_reasons: List[str]
    ir_table_count: int
    ir_context_table_count: int
    ir_semantic_repair_used: bool
    ir_semantic_fallback_used: bool
    ir_semantic_fallback_model: str
    ir_hint_passed: bool  # True if intent IR was passed as soft hint to LLM
    # Jenis query hasil classifier ('lookup', 'aggregate', 'rank', dst.)
    qtype: str
    # Flag apakah pertanyaan ini butuh REGEXP/pattern handling
    requires_regex: bool
    # === CONFIDENCE SCORING ===
    sql_confidence: float  # Overall confidence score (0.0 - 1.0)
    confidence_signals: Dict[str, float]  # Individual signal scores
    confidence_warnings: List[str]  # Warnings from confidence analysis
    confidence_low: bool  # True if confidence < threshold (0.70)
    confidence_retry_used: bool  # True if confidence retry was attempted
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
        "router_model_candidate": "",
        "target_model": "",
        "final_model": "",  # Track final model after any regeneration
        "answer": "",
        "draft_sql": "",
        "draft_source": "",
        "draft_inference_time": 0.0,
        "ranked_cols": [],
        "final_sql": "",
        "intent_ir": {},
        "join_plan": {},
        "ir_used": False,
        "ir_confidence": 0.0,
        "ir_plannable": False,
        "ir_gate_reasons": [],
        "ir_table_count": 0,
        "ir_context_table_count": 0,
        "ir_semantic_repair_used": False,
        "ir_semantic_fallback_used": False,
        "ir_semantic_fallback_model": "",
        "ir_hint_passed": False,
        "qtype": "",
        "requires_regex": False,
        "sql_confidence": 0.0,
        "confidence_signals": {},
        "confidence_warnings": [],
        "confidence_low": False,
        "confidence_retry_used": False,
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
