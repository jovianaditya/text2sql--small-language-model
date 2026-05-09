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
    scenario: str
    scenario_config: Dict[str, Any]
    added: int
    updated: int
    skipped: int
    # retrieval output (context untuk LLM)
    context: str
    retrieval_context_mode: str
    retrieval_retry_used: bool
    retrieval_retry_reason: str
    retrieval_expanded_table_count: int
    retrieval_snapshot: Dict[str, Any]
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
    draft_source: str  # llm (ir_planner deprecated)
    draft_inference_time: float
    ranked_cols: List[tuple[str, str]]
    final_sql: str
    # Agnostic pipeline artifacts
    intent_ir: Dict[str, Any]
    ambiguity_signals: List[str]
    cannot_answer: bool
    cannot_answer_reason: str
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
    ir_hint_group_filters_present: bool  # True if IR soft hint included Group Filters
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
    final_gate_reason_tags: List[str]
    # Ringkasan apa saja yang dikoreksi Agent 2 (join/regex/whitelist)
    critic_feedback: str
    # Optional post-failure Human-in-the-Loop review telemetry
    execution_error: Dict[str, Any]
    hitl_used: bool
    hitl_action: str
    hitl_feedback: str
    # Benchmark policy / audit trail
    group_by_policy: str
    group_by_added_exprs: List[str]
    # Benchmark timing
    processing_start_time: float
    processing_time: float

def make_state(
    paths,
    question: str,
    *,
    scenario: str = "P1",
    scenario_config: Dict[str, Any] | None = None,
) -> RAGState:
    return {
        "jsonl_paths": paths,
        "question": question,
        "scenario": str(scenario or "P1").strip().upper(),
        "scenario_config": dict(scenario_config or {}),
        "added": 0,
        "updated": 0,
        "skipped": 0,
        "context": "",
        "retrieval_context_mode": "primary",
        "retrieval_retry_used": False,
        "retrieval_retry_reason": "",
        "retrieval_expanded_table_count": 0,
        "retrieval_snapshot": {},
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
        "ambiguity_signals": [],
        "cannot_answer": False,
        "cannot_answer_reason": "",
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
        "ir_hint_group_filters_present": False,
        "qtype": "",
        "requires_regex": False,
        "sql_confidence": 0.0,
        "confidence_signals": {},
        "confidence_warnings": [],
        "confidence_low": False,
        "confidence_retry_used": False,
        "agent_notes": "",
        "validation_errors": [],
        "final_gate_reason_tags": [],
        "critic_feedback": "",
        "execution_error": {},
        "hitl_used": False,
        "hitl_action": "",
        "hitl_feedback": "",
        # Benchmark policy / audit trail
        "group_by_policy": "explicit_select_cols",
        "group_by_added_exprs": [],
        # Benchmark timing
        "processing_start_time": 0.0,
        "processing_time": 0.0,
    }
