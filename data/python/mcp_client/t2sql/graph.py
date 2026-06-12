# ======================================================================
# ============================== graph.py ==============================
# ======================================================================
# Purpose:
# - Defines the state graph for the T2SQL application using RAG architecture.
# - Integrates various nodes for ingestion, retrieval, and SQL drafting/refinement.
# - Establishes the flow and conditional logic between nodes.
# - Compiles the graph for execution.

# ======================================================================
# ====================== STANDARD LIBRARY IMPORT =======================
# ======================================================================
from langgraph.graph import StateGraph, END
# ======================================================================
# ======================== LOCAL MODULE IMPORT =========================
# ======================================================================
from .state import RAGState
from .ingestion import ingest_node
from .retrieval import retrieve_node
from .agents import draft_sql_agent, refine_sql_agent, router_agent, question_guard_node

def _scenario_config(state) -> dict:
    cfg = state.get("scenario_config") or {}
    return cfg if isinstance(cfg, dict) else {}

def _after_guard(state) -> str:
    if not state.get("router_decision", {}).get("is_valid_question", True):
        return "rejected"
    cfg = _scenario_config(state)
    if cfg.get("rag_enabled", True):
        return "ingest"
    return "router" if cfg.get("multi_agent_enabled", True) else "draft_sql"

def _after_ingest(state) -> str:
    cfg = _scenario_config(state)
    return "retrieve" if cfg.get("rag_enabled", True) else "draft_sql"

def _after_retrieve(state) -> str:
    cfg = _scenario_config(state)
    return "router" if cfg.get("multi_agent_enabled", True) else "draft_sql"

def _after_draft(state) -> str:
    cfg = _scenario_config(state)
    return "refine_sql" if cfg.get("multi_agent_enabled", True) else "done"

def build_app():
    g = StateGraph(RAGState)
    # Nodes
    g.add_node("guard", question_guard_node)  # Guard runs FIRST (before ingest)
    g.add_node("ingest", ingest_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("router", router_agent)
    g.add_node("draft_sql", draft_sql_agent)
    g.add_node("refine_sql", refine_sql_agent)
    # Entry point: Guard first!
    g.set_entry_point("guard")
    # Conditional: if guard passes → ingest, else → END
    g.add_conditional_edges(
        "guard",
        _after_guard,
        {
            "ingest": "ingest",
            "router": "router",
            "draft_sql": "draft_sql",
            "rejected": END,
        }
    )
    # Normal flow continues
    g.add_conditional_edges(
        "ingest",
        _after_ingest,
        {
            "retrieve": "retrieve",
            "draft_sql": "draft_sql",
        },
    )
    g.add_conditional_edges(
        "retrieve",
        _after_retrieve,
        {
            "router": "router",
            "draft_sql": "draft_sql",
        },
    )
    g.add_edge("router", "draft_sql")
    g.add_conditional_edges(
        "draft_sql",
        _after_draft,
        {
            "refine_sql": "refine_sql",
            "done": END,
        },
    )
    g.add_edge("refine_sql", END)
    return g.compile()
