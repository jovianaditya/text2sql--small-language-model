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
        lambda state: "continue" if state.get("router_decision", {}).get("is_valid_question", True) else "rejected",
        {
            "continue": "ingest",
            "rejected": END,
        }
    )
    # Normal flow continues
    g.add_edge("ingest", "retrieve")
    g.add_edge("retrieve", "router")
    g.add_edge("router", "draft_sql")
    g.add_edge("draft_sql", "refine_sql")
    g.add_edge("refine_sql", END)
    return g.compile()