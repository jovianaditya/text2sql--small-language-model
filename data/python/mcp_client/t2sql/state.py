from typing import TypedDict, List

class RAGState(TypedDict, total=False):
    jsonl_paths: List[str]
    question: str
    added: int
    updated: int
    skipped: int
    # retrieval output
    retrieved_context: str
    # generation outputs
    answer: str
    draft_sql: str
    final_sql: str
    # >>> Jenis query hasil classifier ('lookup', 'aggregate_by', etc.)
    qtype: str
    # >>> Flag apakah pertanyaan ini butuh REGEXP/pattern handling
    requires_regex: bool
    # >>> Catatan dari LLM (Agent 1/2) tentang alasan join, dll.
    agent_notes: str
    # >>> Daftar error whitelist (identifier tidak ada di ALLOWED IDENTIFIERS)
    validation_errors: List[str]
    # >>> Ringkasan apa saja yang dikoreksi Agent 2 (join/regex/whitelist)
    critic_feedback: str

def make_state(paths, question: str) -> RAGState:
    # Kalau kamu set state.py menjadi total=False, ini bisa diperkecil
    return {
        "jsonl_paths": paths,
        "question": question,
        "added": 0,
        "updated": 0,
        "skipped": 0,
        "retrieved_context": "",
        "answer": "",
        "draft_sql": "",
        "final_sql": "",
        "qtype": "",
        "requires_regex": False,
        "agent_notes": "",
        "validation_errors": [],
        "critic_feedback": "",
    }