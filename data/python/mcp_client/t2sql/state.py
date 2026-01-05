from typing import TypedDict, List

class RAGState(TypedDict, total=False):
    jsonl_paths: List[str]
    question: str
    added: int
    updated: int
    skipped: int
    # retrieval output (context untuk LLM)
    context: str
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

def make_state(paths, question: str) -> RAGState:
    return {
        "jsonl_paths": paths,
        "question": question,
        "added": 0,
        "updated": 0,
        "skipped": 0,
        "context": "",
        "answer": "",
        "draft_sql": "",
        "ranked_cols": [],
        "final_sql": "",
        "qtype": "",
        "requires_regex": False,
        "agent_notes": "",
        "validation_errors": [],
        "critic_feedback": "",
    }