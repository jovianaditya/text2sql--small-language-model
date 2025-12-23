from typing import TypedDict, List

class RAGState(TypedDict, total=False):
    jsonl_paths: List[str]
    question: str
    added: int
    updated: int
    skipped: int
    context: str
    answer: str
    draft_sql: str
    final_sql: str
    # >>> Jenis query hasil classifier ('lookup', 'aggregate_by', etc.)
    qtype: str
    # >>> Flag apakah pertanyaan ini butuh REGEXP/pattern handling
    requires_regex: bool
    # >>> Catatan dari LLM (Agent 1/2) tentang alasan join, dll.
    notes: str
    # >>> Daftar error whitelist (identifier tidak ada di ALLOWED IDENTIFIERS)
    validation_errors: List[str]
    # >>> Ringkasan apa saja yang dikoreksi Agent 2 (join/regex/whitelist)
    critic_feedback: str