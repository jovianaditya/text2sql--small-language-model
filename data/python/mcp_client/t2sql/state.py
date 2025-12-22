from typing import TypedDict, List

class RAGState(TypedDict):
    jsonl_paths: List[str]
    question: str
    added: int
    updated: int
    skipped: int
    context: str
    answer: str
    draft_sql: str
    final_sql: str

    # NEW: jenis query hasil classifier ('lookup', 'aggregate_by', etc.)
    qtype: str

    # NEW: flag apakah pertanyaan ini butuh REGEXP/pattern handling
    requires_regex: bool

    # NEW: catatan dari LLM (Agent 1/2) tentang alasan join, dll.
    notes: str

    # NEW: daftar error whitelist (identifier tidak ada di ALLOWED IDENTIFIERS)
    validation_errors: List[str]

    # NEW: ringkasan apa saja yang dikoreksi Agent 2 (join/regex/whitelist)
    critic_feedback: str