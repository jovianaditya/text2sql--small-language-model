# ======================================================================
# ============================= generation.py ==========================
# ======================================================================
# Purpose:
# - Functions for robust JSON extraction and constrained regeneration.
# - Used by Agent 3 for error correction.
# - Focus on extracting SQL and notes from LLM responses.
# - such as code fences, extra text, and malformed JSON.
# - Enables reliable regeneration of SQL queries with additional constraints.

# ======================================================================
# ====================== STANDARD LIBRARY IMPORT =======================
# ======================================================================
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
from langchain_ollama import ChatOllama
# ======================================================================
# ======================== LOCAL MODULE IMPORT =========================
# ======================================================================
from .utils import (
    try_llm_json,
    _strip_code_fences as _utils_strip_code_fences,
    _extract_json_object as _utils_extract_json_object,
)

def _strip_code_fences(s: str) -> str:
    """Remove markdown code fences from string (delegates to utils helper)."""
    return _utils_strip_code_fences(s)

def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Robust JSON extractor (delegates to utils helper)."""
    return _utils_extract_json_object(text)

# ======================================================================
# ===================== CONSTRAINTED REGENERATION (REPAIR) ==============
# ======================================================================
def regenerate_with_extra_constraint(
    llm: ChatOllama,
    make_prompt_fn,
    ctx: str,
    question: str,
    extra_msg: str,
    prompt_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """
    Regenerate SQL with extra constraint message.

    Used by Agent 3 for error correction and regeneration.

    Args:
        llm: LLM instance
        make_prompt_fn: Function to build prompt
        ctx: Context string
        question: User question
        extra_msg: Additional constraint message

    Returns:
        Tuple of (sql, notes)
    """
    kwargs = dict(prompt_kwargs or {})
    strict_prompt = make_prompt_fn(ctx + "\n\n" + extra_msg, question, **kwargs)
    obj = try_llm_json(llm, strict_prompt, max_attempts=2)
    if obj is None:
        return "", ""
    return (obj.get("sql") or "").strip(), (obj.get("notes") or "").strip()
