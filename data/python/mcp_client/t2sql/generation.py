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
import json
from typing import Dict, Any, Optional, Tuple
from langchain_ollama import ChatOllama
# ======================================================================
# ======================== LOCAL MODULE IMPORT =========================
# ======================================================================
from .utils import try_llm_json

def _strip_code_fences(s: str) -> str:
    """Remove markdown code fences from string."""
    s = (s or "").strip()
    if s.startswith("```"):
        # remove first fence line
        lines = s.splitlines()
        if lines:
            lines = lines[1:]
        # drop trailing fence if present
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return s

def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Robust JSON extractor:
    - handle ```json fences
    - handle extra text before/after
    - brace-matching to find first valid JSON object
    """
    txt = _strip_code_fences(text)

    # Fast path: whole string is JSON
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Brace-matching scan for first valid {...}
    n = len(txt)
    i = 0
    while i < n:
        if txt[i] != "{":
            i += 1
            continue
        depth = 0
        j = i
        while j < n:
            ch = txt[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = txt[i : j + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        break  # stop expanding this start; try next "{"
            j += 1
        i += 1

    return None

def _validate_sql_notes(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Validate that object has at least sql or notes field."""
    if not isinstance(obj, dict):
        return None
    sql = (obj.get("sql") or "").strip()
    notes = (obj.get("notes") or "").strip()

    # minimal sanity: must have at least sql or notes
    if not sql and not notes:
        return None

    # optional extra sanity: if sql exists, it should look like SQL
    if sql:
        low = sql.lower()
        if "select" not in low and "with" not in low:
            # allow rare cases, but usually reject junk
            return None

    return {"sql": sql, "notes": notes}
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
