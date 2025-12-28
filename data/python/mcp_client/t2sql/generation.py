import re
import json
from typing import Dict, Any, Optional
from langchain_ollama import ChatOllama

# ======================================================================
# ===================== LLM RESPONSE PARSING (JSON) =====================
# =====================================================================
def try_llm_json_three_times(llm: ChatOllama, base_prompt: str) -> Optional[Dict[str, Any]]:
    prompts = [
        base_prompt,
        base_prompt + "\n\nReturn ONLY the JSON object. Do not add explanation.",
        base_prompt
        + "\n\nReturn ONLY JSON with keys \"sql\" and \"notes\". No extra text. "
          "Example: {\"sql\":\"SELECT 1;\",\"notes\":\"No joins used.\"}",
    ]
    for p in prompts:
        resp = llm.invoke(p)
        content = getattr(resp, "content", None) or str(resp)
        try:
            obj = json.loads(content)
            if isinstance(obj, dict) and ("sql" in obj or "notes" in obj):
                return obj
        except Exception:
            m = re.search(r"\{.*\}", content, flags=re.S)
            if m:
                try:
                    obj = json.loads(m.group(0))
                    if isinstance(obj, dict) and ("sql" in obj or "notes" in obj):
                        return obj
                except Exception:
                    pass
    return None

# ======================================================================
# ===================== CONSTRAINTED REGENERATION (REPAIR) ==============
# ======================================================================
def regenerate_with_extra_constraint(
    llm: ChatOllama,
    make_prompt_fn,
    ctx: str,
    question: str,
    extra_msg: str,
) -> tuple[str, str]:
    strict_prompt = make_prompt_fn(ctx + "\n\n" + extra_msg, question)
    obj = try_llm_json_three_times(llm, strict_prompt)
    if obj is None:
        return "", ""
    draft2 = (obj.get("sql") or "").strip()
    notes2 = (obj.get("notes") or "").strip()
    return draft2, notes2