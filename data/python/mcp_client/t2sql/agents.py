from .state import RAGState
from .llm_utils import build_llm
from .utils import _dbg_print_two
from .generation import try_llm_json_three_times, regenerate_with_extra_constraint
from .helpers import get_single_allowed_table, validate_identifiers, _get_requires_regex, _get_qtype, make_generation_prompt, sanitize_notes, normalize_sql_pipeline, has_join, _question_requires_word_boundary_regex
 
# ======================================================================
# ======================== SQL AGENT 1 (DRAFTER) =======================
# ======================================================================
def draft_sql_agent(state: RAGState) -> RAGState:
    """
    Agent 1:
    - Mengambil question + context
    - Menentukan qtype & requires_regex (sekali saja, via helper)
    - Memanggil LLM sekali untuk membuat draft SQL + notes
    - Menyimpan hasil ke state['draft_sql'] dan state['notes']
    """
    llm = build_llm()

    ctx = state.get("retrieved_context", "")
    question = state.get("question", "")

    # DRY: pakai helper supaya turunan fitur selalu konsisten
    _ = _get_qtype(state)          # set state["qtype"] kalau belum ada
    _ = _get_requires_regex(state) # set state["requires_regex"] kalau belum ada

    base_prompt = make_generation_prompt(ctx, question)

    draft_sql = ""
    notes = ""

    obj = try_llm_json_three_times(llm, base_prompt)
    if obj is not None:
        draft_sql = (obj.get("sql") or "").strip()
        notes = (obj.get("notes") or "").strip()
    else:
        draft_sql = "SELECT 1;"
        notes = "Heuristic trivial fallback. No joins used."

    if draft_sql:
        _dbg_print_two("[Agent1 SQL] Draft (before post-processing):", draft_sql)

    # simpan hasil Agent 1 ke state, tapi belum dijadikan final answer.
    state["draft_sql"] = draft_sql
    state["agent_notes"] = notes

    return state

# ======================================================================
# ======================== SQL AGENT 2 (REFINER) =======================
# ======================================================================
def _regex_requirement_ok(sql: str, requires_regex: bool, requires_boundary: bool) -> bool:
    if not (requires_regex or requires_boundary):
        return True

    low = (sql or "").lower()
    has_regexp = ("regexp" in low) or ("rlike" in low)
    if not has_regexp:
        return False

    if requires_boundary:
        return "(^|[^[:alnum:]_])" in (sql or "")

    # strict-structure: minimal pastikan ada anchor ^ dan $
    return ("^" in (sql or "")) and ("$" in (sql or ""))

def refine_sql_agent(state: RAGState) -> RAGState:
    """
    Agent 2:
    - Ambil draft_sql + notes dari Agent 1
    - Jalankan heuristic pipeline (normalize_sql_pipeline)
    - Kalau perlu, minta LLM regenerate dengan constraint tambahan:
        * single-table only (kalau ALLOWED cuma 1 tabel & qtype lookup)
        * REGEXP wajib (kalau requires_regex True)
        * whitelist identifer (ALLOWED IDENTIFIERS)
    - Simpan final_sql, notes yang sudah disanitasi, validation_errors, dan critic_feedback.
    """
    llm = build_llm()

    ctx = state.get("retrieved_context", "")
    question = state.get("question", "")
    draft_sql = (state.get("draft_sql") or "").strip()
    notes = state.get("agent_notes") or ""
    qtype = _get_qtype(state)
    requires_regex = _get_requires_regex(state)
    requires_boundary = _question_requires_word_boundary_regex(question)

    corrections: list[str] = []  # NEW: catat apa yang diperbaiki agent 2

    if not draft_sql:
        # fallback kalau Agent 1 gagal total
        draft_sql = "SELECT 1;"
        corrections.append("Draft SQL missing, fallback to SELECT 1;")

    # 1) Normalisasi awal (postprocess + alias repair + postprocess)
    final_sql = normalize_sql_pipeline(draft_sql, question, ctx)
    if final_sql:
        _dbg_print_two("[Agent2 SQL] After initial normalization:", final_sql)

    # 2) RULE: kalau ALLOWED cuma 1 tabel & qtype 'lookup' → paksa tanpa JOIN
    single_tbl = get_single_allowed_table(ctx)
    if single_tbl and qtype == "lookup" and has_join(final_sql or ""):
        extra = (
            f"The question is a simple lookup and ALLOWED IDENTIFIERS only lists a single table `{single_tbl}`.\n"
            "Rewrite the SQL so that it uses ONLY that single table and does NOT use any JOIN at all."
        )
        draft2, notes2 = regenerate_with_extra_constraint(llm, make_generation_prompt, ctx, question, extra)
        sql2 = normalize_sql_pipeline(draft2 or "", question, ctx) if draft2 else ""
        if sql2 and not has_join(sql2):
            corrections.append(
                f"Removed unnecessary JOINs because only single table `{single_tbl}` is allowed."
            )
            final_sql = sql2
            state["draft_sql"] = draft2 or draft_sql
            if notes2:
                notes = notes2

    # 3) RULE: enforce REGEXP untuk pertanyaan pattern-ketat
    # 3) RULE: enforce REGEXP hanya untuk:
    #    A) word-boundary intent
    #    B) strict structure/format
    low_sql = (final_sql or "").lower()

    if requires_boundary and final_sql:
        low_sql = (final_sql or "").lower()
        has_regexp = ("regexp" in low_sql) or ("rlike" in low_sql)
        has_boundary = "(^|[^[:alnum:]_])" in (final_sql or "")

        if (not has_regexp) or (not has_boundary):
            extra = (
                "The question requires word-boundary matching (a word that starts with X). "
                "Rewrite using MySQL REGEXP with POSIX boundary style: "
                "(^|[^[:alnum:]_])X[[:alnum:]_]* . "
                "Do NOT use \\b or \\w. Do NOT use LIKE for this case."
            )
            draft2, notes2 = regenerate_with_extra_constraint(
                llm, make_generation_prompt, ctx, question, extra
            )
            sql2 = normalize_sql_pipeline(draft2 or "", question, ctx) if draft2 else ""
            if sql2 and ("regexp" in sql2.lower() or "rlike" in sql2.lower()):
                corrections.append("Enforced REGEXP usage for word-boundary requirement.")
                final_sql = sql2
                state["draft_sql"] = draft2 or draft_sql
                if notes2:
                    notes = notes2

    elif requires_regex and final_sql and "regexp" not in low_sql:
        extra = (
            "The question requires a strict pattern (exact length and/or digit/character structure). "
            "Rewrite using MySQL REGEXP with ^ and $ anchors. "
            "Do NOT use LIKE-only or numeric/substr tricks for the main pattern."
        )
        draft2, notes2 = regenerate_with_extra_constraint(llm, make_generation_prompt, ctx, question, extra)
        sql2 = normalize_sql_pipeline(draft2 or "", question, ctx) if draft2 else ""
        # _dbg_print_two("[Agent2 REGEX ENFORCE] extra:", extra)
        # _dbg_print_two("[Agent2 REGEX ENFORCE] draft2:", (draft2 or "").strip())
        # _dbg_print_two("[Agent2 REGEX ENFORCE] sql2:", (sql2 or "").strip())
        if sql2 and "regexp" in sql2.lower():
            corrections.append("Enforced REGEXP usage for strict structure requirement.")
            final_sql = sql2
            state["draft_sql"] = draft2 or draft_sql
            if notes2:
                notes = notes2

    # 4) RULE: whitelist check (ALLOWED IDENTIFIERS)
    violations = validate_identifiers(final_sql or "", ctx)
    state["validation_errors"] = violations  # NEW: simpan list error pertama

    if violations:
        old_count = len(violations)

        extra = (
            "The previous SQL used identifiers not present in ALLOWED IDENTIFIERS:\n- "
            + "\n- ".join(violations)
            + "\nRewrite the SQL using ONLY identifiers listed in ALLOWED IDENTIFIERS. "
              "If a synonym appears, choose the canonical column."
        )
        if requires_boundary:
            extra += "\nAdditionally: preserve REGEXP word-boundary style using (^|[^[:alnum:]_])X[[:alnum:]_]*. Do NOT switch to LIKE."
        elif requires_regex:
            extra += "\nAdditionally: preserve REGEXP with ^ and $ anchors for strict pattern. Do NOT switch to LIKE."
        draft2, notes2 = regenerate_with_extra_constraint(llm, make_generation_prompt, ctx, question, extra)
        sql2 = normalize_sql_pipeline(draft2 or "", question, ctx) if draft2 else ""

        new_count = len(validate_identifiers(sql2 or "", ctx))

        # HANYA terima kalau pelanggaran berkurang atau jadi 0
        if (sql2 and (new_count == 0 or new_count < old_count) and _regex_requirement_ok(sql2, requires_regex, requires_boundary)):
            final_sql = sql2
            state["draft_sql"] = draft2 or draft_sql
            if notes2:
                notes = notes2
            state["validation_errors"] = validate_identifiers(final_sql or "", ctx)
        else:
            corrections.append("Rejected whitelist rewrite because it broke regex requirement.")


    # 5) sanitasi notes berdasarkan ada/tidaknya JOIN
    notes = sanitize_notes(final_sql or "", notes or "")

    # 6) simpan final result ke state
    if final_sql:
        answer_out = final_sql
    else:
        answer_out = draft_sql or "SELECT 1;"

    state["final_sql"] = final_sql or answer_out
    state["answer"] = answer_out
    state["agent_notes"] = notes

    # NEW: simpan ringkasan pekerjaan Agent 2 (untuk defense / logging)
    if corrections:
        state["critic_feedback"] = " | ".join(corrections)
    else:
        state["critic_feedback"] = "No critical corrections from Agent 2."

    _dbg_print_two("[Agent2 SQL] Final SQL after all refinements:", state["final_sql"])

    return state