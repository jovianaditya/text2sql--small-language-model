from .state import RAGState
from .llm_utils import build_llm, try_llm_json_three_times, regenerate_with_extra_constraint
from .utils import _get_qtype, _dbg_print_two
from .generation import make_generation_prompt, sanitize_notes, normalize_sql_pipeline, has_join
from .helpers import get_single_allowed_table, validate_identifiers, _get_requires_regex

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

    ctx = state.get("context", "")
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
    state["notes"] = notes

    return state

# ======================================================================
# ======================== SQL AGENT 2 (REFINER) =======================
# ======================================================================
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

    ctx = state.get("context", "")
    question = state.get("question", "")
    draft_sql = (state.get("draft_sql") or "").strip()
    notes = state.get("notes") or ""
    qtype = _get_qtype(state)
    requires_regex = _get_requires_regex(state)

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
    if requires_regex and final_sql and "regexp" not in (final_sql or "").lower():
        extra = (
            "The question describes a strict string pattern (exact length and/or digit/character format). "
            "Rewrite the SQL so that this pattern is implemented using MySQL REGEXP with ^ and $ anchors. "
            "Do NOT use LIKE-only or numeric/substr tricks (e.g. RIGHT(...), CAST(...), BETWEEN ...) "
            "for the main pattern condition."
        )
        draft2, notes2 = regenerate_with_extra_constraint(llm, make_generation_prompt, ctx, question, extra)
        sql2 = normalize_sql_pipeline(draft2 or "", question, ctx) if draft2 else ""
        if sql2 and "regexp" in sql2.lower():
            corrections.append("Enforced REGEXP usage for strict pattern requirement.")
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
        draft2, notes2 = regenerate_with_extra_constraint(llm, make_generation_prompt, ctx, question, extra)
        sql2 = normalize_sql_pipeline(draft2 or "", question, ctx) if draft2 else ""

        new_count = len(validate_identifiers(sql2 or "", ctx))

        # HANYA terima kalau pelanggaran berkurang atau jadi 0
        if sql2 and (new_count == 0 or new_count < old_count):
            corrections.append(
                f"Reduced whitelist violations from {old_count} to {new_count}."
            )
            final_sql = sql2
            state["draft_sql"] = draft2 or draft_sql
            if notes2:
                notes = notes2
            # update violations ke hasil terakhir
            state["validation_errors"] = validate_identifiers(final_sql or "", ctx)


    # 5) sanitasi notes berdasarkan ada/tidaknya JOIN
    notes = sanitize_notes(final_sql or "", notes or "")

    # 6) simpan final result ke state
    if final_sql:
        answer_out = final_sql
    else:
        answer_out = draft_sql or "SELECT 1;"

    state["final_sql"] = final_sql or answer_out
    state["answer"] = answer_out
    state["notes"] = notes

    # NEW: simpan ringkasan pekerjaan Agent 2 (untuk defense / logging)
    if corrections:
        state["critic_feedback"] = " | ".join(corrections)
    else:
        state["critic_feedback"] = "No critical corrections from Agent 2."

    _dbg_print_two("[Agent2 SQL] Final SQL after all refinements:", state["final_sql"])

    return state