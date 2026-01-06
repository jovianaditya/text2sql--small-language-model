from .state import RAGState
from .llm_utils import build_llm_json, build_llm_repair
from .utils import _dbg_print_two
from .generation import try_llm_json, regenerate_with_extra_constraint
from .helpers import (
    get_single_allowed_table,
    validate_identifiers,
    _get_requires_regex,
    _get_qtype,
    make_generation_prompt,
    sanitize_notes,
    normalize_sql_pipeline,
    has_join,
    _question_requires_word_boundary_regex,
)


def draft_sql_agent(state: RAGState) -> RAGState:
    llm = build_llm_json()

    ctx = state.get("context", "")
    question = state.get("question", "")

    _ = _get_qtype(state)
    _ = _get_requires_regex(state)

    base_prompt = make_generation_prompt(ctx, question)

    obj = try_llm_json(llm, base_prompt, max_attempts=2)

    if obj is not None:
        draft_sql = (obj.get("sql") or "").strip()
        notes = (obj.get("notes") or "").strip()
    else:
        draft_sql = "SELECT 1;"
        notes = "Heuristic trivial fallback. No joins used."

    if draft_sql:
        _dbg_print_two("[Agent1 SQL] Draft (before post-processing):", draft_sql)

    state["draft_sql"] = draft_sql
    state["agent_notes"] = notes
    return state


def _regex_requirement_ok(sql: str, requires_regex: bool, requires_boundary: bool) -> bool:
    if not (requires_regex or requires_boundary):
        return True

    low = (sql or "").lower()
    has_regexp = ("regexp" in low) or ("rlike" in low)
    if not has_regexp:
        return False

    if requires_boundary:
        return "(^|[^[:alnum:]_])" in (sql or "")

    return ("^" in (sql or "")) and ("$" in (sql or ""))


def refine_sql_agent(state: RAGState) -> RAGState:
    llm = build_llm_repair()
    ctx = state.get("context", "")
    question = state.get("question", "")
    draft_sql = (state.get("draft_sql") or "").strip()
    notes = state.get("agent_notes") or ""
    ranked_cols = state.get("ranked_cols") or []
    qtype = _get_qtype(state)
    requires_regex = _get_requires_regex(state)
    requires_boundary = _question_requires_word_boundary_regex(question)
    _dbg_print_two("[Agent2] Flags:", f"qtype={qtype} requires_regex={requires_regex} requires_boundary={requires_boundary}")


    corrections: list[str] = []

    if not draft_sql:
        draft_sql = "SELECT 1;"
        corrections.append("Draft SQL missing, fallback to SELECT 1;")
    _dbg_print_two("[Agent2] Draft SQL (raw):", draft_sql)
    final_sql = normalize_sql_pipeline(draft_sql, question, ctx, ranked_cols=ranked_cols)

    if final_sql:
        _dbg_print_two("[Agent2 SQL] After initial normalization:", final_sql)

    single_tbl = get_single_allowed_table(ctx)

    if single_tbl and qtype == "lookup" and has_join(final_sql or ""):
        extra = (
            f"The question is a simple lookup and ALLOWED IDENTIFIERS only lists a single table `{single_tbl}`.\n"
            "Rewrite the SQL so that it uses ONLY that single table and does NOT use any JOIN at all."
        )
        _dbg_print_two("[Agent2] Repair LLM called because:", extra)
        draft2, notes2 = regenerate_with_extra_constraint(llm, make_generation_prompt, ctx, question, extra)
        sql2 = normalize_sql_pipeline(draft2 or "", question, ctx, ranked_cols=ranked_cols) if draft2 else ""

        if sql2 and not has_join(sql2):
            corrections.append(f"Removed unnecessary JOINs because only single table `{single_tbl}` is allowed.")
            final_sql = sql2
            state["draft_sql"] = draft2 or draft_sql
            if notes2:
                notes = notes2

    low_sql = (final_sql or "").lower()

    if requires_boundary and final_sql:
        has_regexp = ("regexp" in low_sql) or ("rlike" in low_sql)
        has_boundary = "(^|[^[:alnum:]_])" in (final_sql or "")

        if (not has_regexp) or (not has_boundary):
            extra = (
                "The question requires word-boundary matching (a word that starts with X). "
                "Rewrite using MySQL REGEXP with POSIX boundary style: "
                "(^|[^[:alnum:]_])X[[:alnum:]_]* . "
                "Do NOT use \\b or \\w. Do NOT use LIKE for this case."
            )
            _dbg_print_two("[Agent2] Repair LLM called because:", extra)
            draft2, notes2 = regenerate_with_extra_constraint(llm, make_generation_prompt, ctx, question, extra)
            sql2 = normalize_sql_pipeline(draft2 or "", question, ctx, ranked_cols=ranked_cols) if draft2 else ""

            if sql2 and (("regexp" in sql2.lower()) or ("rlike" in sql2.lower())):
                corrections.append("Enforced REGEXP usage for word-boundary requirement.")
                final_sql = sql2
                state["draft_sql"] = draft2 or draft_sql
                if notes2:
                    notes = notes2

    elif requires_regex and final_sql and ("regexp" not in low_sql) and ("rlike" not in low_sql):
        extra = (
            "The question requires a strict pattern (exact length and/or digit/character structure). "
            "Rewrite using MySQL REGEXP with ^ and $ anchors. "
            "Do NOT use LIKE-only or numeric/substr tricks for the main pattern."
        )
        _dbg_print_two("[Agent2] Repair LLM called because:", extra)
        draft2, notes2 = regenerate_with_extra_constraint(llm, make_generation_prompt, ctx, question, extra)
        sql2 = normalize_sql_pipeline(draft2 or "", question, ctx, ranked_cols=ranked_cols) if draft2 else ""

        if sql2 and (("regexp" in sql2.lower()) or ("rlike" in sql2.lower())):
            corrections.append("Enforced REGEXP usage for strict structure requirement.")
            final_sql = sql2
            state["draft_sql"] = draft2 or draft_sql
            if notes2:
                notes = notes2

    violations = validate_identifiers(final_sql or "", ctx)
    _dbg_print_two("[Agent2] Whitelist violations:", "\n".join(violations) if violations else "None")
    state["validation_errors"] = violations

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
        _dbg_print_two("[Agent2] Repair LLM called because:", extra)
        draft2, notes2 = regenerate_with_extra_constraint(llm, make_generation_prompt, ctx, question, extra)
        sql2 = normalize_sql_pipeline(draft2 or "", question, ctx, ranked_cols=ranked_cols) if draft2 else ""

        new_count = len(validate_identifiers(sql2 or "", ctx))

        if (
            sql2
            and (new_count == 0 or new_count < old_count)
            and _regex_requirement_ok(sql2, requires_regex, requires_boundary)
        ):
            final_sql = sql2
            state["draft_sql"] = draft2 or draft_sql
            if notes2:
                notes = notes2
            state["validation_errors"] = validate_identifiers(final_sql or "", ctx)
        else:
            corrections.append("Rejected whitelist rewrite because it broke regex requirement.")

    notes = sanitize_notes(final_sql or "", notes or "")

    answer_out = final_sql if final_sql else (draft_sql or "SELECT 1;")

    state["final_sql"] = final_sql or answer_out
    state["answer"] = answer_out
    state["agent_notes"] = notes

    if corrections:
        state["critic_feedback"] = " | ".join(corrections)
    else:
        state["critic_feedback"] = "No critical corrections from Agent 2."

    _dbg_print_two("[Agent2 SQL] Final SQL after all refinements:", state["final_sql"])
    return state