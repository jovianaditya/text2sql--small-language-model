# ======================================================================
# ============================ agents.py ===============================
# ======================================================================
# Purpose:
# - Agent 0 --> Guardrail for question validation (hybrid regex + LLM)
# - Agent 1 --> Router agent to determine query complexity and target model
# - Agent 2 --> Drafting agent to generate initial SQL
# - Agent 3 --> Refinement agent to improve SQL based on feedback

# ======================================================================
# ====================== STANDARD LIBRARY IMPORT =======================
# ======================================================================
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple
# ======================================================================
# ======================== LOCAL MODULE IMPORT =========================
# ======================================================================
from .config import (
    ROUTING_MODEL,
    SMALL_MODEL,
    LARGE_MODEL,
    OLLAMA_URL,
    OLLAMA_KEEP_ALIVE,
    RETRIEVE_DEBUG,
    RETRIEVAL_EXPAND_ON_FAIL_ENABLED,
    RETRIEVAL_EXPAND_TRIGGER_REASONS,
    OOM_WAIT_TIME,
    OOM_MAX_RETRIES,
    AGENT_IR_ENABLED,
    AGENT_IR_MIN_CONFIDENCE,
    AGENT_IR_DEBUG,
    AGENT_IR_ALLOW_MULTI_TABLE,
    AGENT_IR_MULTI_TABLE_MIN_CONFIDENCE,
    AGENT_IR_MAX_TABLES,
    AGENT_IR_ALLOW_SIMPLE_HAVING,
    AGENT_IR_SIMPLE_HAVING_MAX_FILTERS,
    AGENT_B4_CANNOT_ANSWER_ENABLED,
    AGENT_B4_AMBIGUITY_ABSTAIN_ENABLED,
    AGENT_B4_AMBIGUITY_MIN_SIGNALS,
    AGENT_B4_EVIDENCE_ABSTAIN_ENABLED,
    AGENT_B4_EVIDENCE_MIN_ALLOWED_TABLES,
    AGENT_B4_EVIDENCE_MIN_ALLOWED_COLUMNS,
    AGENT_B4_EVIDENCE_MIN_RERANKED,
    AGENT_B4_EVIDENCE_TRIGGER_REASONS,
    AGENT_PROMPT_ABSTAIN_ENABLED,
    AGENT_PROMPT_ABSTAIN_NOTES_PREFIX,
    AGENT_AST_GUARDS_ENABLED,
    AGENT_AST_GUARDS_DEBUG,
    AGENT_REGEN_PRECHECK_ENABLED,
    AGENT_CONFIDENCE_THRESHOLD,
    AGENT_CONFIDENCE_RETRY_MIN_IMPROVEMENT,
    AGENT_INSTANT_REJECT_PATTERNS,
    AGENT_DEFINITE_SQL_PATTERNS,
    AGENT_GIBBERISH_KEYBOARD_PATTERNS,
    AGENT_GUARD_EXAMPLE_MAX_ITEMS,
    AGENT_GUARD_EXAMPLE_YEAR,
    AGENT_GUARD_EXAMPLE_VALUE_THRESHOLD,
    AGENT_GUARD_BANNER_CHAR,
    AGENT_GUARD_BANNER_WIDTH,
    AGENT_GUARD_LLM_TEMPERATURE,
    AGENT_GUARD_LLM_NUM_CTX,
    AGENT_GUARD_LLM_NUM_PREDICT,
    AGENT_GUARD_SCHEMA_EXAMPLE_MAX_ITEMS,
    AGENT_GIBBERISH_MIN_TEXT_LENGTH,
    AGENT_GIBBERISH_SENTENCE_WORD_BYPASS,
    AGENT_GIBBERISH_MIN_ALPHA_LENGTH,
    AGENT_GIBBERISH_VOWEL_RATIO_MIN,
    AGENT_GIBBERISH_VOWEL_CHECK_MIN_LEN,
    AGENT_GIBBERISH_CONSONANT_RUN_MIN,
    AGENT_GIBBERISH_CONSONANT_CHECK_MIN_LEN,
    AGENT_GIBBERISH_CONSONANT_CLASS,
    AGENT_GIBBERISH_NO_SPACE_LEN,
    AGENT_GIBBERISH_REPEAT_MIN_LEN,
    AGENT_GIBBERISH_REPEAT_PATTERN_LENGTHS,
    AGENT_GIBBERISH_VOWELS,
    AGENT_GIBBERISH_ALPHA_ONLY_PATTERN,
    AGENT_GIBBERISH_HAS_UPPER_PATTERN,
    AGENT_GIBBERISH_HAS_SEPARATOR_PATTERN,
    AGENT_TABLE_VARIANT_MIN_TAIL_LEN,
    AGENT_COLUMN_ORDERED_TOKEN,
    AGENT_COLUMN_ORDERED_BY_PATTERN,
    AGENT_MISSING_COL_SUFFIX_SYNONYMS,
    AGENT_REGEX_TRIGGERS,
    AGENT_ROUTER_MATH_KEYWORDS,
    AGENT_ROUTER_GROUP_INDICATORS,
    AGENT_ROUTER_VALUE_RANK_TERMS,
    AGENT_ROUTER_TIME_RANK_TERMS,
    AGENT_ROUTER_EXPLICIT_RANK_PATTERNS,
    AGENT_ROUTER_TOP_N_PER_GROUP_PATTERNS,
    AGENT_ROUTER_ORDINAL_WORDS,
    AGENT_ROUTER_ORDINAL_TARGET_TERMS,
    AGENT_ROUTER_WITHIN_SAME_PATTERN,
    AGENT_ROUTER_BY_TOKEN,
    AGENT_ROUTER_EXPLICIT_SORT_TERMS,
    AGENT_ROUTER_DOMAIN_CROSS_TABLE_EXCLUSION_TERMS,
    AGENT_ROUTER_GENERIC_CROSS_TABLE_EXCLUSION_TERMS,
    AGENT_SQL_RESERVED_EXTRA_TOKENS,
    AGENT_COLUMN_SEMANTIC_SUFFIX_KEYWORDS,
    AGENT_COLUMN_STOP_TOKENS,
    AGENT_COLUMN_TOKEN_SUFFIX_Y,
    AGENT_COLUMN_TOKEN_SUFFIX_IES,
    AGENT_COLUMN_TOKEN_SUFFIX_S,
    AGENT_COLUMN_TOKEN_SUFFIX_Y_MIN_LEN,
    AGENT_COLUMN_TOKEN_SUFFIX_IES_MIN_LEN,
    AGENT_COLUMN_TOKEN_SUFFIX_S_MIN_LEN,
    AGENT_COLUMN_CAMEL_SPLIT_PATTERN,
    AGENT_COLUMN_CAMEL_SPLIT_LOWER_PATTERN,
    AGENT_COLUMN_TOKEN_PATTERN,
    AGENT_CONTRACT_BASE_METRIC_TERMS,
    AGENT_CONTRACT_DOMAIN_METRIC_FALLBACK_TERMS,
    AGENT_CONTRACT_GENERIC_COUNT_NOUNS,
    HELPERS_ALL_OTHER_INTENT_PATTERNS,
    HELPERS_AVERAGE_INTENT_PATTERNS,
    HELPERS_SAME_GROUP_INTENT_PATTERNS,
    DOMAIN_HEURISTICS_ENABLED,
    AGENT_ROUTER_FK_SUFFIX_TERMS,
    AGENT_ROUTER_IGNORE_PATTERNS,
    AGENT_ROUTER_WEAK_EVIDENCE_TAGS,
    AGENT_ROUTER_EVIDENCE_TAG_FOR_EACH,
    AGENT_ROUTER_EVIDENCE_TAG_TABLE_ATTRIBUTE,
    AGENT_ROUTER_EVIDENCE_TAG_AGGREGATE_CONTEXT,
    AGENT_ROUTER_FOR_EACH_ENTITY_PATTERNS,
    AGENT_ROUTER_TABLE_ATTRIBUTE_PATTERNS,
    AGENT_ROUTER_AGGREGATE_CONTEXT_PATTERNS,
    AGENT_ROUTER_ID_REFERENCE_TERMS,
    AGENT_ROUTER_FK_VALUE_LIST_PATTERN,
    AGENT_ROUTER_WINNER_ROW_RECORD_PATTERN,
    AGENT_ROUTER_WINNER_ROW_FOR_EACH_PATTERN,
    AGENT_ROUTER_TIEBREAKER_PATTERN,
    AGENT_ROUTER_MISSING_EMPTY_PATTERN,
    AGENT_ROUTER_ESCALATE_ON_MULTI_TABLE,
    AGENT_ROUTER_ESCALATE_ON_WINNER_ROW,
    AGENT_ROUTER_ESCALATE_ON_EXCLUSION,
    AGENT_ROUTER_ESCALATE_ON_WINDOW_AGGREGATE,
    AGENT_ROUTER_ESCALATE_ON_EXPLICIT_RANK,
    AGENT_ROUTER_ESCALATE_ON_SUBQUERY_INTENT,
    AGENT_ROUTER_ESCALATE_ON_INCLUDE_ZERO_AGGREGATE,
    AGENT_ROUTER_SUBQUERY_INTENT_PATTERN,
    AGENT_ROUTER_ALLOW_SIMPLE_MULTI_TABLE_3B,
    AGENT_ROUTER_SIMPLE_MULTI_TABLE_MAX_TABLES,
    AGENT_ROUTER_ALLOW_SIMPLE_SINGLE_TABLE_SUBQUERY_3B,
    AGENT_ROUTER_COMPLEXITY_SIMPLE,
    AGENT_ROUTER_COMPLEXITY_COMPLEX,
    AGENT_ROUTER_COMPLEXITY_REJECTED,
    AGENT_ROUTER_QTYPE_LOOKUP,
    AGENT_ROUTER_QTYPE_AGGREGATE,
    AGENT_ROUTER_QTYPE_RANK,
    AGENT_ROUTER_QTYPE_INVALID,
    AGENT_ROUTER_SIMPLE_SINGLE_TABLE_SUBQUERY_PATTERN,
    AGENT_ROUTER_COMPLEX_SUBQUERY_CUE_PATTERN,
    AGENT_DRAFT_STRIP_DESC_ASC_BEFORE_FROM_PATTERN,
    AGENT_DRAFT_SELECT_FROM_PATTERN,
    AGENT_DRAFT_STRIP_DESC_ASC_BEFORE_COMMA_PATTERN,
    AGENT_DRAFT_LIMIT_NULL_ZERO_PATTERN,
    AGENT_DRAFT_LIMIT_CAPTURE_PATTERN,
    AGENT_DRAFT_LIMIT_REMOVE_PATTERN,
    AGENT_DRAFT_LIMIT_INTENT_TERMS,
    AGENT_DRAFT_TRY_JSON_MAX_ATTEMPTS,
    AGENT_KEEP_ALIVE_PARSE_PATTERN,
    AGENT_PREWARM_MEMORY_ERROR_TOKEN,
    AGENT_KEEP_ALIVE_DEFAULT_SECONDS,
    AGENT_KEEP_ALIVE_UNIT_SECONDS,
    AGENT_ROUTER_PRIMARY_FROM_IN_PATTERNS,
    AGENT_ROUTER_PRIMARY_ALL_ENTITY_PATTERN,
    AGENT_ROUTER_PRIMARY_REASON_FROM_IN,
    AGENT_ROUTER_PRIMARY_REASON_ALL,
    AGENT_ROUTER_PRIMARY_REASON_EXPLICIT_SUBJECT,
    AGENT_ROUTER_EXPLICIT_SUBJECT_ENTITY_PATTERNS,
    AGENT_ROUTER_FK_EVIDENCE_EXTRACT_PATTERN,
    AGENT_ROUTER_FK_EVIDENCE_PREFIX,
    AGENT_ROUTER_FK_PATTERN_CARD,
    AGENT_ROUTER_FK_PATTERN_DOT,
    AGENT_REFINE_WINNER_ROW_LEAK_PATTERN,
    AGENT_REFINE_TUPLE_IN_PATTERN,
    AGENT_REFINE_PARTITION_BY_PATTERN,
    AGENT_REFINE_BROKEN_ALIAS_PATTERN,
    AGENT_REFINE_MONTH_NAME_GROUP_PATTERN,
    AGENT_REFINE_SAFE_MONTH_SORT_FUNC_PATTERN,
    AGENT_REFINE_SAFE_MONTH_SORT_DATE_PATTERN,
    AGENT_REFINE_GROUP_BY_TOKEN,
    AGENT_REFINE_LEFT_JOIN_TOKEN,
    AGENT_REFINE_WILDCARD_TOKEN,
    AGENT_REFINE_LIKE_TOKEN,
    AGENT_REFINE_REGEX_TOKENS,
    AGENT_REFINE_MONTH_TOKEN,
    AGENT_REFINE_ORDER_TOKEN,
    AGENT_REFINE_FUNC_LOOKAHEAD_LEN,
    AGENT_REFINE_ALIAS_WINDOW_CHARS,
    AGENT_REFINE_ALIAS_TOKEN_WITH_SPACE,
    AGENT_REFINE_ALIAS_TOKEN,
    AGENT_REFINE_GROUP_CLAUSE_PATTERN,
    AGENT_REFINE_WHERE_CLAUSE_PATTERN,
    AGENT_REFINE_SELECT_FROM_CLAUSE_PATTERN,
    AGENT_REFINE_LOWER_WHERE_PATTERN,
    AGENT_REFINE_LOWER_HAVING_PATTERN,
    AGENT_REFINE_DEFINED_ALIAS_PATTERN,
    AGENT_REFINE_QUOTED_ALIAS_PATTERN,
    AGENT_REFINE_SUBQUERY_ALIAS_PATTERN,
    AGENT_REFINE_TABLE_ALIAS_PATTERN,
    AGENT_REFINE_SUBQUERY_ALIAS_RESERVED_TOKENS,
    AGENT_REFINE_TABLE_ALIAS_RESERVED_TOKENS,
    AGENT_REFINE_PRIMARY_FROM_PATTERN,
    AGENT_REFINE_HAS_SUBQUERY_PATTERN,
    AGENT_REFINE_MAIN_SELECT_FROM_PATTERN,
    AGENT_REFINE_SINGLE_QUOTED_STRING_PATTERN,
    AGENT_REFINE_DOUBLE_QUOTED_STRING_PATTERN,
    AGENT_REFINE_FUNCTION_CALL_PATTERN,
    AGENT_REFINE_SQL_TOKEN_PATTERN,
    AGENT_REFINE_TABLE_PLACEHOLDER_TOKEN,
    AGENT_REFINE_TABLE_PLACEHOLDER_DISCARD,
    AGENT_REFINE_JOIN_KEYWORD,
    AGENT_REFINE_PRIMARY_FROM_SKIP_TOKENS,
    AGENT_REFINE_TABLE_REF_TOKEN_PATTERN,
    AGENT_REFINE_GROUP_RISKY_COMPARE_PATTERN,
    AGENT_REFINE_GROUP_RISKY_PAREN_PATTERN,
    SQL_PARSE_DIALECT,
    SQL_WRITE_DIALECT,
)
from .llm_utils import build_llm_json, build_llm_repair
from .state import RAGState
from .utils import _dbg_print_two, try_llm_json, try_llm_sql_text, build_router_schema
from .generation import regenerate_with_extra_constraint
from .retrieval import build_expanded_context_from_state
from .intent_ir import build_intent_ir, is_ir_plannable
from .confidence import compute_sql_confidence


def _log_confidence_result(confidence_result: dict) -> None:
    """Log confidence score with threshold-aware formatting."""
    score = confidence_result.get("confidence", 0.0)
    signals = confidence_result.get("signals", {})
    warnings = confidence_result.get("warnings", [])
    
    if score >= AGENT_CONFIDENCE_THRESHOLD:
        _dbg_print_two(
            "[REFINER - CONFIDENCE]",
            f"score={score:.1%} (threshold={AGENT_CONFIDENCE_THRESHOLD:.0%}) ✓"
        )
    else:
        # Log warning with details
        low_signals = [f"{k}={v:.1%}" for k, v in signals.items() if v < 0.7]
        detail = f"score={score:.1%} (threshold={AGENT_CONFIDENCE_THRESHOLD:.0%}) ⚠"
        if low_signals:
            detail += f" | low_signals=[{', '.join(low_signals)}]"
        if warnings:
            detail += f" | warnings={len(warnings)}"
        _dbg_print_two("[REFINER - CONFIDENCE]", detail)


def _build_confidence_retry_instruction(confidence_result: dict) -> str:
    """Build LLM instruction with confidence feedback for retry."""
    warnings = confidence_result.get("warnings", [])
    signals = confidence_result.get("signals", {})
    score = confidence_result.get("confidence", 0.0)
    
    lines = [
        f"⚠ CONFIDENCE RETRY: Previous SQL had low confidence ({score:.0%}).",
        "Issues detected that need fixing:"
    ]
    
    # Add specific warnings (limit to top 5)
    for w in warnings[:5]:
        lines.append(f"  - {w}")
    
    # Add signal-specific hints
    if signals.get("syntax_valid", 1.0) < 0.5:
        lines.append("  - FIX: SQL has syntax errors. Ensure valid MySQL syntax.")
    if signals.get("identifiers_valid", 1.0) < 0.8:
        lines.append("  - FIX: Use ONLY tables/columns from ALLOWED IDENTIFIERS.")
    if signals.get("entity_coverage", 1.0) < 0.8:
        lines.append("  - FIX: Include tables mentioned in the question.")
    if signals.get("join_path_valid", 1.0) < 0.8:
        lines.append("  - FIX: Ensure JOINs follow valid FK relationships.")
    if signals.get("structure_plausible", 1.0) < 0.7:
        lines.append("  - FIX: Match SQL structure to question intent (COUNT/SUM/GROUP BY/ORDER BY).")
    
    lines.append("")
    lines.append("Generate a corrected SQL query that addresses these issues.")
    
    return "\n".join(lines)


def _attempt_confidence_retry(
    question: str,
    ctx: str,
    current_sql: str,
    confidence_result: dict,
    make_prompt_fn,
    prompt_kwargs: Optional[Dict[str, Any]] = None,
    decision: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    Attempt one retry with 7B model when confidence is below threshold.
    Returns dict with: used, sql, confidence, accepted
    """
    from .generation import regenerate_with_extra_constraint
    
    _dbg_print_two("[REFINER - CONFIDENCE]", "Attempting retry with 7B...")
    
    instruction = _build_confidence_retry_instruction(confidence_result)
    llm_7b = build_llm_repair(model=LARGE_MODEL)
    
    new_sql, _ = regenerate_with_extra_constraint(
        llm_7b,
        make_prompt_fn,
        ctx,
        question,
        instruction,
        prompt_kwargs=prompt_kwargs,
    )
    
    if not new_sql:
        _dbg_print_two("[REFINER - CONFIDENCE]", "Retry failed: no SQL generated")
        return {"used": True, "sql": current_sql, "confidence": confidence_result, "accepted": False}
    
    # Normalize and validate new SQL
    from .helpers_main import normalize_sql_pipeline
    candidate_sql = normalize_sql_pipeline(str(new_sql).strip(), question, ctx)
    if not candidate_sql.endswith(";"):
        candidate_sql += ";"
    
    # Compute confidence on new SQL
    new_confidence = compute_sql_confidence(candidate_sql, question, ctx)
    new_score = new_confidence.get("confidence", 0.0)
    old_score = confidence_result.get("confidence", 0.0)
    min_improvement = max(0.0, float(AGENT_CONFIDENCE_RETRY_MIN_IMPROVEMENT))
    improvement = float(new_score - old_score)
    
    _dbg_print_two(
        "[REFINER - CONFIDENCE]",
        (
            f"Retry result: {old_score:.1%} → {new_score:.1%} "
            f"(delta={improvement:.1%}, min={min_improvement:.1%})"
        )
    )

    # Minimum confidence improvement guard (avoid noisy tiny swaps).
    if improvement < min_improvement:
        _dbg_print_two(
            "[REFINER - CONFIDENCE]",
            "Retry rejected (below minimum improvement threshold)"
        )
        return {
            "used": True,
            "sql": current_sql,
            "confidence": confidence_result,
            "accepted": False,
            "reason": "below_min_improvement",
        }

    # Deterministic acceptance gate must pass before accepting confidence retry SQL.
    retry_gate = _deterministic_sql_acceptance_gate(
        candidate_sql,
        ctx=ctx,
        question=question,
        strict_join_key=AGENT_REGEN_PRECHECK_ENABLED,
    )
    candidate_sql = str(retry_gate.get("sql") or candidate_sql).strip()

    retry_parse_only_reasons = {
        str(r) for r in (retry_gate.get("reasons") or []) if str(r).strip()
    }
    if (
        not retry_gate.get("accepted", False)
        and retry_parse_only_reasons == {"parse_error"}
        and _can_bypass_parse_only_gate(sql=candidate_sql, ctx=ctx, decision=decision or {})
    ):
        retry_gate = {
            **dict(retry_gate or {}),
            "accepted": True,
            "reasons": [],
            "parse_error": "",
        }

    if not retry_gate.get("accepted", False):
        retry_reasons = {str(r) for r in (retry_gate.get("reasons") or []) if str(r).strip()}
        old_syntax = float((confidence_result.get("signals") or {}).get("syntax_valid", 1.0) or 0.0)
        new_syntax = float((new_confidence.get("signals") or {}).get("syntax_valid", 1.0) or 0.0)
        # Safety fallback: when the current SQL is syntactically invalid but the
        # retry is syntactically valid and only fails soft semantic-contract checks,
        # keep the valid retry to avoid execution hard-fail.
        if retry_reasons and retry_reasons.issubset({"semantic_contract_mismatch"}) and old_syntax < 0.5 and new_syntax >= 0.99:
            if not candidate_sql.endswith(";"):
                candidate_sql += ";"
            _dbg_print_two(
                "[REFINER - CONFIDENCE]",
                "Retry accepted with soft semantic warning (replaced syntactically invalid SQL)"
            )
            return {
                "used": True,
                "sql": candidate_sql,
                "confidence": new_confidence,
                "accepted": True,
                "reason": "accepted_soft_semantic_after_invalid_syntax",
                "gate": retry_gate,
                "validation_errors": list(retry_gate.get("identifier_errors") or validate_identifiers(candidate_sql, ctx)),
                "gate_pass": bool(retry_gate.get("accepted", False)),
                "gate_reasons": list(retry_gate.get("reasons") or []),
                "gate_reason_tags": _categorize_gate_reasons(list(retry_gate.get("reasons") or [])),
            }

        _dbg_print_two(
            "[REFINER - CONFIDENCE]",
            f"Retry rejected by acceptance gate: {retry_gate.get('reasons')}"
        )
        return {
            "used": True,
            "sql": current_sql,
            "confidence": confidence_result,
            "accepted": False,
            "reason": "gate_rejected",
            "gate": retry_gate,
        }

    if not candidate_sql.endswith(";"):
        candidate_sql += ";"

    # Recompute confidence on gated SQL in case AST/gate rewrites changed semantics.
    accepted_confidence = compute_sql_confidence(candidate_sql, question, ctx)
    accepted_score = accepted_confidence.get("confidence", 0.0)
    accepted_improvement = float(accepted_score - old_score)
    if accepted_improvement < min_improvement:
        _dbg_print_two(
            "[REFINER - CONFIDENCE]",
            "Retry rejected after gate (improvement dropped below threshold)"
        )
        return {
            "used": True,
            "sql": current_sql,
            "confidence": confidence_result,
            "accepted": False,
            "reason": "below_min_improvement_after_gate",
            "gate": retry_gate,
        }

    retry_validation_errors = list(
        retry_gate.get("identifier_errors") or validate_identifiers(candidate_sql, ctx)
    )
    retry_gate_reasons = list(retry_gate.get("reasons") or [])

    _dbg_print_two("[REFINER - CONFIDENCE]", "Retry accepted (improved + gate-passed)")
    return {
        "used": True,
        "sql": candidate_sql,
        "confidence": accepted_confidence,
        "accepted": True,
        "reason": "accepted",
        "gate": retry_gate,
        "validation_errors": retry_validation_errors,
        "gate_pass": bool(retry_gate.get("accepted", False)),
        "gate_reasons": retry_gate_reasons,
        "gate_reason_tags": _categorize_gate_reasons(retry_gate_reasons),
    }


from .schema_analyzer import (
    get_all_tables,
    get_column_description,
    get_column_role,
    get_singular_form,
    get_schema_analyzer,
    get_id_column,
    get_name_column,
    get_fk_relationships,
    get_transaction_tables,
)
from .helpers_main import (get_single_allowed_table, validate_identifiers, _get_requires_regex, _regex_requirement_ok,
    _regex_requirement_ok, _get_qtype, make_router_prompt, make_generation_prompt, sanitize_notes, normalize_sql_pipeline,
    _find_tables_with_implicit_columns, _extract_fk_from_context, _ordered_token_is_sort_only_context)
from .helpers_core import (
    has_join,
    _ast_guardrails,
    _fix_winner_row_metric_order_priority,
    _question_wants_latest_year_scope_with_aggregate,
    _rewrite_missing_temporal_exclusion_from_null_filter,
    _rewrite_grouped_winner_row_ranked_aggregate,
    _rewrite_joined_ranked_subquery_to_outer_winner,
    _rewrite_malformed_window_winner_clause,
    _question_wants_winner_row,
    _question_wants_anti_existence,
    _question_wants_dual_fact_rollup,
    _question_has_set_operation_intent,
    _enforce_set_operation_operator_from_question,
    _wants_distinct_from_question,
    _question_wants_count_distinct,
    _ensure_reasonable_distinct,
    _ensure_reasonable_count_distinct,
    _rewrite_missing_joined_agg_threshold_filter,
    _question_requires_regex,
    _question_requires_structural_regex,
    _question_requires_word_boundary_regex,
    _question_requires_inner_join,
    _question_requests_ordering,
    _question_wants_include_zero,
    _ensure_explicit_simple_projection_columns,
    _align_requested_description_variant,
    _align_requested_joined_description_variant,
    _drop_unrequested_difference_operand_columns,
    _drop_unrequested_missing_filter_columns,
    _drop_raw_column_covered_by_function,
    _repair_misplaced_select_distinct,
    _enforce_requested_order_direction,
    _parse_allowed_identifiers,
    _remove_top_level_order_by_if_not_requested,
    _SQL_KEYWORDS,
)
from .rulebook import (
    any_phrase_in_text,
    case_bucket_aggregate_request_regex_patterns,
    case_bucket_aggregate_request_terms,
    case_bucket_comparison_terms,
    case_bucket_conditional_terms,
    case_bucket_language_regex_patterns,
    case_bucket_missing_terms,
    case_bucket_numeric_terms,
    case_bucket_numeric_threshold_regex_patterns,
    case_bucket_pattern_terms,
    case_bucket_temporal_terms,
    global_avg_baseline_average_terms,
    global_avg_baseline_reference_terms,
    global_avg_baseline_terms,
    global_avg_comparison_terms,
    group_average_difference_window_phrases,
    numeric_threshold_comparison_terms,
    numeric_threshold_number_words,
    order_term_token_stopwords,
    ordering_direction_asc_terms,
    ordering_direction_desc_terms,
    ordering_superlative_keywords,
    ordering_temporal_asc_terms,
    ordering_temporal_desc_terms,
    ordering_temporal_intent_terms,
    offset_window_ordering_context_regex_patterns,
    offset_window_sequence_terms,
    projection_null_fill_action_terms,
    projection_null_fill_conditional_terms,
    projection_null_fill_fallback_terms,
    projection_null_fill_missing_terms,
    rule_list,
    semantic_phrase_token_stopwords,
    semantic_contract_average_terms,
    semantic_contract_count_how_many_phrases,
    semantic_contract_count_number_of_prefixes,
    semantic_contract_ordinal_terms,
    semantic_contract_per_group_terms,
    semantic_contract_count_terms,
    semantic_contract_identifier_like_targets,
    semantic_contract_ratio_count_reference_phrases,
    semantic_contract_ratio_definition_terms,
    semantic_contract_ratio_operator_terms,
    semantic_contract_sum_terms,
    semantic_contract_sum_total_count_phrases,
    semantic_contract_sum_total_prefix_terms,
    temporal_year_column_terms,
)


def _dbg_agent3_retry(check_name: str, detail: str) -> None:
    _dbg_print_two(f"[REFINER] Retry Gate - {check_name}", detail)


def _dbg_safety_block(scope: str, detail: str) -> None:
    label = "REFINER" if str(scope).strip().lower() == "agent3" else str(scope)
    _dbg_print_two(f"[{label}] Safety Block", detail)


def _has_any_phrase(text: str, phrases: List[str]) -> bool:
    return bool(phrases) and any_phrase_in_text(text or "", phrases)


def _phrase_alt_pattern(phrases: List[str]) -> str:
    out: List[str] = []
    for phrase in phrases:
        p = str(phrase or "").strip()
        if not p:
            continue
        out.append(re.escape(p).replace(r"\ ", r"\s+"))
    return "|".join(sorted(out, key=len, reverse=True))


def _temporal_extrema_terms() -> List[str]:
    latest_terms = rule_list(("temporal_intent", "latest_keywords"), ["latest", "newest", "most recent"])
    earliest_terms = rule_list(("temporal_intent", "earliest_keywords"), ["earliest", "oldest"])
    seen: Set[str] = set()
    out: List[str] = []
    for term in latest_terms + earliest_terms:
        norm = str(term or "").strip().lower()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(str(term).strip())
    return out


def _temporal_extrema_alt() -> str:
    return _phrase_alt_pattern(_temporal_extrema_terms())


def _derive_semantic_contract(question: str, ctx: str = "") -> Dict[str, bool]:
    """
    Minimal semantic contract derived from question wording.

    This does not replace routing or the acceptance gate. It centralizes a few
    high-value meaning checks so final SQL can be repaired (or retried) against
    the question's intent in one place.
    """
    wants_count_distinct = _question_wants_count_distinct(question)
    wants_projection_distinct = _wants_distinct_from_question(question) and not wants_count_distinct
    q_lower = str(question or "").strip().lower()
    ordinal_row_contract = bool(
        re.search(
            r"\b(?:first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th)\b"
            r"(?:\s+\w+){0,8}\s+\b(?:row|record|entry)\b",
            q_lower,
            flags=re.I,
        )
        and re.search(
            r"\b(?:sort(?:ed|ing)?|order(?:ed|ing)?)\b[^.]{0,96}\bby\b",
            q_lower,
            flags=re.I,
        )
    )
    wants_winner_row = _question_wants_winner_row(question) or ordinal_row_contract
    try:
        allowed_ids = _parse_allowed_identifiers(ctx or "")
        context_table_count = len(allowed_ids or {})
    except Exception:
        context_table_count = 0
    relational_ctx = context_table_count >= 2
    wants_include_zero = relational_ctx and _question_wants_include_zero(question, ctx)
    wants_projection_null_fill = _question_requests_projection_null_fill(question)
    wants_anti_existence = (
        relational_ctx
        and _question_wants_anti_existence(question, ctx)
        and not wants_include_zero
        and not wants_projection_null_fill
    )
    wants_offset_window = _question_has_offset_window_sequence_intent(question)
    wants_same_group_avg_difference = (
        any(re.search(pat, q_lower, flags=re.I) for pat in HELPERS_AVERAGE_INTENT_PATTERNS)
        and any(re.search(pat, q_lower, flags=re.I) for pat in HELPERS_SAME_GROUP_INTENT_PATTERNS)
        and any_phrase_in_text(q_lower, group_average_difference_window_phrases())
    )
    wants_final_order = _question_requests_ordering(question)
    wants_boundary_regex = _question_requires_word_boundary_regex(question)
    wants_regex = _question_requires_regex(question)
    wants_structural_regex = _question_requires_structural_regex(question)
    wants_scalar_extrema_date_filter = _question_requires_scalar_extrema_date_filter(question)
    if wants_offset_window:
        # Previous/next row context is an offset-window task, not a
        # winner-row selection. Keeping these separate avoids unnecessary
        # ROW_NUMBER/rn retry gates on otherwise valid LAG/LEAD projections.
        wants_winner_row = False
    # Same-group/window-partition requirements belong to window-aggregate
    # families. True winner-row questions should be governed by their
    # ROW_NUMBER / ORDER BY shape instead of an inferred PARTITION BY contract.
    required_window_partition_col = ""
    # "all other ... average ... same <group>" questions are validly solved
    # with correlated subqueries that exclude the current row, so forcing a
    # PARTITION BY contract here would cause unnecessary regeneration.
    all_other_same_group_avg = (
        any(re.search(pat, q_lower, flags=re.I) for pat in HELPERS_ALL_OTHER_INTENT_PATTERNS)
        and any(re.search(pat, q_lower, flags=re.I) for pat in HELPERS_AVERAGE_INTENT_PATTERNS)
        and any(re.search(pat, q_lower, flags=re.I) for pat in HELPERS_SAME_GROUP_INTENT_PATTERNS)
    )
    same_group_avg_comparison_filter = (
        any(re.search(pat, q_lower, flags=re.I) for pat in HELPERS_AVERAGE_INTENT_PATTERNS)
        and any(re.search(pat, q_lower, flags=re.I) for pat in HELPERS_SAME_GROUP_INTENT_PATTERNS)
        and re.search(
            r"\b(?:higher|greater|more|above|over|lower|less|below|under)\s+than\s+(?:the\s+)?(?:avg|average|mean)\b"
            r"|\b(?:avg|average|mean)\b[^.]{0,80}\b(?:higher|greater|more|above|over|lower|less|below|under)\b"
            r"|\bwhose\b[^.]{0,80}\b(?:higher|greater|more|above|over|lower|less|below|under)\b[^.]{0,80}\b(?:avg|average|mean)\b",
            q_lower,
            flags=re.I,
        )
    )
    if not wants_winner_row and not all_other_same_group_avg and not same_group_avg_comparison_filter:
        required_window_partition_col = _resolve_required_window_partition_column(question, ctx)
    return {
        "requires_projection_distinct": wants_projection_distinct,
        "requires_count_distinct": wants_count_distinct,
        "requires_winner_row": wants_winner_row,
        "requires_anti_existence": wants_anti_existence,
        "requires_include_zero": wants_include_zero,
        "requires_final_order": wants_final_order,
        "requires_offset_window": wants_offset_window,
        "requires_same_group_avg_difference": wants_same_group_avg_difference,
        "requires_boundary_regex": wants_boundary_regex,
        "requires_regex": wants_regex,
        "requires_structural_regex": wants_structural_regex,
        "requires_scalar_extrema_date_filter": wants_scalar_extrema_date_filter,
        "requires_window_partition": bool(required_window_partition_col),
        "required_window_partition_col": required_window_partition_col,
    }


def _router_has_dual_fact_rollup(question: str, relevant_tables: List[str]) -> bool:
    """
    Detect the dual-fact rollup family generically for routing escalation.

    The router should only escalate when both conditions hold:
    1. the question asks for a count + separate total/status rollup shape
    2. the routed tables actually contain a base entity with at least two
       direct child fact tables in the schema graph
    """
    q = (question or "").strip()
    relevant_lower = {
        str(table or "").strip().lower()
        for table in (relevant_tables or [])
        if str(table or "").strip()
    }
    if not q or len(relevant_lower) < 3:
        return False
    if not _question_wants_dual_fact_rollup(q):
        return False

    try:
        fk_map_raw = get_fk_relationships() or {}
    except Exception:
        fk_map_raw = {}
    try:
        transaction_tables = {
            str(table).strip().lower()
            for table in (get_transaction_tables() or [])
            if str(table).strip()
        }
    except Exception:
        transaction_tables = set()

    direct_children: Dict[str, Set[str]] = {}
    direct_fact_children: Dict[str, Set[str]] = {}
    for child_table, fk_targets in fk_map_raw.items():
        child_l = str(child_table or "").strip().lower()
        if child_l not in relevant_lower:
            continue
        for target_table in (fk_targets or {}).values():
            target_l = str(target_table or "").strip().lower()
            if not target_l or target_l not in relevant_lower or target_l == child_l:
                continue
            direct_children.setdefault(target_l, set()).add(child_l)
            if transaction_tables and child_l not in transaction_tables:
                continue
            direct_fact_children.setdefault(target_l, set()).add(child_l)

    if any(len(children) >= 2 for children in direct_fact_children.values()):
        return True

    # Some schemas under-classify operational fact tables in transaction metadata
    # even though the FK graph clearly shows multiple child facts under one base.
    # Treat the FK child structure as the primary signal and use transaction-table
    # metadata only as an additional precision hint, not a hard requirement.
    if any(len(children) >= 2 for children in direct_children.values()):
        return True

    if transaction_tables:
        routed_fact_tables = relevant_lower & transaction_tables
        routed_non_fact_tables = relevant_lower - transaction_tables
        return bool(len(routed_fact_tables) >= 2 and routed_non_fact_tables)

    return False


def _has_winner_row_shape(sql: str) -> bool:
    s = str(sql or "")
    s_up = s.upper()
    if not s_up:
        return False
    has_rank_window = bool(re.search(r"\b(?:ROW_NUMBER|RANK|DENSE_RANK)\s*\(", s_up))
    if not has_rank_window:
        return False

    rank_aliases = {
        "rn",
        "rank",
        "row_num",
        "row_number",
    }
    for m_alias in re.finditer(
        r"\b(?:ROW_NUMBER|RANK|DENSE_RANK)\s*\([^)]*\)\s*OVER\s*\([\s\S]{0,500}?\)\s+AS\s+`?([A-Za-z_][A-Za-z0-9_]*)`?",
        s,
        flags=re.I | re.S,
    ):
        alias = (m_alias.group(1) or "").strip().lower()
        if alias:
            rank_aliases.add(alias)

    alias_alt = "|".join(re.escape(a) for a in sorted(rank_aliases, key=len, reverse=True))
    has_rank_filter = bool(
        re.search(
            rf"\bWHERE\b[^;]*\b(?:[A-Za-z_][A-Za-z0-9_]*\.)?`?(?:{alias_alt})`?\s*(?:=|<=)\s*\d+\b",
            s,
            flags=re.I | re.S,
        )
        or re.search(
            rf"\bQUALIFY\b[^;]*\b(?:[A-Za-z_][A-Za-z0-9_]*\.)?`?(?:{alias_alt})`?\s*(?:=|<=)\s*\d+\b",
            s,
            flags=re.I | re.S,
        )
    )
    return has_rank_filter


def _has_top_level_order_by(sql: str) -> bool:
    s = (sql or "").strip()
    if not s:
        return False
    try:
        import sqlglot
        parsed = sqlglot.parse_one(s, read=SQL_PARSE_DIALECT)
        order = parsed.args.get("order") if parsed is not None else None
        return order is not None
    except Exception:
        return bool(re.search(r"\bORDER\s+BY\b", s, flags=re.I))


def _question_requires_scalar_extrema_date_filter(question: str) -> bool:
    q = (question or "").lower().strip()
    if not q:
        return False

    if _question_wants_winner_row(question):
        return False
    temporal_alt = _temporal_extrema_alt()

    if temporal_alt and re.search(
        r"\b(?:starting|beginning|going)\s+from\s+the\s+"
        rf"(?:{temporal_alt})\b",
        q,
        flags=re.I,
    ):
        return False

    if re.search(
        r"\bif\s+(?:there\s+is\s+a\s+tie|there\s+are\s+ties|multiple|same)\b",
        q,
        flags=re.I,
    ) and temporal_alt and re.search(
        rf"\b(?:{temporal_alt})\b[^.]{{0,24}}\bdate\b",
        q,
        flags=re.I,
    ):
        return False

    return bool(
        temporal_alt and re.search(
            r"\b(?:on|for|with|from)\s+the\s+"
            rf"(?:{temporal_alt})\s+"
            r"(?:[a-z][a-z0-9_]*\s+){0,3}date\b",
            q,
            flags=re.I,
        )
    )


def _extract_same_group_phrase(question: str) -> str:
    q = (question or "").strip().lower()
    if not q:
        return ""
    temporal_alt = _temporal_extrema_alt()

    # Tie-scoped temporal phrases like "that same most recent date" are
    # winner-row tie-break cues, not PARTITION BY hints.
    if temporal_alt and re.search(
        r"\b(?:if\s+(?:there\s+are\s+)?multiple|if\s+there\s+is\s+a\s+tie|"
        r"if\s+there\s+are\s+ties|tie|tied|pick|choose)\b",
        q,
        flags=re.I,
    ) and re.search(
        rf"\bsame\s+(?:(?:{temporal_alt})\s+)?"
        r"(?:date|payment\s+date|order\s+date|time)\b",
        q,
        flags=re.I,
    ):
        return ""
    if re.search(r"\bsame\s+year\b", q, flags=re.I) and temporal_alt and re.search(
        rf"\b(?:{temporal_alt})\b[^.]{{0,40}}\byear\b",
        q,
        flags=re.I | re.S,
    ):
        return ""

    boundary_terms = (
        r"when|where|after|before|ordered|sorted|with|and|then|that|"
        r"which|who|whose|including|excluding|counting|except"
    )
    patterns = [
        rf"\b(?:within|in)\s+the\s+same\s+([a-z][a-z0-9_ ]{{1,40}}?)(?=\s+\b(?:{boundary_terms})\b|[.,;]|$)",
        rf"\b(?:that|the)\s+same\s+([a-z][a-z0-9_ ]{{1,40}}?)(?=\s+\b(?:{boundary_terms})\b|[.,;]|$)",
        rf"\bsame\s+([a-z][a-z0-9_ ]{{1,40}}?)(?=\s+\b(?:{boundary_terms})\b|[.,;]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, q, flags=re.I)
        if match:
            return re.sub(r"\s+", " ", (match.group(1) or "").strip())
    return ""


def _semantic_phrase_tokens(text: str) -> List[str]:
    raw = re.sub(AGENT_COLUMN_CAMEL_SPLIT_PATTERN, r"\1 \2", str(text or ""))
    tokens = re.findall(r"[a-z0-9]+", raw.lower())
    stop_words = semantic_phrase_token_stopwords()
    return [tok for tok in tokens if tok and tok not in stop_words]


def _split_top_level_csv(text: str) -> List[str]:
    items: List[str] = []
    buf: List[str] = []
    depth = 0
    in_sq = False
    in_dq = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "'" and not in_dq:
            if in_sq and i + 1 < len(text) and text[i + 1] == "'":
                buf.append(ch)
                buf.append(text[i + 1])
                i += 2
                continue
            in_sq = not in_sq
            buf.append(ch)
            i += 1
            continue
        if ch == '"' and not in_sq:
            in_dq = not in_dq
            buf.append(ch)
            i += 1
            continue
        if not in_sq and not in_dq:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif ch == "," and depth == 0:
                item = "".join(buf).strip()
                if item:
                    items.append(item)
                buf = []
                i += 1
                continue
        buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        items.append(tail)
    return items


def _resolve_required_window_partition_column(question: str, ctx: str = "", sql: str = "") -> str:
    hint = _extract_same_group_phrase(question)
    if not hint:
        return ""

    allowed_ids = _parse_allowed_identifiers(ctx or "")
    if not allowed_ids:
        return ""

    primary_table = ""
    from_match = re.search(
        r"\bFROM\s+`?([A-Za-z_][A-Za-z0-9_]*)`?(?:\s+(?:AS\s+)?[A-Za-z_][A-Za-z0-9_]*)?\b",
        sql or "",
        flags=re.I,
    )
    if from_match:
        primary_table = (from_match.group(1) or "").strip("`").lower()

    candidate_tables: List[Tuple[str, Set[str]]] = []
    if primary_table:
        for table_name, cols in allowed_ids.items():
            if table_name.lower() == primary_table:
                candidate_tables.append((table_name, cols))
                break
    if not candidate_tables:
        candidate_tables = list(allowed_ids.items())

    hint_norm = " ".join(_semantic_phrase_tokens(hint))
    hint_tokens = set(_semantic_phrase_tokens(hint))
    if not hint_tokens:
        return ""

    best_col = ""
    best_score = 0
    q_lower = (question or "").lower()

    for _, cols in candidate_tables:
        for col in cols:
            col_spaced = re.sub(AGENT_COLUMN_CAMEL_SPLIT_PATTERN, r"\1 \2", str(col or "")).replace("_", " ").lower()
            col_tokens = set(_semantic_phrase_tokens(col_spaced))
            if not col_tokens:
                continue

            score = 0
            matched_hint = False
            if col_spaced == hint_norm:
                score += 100
                matched_hint = True
            if hint_norm and hint_norm in col_spaced:
                score += 60
                matched_hint = True
            if hint_tokens.issubset(col_tokens):
                score += 45 + (5 * len(hint_tokens))
                matched_hint = True
            if hint_tokens.intersection(col_tokens):
                score += 10 * len(hint_tokens.intersection(col_tokens))
                matched_hint = True
            if len(hint_tokens) == 1:
                only_hint = next(iter(hint_tokens))
                if only_hint in col_tokens:
                    score += 35
                    matched_hint = True
                if col_spaced.startswith(f"{only_hint} ") and any(
                    suffix in col_tokens for suffix in {"code", "id", "number", "key"}
                ):
                    suffix_terms = {"code", "id", "number", "key"}
                    descriptor_tokens = col_tokens - hint_tokens - suffix_terms
                    # For "same <entity>", prefer the direct entity identifier
                    # (<entity>Number/<entity>Id/...) over detail columns like
                    # <entity>LineNumber that happen to contain the same token.
                    score += 100 if not descriptor_tokens else max(0, 20 - (20 * len(descriptor_tokens)))
                    matched_hint = True
            if matched_hint and _question_mentions_column(q_lower, str(col)):
                score += 25

            if score > best_score:
                best_score = score
                best_col = str(col)

    return best_col if best_score > 0 else ""


def _has_required_window_partition(sql: str, partition_col: str) -> bool:
    if not sql or not partition_col:
        return False
    return bool(
        re.search(
            rf"\bOVER\s*\([^)]*\bPARTITION\s+BY\s+(?:[A-Za-z_][A-Za-z0-9_]*\.)?`?{re.escape(partition_col)}`?\b",
            sql,
            flags=re.I | re.S,
        )
    )


def _rewrite_required_window_partition(sql: str, partition_col: str) -> str:
    s = (sql or "").strip()
    if not s or not partition_col or "OVER" not in s.upper():
        return sql

    def _rewrite_inner(inner_text: str) -> str:
        inner = (inner_text or "").strip()
        if not inner:
            return f"PARTITION BY {partition_col}"

        without_partition = re.sub(
            r"(?is)^\s*PARTITION\s+BY\s+.+?(?=(?:\bORDER\s+BY\b)|$)",
            "",
            inner,
            count=1,
        ).strip()
        if without_partition:
            return f"PARTITION BY {partition_col} {without_partition}".strip()
        return f"PARTITION BY {partition_col}"

    rewritten, count = re.subn(
        r"(?is)OVER\s*\(([^()]*)\)",
        lambda m: f"OVER ({_rewrite_inner(m.group(1))})",
        s,
        count=1,
    )
    return rewritten if count else sql


def _rewrite_grouped_aggregate_to_window_partition(sql: str, partition_col: str, question: str) -> str:
    s = (sql or "").strip().rstrip(";")
    if not s or not partition_col:
        return sql
    if "OVER" in s.upper() or "GROUP BY" not in s.upper():
        return sql
    if re.search(r"\b(?:HAVING|UNION|INTERSECT|EXCEPT)\b", s, flags=re.I):
        return sql

    def _split_expr_alias(item: str) -> Tuple[str, str]:
        m_alias = re.match(
            r"""(?is)^\s*(?P<expr>.*?)\s+AS\s+(?P<alias>`?[A-Za-z_][A-Za-z0-9_]*`?|'[^']+'|"[^"]+")\s*$""",
            str(item or "").strip(),
        )
        if not m_alias:
            return str(item or "").strip(), ""
        return (m_alias.group("expr") or "").strip(), (m_alias.group("alias") or "").strip()

    def _norm_col_ref(expr: str) -> str:
        text = str(expr or "").strip().strip("`")
        if "." in text:
            text = text.split(".")[-1].strip("`")
        return re.sub(r"\s+", "", text).lower()

    def _find_partition_expr(group_clause: str) -> str:
        for group_item in _split_top_level_csv(group_clause or ""):
            group_expr = str(group_item or "").strip()
            if _norm_col_ref(group_expr) == str(partition_col or "").strip().lower():
                return group_expr
        return str(partition_col or "").strip()

    # Join-aware grouped aggregate fallback:
    # convert a grouped per-row metric plus same-arg MAX/MIN companion into a
    # windowed per-group extrema on the grouped metric itself.
    flat_match = re.match(
        r"(?is)^\s*SELECT\s+(?P<select>.+?)\s+FROM\s+(?P<from>.+?)"
        r"(?:\s+WHERE\s+(?P<where>.*?))?"
        r"\s+GROUP\s+BY\s+(?P<group>.*?)(?:\s+ORDER\s+BY\s+(?P<order>.*?))?\s*$",
        s,
    )
    if flat_match:
        select_clause = (flat_match.group("select") or "").strip()
        from_clause = (flat_match.group("from") or "").strip()
        where_clause = (flat_match.group("where") or "").strip()
        group_clause = (flat_match.group("group") or "").strip()
        order_clause = (flat_match.group("order") or "").strip()
        partition_expr = _find_partition_expr(group_clause)
        if select_clause and from_clause and group_clause and partition_expr:
            aggregate_items: List[Dict[str, str]] = []
            select_items = _split_top_level_csv(select_clause)
            for idx, raw_item in enumerate(select_items):
                expr_sql, alias_sql = _split_expr_alias(raw_item)
                m_agg = re.match(
                    r"""(?is)^\s*(?P<expr>(?P<func>AVG|COUNT|SUM|MIN|MAX)\s*\(\s*(?P<arg>.+?)\s*\))\s*$""",
                    expr_sql,
                )
                if not m_agg:
                    continue
                aggregate_items.append(
                    {
                        "index": str(idx),
                        "func": (m_agg.group("func") or "").upper(),
                        "arg": (m_agg.group("arg") or "").strip(),
                        "expr": (m_agg.group("expr") or "").strip(),
                        "alias": alias_sql,
                    }
                )

            candidate_pair: Tuple[Dict[str, str], Dict[str, str]] | None = None
            for target in aggregate_items:
                if target.get("func") not in {"MAX", "MIN"}:
                    continue
                target_arg = _norm_col_ref(target.get("arg") or "")
                if not target_arg:
                    continue
                matches = [
                    src for src in aggregate_items
                    if src is not target
                    and src.get("func") in {"SUM", "COUNT", "AVG"}
                    and _norm_col_ref(src.get("arg") or "") == target_arg
                ]
                if len(matches) == 1:
                    candidate_pair = (target, matches[0])
                    break

            if candidate_pair is not None:
                target_item, source_item = candidate_pair
                target_idx = int(target_item["index"])
                window_expr = (
                    f"{target_item['func']}({source_item['expr']}) "
                    f"OVER (PARTITION BY {partition_expr})"
                )
                if target_item.get("alias"):
                    window_expr += f" AS {target_item['alias']}"
                rebuilt_items = list(select_items)
                rebuilt_items[target_idx] = window_expr
                out = f"SELECT {', '.join(rebuilt_items)} FROM {from_clause}"
                if where_clause:
                    out += f" WHERE {where_clause}"
                out += f" GROUP BY {group_clause}"
                if order_clause:
                    out += f" ORDER BY {order_clause}"
                simple_out = out + ";"
                if simple_out != sql:
                    return simple_out

    match = re.match(
        r"(?is)^\s*SELECT\s+(?P<select>.+?)\s+FROM\s+(?P<table>`?[A-Za-z_][A-Za-z0-9_]*`?)"
        r"(?:\s+(?:AS\s+)?(?P<alias>[A-Za-z_][A-Za-z0-9_]*))?"
        r"(?:\s+WHERE\s+(?P<where>.*?))?"
        r"\s+GROUP\s+BY\s+(?P<group>.*?)(?:\s+ORDER\s+BY\s+(?P<order>.*?))?\s*$",
        s,
    )
    if match:
        select_clause = (match.group("select") or "").strip()
        table = (match.group("table") or "").strip()
        alias = (match.group("alias") or "").strip()
        where_clause = (match.group("where") or "").strip()
        order_clause = (match.group("order") or "").strip()
        if select_clause and table:
            select_items = _split_top_level_csv(select_clause)
            if select_items:
                aggregate_idx = -1
                aggregate_func = ""
                aggregate_arg = ""
                aggregate_alias = ""
                aggregate_expr = ""

                for idx, item in enumerate(select_items):
                    agg_match = re.match(
                        r"(?is)^\s*(?P<func>AVG|COUNT|SUM|MIN|MAX)\s*\(\s*(?P<arg>[^)]*?)\s*\)"
                        r"(?:\s+AS\s+(?P<alias>`?[A-Za-z_][A-Za-z0-9_]*`?|'[^']+'|\"[^\"]+\"))?\s*$",
                        item,
                    )
                    if agg_match:
                        if aggregate_idx >= 0:
                            return sql
                        aggregate_idx = idx
                        aggregate_func = (agg_match.group("func") or "").upper()
                        aggregate_arg = (agg_match.group("arg") or "").strip()
                        aggregate_alias = (agg_match.group("alias") or "").strip()
                        aggregate_expr = item.strip()

                if aggregate_idx >= 0 and aggregate_func:
                    projection_items = [
                        item.strip()
                        for idx, item in enumerate(select_items)
                        if idx != aggregate_idx and item.strip()
                    ]

                    simple_arg_match = re.match(
                        r"(?is)^(?:[A-Za-z_][A-Za-z0-9_]*\.)?([A-Za-z_][A-Za-z0-9_]*)$",
                        aggregate_arg,
                    )
                    arg_col = (simple_arg_match.group(1) if simple_arg_match else "").strip()
                    if arg_col and _question_mentions_column((question or "").lower(), arg_col):
                        projection_tokens = " ".join(projection_items).lower()
                        if arg_col.lower() not in projection_tokens:
                            projection_items.append(arg_col)

                    window_expr = f"{aggregate_func}({aggregate_arg}) OVER (PARTITION BY {partition_col})"
                    if aggregate_alias:
                        window_expr += f" AS {aggregate_alias}"
                    elif aggregate_expr:
                        inferred_alias_match = re.search(r"\bAS\s+([A-Za-z_][A-Za-z0-9_]*)\b", aggregate_expr, flags=re.I)
                        if inferred_alias_match:
                            window_expr += f" AS {inferred_alias_match.group(1)}"
                    projection_items.append(window_expr)

                    if projection_items:
                        out = f"SELECT {', '.join(projection_items)} FROM {table}"
                        if alias:
                            out += f" AS {alias}"
                        if where_clause:
                            out += f" WHERE {where_clause}"
                        if order_clause:
                            out += f" ORDER BY {order_clause}"
                        simple_out = out + ";"
                        if simple_out != sql:
                            return simple_out

    # Conservative join-aware fallback:
    # if a same-group COUNT(*)/COUNT(col) question is answered with a grouped
    # one-join query, rebuild it as a window COUNT over the detail table only.
    q_low = (question or "").lower()
    if not re.search(r"\b(?:how\s+many|count|number\s+of|total)\b", q_low, flags=re.I):
        return sql

    try:
        import sqlglot
        from sqlglot import exp

        parsed = sqlglot.parse_one(s, read=SQL_PARSE_DIALECT)
        sel = parsed.find(exp.Select)
        if sel is None or sel.find(exp.Window):
            return sql
        joins = list(sel.args.get("joins") or [])
        if len(joins) != 1:
            return sql

        from_expr = sel.args.get("from") or sel.args.get("from_")
        base_src = getattr(from_expr, "this", None)
        if not isinstance(base_src, exp.Table):
            return sql

        base_alias = str(base_src.alias_or_name or "").strip()
        if not base_alias:
            base_alias = str(base_src.name or "").strip()
        if not base_alias:
            return sql

        non_agg_exprs: List[str] = []
        agg_alias = ""
        agg_count = 0
        for item in list(sel.expressions or []):
            expr_node = item.this if isinstance(item, exp.Alias) else item
            if expr_node.find(exp.Window):
                return sql
            agg_nodes = list(expr_node.find_all(exp.AggFunc))
            if agg_nodes:
                if len(agg_nodes) != 1 or agg_count > 0:
                    return sql
                agg = agg_nodes[0]
                if not isinstance(agg, exp.Count):
                    return sql
                agg_count += 1
                if isinstance(item, exp.Alias):
                    agg_alias = str(item.alias or "").strip()
                continue

            col_tables = {
                str(col.table or "").strip()
                for col in expr_node.find_all(exp.Column)
                if str(col.table or "").strip()
            }
            if col_tables and col_tables != {base_alias}:
                return sql
            non_agg_exprs.append(item.sql(dialect=SQL_WRITE_DIALECT))

        if agg_count != 1 or not non_agg_exprs:
            return sql

        where_expr = sel.args.get("where")
        if where_expr is not None:
            where_tables = {
                str(col.table or "").strip()
                for col in where_expr.find_all(exp.Column)
                if str(col.table or "").strip()
            }
            if where_tables and where_tables != {base_alias}:
                return sql

        order_expr = sel.args.get("order")
        if order_expr is not None:
            order_tables = {
                str(col.table or "").strip()
                for col in order_expr.find_all(exp.Column)
                if str(col.table or "").strip()
            }
            if order_tables and order_tables != {base_alias}:
                return sql

        partition_ref = f"{base_alias}.{partition_col}"
        window_item = f"COUNT(*) OVER (PARTITION BY {partition_ref})"
        if agg_alias:
            window_item += f" AS {agg_alias}"

        rebuilt = f"SELECT {', '.join(non_agg_exprs + [window_item])} FROM {base_src.sql(dialect=SQL_WRITE_DIALECT)}"
        if where_expr is not None:
            rebuilt += f" WHERE {where_expr.sql(dialect=SQL_WRITE_DIALECT)}"
        if order_expr is not None:
            rebuilt += f" {order_expr.sql(dialect=SQL_WRITE_DIALECT)}"
        return rebuilt + ";"
    except Exception:
        return sql


def _has_scalar_extrema_date_filter_shape(sql: str) -> bool:
    s = (sql or "").strip()
    if not s:
        return False
    return bool(
        re.search(
            r"\bWHERE\b[^;]*\b[a-zA-Z_][a-zA-Z0-9_.`]*date[a-zA-Z0-9_.`]*\s*=\s*"
            r"\(\s*SELECT\s+(?:MAX|MIN)\s*\(\s*[a-zA-Z_][a-zA-Z0-9_.`]*date[a-zA-Z0-9_.`]*\s*\)",
            s,
            flags=re.I | re.S,
        )
    )


def _rewrite_scalar_extrema_date_filter(sql: str, question: str) -> str:
    if not sql or not _question_requires_scalar_extrema_date_filter(question):
        return sql
    if _has_scalar_extrema_date_filter_shape(sql):
        return sql

    s = (sql or "").strip().rstrip(";")
    if not s:
        return sql

    if re.search(r"\b(?:JOIN|GROUP\s+BY|HAVING|UNION|INTERSECT|EXCEPT)\b", s, flags=re.I):
        return sql

    m = re.match(
        r"(?is)^\s*SELECT\s+(?P<select>.+?)\s+FROM\s+(?P<table>`?[A-Za-z_][A-Za-z0-9_]*`?)"
        r"(?:\s+(?:AS\s+)?(?P<alias>[A-Za-z_][A-Za-z0-9_]*))?"
        r"(?:\s+WHERE\s+(?P<where>.*?))?"
        r"(?:\s+ORDER\s+BY\s+(?P<order>.*?))?\s*$",
        s,
    )
    if not m:
        return sql

    select_clause = (m.group("select") or "").strip()
    table = (m.group("table") or "").strip()
    alias = (m.group("alias") or "").strip()
    where_clause = (m.group("where") or "").strip()
    existing_order = (m.group("order") or "").strip()
    if not select_clause or not table:
        return sql

    agg_func = "MAX"
    earliest_terms = rule_list(("temporal_intent", "earliest_keywords"), ["earliest", "oldest"])
    if _has_any_phrase(question or "", earliest_terms):
        agg_func = "MIN"

    date_candidates = re.findall(
        r"\b([A-Za-z_][A-Za-z0-9_]*?(?:Date|Time|Timestamp|date|time|timestamp))\b",
        s,
        flags=re.I,
    )
    if not date_candidates:
        return sql
    date_col = date_candidates[0]

    outer_date_ref = f"{alias}.{date_col}" if alias else date_col
    inner_date_ref = date_col
    extrema_filter = f"{outer_date_ref} = (SELECT {agg_func}({inner_date_ref}) FROM {table})"
    final_where = extrema_filter
    if where_clause:
        final_where = f"{where_clause} AND {extrema_filter}"

    requested_order = ""
    sort_m = re.search(
        r"\b(?:sort(?:ed)?\s+by|order(?:ed)?\s+by)\s+(.+?)(?:[.;]|$)",
        question or "",
        flags=re.I,
    )
    if sort_m:
        sort_clause = sort_m.group(1).strip()
        select_identifiers = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", select_clause)
        unique_identifiers = []
        for ident in select_identifiers:
            if ident.lower() in {"select", "from", "as"}:
                continue
            if ident not in unique_identifiers:
                unique_identifiers.append(ident)

        def _to_words(text: str) -> str:
            return re.sub(r"\s+", " ", re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text.replace("_", " "))).strip().lower()

        for ident in unique_identifiers:
            words = _to_words(ident)
            if words and words in sort_clause.lower():
                direction = "DESC" if _has_any_phrase(sort_clause, ordering_direction_desc_terms()) else "ASC"
                requested_order = f"{ident} {direction}"
                break

    final_order = requested_order or existing_order

    out = f"SELECT {select_clause} FROM {table}"
    if alias:
        out += f" AS {alias}"
    out += f" WHERE {final_where}"
    if final_order:
        out += f" ORDER BY {final_order}"
    return out + ";"


def _has_anti_existence_shape(sql: str) -> bool:
    s = (sql or "").strip()
    if not s:
        return False
    qcol = r"(?:`?[A-Za-z_][A-Za-z0-9_]*`?\s*\.\s*)?`?[A-Za-z_][A-Za-z0-9_]*`?"
    if re.search(r"\bNOT\s+EXISTS\s*\(", s, flags=re.I):
        return True
    if re.search(rf"\b{qcol}\s+NOT\s+IN\s*\(", s, flags=re.I):
        return True
    if re.search(rf"\bNOT\s+{qcol}\s+IN\s*\(", s, flags=re.I):
        return True
    if re.search(r"\bNOT\s+IN\s*\(", s, flags=re.I):
        return True
    if re.search(r"\bLEFT\s+JOIN\b", s, flags=re.I) and re.search(r"\bIS\s+NULL\b", s, flags=re.I):
        return True
    if re.search(r"\bHAVING\b[^;]*\bCOUNT\s*\([^)]*\)\s*=\s*0\b", s, flags=re.I | re.S):
        return True
    return False


def _has_include_zero_shape(sql: str) -> bool:
    s = (sql or "").strip()
    if not s:
        return False
    if has_join(s):
        return bool(re.search(r"\b(?:LEFT|RIGHT|FULL(?:\s+OUTER)?)\s+JOIN\b", s, flags=re.I))
    # Keep this conservative: if the query uses no JOIN, do not force a violation.
    # Some valid include-zero patterns can be expressed with scalar subqueries.
    return True


def _find_top_level_kw(sql: str, keyword: str, start: int = 0) -> int:
    if not sql:
        return -1
    s = str(sql)
    low = s.lower()
    kw = keyword.lower()
    depth = 0
    in_sq = False
    in_dq = False
    i = max(0, start)
    while i <= len(low) - len(kw):
        ch = low[i]
        if ch == "'" and not in_dq:
            in_sq = not in_sq
            i += 1
            continue
        if ch == '"' and not in_sq:
            in_dq = not in_dq
            i += 1
            continue
        if not in_sq and not in_dq:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            if depth == 0 and low.startswith(kw, i):
                before_ok = i == 0 or not low[i - 1].isalnum()
                after_ok = i + len(kw) >= len(low) or not low[i + len(kw)].isalnum()
                if before_ok and after_ok:
                    return i
        i += 1
    return -1


def _ensure_group_by_for_non_agg(sql: str) -> str:
    s = (sql or "").strip()
    if not s:
        return s
    trailing_semicolon = ";" if s.endswith(";") else ""
    if trailing_semicolon:
        s = s[:-1].rstrip()
    if _find_top_level_kw(s, "group by") >= 0:
        return s + trailing_semicolon

    select_match = re.search(r"^\s*select\s+(.*?)\s+from\s", s, flags=re.I | re.S)
    if not select_match:
        return s + trailing_semicolon
    select_items = _split_top_level_csv(select_match.group(1))
    if not select_items:
        return s + trailing_semicolon

    def _is_top_level_window_expr(expr: str) -> bool:
        return bool(re.search(r"\bOVER\s*\(", str(expr or ""), flags=re.I))

    def _has_top_level_plain_agg(expr: str) -> bool:
        expr_l = str(expr or "").lower()
        if not expr_l:
            return False
        if "select" in expr_l:
            return False
        if _is_top_level_window_expr(expr):
            return False
        return bool(re.search(r"\b(count|sum|avg|min|max)\s*\(", expr_l))

    if not any(_has_top_level_plain_agg(item) for item in select_items):
        return s + trailing_semicolon
    if any(_is_top_level_window_expr(item) for item in select_items):
        return s + trailing_semicolon

    def _strip_alias(expr: str) -> str:
        raw = re.sub(r"\s+AS\s+`?[A-Za-z_][A-Za-z0-9_]*`?\s*$", "", expr, flags=re.I)
        raw = re.sub(r"\s+`?[A-Za-z_][A-Za-z0-9_]*`?\s*$", "", raw)
        return raw.strip()

    group_items: List[str] = []
    for item in select_items:
        if not item:
            continue
        if _has_top_level_plain_agg(item) or _is_top_level_window_expr(item):
            continue
        if re.search(r"^\s*['\"].*['\"]\s*$", item.strip()):
            continue
        if re.search(r"^\s*\d+(\.\d+)?\s*$", item.strip()):
            continue
        if re.search(r"\*\s*$", item.strip()):
            continue
        expr = _strip_alias(item)
        if expr:
            group_items.append(expr)

    if not group_items:
        return s + trailing_semicolon

    order_pos = _find_top_level_kw(s, "order by")
    having_pos = _find_top_level_kw(s, "having")
    limit_pos = _find_top_level_kw(s, "limit")
    tail_candidates = [p for p in [having_pos, order_pos, limit_pos] if p >= 0]
    insert_pos = min(tail_candidates) if tail_candidates else len(s)
    group_by_clause = " GROUP BY " + ", ".join(group_items) + " "
    out = s[:insert_pos].rstrip() + group_by_clause + s[insert_pos:].lstrip()
    return out.rstrip() + trailing_semicolon


def _repair_chronological_month_order(sql: str, question: str) -> str:
    s = (sql or "").strip()
    q = str(question or "").lower()
    if not s or "month" not in q:
        return s
    trailing_semicolon = ";" if s.endswith(";") else ""
    if trailing_semicolon:
        s = s[:-1].rstrip()

    group_pos = _find_top_level_kw(s, "group by")
    if group_pos < 0:
        return s + trailing_semicolon

    order_pos = _find_top_level_kw(s, "order by", group_pos + len("group by"))
    having_pos = _find_top_level_kw(s, "having", group_pos + len("group by"))
    limit_pos = _find_top_level_kw(s, "limit", group_pos + len("group by"))
    group_end_candidates = [p for p in [having_pos, order_pos, limit_pos] if p >= 0]
    group_end = min(group_end_candidates) if group_end_candidates else len(s)
    group_clause = s[group_pos + len("group by"):group_end].strip().rstrip(";")
    group_items = _split_top_level_csv(group_clause)
    if not group_items:
        return s + trailing_semicolon

    def _normalize_expr(expr: str) -> str:
        return re.sub(r"\s+", "", str(expr or "").strip()).lower()

    def _extract_month_label_source(expr: str) -> str:
        item_s = str(expr or "").strip()
        date_format_match = re.match(
            r"""(?is)^DATE_FORMAT\s*\(\s*(?P<date>.+?)\s*,\s*(?P<quote>['"])(?P<fmt>%[A-Za-z])(?P=quote)\s*\)$""",
            item_s,
        )
        if date_format_match and str(date_format_match.group("fmt") or "") == "%M":
            return str(date_format_match.group("date") or "").strip()
        monthname_match = re.match(r"""(?is)^MONTHNAME\s*\(\s*(?P<date>.+?)\s*\)$""", item_s)
        if monthname_match:
            return str(monthname_match.group("date") or "").strip()
        return ""

    def _extract_date_format_args(expr: str) -> Tuple[str, str]:
        match = re.match(r"""(?is)^DATE_FORMAT\s*\(\s*(?P<args>.*)\s*\)$""", str(expr or "").strip())
        if not match:
            return "", ""
        args = _split_top_level_csv(match.group("args"))
        if len(args) != 2:
            return "", ""
        fmt = args[1].strip()
        quote_match = re.match(r"""(?s)^(?P<quote>['"])(?P<fmt>.*)(?P=quote)$""", fmt)
        if not quote_match:
            return "", ""
        return args[0].strip(), str(quote_match.group("fmt") or "")

    def _extract_month_numeric_source(expr: str) -> str:
        item_s = str(expr or "").strip()
        month_match = re.match(r"""(?is)^MONTH\s*\(\s*(?P<date>.+?)\s*\)$""", item_s)
        if month_match:
            month_arg = str(month_match.group("date") or "").strip()
            date_format_source, date_format_fmt = _extract_date_format_args(month_arg)
            if date_format_source and re.search(r"%(?:m|c)\b", date_format_fmt):
                return date_format_source
            return month_arg
        date_format_match = re.match(
            r"""(?is)^DATE_FORMAT\s*\(\s*(?P<date>.+?)\s*,\s*(?P<quote>['"])(?P<fmt>%[A-Za-z])(?P=quote)\s*\)$""",
            item_s,
        )
        if date_format_match and str(date_format_match.group("fmt") or "") == "%m":
            return str(date_format_match.group("date") or "").strip()
        return ""

    def _is_month_label_expr(expr: str) -> bool:
        item_s = str(expr or "").strip()
        if re.search(r"\bMONTHNAME\s*\(", item_s, flags=re.I):
            return True
        date_format_match = re.match(
            r"""(?is)^DATE_FORMAT\s*\(\s*(?P<date>.+?)\s*,\s*(?P<quote>['"])(?P<fmt>%[A-Za-z])(?P=quote)\s*\)$""",
            item_s,
        )
        return bool(date_format_match and str(date_format_match.group("fmt") or "") == "%M")

    month_sort_expr = ""
    month_source_expr = ""
    for item in group_items:
        item_s = str(item or "").strip()
        numeric_source = _extract_month_numeric_source(item_s)
        if numeric_source:
            month_sort_expr = item_s
            month_source_expr = numeric_source
            break
        label_source = _extract_month_label_source(item_s)
        if label_source and not month_source_expr:
            month_source_expr = label_source

    added_month_sort_to_group = False
    if not month_sort_expr and month_source_expr:
        month_sort_expr = f"MONTH({month_source_expr})"
        if all(_normalize_expr(item) != _normalize_expr(month_sort_expr) for item in group_items):
            group_items.append(month_sort_expr)
            rebuilt_group = ", ".join(item.strip() for item in group_items if item.strip())
            suffix = s[group_end:]
            if suffix and not suffix[:1].isspace():
                suffix = " " + suffix
            s = (
                s[:group_pos + len("group by")]
                + " "
                + rebuilt_group
                + suffix
            )
            added_month_sort_to_group = True

    if not month_sort_expr:
        return s + trailing_semicolon

    if added_month_sort_to_group:
        group_pos = _find_top_level_kw(s, "group by")
        order_pos = _find_top_level_kw(s, "order by", group_pos + len("group by"))
        having_pos = _find_top_level_kw(s, "having", group_pos + len("group by"))
        limit_pos = _find_top_level_kw(s, "limit", group_pos + len("group by"))

    direction = ""
    if _has_any_phrase(q, ordering_temporal_desc_terms()):
        direction = "DESC"
    elif _has_any_phrase(q, ordering_temporal_asc_terms()):
        direction = "ASC"

    select_match = re.search(r"^\s*select\s+(.*?)\s+from\s", s, flags=re.I | re.S)
    month_label_aliases: Set[str] = set()
    if select_match:
        for item in _split_top_level_csv(select_match.group(1)):
            if not re.search(r"DATE_FORMAT\s*\([^)]*%M", item, flags=re.I):
                continue
            alias_match = re.search(r"\bAS\s+`?([A-Za-z_][A-Za-z0-9_]*)`?\s*$", item, flags=re.I)
            if alias_match:
                month_label_aliases.add(alias_match.group(1).strip().lower())

    if order_pos < 0:
        if not direction:
            return s + trailing_semicolon
        if not _question_requests_ordering(question):
            return s + trailing_semicolon
        insert_pos = limit_pos if limit_pos >= 0 else len(s)
        order_clause = f" ORDER BY {month_sort_expr} {direction} "
        out = s[:insert_pos].rstrip() + order_clause + s[insert_pos:].lstrip()
        return out.rstrip() + trailing_semicolon

    limit_after_order = _find_top_level_kw(s, "limit", order_pos + len("order by"))
    order_end_candidates = [p for p in [limit_after_order] if p >= 0]
    order_end = min(order_end_candidates) if order_end_candidates else len(s)
    order_clause = s[order_pos + len("order by"):order_end].strip().rstrip(";")
    order_items = _split_top_level_csv(order_clause)
    if not order_items:
        return s + trailing_semicolon

    first_item = order_items[0].strip()
    if not direction:
        existing_direction = re.search(r"\s+\b(ASC|DESC)\b\s*$", first_item, flags=re.I)
        if existing_direction:
            direction = existing_direction.group(1).upper()
    if not direction:
        direction = "ASC"
    first_base = re.sub(r"\s+\b(?:ASC|DESC)\b\s*$", "", first_item, flags=re.I).strip()
    first_base_l = first_base.lower()
    first_numeric_source = _extract_month_numeric_source(first_base)
    first_is_month_label = (
        first_base_l in month_label_aliases
        or _is_month_label_expr(first_base)
        or _normalize_expr(first_base) == _normalize_expr(month_sort_expr)
        or (
            bool(first_numeric_source)
            and bool(month_source_expr)
            and _normalize_expr(first_numeric_source) == _normalize_expr(month_source_expr)
        )
    )
    if not first_is_month_label:
        return s + trailing_semicolon

    order_items[0] = f"{month_sort_expr} {direction}"
    rebuilt_order = ", ".join(item.strip() for item in order_items if item.strip())
    out = (
        s[:order_pos + len("order by")]
        + " "
        + rebuilt_order
        + s[order_end:]
    )
    return out.rstrip().rstrip(";") + trailing_semicolon


def _add_missing_final_order(sql: str, question: str) -> str:
    s = (sql or "").strip()
    if not s or _has_top_level_order_by(s):
        return s
    if not _question_requests_ordering(question):
        return s
    trailing_semicolon = ";" if s.endswith(";") else ""
    if trailing_semicolon:
        s = s[:-1].rstrip()

    select_match = re.search(r"^\s*select\s+(.*?)\s+from\s", s, flags=re.I | re.S)
    if not select_match:
        return s + trailing_semicolon
    select_items = _split_top_level_csv(select_match.group(1))
    if not select_items:
        return s + trailing_semicolon

    q = (question or "").lower()
    direction = "DESC" if _has_any_phrase(q, ordering_direction_desc_terms()) else "ASC"
    temporal_order_intent = _has_any_phrase(q, ordering_temporal_intent_terms())

    reserved_tokens = _SQL_KEYWORDS.union(set(AGENT_SQL_RESERVED_EXTRA_TOKENS))

    def _quote_order_identifier_if_reserved(expr: str) -> str:
        token = str(expr or "").strip()
        if not token:
            return token
        if token.startswith("`") and token.endswith("`"):
            return token
        # Keep non-identifier expressions intact (qualified refs/functions/etc.).
        if "." in token or re.search(r"[()\s]", token):
            return token
        bare = token.strip("`")
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", bare):
            return token
        if bare.lower() in reserved_tokens:
            return f"`{bare}`"
        return bare

    def _extract_alias(expr: str) -> str:
        m = re.search(r"\bAS\s+(`?[A-Za-z_][A-Za-z0-9_]*`?)\s*$", expr, flags=re.I)
        if m:
            return _quote_order_identifier_if_reserved(m.group(1))
        m = re.search(r"\s+(`?[A-Za-z_][A-Za-z0-9_]*`?)\s*$", expr.strip())
        return _quote_order_identifier_if_reserved(m.group(1)) if m else ""

    def _looks_temporal(expr: str, alias: str = "") -> bool:
        text = " ".join(x for x in [expr, alias] if x)
        normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text.replace("_", " "))
        if re.search(r"\b(?:date|time|timestamp|year|month|day)\b", normalized, flags=re.I):
            return True
        return bool(re.search(r"\b(?:date|time|timestamp|year|month|day|extract|date_format)\s*\(", expr, flags=re.I))

    order_expr = ""
    fallback_candidates: List[str] = []
    temporal_candidates: List[str] = []
    for item in select_items:
        item_l = item.lower()
        if re.search(r"\b(count|sum|avg|min|max)\s*\(", item_l):
            continue
        alias = _extract_alias(item)
        candidate = alias or item.strip()
        if not candidate:
            continue
        fallback_candidates.append(candidate)
        if temporal_order_intent and _looks_temporal(item, alias):
            temporal_candidates.append(candidate)

    if temporal_candidates:
        order_expr = temporal_candidates[0]
    elif fallback_candidates:
        order_expr = fallback_candidates[0]

    if not order_expr:
        first = select_items[0].strip()
        order_expr = _extract_alias(first) or first

    if not order_expr:
        return s + trailing_semicolon

    limit_pos = _find_top_level_kw(s, "limit")
    insert_pos = limit_pos if limit_pos >= 0 else len(s)
    order_clause = f" ORDER BY {_quote_order_identifier_if_reserved(order_expr)} {direction} "
    out = s[:insert_pos].rstrip() + order_clause + s[insert_pos:].lstrip()
    return out.rstrip() + trailing_semicolon


def _identifier_similarity_distance(a: str, b: str, max_distance: int = 2) -> int:
    """Small bounded edit-distance helper for conservative alias typo repair."""
    a = str(a or "").lower()
    b = str(b or "").lower()
    if a == b:
        return 0
    if abs(len(a) - len(b)) > max_distance:
        return max_distance + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        best = cur[0]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            val = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            cur.append(val)
            best = min(best, val)
        if best > max_distance:
            return max_distance + 1
        prev = cur
    return prev[-1]


def _repair_near_miss_output_alias_references(sql: str, ctx: str = "") -> str:
    """
    Repair bare references that are near-miss typos of SELECT output aliases.

    Guardrails:
    - only touches WHERE/HAVING/ORDER BY clauses
    - never rewrites real schema columns from the active context
    - requires a unique close alias candidate
    """
    s = (sql or "").strip()
    if not s:
        return sql

    alias_names = {
        str(m.group(1) or "").strip("`").lower()
        for m in re.finditer(
            r"\bAS\s+`?([A-Za-z_][A-Za-z0-9_]*)`?",
            s,
            flags=re.I,
        )
        if str(m.group(1) or "").strip()
    }
    relation_aliases = {
        str(m.group(1) or "").strip("`").lower()
        for m in re.finditer(
            r"\b(?:FROM|JOIN)\s+(?:\([^)]*\)|`?[A-Za-z_][A-Za-z0-9_]*`?)\s+(?:AS\s+)?`?([A-Za-z_][A-Za-z0-9_]*)`?",
            s,
            flags=re.I | re.S,
        )
        if str(m.group(1) or "").strip()
    }
    alias_names.difference_update(relation_aliases)
    if not alias_names:
        return sql

    schema_cols: Set[str] = set()
    try:
        for cols in (_parse_allowed_identifiers(ctx or "") or {}).values():
            schema_cols.update(str(c or "").strip().lower() for c in cols if str(c or "").strip())
    except Exception:
        schema_cols = set()

    clause_positions = [
        pos for pos in (
            _find_top_level_kw(s, "where"),
            _find_top_level_kw(s, "having"),
            _find_top_level_kw(s, "order by"),
        )
        if pos >= 0
    ]
    if not clause_positions:
        return sql
    start = min(clause_positions)
    head, tail = s[:start], s[start:]

    protected = {a.lower() for a in alias_names}
    protected.update(c.lower() for c in schema_cols)
    protected.update(str(k or "").lower() for k in _SQL_KEYWORDS)

    def _replace_token(match: re.Match) -> str:
        token = str(match.group(0) or "")
        token_l = token.lower()
        if token_l in protected or len(token_l) < 4:
            return token
        candidates = [
            alias for alias in alias_names
            if alias != token_l and _identifier_similarity_distance(token_l, alias, 2) <= 2
        ]
        if len(candidates) != 1:
            return token
        return candidates[0]

    repaired_tail = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\b", _replace_token, tail)
    return head + repaired_tail


def _rewrite_leading_character_grouping(sql: str, question: str, ctx: str = "") -> str:
    """
    Repair explicit "group by first N characters of <column>" questions.

    This is a schema-driven semantic rewrite for single-table grouped summaries:
    - resolve the referenced text/code column from runtime schema metadata
    - normalize the grouping expression to LEFT(column, N)
    - drop redundant/invalid LENGTH(alias)=N filters introduced by drafts
    """
    s = (sql or "").strip()
    q = str(question or "").strip()
    if not s or not q:
        return sql

    m = re.search(
        r"\b(?:group(?:ed|ing)?(?:\s+\w+){0,4}\s+by|by)\s+"
        r"(?:the\s+)?(?:first|leading|leftmost)\s+"
        r"(?P<count>\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
        r"(?:characters?|chars?|letters?|digits?)\s+of\s+"
        r"(?P<target>[a-z][a-z0-9_ ]{1,48}?)"
        r"(?=(?:[.,;]|\b(?:return|show|list|display|give|keep|only|order|sort|having|where|with)\b|$))",
        q,
        flags=re.I,
    )
    if not m:
        return sql

    count_token = str(m.group("count") or "").strip().lower()
    target_phrase = str(m.group("target") or "").strip()
    if not count_token or not target_phrase:
        return sql

    word_to_int = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
        "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    }
    try:
        n_chars = int(count_token) if count_token.isdigit() else word_to_int.get(count_token)
    except Exception:
        n_chars = None
    if n_chars is None or n_chars <= 0:
        return sql

    try:
        import sqlglot
        from sqlglot import exp
    except Exception:
        return sql

    try:
        ast = sqlglot.parse_one(s, read="mysql")
    except Exception:
        return sql

    sel = ast.find(exp.Select)
    if sel is None or list(ast.find_all(exp.Join)):
        return sql

    tables = []
    relation_alias = ""
    for src in ast.find_all(exp.Table):
        table_name = str(src.name or "").strip("`")
        if not table_name:
            continue
        tables.append(table_name)
        if not relation_alias:
            alias_or_name = str(src.alias_or_name or "").strip("`")
            relation_alias = alias_or_name if alias_or_name and alias_or_name.lower() != table_name.lower() else ""
    table_names = {t.lower() for t in tables if t}
    if len(table_names) != 1:
        return sql
    table_name = next(iter(table_names))

    allowed_map = _parse_allowed_identifiers(ctx or "") or {}
    allowed_cols = list(allowed_map.get(table_name) or [])
    if not allowed_cols:
        try:
            allowed_cols = list(get_schema_analyzer().get_table_columns(table_name) or [])
        except Exception:
            allowed_cols = []
    if not allowed_cols:
        return sql

    def _tokens(text: str) -> Set[str]:
        raw = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(text or ""))
        toks = {
            tok for tok in re.findall(r"[a-z0-9]+", raw.lower())
            if tok and tok not in {
                "the", "a", "an", "of", "for", "to", "by", "in", "on", "at",
                "with", "without", "and", "or", "first", "leading", "leftmost",
                "character", "characters", "char", "chars", "letter", "letters",
                "digit", "digits",
            }
        }
        return toks

    analyzer = get_schema_analyzer()
    target_tokens = _tokens(target_phrase)
    if not target_tokens:
        return sql

    best_col = ""
    best_score = 0.0
    for col in allowed_cols:
        col_name = str(col or "").strip()
        if not col_name:
            continue
        semantic_texts = {col_name, col_name.replace("_", " ")}
        try:
            for syn in analyzer.get_column_synonyms(table_name, col_name):
                syn_s = str(syn or "").strip()
                if syn_s:
                    semantic_texts.add(syn_s)
            desc = str(get_column_description(table_name, col_name) or "").strip()
            if desc:
                semantic_texts.add(desc)
        except Exception:
            pass
        col_tokens: Set[str] = set()
        for text in semantic_texts:
            col_tokens.update(_tokens(text))
        if not col_tokens:
            continue
        overlap = target_tokens.intersection(col_tokens)
        if not overlap:
            continue
        score = float(len(overlap))
        if target_tokens.issubset(col_tokens):
            score += 2.0
        role = str(get_column_role(table_name, col_name) or "").strip().lower()
        if role in {"name", "text", "code"}:
            score += 0.5
        if role in {"metric", "date"}:
            score -= 0.5
        if score > best_score:
            best_col = col_name
            best_score = score

    if not best_col:
        return sql

    grouped_items: List[Tuple[int, Any, Optional[str]]] = []
    for idx, item in enumerate(sel.expressions):
        base = item.this if isinstance(item, exp.Alias) else item
        if base.find(exp.AggFunc) or base.find(exp.Window) or isinstance(base, (exp.Literal, exp.Star)):
            continue
        alias_name = item.alias if isinstance(item, exp.Alias) else None
        grouped_items.append((idx, base, alias_name))
    if len(grouped_items) != 1:
        return sql

    idx, old_dim_expr, old_alias = grouped_items[0]
    qualified_col = f"{relation_alias}.{best_col}" if relation_alias else best_col
    try:
        replacement_expr = sqlglot.parse_one(
            f"SELECT LEFT({qualified_col}, {int(n_chars)})",
            read="mysql",
        ).expressions[0]
    except Exception:
        return sql

    current_dim_sql = old_dim_expr.sql(dialect="mysql")
    replacement_sql = replacement_expr.sql(dialect="mysql")
    if re.sub(r"[\s`]+", "", current_dim_sql).lower() == re.sub(r"[\s`]+", "", replacement_sql).lower():
        return sql

    new_item = exp.alias_(replacement_expr.copy(), old_alias, quoted=False) if old_alias else replacement_expr.copy()
    new_select_exprs = list(sel.expressions)
    new_select_exprs[idx] = new_item
    sel.set("expressions", new_select_exprs)

    grp = sel.args.get("group")
    if grp is None or not getattr(grp, "expressions", None):
        return sql
    grp.set("expressions", [replacement_expr.copy()])

    def _norm_expr(expr_in: Any) -> str:
        if expr_in is None:
            return ""
        try:
            text = expr_in.sql(dialect="mysql")
        except Exception:
            text = str(expr_in or "")
        return re.sub(r"[\s`]+", "", text).lower()

    def _flatten_and(expr_in: Any) -> List[Any]:
        if expr_in is None:
            return []
        if isinstance(expr_in, exp.And):
            return _flatten_and(expr_in.this) + _flatten_and(expr_in.expression)
        return [expr_in]

    def _build_and(exprs: List[Any]) -> Any:
        cleaned = [e for e in exprs if e is not None]
        if not cleaned:
            return None
        out = cleaned[0]
        for nxt in cleaned[1:]:
            out = exp.And(this=out, expression=nxt)
        return out

    def _is_matching_length_filter(cond: Any) -> bool:
        if not isinstance(cond, exp.EQ):
            return False
        left = cond.this
        right = cond.expression
        literal = None
        fn_side = None
        for a, b in ((left, right), (right, left)):
            if isinstance(b, exp.Literal) and not getattr(b, "is_string", False):
                try:
                    literal = int(str(b.this))
                except Exception:
                    literal = None
                fn_side = a
                break
        if literal != int(n_chars) or fn_side is None:
            return False
        if not isinstance(fn_side, (exp.Length, exp.Anonymous)):
            return False
        fn_name = ""
        if isinstance(fn_side, exp.Length):
            fn_name = "length"
        else:
            fn_name = str(getattr(fn_side, "name", "") or getattr(fn_side, "this", "") or "").lower()
        if fn_name not in {"length", "char_length", "character_length"}:
            return False
        inner = getattr(fn_side, "this", None)
        if inner is None and getattr(fn_side, "expressions", None):
            inner = fn_side.expressions[0]
        inner_norm = _norm_expr(inner)
        if old_alias and isinstance(inner, exp.Column) and not inner.table:
            if str(inner.name or "").strip("`").lower() == str(old_alias).strip("`").lower():
                return True
        return inner_norm in {_norm_expr(old_dim_expr), _norm_expr(replacement_expr)}

    where_arg = sel.args.get("where")
    if isinstance(where_arg, exp.Where):
        conds = _flatten_and(where_arg.this)
        kept = [c.copy() for c in conds if not _is_matching_length_filter(c)]
        if len(kept) != len(conds):
            rebuilt_where = _build_and(kept)
            sel.set("where", exp.Where(this=rebuilt_where) if rebuilt_where is not None else None)

    out = ast.sql(dialect="mysql")
    return out if out else sql


def _normalize_order_term_tokens(text: str) -> Set[str]:
    raw = re.sub(AGENT_COLUMN_CAMEL_SPLIT_PATTERN, r"\1 \2", str(text or ""))
    tokens = re.findall(r"[a-z0-9]+", raw.lower())
    stop = order_term_token_stopwords()
    normalized = []
    for tok in tokens:
        if len(tok) == 1:
            continue
        if tok in stop:
            continue
        if tok.endswith("ies") and len(tok) > 4:
            tok = tok[:-3] + "y"
        elif tok.endswith("s") and len(tok) > 3:
            tok = tok[:-1]
        normalized.append(tok)
    return set(normalized)


def _requested_order_phrase_token_variants(question: str, phrase: str) -> List[Set[str]]:
    base_tokens = _normalize_order_term_tokens(phrase)
    if not base_tokens:
        return []

    variants: List[Set[str]] = [base_tokens]
    semantic_stop_tokens = {
        str(tok or "").strip().lower()
        for tok in semantic_phrase_token_stopwords()
        if len(str(tok or "").strip()) > 1
    }

    def _append_variant(tokens: Set[str]) -> None:
        if not tokens:
            return
        if any(tokens == existing for existing in variants):
            return
        variants.append(tokens)

    semantic_reduced = {tok for tok in base_tokens if tok not in semantic_stop_tokens}
    _append_variant(semantic_reduced)

    if not _question_wants_latest_year_scope_with_aggregate(question):
        return variants

    temporal_scope_tokens: Set[str] = set()
    for term in ordering_temporal_intent_terms() + temporal_year_column_terms():
        temporal_scope_tokens.update(_normalize_order_term_tokens(term))

    for token_set in list(variants):
        reduced_tokens = {tok for tok in token_set if tok not in temporal_scope_tokens}
        _append_variant(reduced_tokens)
    return variants


def _normalize_order_reference(expr: str) -> str:
    text = str(expr or "").strip()
    if not text:
        return ""
    text = text.replace("`", "")
    text = re.sub(r"\s+", "", text)
    return text.lower()


def _quote_order_identifier_if_reserved(expr: str) -> str:
    token = str(expr or "").strip()
    if not token:
        return token
    if token.startswith("`") and token.endswith("`"):
        return token
    if "." in token or re.search(r"[()\s]", token):
        return token
    bare = token.strip("`")
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", bare):
        return token
    reserved_tokens = _SQL_KEYWORDS.union(set(AGENT_SQL_RESERVED_EXTRA_TOKENS))
    if bare.lower() in reserved_tokens:
        return f"`{bare}`"
    return bare


def _extract_requested_final_order_terms(question: str) -> List[Tuple[str, str]]:
    q = str(question or "").strip()
    if not q:
        return []
    patterns = [
        r"\b(?:final\s+)?(?:result|results|output|rows?)\s+(?:should\s+be\s+)?(?:sorted|ordered|arranged)\s+by\s+(?P<body>[^.]+)",
        r"\b(?:sort|sorted|order|ordered|arrange|arranged)\s+(?:the\s+)?(?:final\s+)?(?:result|results|output|rows?)?\s*by\s+(?P<body>[^.]+)",
        r"\b(?:sort|sorted|order|ordered|arrange|arranged)\s+(?:so|such\s+that)\s+(?P<body>[^.]+)",
        r"\b(?:sort|order|arrange)\s+by\s+(?P<body>[^.]+)",
    ]
    # Prefer the latest explicit ordering command in the question text.
    # This avoids choosing earlier window-sequence phrases such as
    # "... ordered by payment date. Sort by customer number ...".
    matches: List[Tuple[int, int, str]] = []
    for idx, pat in enumerate(patterns):
        # Higher priority for stronger final-output phrasing patterns.
        priority = 3 - idx
        for m in re.finditer(pat, q, flags=re.I):
            body_txt = str(m.group("body") or "").strip()
            if not body_txt:
                continue
            matches.append((priority, m.start(), body_txt))
    if not matches:
        return []

    def _has_tiebreaker_body(text: str) -> bool:
        return bool(
            re.search(
                r"\b(?:if\s+there\s+is\s+a\s+tie|if\s+there\s+are\s+ties|tie(?:breaker)?|then)\b",
                str(text or ""),
                flags=re.I,
            )
        )

    # Prefer explicit tie-breaker phrasing when present; otherwise keep the
    # prior behavior of using the latest explicit ordering command.
    matches.sort(key=lambda x: (_has_tiebreaker_body(x[2]), x[1], x[0], len(x[2])), reverse=True)
    body = matches[0][2]
    if not body:
        return []

    # Preserve tie-breaker ordering phrases as additional ORDER BY keys.
    # Example: "... appear first; if there is a tie, sort by name A-Z"
    # should map to two keys, not truncate after the tie marker.
    body = re.sub(
        r"\b(?:if\s+there\s+is\s+a\s+tie|if\s+there\s+are\s+ties)\b\s*,?\s*",
        " then ",
        body,
        flags=re.I,
    ).strip()
    body = body.replace(";", ", ")
    raw_parts = [
        p.strip(" ,;")
        for p in re.split(r"\s*[;,]?\s+then\s+|\s*[;,]\s*", body, flags=re.I)
        if p.strip(" ,;")
    ]
    desc_terms = ordering_direction_desc_terms()
    asc_terms = ordering_direction_asc_terms()
    direction_noise_terms = rule_list(
        ("heuristic_repairs", "final_order", "direction_noise_terms"),
        ["first"],
    )
    desc_alt = _phrase_alt_pattern(desc_terms)
    asc_alt = _phrase_alt_pattern(asc_terms)
    out: List[Tuple[str, str]] = []
    for part in raw_parts:
        direction = "ASC"
        if _has_any_phrase(part, desc_terms):
            direction = "DESC"
        elif re.search(r"\bmore\b.*\bfirst\b", part, flags=re.I):
            direction = "DESC"
        elif _has_any_phrase(part, asc_terms):
            direction = "ASC"
        elif re.search(r"\b(?:less|fewer|lower|smaller)\b.*\bfirst\b", part, flags=re.I):
            direction = "ASC"
        clean = part
        if desc_alt and asc_alt:
            clean = re.sub(
                rf"\bfrom\s+(?:{desc_alt}|{asc_alt})\s+to\s+(?:{desc_alt}|{asc_alt})\b",
                " ",
                clean,
                flags=re.I,
            )
        for phrase in desc_terms + asc_terms + direction_noise_terms:
            phrase_clean = str(phrase or "").strip()
            if not phrase_clean:
                continue
            pat = re.escape(phrase_clean).replace(r"\ ", r"\s+")
            clean = re.sub(rf"\b{pat}\b", " ", clean, flags=re.I)
        clean = re.sub(
            r"^\s*(?:sort|sorted|order|ordered|arrange|arranged)\s+by\s+",
            "",
            clean,
            flags=re.I,
        )
        clean = re.sub(r"^\s*(?:so|that)\s+", "", clean, flags=re.I)
        clean = re.sub(r"\s+", " ", clean).strip(" ,;")
        if clean:
            out.append((clean, direction))
    return out


def _build_requested_final_order_mapping(
    sql: str,
    question: str,
) -> Optional[List[Dict[str, Any]]]:
    requested = _extract_requested_final_order_terms(question)
    if not requested:
        return []

    s = (sql or "").strip()
    if not s:
        return None
    if not re.match(r"^\s*SELECT\b", s, flags=re.I):
        return None

    from_pos = _find_top_level_kw(s, "from")
    if from_pos < 0:
        return None

    select_segment = re.sub(r"^\s*select\s+", "", s[:from_pos], flags=re.I | re.S).strip()
    select_items = _split_top_level_csv(select_segment)
    if not select_items:
        return None

    candidates: List[Dict[str, Any]] = []
    for item in select_items:
        item_clean = re.sub(r"(?is)^(?:DISTINCT|ALL)\s+", "", item.strip()).strip()
        if not item_clean:
            continue

        alias_match = re.search(
            r"""\bAS\s+(`[^`]+`|'[^']+'|"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)\s*$""",
            item_clean,
            flags=re.I,
        )
        alias_sql = alias_match.group(1).strip() if alias_match else ""
        alias = alias_sql.strip("`")
        expr = item_clean[:alias_match.start()].strip() if alias_match else item_clean
        simple_col_match = re.search(r"(?:^|\.)(`?[A-Za-z_][A-Za-z0-9_]*`?)\s*$", expr)
        expr_name = simple_col_match.group(1).strip("`") if simple_col_match else expr
        preferred_expr = _quote_order_identifier_if_reserved(alias_sql or expr_name)
        token_text = " ".join(x for x in [alias, expr_name, expr] if x)
        tokens = _normalize_order_term_tokens(token_text)
        if not preferred_expr or not tokens:
            continue

        acceptable_refs: Set[str] = set()
        for ref in {preferred_expr, alias_sql, alias, expr_name, expr}:
            norm = _normalize_order_reference(ref)
            if norm:
                acceptable_refs.add(norm)
                if "." in norm:
                    acceptable_refs.add(norm.split(".")[-1])

        candidates.append(
            {
                "preferred_expr": preferred_expr,
                "tokens": tokens,
                "acceptable_refs": acceptable_refs,
            }
        )

    if not candidates:
        return None

    mapped: List[Dict[str, Any]] = []
    for phrase, direction in requested:
        phrase_token_variants = _requested_order_phrase_token_variants(question, phrase)
        if not phrase_token_variants:
            return None

        scored: List[Tuple[int, Dict[str, Any]]] = []
        for candidate in candidates:
            candidate_tokens = set(candidate.get("tokens") or set())
            best_variant_score: Optional[int] = None
            for phrase_tokens in phrase_token_variants:
                overlap = len(phrase_tokens.intersection(candidate_tokens))
                extra_tokens = candidate_tokens - phrase_tokens
                if phrase_tokens.issubset(candidate_tokens):
                    exact_bonus = 30 if candidate_tokens == phrase_tokens else 0
                    # Prefer the closest projection match when multiple projected
                    # columns share the requested tokens, e.g. "total quantity"
                    # should map to total_quantity before highest_total_quantity.
                    closeness_penalty = 4 * len(extra_tokens)
                    score = 100 + exact_bonus + overlap - closeness_penalty
                elif overlap and overlap == len(phrase_tokens):
                    exact_bonus = 30 if candidate_tokens == phrase_tokens else 0
                    closeness_penalty = 4 * len(extra_tokens)
                    score = 50 + exact_bonus + overlap - closeness_penalty
                else:
                    continue
                if best_variant_score is None or score > best_variant_score:
                    best_variant_score = score
            if best_variant_score is not None:
                scored.append((best_variant_score, candidate))

        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            return None

        best_score = scored[0][0]
        best_candidates = [cand for score, cand in scored if score == best_score]
        best_exprs = {
            str(cand.get("preferred_expr") or "").strip()
            for cand in best_candidates
            if str(cand.get("preferred_expr") or "").strip()
        }
        if len(best_exprs) != 1:
            return None

        chosen = best_candidates[0]
        mapped.append(
            {
                "phrase": phrase,
                "direction": direction,
                "preferred_expr": str(chosen.get("preferred_expr") or "").strip(),
                "acceptable_refs": set(chosen.get("acceptable_refs") or set()),
            }
        )

    return mapped if len(mapped) == len(requested) else None


def _extract_top_level_order_specs_from_ast(
    parsed: Any,
    sel: Any,
    exp_mod: Any,
) -> List[Dict[str, Any]]:
    def _semantic_clause_arg(name: str) -> Any:
        root_arg = getattr(parsed, "args", {}).get(name)
        if root_arg is not None:
            return root_arg
        return getattr(sel, "args", {}).get(name)

    order_arg = _semantic_clause_arg("order")
    if order_arg is None:
        return []

    items: List[Dict[str, Any]] = []
    for ordered in getattr(order_arg, "expressions", None) or []:
        expr_node = getattr(ordered, "this", None)
        if expr_node is None:
            continue
        try:
            expr_sql = expr_node.sql(dialect=SQL_WRITE_DIALECT)
        except Exception:
            expr_sql = str(expr_node or "")

        acceptable_refs: Set[str] = set()
        expr_norm = _normalize_order_reference(expr_sql)
        if expr_norm:
            acceptable_refs.add(expr_norm)
            if "." in expr_norm:
                acceptable_refs.add(expr_norm.split(".")[-1])
        if isinstance(expr_node, exp_mod.Column):
            col_name = str(expr_node.name or "").strip("`")
            table_name = str(expr_node.table or "").strip("`")
            if col_name:
                acceptable_refs.add(_normalize_order_reference(col_name))
                if table_name:
                    acceptable_refs.add(_normalize_order_reference(f"{table_name}.{col_name}"))

        items.append(
            {
                "direction": "DESC" if bool(getattr(ordered, "args", {}).get("desc")) else "ASC",
                "acceptable_refs": acceptable_refs,
            }
        )
    return items


def _evaluate_requested_final_order_sequence(
    *,
    sql: str,
    question: str,
    parsed: Any,
    sel: Any,
    exp_mod: Any,
) -> List[str]:
    mapped = _build_requested_final_order_mapping(sql, question)
    if mapped is None or not mapped:
        return []

    actual_items = _extract_top_level_order_specs_from_ast(parsed, sel, exp_mod)
    if not actual_items:
        return []

    failures: List[str] = []
    if len(actual_items) < len(mapped):
        failures.append(f"missing_order_keys:{len(actual_items)}<{len(mapped)}")

    for idx, spec in enumerate(mapped):
        if idx >= len(actual_items):
            break
        expected_refs = {
            str(ref or "").strip().lower()
            for ref in (spec.get("acceptable_refs") or set())
            if str(ref or "").strip()
        }
        actual_refs = {
            str(ref or "").strip().lower()
            for ref in (actual_items[idx].get("acceptable_refs") or set())
            if str(ref or "").strip()
        }
        if expected_refs and actual_refs.isdisjoint(expected_refs):
            failures.append(f"order_key_mismatch:{idx + 1}")
        expected_dir = str(spec.get("direction") or "").upper()
        actual_dir = str(actual_items[idx].get("direction") or "").upper()
        if expected_dir and actual_dir and expected_dir != actual_dir:
            failures.append(f"order_dir_mismatch:{idx + 1}:{actual_dir}!={expected_dir}")

    return failures


def _complete_requested_final_order(sql: str, question: str) -> str:
    """
    Rebuild a simple top-level ORDER BY when the question gives an explicit
    multi-key final ordering ("..., then ...") and all keys map to projected
    columns/aliases. This is intentionally limited to flat SELECT outputs.
    """
    s = (sql or "").strip()
    if not s or not _has_top_level_order_by(s):
        return sql
    mapped_specs = _build_requested_final_order_mapping(s, question)
    if not mapped_specs:
        return sql
    # Ranking window ORDER BY is often part of row-number semantics.  Still
    # allow a top-level final ORDER BY repair when the question explicitly names
    # the final output order; the scanner below only edits the top-level tail.
    if re.search(r"\b(?:ROW_NUMBER|RANK|DENSE_RANK)\s*\(", s, flags=re.I) and not re.search(
        r"\b(?:final\s+)?(?:result|results|output|rows?)\s+(?:should\s+be\s+)?(?:sorted|ordered|arranged)\s+by\b"
        r"|\b(?:sort|sorted|order|ordered|arrange|arranged)\s+by\b"
        r"|\b(?:sort|order|arrange)\s+(?:the\s+)?final\s+(?:result|results|output|rows?)\s+by\b",
        str(question or ""),
        flags=re.I,
    ):
        return sql

    trailing_semicolon = ";" if s.endswith(";") else ""
    if trailing_semicolon:
        s = s[:-1].rstrip()

    from_pos = _find_top_level_kw(s, "from")
    order_pos = _find_top_level_kw(s, "order by")
    if from_pos < 0 or order_pos < 0 or order_pos <= from_pos:
        return s + trailing_semicolon

    select_segment = re.sub(r"^\s*select\s+", "", s[:from_pos], flags=re.I | re.S).strip()
    select_items = _split_top_level_csv(select_segment)
    if not select_items:
        return s + trailing_semicolon

    limit_pos = _find_top_level_kw(s, "limit", order_pos + len("order by"))
    order_end = limit_pos if limit_pos >= 0 else len(s)
    suffix = s[order_end:].strip()
    mapped = [
        f"{_quote_order_identifier_if_reserved(str(spec.get('preferred_expr') or '').strip())} {str(spec.get('direction') or 'ASC').upper()}"
        for spec in mapped_specs
        if str(spec.get("preferred_expr") or "").strip()
    ]
    if len(mapped) != len(mapped_specs):
        return s + trailing_semicolon

    prefix = s[:order_pos].rstrip()
    rebuilt = f"{prefix} ORDER BY {', '.join(mapped)}"
    if suffix:
        rebuilt += " " + suffix
    return rebuilt.rstrip() + trailing_semicolon


def _question_requests_offset_default_value(question: str) -> bool:
    q = str(question or "").lower()
    if not q:
        return False
    if not re.search(r"\b(?:previous|prior|next|following|lag|lead)\b", q, flags=re.I):
        return False
    return bool(
        re.search(
            r"\b(?:default|fallback|otherwise|coalesce|fill|replace|use)\b"
            r"[^.]{0,48}\b(?:zero|0|n/a|unknown|blank|empty)\b",
            q,
            flags=re.I | re.S,
        )
        or re.search(
            r"\b(?:if|when)\s+(?:there\s+is\s+)?no\s+"
            r"(?:previous|prior|next|following)\b[^.]{0,64}\b(?:zero|0|n/a|unknown|blank|empty)\b",
            q,
            flags=re.I | re.S,
        )
        or re.search(
            r"\b(?:zero|0|n/a|unknown|blank|empty)\b[^.]{0,48}\b"
            r"(?:when|if)\s+(?:there\s+is\s+)?no\s+"
            r"(?:previous|prior|next|following)\b",
            q,
            flags=re.I | re.S,
        )
    )


def _strip_unrequested_offset_window_default(sql: str, question: str) -> str:
    """
    Remove LAG/LEAD default arguments unless the question explicitly asks for
    fallback values. SQL defaults to NULL for missing previous/next rows, which
    matches plain natural-language "previous/next" semantics.
    """
    s = str(sql or "")
    if not s or not _question_has_offset_window_sequence_intent(question):
        return sql
    if _question_requests_offset_default_value(question):
        return sql
    if not re.search(r"\b(?:LAG|LEAD)\s*\(", s, flags=re.I):
        return sql

    out: List[str] = []
    idx = 0
    changed = False
    pattern = re.compile(r"\b(?P<func>LAG|LEAD)\s*\(", flags=re.I)

    while True:
        m = pattern.search(s, idx)
        if not m:
            out.append(s[idx:])
            break
        open_idx = s.find("(", m.start())
        if open_idx < 0:
            out.append(s[idx:])
            break

        depth = 0
        in_sq = False
        in_dq = False
        close_idx = -1
        i = open_idx
        while i < len(s):
            ch = s[i]
            if ch == "'" and not in_dq:
                if in_sq and i + 1 < len(s) and s[i + 1] == "'":
                    i += 2
                    continue
                in_sq = not in_sq
            elif ch == '"' and not in_sq:
                in_dq = not in_dq
            elif not in_sq and not in_dq:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        close_idx = i
                        break
            i += 1

        if close_idx < 0:
            out.append(s[idx:])
            break

        args_body = s[open_idx + 1:close_idx]
        args = _split_top_level_csv(args_body)
        if len(args) == 3:
            out.append(s[idx:m.end()])
            out.append(", ".join(arg.strip() for arg in args[:2]))
            out.append(")")
            changed = True
        else:
            out.append(s[idx:close_idx + 1])
        idx = close_idx + 1

    return "".join(out) if changed else sql


def _rewrite_same_table_correlated_aggregate_comparison(sql: str, partition_col: str) -> str:
    """
    Rewrite a same-table aggregate comparison into a derived window form.

    This handles generic "value > average value within the same group" shapes
    when the draft uses an ambiguous unaliased self-subquery. It only applies
    to simple single-table queries and uses the schema-resolved group column
    from the semantic contract.
    """
    s = (sql or "").strip()
    if not s or not partition_col or re.search(r"\bJOIN\b", s, flags=re.I):
        return sql

    trailing_semicolon = ";" if s.endswith(";") else ""
    if trailing_semicolon:
        s = s[:-1].rstrip()

    pattern = re.compile(
        r"""(?is)^\s*SELECT\s+(?P<select>.+?)\s+
            FROM\s+\(?\s*`?(?P<table>[A-Za-z_][A-Za-z0-9_]*)`?\s*\)?(?:\s+(?:AS\s+)?(?P<outer_alias>[A-Za-z_][A-Za-z0-9_]*))?\s+
            WHERE\s+(?P<metric>(?:`?[A-Za-z_][A-Za-z0-9_]*`?\s*\.\s*)?`?[A-Za-z_][A-Za-z0-9_]*`?)\s*
            (?P<op>>=|<=|<>|!=|=|>|<)\s*
            \(\s*SELECT\s+(?P<agg>AVG|MIN|MAX|SUM)\s*\(\s*(?P<inner_metric>(?:`?[A-Za-z_][A-Za-z0-9_]*`?\s*\.\s*)?`?[A-Za-z_][A-Za-z0-9_]*`?)\s*\)
            \s+FROM\s+\(?\s*`?(?P=table)`?\s*\)?(?:\s+(?:AS\s+)?(?P<inner_alias>[A-Za-z_][A-Za-z0-9_]*))?
            (?:\s+WHERE\s+(?P<inner_where>.*?))?
            \s*\)\s*(?P<tail>.*)$""",
        re.X,
    )
    m = pattern.match(s)
    if not m:
        return sql

    select_items = _split_top_level_csv(m.group("select") or "")
    if not select_items:
        return sql

    def _strip_qual(ref: str) -> str:
        ref = str(ref or "").strip().strip("`")
        if "." in ref:
            ref = ref.split(".")[-1].strip("`")
        return ref

    metric_col = _strip_qual(m.group("metric"))
    inner_metric_col = _strip_qual(m.group("inner_metric")) or metric_col
    metric_col = metric_col or inner_metric_col
    partition_col = str(partition_col or "").strip("`")
    if not metric_col or not partition_col:
        return sql

    # Keep this helper for flat projections only.
    output_items: List[str] = []
    needed_cols: List[str] = []
    for item in select_items:
        raw = item.strip()
        if re.search(r"\b(?:COUNT|SUM|AVG|MIN|MAX|OVER|CASE)\s*\(", raw, flags=re.I):
            return sql
        alias_match = re.search(r"\bAS\s+.+$", raw, flags=re.I)
        expr = raw[:alias_match.start()].strip() if alias_match else raw
        col_match = re.fullmatch(r"(?:`?[A-Za-z_][A-Za-z0-9_]*`?\s*\.\s*)?`?([A-Za-z_][A-Za-z0-9_]*)`?", expr)
        if not col_match:
            return sql
        col = col_match.group(1)
        output_items.append(raw if "." not in raw else re.sub(r"(?:`?[A-Za-z_][A-Za-z0-9_]*`?\s*\.)", "", raw))
        if col not in needed_cols:
            needed_cols.append(col)

    tail = str(m.group("tail") or "").strip()
    order_pos = _find_top_level_kw(tail, "order by")
    if order_pos >= 0:
        order_clause = tail[order_pos + len("order by"):].strip()
        for order_item in _split_top_level_csv(re.sub(r"\bLIMIT\b.*$", "", order_clause, flags=re.I | re.S)):
            col_match = re.search(r"(?:^|\.|`)([A-Za-z_][A-Za-z0-9_]*)`?\s*(?:ASC|DESC)?\s*$", order_item.strip(), flags=re.I)
            if col_match:
                col = col_match.group(1)
                if col not in needed_cols:
                    needed_cols.append(col)

    for col in (metric_col, partition_col):
        if col not in needed_cols:
            needed_cols.append(col)

    agg = str(m.group("agg") or "AVG").upper()
    avg_alias = re.sub(r"[^A-Za-z0-9_]+", "_", f"{agg.lower()}_{metric_col}_by_{partition_col}").strip("_")
    derived_alias = "base_rows"
    inner_select = ", ".join(needed_cols)
    outer_select = ", ".join(output_items)
    op = m.group("op")
    rebuilt = (
        f"SELECT {outer_select} FROM ("
        f"SELECT {inner_select}, {agg}({inner_metric_col}) OVER (PARTITION BY {partition_col}) AS {avg_alias} "
        f"FROM {m.group('table')}"
        f") AS {derived_alias} "
        f"WHERE {metric_col} {op} {avg_alias}"
    )
    if tail:
        rebuilt += " " + re.sub(r"(?:`?[A-Za-z_][A-Za-z0-9_]*`?\s*\.)", "", tail).strip()
    return rebuilt.rstrip() + trailing_semicolon


def _has_same_group_avg_difference_shape(sql: str) -> bool:
    s = str(sql or "")
    if not s:
        return False
    return bool(
        re.search(
            r"""(?is)
            (?:`?[A-Za-z_][A-Za-z0-9_]*`?(?:\s*\.\s*`?[A-Za-z_][A-Za-z0-9_]*`?)?)
            \s*-\s*
            AVG\s*\([^)]*\)\s*OVER\s*\(
            """,
            s,
            flags=re.I | re.X,
        )
    )


def _rewrite_same_group_avg_projection_to_difference(sql: str, question: str) -> str:
    """
    Repair a common near-miss for questions like:
      "gap/difference between each row value and the average value for the same group"

    If the draft already has a row-preserving AVG(...) OVER (PARTITION BY ...)
    projection but forgets the subtraction, rewrite only that projected window
    metric into `base_metric - AVG(base_metric) OVER (...)`.
    """
    s = str(sql or "").strip()
    q = str(question or "").strip().lower()
    if not s or not q:
        return sql
    if not (
        any(re.search(pat, q, flags=re.I) for pat in HELPERS_AVERAGE_INTENT_PATTERNS)
        and any(re.search(pat, q, flags=re.I) for pat in HELPERS_SAME_GROUP_INTENT_PATTERNS)
        and any_phrase_in_text(q, group_average_difference_window_phrases())
    ):
        return sql
    if _has_same_group_avg_difference_shape(s):
        return sql
    if re.search(r"\bJOIN\b", s, flags=re.I):
        return sql
    if not re.search(r"\bAVG\s*\([^)]*\)\s*OVER\s*\(\s*PARTITION\s+BY\b", s, flags=re.I):
        return sql

    from_pos = _find_top_level_kw(s, "from")
    if from_pos < 0:
        return sql

    select_prefix = re.match(r"(?is)^\s*SELECT\s+", s)
    if not select_prefix:
        return sql
    select_body = s[select_prefix.end():from_pos].strip()
    if not select_body:
        return sql

    select_items = _split_top_level_csv(select_body)
    if not select_items:
        return sql

    def _split_expr_alias(item: str) -> Tuple[str, str]:
        raw = str(item or "").strip()
        if not raw:
            return "", ""
        m = re.match(
            r"""(?is)^\s*(?P<expr>.*?)\s+AS\s+(?P<alias>`[^`]+`|'[^']+'|"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)\s*$""",
            raw,
        )
        if m:
            return (m.group("expr") or "").strip(), (m.group("alias") or "").strip()
        return raw, ""

    plain_metric_refs: Dict[str, str] = {}
    for item in select_items:
        expr, _alias = _split_expr_alias(item)
        if not expr or re.search(r"\b(?:COUNT|SUM|AVG|MIN|MAX|OVER|CASE)\s*\(", expr, flags=re.I):
            continue
        m_col = re.fullmatch(
            r"""(?is)(?P<ref>(?:`?[A-Za-z_][A-Za-z0-9_]*`?\s*\.\s*)?`?(?P<col>[A-Za-z_][A-Za-z0-9_]*)`?)""",
            expr.strip(),
        )
        if not m_col:
            continue
        plain_metric_refs[(m_col.group("col") or "").strip("`").lower()] = (m_col.group("ref") or "").strip()

    avg_item_pattern = re.compile(
        r"""(?is)^\s*
        AVG\s*\(\s*(?P<metric>(?:`?[A-Za-z_][A-Za-z0-9_]*`?\s*\.\s*)?`?(?P<metric_col>[A-Za-z_][A-Za-z0-9_]*)`?)\s*\)
        \s*OVER\s*\(\s*PARTITION\s+BY\s+(?P<partition>.*?)\s*\)
        \s*$
        """,
        re.X,
    )

    changed = False
    rewritten_items: List[str] = []
    for item in select_items:
        raw = str(item or "").strip()
        expr, alias = _split_expr_alias(raw)
        if not expr:
            rewritten_items.append(raw)
            continue
        m_avg = avg_item_pattern.match(expr)
        if not m_avg:
            rewritten_items.append(raw)
            continue

        metric_col = (m_avg.group("metric_col") or "").strip("`").lower()
        base_metric_ref = plain_metric_refs.get(metric_col, "")
        if not base_metric_ref:
            rewritten_items.append(raw)
            continue

        diff_expr = f"{base_metric_ref} - {expr.strip()}"
        rewritten_items.append(f"{diff_expr} AS {alias}" if alias else diff_expr)
        changed = True

    if not changed:
        return sql

    rebuilt_select = ", ".join(rewritten_items)
    return s[:select_prefix.end()] + rebuilt_select + " " + s[from_pos:].lstrip()


def _apply_semantic_contract(sql: str, question: str, ctx: str = "") -> Tuple[str, List[str], List[str]]:
    """
    Apply low-risk semantic repairs and report any remaining unmet contracts.
    """
    contract = _derive_semantic_contract(question, ctx)
    if not sql:
        return sql, [], []

    out_sql = sql
    corrections: List[str] = []
    violations: List[str] = []

    fixed = _repair_near_miss_output_alias_references(out_sql, ctx)
    if fixed != out_sql:
        out_sql = fixed
        corrections.append("Semantic contract: repaired near-miss output alias reference")

    fixed = _rewrite_leading_character_grouping(out_sql, question, ctx)
    if fixed != out_sql:
        out_sql = fixed
        corrections.append("Semantic contract: normalized leading-character grouping expression")

    has_set_operation = bool(
        re.search(r"\b(?:UNION(?:\s+ALL)?|INTERSECT|EXCEPT)\b", out_sql, flags=re.I)
    )
    # GROUP-BY backfill is unsafe on set-operation chains because each branch
    # is a full SELECT unit; applying a flat rewrite can corrupt branch tails.
    if not has_set_operation:
        fixed = _ensure_group_by_for_non_agg(out_sql)
        if fixed != out_sql:
            out_sql = fixed
            corrections.append("Semantic contract: added missing GROUP BY for non-aggregate select")

    if contract["requires_projection_distinct"]:
        fixed = _enforce_set_operation_operator_from_question(out_sql, question)
        if fixed != out_sql:
            out_sql = fixed
            corrections.append("Semantic contract: enforced set-operation dedupe operator")
        fixed = _ensure_reasonable_distinct(out_sql, question)
        if fixed != out_sql:
            out_sql = fixed
            corrections.append("Semantic contract: enforced projection DISTINCT")
        has_set_level_dedupe = bool(
            _question_has_set_operation_intent(question)
            and re.search(r"\bUNION\b(?!\s+ALL\b)", out_sql, flags=re.I)
        )
        if not has_set_level_dedupe and not re.search(r"^\s*SELECT\s+DISTINCT\b", out_sql, flags=re.I):
            violations.append("missing_projection_distinct")

    if contract["requires_count_distinct"]:
        fixed = _ensure_reasonable_count_distinct(out_sql, question)
        if fixed != out_sql:
            out_sql = fixed
            corrections.append("Semantic contract: enforced COUNT(DISTINCT)")
        if "COUNT(" in out_sql.upper() and not re.search(r"\bCOUNT\s*\(\s*DISTINCT\b", out_sql, flags=re.I):
            violations.append("missing_count_distinct")

    if contract["requires_winner_row"] and not _has_winner_row_shape(out_sql):
        fixed = _rewrite_joined_ranked_subquery_to_outer_winner(out_sql, question, ctx)
        if fixed != out_sql:
            out_sql = fixed
            corrections.append("Semantic contract: lifted child-ranked join to parent winner-row")
        fixed = _rewrite_malformed_window_winner_clause(out_sql, question)
        if fixed != out_sql:
            out_sql = fixed
            corrections.append("Semantic contract: repaired malformed WINDOW winner-row clause")
        fixed = _rewrite_grouped_winner_row_ranked_aggregate(out_sql, question)
        if fixed != out_sql:
            out_sql = fixed
            corrections.append("Semantic contract: enforced winner-row pattern")
        fixed = _fix_winner_row_metric_order_priority(out_sql, question)
        if fixed != out_sql:
            out_sql = fixed
            corrections.append("Semantic contract: normalized winner-row metric priority")
        if not _has_winner_row_shape(out_sql):
            violations.append("missing_winner_row")

    if contract["requires_anti_existence"]:
        fixed = _rewrite_missing_temporal_exclusion_from_null_filter(out_sql, question, ctx)
        if fixed != out_sql:
            out_sql = fixed
            corrections.append("Semantic contract: synthesized anti-existence from temporal null-filter shortcut")
        if not _has_anti_existence_shape(out_sql):
            violations.append("missing_anti_existence")

    if contract["requires_include_zero"]:
        if not _has_include_zero_shape(out_sql):
            violations.append("missing_include_zero")

    if bool(contract.get("requires_offset_window")):
        fixed = _strip_unrequested_offset_window_default(out_sql, question)
        if fixed != out_sql:
            out_sql = fixed
            corrections.append("Semantic contract: removed unrequested LAG/LEAD default value")
        if not _has_required_offset_window(out_sql, question):
            violations.append("missing_offset_window")

    if bool(contract.get("requires_scalar_extrema_date_filter")):
        fixed = _rewrite_scalar_extrema_date_filter(out_sql, question)
        if fixed != out_sql:
            out_sql = fixed
            corrections.append("Semantic contract: enforced scalar extrema-date filter")
        if not _has_scalar_extrema_date_filter_shape(out_sql):
            violations.append("missing_scalar_extrema_date_filter")

    required_window_partition_col = str(contract.get("required_window_partition_col") or "").strip()
    if required_window_partition_col:
        fixed = _rewrite_same_table_correlated_aggregate_comparison(
            out_sql,
            required_window_partition_col,
        )
        if fixed != out_sql:
            out_sql = fixed
            corrections.append(
                f"Semantic contract: normalized same-table aggregate comparison by {required_window_partition_col}"
            )
        fixed = _rewrite_grouped_aggregate_to_window_partition(
            out_sql,
            required_window_partition_col,
            question,
        )
        if fixed != out_sql:
            out_sql = fixed
            corrections.append(
                f"Semantic contract: rewrote grouped aggregate to window PARTITION BY {required_window_partition_col}"
            )
        fixed = _rewrite_required_window_partition(out_sql, required_window_partition_col)
        if fixed != out_sql:
            out_sql = fixed
            corrections.append(
                f"Semantic contract: enforced window PARTITION BY {required_window_partition_col}"
            )
        if not _has_required_window_partition(out_sql, required_window_partition_col):
            violations.append("missing_required_window_partition")

    if bool(contract.get("requires_same_group_avg_difference")):
        fixed = _rewrite_same_group_avg_projection_to_difference(out_sql, question)
        if fixed != out_sql:
            out_sql = fixed
            corrections.append("Semantic contract: enforced same-group average difference projection")
        if not _has_same_group_avg_difference_shape(out_sql):
            violations.append("missing_same_group_avg_difference")

    # Apply top-level final ordering late, after structural rewrites such as
    # window-partition normalization, so later repairs do not invalidate a
    # freshly repaired ORDER BY and trigger unnecessary regeneration.
    if contract["requires_final_order"]:
        fixed = _add_missing_final_order(out_sql, question)
        if fixed != out_sql:
            out_sql = fixed
            corrections.append("Semantic contract: added missing final ORDER BY")
        fixed = _enforce_requested_order_direction(out_sql, question)
        if fixed != out_sql:
            out_sql = fixed
            corrections.append("Semantic contract: enforced final ORDER BY direction")
        fixed = _complete_requested_final_order(out_sql, question)
        if fixed != out_sql:
            out_sql = fixed
            corrections.append("Semantic contract: completed requested final ORDER BY keys")
        fixed = _repair_chronological_month_order(out_sql, question)
        if fixed != out_sql:
            out_sql = fixed
            corrections.append("Semantic contract: normalized chronological month ordering")
        if not _has_top_level_order_by(out_sql):
            violations.append("missing_final_order")
        elif _build_requested_final_order_mapping(out_sql, question):
            try:
                import sqlglot
                from sqlglot import exp

                parsed_order = sqlglot.parse_one(out_sql, read=SQL_PARSE_DIALECT)
                sel_order = parsed_order.find(exp.Select)
                if sel_order is not None:
                    order_failures = _evaluate_requested_final_order_sequence(
                        sql=out_sql,
                        question=question,
                        parsed=parsed_order,
                        sel=sel_order,
                        exp_mod=exp,
                    )
                    if order_failures:
                        violations.append("incomplete_final_order")
            except Exception:
                pass

    requires_boundary_regex = bool(contract.get("requires_boundary_regex"))
    requires_regex = bool(contract.get("requires_regex"))
    requires_structural_regex = bool(contract.get("requires_structural_regex"))
    if requires_regex or requires_boundary_regex or requires_structural_regex:
        if not _regex_requirement_ok(
            out_sql,
            requires_regex,
            requires_boundary_regex,
            requires_structural_regex,
        ):
            violations.append("missing_regex_semantics")

    active_contracts: List[str] = []
    if contract["requires_projection_distinct"]:
        active_contracts.append("distinct")
    if contract["requires_count_distinct"]:
        active_contracts.append("count_distinct")
    if contract["requires_winner_row"]:
        active_contracts.append("winner_row")
    if contract["requires_anti_existence"]:
        active_contracts.append("anti_existence")
    if contract["requires_include_zero"]:
        active_contracts.append("include_zero")
    if contract["requires_final_order"]:
        active_contracts.append("final_order")
    if bool(contract.get("requires_offset_window")):
        active_contracts.append("offset_window")
    if bool(contract.get("requires_same_group_avg_difference")):
        active_contracts.append("same_group_avg_difference")
    if bool(contract.get("requires_scalar_extrema_date_filter")):
        active_contracts.append("scalar_extrema_date_filter")
    if required_window_partition_col:
        active_contracts.append(f"window_partition:{required_window_partition_col}")
    if requires_regex or requires_boundary_regex or requires_structural_regex:
        active_contracts.append("regex")

    if violations:
        status = "pending_retry"
    elif corrections:
        status = "autofix_passed"
    elif active_contracts:
        status = "passed"
    else:
        status = "not_applicable"

    detail = f"requires={active_contracts}; status={status}"
    if corrections:
        detail += f"; fixes={len(corrections)}"
    if violations:
        detail += f"; violations={violations}"
    _dbg_print_two("[REFINER - CONTRACT]", detail)

    return out_sql, corrections, list(dict.fromkeys(violations))

# ======================================================================
# ======================= AGENT 0 - QUESTION GUARD =====================
# ======================================================================
# Purpose: Reject non-SQL questions BEFORE hitting main LLM to prevent:
#   1. Garbage output from irrelevant questions
#   2. Resource spike from spam/abuse
#
# Strategy: Two-layer hybrid filter
#   Layer 1: REGEX - Instant reject obvious garbage (zero cost)
#   Layer 2: LLM 1.5B - Smart judge for borderline cases
# ======================================================================

# --- Layer 1: Patterns to INSTANT REJECT (obvious garbage) ---
_INSTANT_REJECT_PATTERNS = AGENT_INSTANT_REJECT_PATTERNS

# --- Patterns that BYPASS LLM check (definitely SQL-related) ---
_DEFINITE_SQL_PATTERNS = AGENT_DEFINITE_SQL_PATTERNS

try:
    _TABLE_SINGULAR_ROOTS = {
        get_singular_form(t.lower().strip())
        for t in get_all_tables()
        if t
    }
except Exception:
    _TABLE_SINGULAR_ROOTS = set()

def _table_name_variants(table: str) -> Set[str]:
    """
    Build robust table-name variants from schema names only (no column synonyms).
    Includes table name, singular form, and underscore-spaced forms.
    """
    t = (table or "").lower().strip()
    if not t:
        return set()

    variants = {t}
    singular = get_singular_form(t)
    if singular:
        variants.add(singular.lower())

    if "_" in t:
        variants.add(t.replace("_", " "))
    if singular and "_" in singular:
        variants.add(singular.replace("_", " "))

    # Build spaced variants for concatenated table names using other schema roots.
    # Example: productlines -> product lines, orderdetails -> order details
    for base in list(variants):
        compact = base.replace(" ", "")
        if not compact.isalpha():
            continue
        for root in _TABLE_SINGULAR_ROOTS:
            if not root or root == compact:
                continue
            if compact.startswith(root):
                tail = compact[len(root):]
                if len(tail) >= AGENT_TABLE_VARIANT_MIN_TAIL_LEN and tail.isalpha():
                    variants.add(f"{root} {tail}")

    return {v for v in variants if v}

def _table_entity_pattern(table: str) -> str:
    """
    Regex alternation pattern for table variants.
    Example output: '(?:orders|order)'
    """
    variants = sorted(_table_name_variants(table), key=len, reverse=True)
    if not variants:
        return r"(?:)"

    escaped = [re.escape(v).replace(r"\ ", r"\s+") for v in variants]
    return "(?:" + "|".join(escaped) + ")"


def _is_detail_child_entity(base_table: str, candidate_table: str) -> bool:
    """
    Detect schema-driven "detail/line/item" child naming relationship.
    Example: base=orders, candidate=orderdetails -> True.
    """
    base = (base_table or "").strip().lower()
    cand = (candidate_table or "").strip().lower()
    if not base or not cand or base == cand:
        return False

    base_singular = (get_singular_form(base) or "").strip().lower()
    if not base_singular:
        return False

    detail_suffixes = (
        "detail", "details",
        "line", "lines",
        "item", "items",
        "entry", "entries",
    )
    candidate_variants = {v.replace("_", " ").strip().lower() for v in _table_name_variants(cand)}
    candidate_variants.add(cand.replace("_", " ").strip().lower())
    candidate_variants.add(cand.replace("_", "").strip().lower())

    expected_variants: Set[str] = set()
    for suffix in detail_suffixes:
        expected_variants.add(f"{base_singular} {suffix}")
        expected_variants.add(f"{base_singular}{suffix}")

    return bool(candidate_variants.intersection(expected_variants))


def _question_mentions_entity_display_name(question_lower: str, table: str) -> bool:
    """
    Return True only when the question locally ties a display-name request to
    a specific entity, e.g. "product name" or "name of the customer".

    This avoids treating a generic "name" mention elsewhere in the question as
    evidence that every mentioned entity needs its display-name column.
    """
    q = (question_lower or "").strip().lower()
    entity_pattern = _table_entity_pattern(table)
    if not q or not entity_pattern:
        return False

    name_pat = r"(?:name|names|full\s+name)"
    patterns = [
        rf"\b{entity_pattern}\b(?:\s+\w+){{0,2}}\s+{name_pat}\b",
        rf"\b{name_pat}\b(?:\s+\w+){{0,2}}\s+\bof\s+{entity_pattern}\b",
        rf"\b{name_pat}\b(?:\s+\w+){{0,2}}\s+\bfor\s+{entity_pattern}\b",
    ]
    return any(re.search(p, q, flags=re.I) for p in patterns)


def _instantiate_entity_patterns(templates: List[str], entity_pattern: str) -> List[str]:
    """
    Replace entity placeholders in regex templates.
    Supports both "__ENTITY__" and "{entity}" placeholders for env overrides.
    """
    patterns: List[str] = []
    for tpl in templates or []:
        raw = str(tpl or "").strip()
        if not raw:
            continue
        pat = raw.replace("__ENTITY__", entity_pattern).replace("{entity}", entity_pattern)
        patterns.append(pat)
    return patterns


def _explicit_subject_match_score(question_lower: str, entity_pattern: str) -> int:
    """
    Score explicit-subject match specificity using config-driven templates.
    Higher score means a longer concrete entity phrase was matched
    (e.g., "product lines" > "product").
    """
    if not question_lower or not entity_pattern:
        return 0

    best = 0
    ent_group = f"(?P<ent>{entity_pattern})"
    for tpl in AGENT_ROUTER_EXPLICIT_SUBJECT_ENTITY_PATTERNS or []:
        raw = str(tpl or "").strip()
        if not raw:
            continue
        pat = raw.replace("__ENTITY__", ent_group).replace("{entity}", ent_group)
        try:
            m = re.search(pat, question_lower, flags=re.I)
        except re.error:
            continue
        if not m:
            continue
        ent = (m.groupdict().get("ent") or "").strip()
        if ent:
            best = max(best, len(ent))
        else:
            best = max(best, 1)
    return best


def _entity_search_scope(question_lower: str, table_singular: str) -> str:
    """
    Build a cleaned search scope for one entity by stripping generic grammar
    uses (e.g., "order by") that are not table mentions.
    """
    scope = question_lower or ""
    patterns = AGENT_ROUTER_IGNORE_PATTERNS.get((table_singular or "").lower(), [])
    for pat in patterns:
        scope = re.sub(pat, " ", scope)
    return scope


def _has_router_phrase_term(text: str, phrase: str) -> bool:
    """
    Word-boundary phrase matcher for router keyword lists.

    Avoids substring false positives such as:
    - phrase: "have no"
    - text  : "have not been shipped"
    """
    txt = str(text or "")
    phr = str(phrase or "").strip().lower()
    if not txt or not phr:
        return False
    pat = re.escape(phr).replace(r"\ ", r"\s+")
    return bool(re.search(rf"(?<![A-Za-z0-9_]){pat}(?![A-Za-z0-9_])", txt, flags=re.I))


def _active_router_cross_table_exclusion_terms(question: str) -> List[str]:
    active_terms: List[str] = list(AGENT_ROUTER_GENERIC_CROSS_TABLE_EXCLUSION_TERMS)
    if DOMAIN_HEURISTICS_ENABLED:
        q = str(question or "").lower()
        for term in AGENT_ROUTER_DOMAIN_CROSS_TABLE_EXCLUSION_TERMS:
            if _has_router_phrase_term(q, term):
                active_terms.append(term)
    return list(
        dict.fromkeys(
            str(term or "").strip().lower()
            for term in active_terms
            if str(term or "").strip()
        )
    )


def _question_targets_entity_exclusion(question_lower: str, table: str) -> bool:
    """
    Return True when question expresses exclusion/absence semantics for this entity:
    e.g., "no orders", "never placed any orders", "without payments".
    """
    tbl_lower = (table or "").lower().strip()
    if not tbl_lower:
        return False

    table_singular = get_singular_form(tbl_lower)
    entity_pattern = _table_entity_pattern(tbl_lower)
    if not entity_pattern or entity_pattern == r"(?:)":
        return False

    scope = _entity_search_scope(question_lower, table_singular)

    # Strong direct pattern: negation terms within local window before entity mention.
    if re.search(
        rf"\b(?:no|without(?:\s+any)?|never|not)\b(?:[\s,;:()'\"-]+\w+){{0,8}}[\s,;:()'\"-]+(?:{entity_pattern})\b",
        scope,
        flags=re.I,
    ):
        return True

    # Backup: global exclusion phrase + explicit entity mention in same question.
    active_exclusion_terms = _active_router_cross_table_exclusion_terms(scope)
    if any(_has_router_phrase_term(scope, term) for term in active_exclusion_terms):
        if re.search(rf"\b(?:{entity_pattern})\b", scope, flags=re.I):
            return True

    return False


def _router_id_terms_pattern() -> str:
    terms = [str(t or "").strip().lower() for t in AGENT_ROUTER_ID_REFERENCE_TERMS if str(t or "").strip()]
    if not terms:
        return r"(?:id)"
    return "|".join(re.escape(t) for t in sorted(set(terms), key=len, reverse=True))

def _question_mentions_schema_entity(question_lower: str) -> bool:
    """
    Schema-driven entity detection for guard bypass.
    Uses dynamic table names (plus singular forms) instead of fixed table list.
    """
    try:
        for table in get_all_tables():
            for v in _table_name_variants(table):
                if v and re.search(rf"\b{re.escape(v)}\b", question_lower):
                    return True
    except Exception:
        # Keep guard permissive if schema analyzer is unavailable
        return False

    return False


def _schema_entities_for_examples(max_items: int = AGENT_GUARD_EXAMPLE_MAX_ITEMS) -> List[str]:
    """
    Build user-facing entity examples from schema tables (fallback-safe).
    """
    entities: List[str] = []
    try:
        for table in get_all_tables():
            variants = sorted(_table_name_variants(table), key=len)
            # Prefer simple natural form (usually singular/plural table name).
            if variants:
                entities.append(variants[0])
    except Exception:
        entities = []

    # Deduplicate while preserving order.
    seen = set()
    out = []
    for e in entities:
        k = (e or "").strip().lower()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(e)
        if len(out) >= max_items:
            break
    return out


def _example_data_queries() -> List[str]:
    entities = _schema_entities_for_examples(max_items=AGENT_GUARD_EXAMPLE_MAX_ITEMS)
    if len(entities) >= 3:
        return [
            f"Show all {entities[0]}",
            f"How many {entities[1]} in {AGENT_GUARD_EXAMPLE_YEAR}?",
            f"List {entities[2]} with value above {AGENT_GUARD_EXAMPLE_VALUE_THRESHOLD}",
        ]
    if len(entities) == 2:
        return [
            f"Show all {entities[0]}",
            f"How many {entities[1]} in {AGENT_GUARD_EXAMPLE_YEAR}?",
            f"List {entities[0]} with value above {AGENT_GUARD_EXAMPLE_VALUE_THRESHOLD}",
        ]
    if len(entities) == 1:
        return [
            f"Show all {entities[0]}",
            f"How many {entities[0]} in {AGENT_GUARD_EXAMPLE_YEAR}?",
            f"List {entities[0]} with value above {AGENT_GUARD_EXAMPLE_VALUE_THRESHOLD}",
        ]
    return [
        "Show all records",
        f"How many records in {AGENT_GUARD_EXAMPLE_YEAR}?",
        f"List records with value above {AGENT_GUARD_EXAMPLE_VALUE_THRESHOLD}",
    ]


def _format_example_block() -> str:
    examples = _example_data_queries()
    return "\n".join([f"  - '{ex}'" for ex in examples])

def _check_instant_reject(question: str) -> Tuple[bool, str]:
    """
    Layer 1: Regex-based instant rejection for obvious garbage.
    Returns: (should_reject, rejection_reason)
    """
    if not question or not question.strip():
        return True, "Empty question. Please enter a data query."
    
    q = question.strip()
    q_lower = q.lower()
    
    # --- Check 1: Pattern-based rejection (greetings, etc) ---
    for pattern in _INSTANT_REJECT_PATTERNS:
        try:
            is_match = re.match(pattern, q_lower, re.IGNORECASE)
        except re.error as exc:
            print(f"[ROUTER - GUARD] Skipping invalid reject regex {pattern!r}: {exc}")
            continue
        if is_match:
            return True, (
                "This doesn't look like a data query. Please ask about your data, for example:\n"
                + _format_example_block()
            )
    
    # --- Check 2: Gibberish detection ---
    is_gibberish, gibberish_reason = _detect_gibberish(q)
    if is_gibberish:
        return True, gibberish_reason
    
    return False, ""

def _detect_gibberish(text: str) -> Tuple[bool, str]:
    """
    Detect gibberish/random text using heuristics.
    Returns: (is_gibberish, reason)
    
    Heuristics:
    1. Vowel ratio too low (normal text ~35-45% vowels)
    2. Too many consecutive consonants (>6 in a row) - raised threshold
    3. No spaces in long text (>15 chars without space)
    4. Repeated character patterns
    """
    if not text or len(text) < AGENT_GIBBERISH_MIN_TEXT_LENGTH:
        return False, ""  # Too short to judge, let other checks handle
    
    t = text.lower().strip()
    
    # IMPORTANT: If text has spaces (likely a sentence), be more lenient
    # Real sentences like "Find products whose names start with..." are valid
    has_spaces = ' ' in t
    word_count = len(t.split())
    
    # If it looks like a real sentence (3+ words), skip gibberish check
    if has_spaces and word_count >= AGENT_GIBBERISH_SENTENCE_WORD_BYPASS:
        return False, ""
    
    alpha_only = re.sub(AGENT_GIBBERISH_ALPHA_ONLY_PATTERN, '', t)
    
    if len(alpha_only) < AGENT_GIBBERISH_MIN_ALPHA_LENGTH:
        return False, ""  # Not enough letters to analyze
    
    # --- Heuristic 1: Vowel ratio ---
    vowels = set(AGENT_GIBBERISH_VOWELS or "aiueo")
    vowel_count = sum(1 for c in alpha_only if c in vowels)
    vowel_ratio = vowel_count / len(alpha_only)
    
    # Normal English/Indonesian text has ~35-50% vowels
    # "asdgasdgasd" has ratio ~0.18 (too low)
    # Only apply to single-word inputs (no spaces)
    if vowel_ratio < AGENT_GIBBERISH_VOWEL_RATIO_MIN and len(alpha_only) > AGENT_GIBBERISH_VOWEL_CHECK_MIN_LEN and not has_spaces:
        return True, "Input appears to be random text. Please enter a valid data query."
    
    # --- Heuristic 2: Consecutive consonants ---
    # Raised threshold to 7 to avoid false positives like "strengths", "start with"
    # "qwrtypsdfgh" still caught (10 consonants)
    # Only apply to single-word inputs
    if not has_spaces:
        consonant_run_min = max(1, AGENT_GIBBERISH_CONSONANT_RUN_MIN)
        consonant_class = AGENT_GIBBERISH_CONSONANT_CLASS
        if consonant_class:
            consonant_run = re.search(rf'[{re.escape(consonant_class)}]{{{consonant_run_min},}}', alpha_only)
            if consonant_run and len(alpha_only) > AGENT_GIBBERISH_CONSONANT_CHECK_MIN_LEN:
                return True, "Input appears to be random text. Please enter a valid data query."
    
    # --- Heuristic 3: No spaces in long text ---
    # Normal questions have spaces: "show all customers"
    # Gibberish often doesn't: "asdgasdgasdgasd"
    if len(t) > AGENT_GIBBERISH_NO_SPACE_LEN and ' ' not in t:
        # Exception: could be a single long word like "customerNumber"
        # Check if it looks like camelCase or has some structure
        if not re.search(AGENT_GIBBERISH_HAS_UPPER_PATTERN, text) and not re.search(AGENT_GIBBERISH_HAS_SEPARATOR_PATTERN, text):
            return True, "Input appears to be random text. Please enter a valid data query."
    
    # --- Heuristic 4: Repeated patterns ---
    # "abcabcabc" or "xyxyxyxy"
    # Only for short inputs without spaces
    if len(t) >= AGENT_GIBBERISH_REPEAT_MIN_LEN and not has_spaces:
        # Check for 2-3 char patterns repeated 3+ times
        for pattern_len in AGENT_GIBBERISH_REPEAT_PATTERN_LENGTHS:
            if pattern_len <= 0:
                continue
            pattern = t[:pattern_len]
            if pattern * 3 in t and len(set(pattern)) < pattern_len:
                return True, "Input appears to be random text. Please enter a valid data query."
    
    # --- Heuristic 5: Keyboard mash detection ---
    # Common keyboard patterns: qwerty, asdf, zxcv
    # Only for inputs without spaces (single gibberish words)
    if not has_spaces:
        for kp in AGENT_GIBBERISH_KEYBOARD_PATTERNS:
            if kp in alpha_only:
                return True, "Input appears to be random text. Please enter a valid data query."
    
    return False, ""

def _check_definite_sql(question: str) -> bool:
    """
    Check if question definitely has SQL intent (bypass LLM check).
    """
    q_lower = question.lower()

    # Prefer schema-driven detection first
    if _question_mentions_schema_entity(q_lower):
        return True

    for pattern in _DEFINITE_SQL_PATTERNS:
        if re.search(pattern, q_lower, re.IGNORECASE):
            return True
    return False

def _llm_judge_sql_intent(question: str) -> Tuple[bool, str]:
    """
    Layer 2: Use LLM 1.5B to judge if question has SQL/data query intent.
    Returns: (is_valid, rejection_reason)
    """
    from langchain_ollama import ChatOllama
    
    llm = ChatOllama(
        model=ROUTING_MODEL,  # qwen2.5:1.5b-instruct
        base_url=OLLAMA_URL,
        temperature=AGENT_GUARD_LLM_TEMPERATURE,
        format="json",
        keep_alive=OLLAMA_KEEP_ALIVE,
        num_ctx=AGENT_GUARD_LLM_NUM_CTX,  # Small context for fast inference
        num_predict=AGENT_GUARD_LLM_NUM_PREDICT,  # Short response
    )
    
    schema_entities = _schema_entities_for_examples(max_items=AGENT_GUARD_SCHEMA_EXAMPLE_MAX_ITEMS)
    schema_entities_txt = ", ".join(schema_entities) if schema_entities else "schema entities"

    prompt = f"""You are a query intent classifier for a SQL database system.
Determine if the user's question is asking about DATA or DATABASE QUERIES.

VALID questions (answer "yes"):
- Questions about schema entities such as: {schema_entities_txt}
- Questions asking for lists, counts, totals, averages
- Questions with filters (by date, by country, by status)
- Questions asking for rankings (highest, lowest, top, most)

INVALID questions (answer "no"):
- Greetings, small talk, chitchat
- Questions about weather, news, general knowledge
- Personal questions, opinions, advice
- Random text, gibberish, spam

User question: "{question}"

Return JSON: {{"is_data_query": true/false, "reason": "brief explanation"}}"""

    try:
        response = llm.invoke(prompt)
        import json
        result = json.loads(response.content)
        is_valid = result.get("is_data_query", False)
        reason = result.get("reason", "")
        
        if not is_valid:
            return False, (
                f"This doesn't appear to be a data query. {reason}\n\n"
                "Please ask about your data, for example:\n"
                + _format_example_block()
            )
        
        return True, ""
        
    except Exception as e:
        # If LLM fails, be permissive (let it through)
        print(f"[ROUTER - GUARD] LLM judge failed: {e}, allowing question through")
        return True, ""

def validate_question_intent(question: str) -> Tuple[bool, str]:
    """
    Hybrid question validation: Regex + LLM 1.5B
    
    Returns:
        Tuple[bool, str]: (is_valid, rejection_reason)
        - (True, "") if question is valid
        - (False, reason) if question should be rejected
    
    Flow:
        1. Regex instant reject (obvious garbage) → No LLM cost
        2. Regex definite SQL (obvious valid) → No LLM cost
        3. LLM 1.5B judge (borderline cases) → Smart decision
    """
    print(f"[ROUTER - GUARD] Validating question: '{question[:50]}...'")
    
    # Layer 1: Instant reject obvious garbage
    should_reject, reason = _check_instant_reject(question)
    if should_reject:
        print(f"[ROUTER - GUARD] Layer 1 REJECT (regex): {reason[:50]}")
        return False, reason
    
    # Layer 1b: Instant accept definite SQL
    if _check_definite_sql(question):
        print(f"[ROUTER - GUARD] ✓ Layer 1 ACCEPT (definite SQL pattern)")
        return True, ""
    
    # Layer 2: LLM judge for borderline cases
    print(f"[ROUTER - GUARD] → Layer 2: LLM 1.5B judging...")
    is_valid, reason = _llm_judge_sql_intent(question)
    
    if is_valid:
        print(f"[ROUTER - GUARD] ✓ Layer 2 ACCEPT (LLM approved)")
    else:
        print(f"[ROUTER - GUARD] Layer 2 REJECT (LLM): {reason[:50]}")
    
    return is_valid, reason

def question_guard_node(state: RAGState) -> RAGState:
    """
    Standalone guard node - runs BEFORE retrieve to save resources.
    If rejected, pipeline ends early without calling retrieve/router/draft/refine.
    """
    line_sep = AGENT_GUARD_BANNER_CHAR * AGENT_GUARD_BANNER_WIDTH
    print(f"\n{line_sep}")
    print(f"{'QUESTION GUARD':^{AGENT_GUARD_BANNER_WIDTH}}")
    print(f"{line_sep}")
    
    question = state.get("question", "")
    
    is_valid, rejection_reason = validate_question_intent(question)
    
    if not is_valid:
        print(f"\nREJECTED: Question is not a valid data query")
        print(f"   Reason: {rejection_reason[:100]}...")
        print(f"{line_sep}\n")
        return {
            **state,
            "router_decision": {
                "complexity": AGENT_ROUTER_COMPLEXITY_REJECTED,
                "qtype": AGENT_ROUTER_QTYPE_INVALID,
                "rejection_reason": rejection_reason,
                "is_valid_question": False,
            },
            "final_sql": "",
            "answer": rejection_reason,
        }
    
    print(f"\n✓ PASSED: Question appears to be a valid data query")
    print(f"{line_sep}\n")
    
    return {
        **state,
        "router_decision": {
            "is_valid_question": True,
        }
    }


def _question_mentions_column(question_lower: str, column: str) -> bool:
    """
    Robust column mention detector:
    - direct full column name
    - spaced camelCase/snake_case variant
    - informative token-level match (e.g., quantityOrdered -> "ordered")
    """
    if not column:
        return False

    col_lower = column.lower()
    spaced = re.sub(AGENT_COLUMN_CAMEL_SPLIT_PATTERN, r'\1 \2', column).replace("_", " ").lower()

    # Full-form matches
    if col_lower in question_lower:
        return True
    if spaced != col_lower and spaced in question_lower:
        return True

    # Guardrail: avoid false positives like customerName matching just "customer id".
    # For semantic suffix columns, require suffix intent in question unless full phrase matched.
    semantic_suffix_keywords = AGENT_COLUMN_SEMANTIC_SUFFIX_KEYWORDS
    stop_tokens = AGENT_COLUMN_STOP_TOKENS
    col_tokens = re.findall(AGENT_COLUMN_TOKEN_PATTERN, spaced)
    if len(col_tokens) >= 2:
        suffix = col_tokens[-1]
        suffix_terms = semantic_suffix_keywords.get(suffix)
        if suffix_terms:
            has_suffix_intent = any(
                re.search(rf"\b{re.escape(term)}\b", question_lower)
                for term in suffix_terms
            )
            if not has_suffix_intent:
                return False
            # For semantic columns like productName/customerNumber, do not allow
            # a loose global match such as "customer name ... products".
            # Require at least one informative prefix token to appear locally
            # near a suffix term (e.g., "product name", "customer id").
            informative_prefix_tokens = [
                tok for tok in col_tokens[:-1]
                if len(tok) >= 4 and tok not in stop_tokens
            ]
            for token in informative_prefix_tokens:
                token_forms = {token}
                if (
                    AGENT_COLUMN_TOKEN_SUFFIX_Y
                    and token.endswith(AGENT_COLUMN_TOKEN_SUFFIX_Y)
                    and len(token) >= AGENT_COLUMN_TOKEN_SUFFIX_Y_MIN_LEN
                ):
                    token_forms.add(token[:-len(AGENT_COLUMN_TOKEN_SUFFIX_Y)] + AGENT_COLUMN_TOKEN_SUFFIX_IES)
                elif (
                    AGENT_COLUMN_TOKEN_SUFFIX_IES
                    and token.endswith(AGENT_COLUMN_TOKEN_SUFFIX_IES)
                    and len(token) >= AGENT_COLUMN_TOKEN_SUFFIX_IES_MIN_LEN
                ):
                    token_forms.add(token[:-len(AGENT_COLUMN_TOKEN_SUFFIX_IES)] + AGENT_COLUMN_TOKEN_SUFFIX_Y)
                elif (
                    AGENT_COLUMN_TOKEN_SUFFIX_S
                    and token.endswith(AGENT_COLUMN_TOKEN_SUFFIX_S)
                    and len(token) >= AGENT_COLUMN_TOKEN_SUFFIX_S_MIN_LEN
                ):
                    token_forms.add(token[:-1])
                else:
                    token_forms.add(token + AGENT_COLUMN_TOKEN_SUFFIX_S)

                for tok in token_forms:
                    if not tok:
                        continue
                    for term in suffix_terms:
                        if re.search(
                            rf"\b{re.escape(tok)}\b(?:\s+\w+){{0,2}}\s+\b{re.escape(term)}\b",
                            question_lower,
                        ):
                            return True
            return False

    # Token-level fallback (skip weak generic tokens)
    for token in re.findall(AGENT_COLUMN_TOKEN_PATTERN, spaced):
        if len(token) < 4 or token in stop_tokens:
            continue
        # "ordered by ..." usually indicates sorting intent, not orderdetails semantics
        if (
            AGENT_COLUMN_ORDERED_TOKEN
            and token == AGENT_COLUMN_ORDERED_TOKEN
            and _ordered_token_is_sort_only_context(question_lower)
        ):
            continue
        token_forms = {token}
        if AGENT_COLUMN_TOKEN_SUFFIX_Y and token.endswith(AGENT_COLUMN_TOKEN_SUFFIX_Y) and len(token) >= AGENT_COLUMN_TOKEN_SUFFIX_Y_MIN_LEN:
            token_forms.add(token[:-len(AGENT_COLUMN_TOKEN_SUFFIX_Y)] + AGENT_COLUMN_TOKEN_SUFFIX_IES)
        elif AGENT_COLUMN_TOKEN_SUFFIX_IES and token.endswith(AGENT_COLUMN_TOKEN_SUFFIX_IES) and len(token) >= AGENT_COLUMN_TOKEN_SUFFIX_IES_MIN_LEN:
            token_forms.add(token[:-len(AGENT_COLUMN_TOKEN_SUFFIX_IES)] + AGENT_COLUMN_TOKEN_SUFFIX_Y)
        elif AGENT_COLUMN_TOKEN_SUFFIX_S and token.endswith(AGENT_COLUMN_TOKEN_SUFFIX_S) and len(token) >= AGENT_COLUMN_TOKEN_SUFFIX_S_MIN_LEN:
            token_forms.add(token[:-1])
        else:
            token_forms.add(token + AGENT_COLUMN_TOKEN_SUFFIX_S)

        if any(re.search(rf"\b{re.escape(tok)}\b", question_lower) for tok in token_forms if tok):
            return True

    return False


def _is_conventional_fk_name(table_singular: str, column_name: str) -> bool:
    singular = (table_singular or "").strip().lower()
    col = (column_name or "").strip().lower()
    suffixes = [str(s or "").strip().lower() for s in AGENT_ROUTER_FK_SUFFIX_TERMS if str(s or "").strip()]
    if not singular or not col or not suffixes:
        return False
    suffix_pattern = "|".join(re.escape(s) for s in suffixes)
    fk_pattern = rf'^{re.escape(singular)}(?:{suffix_pattern})$'
    return bool(re.match(fk_pattern, col, re.IGNORECASE))

# ======================================================================
# ======================== AGENT 1 - ROUTER AGENT ======================
# ======================================================================
def _get_scenario_config(state: RAGState) -> Dict[str, Any]:
    cfg = state.get("scenario_config") or {}
    return cfg if isinstance(cfg, dict) else {}

def router_agent(state: RAGState) -> RAGState:
    """
    Router agent - determines query complexity and target model.
    Note: Question guard is now handled by separate question_guard_node (runs before retrieve).
    """
    print(f"\n[ROUTER] Evidence-Based Analysis starting...")
    question_raw = state.get("question", "")
    question = question_raw.lower()
    ctx = state.get("context", "")
    
    # Get existing router_decision from guard node (preserve is_valid_question)
    existing_decision = state.get("router_decision", {})
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 1: IDENTIFIKASI TABEL (Pattern-Based Exclusion)
    # ═══════════════════════════════════════════════════════════════════
    allowed_identifiers = _parse_allowed_identifiers(ctx)
    whitelisted_tables = allowed_identifiers.keys()
    whitelisted_lower = {t.lower() for t in whitelisted_tables}

    # Use full schema columns as fallback when context cards are column-pruned.
    analyzer = get_schema_analyzer()
    allowed_cols_by_lower: Dict[str, Set[str]] = {}
    for t, cols in allowed_identifiers.items():
        allowed_cols_by_lower.setdefault(t.lower(), set()).update(c.lower() for c in cols)

    def _get_table_cols(table_name: str) -> Set[str]:
        t = (table_name or "").lower()
        cols = set(allowed_cols_by_lower.get(t, set()))
        try:
            cols.update(c.lower() for c in analyzer.get_table_columns(t))
        except Exception:
            pass
        return cols
    
    # [FIX] Definisi Pattern yang HARUS DIBUANG sebelum deteksi tabel
    ignore_patterns = AGENT_ROUTER_IGNORE_PATTERNS
    
    relevant_tables = []
    id_terms_pattern = _router_id_terms_pattern()
    for tbl in whitelisted_tables:
        if f"- {tbl}:" not in ctx and f"- `{tbl}`:" not in ctx:
            continue

        table_singular = get_singular_form(tbl.lower())
        table_pattern = _table_entity_pattern(tbl)
        table_singular_token = (table_singular or "").strip().lower()

        # Buat copy pertanyaan khusus untuk scanning tabel ini
        search_scope = question

        if table_singular in ignore_patterns:
            patterns = ignore_patterns[table_singular]
            for pat in patterns:
                search_scope = re.sub(pat, " ", search_scope)

        # Cek 2: Cari kata tabel di search_scope yang SUDAH BERSIH.
        # Fallback: allow simple inflected forms (e.g., "ordered", "ordering")
        # for verb-like entity mentions when direct entity token is absent.
        direct_entity_match = bool(re.search(rf"\b{table_pattern}\b", search_scope))
        inflected_entity_match = False
        if (
            not direct_entity_match
            and table_singular_token
            and table_singular_token.isalpha()
            and len(table_singular_token) >= 4
        ):
            inflected_hit = re.search(
                rf"\b{re.escape(table_singular_token)}(?:ed|ing)\b",
                search_scope,
            )
            inflected_entity_match = bool(inflected_hit)
            if inflected_entity_match and inflected_hit is not None:
                left = max(0, inflected_hit.start() - 16)
                right = min(len(search_scope), inflected_hit.end() + 24)
                local_scope = search_scope[left:right]
                for ignore_pat in ignore_patterns.get(table_singular_token, []):
                    if re.search(ignore_pat, local_scope, flags=re.I):
                        inflected_entity_match = False
                        break

        if direct_entity_match or inflected_entity_match:

            # Deteksi referensi ID saja (FK)
            is_just_id_ref = re.search(
                rf"\b{table_pattern}\b\s+(?:{id_terms_pattern})\b",
                search_scope
            )

            if direct_entity_match and is_just_id_ref:
                if re.search(
                    rf"\b{table_pattern}\b(?!\s+(?:{id_terms_pattern}))",
                    search_scope
                ):
                    relevant_tables.append(tbl)
                    if RETRIEVE_DEBUG:
                        print(f"[ROUTER] ✓ Added {tbl} (with ID ref check)")
            else:
                relevant_tables.append(tbl)
                if RETRIEVE_DEBUG:
                    if inflected_entity_match and not direct_entity_match:
                        print(f"[ROUTER] ✓ Added {tbl} (inflected entity match)")
                    else:
                        print(f"[ROUTER] ✓ Added {tbl} (direct match)")
        elif RETRIEVE_DEBUG:
            pattern_str = rf"\b{table_pattern}\b"
            print(f"[ROUTER] ✗ Skipped {tbl} (no match in question)")
            print(f"[ROUTER]   Pattern: {pattern_str}")
            print(f"[ROUTER]   In question? {bool(re.search(pattern_str, search_scope))}")
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 2: FK-ONLY FILTER (Hapus tabel yang hanya disebutkan untuk FK)
    # ═══════════════════════════════════════════════════════════════════
    filtered_tables = []
    for tbl in relevant_tables:
        table_pattern = _table_entity_pattern(tbl)

        # Pattern: "<table> with <code/id> <angka/list>"
        fk_only_pattern = rf"\b{table_pattern}\b\s+with\s+(?:{id_terms_pattern})s?\s+{AGENT_ROUTER_FK_VALUE_LIST_PATTERN}"

        if re.search(fk_only_pattern, question):
            temp_scope = re.sub(fk_only_pattern, "", question)

            has_other_mention = re.search(
                rf"\b{table_pattern}\b(?!\s+(?:{id_terms_pattern}))",
                temp_scope
            )

            if not has_other_mention:
                continue
        
        filtered_tables.append(tbl)

    relevant_tables = filtered_tables

    # ═══════════════════════════════════════════════════════════════════
    # STEP 2.2: INCLUDE TABLES WITH IMPLICITLY MENTIONED COLUMNS (AGNOSTIC)
    # ═══════════════════════════════════════════════════════════════════
    # AGNOSTIC approach - works for ANY column type (numeric, string, date, etc.)
    # Uses retrieval results (ranked_cols) to find tables whose columns are
    # mentioned in the question, even when table names aren't explicitly stated.
    #
    # Examples:
    #   - "total amount" → finds tables with "amount" column
    #   - "customer names" → finds tables with "customerName" column
    #   - "order dates" → finds tables with "orderDate" column
    #
    # Only includes tables that have FK relationships with already-selected tables.

    ranked_cols = state.get("ranked_cols", [])
    if ranked_cols:
        # Find tables whose columns are mentioned in the question
        tables_with_implicit_cols = _find_tables_with_implicit_columns(
            question=question,
            ranked_cols=ranked_cols,
            relevant_tables=relevant_tables,
            ctx=ctx
        )

        if tables_with_implicit_cols:
            # Get the original table names (case-preserving) from context
            all_table_names = set(allowed_identifiers.keys())
            for tbl_lower in tables_with_implicit_cols:
                # Find the original table name with correct casing
                for original_tbl in all_table_names:
                    if original_tbl.lower() == tbl_lower:
                        relevant_lower = {t.lower() for t in relevant_tables}
                        if tbl_lower not in relevant_lower:
                            relevant_tables.append(original_tbl)
                            if RETRIEVE_DEBUG:
                                print(f"[ROUTER] Including {original_tbl} (implicit column mention)")

    # ═══════════════════════════════════════════════════════════════════
    # STEP 2.5:  DETECT SIGNALS EARLY (needed for FK inference)
    # ═══════════════════════════════════════════════════════════════════
    question_lower = question.lower()
    
    # Detect aggregation context
    math_keywords = AGENT_ROUTER_MATH_KEYWORDS
    has_math = any(re.search(rf"\b{re.escape(kw)}\b", question_lower) for kw in math_keywords)
    
    group_indicators = AGENT_ROUTER_GROUP_INDICATORS
    has_group_word = any(re.search(rf"\b{re.escape(kw)}\b", question_lower) for kw in group_indicators)
    
    is_aggregation = has_math or has_group_word
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 3: GRAPH-BASED TABLE SELECTION (Robust, Schema-Agnostic)
    # ═══════════════════════════════════════════════════════════════════
    # This approach uses graph theory to find the minimal set of tables needed:
    # 1. Build FK relationship graph
    # 2. Identify "evidence" tables (tables directly referenced in question)
    # 3. Find minimal connecting subgraph (shortest paths between evidence tables)
    # 4. Remove tables not in the connecting subgraph
    #
    # Advantages:
    # - Works for any schema (no hardcoded patterns)
    # - Based on actual FK relationships
    # - Finds minimal table set for valid joins
    # - Handles multi-hop joins correctly

    question_lower = question.lower()

    # --- STEP 3.1: Build FK Relationship Graph ---
    # Graph structure: adjacency list with bidirectional edges
    # fk_graph: {table: {neighbor: (fk_column, reverse_fk_column)}}
    fk_graph = {}
    fk_columns_by_table = {}  # {source_table: {fk_column: target_table}}

    # Pattern 1: Card format "table: X | column: Y | ... FK to Z.W"
    fk_pattern_card = re.compile(AGENT_ROUTER_FK_PATTERN_CARD, re.IGNORECASE)
    for match in fk_pattern_card.finditer(ctx):
        source_table = match.group(1).lower()
        source_col = match.group(2).lower()
        target_table = match.group(3).lower()

        # Add to fk_columns_by_table for reference
        if source_table not in fk_columns_by_table:
            fk_columns_by_table[source_table] = {}
        fk_columns_by_table[source_table][source_col] = target_table

        # Build bidirectional graph
        if source_table not in fk_graph:
            fk_graph[source_table] = {}
        if target_table not in fk_graph:
            fk_graph[target_table] = {}

        # Forward edge: source -> target with FK column
        fk_graph[source_table][target_table] = (source_col, None)
        # Reverse edge: target -> source (for path finding)
        fk_graph[target_table][source_table] = (None, source_col)

    # Pattern 2: Dot notation "X.Y ... FK to Z.W"
    fk_pattern_dot = re.compile(AGENT_ROUTER_FK_PATTERN_DOT, re.IGNORECASE)
    for match in fk_pattern_dot.finditer(ctx):
        source_table = match.group(1).lower()
        source_col = match.group(2).lower()
        target_table = match.group(3).lower()

        if source_table not in fk_columns_by_table:
            fk_columns_by_table[source_table] = {}
        fk_columns_by_table[source_table][source_col] = target_table

        if source_table not in fk_graph:
            fk_graph[source_table] = {}
        if target_table not in fk_graph:
            fk_graph[target_table] = {}

        fk_graph[source_table][target_table] = (source_col, None)
        fk_graph[target_table][source_table] = (None, source_col)

    # Pattern 3: Global schema FK metadata fallback (works even when cards are sparse)
    try:
        schema_fk = get_fk_relationships()
    except Exception:
        schema_fk = {}
    for source_table, fk_map in schema_fk.items():
        src = (source_table or "").lower()
        for source_col, target_table in (fk_map or {}).items():
            tgt = (target_table or "").lower()
            if not src or not tgt:
                continue

            if src not in fk_columns_by_table:
                fk_columns_by_table[src] = {}
            fk_columns_by_table[src][str(source_col).lower()] = tgt

            if src not in fk_graph:
                fk_graph[src] = {}
            if tgt not in fk_graph:
                fk_graph[tgt] = {}
            fk_graph[src][tgt] = (str(source_col).lower(), None)
            if src not in fk_graph[tgt]:
                fk_graph[tgt][src] = (None, str(source_col).lower())

    # --- STEP 3.2: Identify Seed Tables ---
    # Evidence tables are tables that have direct evidence from the question:
    # 1. Columns mentioned (including camelCase variants)
    # 2. "For each X" patterns
    # 3. Table name + attribute patterns ("employee name", "customer number")
    # 4. Aggregate keywords with table context

    evidence_tables = set()
    evidence_details = {}  # {table: [evidence_strings]}

    for tbl in relevant_tables:
        tbl_lower = tbl.lower()
        tbl_cols = allowed_identifiers.get(tbl, [])
        evidence = []

        # Check 1: Direct column mentions (all columns, including FKs)
        # FK columns are treated as WEAK evidence if they exist in other tables
        table_singular = get_singular_form(tbl_lower)
        table_entity_pattern = _table_entity_pattern(tbl_lower)
        weak_col = None
        strong_col = None
        for col in tbl_cols:
            col_lower = col.lower()
            if not _question_mentions_column(question_lower, col):
                continue

            # Check if this is an FK/bridge-like key reference:
            # 1) Conventional FK naming (<entity>Number/<entity>Code/<entity>Id)
            # 2) Explicit FK metadata from context (fk_columns_by_table)
            # 3) Parent key reused in child tables (e.g., productLine shared by products/productlines)
            is_fk_col = _is_conventional_fk_name(table_singular, col_lower)
            if col_lower in fk_columns_by_table.get(tbl_lower, {}):
                is_fk_col = True
            table_id_col = (get_id_column(tbl_lower) or "").lower()
            is_parent_key = bool(table_id_col and col_lower == table_id_col)
            matches_other_table_id = False
            for other_tbl in relevant_tables:
                if other_tbl.lower() == tbl_lower:
                    continue
                other_id_col = (get_id_column(other_tbl.lower()) or "").lower()
                if other_id_col and other_id_col == col_lower:
                    matches_other_table_id = True
                    break
            if matches_other_table_id and not is_parent_key:
                is_fk_col = True

            # Check if this column exists in ANOTHER table
            col_exists_elsewhere = False
            for other_tbl in relevant_tables:
                if other_tbl.lower() == tbl_lower:
                    continue
                other_cols = _get_table_cols(other_tbl)
                for other_col in other_cols:
                    if other_col.lower() == col_lower:
                        col_exists_elsewhere = True
                        break
                if col_exists_elsewhere:
                    break

            # FK-like columns should be weak evidence even when unique in schema
            # (e.g., customers.salesRepEmployeeNumber), because they often indicate
            # relational linkage rather than requested output attributes.
            is_weak_key_like = is_fk_col or (col_exists_elsewhere and is_parent_key)

            if is_weak_key_like:
                # Keep first weak match, but continue scanning for stronger
                # non-FK/non-key column evidence.
                if weak_col is None:
                    weak_col = col
            else:
                # Non-FK column → strong evidence
                strong_col = col
                break

        if strong_col:
            evidence.append(f"column '{strong_col}'")
        elif weak_col:
            evidence.append(f"FK column '{weak_col}'")

        # Check 2: "For each X" patterns
        # Handle both plural and singular, with optional words in between (e.g., "for each such customer")
        for_each_patterns = _instantiate_entity_patterns(
            AGENT_ROUTER_FOR_EACH_ENTITY_PATTERNS,
            table_entity_pattern,
        )
        if any(re.search(p, question_lower) for p in for_each_patterns):
            evidence.append(AGENT_ROUTER_EVIDENCE_TAG_FOR_EACH)

        # Check 3: Table name + attribute patterns
        table_patterns = _instantiate_entity_patterns(
            AGENT_ROUTER_TABLE_ATTRIBUTE_PATTERNS,
            table_entity_pattern,
        )
        if any(re.search(p, question_lower) for p in table_patterns):
            evidence.append(AGENT_ROUTER_EVIDENCE_TAG_TABLE_ATTRIBUTE)

        # Check 4: Aggregate keywords with table mention
        if has_math or has_group_word:
            # Patterns like "total orders", "count of customers", "number of orders", etc.
            # More flexible: allows extra words after table name (e.g., "orders placed")
            aggregate_patterns = _instantiate_entity_patterns(
                AGENT_ROUTER_AGGREGATE_CONTEXT_PATTERNS,
                table_entity_pattern,
            )
            if any(re.search(p, question_lower) for p in aggregate_patterns):
                evidence.append(AGENT_ROUTER_EVIDENCE_TAG_AGGREGATE_CONTEXT)

        if evidence:
            evidence_tables.add(tbl_lower)
            evidence_details[tbl_lower] = evidence
            if RETRIEVE_DEBUG:
                print(f"[ROUTER] Evidence table: {tbl} (evidence: {', '.join(evidence)})")

    # STEP 2.8: Remove parent tables that are only referenced by "for each" pattern
    # If a table is ONLY added because of "for each/per/each" pattern (no column mentions),
    # AND another table has the FK column linking to it, skip the parent table.
    # Example: "per customer" + payments.has(customerNumber) → skip customers table
    tables_to_remove = []
    try:
        schema_analyzer = get_schema_analyzer()
    except Exception:
        schema_analyzer = None
    for tbl_lower in list(evidence_tables):
        evidence_list = evidence_details.get(tbl_lower, [])
        weak_evidence = AGENT_ROUTER_WEAK_EVIDENCE_TAGS
        aggregate_or_weak_evidence = set(weak_evidence) | {AGENT_ROUTER_EVIDENCE_TAG_AGGREGATE_CONTEXT}

        # Preserve weak-evidence parent table when it is explicitly the
        # output subject in the question (e.g., "Show product lines ...")
        # and has table+attribute evidence.
        #
        # Exception:
        # If this "explicit subject" is only a base-entity token inside a
        # composite detail phrase (e.g., "order details") and another detail
        # sibling table has strong evidence, keep the detail table and allow
        # pruning of the weak parent to avoid unnecessary joins.
        table_entity_pattern = _table_entity_pattern(tbl_lower)
        has_table_attr_evidence = AGENT_ROUTER_EVIDENCE_TAG_TABLE_ATTRIBUTE in evidence_list
        explicit_subject_score = _explicit_subject_match_score(question_lower, table_entity_pattern)
        is_explicit_subject = has_table_attr_evidence and (explicit_subject_score > 0)
        if is_explicit_subject:
            detail_sibling_override = False
            for other_tbl in relevant_tables:
                other_lower = other_tbl.lower()
                if other_lower == tbl_lower:
                    continue
                if not _is_detail_child_entity(tbl_lower, other_lower):
                    continue
                other_entity_pattern = _table_entity_pattern(other_lower)
                if not other_entity_pattern or not re.search(rf"\b{other_entity_pattern}\b", question_lower):
                    continue

                other_evidence = evidence_details.get(other_lower, [])
                other_has_strong_evidence = any(
                    (e not in weak_evidence) and (not e.startswith(AGENT_ROUTER_FK_EVIDENCE_PREFIX))
                    for e in other_evidence
                )
                if other_has_strong_evidence:
                    detail_sibling_override = True
                    if RETRIEVE_DEBUG:
                        print(
                            f"[ROUTER] Explicit subject override: {tbl_lower} appears as base entity "
                            f"inside detail phrase; prefer strong detail table {other_lower}"
                        )
                    break

            if detail_sibling_override:
                # continue evaluating weak-evidence pruning below
                pass
            else:
                if RETRIEVE_DEBUG:
                    print(f"[ROUTER] Keeping {tbl_lower}: explicit subject entity in question")
                continue

        # Preserve weak tables that are explicitly the target of exclusion intent
        # (e.g., "customers with no orders", "never paid").
        if _question_targets_entity_exclusion(question_lower, tbl_lower):
            if RETRIEVE_DEBUG:
                print(f"[ROUTER] Keeping {tbl_lower}: exclusion target entity in question")
            continue

        # Check if evidence is ONLY weak patterns
        # Strong evidence: non-FK column mentions, aggregate context
        # Weak evidence: 'for each' pattern, table+attribute pattern, FK column mentions
        weak_evidence = AGENT_ROUTER_WEAK_EVIDENCE_TAGS
        # Also check for FK column evidence (format: "FK column 'X'")
        is_only_weak = all(e in weak_evidence or e.startswith(AGENT_ROUTER_FK_EVIDENCE_PREFIX) for e in evidence_list)
        has_fk_col_evidence = any(e.startswith(AGENT_ROUTER_FK_EVIDENCE_PREFIX) for e in evidence_list)
        has_non_id_entity_mention = bool(
            re.search(
                rf"\b{table_entity_pattern}\b(?!\s+(?:{id_terms_pattern})\b)",
                question_lower,
            )
        )
        relevant_lowers = {t.lower() for t in relevant_tables}
        neighbor_count_in_relevant = sum(
            1 for n in fk_graph.get(tbl_lower, {}) if n in relevant_lowers and n != tbl_lower
        )
        is_multi_hop_bridge = neighbor_count_in_relevant >= 2

        table_singular = get_singular_form(tbl_lower)
        table_id_col = (get_id_column(tbl_lower) or "").lower()
        fk_exists_in_other_table = False
        for other_tbl in relevant_tables:
            if other_tbl.lower() == tbl_lower:
                continue
            other_cols = _get_table_cols(other_tbl)
            for col in other_cols:
                col_lower = str(col).lower()
                if (table_id_col and col_lower == table_id_col) or _is_conventional_fk_name(table_singular, col_lower):
                    fk_exists_in_other_table = True
                    break
            if fk_exists_in_other_table:
                break

        tbl_cols = sorted(_get_table_cols(tbl_lower))
        fk_cols = set()
        for col in tbl_cols:
            col_lower = str(col).lower()
            if (table_id_col and col_lower == table_id_col) or _is_conventional_fk_name(table_singular, col_lower):
                fk_cols.add(col_lower)

        non_fk_mentioned = False
        for col in tbl_cols:
            col_low = str(col).lower()
            if col_low in fk_cols:
                continue
            spaced = re.sub(AGENT_COLUMN_CAMEL_SPLIT_LOWER_PATTERN, r'\1 \2', str(col)).lower()
            if col_low in question_lower or spaced in question_lower:
                non_fk_mentioned = True
                break

        if not non_fk_mentioned and schema_analyzer is not None:
            try:
                schema_cols = sorted(schema_analyzer.get_table_columns(tbl_lower))
            except Exception:
                schema_cols = []
            for col in schema_cols:
                col_low = str(col).lower()
                if col_low in fk_cols:
                    continue
                if _question_mentions_column(question_lower, str(col)):
                    non_fk_mentioned = True
                    break

        if not non_fk_mentioned:
            name_col = (get_name_column(tbl_lower) or "").strip()
            if (
                name_col
                and str(name_col).lower() not in fk_cols
                and _question_mentions_entity_display_name(question_lower, tbl_lower)
            ):
                non_fk_mentioned = True

        # If the entity is explicitly mentioned in non-ID form and evidence comes
        # from FK columns, keep it only when:
        # 1) it is a true bridge node between at least two relevant tables, AND
        # 2) the question also asks for at least one non-FK attribute from that
        #    table. This prevents unnecessary parent-table joins in single-table
        #    aggregates/window queries like payments/customer where only the FK
        #    key is needed, but still preserves cases that truly need attributes
        #    like customerName, city, etc.
        if has_fk_col_evidence and has_non_id_entity_mention and is_multi_hop_bridge and non_fk_mentioned:
            if RETRIEVE_DEBUG:
                print(f"[ROUTER] Keeping {tbl_lower}: explicit entity mention + FK evidence")
            continue

        # Aggregate/count mentions of a leaf parent entity (e.g., "distinct products")
        # can often be satisfied via a child FK table without joining the parent table,
        # as long as no non-key parent attributes are requested.
        is_aggregate_like_only = (
            AGENT_ROUTER_EVIDENCE_TAG_AGGREGATE_CONTEXT in evidence_list
            and all(
                e in aggregate_or_weak_evidence or e.startswith(AGENT_ROUTER_FK_EVIDENCE_PREFIX)
                for e in evidence_list
            )
        )
        if (
            is_aggregate_like_only
            and fk_exists_in_other_table
            and not non_fk_mentioned
            and neighbor_count_in_relevant <= 1
        ):
            covering_child_lowers: List[str] = []
            for other_tbl in relevant_tables:
                other_lower = other_tbl.lower()
                if other_lower == tbl_lower:
                    continue
                for col in _get_table_cols(other_tbl):
                    col_lower = str(col).lower()
                    if (table_id_col and col_lower == table_id_col) or _is_conventional_fk_name(table_singular, col_lower):
                        covering_child_lowers.append(other_lower)
                        break

            for child_lower in dict.fromkeys(covering_child_lowers):
                evidence_tables.add(child_lower)
                child_evidence = evidence_details.setdefault(child_lower, [])
                if AGENT_ROUTER_EVIDENCE_TAG_AGGREGATE_CONTEXT not in child_evidence:
                    child_evidence.append(AGENT_ROUTER_EVIDENCE_TAG_AGGREGATE_CONTEXT)
                if RETRIEVE_DEBUG:
                    print(
                        f"[ROUTER] Promoting {child_lower}: aggregate support for {tbl_lower}"
                    )
            tables_to_remove.append(tbl_lower)
            if RETRIEVE_DEBUG:
                print(
                    f"[ROUTER] Skipping {tbl_lower}: aggregate-only leaf entity "
                    f"covered by child FK table"
                )
            continue

        if is_only_weak and evidence_list:  # Has evidence, but all are weak
            if fk_exists_in_other_table:
                if not non_fk_mentioned:
                    # Safe to remove - only referenced via "for each" and FK exists elsewhere
                    tables_to_remove.append(tbl_lower)
                    if RETRIEVE_DEBUG:
                        print(f"[ROUTER] Skipping {tbl_lower}: only 'for each' ref, FK exists in other table")
            else:
                # Fallback: remove weak-only table if another table has strong evidence
                # and shares the weak FK column(s). This handles key-sharing schemas like
                # products.productLine <-> productlines.productLine even without explicit FK cards.
                weak_fk_cols = set()
                for ev in evidence_list:
                    m_fk = re.match(AGENT_ROUTER_FK_EVIDENCE_EXTRACT_PATTERN, ev)
                    if m_fk:
                        weak_fk_cols.add(m_fk.group(1).lower())

                if weak_fk_cols:
                    has_strong_other = False
                    for other_tbl in relevant_tables:
                        other_lower = other_tbl.lower()
                        if other_lower == tbl_lower:
                            continue
                        other_evidence = evidence_details.get(other_lower, [])
                        if any((e not in weak_evidence) and (not e.startswith(AGENT_ROUTER_FK_EVIDENCE_PREFIX)) for e in other_evidence):
                            has_strong_other = True
                            break

                    if has_strong_other:
                        shared_with_other = False
                        for other_tbl in relevant_tables:
                            other_lower = other_tbl.lower()
                            if other_lower == tbl_lower:
                                continue
                            other_cols = _get_table_cols(other_tbl)
                            if weak_fk_cols & other_cols:
                                shared_with_other = True
                                break

                        if shared_with_other:
                            tables_to_remove.append(tbl_lower)
                            if RETRIEVE_DEBUG:
                                print(f"[ROUTER] Skipping {tbl_lower}: weak-only FK evidence, strong sibling table covers shared key")

    # Remove identified tables
    for tbl_lower in tables_to_remove:
        evidence_tables.discard(tbl_lower)
        evidence_details.pop(tbl_lower, None)
        # Also remove from relevant_tables
        tbl_original = None
        for t in relevant_tables:
            if t.lower() == tbl_lower:
                tbl_original = t
                break
        if tbl_original and tbl_original in relevant_tables:
            relevant_tables.remove(tbl_original)

    # --- STEP 3.3: Find Minimal Connecting Subgraph ---
    # Use BFS (Breadth-First Search) to find shortest paths between all pairs of evidence tables
    # BFS = explores all neighbors at current depth before moving deeper
    # The union of all paths = minimal table set needed for valid JOINs

    def bfs_shortest_path(start, end, graph):
        """
        Find shortest path between start and end using BFS (Breadth-First Search).

        BFS explores graph level by level, guaranteeing the shortest path in unweighted graphs.
        Used here to find minimal JOIN path between two evidence tables via FK relationships.
        """
        if start == end:
            return [start]

        queue = [(start, [start])]
        visited = {start}

        while queue:
            node, path = queue.pop(0)

            for neighbor in graph.get(node, {}):
                if neighbor == end:
                    return path + [neighbor]

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None  # No path found

    # Collect all tables in the minimal connecting subgraph
    tables_in_subgraph = set(evidence_tables)

    if len(evidence_tables) >= 2:
        # Find paths between all pairs of evidence tables
        evidence_list = list(evidence_tables)
        for i, start in enumerate(evidence_list):
            for end in evidence_list[i+1:]:
                path = bfs_shortest_path(start, end, fk_graph)
                if path:
                    tables_in_subgraph.update(path)
                # Path removed: not passed to LLM, only internal use

    # Add missing connector tables from the minimal FK subgraph.
    # Only add true connectors (degree >= 2 within the subgraph), not leaves.
    if tables_in_subgraph:
        relevant_lowers = {t.lower() for t in relevant_tables}
        original_tables = list(allowed_identifiers.keys())
        for tbl_lower in sorted(tables_in_subgraph):
            if tbl_lower in relevant_lowers:
                continue
            degree_in_subgraph = sum(
                1 for n in fk_graph.get(tbl_lower, {}) if n in tables_in_subgraph and n != tbl_lower
            )
            if degree_in_subgraph < 2:
                continue
            tbl_original = next((t for t in original_tables if t.lower() == tbl_lower), tbl_lower)
            relevant_tables.append(tbl_original)
            relevant_lowers.add(tbl_lower)
            if RETRIEVE_DEBUG:
                print(f"[ROUTER] Including {tbl_original}: connector in minimal subgraph")

    # --- STEP 3.4: Remove Tables Not in Connecting Subgraph ---
    # Keep tables that are:
    # 1. Seed tables (direct evidence)
    # 2. In the minimal connecting subgraph (needed for joins)
    # 3. NOT in the subgraph but might be needed for aggregation (edge case)

    # Temporal intent guardrail (schema-driven + config-driven):
    # keep date-bearing tables when question expresses temporal ranking/filtering.
    def _is_temporal_col_name(col_name: str) -> bool:
        c = str(col_name or "").lower()
        return any(k in c for k in ("date", "time", "timestamp", "year", "month", "day"))

    # Derive temporal unit hints from schema columns (agnostic across domains).
    # Generic temporal units are domain-agnostic and common in NL prompts.
    temporal_unit_terms = {"year", "month", "day", "date", "time"}
    temporal_cols_by_table = {}
    for _tbl in whitelisted_tables:
        t_cols = []
        for c in _get_table_cols(_tbl):
            if _is_temporal_col_name(c):
                t_cols.append(str(c))
        if t_cols:
            temporal_cols_by_table[_tbl.lower()] = t_cols
            for c in t_cols:
                cl = c.lower()
                if "date" in cl:
                    temporal_unit_terms.add("date")
                if "time" in cl:
                    temporal_unit_terms.add("time")
                if "timestamp" in cl:
                    temporal_unit_terms.add("timestamp")
                if "year" in cl:
                    temporal_unit_terms.add("year")
                if "month" in cl:
                    temporal_unit_terms.add("month")
                if "day" in cl:
                    temporal_unit_terms.add("day")

    # Config-driven temporal ranking terms (latest/newest/earliest, etc.).
    has_temporal_rank_intent = any(
        re.search(rf"\b{re.escape(str(t).lower())}\b", question_lower)
        for t in AGENT_ROUTER_TIME_RANK_TERMS
        if str(t or "").strip()
    )
    has_temporal_unit_mention = any(
        re.search(rf"\b{re.escape(t)}\b", question_lower)
        for t in temporal_unit_terms
    )
    has_temporal_col_mention = False
    for _tbl, cols in temporal_cols_by_table.items():
        if any(_question_mentions_column(question_lower, c) for c in cols):
            has_temporal_col_mention = True
            break
    has_four_digit_year = bool(re.search(r"\b(?:19|20)\d{2}\b", question_lower))

    has_temporal_intent = (
        has_temporal_rank_intent
        or has_temporal_unit_mention
        or has_temporal_col_mention
        or has_four_digit_year
    )

    def _table_has_temporal_column(tbl_name: str) -> bool:
        tbl_l = (tbl_name or "").lower()
        if tbl_l in temporal_cols_by_table:
            return True
        cols = _get_table_cols(tbl_name)
        return any(_is_temporal_col_name(c) for c in cols)

    def _question_mentions_table_entity(question_text: str, tbl_name: str) -> bool:
        tbl_lower = (tbl_name or "").lower()
        pat = _table_entity_pattern(tbl_lower)
        if not pat:
            return False

        table_singular = get_singular_form(tbl_lower)
        search_scope = question_text
        for ignore_pat in AGENT_ROUTER_IGNORE_PATTERNS.get((table_singular or "").lower(), []):
            search_scope = re.sub(ignore_pat, " ", search_scope)

        if re.search(rf"\b{pat}\b", search_scope):
            return True

        singular_token = (table_singular or "").strip().lower()
        if singular_token and singular_token.isalpha() and len(singular_token) >= 4:
            if re.search(rf"\b{re.escape(singular_token)}(?:ed|ing)\b", search_scope):
                return True

        return False

    def _question_mentions_table_temporal_column(question_text: str, tbl_name: str) -> bool:
        tbl_l = (tbl_name or "").lower()
        for c in temporal_cols_by_table.get(tbl_l, _get_table_cols(tbl_name)):
            if not _is_temporal_col_name(c):
                continue
            if _question_mentions_column(question_text, str(c)):
                return True
        return False

    temporal_column_mentioned = any(
        _question_mentions_table_temporal_column(question_lower, tbl)
        for tbl in relevant_tables
    )
    current_temporal_tables = [
        tbl for tbl in relevant_tables
        if _table_has_temporal_column(tbl)
    ]

    # Generic temporal bridge enrichment:
    # if the question clearly expresses a year/date intent but none of the
    # currently kept tables expose temporal columns, pull in a uniquely
    # connected temporal table so the LLM has the right date source.
    # This stays schema-agnostic and avoids per-question/per-table hardcoding.
    if has_temporal_intent and len(relevant_tables) >= 2 and not current_temporal_tables:
        relevant_lowers_now = {t.lower() for t in relevant_tables}
        temporal_connector_candidates: List[str] = []
        for tbl in allowed_identifiers.keys():
            tbl_lower = str(tbl or "").lower()
            if not tbl_lower or tbl_lower in relevant_lowers_now:
                continue
            if not _table_has_temporal_column(tbl):
                continue

            directly_connected = any(
                tbl_lower in fk_graph.get(rel_lower, {})
                or rel_lower in fk_graph.get(tbl_lower, {})
                for rel_lower in relevant_lowers_now
            )
            if not directly_connected:
                continue

            temporal_connector_candidates.append(tbl)

        if len(temporal_connector_candidates) == 1:
            temporal_tbl = temporal_connector_candidates[0]
            relevant_tables.append(temporal_tbl)
            if RETRIEVE_DEBUG:
                print(
                    f"[ROUTER] Including {temporal_tbl}: unique connected temporal table "
                    f"for time/year intent"
                )

    tables_to_remove = []
    for tbl in relevant_tables:
        tbl_lower = tbl.lower()

        # Keep if it's an evidence table
        if tbl_lower in evidence_tables:
            if RETRIEVE_DEBUG:
                print(f"[ROUTER] Keeping {tbl}: evidence table")
            continue

        # Keep if it's in the connecting subgraph
        if tbl_lower in tables_in_subgraph:
            if RETRIEVE_DEBUG:
                print(f"[ROUTER] Keeping {tbl}: in connecting subgraph")
            continue

        # Keep if it has outgoing FKs to an evidence table (might be needed for query)
        has_fk_to_evidence = False
        for target in fk_graph.get(tbl_lower, {}):
            if target in evidence_tables:
                has_fk_to_evidence = True
                break

        if has_fk_to_evidence:
            # This table references an evidence table via FK
            # Check if we need any non-FK columns from it
            non_fk_cols = [
                c for c in _get_table_cols(tbl)
                if c.lower() not in fk_columns_by_table.get(tbl_lower, {})
            ]
            needs_non_fk = False
            for col in non_fk_cols:
                col_variants = [col.lower()]
                spaced = re.sub(AGENT_COLUMN_CAMEL_SPLIT_LOWER_PATTERN, r'\1 \2', col).lower()
                if spaced != col.lower():
                    col_variants.append(spaced)
                if any(v in question_lower for v in col_variants):
                    needs_non_fk = True
                    break

            if needs_non_fk:
                if RETRIEVE_DEBUG:
                    print(f"[ROUTER] Keeping {tbl}: has FK to evidence table + needs non-FK columns")
                continue

        # Temporal recency/year/date questions may need a date-bearing table
        # that is not part of the minimal FK connector among seed tables.
        if has_temporal_intent and _table_has_temporal_column(tbl):
            temporal_bridge_to_evidence = bool(
                any(target in evidence_tables for target in fk_graph.get(tbl_lower, {}))
                or any(tbl_lower in fk_graph.get(src, {}) for src in evidence_tables)
            )
            if (
                _question_mentions_table_entity(question_lower, tbl)
                or _question_mentions_table_temporal_column(question_lower, tbl)
                or (
                    temporal_bridge_to_evidence
                    and not temporal_column_mentioned
                )
            ):
                if RETRIEVE_DEBUG:
                    print(f"[ROUTER] Keeping {tbl}: temporal intent + date-bearing table")
                continue

        # Mark for removal
        tables_to_remove.append(tbl)
        if RETRIEVE_DEBUG:
            print(f"[ROUTER] Removing {tbl}: not in minimal subgraph")

    # Apply removals
    for tbl in tables_to_remove:
        if tbl in relevant_tables:
            relevant_tables.remove(tbl)

    # SAFETY: Never leave relevant_tables empty - fallback to first whitelisted table
    if not relevant_tables and whitelisted_tables:
        fallback_table = list(whitelisted_tables)[0]
        relevant_tables.append(fallback_table)
        if RETRIEVE_DEBUG:
            print(f"[ROUTER] All tables removed, falling back to: {fallback_table}")

    # --- STEP 3.45: Ensure JOIN Connectivity via FK Bridge Tables ---
    # If multiple kept tables are not directly connected, include minimal bridge
    # tables from FK shortest paths so SQL generation has a valid join chain.
    # Kept conservative: only for aggregation/temporal-intent workloads.
    if len(relevant_tables) >= 2 and (is_aggregation or has_temporal_intent):
        relevant_lower_now = [t.lower() for t in relevant_tables]
        bridge_lowers = set()

        for i, start in enumerate(relevant_lower_now):
            for end in relevant_lower_now[i + 1:]:
                path = bfs_shortest_path(start, end, fk_graph)
                if not path:
                    continue
                for node in path:
                    if node not in relevant_lower_now:
                        bridge_lowers.add(node)

        if bridge_lowers:
            lower_to_original = {t.lower(): t for t in allowed_identifiers.keys()}
            for node in sorted(bridge_lowers):
                original = lower_to_original.get(node, node)
                if original not in relevant_tables:
                    relevant_tables.append(original)
                    if RETRIEVE_DEBUG:
                        print(f"[ROUTER] Keeping {original}: bridge table for FK connectivity")

    # FINAL NORMALIZATION: drop redundant aggregate-only leaf entities that are
    # covered by child FK tables, then restore any missing bridge connectors.
    if len(relevant_tables) >= 2:
        final_drop: List[str] = []
        weak_evidence = AGENT_ROUTER_WEAK_EVIDENCE_TAGS
        aggregate_or_weak_evidence = set(weak_evidence) | {AGENT_ROUTER_EVIDENCE_TAG_AGGREGATE_CONTEXT}
        lower_to_original = {t.lower(): t for t in allowed_identifiers.keys()}

        for tbl in list(relevant_tables):
            tbl_lower = tbl.lower()
            evidence_list = evidence_details.get(tbl_lower, [])
            if not evidence_list:
                continue

            is_aggregate_like_only = (
                AGENT_ROUTER_EVIDENCE_TAG_AGGREGATE_CONTEXT in evidence_list
                and all(
                    e in aggregate_or_weak_evidence or e.startswith(AGENT_ROUTER_FK_EVIDENCE_PREFIX)
                    for e in evidence_list
                )
            )
            if not is_aggregate_like_only:
                continue

            table_singular = get_singular_form(tbl_lower)
            table_id_col = (get_id_column(tbl_lower) or "").lower()
            fk_exists_in_other_table = False
            for other_tbl in relevant_tables:
                if other_tbl.lower() == tbl_lower:
                    continue
                for col in _get_table_cols(other_tbl):
                    col_lower = str(col).lower()
                    if (table_id_col and col_lower == table_id_col) or _is_conventional_fk_name(table_singular, col_lower):
                        fk_exists_in_other_table = True
                        break
                if fk_exists_in_other_table:
                    break
            if not fk_exists_in_other_table:
                continue

            tbl_cols = sorted(_get_table_cols(tbl_lower))
            fk_cols = set()
            for col in tbl_cols:
                col_lower = str(col).lower()
                if (table_id_col and col_lower == table_id_col) or _is_conventional_fk_name(table_singular, col_lower):
                    fk_cols.add(col_lower)

            non_fk_mentioned = False
            for col in tbl_cols:
                col_low = str(col).lower()
                if col_low in fk_cols:
                    continue
                spaced = re.sub(AGENT_COLUMN_CAMEL_SPLIT_LOWER_PATTERN, r'\1 \2', str(col)).lower()
                if col_low in question_lower or spaced in question_lower:
                    non_fk_mentioned = True
                    break
            if non_fk_mentioned:
                continue

            relevant_lowers_now = {t.lower() for t in relevant_tables}
            degree_now = sum(
                1 for n in fk_graph.get(tbl_lower, {}) if n in relevant_lowers_now and n != tbl_lower
            )
            if degree_now > 1:
                continue

            final_drop.append(tbl)

        if final_drop:
            for tbl in final_drop:
                if tbl in relevant_tables:
                    relevant_tables.remove(tbl)
                    if RETRIEVE_DEBUG:
                        print(f"[ROUTER] Final cleanup: removing {tbl} (redundant aggregate-only leaf)")

            if len(relevant_tables) >= 2:
                relevant_lower_now = [t.lower() for t in relevant_tables]
                bridge_lowers = set()
                for i, start in enumerate(relevant_lower_now):
                    for end in relevant_lower_now[i + 1:]:
                        path = bfs_shortest_path(start, end, fk_graph)
                        if not path:
                            continue
                        for node in path:
                            if node not in relevant_lower_now:
                                bridge_lowers.add(node)

                for node in sorted(bridge_lowers):
                    original = lower_to_original.get(node, node)
                    if original not in relevant_tables:
                        relevant_tables.append(original)
                        if RETRIEVE_DEBUG:
                            print(f"[ROUTER] Final cleanup: adding {original} (bridge table)")

    # ═══════════════════════════════════════════════════════════════════
    # STEP 4: DETEKSI SINYAL (Complete Detection)
    # ═══════════════════════════════════════════════════════════════════
    # Note: "rank/ranking" NOT included here - explicit ranking with RANK() OVER 
    # is simpler than winner-row patterns (highest/lowest/top) which need subqueries
    value_ranks = AGENT_ROUTER_VALUE_RANK_TERMS
    time_ranks = AGENT_ROUTER_TIME_RANK_TERMS

    # "at least/at most" are quantity filters, not ranking intent.
    # Keep ranking terms, but suppress false positives for these quantifiers.
    rank_hits: List[str] = []
    for kw in value_ranks:
        kw_l = str(kw or "").strip().lower()
        if not kw_l:
            continue
        for m in re.finditer(rf"\b{re.escape(kw_l)}\b", question_lower):
            if kw_l in {"least", "most"}:
                left = question_lower[max(0, m.start() - 8):m.start()]
                if re.search(r"\bat\s+$", left):
                    # e.g. "at least once", "at most 10"
                    continue
            rank_hits.append(kw_l)
            break

    is_ranking_value = bool(rank_hits)
    has_percentage_family = bool(
        re.search(r"\b(percent|percentage|ratio|rate|share)\b", question_lower, flags=re.I)
    )
    percentage_ordering_only = bool(
        has_percentage_family
        and re.search(
            r"\b(?:highest|lowest|largest|smallest|most|least)\s+"
            r"(?:[a-z][a-z0-9_]*\s+){0,2}"
            r"(?:percentage|percent|ratio|rate|share)\b",
            question_lower,
            flags=re.I,
        )
        and not re.search(AGENT_ROUTER_WINNER_ROW_RECORD_PATTERN, question_lower)
        and not re.search(AGENT_ROUTER_WINNER_ROW_FOR_EACH_PATTERN, question_lower)
        and not re.search(AGENT_ROUTER_TIEBREAKER_PATTERN, question_lower)
    )
    if percentage_ordering_only:
        is_ranking_value = False
    is_ranking_time = any(re.search(rf"\b{re.escape(kw)}\b", question_lower) for kw in time_ranks)
    is_ranking = is_ranking_value or is_ranking_time
    
    # EXPLICIT RANK FUNCTION DETECTION: "a rank of X" / "rank employees" / "ranking"
    # This means user explicitly wants RANK() OVER window function
    # E.g., "return a rank of employees within each job title"
    explicit_rank_patterns = AGENT_ROUTER_EXPLICIT_RANK_PATTERNS
    is_explicit_rank_function = any(re.search(pat, question_lower) for pat in explicit_rank_patterns)
    
    # ORDINAL RANKING DETECTION: "second highest", "third largest", "2nd lowest", etc.
    # Also cover row-selection phrasings like "second payment row ... sorted by amount".
    # These patterns ALWAYS need ROW_NUMBER() with rn = N, much more complex than simple MAX/MIN
    ordinal_words = AGENT_ROUTER_ORDINAL_WORDS
    ordinal_targets = AGENT_ROUTER_ORDINAL_TARGET_TERMS
    ordinal_pattern = rf"\b({'|'.join(re.escape(w) for w in ordinal_words)})(?:\s+|-)+({'|'.join(re.escape(w) for w in ordinal_targets)})\b"
    ordinal_row_pattern = rf"\b({'|'.join(re.escape(w) for w in ordinal_words)})\b(?:\s+\w+){{0,8}}\s+(?:row|record|entry)\b"
    has_explicit_sort_sequence = bool(
        re.search(
            r"\b(?:sort(?:ed|ing)?|order(?:ed|ing)?)\b[^.]{0,96}\bby\b",
            question_lower,
            flags=re.I,
        )
    )
    is_ordinal_ranking = bool(
        re.search(ordinal_pattern, question_lower)
        or (re.search(ordinal_row_pattern, question_lower) and has_explicit_sort_sequence)
    )
    if not is_ordinal_ranking:
        is_ordinal_ranking = bool(
            re.search(
                r"\b(?:first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th)\b"
                r"[^.]{0,40}\b(?:row|record|entry)\b",
                question_lower,
                flags=re.I,
            )
            and has_explicit_sort_sequence
        )
    
    # WINDOW AGGREGATE DETECTION: "within the same X" / "in the same X" pattern
    # This pattern requires COUNT(*) OVER (PARTITION BY X), NOT GROUP BY
    # E.g., "total employees within the same office" → PARTITION BY officeCode
    within_same_match = re.search(
        r"\b(?:within|in)\s+the\s+same\s+([a-z][a-z0-9_ ]{1,40}?)(?=\s+\b(?:when|where|after|before|ordered|sorted|with|and|then|that|which|who|whose)\b|[.,;]|$)",
        question_lower,
        flags=re.I,
    )
    if not within_same_match:
        within_same_pattern = AGENT_ROUTER_WITHIN_SAME_PATTERN
        within_same_match = re.search(within_same_pattern, question_lower)
    if not within_same_match:
        same_group_fallback = re.search(
            r"\b(?:that\s+same|the\s+same|same)\s+([a-z][a-z0-9_ ]{1,40}?)(?=\s+\b(?:when|where|after|before|ordered|sorted|with|and|then|that|which|who|whose)\b|[.,;]|$)",
            question_lower,
            flags=re.I,
        )
        has_window_agg_cue = bool(
            re.search(r"\b(?:how\s+many|count|number\s+of|total|average|avg|mean)\b", question_lower, flags=re.I)
        )
        if same_group_fallback and has_window_agg_cue:
            within_same_match = same_group_fallback
    is_window_aggregate = bool(within_same_match)
    partition_hint = None
    if within_same_match:
        partition_hint = _extract_same_group_phrase(question_lower)
        if not partition_hint:
            try:
                partition_hint = within_same_match.group(2)
            except Exception:
                partition_hint = None
            if not partition_hint:
                try:
                    partition_hint = within_same_match.group(1)
                except Exception:
                    partition_hint = None
    
    explicit_sort = AGENT_ROUTER_EXPLICIT_SORT_TERMS
    is_sorting = any(re.search(rf"\b{re.escape(kw)}\b", question_lower) for kw in explicit_sort) or is_ranking
    
    is_by_logic = (
        AGENT_ROUTER_BY_TOKEN in question_lower
        and not any(re.search(rf"\b{re.escape(s)}\b", question_lower) for s in explicit_sort)
        and (has_math or is_ranking)
    )
    has_grouping = has_group_word or is_by_logic

    # ═══════════════════════════════════════════════════════════════════
    # STEP 4.5: PRIMARY TABLE DETERMINATION
    # ═══════════════════════════════════════════════════════════════════
    # Determine which table should be the FROM table (primary table)
    # This provides guidance to LLM for SQL generation
    #
    # Priority rules:
    # 0. Explicit lead subject ("Show/List/Return X ..."), most specific match wins
    # 1. "For each X" → X is primary table (GROUP BY context)
    # 2. "Show X from Y" / "X in Y" → Y is primary table
    # 3. "All X that Y" → X is primary table (filtering on X)
    # 4. Table with most direct evidence (column mentions)
    # 5. First table in list (fallback)

    primary_table = None
    primary_reason = None

    if len(relevant_tables) == 1:
        # Single table - easy decision
        primary_table = relevant_tables[0]
        primary_reason = "single table"
    else:
        # Multiple tables - apply priority rules

        # Rule 0: explicit subject entity at question lead (config-driven).
        # This reduces false anchors like "products" when prompt starts with
        # a more specific entity such as "product lines".
        explicit_candidates = []
        for tbl in relevant_tables:
            tbl_lower = tbl.lower()
            evidence_list = evidence_details.get(tbl_lower, [])
            has_table_attr_evidence = AGENT_ROUTER_EVIDENCE_TAG_TABLE_ATTRIBUTE in evidence_list
            if not has_table_attr_evidence:
                continue
            score = _explicit_subject_match_score(
                question_lower,
                _table_entity_pattern(tbl_lower),
            )
            if score <= 0:
                continue
            explicit_candidates.append((score, len(evidence_list), tbl))

        if explicit_candidates:
            explicit_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
            top_score = explicit_candidates[0][0]
            top_scored = [c for c in explicit_candidates if c[0] == top_score]
            if len(top_scored) == 1:
                primary_table = top_scored[0][2]
                primary_reason = AGENT_ROUTER_PRIMARY_REASON_EXPLICIT_SUBJECT
            else:
                top_evidence = max(c[1] for c in top_scored)
                top_evidence_scored = [c for c in top_scored if c[1] == top_evidence]
                if len(top_evidence_scored) == 1:
                    primary_table = top_evidence_scored[0][2]
                    primary_reason = AGENT_ROUTER_PRIMARY_REASON_EXPLICIT_SUBJECT

        # Rule 1: "For each X" pattern (highest priority)
        # Handle both plural and singular, with optional words in between
        if not primary_table:
            for tbl in relevant_tables:
                tbl_lower = tbl.lower()
                table_entity_pattern = _table_entity_pattern(tbl_lower)
                for_each_patterns = _instantiate_entity_patterns(
                    AGENT_ROUTER_FOR_EACH_ENTITY_PATTERNS,
                    table_entity_pattern,
                )
                if any(re.search(p, question_lower) for p in for_each_patterns):
                    primary_table = tbl
                    primary_reason = AGENT_ROUTER_EVIDENCE_TAG_FOR_EACH
                    break

        # Rule 2: Not yet found, check "show X from Y" / "X in Y" patterns
        if not primary_table:
            for tbl in relevant_tables:
                tbl_lower = tbl.lower()
                table_entity_pattern = _table_entity_pattern(tbl_lower)
                # Patterns: "customers who placed", "offices in", "products from"
                from_in_patterns = _instantiate_entity_patterns(
                    AGENT_ROUTER_PRIMARY_FROM_IN_PATTERNS,
                    table_entity_pattern,
                )
                if any(re.search(p, question_lower) for p in from_in_patterns):
                    primary_table = tbl
                    primary_reason = AGENT_ROUTER_PRIMARY_REASON_FROM_IN
                    break

        # Rule 3: Not yet found, check "all X that Y" pattern
        if not primary_table:
            for tbl in relevant_tables:
                table_entity_pattern = _table_entity_pattern(tbl.lower())
                all_x_pattern = AGENT_ROUTER_PRIMARY_ALL_ENTITY_PATTERN.replace(
                    "__ENTITY__", table_entity_pattern
                ).replace("{entity}", table_entity_pattern)
                if re.search(all_x_pattern, question_lower):
                    primary_table = tbl
                    primary_reason = AGENT_ROUTER_PRIMARY_REASON_ALL
                    break

        # Rule 4: Not yet found, use table with most evidence
        if not primary_table:
            # Use the evidence_tables set from earlier (tables with direct evidence)
            best_score = -1
            for tbl in relevant_tables:
                tbl_lower = tbl.lower()
                if tbl_lower in evidence_details:
                    score = len(evidence_details[tbl_lower])
                    if score > best_score:
                        best_score = score
                        primary_table = tbl
                        primary_reason = f"most evidence ({score} factors)"

        # Rule 5: Fallback - first table
        if not primary_table:
            primary_table = relevant_tables[0]
            primary_reason = "first table (fallback)"

    if RETRIEVE_DEBUG:
        print(f"[ROUTER] Primary Table: {primary_table} (reason: {primary_reason})")

    def _detect_router_exclusion(is_multi_table_context: bool) -> bool:
        # Exclusion detection - only trigger for TRUE exclusion patterns.
        # These patterns imply NOT EXISTS / NOT IN / LEFT JOIN ... IS NULL (cross-table exclusion)
        # NOT triggered by: "missing" in single-table context (just IS NULL check)
        cross_table_exclusion = _active_router_cross_table_exclusion_terms(question_lower)
        if any(_has_router_phrase_term(question_lower, kw) for kw in cross_table_exclusion):
            return True

        # "no" + entity noun (dynamic, schema-driven) implies cross-table exclusion.
        entity_terms = set()
        for tbl in whitelisted_tables:
            for v in _table_name_variants(tbl):
                entity_terms.add(re.escape(v).replace(r"\ ", r"\s+"))

        if entity_terms:
            entity_pattern = "|".join(sorted(entity_terms, key=len, reverse=True))
            if re.search(rf"\bno\s+(?:{entity_pattern})\b", question_lower):
                return True

        # "missing" or "empty" only implies exclusion in multi-table context.
        return bool(
            is_multi_table_context
            and re.search(AGENT_ROUTER_MISSING_EMPTY_PATTERN, question_lower)
        )

    has_exclusion = _detect_router_exclusion(len(relevant_tables) > 1)

    # Subquery intent cues (e.g., compare against average/overall baseline, EXISTS wording).
    # This is intentionally lightweight and only used for escalation policy.
    has_global_avg_baseline = _question_has_global_avg_baseline_compare(question_lower)
    is_subquery_intent = bool(
        re.search(AGENT_ROUTER_SUBQUERY_INTENT_PATTERN, question_lower)
        or has_global_avg_baseline
    )
    has_complex_subquery_cue = bool(
        re.search(AGENT_ROUTER_COMPLEX_SUBQUERY_CUE_PATTERN, question_lower)
    )
    has_exists_keyword = bool(re.search(r"\b(?:exists|not\s+exists)\b", question_lower))

    compact_single_table_family = bool(
        primary_table
        and len(relevant_tables) > 1
        and (
            is_explicit_rank_function
            or is_window_aggregate
            or is_ordinal_ranking
            or percentage_ordering_only
        )
    )
    if compact_single_table_family:
        weak_evidence = AGENT_ROUTER_WEAK_EVIDENCE_TAGS
        retained_tables: List[str] = [primary_table]

        for tbl in relevant_tables:
            if tbl == primary_table:
                continue

            tbl_lower = tbl.lower()
            evidence_list = evidence_details.get(tbl_lower, [])
            evidence_is_compact_safe = bool(evidence_list) and all(
                e in weak_evidence
                or e == AGENT_ROUTER_EVIDENCE_TAG_AGGREGATE_CONTEXT
                or e.startswith(AGENT_ROUTER_FK_EVIDENCE_PREFIX)
                for e in evidence_list
            )
            if not evidence_is_compact_safe:
                retained_tables.append(tbl)
                continue

            if _question_mentions_table_entity(question_lower, tbl):
                retained_tables.append(tbl)
                continue

            table_id_col = (get_id_column(tbl_lower) or "").lower()
            fk_cols = set()
            for col in _get_table_cols(tbl):
                col_lower = str(col).lower()
                if (table_id_col and col_lower == table_id_col) or _is_conventional_fk_name(get_singular_form(tbl_lower), col_lower):
                    fk_cols.add(col_lower)

            non_fk_mentioned = False
            for col in _get_table_cols(tbl):
                col_lower = str(col).lower()
                if col_lower in fk_cols:
                    continue
                if _question_mentions_column(question_lower, str(col)):
                    non_fk_mentioned = True
                    break

            if non_fk_mentioned:
                retained_tables.append(tbl)
                continue

            if RETRIEVE_DEBUG:
                print(f"[ROUTER] Compact family prune: dropping {tbl} in favor of primary table {primary_table}")

        relevant_tables = list(dict.fromkeys(retained_tables))
        if RETRIEVE_DEBUG and relevant_tables == [primary_table]:
            print(f"[ROUTER] Compact family prune: keeping only primary table {primary_table}")

    # Explicit rank/window phrasing often triggers spurious table matches via
    # inflected tokens (e.g., "ordered ..."), even when the request is truly
    # single-table. If no strong non-primary semantic mention exists, keep only
    # the primary table so the query can stay in the 3B lane.
    explicit_rank_single_table_candidate = bool(
        primary_table
        and len(relevant_tables) > 1
        and is_explicit_rank_function
        and bool(within_same_match)
        and not is_ordinal_ranking
        and not has_exclusion
        and not is_subquery_intent
    )
    if explicit_rank_single_table_candidate:
        non_primary_semantic_mention = False
        for tbl in relevant_tables:
            if tbl == primary_table:
                continue

            tbl_lower = str(tbl or "").strip().lower()
            if _question_mentions_table_entity(question_lower, tbl):
                non_primary_semantic_mention = True
                break

            table_id_col = (get_id_column(tbl_lower) or "").strip().lower()
            fk_like_cols = set()
            for col in _get_table_cols(tbl):
                col_lower = str(col or "").strip().lower()
                if not col_lower:
                    continue
                if (table_id_col and col_lower == table_id_col) or _is_conventional_fk_name(get_singular_form(tbl_lower), col_lower):
                    fk_like_cols.add(col_lower)

            for col in _get_table_cols(tbl):
                col_lower = str(col or "").strip().lower()
                if not col_lower or col_lower in fk_like_cols:
                    continue
                if _question_mentions_column(question_lower, str(col)):
                    non_primary_semantic_mention = True
                    break
            if non_primary_semantic_mention:
                break

        if not non_primary_semantic_mention:
            relevant_tables = [primary_table]
            if RETRIEVE_DEBUG:
                print(
                    f"[ROUTER] Explicit-rank single-table compaction: keeping only {primary_table}"
                )

    # ═══════════════════════════════════════════════════════════════════
    # STEP 5: LOGIKA ESKALASI (Smarter Multi-Table Handling)
    # ═══════════════════════════════════════════════════════════════════
    is_multi_table = len(relevant_tables) > 1
    has_exclusion = _detect_router_exclusion(is_multi_table)
    effective_table_scope_count = max(len(relevant_tables), len(allowed_identifiers or {}))
    has_multi_table_scope = effective_table_scope_count > 1
    has_numeric_threshold = _question_has_numeric_threshold(question_lower)
    has_distinct_term = bool(
        re.search(r"\b(?:distinct|unique|different)\b", question_lower, flags=re.I)
    )
    is_multi_table_distinct_threshold = bool(
        is_multi_table
        and has_math
        and has_numeric_threshold
        and has_distinct_term
    )
    
    # Winner-row detection: needs ROW_NUMBER/RANK subquery pattern
    # Case 1: Multi-table + ranking + grouping (e.g., "customer with highest total payment")
    # Case 2: Single-table but explicit "X with largest/highest Y for each Z" pattern
    # Case 3: Tiebreaker pattern ("if multiple, choose the one with...")
    # Case 4: Ordinal ranking ("second highest", "third largest") - ALWAYS needs ROW_NUMBER
    is_winner_row = False
    
    # Pattern D: ORDINAL RANKING (highest priority - always complex)
    # "second highest", "third largest", "2nd lowest" - requires ROW_NUMBER() WHERE rn = N
    if is_ordinal_ranking:
        is_winner_row = True
        print(f"[ROUTER] Ordinal ranking detected ('second highest' etc) -> requires ROW_NUMBER subquery")
    
    # Pattern A: ranking + grouping + multi-table (original logic)
    if not is_winner_row and is_ranking_value and has_grouping and is_multi_table:
        is_winner_row = True
    
    # Pattern B: "X with the largest/highest Y" (even single table)
    # E.g., "payment record with the largest payment amount"
    if not is_winner_row and is_ranking_value:
        # "record/row with the largest/highest/lowest"
        if re.search(AGENT_ROUTER_WINNER_ROW_RECORD_PATTERN, question_lower):
            is_winner_row = True
        # "the largest X for each Y" / "largest X per Y"
        if re.search(AGENT_ROUTER_WINNER_ROW_FOR_EACH_PATTERN, question_lower):
            is_winner_row = True
    
    # Pattern C: Tiebreaker pattern (always complex, needs ROW_NUMBER with multiple ORDER BY)
    # "if multiple X have the same Y, choose the one with..."
    # "if there are ties, pick..."
    has_tiebreaker = bool(re.search(AGENT_ROUTER_TIEBREAKER_PATTERN, question_lower))
    if has_tiebreaker:
        is_winner_row = True
    if not is_winner_row and _question_wants_winner_row(question_lower):
        is_winner_row = True
    if is_winner_row and _question_has_offset_window_sequence_intent(question_lower):
        # LAG/LEAD-style "previous/next row" questions may contain words such
        # as "row" and "order", but they do not ask to pick a single winner
        # row per group. Keep them in the offset-window family.
        is_winner_row = False
    
    # Keep simple single-table scalar subquery in 3B lane:
    # e.g. "creditLimit greater than the average creditLimit".
    is_simple_scalar_subquery_single_table = (
        AGENT_ROUTER_ALLOW_SIMPLE_SINGLE_TABLE_SUBQUERY_3B
        and is_subquery_intent
        and not is_multi_table
        and not is_winner_row
        and not is_window_aggregate
        and not is_explicit_rank_function
        and not is_ordinal_ranking
        and not has_exclusion
        and bool(
            re.search(AGENT_ROUTER_SIMPLE_SINGLE_TABLE_SUBQUERY_PATTERN, question_lower)
            or has_global_avg_baseline
        )
        and not has_complex_subquery_cue
    )

    # Allow simple EXISTS/NOT EXISTS in 3B lane when it is not mixed with
    # harder intents (winner-row/window/rank/tie-break) and table fanout is small.
    is_simple_exists_subquery = (
        is_subquery_intent
        and has_exists_keyword
        and not is_winner_row
        and not is_window_aggregate
        and not is_explicit_rank_function
        and not is_ordinal_ranking
        and not has_tiebreaker
        and len(relevant_tables) <= 2
    )

    is_complex_subquery_intent = bool(
        is_subquery_intent
        and not is_simple_scalar_subquery_single_table
        and not is_simple_exists_subquery
    )

    is_set_operation_intent = _question_has_set_operation_intent(question)
    is_simple_set_operation = bool(
        is_set_operation_intent
        and not is_multi_table
        and not is_winner_row
        and not is_window_aggregate
        and not is_explicit_rank_function
        and not is_ordinal_ranking
        and not is_complex_subquery_intent
    )

    has_dual_fact_rollup = bool(
        is_multi_table and _router_has_dual_fact_rollup(question_raw, relevant_tables)
    )
    temporal_alt = _temporal_extrema_alt()
    has_latest_year_same_year_aggregate_intent = bool(
        is_multi_table
        and _question_wants_latest_year_scope_with_aggregate(question)
        and temporal_alt
        and re.search(
            rf"\b(?:{temporal_alt})\b[^.]{{0,64}}\bdate\b",
            question_lower,
            flags=re.I | re.S,
        )
    )
    has_group_eligibility_phrase = bool(
        re.search(
            r"\b(?:only\s+include|include\s+only|keep\s+only|only\s+keep|retain\s+only)\b"
            r"[^.]{0,220}\bat\s+least\s+one\b",
            question_lower,
            flags=re.I | re.S,
        )
    )
    has_relative_existence_detail = bool(
        re.search(
            r"\bat\s+least\s+one\b[^.]{0,160}\b(?:who|that|whose)\b",
            question_lower,
            flags=re.I | re.S,
        )
    )
    has_conditioned_subgroup_group_metric = bool(
        is_multi_table
        and has_grouping
        and re.search(
            r"\b(?:how\s+many(?:\s+of\s+(?:them|those|these))?|count|number\s+of)\b"
            r"[^.]{0,120}\b(?:are|have|with|whose|where)\b",
            question_lower,
            flags=re.I | re.S,
        )
        and has_group_eligibility_phrase
        and (
            re.search(r"\b(?:is\s+not\s+null|not\s+null|is\s+null)\b", question_lower, flags=re.I)
            or bool(re.findall(r"""['"“”][^'"“”]+['"“”]""", question or ""))
            or bool(re.search(r"=\s*['\"“”]", question or ""))
            or has_relative_existence_detail
        )
    )
    has_include_zero_aggregate_intent = bool(
        is_multi_table
        and has_grouping
        and _question_wants_include_zero(question_raw, ctx)
    )
    has_top_n_per_group_intent = bool(
        has_multi_table_scope
        and has_grouping
        and any(re.search(pat, question_lower, flags=re.I | re.S) for pat in AGENT_ROUTER_TOP_N_PER_GROUP_PATTERNS)
    )
    
    # ESCALATION LOGIC
    # Keep 3B for simple multi-table joins; escalate to 7B only for hard patterns.
    max_small_tables = max(1, int(AGENT_ROUTER_SIMPLE_MULTI_TABLE_MAX_TABLES))
    escalation_reasons: List[str] = []

    if AGENT_ROUTER_ESCALATE_ON_MULTI_TABLE and is_multi_table:
        if not AGENT_ROUTER_ALLOW_SIMPLE_MULTI_TABLE_3B:
            escalation_reasons.append("multi_table_forced")
        elif len(relevant_tables) > max_small_tables:
            escalation_reasons.append(
                f"multi_table_too_wide:{len(relevant_tables)}>{max_small_tables}"
            )

    if AGENT_ROUTER_ESCALATE_ON_WINNER_ROW and is_winner_row and is_multi_table:
        escalation_reasons.append("winner_row_subquery_multi_table")
    if AGENT_ROUTER_ESCALATE_ON_EXCLUSION and has_exclusion:
        escalation_reasons.append("cross_table_exclusion")
    if AGENT_ROUTER_ESCALATE_ON_WINDOW_AGGREGATE and is_window_aggregate and is_multi_table:
        escalation_reasons.append("window_aggregate_multi_table")
    if AGENT_ROUTER_ESCALATE_ON_EXPLICIT_RANK and is_explicit_rank_function and is_multi_table:
        escalation_reasons.append("explicit_rank_function_multi_table")
    if AGENT_ROUTER_ESCALATE_ON_SUBQUERY_INTENT and is_complex_subquery_intent:
        if is_multi_table or has_complex_subquery_cue:
            escalation_reasons.append("subquery_intent_complex")
    if is_multi_table_distinct_threshold:
        escalation_reasons.append("multi_table_distinct_threshold")
    if is_set_operation_intent and not is_simple_set_operation:
        escalation_reasons.append("set_operation")
    if has_dual_fact_rollup:
        escalation_reasons.append("multi_fact_rollup_fanout_risk")
    if has_latest_year_same_year_aggregate_intent:
        escalation_reasons.append("latest_year_same_year_aggregate")
    if has_conditioned_subgroup_group_metric:
        escalation_reasons.append("conditioned_subgroup_group_metric")
    if AGENT_ROUTER_ESCALATE_ON_INCLUDE_ZERO_AGGREGATE and has_include_zero_aggregate_intent:
        escalation_reasons.append("include_zero_aggregate")
    if has_top_n_per_group_intent:
        escalation_reasons.append("top_n_per_group_multi_table")

    # keep order, remove duplicates
    escalation_reasons = list(dict.fromkeys(escalation_reasons))
    should_escalate = bool(escalation_reasons)

    if is_multi_table and not should_escalate and AGENT_ROUTER_ALLOW_SIMPLE_MULTI_TABLE_3B:
        print(f"[ROUTER] Multi-table simple pattern detected: {relevant_tables} -> keeping 3B")
    elif escalation_reasons:
        print(f"[ROUTER] Escalating to 7B. reasons={escalation_reasons}")

    chosen_model = LARGE_MODEL if should_escalate else SMALL_MODEL
    complexity = AGENT_ROUTER_COMPLEXITY_COMPLEX if should_escalate else AGENT_ROUTER_COMPLEXITY_SIMPLE
    
    # Tentukan QTYPE
    if is_winner_row or is_explicit_rank_function:
        qtype = AGENT_ROUTER_QTYPE_RANK
    elif has_math or has_grouping:
        qtype = AGENT_ROUTER_QTYPE_AGGREGATE
    else: 
        qtype = AGENT_ROUTER_QTYPE_LOOKUP
    
    print(f"[ROUTER] Detected Tables: {relevant_tables}")
    print(
        f"[ROUTER] Signals -> Math:{has_math}, Sort:{is_sorting}, Rank:{is_ranking}, "
        f"Group:{has_grouping}, WinnerRow:{is_winner_row}, OrdinalRank:{is_ordinal_ranking}, "
        f"Exclusion:{has_exclusion}, WindowAgg:{is_window_aggregate}, "
        f"ExplicitRank:{is_explicit_rank_function}, SubqueryIntent:{is_subquery_intent}, "
        f"SetOp:{is_set_operation_intent}, "
        f"SimpleScalarSubquery:{is_simple_scalar_subquery_single_table}, "
        f"SimpleExistsSubquery:{is_simple_exists_subquery}, "
        f"TopNPerGroup:{has_top_n_per_group_intent}, "
        f"DualFactRollup:{has_dual_fact_rollup}"
    )
    print(f"[ROUTER] LLM Fallback Candidate: {complexity.upper()} -> {chosen_model}")

    return {
        **state,
        "router_tables": relevant_tables,
        "primary_table": primary_table,
        "primary_table_reason": primary_reason,
        "router_model_candidate": chosen_model,
        "target_model": "",
        "router_decision": {
            "complexity": complexity,
            "qtype": qtype,
            "has_grouping": has_grouping,
            "is_ranking": is_ranking,
            "is_winner_row": is_winner_row,
            "is_ordinal_ranking": is_ordinal_ranking,
            "has_exclusion": has_exclusion,
            "is_window_aggregate": is_window_aggregate,
            "partition_hint": partition_hint,
            "is_explicit_rank_function": is_explicit_rank_function,
            "is_subquery_intent": is_subquery_intent,
            "is_set_operation_intent": is_set_operation_intent,
            "is_simple_set_operation": is_simple_set_operation,
            "is_simple_scalar_subquery_single_table": is_simple_scalar_subquery_single_table,
            "is_simple_exists_subquery": is_simple_exists_subquery,
            "is_complex_subquery_intent": is_complex_subquery_intent,
            "has_dual_fact_rollup": has_dual_fact_rollup,
            "table_count": len(relevant_tables),
            "escalation_reasons": escalation_reasons,
            "simple_multi_table_candidate": bool(is_multi_table and not should_escalate),
        }
    }
# ======================================================================
# ======================= AGENT 2 - SQL DRAFTER ========================
# ======================================================================
import time as _time  # For inference timing
# Track last warmup time per model
_last_model_warmup = {}
_CONTRACT_METRIC_TERMS_CACHE: Optional[Set[str]] = None
_CONTRACT_ENTITY_TERMS_CACHE: Optional[Set[str]] = None


def _contract_normalize_token(token: str) -> str:
    t = str(token or "").strip().lower()
    if not t:
        return ""
    if t.endswith("ies") and len(t) > 3:
        return t[:-3] + "y"
    if t.endswith("s") and len(t) > 2 and not t.endswith(("ss", "us")):
        return t[:-1]
    return t


def _contract_tokenize(text: str) -> Set[str]:
    return {
        tok
        for tok in (
            _contract_normalize_token(raw)
            for raw in re.findall(r"[a-z0-9]+", str(text or "").lower())
        )
        if tok and not tok.isdigit()
    }


def _contract_identifier_terms(name: str) -> Set[str]:
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name or ""))
    words = words.replace("_", " ")
    return _contract_tokenize(words)


def _contract_metric_terms(question_tokens: Optional[Set[str]] = None) -> Set[str]:
    global _CONTRACT_METRIC_TERMS_CACHE
    base_terms = (
        set(_CONTRACT_METRIC_TERMS_CACHE)
        if _CONTRACT_METRIC_TERMS_CACHE is not None
        else None
    )
    if base_terms is None:
        terms: Set[str] = set(AGENT_CONTRACT_BASE_METRIC_TERMS)
        analyzer = None
        try:
            analyzer = get_schema_analyzer()
        except Exception:
            analyzer = None

        schema_enriched = False
        for table in get_all_tables() or []:
            t = str(table or "").strip().lower()
            if not t:
                continue
            try:
                cols = list(analyzer.get_table_columns(t)) if analyzer is not None else []
            except Exception:
                cols = []
            for col in cols:
                c = str(col or "").strip()
                if not c:
                    continue
                c_low = c.lower()
                if c_low.endswith(("id", "number", "code", "key")):
                    continue
                role = ""
                try:
                    if analyzer is not None:
                        role = str(analyzer._column_roles.get(t, {}).get(c_low, "") or "").lower()
                except Exception:
                    role = ""
                col_terms = _contract_identifier_terms(c)
                if role in {"metric", "measure", "amount", "quantity"} or col_terms.intersection(terms):
                    schema_enriched = True
                    terms.update(col_terms)

        _CONTRACT_METRIC_TERMS_CACHE = set(terms)
        base_terms = set(_CONTRACT_METRIC_TERMS_CACHE)

    terms = set(base_terms)
    if question_tokens is not None:
        q_tokens = {str(tok).strip().lower() for tok in question_tokens if str(tok).strip()}
        if not DOMAIN_HEURISTICS_ENABLED:
            return terms
        # Domain lexical priors are only a compatibility fallback here.
        # The schema-enriched term set above remains the primary signal.
        schema_or_generic_hits = q_tokens.intersection(terms)
        domain_hits = q_tokens.intersection(AGENT_CONTRACT_DOMAIN_METRIC_FALLBACK_TERMS)
        if domain_hits and not schema_or_generic_hits:
            terms.update(domain_hits)
    return terms


def _contract_entity_terms() -> Set[str]:
    global _CONTRACT_ENTITY_TERMS_CACHE
    if _CONTRACT_ENTITY_TERMS_CACHE is not None:
        return set(_CONTRACT_ENTITY_TERMS_CACHE)

    terms: Set[str] = set(AGENT_CONTRACT_GENERIC_COUNT_NOUNS)
    for table in get_all_tables() or []:
        t = str(table or "").strip().lower()
        if not t:
            continue
        singular = str(get_singular_form(t) or t).strip().lower()
        for variant in {t, singular, t.replace("_", " "), singular.replace("_", " ")}:
            terms.update(_contract_tokenize(variant))

    _CONTRACT_ENTITY_TERMS_CACHE = set(terms)
    return set(_CONTRACT_ENTITY_TERMS_CACHE)


def _collect_hard_identifier_errors(errors: List[str]) -> List[str]:
    if not errors:
        return []
    return [
        e for e in errors
        if "not in ALLOWED" in e or "invalid column" in e or "invalid columns" in e
    ]


def _categorize_gate_reasons(reasons: List[str]) -> List[str]:
    """
    Stable reason taxonomy for retry/analysis observability.
    This is metadata only; it does not affect acceptance behavior.
    """
    tags: Set[str] = set()
    for reason in (reasons or []):
        r = str(reason or "").strip().lower()
        if not r:
            continue
        if r.startswith("parse_") or "parse" in r:
            tags.add("syntax")
        if "identifier" in r:
            tags.add("identifier")
        if "join" in r or "fanout" in r:
            tags.add("join_path")
        if "semantic" in r:
            tags.add("semantic")
        if "group" in r:
            tags.add("grouping")
        if "ambiguous" in r:
            tags.add("ambiguity")
        if "select" in r:
            tags.add("select_shape")
    if not tags and (reasons or []):
        tags.add("other")
    return sorted(tags)


def _should_expand_retrieval_context(gate_reasons: List[str]) -> bool:
    if not RETRIEVAL_EXPAND_ON_FAIL_ENABLED:
        return False

    reasons = [str(r or "").strip().lower() for r in (gate_reasons or []) if str(r or "").strip()]
    if not reasons:
        return False

    triggers = [
        str(t or "").strip().lower()
        for t in (RETRIEVAL_EXPAND_TRIGGER_REASONS or [])
        if str(t or "").strip()
    ]
    if not triggers:
        return False

    for trig in triggers:
        for reason in reasons:
            if reason == trig or reason.startswith(f"{trig}:") or trig in reason:
                return True
    return False


def _is_low_retrieval_evidence(
    retrieval_snapshot: Optional[Dict[str, Any]],
) -> Tuple[bool, List[str]]:
    snap = dict(retrieval_snapshot or {})
    reasons: List[str] = []

    def _to_int(v: Any, default: int = 0) -> int:
        try:
            return int(v)
        except Exception:
            return int(default)

    allowed_tables = _to_int(snap.get("allowed_tables_after_filter"))
    allowed_cols = _to_int(snap.get("allowed_columns_after_filter"))
    reranked = _to_int(snap.get("reranked_candidates"))
    sparse = _to_int(snap.get("sparse_candidates"))
    dense = _to_int(snap.get("dense_candidates"))

    if allowed_tables < max(1, int(AGENT_B4_EVIDENCE_MIN_ALLOWED_TABLES)):
        reasons.append(
            f"allowed_tables_low:{allowed_tables}<{int(AGENT_B4_EVIDENCE_MIN_ALLOWED_TABLES)}"
        )
    if allowed_cols < max(1, int(AGENT_B4_EVIDENCE_MIN_ALLOWED_COLUMNS)):
        reasons.append(
            f"allowed_columns_low:{allowed_cols}<{int(AGENT_B4_EVIDENCE_MIN_ALLOWED_COLUMNS)}"
        )
    if reranked < max(1, int(AGENT_B4_EVIDENCE_MIN_RERANKED)):
        reasons.append(
            f"reranked_low:{reranked}<{int(AGENT_B4_EVIDENCE_MIN_RERANKED)}"
        )
    if sparse <= 0 and dense <= 0:
        reasons.append("no_sparse_dense_candidates")

    return bool(reasons), reasons


def _evaluate_cannot_answer_from_ir(
    intent_ir: Dict[str, Any],
    *,
    retrieval_snapshot: Optional[Dict[str, Any]] = None,
    gate_reasons: Optional[List[str]] = None,
) -> Tuple[bool, str, List[str]]:
    """
    Decide whether to abstain before SQL drafting for ambiguous/unanswerable intents.
    This gate is flag-driven and defaults to inactive.
    """
    signals = [str(s).strip() for s in (intent_ir.get("ambiguity_signals") or []) if str(s).strip()]
    notes = {str(n or "").strip().lower() for n in (intent_ir.get("notes") or []) if str(n or "").strip()}
    subject = str(intent_ir.get("subject_entity") or "").strip()
    mention_count = len(intent_ir.get("mentioned_tables") or [])

    if AGENT_B4_AMBIGUITY_ABSTAIN_ENABLED and len(signals) >= max(1, int(AGENT_B4_AMBIGUITY_MIN_SIGNALS)):
        return True, f"Ambiguous question across schema context: {signals}", signals

    if AGENT_B4_CANNOT_ANSWER_ENABLED:
        if not subject and mention_count >= 2 and signals:
            return True, f"Cannot resolve subject entity reliably: {signals}", signals
        if "disconnected_mentions" in notes and not subject:
            return True, "Cannot map disconnected table mentions to a safe join path", signals

    if AGENT_B4_EVIDENCE_ABSTAIN_ENABLED:
        low_evidence, evidence_reasons = _is_low_retrieval_evidence(retrieval_snapshot)
        gate_reason_set = {
            str(r or "").strip().lower()
            for r in (gate_reasons or [])
            if str(r or "").strip()
        }
        trigger_reasons = {
            str(r or "").strip().lower()
            for r in (AGENT_B4_EVIDENCE_TRIGGER_REASONS or [])
            if str(r or "").strip()
        }
        trigger_hit = False
        for reason in gate_reason_set:
            if reason in trigger_reasons:
                trigger_hit = True
                break
            if any(
                reason.startswith(f"{trig}:") or trig in reason
                for trig in trigger_reasons
            ):
                trigger_hit = True
                break

        if low_evidence and (signals or trigger_hit):
            reason_bits: List[str] = []
            if trigger_hit:
                reason_bits.append(f"gate_reasons={sorted(gate_reason_set)}")
            if signals:
                reason_bits.append(f"ambiguity={signals}")
            reason_bits.append(f"evidence={evidence_reasons}")
            return True, "Cannot answer safely from weak schema evidence (" + "; ".join(reason_bits) + ")", signals

    return False, "", signals


def _extract_llm_cannot_answer_reason(obj: Dict[str, Any]) -> str:
    reason = str(obj.get("cannot_answer_reason") or "").strip()
    notes = str(obj.get("notes") or "").strip()
    prefix = str(AGENT_PROMPT_ABSTAIN_NOTES_PREFIX or "CANNOT_ANSWER:").strip()
    if not reason and prefix and notes.upper().startswith(prefix.upper()):
        reason = notes[len(prefix):].strip(" :-")
    if not reason and notes:
        reason = notes
    return reason


def _llm_output_is_cannot_answer(obj: Dict[str, Any]) -> bool:
    if not AGENT_PROMPT_ABSTAIN_ENABLED:
        return False
    if not isinstance(obj, dict):
        return False
    if bool(obj.get("cannot_answer", False)):
        return True
    sql_text = str(obj.get("sql") or "").strip()
    notes = str(obj.get("notes") or "").strip()
    prefix = str(AGENT_PROMPT_ABSTAIN_NOTES_PREFIX or "CANNOT_ANSWER:").strip()
    return (not sql_text) and bool(prefix) and notes.upper().startswith(prefix.upper())


def _extract_explicit_row_limit_value(question: str) -> Optional[int]:
    """
    Extract a row-limit value only from phrases that constrain result rows.
    Transformation phrases like "first 2 characters" are not row limits.
    """
    q = str(question or "").strip().lower()
    if not q:
        return None

    non_row_unit = r"(?!\s+(?:characters?|chars?|letters?|digits?|bytes?|positions?|tokens?|words?)\b)"
    superlative_alt = _phrase_alt_pattern(
        ordering_superlative_keywords()
    )
    patterns = [
        r"\blimit\s+(\d{1,4})\b",
        r"\btop\s+(\d{1,4})\b",
        r"\bbottom\s+(\d{1,4})\b",
        rf"\b(?:first|last)\s+(\d{{1,4}})\b{non_row_unit}",
        rf"\b(?:show|return|list|display|give\s+me)\s+(\d{{1,4}})\b{non_row_unit}",
        rf"\b(\d{{1,4}})\s+(?:{superlative_alt})\b" if superlative_alt else r"$^",
    ]
    for pat in patterns:
        m = re.search(pat, q, flags=re.I)
        if not m:
            continue
        try:
            value = int(m.group(1))
        except Exception:
            value = 0
        if value > 0:
            return value
    return None


def _extract_semantic_contract(question: str) -> Dict[str, Any]:
    """
    Build a conservative semantic contract directly from the NL question.
    Contract is intentionally minimal to avoid false positives:
    - Explicit ORDER BY intent
    - Explicit Top-N/LIMIT intent
    - Explicit aggregate intent (COUNT / SUM / AVG)
    """
    q = str(question or "").strip().lower()
    contract: Dict[str, Any] = {
        "requires_order_by": False,
        "required_limit": None,
        "required_aggs": [],
        "requested_order_terms": [],
    }
    if not q:
        return contract

    # Reuse the shared final-order detector so acceptance-gate semantics stay
    # aligned with the refiner contract logic. This avoids false positives on
    # window-ranking prompts where "descending order" belongs inside OVER (...),
    # not as a top-level ORDER BY requirement.
    explicit_sort_phrase = _question_requests_ordering(question)
    contract["requires_order_by"] = explicit_sort_phrase
    if explicit_sort_phrase:
        contract["requested_order_terms"] = _extract_requested_final_order_terms(question)

    superlative_alt = _phrase_alt_pattern(
        ordering_superlative_keywords()
    )
    ordinal_alt = _phrase_alt_pattern(
        semantic_contract_ordinal_terms()
    )
    # Ordinal per-group winner pattern (e.g. "second highest per customer")
    # naturally relies on window ranking + rn filter, and does not require
    # top-level ORDER BY unless user explicitly asks sorting.
    ordinal_per_group_pattern = bool(
        superlative_alt
        and ordinal_alt
        and re.search(
            rf"\b(?:{ordinal_alt})\s+(?:{superlative_alt})\b",
            q,
        )
        and any_phrase_in_text(q, semantic_contract_per_group_terms())
    )
    if ordinal_per_group_pattern and not explicit_sort_phrase:
        contract["requires_order_by"] = False

    contract["required_limit"] = _extract_explicit_row_limit_value(question)

    # Aggregate intent extraction (conservative to avoid lookup false positives).
    required_aggs: Set[str] = set()

    # COUNT intent:
    # - strong cues: "how many", "count"
    # - phrase cue: "number of <entity>", excluding identifier-like projections
    #   such as "employee numbers", "office codes", "customer ids".
    has_how_many = any_phrase_in_text(q, semantic_contract_count_how_many_phrases())
    has_count_word = any_phrase_in_text(q, semantic_contract_count_terms())

    number_of_target = ""
    number_of_prefix_terms = [
        str(term).strip().lower()
        for term in semantic_contract_count_number_of_prefixes()
        if str(term).strip()
    ]
    prefix_patterns = [
        re.escape(term).replace(r"\ ", r"\s+")
        for term in number_of_prefix_terms
    ]
    if prefix_patterns:
        m_number_of = re.search(
            rf"\b(?:{'|'.join(sorted(prefix_patterns, key=len, reverse=True))})\s+([a-z][a-z0-9_]*)\b",
            q,
            flags=re.I,
        )
        if m_number_of:
            number_of_target = str(m_number_of.group(1) or "").strip().lower()

    identifier_like_targets = semantic_contract_identifier_like_targets()
    has_number_of_entity = bool(
        number_of_target and number_of_target not in identifier_like_targets
    )
    if has_how_many or has_count_word or has_number_of_entity:
        required_aggs.add("COUNT")

    avg_terms = semantic_contract_average_terms()
    avg_alt = _phrase_alt_pattern(avg_terms)
    ratio_definition_alt = _phrase_alt_pattern(semantic_contract_ratio_definition_terms())
    ratio_count_reference_alt = _phrase_alt_pattern(
        semantic_contract_ratio_count_reference_phrases()
    )
    ratio_operator_alt = _phrase_alt_pattern(semantic_contract_ratio_operator_terms())
    defines_average_as_ratio = bool(
        (avg_alt and re.search(rf"\b(?:{avg_alt})\b", q))
        and ratio_count_reference_alt
        and re.search(rf"\b(?:{ratio_count_reference_alt})\b", q)
        and (
            (ratio_definition_alt and ratio_operator_alt and re.search(rf"\b(?:{ratio_definition_alt})\b[^.]{{0,140}}\b(?:{ratio_operator_alt})\b", q))
            or (
                ratio_operator_alt
                and re.search(
                rf"\b(?:{ratio_operator_alt})\s+(?:the\s+)?(?:{ratio_count_reference_alt})\b",
                q,
                )
            )
        )
    )

    # AVG intent.
    if avg_alt and re.search(rf"\b(?:{avg_alt})\b", q) and not defines_average_as_ratio:
        required_aggs.add("AVG")

    # SUM intent:
    # - explicit "sum"/"sum of"
    # - "total" only when attached to quantity/value-like metrics.
    has_explicit_sum = any_phrase_in_text(q, semantic_contract_sum_terms())
    q_tokens = _contract_tokenize(q)
    metric_terms = _contract_metric_terms(q_tokens)
    entity_terms = _contract_entity_terms()
    total_prefix_alt = _phrase_alt_pattern(
        semantic_contract_sum_total_prefix_terms()
    )
    total_phrases = [
        str(m.group(1) or "").strip()
        for m in (
            re.finditer(
                rf"\b(?:{total_prefix_alt})\s+([a-z][a-z0-9_ ]{{0,48}})\b",
                q,
            )
            if total_prefix_alt
            else []
        )
    ]
    has_total_metric = any(
        _contract_tokenize(phrase).intersection(metric_terms)
        for phrase in total_phrases
    )
    looks_like_count_phrase = any_phrase_in_text(
        q,
        semantic_contract_sum_total_count_phrases(),
    )
    if not looks_like_count_phrase:
        for phrase in total_phrases:
            phrase_tokens = _contract_tokenize(phrase)
            if phrase_tokens.intersection(entity_terms) and not phrase_tokens.intersection(metric_terms):
                looks_like_count_phrase = True
                break
    if (has_explicit_sum or has_total_metric) and not looks_like_count_phrase:
        required_aggs.add("SUM")

    contract["required_aggs"] = sorted(required_aggs)
    return contract


def _extract_limit_value_from_clause_sql(limit_sql: str) -> Optional[int]:
    if not limit_sql:
        return None
    ls = str(limit_sql).strip().lower()
    try:
        # LIMIT offset, count
        m = re.search(r"\blimit\s+\d+\s*,\s*(\d+)\b", ls)
        if m:
            return int(m.group(1))
        # LIMIT count [OFFSET n]
        m = re.search(r"\blimit\s+(\d+)\b", ls)
        if m:
            return int(m.group(1))
    except Exception:
        return None
    return None


def _evaluate_semantic_contract(
    sql: str,
    question: str,
    parsed: Any,
    sel: Any,
    exp_mod: Any,
    contract: Dict[str, Any],
) -> List[str]:
    failures: List[str] = []
    if not contract:
        return failures

    def _semantic_clause_arg(name: str) -> Any:
        # For set-operation roots (e.g. UNION ... ORDER BY), semantic clauses
        # live on the top-level parsed expression, not necessarily on the first
        # inner SELECT returned by parsed.find(exp.Select).
        root_arg = getattr(parsed, "args", {}).get(name)
        if root_arg is not None:
            return root_arg
        return getattr(sel, "args", {}).get(name)

    requires_order = bool(contract.get("requires_order_by"))
    required_limit = contract.get("required_limit")
    required_aggs = {str(a).upper() for a in (contract.get("required_aggs") or []) if str(a).strip()}

    if requires_order:
        order_arg = _semantic_clause_arg("order")
        has_order = bool(order_arg is not None and (getattr(order_arg, "expressions", None) or []))
        if not has_order:
            failures.append("missing_order_by")
        else:
            failures.extend(
                _evaluate_requested_final_order_sequence(
                    sql=sql,
                    question=question,
                    parsed=parsed,
                    sel=sel,
                    exp_mod=exp_mod,
                )
            )

    if required_limit is not None:
        limit_arg = _semantic_clause_arg("limit")
        if limit_arg is None:
            failures.append("missing_limit")
        else:
            parsed_limit = _extract_limit_value_from_clause_sql(limit_arg.sql(dialect=SQL_WRITE_DIALECT))
            if parsed_limit is None:
                failures.append("unparsed_limit")
            elif int(parsed_limit) != int(required_limit):
                failures.append(f"limit_mismatch:{parsed_limit}!={required_limit}")

    if required_aggs:
        present_aggs: Set[str] = set()
        for agg in parsed.find_all(exp_mod.AggFunc):
            if isinstance(agg, exp_mod.Count):
                present_aggs.add("COUNT")
            elif isinstance(agg, exp_mod.Sum):
                present_aggs.add("SUM")
            elif isinstance(agg, exp_mod.Avg):
                present_aggs.add("AVG")
        for req in sorted(required_aggs):
            if req not in present_aggs:
                failures.append(f"missing_agg_{req.lower()}")

    return failures

def _deterministic_sql_acceptance_gate(
    sql: str,
    ctx: str,
    question: str = "",
    *,
    strict_join_key: bool = False,
) -> Dict[str, Any]:
    """
    Deterministic SQL acceptance gate before execution.
    Checks:
    - SQL parseability (sqlglot / mysql dialect)
    - Identifier validity (table/column/alias scope)
    - Ambiguous unqualified columns (multi-table)
    - GROUP BY projection consistency (obvious 1055 risks)
    - Conservative semantic contract (ORDER/LIMIT/aggregate intent)
    - Optional join-key sanity strictness (from AST warnings)
    """
    verdict: Dict[str, Any] = {
        "accepted": False,
        "sql": (sql or "").strip(),
        "reasons": [],
        "parse_error": "",
        "identifier_errors": [],
        "hard_identifier_errors": [],
        "ambiguous_unqualified_columns": [],
        "group_projection_violations": [],
        "invalid_join_edges": [],
        "fanout_risk_tables": [],
        "join_risk_warnings": [],
        "semantic_contract": _extract_semantic_contract(question),
        "semantic_failures": [],
        "ast_fixes": [],
        "ast_warnings": [],
    }

    candidate_sql = (sql or "").strip()
    if not candidate_sql:
        verdict["reasons"].append("empty_sql")
        return verdict

    def _find_top_level_kw_local(text: str, kw: str, start: int = 0) -> int:
        target = str(kw or "").upper()
        if not target:
            return -1
        t = str(text or "")
        t_up = t.upper()
        depth = 0
        quote = ""
        i = max(0, int(start))
        while i < len(t):
            ch = t[i]
            if quote:
                if ch == quote and (i == 0 or t[i - 1] != "\\"):
                    quote = ""
                i += 1
                continue
            if ch in {"'", '"', "`"}:
                quote = ch
                i += 1
                continue
            if ch == "(":
                depth += 1
                i += 1
                continue
            if ch == ")":
                depth = max(0, depth - 1)
                i += 1
                continue
            if depth == 0 and t_up.startswith(target, i):
                prev_ok = i == 0 or not (t_up[i - 1].isalnum() or t_up[i - 1] == "_")
                after = i + len(target)
                next_ok = after >= len(t_up) or not (t_up[after].isalnum() or t_up[after] == "_")
                if prev_ok and next_ok:
                    return i
            i += 1
        return -1

    def _top_level_union_count_local(text: str) -> int:
        body = str(text or "")
        count = 0
        idx = 0
        while idx < len(body):
            pos_all = _find_top_level_kw_local(body, "UNION ALL", idx)
            pos_union = _find_top_level_kw_local(body, "UNION", idx)
            candidates = [p for p in (pos_all, pos_union) if p >= 0]
            if not candidates:
                break
            pos = min(candidates)
            count += 1
            idx = pos + 5
        return count

    repaired_distinct_sql = _repair_misplaced_select_distinct(candidate_sql)
    if repaired_distinct_sql and repaired_distinct_sql != candidate_sql:
        candidate_sql = str(repaired_distinct_sql).strip()

    ast_sql, ast_meta = _ast_guardrails(candidate_sql, ctx=ctx, question=question)
    if ast_meta.get("parse_ok", False):
        if ast_meta.get("fixes"):
            original_sql = candidate_sql
            ast_fixed_sql = str(ast_sql or "").strip()
            original_union_cnt = _top_level_union_count_local(original_sql)
            ast_union_cnt = _top_level_union_count_local(ast_fixed_sql)
            original_has_setop = original_union_cnt > 0
            original_has_group = bool(_find_top_level_kw_local(original_sql, "GROUP BY") >= 0)
            original_has_having = bool(_find_top_level_kw_local(original_sql, "HAVING") >= 0)
            ast_has_group = bool(_find_top_level_kw_local(ast_fixed_sql, "GROUP BY") >= 0)
            ast_has_having = bool(_find_top_level_kw_local(ast_fixed_sql, "HAVING") >= 0)

            # Keep AST fixes, except when they reshape top-level set-operation
            # queries in a way that can corrupt branch semantics.
            if original_has_setop and (
                ast_union_cnt != original_union_cnt
                or (not original_has_group and ast_has_group)
                or (not original_has_having and ast_has_having)
            ):
                verdict["ast_warnings"] = list(verdict.get("ast_warnings") or []) + [
                    "ast_fix_skipped:setop_shape_change"
                ]
            else:
                verdict["ast_fixes"] = list(ast_meta.get("fixes") or [])
                candidate_sql = ast_fixed_sql
    else:
        verdict["parse_error"] = str(ast_meta.get("error") or "parse_failed")
        verdict["reasons"].append("parse_error")
        return verdict

    ast_warnings = [str(w) for w in (ast_meta.get("warnings") or [])]
    verdict["ast_warnings"] = ast_warnings
    join_warnings = [w for w in ast_warnings if w.startswith("suspicious_join_key:")]
    verdict["join_risk_warnings"] = join_warnings
    if strict_join_key and join_warnings:
        verdict["reasons"].append("suspicious_join_key")

    identifier_errors = validate_identifiers(candidate_sql, ctx)
    hard_identifier_errors = _collect_hard_identifier_errors(identifier_errors)
    verdict["identifier_errors"] = identifier_errors
    verdict["hard_identifier_errors"] = hard_identifier_errors
    if hard_identifier_errors:
        verdict["reasons"].append("hard_identifier")
    elif identifier_errors:
        verdict["reasons"].append("identifier")

    try:
        import sqlglot
        from sqlglot import exp

        parsed = sqlglot.parse_one(candidate_sql, read=SQL_PARSE_DIALECT)
        sel = parsed.find(exp.Select)
        if sel is None:
            verdict["reasons"].append("no_select")
            verdict["sql"] = candidate_sql
            return verdict

        def _is_nested_subquery_column(col_expr: "exp.Column") -> bool:
            anc = getattr(col_expr, "parent", None)
            while anc is not None:
                if isinstance(anc, exp.Select):
                    return anc is not sel
                anc = getattr(anc, "parent", None)
            return False

        def _norm_expr(expr_in: "exp.Expression") -> str:
            s = expr_in.sql(dialect=SQL_WRITE_DIALECT)
            s = s.replace("`", "").strip().lower()
            return re.sub(r"\s+", " ", s)

        # Build top-level table scope
        alias_to_table: Dict[str, str] = {}
        explicit_alias_to_table: Dict[str, str] = {}

        def _register_src(src: Optional["exp.Expression"]) -> None:
            if src is None:
                return
            if isinstance(src, exp.Table):
                table_name = str(src.name or "").strip("`").lower()
                if not table_name:
                    return
                alias_name = str(src.alias_or_name or table_name).strip("`").lower()
                if alias_name:
                    alias_to_table.setdefault(alias_name, table_name)
                    explicit_alias_to_table.setdefault(alias_name, table_name)
                alias_to_table.setdefault(table_name, table_name)
            elif isinstance(src, exp.Subquery):
                alias_name = str(src.alias_or_name or "").strip("`").lower()
                if alias_name:
                    alias_to_table.setdefault(alias_name, f"__derived__:{alias_name}")
                    explicit_alias_to_table.setdefault(alias_name, f"__derived__:{alias_name}")

        from_arg = sel.args.get("from") or sel.args.get("from_")
        if from_arg is not None:
            _register_src(getattr(from_arg, "this", None))
        for j in sel.args.get("joins") or []:
            _register_src(getattr(j, "this", None))

        # Build table->column map from context + schema metadata
        allowed_ids = _parse_allowed_identifiers(ctx or "")
        allowed_by_lower: Dict[str, Set[str]] = {}
        for t, cols in (allowed_ids or {}).items():
            tl = str(t or "").strip("`").lower()
            if not tl:
                continue
            allowed_by_lower.setdefault(tl, set()).update(
                str(c or "").strip("`").lower() for c in (cols or set()) if str(c or "").strip()
            )

        analyzer = None
        try:
            analyzer = get_schema_analyzer()
        except Exception:
            analyzer = None

        table_cols: Dict[str, Set[str]] = {}
        for _, table_name in alias_to_table.items():
            if str(table_name).startswith("__derived__:"):
                continue
            table_lower = str(table_name).lower()
            cols = set(allowed_by_lower.get(table_lower, set()))
            if analyzer is not None:
                try:
                    cols.update(
                        str(c).strip("`").lower()
                        for c in analyzer.get_table_columns(table_lower)
                    )
                except Exception:
                    pass
            table_cols[table_lower] = cols

        # FK graph-backed JOIN path validation (hard check).
        fk_map_raw = {}
        try:
            fk_map_raw = get_fk_relationships() or {}
        except Exception:
            fk_map_raw = {}

        fk_map: Dict[str, Dict[str, str]] = {}
        for src, rels in fk_map_raw.items():
            src_l = str(src or "").strip("`").lower()
            if not src_l:
                continue
            fk_map.setdefault(src_l, {})
            for fk_col, tgt in (rels or {}).items():
                fk_l = str(fk_col or "").strip("`").lower()
                tgt_l = str(tgt or "").strip("`").lower()
                if fk_l and tgt_l:
                    fk_map[src_l][fk_l] = tgt_l

        def _is_id_like(col_name: str) -> bool:
            c = (col_name or "").lower()
            return bool(c) and (c.endswith("id") or c.endswith("number") or c.endswith("code"))

        hierarchy_question = bool(
            re.search(
                r"\b(?:manager|subordinate|supervisor|parent|child|owner|member|leader|mentor|boss|report(?:s|ing)?\s+to)\b",
                str(question or ""),
                flags=re.I,
            )
        )

        def _looks_like_self_reference_col(table_name: str, col_name: str, id_col_name: str) -> bool:
            table_l = str(table_name or "").strip().lower()
            col_l = str(col_name or "").strip().lower()
            id_l = str(id_col_name or "").strip().lower()
            if not table_l or not col_l or not id_l or col_l == id_l:
                return False

            role_l = ""
            desc_l = ""
            try:
                role_l = str(get_column_role(table_l, col_l) or "").strip().lower()
            except Exception:
                role_l = ""
            try:
                desc_l = str(get_column_description(table_l, col_l) or "").strip().lower()
            except Exception:
                desc_l = ""

            semantic_text = " ".join(part for part in (col_l, role_l, desc_l) if part).strip()
            if not semantic_text:
                return False

            if re.search(
                r"\b(?:manager|subordinate|supervisor|parent|child|owner|member|leader|mentor|boss|report(?:s|ing)?\s+to|hierarch)\b",
                semantic_text,
                flags=re.I,
            ):
                return True

            if id_l and id_l in semantic_text.replace("`", ""):
                return True

            return False

        def _is_plausible_join_edge(lt: str, lc: str, rt: str, rc: str) -> bool:
            lt_l = (lt or "").lower()
            rt_l = (rt or "").lower()
            lc_l = (lc or "").lower()
            rc_l = (rc or "").lower()
            if not lt_l or not rt_l or not lc_l or not rc_l:
                return False

            # Direct FK edge: lt.lc -> rt.<id>
            if fk_map.get(lt_l, {}).get(lc_l) == rt_l:
                rt_id = (get_id_column(rt_l) or "").strip("`").lower()
                if not rt_id or rc_l == rt_id or rc_l == lc_l:
                    return True

            # Reverse FK edge: rt.rc -> lt.<id>
            if fk_map.get(rt_l, {}).get(rc_l) == lt_l:
                lt_id = (get_id_column(lt_l) or "").strip("`").lower()
                if not lt_id or lc_l == lt_id or lc_l == rc_l:
                    return True

            # Heuristic fallback: same key-like column on both sides.
            if lc_l == rc_l and _is_id_like(lc_l):
                if lc_l in table_cols.get(lt_l, set()) and rc_l in table_cols.get(rt_l, set()):
                    return True

            # Conservative self-join fallback when metadata misses explicit
            # self-referential FK edges (e.g., hierarchy parent pointers).
            if lt_l == rt_l:
                table_id = (get_id_column(lt_l) or "").strip("`").lower()
                if table_id:
                    for fk_col, id_col in ((lc_l, rc_l), (rc_l, lc_l)):
                        if id_col != table_id:
                            continue
                        if fk_col == table_id:
                            continue
                        if fk_col in table_cols.get(lt_l, set()) and id_col in table_cols.get(rt_l, set()):
                            if (
                                _is_id_like(fk_col)
                                or _looks_like_self_reference_col(lt_l, fk_col, table_id)
                                or hierarchy_question
                            ):
                                return True

            return False

        child_to_parents_alias: Dict[str, Set[str]] = {}

        def _record_fk_alias_edge(
            lt_alias: str,
            lc: str,
            rt_alias: str,
            rc: str,
            lt_table: str,
            rt_table: str,
        ) -> None:
            lt_alias_l = str(lt_alias or "").strip().lower()
            rt_alias_l = str(rt_alias or "").strip().lower()
            lc_l = str(lc or "").strip().lower()
            rc_l = str(rc or "").strip().lower()
            lt_table_l = str(lt_table or "").strip().lower()
            rt_table_l = str(rt_table or "").strip().lower()
            if not lt_alias_l or not rt_alias_l or not lc_l or not rc_l:
                return
            if not lt_table_l or not rt_table_l:
                return
            if fk_map.get(lt_table_l, {}).get(lc_l) == rt_table_l:
                child_to_parents_alias.setdefault(lt_alias_l, set()).add(rt_alias_l)
            if fk_map.get(rt_table_l, {}).get(rc_l) == lt_table_l:
                child_to_parents_alias.setdefault(rt_alias_l, set()).add(lt_alias_l)

        def _has_parent_path(child_alias: str, parent_alias: str) -> bool:
            child_l = str(child_alias or "").strip().lower()
            parent_l = str(parent_alias or "").strip().lower()
            if not child_l or not parent_l:
                return False
            if child_l == parent_l:
                return True
            seen = {child_l}
            queue = [child_l]
            q_idx = 0
            while q_idx < len(queue):
                cur = queue[q_idx]
                q_idx += 1
                for nxt in child_to_parents_alias.get(cur, set()):
                    if nxt == parent_l:
                        return True
                    if nxt in seen:
                        continue
                    seen.add(nxt)
                    queue.append(nxt)
            return False

        def _is_top_level_expr(node: Optional["exp.Expression"]) -> bool:
            if node is None:
                return False
            anc = getattr(node, "parent", None)
            while anc is not None:
                if anc is sel:
                    return True
                if isinstance(anc, exp.Subquery):
                    return False
                anc = getattr(anc, "parent", None)
            return False

        def _expr_aliases(expr_node: Optional["exp.Expression"]) -> Set[str]:
            aliases: Set[str] = set()
            if expr_node is None:
                return aliases
            for col in expr_node.find_all(exp.Column):
                if _is_nested_subquery_column(col):
                    continue
                alias_name = str(col.table or "").strip("`").lower()
                if alias_name:
                    aliases.add(alias_name)
            return aliases

        def _has_safe_ratio_rollup_pattern() -> bool:
            # Accept fanout-sensitive multi-table rollups when the SQL already
            # expresses average-like metric as SUM(...) / COUNT(DISTINCT parent_key)
            # over a valid FK parent chain.
            def _is_count_distinct_node(node: "exp.Count") -> bool:
                if not isinstance(node, exp.Count):
                    return False
                if bool(node.args.get("distinct")):
                    return True
                this_arg = node.args.get("this")
                return isinstance(this_arg, exp.Distinct)

            for div_node in sel.find_all(exp.Div):
                if not _is_top_level_expr(div_node):
                    continue

                sum_nodes = [
                    s for s in div_node.find_all(exp.Sum)
                    if _is_top_level_expr(s)
                ]
                count_nodes = [
                    c for c in div_node.find_all(exp.Count)
                    if _is_top_level_expr(c) and _is_count_distinct_node(c)
                ]
                if not sum_nodes or not count_nodes:
                    continue

                for sum_node in sum_nodes:
                    sum_aliases = _expr_aliases(sum_node.this)
                    if not sum_aliases:
                        continue
                    for count_node in count_nodes:
                        count_cols = [
                            c for c in count_node.find_all(exp.Column)
                            if not _is_nested_subquery_column(c)
                        ]
                        if len(count_cols) != 1:
                            continue
                        count_alias = str(count_cols[0].table or "").strip("`").lower()
                        if not count_alias:
                            continue
                        for sum_alias in sum_aliases:
                            if sum_alias == count_alias or _has_parent_path(sum_alias, count_alias):
                                return True
            return False

        def _has_safe_single_branch_rollup_pattern(
            agg_nodes: List["exp.AggFunc"],
        ) -> bool:
            sum_like_aliases: Set[str] = set()
            count_aliases: Set[str] = set()

            for agg in agg_nodes:
                alias_refs = _expr_aliases(agg)
                if not alias_refs:
                    continue
                if isinstance(agg, (exp.Sum, exp.Avg)):
                    sum_like_aliases.update(alias_refs)
                elif isinstance(agg, exp.Count):
                    count_aliases.update(alias_refs)

            if len(sum_like_aliases) != 1 or not count_aliases:
                return False

            sum_alias = next(iter(sum_like_aliases))
            return all(
                count_alias == sum_alias or _has_parent_path(sum_alias, count_alias)
                for count_alias in count_aliases
            )

        invalid_join_edges: List[str] = []
        joins = sel.args.get("joins") or []
        for j in joins:
            on_expr = j.args.get("on")
            if on_expr is None:
                continue
            eq_pairs: List[Tuple[str, str, str, str, str, str]] = []
            for eq in on_expr.find_all(exp.EQ):
                left = getattr(eq, "left", None)
                right = getattr(eq, "right", None)
                if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
                    continue
                lt_alias = str(left.table or "").strip("`").lower()
                rt_alias = str(right.table or "").strip("`").lower()
                lc = str(left.name or "").strip("`").lower()
                rc = str(right.name or "").strip("`").lower()
                if not lt_alias or not rt_alias or not lc or not rc:
                    continue
                lt = str(alias_to_table.get(lt_alias, lt_alias))
                rt = str(alias_to_table.get(rt_alias, rt_alias))
                if lt.startswith("__derived__:") or rt.startswith("__derived__:"):
                    continue
                _record_fk_alias_edge(lt_alias, lc, rt_alias, rc, lt, rt)
                eq_pairs.append((lt_alias, lc, rt_alias, rc, lt.lower(), rt.lower()))

            if not eq_pairs:
                continue

            if not any(_is_plausible_join_edge(lt, lc, rt, rc) for _, lc, _, rc, lt, rt in eq_pairs):
                p0 = eq_pairs[0]
                invalid_join_edges.append(f"{p0[0]}.{p0[1]} = {p0[2]}.{p0[3]}")

        if invalid_join_edges:
            verdict["invalid_join_edges"] = sorted(set(invalid_join_edges))
            verdict["reasons"].append("invalid_join_path")

        # Dual-fact fanout hard check:
        # If >1 direct transaction table contributes top-level aggregates,
        # require pre-aggregation via derived subqueries.
        txn_tables = set()
        try:
            txn_tables = {
                str(t or "").strip("`").lower()
                for t in (get_transaction_tables() or [])
                if str(t or "").strip()
            }
        except Exception:
            txn_tables = set()

        if txn_tables:
            direct_txn_alias_to_table = {
                a: t
                for a, t in explicit_alias_to_table.items()
                if not str(t).startswith("__derived__:") and str(t).lower() in txn_tables
            }
            if len(set(direct_txn_alias_to_table.values())) >= 2:
                group_arg = sel.args.get("group")
                top_level_aggs: List["exp.AggFunc"] = []
                for agg in sel.find_all(exp.AggFunc):
                    anc = getattr(agg, "parent", None)
                    inside_subquery = False
                    while anc is not None:
                        if anc is sel:
                            break
                        if isinstance(anc, exp.Subquery):
                            inside_subquery = True
                            break
                        anc = getattr(anc, "parent", None)
                    if not inside_subquery:
                        top_level_aggs.append(agg)

                txn_tables_in_metrics: Set[str] = set()
                for agg in top_level_aggs:
                    for col in agg.find_all(exp.Column):
                        if _is_nested_subquery_column(col):
                            continue
                        c_alias = str(col.table or "").strip("`").lower()
                        if not c_alias:
                            continue
                        c_table = str(alias_to_table.get(c_alias, c_alias)).lower()
                        if c_alias in direct_txn_alias_to_table and c_table in txn_tables:
                            txn_tables_in_metrics.add(c_table)

                has_group_or_agg = bool(group_arg is not None or top_level_aggs)
                safe_ratio_rollup = _has_safe_ratio_rollup_pattern()
                if has_group_or_agg and len(txn_tables_in_metrics) >= 2 and not safe_ratio_rollup:
                    verdict["fanout_risk_tables"] = sorted(txn_tables_in_metrics)
                    verdict["reasons"].append("fanout_risk_dual_fact")

        # Fallback dual-fact heuristic when transaction metadata is incomplete:
        # if top-level aggregates reference >=2 joined base tables with mixed
        # COUNT and SUM/AVG, this is highly likely fanout-prone.
        group_arg = sel.args.get("group")
        top_level_aggs_fb: List["exp.AggFunc"] = []
        for agg in sel.find_all(exp.AggFunc):
            anc = getattr(agg, "parent", None)
            inside_subquery = False
            while anc is not None:
                if anc is sel:
                    break
                if isinstance(anc, exp.Subquery):
                    inside_subquery = True
                    break
                anc = getattr(anc, "parent", None)
            if not inside_subquery:
                top_level_aggs_fb.append(agg)

        has_group_or_agg_fb = bool(group_arg is not None or top_level_aggs_fb)
        has_multi_join_fb = len(sel.args.get("joins") or []) >= 2
        if has_group_or_agg_fb and has_multi_join_fb:
            safe_ratio_rollup_fb = _has_safe_ratio_rollup_pattern()
            safe_single_branch_rollup_fb = _has_safe_single_branch_rollup_pattern(top_level_aggs_fb)
            agg_kinds_by_table: Dict[str, Set[str]] = {}
            for agg in top_level_aggs_fb:
                if isinstance(agg, exp.Count):
                    agg_kind = "COUNT"
                elif isinstance(agg, exp.Sum):
                    agg_kind = "SUM"
                elif isinstance(agg, exp.Avg):
                    agg_kind = "AVG"
                else:
                    agg_kind = "OTHER"

                for col in agg.find_all(exp.Column):
                    if _is_nested_subquery_column(col):
                        continue
                    c_alias = str(col.table or "").strip("`").lower()
                    if not c_alias:
                        continue
                    c_table = str(alias_to_table.get(c_alias, c_alias)).lower()
                    if c_table.startswith("__derived__:"):
                        continue
                    agg_kinds_by_table.setdefault(c_table, set()).add(agg_kind)

            if len(agg_kinds_by_table) >= 2:
                has_count = any("COUNT" in ks for ks in agg_kinds_by_table.values())
                has_sum_like = any(bool({"SUM", "AVG"} & ks) for ks in agg_kinds_by_table.values())
                if (
                    has_count
                    and has_sum_like
                    and not safe_ratio_rollup_fb
                    and not safe_single_branch_rollup_fb
                ):
                    existing = set(verdict.get("fanout_risk_tables") or [])
                    existing.update(agg_kinds_by_table.keys())
                    verdict["fanout_risk_tables"] = sorted(existing)
                    if "fanout_risk_dual_fact" not in verdict["reasons"]:
                        verdict["reasons"].append("fanout_risk_dual_fact")

        # Detect ambiguous unqualified columns only when >1 base table
        base_tables = [t for t in table_cols.keys() if t and not t.startswith("__derived__:")]
        if len(base_tables) > 1:
            col_to_owners: Dict[str, Set[str]] = {}
            for t, cols in table_cols.items():
                for c in cols:
                    col_to_owners.setdefault(c, set()).add(t)

            projection_aliases = {
                str(it.alias or "").strip("`").lower()
                for it in (sel.expressions or [])
                if isinstance(it, exp.Alias) and str(it.alias or "").strip()
            }

            def _is_order_by_ref(col_expr: "exp.Column") -> bool:
                anc = getattr(col_expr, "parent", None)
                while anc is not None:
                    if anc is sel:
                        return False
                    if isinstance(anc, exp.Ordered):
                        return True
                    anc = getattr(anc, "parent", None)
                return False

            ambiguous_cols: Set[str] = set()
            for col in sel.find_all(exp.Column):
                if _is_nested_subquery_column(col):
                    continue
                if str(col.table or "").strip():
                    continue
                cname = str(col.name or "").strip("`").lower()
                if not cname:
                    continue
                # ORDER BY can legally refer to SELECT alias; skip that case.
                if cname in projection_aliases and _is_order_by_ref(col):
                    continue
                owners = col_to_owners.get(cname, set())
                if len(owners) > 1:
                    ambiguous_cols.add(cname)

            if ambiguous_cols:
                verdict["ambiguous_unqualified_columns"] = sorted(ambiguous_cols)
                verdict["reasons"].append("ambiguous_unqualified")

        # GROUP BY projection consistency check (obvious 1055 risk)
        group_arg = sel.args.get("group")
        if group_arg is not None:
            group_norm = {_norm_expr(g) for g in (group_arg.expressions or [])}
            group_col_names = {
                str(g.name or "").strip("`").lower()
                for g in (group_arg.expressions or [])
                if isinstance(g, exp.Column)
            }
            projection_violations: Set[str] = set()
            for item in sel.expressions or []:
                expr_node = item.this if isinstance(item, exp.Alias) else item
                if expr_node is None:
                    continue
                if isinstance(expr_node, (exp.Star, exp.Literal)):
                    continue
                if expr_node.find(exp.AggFunc) or expr_node.find(exp.Window):
                    continue
                if isinstance(expr_node, exp.Column):
                    expr_norm = _norm_expr(expr_node)
                    expr_name = str(expr_node.name or "").strip("`").lower()
                    alias_name = str(item.alias_or_name or "").strip("`").lower()
                    if (
                        expr_norm not in group_norm
                        and expr_name not in group_col_names
                        and alias_name not in group_col_names
                    ):
                        projection_violations.add(expr_norm)

            if projection_violations:
                verdict["group_projection_violations"] = sorted(projection_violations)
                verdict["reasons"].append("group_projection_mismatch")

        semantic_failures = _evaluate_semantic_contract(
            sql=candidate_sql,
            question=question,
            parsed=parsed,
            sel=sel,
            exp_mod=exp,
            contract=dict(verdict.get("semantic_contract") or {}),
        )
        if semantic_failures:
            verdict["semantic_failures"] = semantic_failures
            verdict["reasons"].append("semantic_contract_mismatch")

    except Exception as e:
        verdict["parse_error"] = f"acceptance_gate_error:{e}"
        verdict["reasons"].append("acceptance_gate_internal_error")

    verdict["reasons"] = list(dict.fromkeys(str(r) for r in (verdict.get("reasons") or [])))
    verdict["sql"] = candidate_sql
    verdict["accepted"] = len(verdict["reasons"]) == 0
    return verdict


def _build_gate_retry_instruction(base_instruction: str, gate: Dict[str, Any]) -> str:
    lines: List[str] = []
    if base_instruction:
        lines.append(base_instruction.strip())
    lines.append(
        "STRICT REQUIREMENT: SQL failed deterministic acceptance checks. "
        "Fix every blocker below and return one valid MySQL SELECT query only."
    )

    reasons = gate.get("reasons") or []
    if reasons:
        lines.append(f"- Gate reasons: {reasons}")

    hard_errors = gate.get("hard_identifier_errors") or []
    if hard_errors:
        lines.append(f"- Identifier blockers: {hard_errors}")

    identifier_errors = gate.get("identifier_errors") or []
    if identifier_errors and not hard_errors:
        lines.append(f"- Identifier warnings: {identifier_errors}")

    amb_cols = gate.get("ambiguous_unqualified_columns") or []
    if amb_cols:
        lines.append(
            "- Ambiguous unqualified columns detected: "
            f"{amb_cols}. You MUST qualify them with explicit table aliases."
        )

    group_issues = gate.get("group_projection_violations") or []
    if group_issues:
        lines.append(
            "- GROUP BY mismatch: these raw projected columns are not grouped/aggregated: "
            f"{group_issues}."
        )

    join_warnings = gate.get("join_risk_warnings") or []
    if join_warnings:
        lines.append(f"- Suspicious join keys: {join_warnings}")

    invalid_join_edges = gate.get("invalid_join_edges") or []
    if invalid_join_edges:
        lines.append(
            "- Invalid join path detected: "
            f"{invalid_join_edges}. Use FK-consistent join keys only."
        )

    fanout_tables = gate.get("fanout_risk_tables") or []
    if fanout_tables:
        lines.append(
            "- Dual-fact fanout risk across transaction tables: "
            f"{fanout_tables}. Pre-aggregate each fact per key before joining."
        )

    semantic_failures = gate.get("semantic_failures") or []
    if semantic_failures:
        lines.append(
            "- Semantic contract mismatch: "
            f"{semantic_failures}. Match ORDER/LIMIT/aggregate requirements from the question."
        )
    semantic_contract = gate.get("semantic_contract") or {}
    if semantic_contract:
        semantic_parts: List[str] = []
        if semantic_contract.get("requires_order_by"):
            semantic_parts.append("ORDER BY is required")
        requested_order_terms = semantic_contract.get("requested_order_terms") or []
        if requested_order_terms:
            order_text = ", ".join(
                f"{phrase} {direction}".strip()
                for phrase, direction in requested_order_terms
                if str(phrase or "").strip()
            )
            if order_text:
                semantic_parts.append(f"Requested final order sequence: {order_text}")
        if semantic_contract.get("required_limit") is not None:
            semantic_parts.append(f"LIMIT {semantic_contract.get('required_limit')} is required")
        req_aggs = semantic_contract.get("required_aggs") or []
        if req_aggs:
            semantic_parts.append(f"Required aggregates: {req_aggs}")
        if semantic_parts:
            lines.append("- Semantic requirements: " + "; ".join(semantic_parts))

    parse_error = gate.get("parse_error") or ""
    if parse_error:
        lines.append(f"- Parse error: {parse_error}")

    return "\n".join(lines)


def _is_critical_gate_failure(gate: Dict[str, Any]) -> bool:
    reasons = {str(r) for r in (gate.get("reasons") or [])}
    critical_reasons = {
        "parse_error",
        "no_select",
        "hard_identifier",
        "ambiguous_unqualified",
        "group_projection_mismatch",
        "invalid_join_path",
        "fanout_risk_dual_fact",
    }
    if reasons.intersection(critical_reasons):
        return True
    return bool(gate.get("parse_error"))


def _has_balanced_sql_delimiters(sql: str) -> bool:
    depth = 0
    in_sq = False
    in_dq = False
    i = 0
    s = str(sql or "")
    while i < len(s):
        ch = s[i]
        if ch == "'" and not in_dq:
            if in_sq and i + 1 < len(s) and s[i + 1] == "'":
                i += 2
                continue
            in_sq = not in_sq
            i += 1
            continue
        if ch == '"' and not in_sq:
            in_dq = not in_dq
            i += 1
            continue
        if in_sq or in_dq:
            i += 1
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        i += 1
    return depth == 0 and not in_sq and not in_dq


def _can_bypass_parse_only_gate(
    *,
    sql: str,
    ctx: str,
    decision: Dict[str, Any],
) -> bool:
    def _is_safe_single_table_grouped_aggregate(candidate_sql: str) -> bool:
        stmt = str(candidate_sql or "").strip().rstrip(";").strip()
        if not stmt:
            return False
        if has_join(stmt):
            return False
        if re.search(r"\(\s*SELECT\b", stmt, flags=re.I):
            return False
        if re.search(r"\bUNION(?:\s+ALL)?\b", stmt, flags=re.I):
            return False
        if re.search(r"\bOVER\s*\(", stmt, flags=re.I):
            return False

        from_pos = _find_top_level_kw(stmt, "from")
        if from_pos < 0:
            return False
        group_pos = _find_top_level_kw(stmt, "group by", from_pos + len("from"))
        if group_pos < 0:
            return False

        select_part = stmt[len("select"):from_pos].strip()
        if not re.search(r"\b(?:count|sum|avg|min|max)\s*\(", select_part, flags=re.I):
            return False

        where_pos = _find_top_level_kw(stmt, "where", from_pos + len("from"))
        from_end_candidates = [p for p in [where_pos, group_pos] if p >= 0]
        from_end = min(from_end_candidates) if from_end_candidates else len(stmt)
        from_core = stmt[from_pos + len("from"):from_end].strip()
        if not from_core:
            return False
        if re.search(r"\bJOIN\b", from_core, flags=re.I):
            return False
        if len([part for part in _split_top_level_csv(from_core) if part.strip()]) != 1:
            return False

        order_pos = _find_top_level_kw(stmt, "order by", group_pos + len("group by"))
        having_pos = _find_top_level_kw(stmt, "having", group_pos + len("group by"))
        limit_pos = _find_top_level_kw(stmt, "limit", group_pos + len("group by"))
        group_end_candidates = [p for p in [having_pos, order_pos, limit_pos] if p >= 0]
        group_end = min(group_end_candidates) if group_end_candidates else len(stmt)
        group_clause = stmt[group_pos + len("group by"):group_end].strip()
        if not group_clause:
            return False
        if order_pos < 0 and re.search(r"\b(?:asc|desc)\b\s*$", group_clause, flags=re.I):
            return False

        group_items = [item.strip() for item in _split_top_level_csv(group_clause) if item.strip()]
        return bool(group_items)

    s = str(sql or "").strip()
    if not s or not re.match(r"^\s*SELECT\b", s, flags=re.I):
        return False
    if not _has_balanced_sql_delimiters(s):
        return False
    if re.search(AGENT_REFINE_BROKEN_ALIAS_PATTERN, s.lower()):
        return False

    identifier_errors = validate_identifiers(s, ctx)
    if identifier_errors or _collect_hard_identifier_errors(identifier_errors):
        return False

    has_window = bool(re.search(r"\bOVER\s*\(", s, flags=re.I))
    has_subquery = bool(re.search(r"\(\s*SELECT\b", s, flags=re.I))
    has_set_op = bool(re.search(r"\bUNION(?:\s+ALL)?\b", s, flags=re.I))
    has_top_level_join = has_join(s)

    if has_window and not has_top_level_join and bool(decision.get("is_window_aggregate", False)):
        return True
    if has_subquery and not has_top_level_join and bool(decision.get("is_simple_scalar_subquery_single_table", False)):
        return True
    if has_set_op and not has_top_level_join and bool(decision.get("is_set_operation_intent", False)):
        return True
    if _is_safe_single_table_grouped_aggregate(s):
        return True
    return False


def _allowed_identifiers_snapshot(ctx: str, max_tables: int = 4, max_cols: int = 12) -> str:
    allowed_map = _parse_allowed_identifiers(ctx)
    if not allowed_map:
        return ""
    lines: List[str] = []
    for i, (tbl, cols) in enumerate(allowed_map.items()):
        if i >= max_tables:
            break
        cols_sorted = sorted(str(c) for c in (cols or set()) if str(c).strip())
        if len(cols_sorted) > max_cols:
            cols_sorted = cols_sorted[:max_cols]
        lines.append(f"- {tbl}: {', '.join(cols_sorted)}")
    if not lines:
        return ""
    return "ALLOWED IDENTIFIERS SNAPSHOT:\n" + "\n".join(lines)


def _attempt_last_chance_rescue_7b(
    *,
    question: str,
    ctx: str,
    current_sql: str,
    gate: Dict[str, Any],
    strict_join_key: bool = False,
    router_decision: Optional[Dict[str, Any]] = None,
    state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    One-shot rescue with 7B when final gate fails critically.
    Returns a payload with the rescued SQL and gate verdict.
    """
    result: Dict[str, Any] = {
        "used": False,
        "accepted": False,
        "sql": (current_sql or "").strip(),
        "gate": gate,
        "error": "",
    }

    if not _is_critical_gate_failure(gate):
        return result

    result["used"] = True
    llm_rescue = build_llm_repair(model=LARGE_MODEL)
    _dbg_print_two("[REFINER] Last-chance rescue", "Regenerating with 7B")

    rescue_instruction = _build_gate_retry_instruction(
        "LAST-CHANCE RESCUE: Rewrite SQL to satisfy deterministic acceptance gate. Return one valid MySQL SELECT only.",
        gate,
    )
    allowed_snapshot = _allowed_identifiers_snapshot(ctx)
    if allowed_snapshot:
        rescue_instruction = f"{rescue_instruction}\n\n{allowed_snapshot}"

    rescued_sql, _ = regenerate_with_extra_constraint(
        llm_rescue,
        make_generation_prompt,
        ctx,
        question,
        rescue_instruction,
        prompt_kwargs={
            "router_decision": dict(router_decision or {}),
            "state": state,
        },
    )
    if not rescued_sql:
        result["error"] = "rescue_empty_sql"
        return result

    candidate_sql = normalize_sql_pipeline(str(rescued_sql).strip(), question, ctx)
    rescue_gate = _deterministic_sql_acceptance_gate(
        candidate_sql,
        ctx=ctx,
        question=question,
        strict_join_key=strict_join_key,
    )
    candidate_sql = str(rescue_gate.get("sql") or candidate_sql).strip()
    if candidate_sql and not candidate_sql.endswith(";"):
        candidate_sql += ";"

    result["sql"] = candidate_sql
    result["gate"] = rescue_gate
    result["accepted"] = bool(rescue_gate.get("accepted", False))
    if not result["accepted"]:
        result["error"] = "rescue_gate_failed"
    return result


_LLM_SEMANTIC_RETRY_FAILURE_PREFIXES = (
    "limit_mismatch:",
    "missing_order_keys:",
    "order_key_mismatch:",
    "order_dir_mismatch:",
)
_LLM_SEMANTIC_RETRY_FAILURES = {
    "missing_order_by",
    "missing_limit",
    "unparsed_limit",
    "missing_agg_count",
    "missing_agg_sum",
    "missing_agg_avg",
}


def _is_llm_semantic_retry_gate(gate: Dict[str, Any]) -> bool:
    if not gate or bool(gate.get("accepted", False)):
        return False
    if _is_critical_gate_failure(gate):
        return False

    reasons = {str(r) for r in (gate.get("reasons") or []) if str(r).strip()}
    if not reasons or not reasons.issubset({"semantic_contract_mismatch"}):
        return False

    failures = [str(f) for f in (gate.get("semantic_failures") or []) if str(f).strip()]
    if not failures:
        return False

    for failure in failures:
        if failure in _LLM_SEMANTIC_RETRY_FAILURES:
            continue
        if any(failure.startswith(prefix) for prefix in _LLM_SEMANTIC_RETRY_FAILURE_PREFIXES):
            continue
        return False
    return True


def _prewarm_model_with_oom_retry(llm, target_model: str, max_retries: int = OOM_MAX_RETRIES) -> bool:
        global _last_model_warmup
        # Get keep_alive in seconds
        import re
        from .config import OLLAMA_KEEP_ALIVE
        m = re.match(AGENT_KEEP_ALIVE_PARSE_PATTERN, str(OLLAMA_KEEP_ALIVE).lower())
        if m:
            val, unit = int(m.group(1)), m.group(2)
            scale = AGENT_KEEP_ALIVE_UNIT_SECONDS.get(unit)
            if scale is not None:
                keep_alive_sec = val * scale
            else:
                keep_alive_sec = AGENT_KEEP_ALIVE_DEFAULT_SECONDS
        else:
            keep_alive_sec = AGENT_KEEP_ALIVE_DEFAULT_SECONDS

        now = _time.time()
        last_warmup = _last_model_warmup.get(target_model, 0)
        idle_time = now - last_warmup
        # Only warmup if idle_time > keep_alive
        if idle_time < keep_alive_sec:
            print(f"[STATUS] Model {target_model} still warm (idle {idle_time:.1f}s < keep_alive {keep_alive_sec}s), skip warmup.")
            return True

        """
        Pre-warm model with OOM retry logic.
        Returns True if model is ready, False if failed after all retries.
        """
        for attempt in range(max_retries):
            try:
                print(f"[STATUS] Loading model: {target_model}..." + (f" (attempt {attempt+1})" if attempt > 0 else ""))
                llm.invoke("Say OK")  # Quick call to load model into memory
                print(f"[STATUS] Model ready: {target_model}")
                _last_model_warmup[target_model] = _time.time()
                return True
            except Exception as e:
                error_msg = str(e).lower()
                if AGENT_PREWARM_MEMORY_ERROR_TOKEN in error_msg:
                    print(f"[STATUS] Memory issue detected. Waiting {OOM_WAIT_TIME}s for previous model to unload...")
                    _time.sleep(OOM_WAIT_TIME)
                    print(f"[STATUS] Retrying model load...")
                else:
                    print(f"[STATUS] Pre-warm failed: {e}")
                    return False
        
        print(f"[STATUS] Failed to load model after {max_retries} OOM retries")
        return False


_IR_HAVING_THRESHOLD_OPS = {"eq", "gt", "gte", "lt", "lte", "between"}
_IR_HAVING_ALLOWED_AGGS = {"count", "sum", "avg", "min", "max"}


def _is_numeric_literal(v: Any) -> bool:
    if isinstance(v, (int, float)):
        return True
    s = str(v or "").strip().replace(",", "")
    if not s:
        return False
    try:
        float(s)
        return True
    except Exception:
        return False


def _is_ir_metric_threshold_filter(f: Dict[str, Any], m: Dict[str, Any]) -> bool:
    if not isinstance(f, dict) or not isinstance(m, dict):
        return False

    op = str(f.get("op") or "").strip().lower()
    if op not in _IR_HAVING_THRESHOLD_OPS:
        return False

    agg = str(m.get("agg") or "").strip().lower()
    if agg not in _IR_HAVING_ALLOWED_AGGS:
        return False

    if op == "between":
        if not (_is_numeric_literal(f.get("value")) and _is_numeric_literal(f.get("value2"))):
            return False
    elif not _is_numeric_literal(f.get("value")):
        return False

    f_table = str(f.get("table") or "").strip().lower()
    m_table = str(m.get("table") or "").strip().lower()
    if not f_table or not m_table or f_table != m_table:
        return False

    f_col = str(f.get("column") or "").strip().lower()
    m_col = str(m.get("column") or "").strip().lower()
    if not f_col or not m_col:
        return False
    if m_col == "*":
        return f_col == "*"
    return f_col == m_col


def _collect_ir_metric_threshold_pairs(intent_ir: Dict[str, Any]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    metrics = [m for m in (intent_ir.get("metrics") or []) if isinstance(m, dict)]
    filters = [f for f in (intent_ir.get("filters") or []) if isinstance(f, dict)]
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    seen: Set[Tuple[str, str, str, str, str, str, str, str, str]] = set()
    for f in filters:
        for m in metrics:
            if not _is_ir_metric_threshold_filter(f, m):
                continue
            key = (
                str(f.get("table") or "").strip().lower(),
                str(f.get("column") or "").strip().lower(),
                str(f.get("op") or "").strip().lower(),
                str(f.get("value")),
                str(f.get("value2")),
                str(m.get("agg") or "").strip().lower(),
                str(m.get("table") or "").strip().lower(),
                str(m.get("column") or "").strip().lower(),
                str(bool(m.get("distinct"))),
            )
            if key in seen:
                continue
            seen.add(key)
            pairs.append((f, m))
    return pairs


def _is_simple_having_shape(
    intent_ir: Dict[str, Any],
    decision: Dict[str, Any],
    threshold_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]],
) -> Tuple[bool, str]:
    metrics = [m for m in (intent_ir.get("metrics") or []) if isinstance(m, dict)]
    dims = [d for d in (intent_ir.get("dimensions") or []) if isinstance(d, dict)]
    existence = [e for e in (intent_ir.get("existence") or []) if isinstance(e, dict)]
    filters = [f for f in (intent_ir.get("filters") or []) if isinstance(f, dict)]

    if str(intent_ir.get("qtype") or "").strip().lower() != AGENT_ROUTER_QTYPE_AGGREGATE:
        return False, "simple_having_requires_aggregate_qtype"
    if len(metrics) != 1:
        return False, "simple_having_requires_single_metric"
    if not dims:
        return False, "simple_having_requires_group_dimension"
    if existence:
        return False, "simple_having_with_existence_not_supported"
    if len(threshold_pairs) == 0:
        return False, "simple_having_no_metric_threshold"

    max_filters = max(1, int(AGENT_IR_SIMPLE_HAVING_MAX_FILTERS))
    if len(threshold_pairs) > max_filters:
        return False, f"simple_having_too_many_threshold_filters:{len(threshold_pairs)}>{max_filters}"

    referenced_tables: Set[str] = set()

    def _add_table_refs(items: List[Dict[str, Any]]) -> None:
        for item in items:
            table = str(item.get("table") or "").strip().lower()
            if table:
                referenced_tables.add(table)

    _add_table_refs(metrics)
    _add_table_refs(dims)
    _add_table_refs(filters)
    _add_table_refs(existence)
    subject_table = str(intent_ir.get("subject_entity") or "").strip().lower()
    if subject_table:
        referenced_tables.add(subject_table)
    if len(referenced_tables) > 1:
        return False, "simple_having_multi_table_scope"

    m0 = metrics[0]
    m0_sig = (
        str(m0.get("agg") or "").strip().lower(),
        str(m0.get("table") or "").strip().lower(),
        str(m0.get("column") or "").strip().lower(),
        bool(m0.get("distinct")),
    )
    for _, m in threshold_pairs:
        sig = (
            str(m.get("agg") or "").strip().lower(),
            str(m.get("table") or "").strip().lower(),
            str(m.get("column") or "").strip().lower(),
            bool(m.get("distinct")),
        )
        if sig != m0_sig:
            return False, "simple_having_multiple_metric_targets"

    if bool(decision.get("has_dual_fact_rollup", False)):
        return False, "simple_having_dual_fact_rollup"

    return True, ""


def _is_ir_low_risk_single_table_lookup(
    intent_ir: Dict[str, Any],
    qtype: str,
    safe_filter_ops: Set[str],
) -> bool:
    if str(qtype or "").strip().lower() != AGENT_ROUTER_QTYPE_LOOKUP:
        return False
    if not bool(intent_ir.get("planner_compatible", True)):
        return False

    subject_table = str(intent_ir.get("subject_entity") or "").strip().lower()
    if not subject_table:
        return False

    metrics = [m for m in (intent_ir.get("metrics") or []) if isinstance(m, dict)]
    existence = [e for e in (intent_ir.get("existence") or []) if isinstance(e, dict)]
    if metrics or existence:
        return False

    dims = [d for d in (intent_ir.get("dimensions") or []) if isinstance(d, dict)]
    filters = [f for f in (intent_ir.get("filters") or []) if isinstance(f, dict)]
    ordering = [o for o in (intent_ir.get("ordering") or []) if isinstance(o, dict)]

    visible_dim_aliases = {
        str(d.get("alias") or "").strip().lower()
        for d in dims
        if bool(d.get("visible", True)) and str(d.get("alias") or "").strip()
    }

    for d in dims:
        table = str(d.get("table") or "").strip().lower()
        if not table or table != subject_table:
            return False

    for f in filters:
        table = str(f.get("table") or "").strip().lower()
        op = str(f.get("op") or "").strip().lower()
        if not table or table != subject_table:
            return False
        if op and op not in safe_filter_ops:
            return False

    for o in ordering:
        kind = str(o.get("kind") or "").strip().lower()
        if kind == "column":
            table = str(o.get("table") or "").strip().lower()
            if not table or table != subject_table:
                return False
            continue
        if kind == "alias":
            by = str(o.get("by") or "").strip().lower()
            if not by or by not in visible_dim_aliases:
                return False
            continue
        return False

    return True


def _ir_regex_family_status(intent_ir: Dict[str, Any], question: str) -> Tuple[bool, bool]:
    """
    Decide whether regex intent is deterministic enough for IR.

    Returns:
      (regex_is_ir_safe, has_regex_family)
    """
    filters = [f for f in (intent_ir.get("filters") or []) if isinstance(f, dict)]
    regex_filters = [
        f for f in filters
        if str(f.get("op") or "").strip().lower() == "regexp"
    ]

    requires_boundary = _question_requires_word_boundary_regex(question)
    requires_regex = _question_requires_regex(question)
    requires_structural_regex = _question_requires_structural_regex(question)
    has_regex_family = bool(requires_boundary or requires_regex or regex_filters)
    if not has_regex_family:
        return True, False

    # Broad/open regex stays LLM-first.
    if requires_boundary:
        return False, True
    if requires_structural_regex:
        return False, True

    # If the question explicitly requires regex but intent_ir did not resolve a
    # regexp filter, do not force IR.
    if requires_regex and not regex_filters:
        return False, True

    # Only allow narrow, template-generated regex forms in IR.
    deterministic_re = re.compile(
        r"^\^(?:[A-Za-z0-9_\\-]+|\[A-Z\]\{\d+\}|\[0-9\]\{\d+\}|\[0-9\])+\$$"
    )
    for f in regex_filters:
        value = str(f.get("value") or "")
        if not value or not deterministic_re.match(value):
            return False, True

    return True, True


def _question_requests_projection_null_fill(question: str) -> bool:
    q_raw = str(question or "")
    q_low = q_raw.lower()
    if not any_phrase_in_text(q_low, projection_null_fill_missing_terms()):
        return False
    if not any_phrase_in_text(q_low, projection_null_fill_conditional_terms()):
        return False
    if not any_phrase_in_text(q_low, projection_null_fill_action_terms()):
        return False
    if re.search(r"""["'“”][^"'“”]{1,40}["'“”]""", q_raw):
        return True
    return any_phrase_in_text(q_low, projection_null_fill_fallback_terms())


def _question_has_numeric_threshold(question: str) -> bool:
    q = str(question or "").strip().lower()
    if not q:
        return False
    comparison_alt = _phrase_alt_pattern(numeric_threshold_comparison_terms())
    number_words_alt = _phrase_alt_pattern(numeric_threshold_number_words())
    if number_words_alt:
        number_pat = rf"(?:\d[\d,]*(?:\.\d+)?|{number_words_alt})"
    else:
        number_pat = r"(?:\d[\d,]*(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
    return bool(
        comparison_alt and re.search(
            rf"\b(?:{comparison_alt})\s+{number_pat}\b",
            q,
            flags=re.I,
        )
    )


def _question_has_global_avg_baseline_compare(question: str) -> bool:
    q = str(question or "").strip().lower()
    if not q:
        return False
    avg_alt = _phrase_alt_pattern(global_avg_baseline_average_terms())
    baseline_alt = _phrase_alt_pattern(global_avg_baseline_terms())
    comparison_alt = _phrase_alt_pattern(global_avg_comparison_terms())
    baseline_ref_alt = _phrase_alt_pattern(global_avg_baseline_reference_terms())
    has_avg = bool(avg_alt and re.search(rf"\b(?:{avg_alt})\b", q, flags=re.I))
    has_baseline = bool(baseline_alt and re.search(rf"\b(?:{baseline_alt})\b", q, flags=re.I))
    has_cmp = bool(
        (
            comparison_alt
            and re.search(
                rf"\b(?:{comparison_alt})\s+than\b",
                q,
                flags=re.I,
            )
        )
        or (
            comparison_alt
            and baseline_ref_alt
            and re.search(
                rf"\b(?:{comparison_alt})\b[^.]{{0,32}}\b(?:{baseline_ref_alt})\b",
                q,
                flags=re.I,
            )
        )
    )
    return has_avg and has_baseline and has_cmp


def _question_has_offset_window_sequence_intent(question: str) -> bool:
    q = str(question or "").strip().lower()
    if not q:
        return False
    if not any_phrase_in_text(q, offset_window_sequence_terms()):
        return False
    ordering_context_patterns = [
        str(p).strip()
        for p in offset_window_ordering_context_regex_patterns()
        if str(p).strip()
    ]
    return any(re.search(pat, q, flags=re.I) for pat in ordering_context_patterns)


def _question_has_case_bucketing_intent(question: str) -> bool:
    q = str(question or "").strip().lower()
    if not q:
        return False

    bucket_language_patterns = [
        str(p).strip()
        for p in case_bucket_language_regex_patterns()
        if str(p).strip()
    ]
    has_bucket_language = any(re.search(pat, q, flags=re.I) for pat in bucket_language_patterns)
    if not has_bucket_language:
        return False

    numeric_threshold_patterns = [
        str(p).strip()
        for p in case_bucket_numeric_threshold_regex_patterns()
        if str(p).strip()
    ]
    numeric_term_alt = _phrase_alt_pattern(case_bucket_numeric_terms())
    has_numeric_threshold_shape = bool(
        any(re.search(pat, q, flags=re.I) for pat in numeric_threshold_patterns)
        or (numeric_term_alt and re.search(rf"\b(?:{numeric_term_alt})\b", q, flags=re.I))
    )

    comparison_alt = _phrase_alt_pattern(case_bucket_comparison_terms())
    temporal_alt = _phrase_alt_pattern(case_bucket_temporal_terms())
    pattern_alt = _phrase_alt_pattern(case_bucket_pattern_terms())
    has_conditional_bucket_shape = bool(
        any_phrase_in_text(q, case_bucket_conditional_terms())
        and (
            any_phrase_in_text(q, case_bucket_missing_terms())
            or (comparison_alt and re.search(rf"\b(?:{comparison_alt})\b", q, flags=re.I))
            or (temporal_alt and re.search(rf"\b(?:{temporal_alt})\b", q, flags=re.I))
            or (pattern_alt and re.search(rf"\b(?:{pattern_alt})\b", q, flags=re.I))
        )
    )
    if not (has_numeric_threshold_shape or has_conditional_bucket_shape):
        return False

    aggregate_request_patterns = [
        str(p).strip()
        for p in case_bucket_aggregate_request_regex_patterns()
        if str(p).strip()
    ]
    return bool(
        any(re.search(pat, q, flags=re.I) for pat in aggregate_request_patterns)
        or any_phrase_in_text(q, case_bucket_aggregate_request_terms())
    )


def _make_case_bucketing_direct_text_prompt(base_prompt: str, question: str, ctx: str) -> str:
    """
    Text-mode prompt for CASE-style bucket aggregations.

    Small JSON-mode models often produce valid SQL but malformed JSON for long
    CASE expressions. Keeping this as a narrow intent-family override avoids
    changing the normal draft path for non-bucketing queries.
    """
    prompt_lines = [
        str(base_prompt or "").rstrip(),
        "",
        "TEMPORARY OUTPUT OVERRIDE FOR CASE BUCKET AGGREGATION:",
        "Return ONLY one valid MySQL SELECT statement as plain text.",
        "Do NOT return JSON. Do NOT return notes. Do NOT use Markdown.",
        "Use a single SELECT with CASE WHEN ... THEN ... ELSE ... END for the bucket label.",
        "Put bucket conditions inside the CASE expression, not as separate WHERE filters, unless the question explicitly asks for an additional row filter.",
        "For ranges like 'between X and less than Y', use `col >= X AND col < Y`; do NOT use `BETWEEN X AND Y` because BETWEEN includes Y.",
        "Do NOT use UNION/UNION ALL unless the question explicitly asks to combine separate result sets.",
        "When grouping by the bucket, repeat the full CASE expression in GROUP BY instead of grouping by the alias.",
        "Include every aggregate output requested by the question, such as COUNT, MIN, MAX, SUM, or AVG.",
        "Return exactly one SQL statement; use only one semicolon, at the very end.",
    ]
    allowed_snapshot = _allowed_identifiers_snapshot(ctx, max_tables=2, max_cols=10)
    if allowed_snapshot:
        prompt_lines.extend(["", allowed_snapshot])
    prompt_lines.extend(["", f"Question: {str(question or '').strip()}"])
    return "\n".join(prompt_lines).strip()


def _make_sql_only_draft_prompt(base_prompt: str) -> str:
    p = str(base_prompt or "").rstrip()
    return (
        f"{p}\n\n"
        "TEMPORARY OUTPUT OVERRIDE:\n"
        "Ignore any JSON requirement above.\n"
        "Return ONLY the final MySQL query as plain text on a single line.\n"
        "Do NOT return JSON. Do NOT return notes. Do NOT use Markdown.\n"
        "End the SQL with a semicolon."
    )


def _make_case_bucketing_ultra_minimal_prompt(question: str, ctx: str) -> str:
    prompt_lines = [
        "Return exactly one valid MySQL SELECT query on a single line.",
        "Do NOT return JSON. Do NOT return notes. Do NOT use Markdown.",
        f"Question: {str(question or '').strip()}",
        "Use CASE WHEN ... THEN ... ELSE ... END for the bucket labels.",
        "For half-open ranges like 'between X and less than Y', use `col >= X AND col < Y`, not BETWEEN.",
        "If grouping by the bucket, repeat the full CASE expression inside GROUP BY instead of using the alias.",
        "If the question asks for counts per bucket, return the bucket label column and a COUNT(...) metric.",
        "If the question asks for descending count order, sort by COUNT(...) DESC.",
    ]

    allowed_snapshot = _allowed_identifiers_snapshot(ctx, max_tables=2, max_cols=10)
    if allowed_snapshot:
        prompt_lines.append(allowed_snapshot)

    prompt_lines.append("End the SQL with a semicolon.")
    return "\n".join(prompt_lines)


def _make_case_bucketing_micro_prompt(question: str, ctx: str) -> str:
    q = str(question or "").strip()
    prompt_lines = [
        "Write one valid MySQL SELECT query for a single-table numeric bucket aggregation.",
        "Return plain SQL only on one line.",
        "Do NOT return JSON. Do NOT return notes. Do NOT use Markdown.",
        f"Question: {q}",
        "Use CASE WHEN ... THEN ... ELSE ... END to assign the bucket labels requested by the question.",
        "For half-open ranges like 'between X and less than Y', use `col >= X AND col < Y`, not BETWEEN.",
        "Return exactly two outputs: the bucket label and a count metric.",
        "Repeat the full CASE expression inside GROUP BY instead of using the alias.",
        "If descending count order is requested, use ORDER BY COUNT(...) DESC.",
    ]

    allowed_map = _parse_allowed_identifiers(ctx or "")
    if len(allowed_map) == 1:
        table_name, cols = next(iter(allowed_map.items()))
        cols_sorted = sorted(str(c) for c in (cols or set()) if str(c).strip())
        if table_name:
            prompt_lines.append(f"Use this table only: {table_name}")
        if cols_sorted:
            prompt_lines.append(f"Allowed columns: {', '.join(cols_sorted[:8])}")
    else:
        allowed_snapshot = _allowed_identifiers_snapshot(ctx, max_tables=2, max_cols=8)
        if allowed_snapshot:
            prompt_lines.append(allowed_snapshot)

    prompt_lines.append("End the SQL with a semicolon.")
    return "\n".join(prompt_lines)


def _recover_missing_outer_winner_row_columns(sql: str, missing_cols: List[str]) -> Optional[str]:
    s = str(sql or "").strip()
    if not s or not missing_cols:
        return None
    if "ROW_NUMBER" not in s.upper():
        return None

    pattern = re.compile(
        r"(?is)^\s*SELECT\s+(?P<outer>.*?)\s+FROM\s*\(\s*SELECT\s+(?P<inner>.*?)\s+FROM\s+(?P<rest>.+?)\)\s*(?:AS\s+)?(?P<alias>[A-Za-z_][A-Za-z0-9_]*)\s+WHERE\s+(?P=alias)\.rn\s*=\s*\d+\s*;?\s*$"
    )
    m = pattern.match(s)
    if not m:
        return None

    def _split_top_level_csv(text: str) -> List[str]:
        items: List[str] = []
        buf: List[str] = []
        depth = 0
        in_sq = False
        in_dq = False
        i = 0
        while i < len(text):
            ch = text[i]
            if ch == "'" and not in_dq:
                if in_sq and i + 1 < len(text) and text[i + 1] == "'":
                    buf.append(ch)
                    buf.append(text[i + 1])
                    i += 2
                    continue
                in_sq = not in_sq
                buf.append(ch)
                i += 1
                continue
            if ch == '"' and not in_sq:
                in_dq = not in_dq
                buf.append(ch)
                i += 1
                continue
            if not in_sq and not in_dq:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth = max(0, depth - 1)
                elif ch == "," and depth == 0:
                    item = "".join(buf).strip()
                    if item:
                        items.append(item)
                    buf = []
                    i += 1
                    continue
            buf.append(ch)
            i += 1
        tail = "".join(buf).strip()
        if tail:
            items.append(tail)
        return items

    def _extract_output_label(select_item: str) -> str:
        item = str(select_item or "").strip()
        if not item:
            return ""
        alias_match = re.search(
            r"""(?is)\bAS\s+(`[^`]+`|'[^']+'|"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)\s*$""",
            item,
        )
        if alias_match:
            return alias_match.group(1).strip("`'\"").lower()
        simple_match = re.match(
            r"(?is)^\s*(?:`?[A-Za-z_][A-Za-z0-9_]*`?\s*\.\s*)?`?([A-Za-z_][A-Za-z0-9_]*)`?\s*$",
            item,
        )
        return (simple_match.group(1).lower() if simple_match else "")

    outer_select = m.group("outer").strip()
    inner_select = m.group("inner").strip()
    alias = m.group("alias").strip()

    outer_items = _split_top_level_csv(outer_select)
    inner_items = _split_top_level_csv(inner_select)
    outer_labels = {lbl for lbl in (_extract_output_label(it) for it in outer_items) if lbl}
    inner_labels = {lbl for lbl in (_extract_output_label(it) for it in inner_items) if lbl}

    prepend_items: List[str] = []
    for col_raw in missing_cols:
        col = str(col_raw or "").strip().strip("`").strip("'").strip('"')
        if not col:
            continue
        col_l = col.lower()
        if col_l in outer_labels:
            continue
        if col_l not in inner_labels:
            continue
        prepend_items.append(f"{alias}.{col}")
        outer_labels.add(col_l)

    if not prepend_items:
        return None

    rebuilt_outer = ", ".join(prepend_items + outer_items)
    fixed = re.sub(
        r"(?is)^\s*SELECT\s+.*?\s+FROM\s*\(",
        f"SELECT {rebuilt_outer} FROM (",
        s,
        count=1,
    ).strip()
    if sql.rstrip().endswith(";") and not fixed.endswith(";"):
        fixed += ";"
    return fixed


def _recover_missing_joined_lookup_columns(
    sql: str,
    missing_cols: List[str],
    question: str,
    ctx: str = "",
) -> Optional[str]:
    """
    Deterministically recover missing output columns for flat joined lookup SQL.

    Scope is intentionally narrow:
    - top-level SELECT ... FROM ... with JOINs
    - no GROUP BY/HAVING/UNION/WINDOW/subquery shapes
    - add at most one-hop FK LEFT JOIN per missing column
    """
    s = str(sql or "").strip()
    if not s or not missing_cols:
        return None
    if not has_join(s):
        return None
    if re.search(r"\b(?:group\s+by|having|union|intersect|except|with)\b", s, flags=re.I):
        return None
    if re.search(r"\bOVER\s*\(", s, flags=re.I):
        return None
    if re.search(r"(?is)\b(?:from|exists|in)\s*\(\s*select\b", s):
        return None

    s_core = s.rstrip().rstrip(";").strip()
    if not re.match(r"(?is)^SELECT\b", s_core):
        return None

    def _find_top_level_keyword(text: str, keyword: str, start: int = 0) -> int:
        kw = str(keyword or "").upper()
        if not kw:
            return -1
        t = str(text or "")
        up = t.upper()
        depth = 0
        in_sq = False
        in_dq = False
        i = max(0, int(start))
        while i < len(t):
            ch = t[i]
            if ch == "'" and not in_dq:
                if in_sq and i + 1 < len(t) and t[i + 1] == "'":
                    i += 2
                    continue
                in_sq = not in_sq
                i += 1
                continue
            if ch == '"' and not in_sq:
                in_dq = not in_dq
                i += 1
                continue
            if in_sq or in_dq:
                i += 1
                continue
            if ch == "(":
                depth += 1
                i += 1
                continue
            if ch == ")":
                depth = max(0, depth - 1)
                i += 1
                continue
            if depth == 0 and up.startswith(kw, i):
                left_ok = i == 0 or not up[i - 1].isalnum()
                right_idx = i + len(kw)
                right_ok = right_idx >= len(up) or not up[right_idx].isalnum()
                if left_ok and right_ok:
                    return i
            i += 1
        return -1

    def _split_top_level_csv(text: str) -> List[str]:
        items: List[str] = []
        buf: List[str] = []
        depth = 0
        in_sq = False
        in_dq = False
        i = 0
        while i < len(text):
            ch = text[i]
            if ch == "'" and not in_dq:
                if in_sq and i + 1 < len(text) and text[i + 1] == "'":
                    buf.append(ch)
                    buf.append(text[i + 1])
                    i += 2
                    continue
                if in_sq and i > 0 and text[i - 1] == "\\":
                    buf.append(ch)
                    i += 1
                    continue
                in_sq = not in_sq
                buf.append(ch)
                i += 1
                continue
            if ch == '"' and not in_sq:
                if in_dq and i > 0 and text[i - 1] == "\\":
                    buf.append(ch)
                    i += 1
                    continue
                in_dq = not in_dq
                buf.append(ch)
                i += 1
                continue
            if not in_sq and not in_dq:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth = max(0, depth - 1)
                elif ch == "," and depth == 0:
                    item = "".join(buf).strip()
                    if item:
                        items.append(item)
                    buf = []
                    i += 1
                    continue
            buf.append(ch)
            i += 1
        tail = "".join(buf).strip()
        if tail:
            items.append(tail)
        return items

    from_pos = _find_top_level_keyword(s_core, "FROM", start=0)
    if from_pos < 0:
        return None
    select_part_full = s_core[len("SELECT"):from_pos].strip()
    rest = s_core[from_pos + len("FROM"):].strip()
    if not select_part_full or not rest:
        return None

    clause_positions: List[Tuple[int, str]] = []
    for kw in ("WHERE", "GROUP BY", "HAVING", "ORDER BY", "LIMIT"):
        pos = _find_top_level_keyword(rest, kw, start=0)
        if pos >= 0:
            clause_positions.append((pos, kw))
    clause_positions.sort(key=lambda x: x[0])
    if clause_positions:
        first_pos = clause_positions[0][0]
        from_core = rest[:first_pos].strip()
        tail = rest[first_pos:].strip()
    else:
        from_core = rest
        tail = ""
    if not from_core:
        return None

    has_select_distinct = bool(re.match(r"(?is)^DISTINCT\b", select_part_full))
    select_part = re.sub(r"(?is)^DISTINCT\s+", "", select_part_full, count=1).strip() if has_select_distinct else select_part_full
    select_items = _split_top_level_csv(select_part)
    if not select_items:
        return None

    alias_to_table = _parse_from_join_aliases(f"SELECT 1 FROM {from_core}")
    if not alias_to_table:
        return None

    source_pairs: List[Tuple[str, str]] = []
    seen_aliases: Set[str] = set()
    reserved_alias_tokens = {
        "left", "right", "inner", "outer", "full", "cross", "join",
        "on", "where", "group", "order", "having", "limit",
    }
    for m_tbl in re.finditer(
        r"""(?is)\b(?:FROM|JOIN)\s+(`?[A-Za-z_][A-Za-z0-9_]*`?(?:\.`?[A-Za-z_][A-Za-z0-9_]*`?)?)\s*(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*)?""",
        f"FROM {from_core}",
    ):
        raw_tbl = (m_tbl.group(1) or "").replace("`", "").strip()
        tbl = raw_tbl.split(".")[-1].lower()
        if not tbl:
            continue
        alias_raw = (m_tbl.group(2) or "").strip()
        if alias_raw.lower() in reserved_alias_tokens:
            alias_raw = ""
        alias = (alias_raw or tbl).strip()
        alias_l = alias.lower()
        if not alias_l or alias_l in seen_aliases:
            continue
        seen_aliases.add(alias_l)
        source_pairs.append((alias, tbl))
    if not source_pairs:
        return None

    try:
        analyzer = get_schema_analyzer()
    except Exception:
        analyzer = None
    allowed_identifiers = _parse_allowed_identifiers(ctx or "")

    def _table_cols_lower(table_name: str) -> Set[str]:
        t_l = str(table_name or "").strip().lower()
        if not t_l:
            return set()
        if analyzer is not None:
            try:
                return {
                    str(c).strip("`").lower()
                    for c in analyzer.get_table_columns(t_l)
                    if str(c or "").strip()
                }
            except Exception:
                pass
        for t_ctx, cols_ctx in (allowed_identifiers or {}).items():
            if str(t_ctx or "").strip().lower() != t_l:
                continue
            return {
                str(c).strip("`").lower()
                for c in (cols_ctx or set())
                if str(c or "").strip()
            }
        return set()

    try:
        all_tables = {str(t).strip().lower() for t in (get_all_tables() or []) if str(t).strip()}
    except Exception:
        all_tables = {str(t).strip().lower() for t in (allowed_identifiers or {}).keys() if str(t).strip()}

    joined_tables = {tbl for _alias, tbl in source_pairs}
    alias_keys = {str(k or "").strip().lower() for k in alias_to_table.keys() if str(k or "").strip()}
    order_segment = ""
    m_order = re.search(r"(?is)\bORDER\s+BY\b\s+(?P<ord>.+?)(?:\bLIMIT\b.+)?$", tail)
    if m_order:
        order_segment = str(m_order.group("ord") or "").strip()

    preferred_table_for_col: Dict[str, str] = {}
    if order_segment:
        for m_ref in re.finditer(
            r"(?is)\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)\b",
            order_segment,
        ):
            qual = str(m_ref.group(1) or "").strip().lower()
            col = str(m_ref.group(2) or "").strip().lower()
            if not qual or not col:
                continue
            if qual in alias_keys:
                continue
            if qual in all_tables and col in _table_cols_lower(qual):
                preferred_table_for_col.setdefault(col, qual)

    try:
        fk_map_raw = get_fk_relationships() or {}
    except Exception:
        fk_map_raw = {}
    fk_map: Dict[str, Dict[str, str]] = {}
    for src, mapping in (fk_map_raw or {}).items():
        src_l = str(src or "").strip().lower()
        if not src_l:
            continue
        fk_map[src_l] = {}
        for fk_col, tgt in (mapping or {}).items():
            fk_l = str(fk_col or "").strip().lower()
            tgt_l = str(tgt or "").strip().lower()
            if fk_l and tgt_l:
                fk_map[src_l][fk_l] = tgt_l

    used_aliases: Set[str] = {a.lower() for a, _t in source_pairs}
    used_aliases.update(alias_keys)

    def _next_alias_for_table(table_name: str) -> str:
        t_l = str(table_name or "").strip().lower() or "t"
        if t_l and t_l not in used_aliases:
            used_aliases.add(t_l)
            return t_l
        base = re.sub(r"[^a-z0-9_]", "", t_l) or "t"
        first = base[0]
        if first not in used_aliases:
            used_aliases.add(first)
            return first
        for i in range(2, 200):
            cand = f"{first}{i}"
            if cand not in used_aliases:
                used_aliases.add(cand)
                return cand
        fallback = f"{first}_x"
        used_aliases.add(fallback)
        return fallback

    def _find_join_plan(target_table: str) -> Optional[Tuple[str, str, str, str]]:
        t_l = str(target_table or "").strip().lower()
        if not t_l:
            return None
        target_cols = _table_cols_lower(t_l)
        for conn_alias, conn_table in source_pairs:
            conn_l = str(conn_table or "").strip().lower()
            conn_cols = _table_cols_lower(conn_l)
            for fk_col, tgt in (fk_map.get(conn_l, {}) or {}).items():
                if tgt != t_l:
                    continue
                if fk_col not in conn_cols:
                    continue
                try:
                    target_key = str(get_id_column(t_l) or "").strip().lower()
                except Exception:
                    target_key = ""
                if not target_key or target_key not in target_cols:
                    target_key = fk_col if fk_col in target_cols else ""
                if target_key:
                    return ("conn_to_target", conn_alias, fk_col, target_key)
            for fk_col, tgt in (fk_map.get(t_l, {}) or {}).items():
                if tgt != conn_l:
                    continue
                if fk_col not in target_cols:
                    continue
                try:
                    conn_key = str(get_id_column(conn_l) or "").strip().lower()
                except Exception:
                    conn_key = ""
                if not conn_key or conn_key not in conn_cols:
                    conn_key = fk_col if fk_col in conn_cols else ""
                if conn_key:
                    return ("target_to_conn", conn_alias, fk_col, conn_key)
        return None

    select_low = select_part.lower()
    rebuilt_select_items = list(select_items)
    rebuilt_tail = tail
    rebuilt_from = from_core
    additions = 0

    for col_raw in missing_cols:
        col = str(col_raw or "").strip().strip("`")
        col_l = col.lower()
        if not col_l:
            continue
        if re.search(rf"(?is)\b{re.escape(col_l)}\b", select_low):
            continue

        preferred_table = preferred_table_for_col.get(col_l, "")
        candidate_tables: List[str] = []
        if preferred_table and preferred_table not in joined_tables:
            candidate_tables.append(preferred_table)
        for t in sorted(all_tables):
            if t in joined_tables:
                continue
            if col_l not in _table_cols_lower(t):
                continue
            if t not in candidate_tables:
                candidate_tables.append(t)
        if not candidate_tables:
            continue

        chosen_table = ""
        chosen_plan = None
        for cand_table in candidate_tables:
            plan = _find_join_plan(cand_table)
            if plan is None:
                continue
            chosen_table = cand_table
            chosen_plan = plan
            break
        if not chosen_table or chosen_plan is None:
            continue

        join_alias = (
            chosen_table
            if chosen_table not in used_aliases
            else _next_alias_for_table(chosen_table)
        )
        used_aliases.add(join_alias)
        alias_keys.add(join_alias)
        alias_to_table[join_alias] = chosen_table

        join_mode, conn_alias, lhs_col, rhs_col = chosen_plan
        if join_mode == "conn_to_target":
            on_clause = f"{conn_alias}.{lhs_col} = {join_alias}.{rhs_col}"
        else:
            on_clause = f"{join_alias}.{lhs_col} = {conn_alias}.{rhs_col}"

        rebuilt_from = f"{rebuilt_from} LEFT JOIN {chosen_table} AS {join_alias} ON {on_clause}"
        joined_tables.add(chosen_table)
        source_pairs.append((join_alias, chosen_table))

        rebuilt_select_items.append(f"{join_alias}.{col}")
        select_low += f" {col_l}"
        additions += 1

        if preferred_table and preferred_table != join_alias and rebuilt_tail:
            rebuilt_tail = re.sub(
                rf"(?is)\b{re.escape(preferred_table)}\s*\.\s*{re.escape(col)}\b",
                f"{join_alias}.{col}",
                rebuilt_tail,
            )

    if additions <= 0:
        return None

    rebuilt_select = ", ".join(rebuilt_select_items)
    distinct_prefix = "DISTINCT " if has_select_distinct else ""
    out = f"SELECT {distinct_prefix}{rebuilt_select} FROM {rebuilt_from}"
    if rebuilt_tail:
        out += f" {rebuilt_tail}"
    out = out.strip()
    if sql.rstrip().endswith(";") and not out.endswith(";"):
        out += ";"
    return out


def _repair_single_table_winner_row_projection(sql: str, question: str, ctx: str = "") -> Optional[str]:
    """
    Repair narrow winner-row near-misses for single-table ranked queries.

    Typical malformed shape:
      SELECT t.customer
      FROM (
        SELECT customer, amount,
               ROW_NUMBER() OVER (PARTITION BY customer ORDER BY amount DESC) AS rn
        FROM payments AS t1
      ) AS t
      WHERE t.rn = 2;

    Correct generic shape:
      SELECT t.customerNumber, t.amount
      FROM (
        SELECT customerNumber, amount,
               ROW_NUMBER() OVER (PARTITION BY customerNumber ORDER BY amount DESC) AS rn
        FROM payments AS t1
      ) AS t
      WHERE t.rn = 2;

    Guardrails:
    - only for winner-row questions
    - only for single-table ROW_NUMBER derived-table patterns
    - only when a generic entity token can be mapped uniquely to a key-like
      schema column (e.g. customer -> customerNumber)
    - only auto-adds projected columns that are explicitly requested and
      already available from the ranked inner rows
    """
    s = str(sql or "").strip()
    if not s or "ROW_NUMBER" not in s.upper():
        return None
    if not _question_wants_winner_row(question):
        return None

    pattern = re.compile(
        r"(?is)^\s*SELECT\s+(?P<outer>.*?)\s+FROM\s*\(\s*SELECT\s+(?P<inner>.*?)\s+FROM\s+(?P<rest>.+?)\)\s*(?:AS\s+)?(?P<alias>[A-Za-z_][A-Za-z0-9_]*)\s+WHERE\s+(?P=alias)\.rn\s*=\s*(?P<rank>\d+)\s*(?P<tail>.*?);?\s*$"
    )
    m = pattern.match(s)
    if not m:
        return None

    inner_rest = str(m.group("rest") or "").strip()
    if not inner_rest or re.search(r"(?is)\bJOIN\b", inner_rest):
        return None
    if re.search(r"(?is)\bSELECT\b", inner_rest):
        return None

    base_match = re.match(
        r"""(?is)^\s*(`?[A-Za-z_][A-Za-z0-9_]*`?(?:\.`?[A-Za-z_][A-Za-z0-9_]*`?)?)\s*(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*)?(?P<tail>.*)$""",
        inner_rest,
    )
    if not base_match:
        return None

    inner_table_expr = str(base_match.group(1) or "").strip()
    inner_tail = str(base_match.group("tail") or "").strip()
    if "," in inner_table_expr or "," in inner_tail:
        return None

    table_name = inner_table_expr.replace("`", "").split(".")[-1].strip().lower()
    if not table_name:
        return None

    allowed_map = _parse_allowed_identifiers(ctx or "")
    valid_cols_by_lower: Dict[str, str] = {
        str(c).strip("`").lower(): str(c).strip("`")
        for t, cols in (allowed_map or {}).items()
        if str(t).strip("`").lower() == table_name
        for c in (cols or set())
        if str(c).strip("`")
    }
    try:
        analyzer = get_schema_analyzer()
    except Exception:
        analyzer = None
    if analyzer is not None:
        try:
            for col_name in analyzer.get_table_columns(table_name):
                raw_col = str(col_name).strip("`")
                if raw_col:
                    valid_cols_by_lower.setdefault(raw_col.lower(), raw_col)
        except Exception:
            pass
    valid_cols = set(valid_cols_by_lower.keys())
    if not valid_cols:
        return None

    def _split_top_level_csv(text: str) -> List[str]:
        items: List[str] = []
        buf: List[str] = []
        depth = 0
        in_sq = False
        in_dq = False
        i = 0
        while i < len(text):
            ch = text[i]
            if ch == "'" and not in_dq:
                if in_sq and i + 1 < len(text) and text[i + 1] == "'":
                    buf.append(ch)
                    buf.append(text[i + 1])
                    i += 2
                    continue
                in_sq = not in_sq
                buf.append(ch)
                i += 1
                continue
            if ch == '"' and not in_sq:
                in_dq = not in_dq
                buf.append(ch)
                i += 1
                continue
            if not in_sq and not in_dq:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth = max(0, depth - 1)
                elif ch == "," and depth == 0:
                    item = "".join(buf).strip()
                    if item:
                        items.append(item)
                    buf = []
                    i += 1
                    continue
            buf.append(ch)
            i += 1
        tail_item = "".join(buf).strip()
        if tail_item:
            items.append(tail_item)
        return items

    def _extract_output_label(select_item: str) -> str:
        item = str(select_item or "").strip()
        if not item:
            return ""
        alias_match = re.search(
            r"""(?is)\bAS\s+(`[^`]+`|'[^']+'|"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)\s*$""",
            item,
        )
        if alias_match:
            return alias_match.group(1).strip("`'\"").lower()
        simple_match = re.match(
            r"(?is)^\s*(?:`?[A-Za-z_][A-Za-z0-9_]*`?\s*\.\s*)?`?([A-Za-z_][A-Za-z0-9_]*)`?\s*$",
            item,
        )
        return (simple_match.group(1).lower() if simple_match else "")

    def _simple_ref(item: str) -> Optional[Tuple[Optional[str], str]]:
        item_s = str(item or "").strip()
        m_ref = re.match(
            r"(?is)^\s*(?:`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\.\s*)?`?([A-Za-z_][A-Za-z0-9_]*)`?\s*$",
            item_s,
        )
        if not m_ref:
            return None
        qualifier = m_ref.group(1)
        col = m_ref.group(2)
        return (qualifier.lower() if qualifier else None, col.lower())

    def _name_tokens(name: str) -> List[str]:
        spaced = re.sub(AGENT_COLUMN_CAMEL_SPLIT_LOWER_PATTERN, r"\1 \2", str(name or ""))
        spaced = spaced.replace("_", " ")
        tokens = [tok for tok in re.findall(r"[a-z0-9]+", spaced.lower()) if tok]
        if len(tokens) == 1:
            raw = tokens[0]
            for suffix in ("number", "date", "amount", "code", "name", "time", "limit", "id", "key"):
                if raw.endswith(suffix) and len(raw) > len(suffix):
                    prefix = raw[: -len(suffix)].strip("_")
                    if prefix:
                        return [prefix, suffix]
        return tokens

    key_suffixes = {"id", "number", "code", "key"}

    def _question_mentions_col(col_name: str) -> bool:
        q = str(question or "").lower()
        if not q:
            return False
        if re.search(rf"\b{re.escape(col_name.lower())}\b", q):
            return True
        tokens = _name_tokens(col_name)
        if not tokens:
            return False
        spaced = " ".join(tokens)
        if spaced and re.search(rf"\b{re.escape(spaced)}\b", q):
            return True
        entity_tokens = [tok for tok in tokens if tok not in key_suffixes]
        if entity_tokens and key_suffixes.intersection(tokens):
            entity_phrase = " ".join(entity_tokens)
            if entity_phrase and re.search(
                rf"\b{re.escape(entity_phrase)}\s+(?:id|number|code|key)\b",
                q,
                flags=re.I,
            ):
                return True
        return False

    def _pick_key_like_match(entity_token: str) -> Optional[str]:
        entity = str(entity_token or "").strip().lower()
        if not entity:
            return None
        ranked: List[Tuple[int, str]] = []
        for col_lower, col_name in valid_cols_by_lower.items():
            tokens = _name_tokens(col_name)
            if entity not in tokens:
                continue
            if not key_suffixes.intersection(tokens):
                continue
            score = 0
            if tokens and tokens[0] == entity:
                score += 4
            if tokens and tokens[-1] in key_suffixes:
                score += 2
            if col_lower.startswith(entity):
                score += 1
            ranked.append((score, col_name))
        if not ranked:
            return None
        ranked.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
        if len(ranked) > 1 and ranked[0][0] == ranked[1][0] and ranked[0][1] != ranked[1][1]:
            return None
        return ranked[0][1]

    outer_select = str(m.group("outer") or "").strip()
    inner_select = str(m.group("inner") or "").strip()
    alias = str(m.group("alias") or "").strip()
    tail = str(m.group("tail") or "").strip()

    outer_items = _split_top_level_csv(outer_select)
    inner_items = _split_top_level_csv(inner_select)

    generic_map: Dict[str, str] = {}
    for item in inner_items + outer_items:
        ref = _simple_ref(item)
        if not ref:
            continue
        _, col = ref
        if col in valid_cols or col == "rn":
            continue
        mapped = _pick_key_like_match(col)
        if mapped and mapped != col:
            generic_map[col] = mapped

    if not generic_map:
        return None

    def _replace_identifier_token(text: str, old: str, new: str) -> str:
        out = str(text or "")
        out = re.sub(
            rf"(?i)(\b[A-Za-z_][A-Za-z0-9_]*\s*\.\s*)`?{re.escape(old)}`?(?![A-Za-z0-9_])",
            lambda m_ref: f"{m_ref.group(1)}{new}",
            out,
        )
        out = re.sub(
            rf"(?i)(?<![A-Za-z0-9_`])`?{re.escape(old)}`?(?![A-Za-z0-9_])",
            new,
            out,
        )
        return out

    rebuilt_inner = inner_select
    rebuilt_outer = outer_select
    rebuilt_tail = tail
    for old_col, new_col in generic_map.items():
        rebuilt_inner = _replace_identifier_token(rebuilt_inner, old_col, new_col)
        rebuilt_outer = _replace_identifier_token(rebuilt_outer, old_col, new_col)
        rebuilt_tail = _replace_identifier_token(rebuilt_tail, old_col, new_col)

    outer_items_after = _split_top_level_csv(rebuilt_outer)
    inner_items_after = _split_top_level_csv(rebuilt_inner)
    outer_labels = {lbl for lbl in (_extract_output_label(it) for it in outer_items_after) if lbl}
    inner_labels = {lbl for lbl in (_extract_output_label(it) for it in inner_items_after) if lbl}

    append_items: List[str] = []
    for inner_label in inner_labels:
        if inner_label in {"rn"} or inner_label in outer_labels:
            continue
        if _question_mentions_col(inner_label):
            append_items.append(f"{alias}.{inner_label}")
            outer_labels.add(inner_label)

    if append_items:
        rebuilt_outer = ", ".join(_split_top_level_csv(rebuilt_outer) + append_items)

    if (
        rebuilt_outer.strip() == outer_select.strip()
        and rebuilt_inner.strip() == inner_select.strip()
        and rebuilt_tail.strip() == tail.strip()
    ):
        return None

    fixed = (
        f"SELECT {rebuilt_outer} FROM (SELECT {rebuilt_inner} FROM {inner_rest}) {alias} "
        f"WHERE {alias}.rn = {m.group('rank')}"
    ).strip()
    if rebuilt_tail:
        fixed += " " + rebuilt_tail
    fixed = re.sub(r"\s{2,}", " ", fixed).strip()
    if s.endswith(";") and not fixed.endswith(";"):
        fixed += ";"
    return fixed


def _has_required_offset_window(sql: str, question: str) -> bool:
    s = str(sql or "")
    q = str(question or "").lower()
    if not s or not _question_has_offset_window_sequence_intent(question):
        return True

    wants_previous = bool(re.search(r"\b(?:previous|prior)\b", q, flags=re.I))
    wants_next = bool(re.search(r"\b(?:next|following)\b", q, flags=re.I))
    if wants_previous and not re.search(r"\bLAG\s*\(", s, flags=re.I):
        return False
    if wants_next and not re.search(r"\bLEAD\s*\(", s, flags=re.I):
        return False
    return True


def _is_ir_family_preferred(
    intent_ir: Dict[str, Any],
    decision: Dict[str, Any],
    qtype: str,
    mentioned_tables: List[str],
    router_tables: List[str],
    question: str,
) -> bool:
    """
    Family-first IR preference for compact, planner-safe query shapes.

    This is intentionally narrower than "anything plannable". It keeps IR
    aligned to the deterministic safe subset while making retrieval breadth a
    secondary signal instead of a hard veto for compact supported families.
    """
    qtype_norm = str(qtype or "").strip().lower()
    if qtype_norm not in {AGENT_ROUTER_QTYPE_LOOKUP, AGENT_ROUTER_QTYPE_AGGREGATE}:
        return False
    if not bool(intent_ir.get("planner_compatible", True)):
        return False

    compact_scope = max(len(mentioned_tables or []), len(router_tables or []), 1)
    if compact_scope > int(AGENT_IR_MAX_TABLES):
        return False

    # Multi-table COUNT(DISTINCT ...) is still too semantically fragile for the
    # current generic planner. Keep it in the LLM lane unless/ until a direct
    # planner-safe family is implemented explicitly.
    if compact_scope >= 2 and _question_wants_count_distinct(question):
        return False

    if bool(decision.get("has_dual_fact_rollup", False)):
        return False

    # These families remain LLM-first by design.
    family_blocking_flags = (
        "is_winner_row",
        "has_exclusion",
        "is_window_aggregate",
        "is_explicit_rank_function",
        "is_ordinal_ranking",
        "is_subquery_intent",
        "is_simple_scalar_subquery_single_table",
        "is_set_operation_intent",
    )
    if any(bool(decision.get(flag, False)) for flag in family_blocking_flags):
        return False

    regex_is_ir_safe, _ = _ir_regex_family_status(intent_ir, question)
    if not regex_is_ir_safe:
        return False

    return True


def draft_sql_agent(state: RAGState) -> RAGState:
    # --- Guard: Skip if question was rejected by router ---
    decision = state.get("router_decision", {})
    if decision.get("complexity") == AGENT_ROUTER_COMPLEXITY_REJECTED or decision.get("is_valid_question") is False:
        print("[DRAFTER] Skipped: Question was rejected by router guard")
        return state  # Pass through unchanged

    question = state.get("question", "")
    ctx = state.get("context", "")
    scenario_cfg = _get_scenario_config(state)
    intent_ir_enabled = bool(scenario_cfg.get("intent_ir_enabled", AGENT_IR_ENABLED))
    adaptive_model_enabled = bool(scenario_cfg.get("adaptive_model_enabled", True))
    forced_model = str(scenario_cfg.get("forced_model") or "").strip()
    # Route telemetry defaults
    state["ir_used"] = False
    state["ir_confidence"] = 0.0
    state["ir_plannable"] = False
    state["ir_gate_reasons"] = []
    state["ir_table_count"] = 0
    state["ir_context_table_count"] = 0
    state["ambiguity_signals"] = []
    state["cannot_answer"] = False
    state["cannot_answer_reason"] = ""

    # ------------------------------------------------------------------
    # IR-FIRST deterministic planning (schema-agnostic)
    # ------------------------------------------------------------------
    if intent_ir_enabled:
        intent_ir = build_intent_ir(
            question=question,
            ctx=ctx,
            router_decision=decision,
            primary_table=state.get("primary_table", ""),
        )
        state["intent_ir"] = intent_ir
        state["ambiguity_signals"] = list(intent_ir.get("ambiguity_signals") or [])

        should_abstain, abstain_reason, ambiguity_signals = _evaluate_cannot_answer_from_ir(
            intent_ir,
            retrieval_snapshot=state.get("retrieval_snapshot"),
        )
        if should_abstain:
            print(f"[DRAFTER] Abstain: {abstain_reason}")
            updated_decision = dict(decision or {})
            updated_decision["is_valid_question"] = False
            updated_decision["rejection_reason"] = abstain_reason
            return {
                **state,
                "ambiguity_signals": ambiguity_signals,
                "cannot_answer": True,
                "cannot_answer_reason": abstain_reason,
                "router_decision": updated_decision,
                "draft_source": "cannot_answer_guard",
                "draft_sql": "",
                "final_sql": "",
                "answer": abstain_reason,
            }

        mentioned_tables = intent_ir.get("mentioned_tables") or []
        router_tables = state.get("router_tables") or []
        ctx_allowed_ids = _parse_allowed_identifiers(ctx)
        ctx_table_count = len(ctx_allowed_ids)
        dynamic_min_conf = float(AGENT_IR_MIN_CONFIDENCE)
        ir_gate_reasons: List[str] = []
        ir_confidence = float(intent_ir.get("confidence", 0.0) or 0.0)
        ir_qtype = str(intent_ir.get("qtype") or "").strip().lower()
        ir_metrics = list(intent_ir.get("metrics") or [])
        ir_existence = list(intent_ir.get("existence") or [])
        ir_dims = list(intent_ir.get("dimensions") or [])
        safe_filter_ops = {
            "eq",
            "in",
            "not_in",
            "gt",
            "gte",
            "lt",
            "lte",
            "between",
            "year_eq",
            "month_eq",
            "month_between",
            "weekday_eq",
            "like_prefix",
            "like_suffix",
            "like_contains",
            "not_like_contains",
            "not_like_prefix",
            "not_like_suffix",
            "regexp",
            "word_count_eq",
            "len_eq",
            "len_gt",
            "len_gte",
            "len_lt",
            "len_lte",
            "is_null_or_empty",
            "is_not_null_or_empty",
            "col_eq",
            "col_gt",
            "col_gte",
            "col_lt",
            "col_lte",
        }
        prefer_ir_lookup_qtype = (
            ir_qtype == AGENT_ROUTER_QTYPE_LOOKUP
            and bool(intent_ir.get("select_distinct"))
            and bool(ir_dims)
            and not ir_metrics
            and not ir_existence
        )
        qtype = (
            ir_qtype
            if prefer_ir_lookup_qtype
            else str(decision.get("qtype") or ir_qtype or "").strip().lower()
        )
        lookup_subject_focused = (
            qtype == AGENT_ROUTER_QTYPE_LOOKUP
            and bool(intent_ir.get("subject_entity"))
            and len(mentioned_tables) <= 1
            and len(router_tables) <= 1
        )
        low_risk_single_table_lookup = _is_ir_low_risk_single_table_lookup(
            intent_ir=intent_ir,
            qtype=qtype,
            safe_filter_ops=safe_filter_ops,
        )
        family_ir_preferred = _is_ir_family_preferred(
            intent_ir=intent_ir,
            decision=decision,
            qtype=qtype,
            mentioned_tables=mentioned_tables,
            router_tables=router_tables,
            question=question,
        )
        regex_is_ir_safe, has_regex_family = _ir_regex_family_status(intent_ir, question)
        if low_risk_single_table_lookup:
            table_count = 1
        elif family_ir_preferred:
            table_count = max(len(mentioned_tables), len(router_tables), 1)
        elif lookup_subject_focused:
            table_count = max(len(mentioned_tables), len(router_tables), 1)
        else:
            table_count = max(len(mentioned_tables), len(router_tables), ctx_table_count)

        if qtype not in {AGENT_ROUTER_QTYPE_LOOKUP, AGENT_ROUTER_QTYPE_AGGREGATE}:
            ir_gate_reasons.append(f"unsupported_qtype_for_ir:{qtype or 'unknown'}")

        if has_regex_family and not regex_is_ir_safe:
            ir_gate_reasons.append("open_regex_prefers_llm")

        if table_count > AGENT_IR_MAX_TABLES:
            ir_gate_reasons.append(
                f"table_count_exceeds_ir_max:{table_count}>{AGENT_IR_MAX_TABLES}"
            )
        if (
            ctx_table_count > AGENT_IR_MAX_TABLES
            and not lookup_subject_focused
            and not low_risk_single_table_lookup
            and not family_ir_preferred
        ):
            ir_gate_reasons.append(
                f"context_table_count_exceeds_ir_max:{ctx_table_count}>{AGENT_IR_MAX_TABLES}"
            )

        if table_count > 1:
            dynamic_min_conf = max(dynamic_min_conf, float(AGENT_IR_MULTI_TABLE_MIN_CONFIDENCE))
            if not AGENT_IR_ALLOW_MULTI_TABLE:
                ir_gate_reasons.append("multi_table_ir_disabled")

        if bool(decision.get("has_dual_fact_rollup", False)):
            ir_gate_reasons.append("router_hard:dual_fact_rollup")
        if str(decision.get("complexity") or "").strip().lower() == AGENT_ROUTER_COMPLEXITY_COMPLEX:
            ir_gate_reasons.append("router_complex_prefers_llm")

        hard_router_flags = {
            "is_winner_row": "winner_row",
            "has_exclusion": "exclusion",
            "is_window_aggregate": "window_aggregate",
            "is_explicit_rank_function": "explicit_rank",
            "is_ordinal_ranking": "ordinal_rank",
            "is_subquery_intent": "subquery_intent",
            "is_simple_scalar_subquery_single_table": "simple_scalar_subquery",
        }
        for flag, label in hard_router_flags.items():
            if bool(decision.get(flag, False)):
                ir_gate_reasons.append(f"router_hard:{label}")

        # Percentage intent: allow IR only for supported metric patterns.
        q_low = (question or "").lower()
        # HAVING-like aggregate threshold (e.g., "average ... above 50,000").
        # Allow only a conservative, metric-aligned subset.
        has_numeric_threshold = _question_has_numeric_threshold(q_low)
        has_aggregate_keyword = bool(
            re.search(r"\b(total|sum|avg|average|mean|count|number\s+of)\b", q_low, flags=re.I)
        )
        has_metrics = bool(intent_ir.get("metrics") or [])
        threshold_pairs = _collect_ir_metric_threshold_pairs(intent_ir)
        if has_numeric_threshold and has_aggregate_keyword and has_metrics:
            if not threshold_pairs:
                ir_gate_reasons.append("having_like_threshold_unresolved")
            elif not AGENT_IR_ALLOW_SIMPLE_HAVING:
                ir_gate_reasons.append("having_like_threshold_intent_ir_disabled")
            else:
                having_ok, having_reason = _is_simple_having_shape(intent_ir, decision, threshold_pairs)
                if not having_ok:
                    ir_gate_reasons.append(having_reason or "unsupported_simple_having_shape")

        if re.search(r"\b(percent|percentage|%\s*of|ratio|rate)\b", q_low):
            ir_metrics = intent_ir.get("metrics") or []
            has_supported_pct_metric = any(
                str(m.get("agg") or "").lower() == "pct_non_null"
                for m in ir_metrics
                if isinstance(m, dict)
            )
            if not has_supported_pct_metric:
                ir_gate_reasons.append("percentage_intent_ir_disabled")

        safe_metric_aggs = {"count", "sum", "avg", "min", "max", "pct_non_null"}

        ir_metrics = list(intent_ir.get("metrics") or [])
        ir_existence = list(intent_ir.get("existence") or [])
        ir_dims = list(intent_ir.get("dimensions") or [])
        ir_filters = list(intent_ir.get("filters") or [])
        ir_ordering = list(intent_ir.get("ordering") or [])
        visible_dim_cols = [
            str(d.get("column") or "").strip().lower()
            for d in ir_dims
            if isinstance(d, dict) and bool(d.get("visible", True))
        ]
        date_like_dim_count = sum(
            1
            for col in visible_dim_cols
            if any(tok in col for tok in ("date", "time", "timestamp", "year", "month", "day"))
        )
        has_column_compare_filter = any(
            isinstance(f, dict)
            and str(f.get("op") or "").strip().lower() in {"col_eq", "col_gt", "col_gte", "col_lt", "col_lte"}
            for f in ir_filters
        )
        has_resolved_ordering = any(
            isinstance(o, dict)
            and str(o.get("kind") or "").strip().lower() in {"alias", "column", "expression"}
            for o in ir_ordering
        )

        if (
            qtype == AGENT_ROUTER_QTYPE_LOOKUP
            and not ir_metrics
            and _question_requests_projection_null_fill(question)
        ):
            ir_gate_reasons.append("projection_null_fill_ir_unsupported")

        if (
            qtype == AGENT_ROUTER_QTYPE_LOOKUP
            and not ir_metrics
            and not ir_existence
            and date_like_dim_count >= 2
            and bool(re.search(r"\b(?:after|before|later\s+than|earlier\s+than)\b", q_low, flags=re.I))
            and not has_column_compare_filter
        ):
            ir_gate_reasons.append("column_to_column_compare_ir_unsupported")

        temporal_alt = _temporal_extrema_alt()
        has_temporal_display_order_cue = bool(
            temporal_alt and re.search(
                r"\b(?:starting|beginning|going)\s+from\s+the\s+"
                rf"(?:{temporal_alt})\b",
                q_low,
                flags=re.I,
            )
            or (temporal_alt and re.search(
                r"\b(?:sorted|ordered)\s+from\s+the\s+"
                rf"(?:{temporal_alt})\b",
                q_low,
                flags=re.I,
            ))
            or (temporal_alt and re.search(
                r"\b(?:ordered|sorted)\s+by\b[^.;]{0,96}\bfrom\s+the\s+"
                rf"(?:{temporal_alt})\b",
                q_low,
                flags=re.I,
            ))
        )

        if (
            qtype == AGENT_ROUTER_QTYPE_LOOKUP
            and not ir_metrics
            and not ir_existence
            and len(mentioned_tables) <= 1
            and not has_temporal_display_order_cue
            and bool(
                temporal_alt and re.search(
                    r"\b(?:on|for|with|from)\s+the\s+"
                    rf"(?:{temporal_alt})\s+"
                    r"(?:[a-z][a-z0-9_]*\s+){0,3}date\b",
                    q_low,
                    flags=re.I,
                )
                or (temporal_alt and re.search(
                    r"\b(?:equal\s+to|same\s+as)\s+the\s+"
                    rf"(?:{temporal_alt})\s+"
                    r"(?:[a-z][a-z0-9_]*\s+){0,3}date\b",
                    q_low,
                    flags=re.I,
                ))
            )
        ):
            ir_gate_reasons.append("scalar_extrema_date_subquery_ir_unsupported")

        if (
            not ir_metrics
            and not ir_existence
            and _question_has_offset_window_sequence_intent(q_low)
        ):
            ir_gate_reasons.append("offset_window_ir_unsupported")

        if (
            qtype == AGENT_ROUTER_QTYPE_LOOKUP
            and not ir_metrics
            and not ir_existence
            and isinstance(intent_ir.get("limit"), int)
            and int(intent_ir.get("limit") or 0) > 0
            and bool(re.search(r"\b(?:top|bottom|highest|lowest|largest|smallest)\b", q_low, flags=re.I))
            and not has_resolved_ordering
        ):
            ir_gate_reasons.append("top_n_lookup_ir_unresolved")

        if _question_has_set_operation_intent(question):
            ir_gate_reasons.append("set_operation_ir_unsupported")

        for f in ir_filters:
            if not isinstance(f, dict):
                continue
            op = str(f.get("op") or "").strip().lower()
            if op and op not in safe_filter_ops:
                ir_gate_reasons.append(f"unsupported_ir_filter_op:{op}")

        for m in (intent_ir.get("metrics") or []):
            if not isinstance(m, dict):
                continue
            agg = str(m.get("agg") or "").strip().lower()
            if agg and agg not in safe_metric_aggs:
                ir_gate_reasons.append(f"unsupported_ir_metric:{agg}")

        if ir_gate_reasons:
            ir_gate_reasons = list(dict.fromkeys(ir_gate_reasons))

        base_gate_reasons: List[str] = []
        if not bool(intent_ir.get("planner_compatible", True)):
            base_gate_reasons.append("planner_incompatible")
        if ir_confidence < dynamic_min_conf:
            base_gate_reasons.append(
                f"confidence_below_min:{ir_confidence:.2f}<{dynamic_min_conf:.2f}"
            )
        if not str(intent_ir.get("subject_entity") or "").strip():
            base_gate_reasons.append("missing_subject_entity")
        if not (
            intent_ir.get("dimensions")
            or intent_ir.get("metrics")
            or intent_ir.get("filters")
            or intent_ir.get("existence")
        ):
            base_gate_reasons.append("empty_ir_payload")
        if base_gate_reasons:
            base_gate_reasons = list(dict.fromkeys(base_gate_reasons))

        base_plannable = is_ir_plannable(intent_ir, dynamic_min_conf)
        ir_plannable = base_plannable and not ir_gate_reasons
        ir_log_reasons: List[str] = list(
            dict.fromkeys(list(ir_gate_reasons) + list(base_gate_reasons))
        )
        state["ir_confidence"] = ir_confidence
        state["ir_plannable"] = bool(ir_plannable)
        state["ir_gate_reasons"] = list(ir_gate_reasons)
        state["ir_base_gate_reasons"] = list(base_gate_reasons)
        state["ir_table_count"] = int(table_count)
        state["ir_context_table_count"] = int(ctx_table_count)
        state["ir_family_preferred"] = bool(family_ir_preferred)

        if AGENT_IR_DEBUG:
            print(
                "[IR] confidence="
                f"{ir_confidence:.2f}, "
                f"plannable={ir_plannable}, "
                f"family_preferred={bool(family_ir_preferred)}, "
                f"table_count={table_count}, "
                f"context_table_count={ctx_table_count}, "
                f"min_conf={dynamic_min_conf:.2f}, "
                f"gate_reasons={ir_log_reasons or ['none']}"
            )

        if ir_plannable:
            if AGENT_IR_DEBUG:
                print("[IR] planner disabled -> LLM. intent IR used as hint only")
        elif AGENT_IR_DEBUG and not ir_plannable:
            print(
                f"[IR] gated off -> LLM. reasons={ir_log_reasons or ['unknown_ir_gate_reason']}"
            )
        if not ir_plannable:
            should_abstain, abstain_reason, ambiguity_signals = _evaluate_cannot_answer_from_ir(
                intent_ir,
                retrieval_snapshot=state.get("retrieval_snapshot"),
                gate_reasons=ir_log_reasons,
            )
            if should_abstain:
                print(f"[DRAFTER] Abstain (evidence-aware): {abstain_reason}")
                updated_decision = dict(decision or {})
                updated_decision["is_valid_question"] = False
                updated_decision["rejection_reason"] = abstain_reason
                return {
                    **state,
                    "ambiguity_signals": ambiguity_signals,
                    "cannot_answer": True,
                    "cannot_answer_reason": abstain_reason,
                    "router_decision": updated_decision,
                    "draft_source": "cannot_answer_guard",
                    "draft_sql": "",
                    "final_sql": "",
                    "answer": abstain_reason,
                }

    fallback_model = forced_model or state.get("router_model_candidate")
    if not fallback_model:
        if adaptive_model_enabled:
            fallback_model = LARGE_MODEL if (decision.get("escalation_reasons") or []) else SMALL_MODEL
        else:
            fallback_model = LARGE_MODEL
    fallback_complexity = str(decision.get("complexity") or "").strip().lower()
    if not fallback_complexity:
        fallback_complexity = (
            AGENT_ROUTER_COMPLEXITY_COMPLEX
            if fallback_model == LARGE_MODEL
            else AGENT_ROUTER_COMPLEXITY_SIMPLE
        )
    target_model = str(fallback_model or LARGE_MODEL)
    if forced_model:
        state["router_model_candidate"] = forced_model
    state["target_model"] = target_model
    print(f"[DRAFTER] Using LLM fallback: {str(fallback_complexity).upper()} -> {target_model}")

    use_case_bucket_text_mode = bool(
        target_model == SMALL_MODEL
        and _question_has_case_bucketing_intent(question)
    )
    llm = (
        build_llm_repair(model=target_model)
        if use_case_bucket_text_mode
        else build_llm_json(model=target_model)
    )

    # ===== PRE-WARM WITH OOM RETRY =====
    # This separates cold-start time from actual inference time
    # Also handles OOM by waiting for previous model to unload
    model_ready = _prewarm_model_with_oom_retry(llm, target_model)

    if not model_ready:
        return {
            **state,
            "target_model": target_model,
            "draft_source": "llm",
            "draft_sql": "SELECT 'Error: Model OOM after retries';",
        }

    # START TIMER: Model ready → Final SQL complete (for thesis latency benchmark)
    # This measures the core text-to-SQL processing time, excluding:
    # - Model loading/warmup (one-time cost)
    # - Retrieval (vector store, depends on index size)
    # - SQL execution (depends on database performance)
    state["processing_start_time"] = _time.perf_counter()
    print(f"[DEBUG] processing_start_time set")

    # Panggil Dynamic Prompt Builder (helpers_main.py)
    # Segala aturan heuristik (Anti-MAX, Anti-1055, Anti-Inner Join) sudah diurus di sana.
    # Pass state for dynamic join context from retrieval results
    prompt = make_generation_prompt(ctx, question, decision, state)

    if state.get("ir_hint_passed"):
        print("[DRAFTER] IR-informed LLM: intent IR hint injected into prompt")
    else:
        print("[DRAFTER] Pure LLM: no intent IR hint available")

    # ===== TIMED INFERENCE (model already loaded) =====
    _t0 = _time.perf_counter()
    if use_case_bucket_text_mode:
        print("[DRAFTER] CASE-bucketing direct text mode enabled for 3B")
        text_sql_direct = try_llm_sql_text(
            llm,
            _make_case_bucketing_direct_text_prompt(prompt, question, ctx),
            max_attempts=1,
            oom_retry_wait=OOM_WAIT_TIME,
        )
        obj = (
            {"sql": text_sql_direct, "notes": "No reasoning provided."}
            if text_sql_direct
            else None
        )
    else:
        obj = try_llm_json(llm, prompt, max_attempts=AGENT_DRAFT_TRY_JSON_MAX_ATTEMPTS, oom_retry_wait=OOM_WAIT_TIME)
    _inference_time = _time.perf_counter() - _t0

    # Refresh warmup heartbeat after real inference.
    # Without this, long generations (e.g., 7B > keep_alive seconds) make
    # the next request look "cold" even when the model has just been used.
    _last_model_warmup[target_model] = _time.time()

    # Store inference time in state for debugging (not logged to reduce noise)
    state["draft_inference_time"] = _inference_time
    print(f"[DEBUG] Draft inference time: {_inference_time:.2f}s")

    if obj is None and target_model == SMALL_MODEL and _question_has_case_bucketing_intent(question):
        print("[DRAFTER] CASE-bucketing fallback -> retry 3B text mode")
        text_retry_prompt = _make_sql_only_draft_prompt(prompt)
        text_llm = build_llm_repair(model=target_model)
        text_ready = _prewarm_model_with_oom_retry(text_llm, target_model)
        if text_ready:
            _t_text = _time.perf_counter()
            text_sql = try_llm_sql_text(
                text_llm,
                text_retry_prompt,
                max_attempts=1,
                oom_retry_wait=OOM_WAIT_TIME,
            )
            text_inference_time = _time.perf_counter() - _t_text
            _last_model_warmup[target_model] = _time.time()
            _inference_time += text_inference_time
            state["draft_inference_time"] = _inference_time
            print(f"[DEBUG] Draft text-retry inference time: {text_inference_time:.2f}s")
            if text_sql:
                obj = {"sql": text_sql, "notes": "No reasoning provided."}
                print("[DRAFTER] CASE-bucketing text retry succeeded with 3B")
            else:
                print("[DRAFTER] CASE-bucketing text retry failed -> retry 3B ultra-minimal prompt")
                _t_text_min = _time.perf_counter()
                text_sql_min = try_llm_sql_text(
                    text_llm,
                    _make_case_bucketing_ultra_minimal_prompt(question, ctx),
                    max_attempts=1,
                    oom_retry_wait=OOM_WAIT_TIME,
                )
                text_min_inference_time = _time.perf_counter() - _t_text_min
                _last_model_warmup[target_model] = _time.time()
                _inference_time += text_min_inference_time
                state["draft_inference_time"] = _inference_time
                print(f"[DEBUG] Draft ultra-minimal text-retry inference time: {text_min_inference_time:.2f}s")
                if text_sql_min:
                    obj = {"sql": text_sql_min, "notes": "No reasoning provided."}
                    print("[DRAFTER] CASE-bucketing ultra-minimal text retry succeeded with 3B")
                else:
                    print("[DRAFTER] CASE-bucketing ultra-minimal retry failed -> retry 3B micro prompt")
                    _t_text_micro = _time.perf_counter()
                    text_sql_micro = try_llm_sql_text(
                        text_llm,
                        _make_case_bucketing_micro_prompt(question, ctx),
                        max_attempts=1,
                        oom_retry_wait=OOM_WAIT_TIME,
                    )
                    text_micro_inference_time = _time.perf_counter() - _t_text_micro
                    _last_model_warmup[target_model] = _time.time()
                    _inference_time += text_micro_inference_time
                    state["draft_inference_time"] = _inference_time
                    print(f"[DEBUG] Draft micro-prompt text-retry inference time: {text_micro_inference_time:.2f}s")
                    if text_sql_micro:
                        obj = {"sql": text_sql_micro, "notes": "No reasoning provided."}
                        print("[DRAFTER] CASE-bucketing micro prompt retry succeeded with 3B")

    if obj is None and target_model != LARGE_MODEL and not forced_model:
        recovery_model = LARGE_MODEL
        print(f"[DRAFTER] JSON recovery fallback -> {recovery_model}")
        recovery_llm = build_llm_json(model=recovery_model)
        recovery_ready = _prewarm_model_with_oom_retry(recovery_llm, recovery_model)
        if recovery_ready:
            _t1 = _time.perf_counter()
            recovery_obj = try_llm_json(
                recovery_llm,
                prompt,
                max_attempts=1,
                oom_retry_wait=OOM_WAIT_TIME,
            )
            recovery_inference_time = _time.perf_counter() - _t1
            _last_model_warmup[recovery_model] = _time.time()
            _inference_time += recovery_inference_time
            state["draft_inference_time"] = _inference_time
            print(f"[DEBUG] Draft recovery inference time: {recovery_inference_time:.2f}s")
            if recovery_obj is not None:
                obj = recovery_obj
                target_model = recovery_model
                state["target_model"] = target_model
                print(f"[DRAFTER] JSON recovery succeeded with {recovery_model}")

    if obj is None:
        return {
            **state,
            "target_model": target_model,
            "draft_source": "llm",
            "draft_sql": "SELECT 'Error: LLM failed';",
        }

    if _llm_output_is_cannot_answer(obj):
        reason = _extract_llm_cannot_answer_reason(obj) or "Cannot answer from available schema evidence"
        print(f"[DRAFTER] LLM abstain contract triggered: {reason}")
        updated_decision = dict(decision or {})
        updated_decision["is_valid_question"] = False
        updated_decision["rejection_reason"] = reason
        return {
            **state,
            "router_decision": updated_decision,
            "draft_source": "cannot_answer_llm",
            "draft_sql": "",
            "final_sql": "",
            "answer": reason,
            "cannot_answer": True,
            "cannot_answer_reason": reason,
        }
        
    draft_sql = (obj.get("sql") or "SELECT 1;").strip()
    reasoning = (obj.get("notes") or "No reasoning provided.").strip()
    print(f"\n[DRAFTER] Raw from LLM: {draft_sql}")
    print(f"[DRAFTER] LLM Reasoning: {reasoning}")
    draft_sql = _remove_top_level_order_by_if_not_requested(draft_sql, question)
    # 2. [NEW] Hapus syntax error: SELECT col DESC FROM ...
    # Regex ini mencari DESC/ASC sebelum FROM (hanya di SELECT clause, bukan di ORDER BY)
    # Hanya strip DESC/ASC yang langsung sebelum FROM (dalam SELECT list)
    draft_sql = re.sub(AGENT_DRAFT_STRIP_DESC_ASC_BEFORE_FROM_PATTERN, " ", draft_sql, flags=re.IGNORECASE)
    # FIX: Jangan strip DESC/ASC sebelum koma jika ada di dalam ORDER BY atau OVER clause
    # Pattern lama: r"\s+(DESC|ASC)\s*(?=,)" - ini salah strip "ORDER BY amount DESC, date"
    # Pattern baru: hanya strip jika di SELECT clause (sebelum FROM)
    # Gunakan negative lookbehind untuk memastikan bukan dalam ORDER BY context
    def _strip_desc_in_select_only(sql: str) -> str:
        """Strip invalid top-level SELECT item directions, not window ORDER BY."""
        # Find SELECT ... FROM boundary
        match = re.search(AGENT_DRAFT_SELECT_FROM_PATTERN, sql, re.IGNORECASE | re.DOTALL)
        if not match:
            return sql
        select_part = match.group(1)
        select_items = _split_top_level_csv(select_part)
        if not select_items:
            return sql
        leading_ws = re.match(r"^\s*", select_part).group(0)
        trailing_ws = re.search(r"\s*$", select_part).group(0)
        fixed_items = [
            re.sub(r"\s+\b(?:DESC|ASC)\b\s*$", "", item, flags=re.I).strip()
            for item in select_items
        ]
        fixed_select = leading_ws + ", ".join(fixed_items) + (trailing_ws or " ")
        if fixed_select != select_part:
            sql = sql[:match.start(1)] + fixed_select + sql[match.end(1):]
        return sql
    draft_sql = _strip_desc_in_select_only(draft_sql)
    # Basic Cleanup (Limit Halusinasi)
    draft_sql = re.sub(AGENT_DRAFT_LIMIT_NULL_ZERO_PATTERN, "", draft_sql, flags=re.IGNORECASE).strip()
    match = re.search(AGENT_DRAFT_LIMIT_CAPTURE_PATTERN, draft_sql, re.IGNORECASE)
    if match:
        limit_n = match.group(1)
        limit_terms_pattern = "|".join(re.escape(term) for term in AGENT_DRAFT_LIMIT_INTENT_TERMS if term)
        explicit_limit_value = _extract_explicit_row_limit_value(question)
        has_limit_intent = bool(
            (explicit_limit_value is not None and int(limit_n) == int(explicit_limit_value)) or
            (limit_terms_pattern and re.search(rf"\b(?:{limit_terms_pattern})\b", question.lower()))
        )
        if not has_limit_intent:
            draft_sql = re.sub(AGENT_DRAFT_LIMIT_REMOVE_PATTERN, "", draft_sql, flags=re.IGNORECASE).strip()
    
    if not draft_sql.endswith(";"): draft_sql += ";"
    print(f"[DRAFTER] After cleanup: {draft_sql}")
    group_filters_hint_present = bool(state.get("ir_hint_group_filters_present", False))
    draft_has_group_by = bool(re.search(r"\bGROUP\s+BY\b", draft_sql, flags=re.I))
    draft_has_where = bool(re.search(r"\bWHERE\b", draft_sql, flags=re.I))
    draft_has_having = bool(re.search(r"\bHAVING\b", draft_sql, flags=re.I))
    draft_has_agg_in_where = bool(
        re.search(
            r"\bWHERE\b(?:(?!\bGROUP\s+BY\b|\bORDER\s+BY\b|\bLIMIT\b).)*\b(?:COUNT|SUM|AVG|MIN|MAX)\s*\(",
            draft_sql,
            flags=re.I | re.S,
        )
    )
    draft_has_where_in_subquery_having = bool(
        re.search(
            r"\bWHERE\b[\s\S]*?\bIN\s*\(\s*SELECT\b[\s\S]*?\bHAVING\b",
            draft_sql,
            flags=re.I,
        )
    )
    print(
        "[DRAFTER] HAVING telemetry: "
        f"group_filters_hint_present={group_filters_hint_present}, "
        f"draft_has_group_by={draft_has_group_by}, "
        f"draft_has_where={draft_has_where}, "
        f"draft_has_having={draft_has_having}, "
        f"draft_has_agg_in_where={draft_has_agg_in_where}, "
        f"draft_has_where_in_subquery_having={draft_has_where_in_subquery_having}"
    )
    return {
        **state,
        "target_model": target_model,
        "final_sql": draft_sql,
        "answer": draft_sql,
        "final_model": target_model,
        "draft_source": "llm",
        "draft_sql": draft_sql,
        "agent_notes": (obj.get("notes") or "").strip(),
    }
# ======================================================================
# ======================= AGENT 3 - SQL REFINER ========================
# ======================================================================
def refine_sql_agent(state: RAGState) -> RAGState:
    # --- Guard: Skip if question was rejected by router ---
    decision = state.get("router_decision", {})
    if decision.get("complexity") == AGENT_ROUTER_COMPLEXITY_REJECTED or decision.get("is_valid_question") is False:
        print("[REFINE] Skipped: Question was rejected by router guard")
        return state  # Pass through unchanged

    question = state.get("question", "")
    ctx = state.get("context", "")
    draft_sql = (state.get("draft_sql") or "").strip()
    draft_source = (state.get("draft_source") or "").strip().lower()
    retrieval_context_mode = str(state.get("retrieval_context_mode") or "primary")
    retrieval_retry_used = bool(state.get("retrieval_retry_used", False))
    retrieval_retry_reason = str(state.get("retrieval_retry_reason") or "")
    retrieval_expanded_table_count = int(state.get("retrieval_expanded_table_count") or 0)

    # DYNAMIC MODEL SELECTION berdasarkan draft quality
    draft_model = state.get("target_model") or SMALL_MODEL  # Track model asal
    target_model = LARGE_MODEL  # Default untuk regeneration
    
    llm = build_llm_repair(model=target_model)
    
    decision = state.get("router_decision", {})
    detected_tables = state.get("router_tables") or []
    complexity = decision.get("complexity", AGENT_ROUTER_COMPLEXITY_COMPLEX)
    qtype = decision.get("qtype", AGENT_ROUTER_QTYPE_LOOKUP)
    
    # 1. HEURISTIC PIPELINE (Versi Bersih/Safe Mode)
    final_sql = normalize_sql_pipeline(draft_sql, question, ctx)
    corrections = ["Applied Heuristic Pipeline"]
    repaired_distinct_sql = _repair_misplaced_select_distinct(final_sql)
    if repaired_distinct_sql and repaired_distinct_sql != final_sql:
        final_sql = str(repaired_distinct_sql).strip()
        corrections.append("Repaired misplaced SELECT DISTINCT")
    explicit_projection_sql = _ensure_explicit_simple_projection_columns(final_sql, question, ctx)
    if explicit_projection_sql and explicit_projection_sql != final_sql:
        final_sql = str(explicit_projection_sql).strip()
        corrections.append("Applied explicit simple projection preservation")
    aligned_description_sql = _align_requested_description_variant(final_sql, question, ctx)
    if aligned_description_sql and aligned_description_sql != final_sql:
        final_sql = str(aligned_description_sql).strip()
        corrections.append("Aligned description-column variant with question wording")
    aligned_joined_description_sql = _align_requested_joined_description_variant(final_sql, question, ctx)
    if aligned_joined_description_sql and aligned_joined_description_sql != final_sql:
        final_sql = str(aligned_joined_description_sql).strip()
        corrections.append("Aligned joined description-column variant with question wording")
    pruned_difference_sql = _drop_unrequested_difference_operand_columns(final_sql, question, ctx)
    if pruned_difference_sql and pruned_difference_sql != final_sql:
        final_sql = str(pruned_difference_sql).strip()
        corrections.append("Dropped unrequested arithmetic operand columns")
    dropped_filter_only_sql = _drop_unrequested_missing_filter_columns(final_sql, question, ctx)
    if dropped_filter_only_sql and dropped_filter_only_sql != final_sql:
        final_sql = str(dropped_filter_only_sql).strip()
        corrections.append("Dropped unrequested missing-filter projection columns")

    # 1a-2. Drop raw columns already covered by function expressions (e.g., textDescription when LEFT(textDescription,50) exists)
    pruned_func_covered_sql = _drop_raw_column_covered_by_function(final_sql, question)
    if pruned_func_covered_sql and pruned_func_covered_sql != final_sql:
        final_sql = str(pruned_func_covered_sql).strip()
        corrections.append("Dropped raw columns covered by function expressions")

    # 1b. AST guardrails (sqlglot) for structural safety checks/fixes.
    # This pass is conservative: if AST parse fails, we keep legacy behavior.
    if AGENT_AST_GUARDS_ENABLED:
        ast_sql, ast_meta = _ast_guardrails(final_sql, ctx=ctx, question=question)
        if ast_meta.get("parse_ok", False):
            ast_fixes = ast_meta.get("fixes") or []
            ast_warnings = ast_meta.get("warnings") or []
            if ast_fixes:
                final_sql = ast_sql
                corrections.append(f"AST guardrails fixes: {ast_fixes}")
            if AGENT_AST_GUARDS_DEBUG and ast_warnings:
                _dbg_print_two("[REFINER - AST] warnings", f"{ast_warnings}")
                corrections.append(f"AST guardrails warnings: {ast_warnings}")
        elif AGENT_AST_GUARDS_DEBUG and ast_meta.get("error"):
            _dbg_print_two("[REFINER - AST] parse skipped", f"{ast_meta.get('error')}")

    final_sql, contract_corrections, contract_violations = _apply_semantic_contract(final_sql, question, ctx)
    if contract_corrections:
        corrections.extend(contract_corrections)
    dropped_filter_only_sql = _drop_unrequested_missing_filter_columns(final_sql, question, ctx)
    if dropped_filter_only_sql and dropped_filter_only_sql != final_sql:
        final_sql = str(dropped_filter_only_sql).strip()
        corrections.append("Dropped unrequested missing-filter projection columns after semantic contract")
    repaired_distinct_sql = _repair_misplaced_select_distinct(final_sql)
    if repaired_distinct_sql and repaired_distinct_sql != final_sql:
        final_sql = str(repaired_distinct_sql).strip()
        corrections.append("Repaired misplaced SELECT DISTINCT after semantic contract")
    aligned_description_sql = _align_requested_description_variant(final_sql, question, ctx)
    if aligned_description_sql and aligned_description_sql != final_sql:
        final_sql = str(aligned_description_sql).strip()
        corrections.append("Aligned description-column variant after semantic contract")
    aligned_joined_description_sql = _align_requested_joined_description_variant(final_sql, question, ctx)
    if aligned_joined_description_sql and aligned_joined_description_sql != final_sql:
        final_sql = str(aligned_joined_description_sql).strip()
        corrections.append("Aligned joined description-column variant after semantic contract")
    pruned_difference_sql = _drop_unrequested_difference_operand_columns(final_sql, question, ctx)
    if pruned_difference_sql and pruned_difference_sql != final_sql:
        final_sql = str(pruned_difference_sql).strip()
        corrections.append("Dropped unrequested arithmetic operand columns after semantic contract")

    repaired_winner_projection = _repair_single_table_winner_row_projection(final_sql, question, ctx)
    if repaired_winner_projection and repaired_winner_projection != final_sql:
        final_sql = str(repaired_winner_projection).strip()
        corrections.append("Applied single-table winner-row projection repair")

    sql_up = final_sql.upper()

    # 2. PREPARE CHECKS (FULL PROTECTION)
    sql_for_tokenizing = final_sql
    sql_for_tokenizing = re.sub(AGENT_REFINE_SINGLE_QUOTED_STRING_PATTERN, " ", sql_for_tokenizing)
    sql_for_tokenizing = re.sub(AGENT_REFINE_DOUBLE_QUOTED_STRING_PATTERN, " ", sql_for_tokenizing)
    # Quoted aliases are display labels, not schema identifiers.  Leave the
    # expression before AS intact, but remove the alias body so labels like
    # `largest payment` are not treated as hallucinated columns.
    sql_for_tokenizing = re.sub(
        r"""\bAS\s+(?:`[^`]*`|'(?:''|[^'])*'|"(?:""|[^"])*")""",
        " AS ",
        sql_for_tokenizing,
        flags=re.IGNORECASE,
    )
    # FIX 1: Remove function calls (word followed by parenthesis)
    sql_for_tokenizing = re.sub(
        AGENT_REFINE_FUNCTION_CALL_PATTERN,
        "(",
        sql_for_tokenizing
    )

    # FIX 2: Remove table names from tokenizing
    # This prevents "payments" in "FROM payments" or subquery being treated as column
    allowed_ids = _parse_allowed_identifiers(ctx)
    allowed_cols_by_table_lower = {
        str(t).strip("`").lower(): {str(c).strip("`").lower() for c in cols}
        for t, cols in allowed_ids.items()
    }
    try:
        schema_analyzer = get_schema_analyzer()
    except Exception:
        schema_analyzer = None

    def _table_cols_lower(table_name: str) -> Set[str]:
        """
        Robust table column lookup for hallucination checks.
        Uses both request-scoped ALLOWED IDENTIFIERS and full schema metadata.
        This prevents false positives when ALLOWED cards are sparse.
        """
        t = str(table_name or "").strip("`").lower()
        if not t:
            return set()

        cols = set(allowed_cols_by_table_lower.get(t, set()))
        if schema_analyzer is not None:
            try:
                cols.update(c.lower() for c in schema_analyzer.get_table_columns(t))
            except Exception:
                pass
        return cols
    for tbl in allowed_ids.keys():
        sql_for_tokenizing = re.sub(
            rf"\b{re.escape(tbl)}\b",
            AGENT_REFINE_TABLE_PLACEHOLDER_TOKEN,
            sql_for_tokenizing,
            flags=re.IGNORECASE
        )

    sql_tokens = set(re.findall(AGENT_REFINE_SQL_TOKEN_PATTERN, sql_for_tokenizing.lower()))

    # Remove placeholder tokens
    sql_tokens.discard(AGENT_REFINE_TABLE_PLACEHOLDER_DISCARD)
    # Window-frame syntax tokens are SQL grammar, not schema columns.
    # Keep this local guard (in addition to config defaults) so env overrides
    # cannot accidentally re-enable hallucination false positives.
    window_grammar_tokens = {
        "rows", "row", "range", "unbounded", "preceding", "following", "current",
    }
    reserved = _SQL_KEYWORDS.union(set(AGENT_SQL_RESERVED_EXTRA_TOKENS)).union(window_grammar_tokens)
    
    defined_aliases = set()
    defined_aliases.update(re.findall(AGENT_REFINE_DEFINED_ALIAS_PATTERN, sql_up, re.IGNORECASE))
    quoted_aliases = re.findall(AGENT_REFINE_QUOTED_ALIAS_PATTERN, sql_up, re.IGNORECASE)
    defined_aliases.update(quoted_aliases)
    
    # Also capture subquery aliases: FROM (...) alias or FROM (...) AS alias
    subquery_aliases = re.findall(AGENT_REFINE_SUBQUERY_ALIAS_PATTERN, sql_up, re.IGNORECASE)
    for sa in subquery_aliases:
        if sa.lower() not in AGENT_REFINE_SUBQUERY_ALIAS_RESERVED_TOKENS:
            defined_aliases.add(sa)

    # Also capture table aliases: FROM orders o, JOIN customers c
    # Pattern: FROM table alias or JOIN table alias (with or without AS)
    table_alias_pattern = AGENT_REFINE_TABLE_ALIAS_PATTERN
    for m in re.finditer(table_alias_pattern, sql_up, re.IGNORECASE):
        alias = m.group(1).lower()
        # Skip SQL keywords that might be captured
        if alias not in AGENT_REFINE_TABLE_ALIAS_RESERVED_TOKENS:
            defined_aliases.add(alias)

    reserved.update(a.lower() for a in defined_aliases)

    # Fix: For subqueries, FROM (SELECT...) matches "SELECT" - skip that
    # Look for real table name or subquery alias
    primary_match = re.search(AGENT_REFINE_PRIMARY_FROM_PATTERN, sql_up)
    primary_tbl = ""
    if primary_match:
        candidate = primary_match.group(1).lower()
        # Skip if it's a keyword (subquery case: FROM (SELECT...))
        if candidate not in AGENT_REFINE_PRIMARY_FROM_SKIP_TOKENS:
            primary_tbl = candidate
    
    if primary_tbl: reserved.add(primary_tbl)

    primary_cols = _table_cols_lower(primary_tbl)
    alien_cols = [t for t in sql_tokens if t not in reserved and t not in primary_cols and not t.isdigit()]
    
    # --- FLATTEN VALID COLS FOR KASUS J ---
    all_valid_cols = set()
    for cols in allowed_ids.values():
        all_valid_cols.update(cols)

    # Static identifier validation catches undefined aliases/tables early
    # (e.g., SUM(p.amount) without joining alias `p`).
    identifier_errors = validate_identifiers(final_sql, ctx)
    hard_identifier_errors = _collect_hard_identifier_errors(identifier_errors)

    # 3. DECISION MATRIX
    needs_regeneration = False
    regeneration_model = LARGE_MODEL  # Default 7B untuk regeneration
    extra_instruction = ""

    # --- KASUS 0: IDENTIFIER VALIDATION (HARD ERROR) ---
    if hard_identifier_errors:
        _dbg_agent3_retry("Identifier Validation", f"{hard_identifier_errors}")
        table_chain = " -> ".join(detected_tables) if detected_tables else "schema tables"
        extra_instruction = (
            "STRICT REQUIREMENT: Your SQL references undefined aliases/tables/columns. "
            f"Errors detected: {hard_identifier_errors}. "
            "Use only aliases that are declared in FROM/JOIN and only valid schema columns. "
            f"Expected join scope from router tables: {table_chain}. "
            "If selecting from child facts, explicitly JOIN them first before using their alias."
        )
        needs_regeneration = True
        is_simple_multi_table_candidate = bool(
            decision.get("simple_multi_table_candidate", False)
        )
        if (
            draft_model == SMALL_MODEL
            and complexity == AGENT_ROUTER_COMPLEXITY_SIMPLE
            and is_simple_multi_table_candidate
            and qtype == AGENT_ROUTER_QTYPE_LOOKUP
            and len(detected_tables) <= 2
        ):
            regeneration_model = SMALL_MODEL
        else:
            regeneration_model = LARGE_MODEL

    # --- SEMANTIC CONTRACT (centralized high-value intent checks) ---
    if contract_violations and not needs_regeneration:
        contract_detail = ", ".join(contract_violations)
        _dbg_agent3_retry("Semantic Contract", contract_detail)
        if "missing_winner_row" in contract_violations:
            extra_instruction = (
                "STRICT REQUIREMENT: The question requires one winning row per group. "
                "You MUST use ROW_NUMBER() (or equivalent winner-row logic), rank within each group, "
                "and keep only rn = 1 in the outer query."
            )
        elif "missing_count_distinct" in contract_violations:
            extra_instruction = (
                "STRICT REQUIREMENT: The question asks for a count of distinct values. "
                "You MUST use COUNT(DISTINCT target_column) and avoid grouping by the distinct target "
                "unless the question explicitly asks to display that target column."
            )
        elif "missing_anti_existence" in contract_violations:
            extra_instruction = (
                "STRICT REQUIREMENT: The question asks for rows WITHOUT related child rows. "
                "You MUST use an anti-existence pattern such as NOT EXISTS, NOT IN, "
                "or LEFT JOIN ... WHERE child_key IS NULL. Do NOT use positive existence filtering."
            )
        elif "missing_include_zero" in contract_violations:
            extra_instruction = (
                "STRICT REQUIREMENT: The question explicitly asks to keep parent rows even when no child rows exist. "
                "You MUST preserve zero-match parents with LEFT JOIN (or an equivalent zero-preserving pattern) "
                "instead of dropping them with INNER JOIN."
            )
        elif "missing_final_order" in contract_violations:
            extra_instruction = (
                "STRICT REQUIREMENT: The question explicitly requests final result ordering. "
                "You MUST include a top-level ORDER BY in the final query with the requested sort direction. "
                "Preserve FROM/JOIN/WHERE/GROUP BY/HAVING and selected columns; fix only the final ORDER BY."
            )
        elif "incomplete_final_order" in contract_violations:
            requested_order_terms = _extract_requested_final_order_terms(question)
            requested_order_text = ", ".join(
                f"{phrase} {direction}".strip()
                for phrase, direction in requested_order_terms
                if str(phrase or "").strip()
            )
            extra_instruction = (
                "STRICT REQUIREMENT: The question explicitly requests a multi-key final ordering and the current SQL "
                "does not preserve the full ORDER BY sequence. "
                "Preserve FROM/JOIN/WHERE/GROUP BY/HAVING, selected columns, metrics, and aliases exactly as they are. "
                "Rewrite ONLY the top-level ORDER BY clause so it matches the requested order sequence"
                + (f": {requested_order_text}." if requested_order_text else ".")
            )
        elif "missing_scalar_extrema_date_filter" in contract_violations:
            extra_instruction = (
                "STRICT REQUIREMENT: The question asks for all rows on the latest or earliest date, not just one row. "
                "You MUST filter with a scalar subquery such as "
                "`WHERE date_col = (SELECT MAX(date_col) FROM base_table)` "
                "or the MIN(...) equivalent, then apply any requested final ORDER BY."
            )
        elif "missing_required_window_partition" in contract_violations:
            required_partition_col = _resolve_required_window_partition_column(question, ctx, final_sql)
            extra_instruction = (
                "STRICT REQUIREMENT: The question asks for a window calculation within the same group, "
                "so you MUST use a window function with the correct PARTITION BY column. "
                f"Use `PARTITION BY {required_partition_col}` and do NOT partition by an unrelated display column."
            )
        elif "missing_offset_window" in contract_violations:
            extra_instruction = (
                "STRICT REQUIREMENT: The question asks for a previous/next value in a sequence. "
                "You MUST use a window offset function. "
                "Use `LAG(...)` for previous/prior and `LEAD(...)` for next/following. "
                "Partition by the entity group when stated and order the window by the sequence column named in the question. "
                "If the question does not explicitly request descending sequence order, use ascending order inside the window."
            )
        elif "missing_regex_semantics" in contract_violations:
            extra_instruction = (
                "STRICT REQUIREMENT: The question requires regex-level structural matching. "
                "You MUST use REGEXP (or RLIKE), not LIKE, for exact structural patterns. "
                "If the question describes an exact shape, use anchored regex with ^ and $. "
                "Do NOT weaken exact multi-word or exact-format intent into a loose substring match."
            )
        else:
            extra_instruction = (
                "STRICT REQUIREMENT: The question requires deduplicated output rows. "
                "You MUST use SELECT DISTINCT (or an equivalent deduplication pattern) so duplicates are not returned."
            )
        needs_regeneration = True
        regeneration_model = LARGE_MODEL

    # --- KASUS A: Schema Validation & Hallucination Check (HARD ERROR) ---
    if alien_cols and AGENT_REFINE_JOIN_KEYWORD not in sql_up and not needs_regeneration:
        # SMART SUBQUERY HANDLING
        # Only skip validation if alien columns are INSIDE subquery (not in main SELECT)
        has_subquery = re.search(AGENT_REFINE_HAS_SUBQUERY_PATTERN, sql_up, re.IGNORECASE)
        
        if has_subquery:
            # Extract main SELECT clause (before first subquery)
            # Pattern: SELECT ... FROM ... WHERE ... (SELECT ...)
            main_select_match = re.search(AGENT_REFINE_MAIN_SELECT_FROM_PATTERN, sql_up, re.IGNORECASE | re.DOTALL)
            
            if main_select_match:
                main_select = main_select_match.group(1)
                
                # Check if alien columns appear in MAIN SELECT
                aliens_in_main = [
                    col for col in alien_cols 
                    if re.search(rf"\b{re.escape(col)}\b", main_select, re.IGNORECASE)
                ]
                
                if aliens_in_main:
                    # Alien columns in main SELECT → NOT safe to skip
                    _dbg_agent3_retry(
                        "Hallucination in Main Query",
                        f"Columns {aliens_in_main} in SELECT but not in FROM table",
                    )
                    
                    # Check if can be solved by JOIN
                    can_be_solved_by_join = False
                    potential_table = ""
                    for tbl in detected_tables:
                        if tbl == primary_tbl: continue
                        tbl_cols = _table_cols_lower(tbl)
                        if set(aliens_in_main).intersection(tbl_cols):
                            can_be_solved_by_join = True
                            potential_table = tbl
                            break
                    
                    if can_be_solved_by_join:
                        extra_instruction = (
                            f"CRITICAL ERROR: You selected columns {aliens_in_main} from table '{primary_tbl}', "
                            f"but these columns do NOT exist in '{primary_tbl}'. They exist in '{potential_table}'. "
                            f"You MUST use JOIN to access columns from multiple tables.\n"
                            f"CORRECT PATTERN:\n"
                            f"SELECT {potential_table[0]}.{aliens_in_main[0]}, {primary_tbl[0]}.column, AGG()\n"
                            f"FROM {potential_table} {potential_table[0]}\n"
                            f"JOIN {primary_tbl} {primary_tbl[0]} ON ...\n"
                            f"GROUP BY ..."
                        )
                        needs_regeneration = True
                        regeneration_model = LARGE_MODEL
                    else:
                        extra_instruction = (
                            f"STRICT REQUIREMENT: The columns {aliens_in_main} do NOT exist in table '{primary_tbl}'. "
                            "Check the schema and use only valid column names."
                        )
                        needs_regeneration = True
                        regeneration_model = LARGE_MODEL
                else:
                    # Alien columns only in subquery → safe to skip
                    _dbg_print_two(
                        "[REFINER] Skipping hallucination check", 
                        "Alien columns only in subquery (different scope)"
                    )
            else:
                # Can't parse SELECT clause → skip conservatively
                _dbg_print_two("[REFINER] Skipping hallucination check", "Cannot parse SELECT clause")
        else:
            # No subquery → original logic
            can_be_solved_by_join = False
            potential_table = ""
            for tbl in detected_tables:
                if tbl == primary_tbl: continue
                tbl_cols = _table_cols_lower(tbl)
                if set(alien_cols).intersection(tbl_cols):
                    can_be_solved_by_join = True
                    potential_table = tbl
                    break
            
            if can_be_solved_by_join:
                _dbg_agent3_retry("Missing JOIN", f"Columns {alien_cols} belong to {potential_table}")
                extra_instruction = (
                    f"STRICT REQUIREMENT: Columns {alien_cols} do NOT exist in table '{primary_tbl}', but they belong to '{potential_table}'. "
                    f"Detected intent implies multiple tables: {detected_tables}. "
                    "You MUST use explicit JOIN (e.g., JOIN {potential_table} ON ...) to link the data."
                )
                needs_regeneration = True
                regeneration_model = LARGE_MODEL
                
            elif not can_be_solved_by_join:
                _dbg_agent3_retry("Hallucination", f"Columns {alien_cols} do not exist in Schema.")
                extra_instruction = (
                    f"STRICT REQUIREMENT: The columns {alien_cols} do NOT exist in the database schema. "
                    "1. Check the exact column names in the schema context. "
                    "2. Use the exact same casing and spelling as shown in the schema. "
                    "3. Do NOT change the casing style (e.g., do NOT convert camelCase to snake_case)."
                )
                needs_regeneration = True
                regeneration_model = LARGE_MODEL

    # --- KASUS B: Winner-Row Leak Check (HARD ERROR) ---
    is_winner_row = decision.get("is_winner_row", False)
    wants_true_winner_row = (
        _question_wants_winner_row(question)
        and not _question_has_offset_window_sequence_intent(question)
    )
    if wants_true_winner_row and not needs_regeneration:
        if not _has_winner_row_shape(final_sql):
            rewritten_winner_sql = _rewrite_grouped_winner_row_ranked_aggregate(final_sql, question)
            if rewritten_winner_sql != final_sql:
                final_sql = rewritten_winner_sql
                sql_up = final_sql.upper()
                corrections.append("Applied grouped winner-row ranking rewrite")
            else:
                _dbg_agent3_retry("Missing Winner-Row Pattern", "ROW_NUMBER() winner-row filter missing.")
                extra_instruction = (
                    "STRICT REQUIREMENT: This question asks for one winning row per group. "
                    "You MUST use a grouped winner-row pattern with a derived table and ROW_NUMBER(). "
                    "First aggregate at the candidate grain, then rank inside each group, then keep rn = 1. "
                    "CORRECT PATTERN: "
                    "`SELECT * FROM (SELECT group_col, candidate_cols, metric1, metric2, "
                    "ROW_NUMBER() OVER (PARTITION BY group_col ORDER BY primary_metric DESC, tiebreak_metric DESC, stable_key ASC) AS rn "
                    "FROM (...) aggregated_candidates) t WHERE t.rn = 1`. "
                    "Do NOT return all grouped rows. Do NOT use plain GROUP BY + ORDER BY only."
                )
                needs_regeneration = True
                regeneration_model = LARGE_MODEL

    if is_winner_row and not needs_regeneration:
        if re.search(AGENT_REFINE_WINNER_ROW_LEAK_PATTERN, sql_up):
             _dbg_agent3_retry("Winner-Row Leak", "rn column found in outer select.")
             extra_instruction = (
                "STRICT REQUIREMENT: Use standard Winner-Row pattern. "
                "Do NOT include the 'rn' (row number) column in the final output."
             )
             needs_regeneration = True
             regeneration_model = LARGE_MODEL  # Force 7B

    # --- KASUS C: ANTI-PATTERN (WHERE IN Subquery) (HARD ERROR) ---
    tuple_in_pattern = re.search(AGENT_REFINE_TUPLE_IN_PATTERN, sql_up)
    if tuple_in_pattern and not needs_regeneration:
        _dbg_agent3_retry("Anti-Pattern", "Tuple comparison inside WHERE IN (...)")
        extra_instruction = (
            "STRICT REQUIREMENT: Do NOT use the `WHERE (col1, col2) IN (SELECT ...)` pattern. "
            "It causes Error 1055. INSTEAD, use the Window Function pattern:\n"
            "SELECT ... FROM (SELECT ..., ROW_NUMBER() OVER (...) as rn FROM table) t WHERE rn = 1;"
        )
        needs_regeneration = True
        regeneration_model = LARGE_MODEL  # Force 7B

    # --- KASUS D: MISSING PARTITION KEY IN OUTPUT (HARD ERROR) ---
    if is_winner_row and not needs_regeneration:
        part_match = re.search(AGENT_REFINE_PARTITION_BY_PATTERN, final_sql, re.IGNORECASE)
        if part_match:
            part_col = part_match.group(1)
            select_clause_match = re.search(AGENT_REFINE_SELECT_FROM_CLAUSE_PATTERN, final_sql, re.IGNORECASE | re.DOTALL)
            if select_clause_match:
                select_clause = select_clause_match.group(1)
                select_clause_up = select_clause.upper()

                def _partition_key_present_semantically(part_col_up: str, outer_select_up: str, full_sql: str) -> bool:
                    part_col_norm = (part_col_up or "").strip().lower()
                    outer_select_norm = (outer_select_up or "").strip().lower()
                    # Direct presence in outer SELECT.
                    if part_col_norm and part_col_norm in outer_select_norm:
                        return True

                    outer_select_tokens = set(
                        re.findall(AGENT_REFINE_SQL_TOKEN_PATTERN, outer_select_up.lower())
                    )

                    # Support alias projection from inner query:
                    # PARTITION BY customerNumber, inner SELECT customerNumber AS customer_id,
                    # outer SELECT customer_id ...
                    alias_matches = re.findall(
                        rf"(?:\b[A-Za-z_][A-Za-z0-9_]*\.)?{re.escape(part_col_up)}\b\s+AS\s+([A-Za-z_][A-Za-z0-9_]*)",
                        full_sql,
                        flags=re.IGNORECASE,
                    )
                    for alias_name in alias_matches:
                        if alias_name.lower() in outer_select_tokens:
                            return True

                    # Semantic equivalence: customerNumber ~ customer_id / customerCode.
                    part_low = part_col_up.lower()
                    part_stem = re.sub(r"(?:_)?(?:number|id|code|key)$", "", part_low).strip("_")
                    if part_stem:
                        for tok in outer_select_tokens:
                            tok_stem = re.sub(r"(?:_)?(?:number|id|code|key)$", "", tok).strip("_")
                            if tok_stem == part_stem and re.search(r"(?:number|id|code|key)$", tok):
                                return True

                    return False

                if not _partition_key_present_semantically(part_col, select_clause_up, final_sql):
                    def _inject_partition_key_into_outer_select(full_sql: str, partition_col: str) -> Optional[str]:
                        s = (full_sql or "").strip()
                        if not s:
                            return None

                        def _split_top_level_csv_local(text: str) -> List[str]:
                            items: List[str] = []
                            buf: List[str] = []
                            depth = 0
                            in_sq = False
                            in_dq = False
                            i = 0
                            while i < len(text):
                                ch = text[i]
                                if ch == "'" and not in_dq:
                                    if in_sq and i + 1 < len(text) and text[i + 1] == "'":
                                        buf.append(ch)
                                        buf.append(text[i + 1])
                                        i += 2
                                        continue
                                    in_sq = not in_sq
                                    buf.append(ch)
                                    i += 1
                                    continue
                                if ch == '"' and not in_sq:
                                    in_dq = not in_dq
                                    buf.append(ch)
                                    i += 1
                                    continue
                                if not in_sq and not in_dq:
                                    if ch == "(":
                                        depth += 1
                                    elif ch == ")":
                                        depth = max(0, depth - 1)
                                    elif ch == "," and depth == 0:
                                        item = "".join(buf).strip()
                                        if item:
                                            items.append(item)
                                        buf = []
                                        i += 1
                                        continue
                                buf.append(ch)
                                i += 1
                            tail = "".join(buf).strip()
                            if tail:
                                items.append(tail)
                            return items

                        select_match = re.match(r"(?is)^\s*SELECT\s+(?P<outer>.+?)\s+FROM\s*\(", s)
                        if not select_match:
                            return None

                        from_match = re.search(r"\bFROM\s*\(", s, re.IGNORECASE)
                        if not from_match:
                            return None
                        open_idx = s.find("(", from_match.start())
                        if open_idx < 0:
                            return None

                        depth = 0
                        in_sq = False
                        in_dq = False
                        close_idx = -1
                        for i in range(open_idx, len(s)):
                            ch = s[i]
                            if ch == "'" and not in_dq:
                                in_sq = not in_sq
                            elif ch == '"' and not in_sq:
                                in_dq = not in_dq
                            elif not in_sq and not in_dq:
                                if ch == "(":
                                    depth += 1
                                elif ch == ")":
                                    depth -= 1
                                    if depth == 0:
                                        close_idx = i
                                        break

                        if close_idx < 0:
                            return None

                        inner_sql = s[open_idx + 1:close_idx]
                        outer_tail = s[close_idx + 1:].strip()
                        alias_match = re.match(r"(?is)^(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*)\b", outer_tail)
                        if not alias_match:
                            return None
                        outer_alias = alias_match.group(1)

                        projected_name = partition_col
                        alias_from_inner = re.search(
                            rf"(?:\b[A-Za-z_][A-Za-z0-9_]*\.)?{re.escape(partition_col)}\b\s+AS\s+([A-Za-z_][A-Za-z0-9_]*)",
                            inner_sql,
                            flags=re.IGNORECASE,
                        )
                        if alias_from_inner:
                            projected_name = alias_from_inner.group(1)
                        else:
                            has_inner_projection = re.search(
                                rf"(?:\b[A-Za-z_][A-Za-z0-9_]*\.)?{re.escape(partition_col)}\b",
                                inner_sql,
                                flags=re.IGNORECASE,
                            )
                            if not has_inner_projection:
                                return None

                        outer_select = (select_match.group("outer") or "").strip()
                        qualified_outer_items: List[str] = []
                        for item in _split_top_level_csv_local(outer_select):
                            bare_col_match = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*$", item)
                            if bare_col_match:
                                qualified_outer_items.append(f"{outer_alias}.{bare_col_match.group(1)}")
                            else:
                                qualified_outer_items.append(item)
                        outer_select_qualified = ", ".join(qualified_outer_items)
                        candidate_expr = f"{outer_alias}.{projected_name}"
                        fixed = "SELECT " + candidate_expr + ", " + outer_select_qualified + s[select_match.end("outer"):]
                        fixed = re.sub(r"\s{2,}", " ", fixed).strip()
                        if full_sql.rstrip().endswith(";") and not fixed.endswith(";"):
                            fixed += ";"
                        return fixed

                    repaired_partition_sql = _inject_partition_key_into_outer_select(final_sql, part_col)
                    if repaired_partition_sql:
                        repaired_select_match = re.search(
                            AGENT_REFINE_SELECT_FROM_CLAUSE_PATTERN,
                            repaired_partition_sql,
                            re.IGNORECASE | re.DOTALL,
                        )
                        repaired_select_up = repaired_select_match.group(1).upper() if repaired_select_match else ""
                        if _partition_key_present_semantically(part_col, repaired_select_up, repaired_partition_sql):
                            final_sql = repaired_partition_sql
                            sql_up = final_sql.upper()
                            corrections.append(f"Restored missing partition key '{part_col}' to outer SELECT")
                        else:
                            _dbg_agent3_retry("Missing Partition Key", f"'{part_col}' missing from SELECT")
                            extra_instruction = (
                                f"STRICT REQUIREMENT: The query partitions by '{part_col}', but it is MISSING from the final SELECT list. "
                                f"You MUST include '{part_col}' in the output so we know who the record belongs to. "
                                f"Target format: SELECT {part_col}, [other columns] FROM ..."
                            )
                            needs_regeneration = True
                            regeneration_model = LARGE_MODEL  # Force 7B
                    else:
                        _dbg_agent3_retry("Missing Partition Key", f"'{part_col}' missing from SELECT")
                        extra_instruction = (
                            f"STRICT REQUIREMENT: The query partitions by '{part_col}', but it is MISSING from the final SELECT list. "
                            f"You MUST include '{part_col}' in the output so we know who the record belongs to. "
                            f"Target format: SELECT {part_col}, [other columns] FROM ..."
                        )
                        needs_regeneration = True
                        regeneration_model = LARGE_MODEL  # Force 7B

    # --- KASUS E: INNER JOIN TRAP (HARD ERROR) ---
    # Apply this guard ONLY when question intent expects inclusive parent rows
    # (include-zero semantics). For existence-filter intent (e.g. "has at least one"),
    # INNER JOIN is correct and this trap must be skipped.
    has_groupby = AGENT_REFINE_GROUP_BY_TOKEN in sql_up
    is_left_join = AGENT_REFINE_LEFT_JOIN_TOKEN in sql_up
    wants_inner_join = _question_requires_inner_join(question, ctx)
    wants_include_zero = _question_wants_include_zero(question, ctx)
    if has_groupby and not is_left_join and not needs_regeneration and wants_include_zero and not wants_inner_join:
        where_match = re.search(AGENT_REFINE_WHERE_CLAUSE_PATTERN, sql_up, re.DOTALL)
        group_match = re.search(AGENT_REFINE_GROUP_CLAUSE_PATTERN, sql_up, re.DOTALL)
        if where_match and group_match:
            where_clause = where_match.group(1).lower()
            group_clause = group_match.group(1).lower()
            where_refs = set(re.findall(AGENT_REFINE_TABLE_REF_TOKEN_PATTERN, where_clause))
            group_refs = set(re.findall(AGENT_REFINE_TABLE_REF_TOKEN_PATTERN, group_clause))
            risky_filters = where_refs - group_refs
            if risky_filters:
                 _dbg_agent3_retry("INNER JOIN Trap", f"Filtering child table {risky_filters}")
                 extra_instruction = (
                    "STRICT REQUIREMENT: You are aggregating data (GROUP BY), but you are filtering a related table in the WHERE clause. "
                    "This behaves like an INNER JOIN and removes master records with no matches. "
                    "1. You MUST use 'LEFT JOIN'. "
                    "2. You MUST move the filters for the related table from WHERE to the ON clause."
                 )
                 needs_regeneration = True
                 regeneration_model = LARGE_MODEL  # Force 7B

    # --- KASUS F: CRITICAL COLUMN OMISSION (SOFT/MEDIUM/HARD ERROR) ---
    if not needs_regeneration:
        missing_cols = []
        
        # STEP 1: Ambil tokens HANYA dari klausa SELECT (Bukan seluruh SQL)
        # Tujuannya agar kolom yang hanya ada di WHERE/ORDER BY tetap terdeteksi sebagai "missing" di output
        select_match_chk = re.search(AGENT_REFINE_SELECT_FROM_CLAUSE_PATTERN, final_sql, re.IGNORECASE | re.DOTALL)
        select_text_chk_raw = select_match_chk.group(1) if select_match_chk else ""
        select_text_chk = select_text_chk_raw.lower()
        # Tokenize hanya bagian SELECT
        select_tokens_chk = set(re.findall(AGENT_REFINE_SQL_TOKEN_PATTERN, select_text_chk))

        # Semantic SELECT tokens:
        # - direct token (customernumber)
        # - underscore parts (total_payment_amount -> total/payment/amount)
        # - camel parts (totalPaymentAmount -> total/payment/amount)
        select_semantic_tokens = set(select_tokens_chk)
        select_tokens_raw_case = set(re.findall(AGENT_REFINE_SQL_TOKEN_PATTERN, select_text_chk_raw))
        for tok_raw in select_tokens_raw_case:
            tok_low = tok_raw.lower()
            if not tok_low:
                continue
            select_semantic_tokens.add(tok_low)
            for part in tok_low.split("_"):
                if part:
                    select_semantic_tokens.add(part)
            camel_spaced = re.sub(AGENT_COLUMN_CAMEL_SPLIT_LOWER_PATTERN, r"\1 \2", tok_raw)
            for part in re.findall(r"[a-z0-9]+", camel_spaced.lower()):
                if part:
                    select_semantic_tokens.add(part)

        def _split_top_level_csv(text: str) -> List[str]:
            items: List[str] = []
            buf: List[str] = []
            depth = 0
            in_sq = False
            in_dq = False
            i = 0
            while i < len(text):
                ch = text[i]
                if ch == "'" and not in_dq:
                    if in_sq and i + 1 < len(text) and text[i + 1] == "'":
                        buf.append(ch)
                        buf.append(text[i + 1])
                        i += 2
                        continue
                    in_sq = not in_sq
                    buf.append(ch)
                    i += 1
                    continue
                if ch == '"' and not in_sq:
                    in_dq = not in_dq
                    buf.append(ch)
                    i += 1
                    continue
                if not in_sq and not in_dq:
                    if ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth = max(0, depth - 1)
                    elif ch == "," and depth == 0:
                        item = "".join(buf).strip()
                        if item:
                            items.append(item)
                        buf = []
                        i += 1
                        continue
                buf.append(ch)
                i += 1
            tail_item = "".join(buf).strip()
            if tail_item:
                items.append(tail_item)
            return items

        def _semantic_parts(text: str) -> Set[str]:
            out: Set[str] = set()
            if not text:
                return out
            identifier_terms = {
                "id", "ids", "identifier", "identifiers",
                "number", "numbers", "code", "codes", "key", "keys",
            }

            def _expand_term(term: str) -> Set[str]:
                expanded: Set[str] = set()
                low = str(term or "").strip().lower()
                if not low:
                    return expanded
                expanded.add(low)
                if low.endswith("ies") and len(low) > 3:
                    expanded.add(low[:-3] + "y")
                elif low.endswith("s") and len(low) > 3:
                    expanded.add(low[:-1])
                if low in identifier_terms:
                    expanded.update(identifier_terms)
                return expanded

            for token in re.findall(r"[A-Za-z0-9_]+", text):
                low = token.lower()
                if low:
                    out.update(_expand_term(low))
                for part in low.split("_"):
                    if part:
                        out.update(_expand_term(part))
                camel_spaced = re.sub(AGENT_COLUMN_CAMEL_SPLIT_LOWER_PATTERN, r"\1 \2", token)
                for part in re.findall(r"[a-z0-9]+", camel_spaced.lower()):
                    if part:
                        out.update(_expand_term(part))
            return out

        projected_alias_token_sets: List[Set[str]] = []
        for select_item in _split_top_level_csv(select_text_chk_raw):
            alias_match = re.search(r"\bAS\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", select_item, re.IGNORECASE)
            if alias_match:
                label = alias_match.group(1)
            else:
                simple_col_match = re.match(
                    r"\s*(?:[A-Za-z_][A-Za-z0-9_]*\.)?([A-Za-z_][A-Za-z0-9_]*)\s*$",
                    select_item,
                )
                label = simple_col_match.group(1) if simple_col_match else ""
            token_set = _semantic_parts(label)
            if token_set:
                projected_alias_token_sets.append(token_set)

        identifier_terms = {
            "id", "ids", "identifier", "identifiers",
            "number", "numbers", "code", "codes", "key", "keys",
        }

        def _parts_covered_by_alias(required_parts: Set[str], alias_tokens: Set[str]) -> bool:
            if not required_parts or not alias_tokens:
                return False

            def _equiv(a: str, b: str) -> bool:
                a0 = str(a or "").strip().lower()
                b0 = str(b or "").strip().lower()
                if not a0 or not b0:
                    return False
                if a0 == b0:
                    return True
                if a0 in identifier_terms and b0 in identifier_terms:
                    return True
                short, long = (a0, b0) if len(a0) <= len(b0) else (b0, a0)
                if len(short) >= 3 and len(long) >= 5 and long.startswith(short):
                    return True
                return False

            return all(
                any(_equiv(req, tok) for tok in alias_tokens)
                for req in required_parts
            )
        
        # STEP 1b: Ambil tokens dari WHERE/HAVING/JOIN ON (kolom predicate/filter)
        # Kolom yang sudah dipakai sebagai predicate tidak perlu dipaksa keluar
        # sebagai output jika pertanyaan tidak memintanya secara eksplisit.
        sql_lower = final_sql.lower()
        where_having_tokens = set()
        # Extract WHERE clause
        where_match = re.search(AGENT_REFINE_LOWER_WHERE_PATTERN, sql_lower, re.DOTALL)
        if where_match:
            where_having_tokens.update(re.findall(AGENT_REFINE_SQL_TOKEN_PATTERN, where_match.group(1)))
        # Extract HAVING clause
        having_match = re.search(AGENT_REFINE_LOWER_HAVING_PATTERN, sql_lower, re.DOTALL)
        if having_match:
            where_having_tokens.update(re.findall(AGENT_REFINE_SQL_TOKEN_PATTERN, having_match.group(1)))
        # Extract JOIN ON predicates
        for on_match in re.finditer(
            r"(?is)\bON\b\s+(.*?)(?=\b(?:INNER\s+JOIN|LEFT\s+JOIN|RIGHT\s+JOIN|FULL\s+JOIN|JOIN|WHERE|GROUP\s+BY|HAVING|ORDER\s+BY|LIMIT)\b|$)",
            sql_lower,
        ):
            where_having_tokens.update(
                re.findall(AGENT_REFINE_SQL_TOKEN_PATTERN, on_match.group(1))
            )

        def _extract_requested_output_segment_local(text: str) -> str:
            q_low_local = str(text or "").strip().lower()
            if not q_low_local:
                return ""

            head_patterns = [
                r"\b(?:return|show|display|get|list|find|identify|retrieve|extract)\s+(?:the\s+)?(?P<seg>.+?)\s+\bfor\b",
                r"\b(?:return|show|display|get|list|find|identify|retrieve|extract)\s+(?:the\s+)?(?P<seg>.+?)\s+\bwhere\b",
            ]
            for pat in head_patterns:
                m = re.search(pat, q_low_local, flags=re.I)
                if not m:
                    continue
                seg = str(m.group("seg") or "").strip(" .;")
                if seg:
                    return seg

            tail_patterns = [
                r"\breturn\b\s+(?P<seg>.+)$",
                r"\bshow\b\s+(?P<seg>.+)$",
                r"\bdisplay\b\s+(?P<seg>.+)$",
                r"\bget\b\s+(?P<seg>.+)$",
                r"\bshowing\b\s+(?P<seg>.+)$",
                r"\bincluding\b\s+(?P<seg>.+)$",
                r"\binclude\b\s+(?P<seg>.+)$",
                r"\bwith\b\s+(?P<seg>.+)$",
            ]
            for pat in tail_patterns:
                m = re.search(pat, q_low_local, flags=re.I)
                if not m:
                    continue
                seg = str(m.group("seg") or "").strip(" .;")
                if not seg:
                    continue
                stop_pat = re.compile(
                    r"(?is)(?:,\s*|\.\s*|\s+)(?:sorted|ordered|starting|beginning|ranked|grouped|within|among|across|inside|only\s+include|include\s+only|keep\s+only|only\s+keep|retain\s+only)\b"
                )
                stop_m = stop_pat.search(seg)
                if stop_m:
                    seg = seg[:stop_m.start()].strip(" ,.;")
                return seg
            return ""

        requested_output_segment = _extract_requested_output_segment_local(question)
        requested_output_text = str(requested_output_segment or question or "")

        def _question_explicitly_requests_column(col_name: str) -> bool:
            seg = str(requested_output_segment or "").strip().lower()
            if not seg:
                return False
            clean = re.sub(AGENT_COLUMN_CAMEL_SPLIT_LOWER_PATTERN, r"\1 \2", str(col_name or ""))
            clean = clean.replace("_", " ").lower().strip()
            if not clean:
                return False
            parts_local = [p for p in clean.split() if p]
            if not parts_local:
                return False
            phrase_patterns = {clean}
            phrase_patterns.add(r"\s+".join(re.escape(p) for p in parts_local))
            suffix_synonyms_local = AGENT_MISSING_COL_SUFFIX_SYNONYMS
            last_word_local = parts_local[-1]
            if last_word_local in suffix_synonyms_local and len(parts_local) >= 2:
                prefix_local = r"\s+".join(re.escape(p) for p in parts_local[:-1])
                phrase_patterns.add(rf"{prefix_local}\s*{suffix_synonyms_local[last_word_local]}")

            condition_before = re.compile(
                r"\b(?:with|where|having|who|that|whose|average|avg|mean|overall|global)\s*$",
                flags=re.I,
            )
            condition_after = re.compile(
                r"^\s*(?:"
                r"greater(?:\s+than)?|less(?:\s+than)?|higher(?:\s+than)?|lower(?:\s+than)?|"
                r"above|below|over|under|between|before|after|than|"
                r"equal(?:s|(?:\s+to)?)|not\s+null|is\s+null|is\s+not\s+null|"
                r"like|regexp|matches?|matching|condition\b"
                r")",
                flags=re.I,
            )

            for pat in phrase_patterns:
                for m in re.finditer(rf"\b(?:the\s+)?{pat}\b", seg, flags=re.I):
                    before = seg[max(0, m.start() - 24):m.start()]
                    after = seg[m.end():m.end() + 48]
                    if condition_before.search(before) or condition_after.search(after):
                        continue
                    return True
            return False

        suffix_synonyms = AGENT_MISSING_COL_SUFFIX_SYNONYMS
        
        # all_valid_cols sudah disiapkan di atas
        for col_raw in all_valid_cols:
            col_lower = col_raw.lower()
            
            # STEP 2: Logic Check yang Lebih Akurat
            # Skip jika kolom sudah ada di SELECT tokens, atau jika ada wildcard (*)
            if col_lower in select_semantic_tokens or AGENT_REFINE_WILDCARD_TOKEN in select_text_chk:
                continue
            
            # Skip jika kolom sudah dipakai di WHERE/HAVING (artinya dipakai sebagai filter, bukan output)
            if col_lower in where_having_tokens and not _question_explicitly_requests_column(col_raw):
                continue

            # Treat common *_number and *_id forms as equivalent semantic outputs.
            # Example: employeeNumber requested, employee_id selected.
            m_num = re.match(r"^([a-z_]+)number$", col_lower)
            if m_num:
                ent = (m_num.group(1) or "").rstrip("_")
                if ent and (
                    f"{ent}id" in select_semantic_tokens
                    or f"{ent}_id" in select_semantic_tokens
                    or (ent in select_semantic_tokens and "id" in select_semantic_tokens)
                ):
                    continue

            m_id = re.match(r"^([a-z_]+)id$", col_lower)
            if m_id:
                ent = (m_id.group(1) or "").rstrip("_")
                if ent and (
                    f"{ent}number" in select_semantic_tokens
                    or f"{ent}_number" in select_semantic_tokens
                    or (ent in select_semantic_tokens and "number" in select_semantic_tokens)
                ):
                    continue
            
            clean_name = re.sub(AGENT_COLUMN_CAMEL_SPLIT_LOWER_PATTERN, r'\1 \2', col_raw)
            clean_name = clean_name.replace('_', ' ').lower()
            parts = clean_name.split()
            if not parts: continue

            parts_set = {p for p in parts if p}
            if parts_set and any(_parts_covered_by_alias(parts_set, token_set) for token_set in projected_alias_token_sets):
                continue

            last_word = parts[-1]
            if last_word in suffix_synonyms:
                prefix = " ".join(parts[:-1])
                pattern_suffix = suffix_synonyms[last_word]
                # FIX: Allow optional words BEFORE the pattern (e.g., "earliest order date" should match "orderDate")
                # Pattern: (any words) + prefix + suffix
                pattern = rf'\b(?:\w+\s+)*{re.escape(prefix)}\s*{pattern_suffix}\b|\b{re.escape(prefix)}\s*{pattern_suffix}\b'
            else:
                # Also allow optional words before for non-suffix columns
                pattern = rf'\b(?:\w+\s+)*{re.escape(clean_name)}\b|\b{re.escape(clean_name)}\b'

            if re.search(pattern, requested_output_text, re.IGNORECASE):
                missing_cols.append(col_raw)

        if missing_cols:
            if _question_wants_winner_row(question):
                recovered_sql = _recover_missing_outer_winner_row_columns(final_sql, missing_cols)
                if recovered_sql:
                    final_sql = str(recovered_sql).strip()
                    if final_sql and not final_sql.endswith(";"):
                        final_sql += ";"
                    sql_up = final_sql.upper()
                    select_match_chk = re.search(AGENT_REFINE_SELECT_FROM_CLAUSE_PATTERN, final_sql, re.IGNORECASE | re.DOTALL)
                    select_text_chk_raw = select_match_chk.group(1) if select_match_chk else ""
                    select_text_chk = select_text_chk_raw.lower()
                    select_tokens_chk = set(re.findall(AGENT_REFINE_SQL_TOKEN_PATTERN, select_text_chk))
                    select_semantic_tokens = set(select_tokens_chk)
                    select_tokens_raw_case = set(re.findall(AGENT_REFINE_SQL_TOKEN_PATTERN, select_text_chk_raw))
                    for tok_raw in select_tokens_raw_case:
                        tok_low = tok_raw.lower()
                        if not tok_low:
                            continue
                        select_semantic_tokens.add(tok_low)
                        for part in tok_low.split("_"):
                            if part:
                                select_semantic_tokens.add(part)
                        camel_spaced = re.sub(AGENT_COLUMN_CAMEL_SPLIT_LOWER_PATTERN, r"\1 \2", tok_raw)
                        for part in re.findall(r"[a-z0-9]+", camel_spaced.lower()):
                            if part:
                                select_semantic_tokens.add(part)

                    missing_cols = [c for c in missing_cols if str(c or "").lower() not in select_semantic_tokens]
                    if not missing_cols:
                        _dbg_agent3_retry("Missing Columns", "Autofixed winner-row projection deterministically.")

        if missing_cols and complexity == AGENT_ROUTER_COMPLEXITY_SIMPLE and qtype == AGENT_ROUTER_QTYPE_LOOKUP:
            recovered_joined_lookup_sql = _recover_missing_joined_lookup_columns(
                final_sql,
                missing_cols,
                question,
                ctx,
            )
            if recovered_joined_lookup_sql:
                final_sql = str(recovered_joined_lookup_sql).strip()
                if final_sql and not final_sql.endswith(";"):
                    final_sql += ";"
                sql_up = final_sql.upper()
                corrections.append("Applied deterministic joined-lookup projection recovery")

                select_match_chk = re.search(
                    AGENT_REFINE_SELECT_FROM_CLAUSE_PATTERN,
                    final_sql,
                    re.IGNORECASE | re.DOTALL,
                )
                select_text_chk_raw = select_match_chk.group(1) if select_match_chk else ""
                select_text_chk = select_text_chk_raw.lower()
                select_tokens_chk = set(re.findall(AGENT_REFINE_SQL_TOKEN_PATTERN, select_text_chk))
                select_semantic_tokens = set(select_tokens_chk)
                select_tokens_raw_case = set(
                    re.findall(AGENT_REFINE_SQL_TOKEN_PATTERN, select_text_chk_raw)
                )
                for tok_raw in select_tokens_raw_case:
                    tok_low = tok_raw.lower()
                    if not tok_low:
                        continue
                    select_semantic_tokens.add(tok_low)
                    for part in tok_low.split("_"):
                        if part:
                            select_semantic_tokens.add(part)
                    camel_spaced = re.sub(
                        AGENT_COLUMN_CAMEL_SPLIT_LOWER_PATTERN,
                        r"\1 \2",
                        tok_raw,
                    )
                    for part in re.findall(r"[a-z0-9]+", camel_spaced.lower()):
                        if part:
                            select_semantic_tokens.add(part)

                missing_cols = [
                    c for c in missing_cols
                    if str(c or "").lower() not in select_semantic_tokens
                ]
                if not missing_cols:
                    _dbg_agent3_retry(
                        "Missing Columns",
                        "Autofixed joined lookup projection deterministically.",
                    )

            # TIERING LOGIC - Smart escalation based on complexity
            if not missing_cols:
                pass
            elif complexity == AGENT_ROUTER_COMPLEXITY_SIMPLE and qtype == AGENT_ROUTER_QTYPE_LOOKUP:
                # For flat lookup queries, prefer deterministic projection repair
                # over regeneration so we do not accidentally rewrite a correct
                # WHERE/REGEXP clause just to add a missing output column.
                projection_fixed_sql = _ensure_explicit_simple_projection_columns(final_sql, question, ctx)
                if projection_fixed_sql and projection_fixed_sql != final_sql:
                    final_sql = str(projection_fixed_sql).strip()
                    if final_sql and not final_sql.endswith(";"):
                        final_sql += ";"
                    projection_fix_post = _align_requested_description_variant(final_sql, question, ctx)
                    if projection_fix_post and projection_fix_post != final_sql:
                        final_sql = str(projection_fix_post).strip()
                        if final_sql and not final_sql.endswith(";"):
                            final_sql += ";"
                    projection_fix_post = _drop_unrequested_difference_operand_columns(final_sql, question, ctx)
                    if projection_fix_post and projection_fix_post != final_sql:
                        final_sql = str(projection_fix_post).strip()
                        if final_sql and not final_sql.endswith(";"):
                            final_sql += ";"
                    sql_up = final_sql.upper()
                    corrections.append("Applied deterministic projection-only repair")

                    select_match_chk = re.search(
                        AGENT_REFINE_SELECT_FROM_CLAUSE_PATTERN,
                        final_sql,
                        re.IGNORECASE | re.DOTALL,
                    )
                    select_text_chk_raw = select_match_chk.group(1) if select_match_chk else ""
                    select_text_chk = select_text_chk_raw.lower()
                    select_tokens_chk = set(re.findall(AGENT_REFINE_SQL_TOKEN_PATTERN, select_text_chk))
                    select_semantic_tokens = set(select_tokens_chk)
                    select_tokens_raw_case = set(
                        re.findall(AGENT_REFINE_SQL_TOKEN_PATTERN, select_text_chk_raw)
                    )
                    for tok_raw in select_tokens_raw_case:
                        tok_low = tok_raw.lower()
                        if not tok_low:
                            continue
                        select_semantic_tokens.add(tok_low)
                        for part in tok_low.split("_"):
                            if part:
                                select_semantic_tokens.add(part)
                        camel_spaced = re.sub(
                            AGENT_COLUMN_CAMEL_SPLIT_LOWER_PATTERN,
                            r"\1 \2",
                            tok_raw,
                        )
                        for part in re.findall(r"[a-z0-9]+", camel_spaced.lower()):
                            if part:
                                select_semantic_tokens.add(part)

                    still_missing_cols = []
                    for col_raw in missing_cols:
                        col_lower = str(col_raw or "").lower()
                        if not col_lower:
                            continue
                        if col_lower in select_semantic_tokens:
                            continue
                        still_missing_cols.append(col_raw)
                    missing_cols = still_missing_cols
                    if not missing_cols:
                        _dbg_agent3_retry(
                            "Missing Columns",
                            "Autofixed projection deterministically without regeneration.",
                        )
                if missing_cols:
                    _dbg_agent3_retry("Missing Columns", f"Retry (no auto-append): {missing_cols}")
                    extra_instruction = (
                        f"STRICT REQUIREMENT: The user explicitly asked for these columns: {missing_cols}. "
                        f"But they are MISSING from the SELECT clause. "
                        f"You MUST include: {', '.join(missing_cols)} in the SELECT output."
                    )
                    needs_regeneration = True
                    regeneration_model = draft_model if draft_model in {SMALL_MODEL, LARGE_MODEL} else SMALL_MODEL
                    if has_join(final_sql) and regeneration_model == SMALL_MODEL:
                        regeneration_model = LARGE_MODEL
                        _dbg_agent3_retry(
                            "Missing Columns",
                            "Escalating regeneration to 7B for joined lookup query.",
                        )
            
            elif draft_model == SMALL_MODEL and complexity == AGENT_ROUTER_COMPLEXITY_SIMPLE:
                # MEDIUM ERROR: Regenerate dengan 3B (simple query dari 3B yang salah)
                _dbg_agent3_retry("Missing Columns", f"Retry with 3B: {missing_cols}")
                extra_instruction = (
                    f"STRICT REQUIREMENT: The user explicitly asked for these columns: {missing_cols}. "
                    f"But they are MISSING from the SELECT clause. "
                    f"You MUST include: {', '.join(missing_cols)} in the output."
                )
                needs_regeneration = True
                regeneration_model = SMALL_MODEL  # Retry dengan 3B
            
            else:
                # HARD ERROR: Complex query atau 7B draft → escalate ke 7B
                _dbg_agent3_retry("Critical Omission", f"{missing_cols}")
                extra_instruction = (
                    f"STRICT REQUIREMENT: The user explicitly asked for these columns: {missing_cols}. "
                    f"Please add {missing_cols} to the SELECT clause."
                )
                needs_regeneration = True
                regeneration_model = LARGE_MODEL  # Force 7B

    # --- KASUS G: BROKEN ALIAS SYNTAX (HARD ERROR) ---
    if not needs_regeneration:
        if re.search(AGENT_REFINE_BROKEN_ALIAS_PATTERN, sql_up.lower()):
             _dbg_agent3_retry("Syntax Error", "Broken alias syntax `alias.(...)` detected.")
             extra_instruction = (
                "SYNTAX ERROR: You generated invalid SQL syntax like `table_alias.(condition)`. "
                "CORRECT: `(table_alias.column IS NULL ...)`."
             )
             needs_regeneration = True
             regeneration_model = LARGE_MODEL  # Force 7B

    # --- KASUS H: REGEX PATTERN MISMATCH (HARD ERROR) ---
    if not needs_regeneration:
        requires_strict_pattern = any(
            re.search(p, question, re.IGNORECASE) for p in AGENT_REGEX_TRIGGERS
        )
        uses_simple_like = AGENT_REFINE_LIKE_TOKEN in sql_up and not any(
            tok in sql_up for tok in AGENT_REFINE_REGEX_TOKENS
        )
        
        if requires_strict_pattern and uses_simple_like:
             _dbg_agent3_retry("Pattern Mismatch", "Question asks for strict pattern (digits), but SQL uses simple LIKE.")
             extra_instruction = (
                "STRICT REQUIREMENT: The user asks for a strict pattern (e.g. 'digits' vs 'characters', or exact length). "
                "Simple 'LIKE' cannot distinguish digits from characters. "
                "1. You MUST use `REGEXP` (or `REGEXP_LIKE`). "
                "2. Use `[0-9]` for digits. "
                "3. Use anchors `^` (start) and `$` (end) if exact length is needed. "
             )
             needs_regeneration = True
             regeneration_model = LARGE_MODEL  # Force 7B

    # --- KASUS I: UNSAFE MONTH SORTING (HARD ERROR) ---
    if AGENT_REFINE_MONTH_TOKEN in sql_up and AGENT_REFINE_ORDER_TOKEN in sql_up and not needs_regeneration:
        has_month_name_group = re.search(AGENT_REFINE_MONTH_NAME_GROUP_PATTERN, sql_up, re.DOTALL)
        # FIX: Accept sorting by date columns (any alias) as safe - gives chronological order
        # Pattern matches: MONTH(date), YEAR(date), MIN(...), MAX(...), or any *Date column
        has_safe_sort = re.search(
            AGENT_REFINE_SAFE_MONTH_SORT_FUNC_PATTERN,
            sql_up, re.DOTALL | re.IGNORECASE
        ) or re.search(
            AGENT_REFINE_SAFE_MONTH_SORT_DATE_PATTERN,
            sql_up, re.IGNORECASE
        )

        if has_month_name_group and not has_safe_sort:
            repaired_month_sort = _repair_chronological_month_order(final_sql, question)
            if repaired_month_sort != final_sql:
                final_sql = repaired_month_sort
                corrections.append("Deterministic repair: enforced chronological month ordering")
                sql_up = final_sql.upper()
                has_safe_sort = re.search(
                    AGENT_REFINE_SAFE_MONTH_SORT_FUNC_PATTERN,
                    sql_up, re.DOTALL | re.IGNORECASE
                ) or re.search(
                    AGENT_REFINE_SAFE_MONTH_SORT_DATE_PATTERN,
                    sql_up, re.IGNORECASE
                )

        if has_month_name_group and not has_safe_sort:
            _dbg_agent3_retry("Unsafe Month Sorting", "Sorting by Month Name (Alphabetical) detected.")
            extra_instruction = (
                "STRICT REQUIREMENT: You are grouping by Month Name, which sorts Alphabetically (Apr, Aug...). "
                "FIX: You MUST Group and Sort by the Month Number as well. "
                "Example: `GROUP BY MONTH(date_col), month_name ORDER BY MONTH(date_col) DESC`."
            )
            needs_regeneration = True
            regeneration_model = LARGE_MODEL  # Force 7B

    # --- KASUS K: AGGREGATION VIOLATION (ERROR 1055) (HARD ERROR) ---
    # Menangkap kasus SELECT raw_col FROM... GROUP BY derived_col (Error 1055)
    if AGENT_REFINE_GROUP_BY_TOKEN in sql_up and not needs_regeneration:
        group_clause = re.search(AGENT_REFINE_GROUP_CLAUSE_PATTERN, sql_up, re.DOTALL)
        if group_clause:
            group_text = group_clause.group(1).lower()
            # Risky cols: kolom yang ada di dalam ekspresi Group By (e.g. CASE WHEN msrp < ...)
            risky_cols = set(re.findall(AGENT_REFINE_GROUP_RISKY_COMPARE_PATTERN, group_text))
            risky_cols.update(re.findall(AGENT_REFINE_GROUP_RISKY_PAREN_PATTERN, group_text))
            risky_cols = {c for c in risky_cols if c in all_valid_cols}
            
            select_clause = re.search(AGENT_REFINE_SELECT_FROM_CLAUSE_PATTERN, sql_up, re.DOTALL).group(1).lower()
            
            for col in risky_cols:
                # Cek apakah 'col' muncul di SELECT tanpa fungsi agregasi
                matches = re.finditer(rf"\b{col}\b", select_clause)
                for m in matches:
                    start, end = m.span()
                    is_func = select_clause[end:end + AGENT_REFINE_FUNC_LOOKAHEAD_LEN].strip().startswith("(")
                    prev_text = select_clause[max(0, start - AGENT_REFINE_ALIAS_WINDOW_CHARS):start].upper()
                    is_alias = (
                        AGENT_REFINE_ALIAS_TOKEN_WITH_SPACE in prev_text
                        or AGENT_REFINE_ALIAS_TOKEN in prev_text
                    )
                    
                    if not is_func and not is_alias:
                         _dbg_agent3_retry("Aggregation Violation", f"Raw column '{col}' selected but grouped by derived value.")
                         extra_instruction = (
                            f"STRICT REQUIREMENT: You are grouping by a derived value of '{col}' (e.g. CASE or DATE_FORMAT), "
                            f"but you are selecting the raw '{col}' column. This causes SQL Error 1055. "
                            f"FIX: Remove '{col}' from the SELECT clause. Only select the grouping key and aggregates (COUNT, SUM)."
                         )
                         needs_regeneration = True
                         regeneration_model = LARGE_MODEL  # Force 7B
                         break
                if needs_regeneration: break

    # --- FINAL PRE-EXEC ACCEPTANCE GATE (deterministic) ---
    pre_gate = _deterministic_sql_acceptance_gate(final_sql, ctx=ctx, question=question)
    gated_sql = str(pre_gate.get("sql") or final_sql)
    if gated_sql and gated_sql != final_sql:
        final_sql = gated_sql
        corrections.append("Deterministic acceptance gate rewrites applied")

    parse_only_reasons = {str(r) for r in (pre_gate.get("reasons") or []) if str(r).strip()}
    if (
        not pre_gate.get("accepted", False)
        and parse_only_reasons == {"parse_error"}
        and _can_bypass_parse_only_gate(sql=final_sql, ctx=ctx, decision=decision)
    ):
        pre_gate = {
            **dict(pre_gate or {}),
            "accepted": True,
            "reasons": [],
            "parse_error": "",
        }
        corrections.append("Deterministic acceptance gate: bypassed parse-only false positive for safe family")

    pre_reason_labels = {str(r) for r in (pre_gate.get("reasons") or []) if str(r).strip()}
    llm_semantic_retry_eligible = _is_llm_semantic_retry_gate(pre_gate)
    if pre_gate.get("accepted", False):
        print("[LLM] Semantic retry: not needed")
    elif llm_semantic_retry_eligible:
        print("[LLM] Semantic retry: eligible")
    elif pre_reason_labels and pre_reason_labels.issubset({"semantic_contract_mismatch"}):
        print("[LLM] Semantic retry: skipped (non-whitelisted)")

    if not needs_regeneration and not pre_gate.get("accepted", False):
        pre_reasons = pre_reason_labels
        if llm_semantic_retry_eligible:
            _dbg_agent3_retry("Acceptance Gate - Semantic", f"{pre_gate.get('semantic_failures')}")
            extra_instruction = _build_gate_retry_instruction(
                (
                    "STRICT REQUIREMENT: The SQL is structurally valid but still mismatches the "
                    "question's semantic contract. Fix only the ORDER BY/LIMIT/aggregate mismatch "
                    "while preserving already-valid identifiers and join structure."
                ),
                pre_gate,
            )
            needs_regeneration = True
            regeneration_model = draft_model if draft_model in {SMALL_MODEL, LARGE_MODEL} else SMALL_MODEL
            corrections.append("Bounded LLM semantic retry requested")
        # Non-whitelisted soft semantic mismatch: keep original SQL to avoid
        # churn on lower-confidence semantics, but keep the warning visible.
        elif pre_reasons and pre_reasons.issubset({"semantic_contract_mismatch"}):
            corrections.append(f"Soft semantic warning: {sorted(pre_reasons)}")
        else:
            _dbg_agent3_retry("Acceptance Gate", f"{pre_gate.get('reasons')}")
            extra_instruction = _build_gate_retry_instruction("", pre_gate)
            needs_regeneration = True
            regeneration_model = draft_model if draft_model in {SMALL_MODEL, LARGE_MODEL} else SMALL_MODEL

    # 4. EXECUTE REGENERATION (validation-aware, AST-stabilized)
    final_model = state.get("target_model", "")  # Default to initial model
    if needs_regeneration:
        def _allowed_identifiers_hint(max_tables: int = 3, max_cols: int = 10) -> str:
            allowed_map = _parse_allowed_identifiers(ctx)
            if not allowed_map:
                return ""
            chunks: List[str] = []
            for i, (tbl, cols) in enumerate(allowed_map.items()):
                if i >= max_tables:
                    break
                cols_sorted = sorted(str(c) for c in (cols or set()) if str(c).strip())
                if len(cols_sorted) > max_cols:
                    cols_sorted = cols_sorted[:max_cols]
                chunks.append(f"{tbl}: {', '.join(cols_sorted)}")
            if not chunks:
                return ""
            return "ALLOWED IDENTIFIERS SNAPSHOT:\n- " + "\n- ".join(chunks)

        def _join_constraints_hint(max_edges: int = 8) -> str:
            allowed_map = _parse_allowed_identifiers(ctx)
            if not allowed_map:
                return ""

            allowed_tables = {str(t).strip("`").lower() for t in allowed_map.keys() if str(t).strip()}
            routed_tables = {str(t).strip("`").lower() for t in (detected_tables or []) if str(t).strip()}
            if routed_tables:
                table_scope = allowed_tables.intersection(routed_tables) or routed_tables
            else:
                table_scope = allowed_tables

            if not table_scope:
                return ""

            edges: List[str] = []
            seen: Set[str] = set()

            fk_map = {}
            try:
                fk_map = get_fk_relationships() or {}
            except Exception:
                fk_map = {}

            for src, rels in fk_map.items():
                src_l = str(src or "").strip("`").lower()
                if not src_l or src_l not in table_scope:
                    continue
                for fk_col, tgt in (rels or {}).items():
                    tgt_l = str(tgt or "").strip("`").lower()
                    fk_l = str(fk_col or "").strip("`").lower()
                    if not tgt_l or not fk_l or tgt_l not in table_scope:
                        continue
                    tgt_id = (get_id_column(tgt_l) or "").strip("`")
                    if tgt_id:
                        edge = f"{src_l}.{fk_l} = {tgt_l}.{tgt_id}"
                    else:
                        edge = f"{src_l}.{fk_l} -> {tgt_l}"
                    if edge not in seen:
                        seen.add(edge)
                        edges.append(edge)
                    if len(edges) >= max_edges:
                        break
                if len(edges) >= max_edges:
                    break

            fk_ctx_map = _extract_fk_from_context(ctx)
            for src, rels in (fk_ctx_map or {}).items():
                src_l = str(src or "").strip("`").lower()
                if not src_l or src_l not in table_scope:
                    continue
                for fk_col, tgt in (rels or {}).items():
                    tgt_l = str(tgt or "").strip("`").lower()
                    fk_l = str(fk_col or "").strip("`").lower()
                    if not tgt_l or not fk_l or tgt_l not in table_scope:
                        continue
                    tgt_id = (get_id_column(tgt_l) or "").strip("`")
                    if tgt_id:
                        edge = f"{src_l}.{fk_l} = {tgt_l}.{tgt_id}"
                    else:
                        edge = f"{src_l}.{fk_l} -> {tgt_l}"
                    if edge not in seen:
                        seen.add(edge)
                        edges.append(edge)
                    if len(edges) >= max_edges:
                        break
                if len(edges) >= max_edges:
                    break

            if not edges:
                return ""
            return "JOIN KEY HINTS:\n- " + "\n- ".join(edges)

        def _regen_constraint_pack(instruction: str, strict_schema: bool = False) -> str:
            pieces = [instruction]
            base_rules = [
                "STRICT OUTPUT CONSTRAINTS:",
                "- Use ONLY tables/columns from ALLOWED IDENTIFIERS SNAPSHOT.",
                "- Use JOIN predicates only from JOIN KEY HINTS (or exact FK-equivalent).",
                "- Do NOT invent table names, columns, or aliases outside allowed scope.",
            ]
            if strict_schema:
                base_rules.extend([
                    "- Keep SQL minimal and deterministic; avoid unnecessary derived tables.",
                    "- If single-source query is possible, avoid extra joins.",
                ])
            pieces.append("\n".join(base_rules))

            allowed_hint = _allowed_identifiers_hint()
            if allowed_hint:
                pieces.append(allowed_hint)
            join_hint = _join_constraints_hint()
            if join_hint:
                pieces.append(join_hint)
            return "\n\n".join(pieces)

        def _regen_once(model_name: str, instruction: str, tag: str, strict_schema: bool = False) -> Optional[Dict[str, Any]]:
            llm_regen_local = build_llm_repair(model=model_name)
            model_label_local = "7B" if model_name == LARGE_MODEL else "3B"
            _dbg_print_two("[REFINER] Regenerating with", f"{model_label_local} ({tag})")

            packed_instruction = _regen_constraint_pack(instruction, strict_schema=strict_schema)
            new_sql_local, _ = regenerate_with_extra_constraint(
                llm_regen_local,
                make_generation_prompt,
                ctx,
                question,
                packed_instruction,
                prompt_kwargs={
                    "router_decision": decision,
                    "state": state,
                },
            )
            if not new_sql_local:
                return None

            candidate_sql = normalize_sql_pipeline(str(new_sql_local).strip(), question, ctx)
            gate_local = _deterministic_sql_acceptance_gate(
                candidate_sql,
                ctx=ctx,
                question=question,
                strict_join_key=AGENT_REGEN_PRECHECK_ENABLED,
            )
            candidate_sql = str(gate_local.get("sql") or candidate_sql).strip()
            if not candidate_sql.endswith(";"):
                candidate_sql += ";"

            id_errors_local = list(gate_local.get("identifier_errors") or [])
            hard_errors_local = list(gate_local.get("hard_identifier_errors") or [])
            join_warnings_local = list(gate_local.get("join_risk_warnings") or [])
            gate_reasons_local = list(gate_local.get("reasons") or [])
            return {
                "sql": candidate_sql,
                "model": model_name,
                "tag": tag,
                "identifier_errors": id_errors_local,
                "hard_identifier_errors": hard_errors_local,
                "join_risk_warnings": join_warnings_local,
                "join_risk_count": len(join_warnings_local),
                "gate_accepted": bool(gate_local.get("accepted", False)),
                "gate_reasons": gate_reasons_local,
                "gate_parse_error": str(gate_local.get("parse_error") or ""),
                "group_projection_violations": list(gate_local.get("group_projection_violations") or []),
                "ambiguous_unqualified_columns": list(gate_local.get("ambiguous_unqualified_columns") or []),
                "invalid_join_edges": list(gate_local.get("invalid_join_edges") or []),
                "fanout_risk_tables": list(gate_local.get("fanout_risk_tables") or []),
                "semantic_contract": dict(gate_local.get("semantic_contract") or {}),
                "semantic_failures": list(gate_local.get("semantic_failures") or []),
                "early_rejected": bool(
                    AGENT_REGEN_PRECHECK_ENABLED and not bool(gate_local.get("accepted", False))
                ),
            }

        attempts: List[Dict[str, Any]] = []
        primary_attempt = _regen_once(regeneration_model, extra_instruction, "primary")
        if primary_attempt:
            attempts.append(primary_attempt)

        current_attempt = primary_attempt

        # Single targeted retry with the SAME model when gate still fails.
        if current_attempt and not current_attempt.get("gate_accepted", False):
            current_reasons = list(current_attempt.get("gate_reasons") or [])
            if _should_expand_retrieval_context(current_reasons):
                expanded_ctx_result = build_expanded_context_from_state(state)
                if expanded_ctx_result.get("used", False):
                    ctx = str(expanded_ctx_result.get("context") or ctx)
                    state["context"] = ctx
                    retrieval_context_mode = "expanded"
                    retrieval_retry_used = True
                    retrieval_retry_reason = str(expanded_ctx_result.get("reason") or "expanded_on_gate_failure")
                    retrieval_expanded_table_count = int(expanded_ctx_result.get("expanded_table_count") or 0)
                    state["retrieval_context_mode"] = retrieval_context_mode
                    state["retrieval_retry_used"] = retrieval_retry_used
                    state["retrieval_retry_reason"] = retrieval_retry_reason
                    state["retrieval_expanded_table_count"] = retrieval_expanded_table_count
                    corrections.append(
                        f"Expanded retrieval context fallback used ({retrieval_expanded_table_count} tables)"
                    )
                elif RETRIEVE_DEBUG:
                    print(
                        "[RETRIEVAL - EXPAND] Skipped: "
                        f"{expanded_ctx_result.get('reason') or 'unknown_reason'}"
                    )

            retry_instruction = _build_gate_retry_instruction(
                extra_instruction,
                {
                    "reasons": current_reasons,
                    "hard_identifier_errors": current_attempt.get("hard_identifier_errors") or [],
                    "identifier_errors": current_attempt.get("identifier_errors") or [],
                    "ambiguous_unqualified_columns": current_attempt.get("ambiguous_unqualified_columns") or [],
                    "group_projection_violations": current_attempt.get("group_projection_violations") or [],
                    "join_risk_warnings": current_attempt.get("join_risk_warnings") or [],
                    "invalid_join_edges": current_attempt.get("invalid_join_edges") or [],
                    "fanout_risk_tables": current_attempt.get("fanout_risk_tables") or [],
                    "semantic_contract": current_attempt.get("semantic_contract") or {},
                    "semantic_failures": current_attempt.get("semantic_failures") or [],
                    "parse_error": current_attempt.get("gate_parse_error") or "",
                },
            )
            targeted_retry = _regen_once(
                regeneration_model,
                retry_instruction,
                "targeted_retry",
                strict_schema=True,
            )
            if targeted_retry:
                attempts.append(targeted_retry)
                current_attempt = targeted_retry

        # Escalate to 7B only when 3B primary+retry still fail.
        if (
            regeneration_model == SMALL_MODEL
            and (not current_attempt or not current_attempt.get("gate_accepted", False))
        ):
            escalation_instruction = _build_gate_retry_instruction(
                extra_instruction,
                {
                    "reasons": (current_attempt or {}).get("gate_reasons") if current_attempt else [],
                    "hard_identifier_errors": (current_attempt or {}).get("hard_identifier_errors") if current_attempt else [],
                    "identifier_errors": (current_attempt or {}).get("identifier_errors") if current_attempt else [],
                    "ambiguous_unqualified_columns": (current_attempt or {}).get("ambiguous_unqualified_columns") if current_attempt else [],
                    "group_projection_violations": (current_attempt or {}).get("group_projection_violations") if current_attempt else [],
                    "join_risk_warnings": (current_attempt or {}).get("join_risk_warnings") if current_attempt else [],
                    "invalid_join_edges": (current_attempt or {}).get("invalid_join_edges") if current_attempt else [],
                    "fanout_risk_tables": (current_attempt or {}).get("fanout_risk_tables") if current_attempt else [],
                    "semantic_contract": (current_attempt or {}).get("semantic_contract") if current_attempt else {},
                    "semantic_failures": (current_attempt or {}).get("semantic_failures") if current_attempt else [],
                    "parse_error": (current_attempt or {}).get("gate_parse_error") if current_attempt else "",
                },
            )
            fallback_7b = _regen_once(
                LARGE_MODEL,
                escalation_instruction,
                "fallback_7b",
                strict_schema=True,
            )
            if fallback_7b:
                attempts.append(fallback_7b)
                current_attempt = fallback_7b

        if attempts:
            accepted_attempt = next(
                (a for a in attempts if bool(a.get("gate_accepted", False))),
                None,
            )
            if accepted_attempt is not None:
                best = accepted_attempt
            else:
                best = min(
                    attempts,
                    key=lambda a: (
                        len(a.get("hard_identifier_errors") or []),
                        len(a.get("gate_reasons") or []),
                        int(a.get("join_risk_count", 0)),
                        len(a.get("identifier_errors") or []),
                    ),
                )

            final_sql = str(best.get("sql") or final_sql)
            final_model = str(best.get("model") or final_model)
            best_label = "7B" if final_model == LARGE_MODEL else "3B"
            corrections.append(f"Regenerated ({best_label}) [{best.get('tag')}]")

            if best.get("gate_reasons"):
                corrections.append(f"Residual acceptance warnings: {best.get('gate_reasons')}")
            if best.get("hard_identifier_errors"):
                corrections.append(f"Residual identifier warnings: {best.get('hard_identifier_errors')}")
            if best.get("join_risk_warnings"):
                corrections.append(f"Join-key warnings: {best.get('join_risk_warnings')}")
            if best.get("invalid_join_edges"):
                corrections.append(f"Invalid join edges: {best.get('invalid_join_edges')}")
            if best.get("fanout_risk_tables"):
                corrections.append(f"Dual-fact fanout risk: {best.get('fanout_risk_tables')}")
            if best.get("semantic_failures"):
                corrections.append(f"Semantic contract warnings: {best.get('semantic_failures')}")
        else:
            corrections.append("Regeneration failed: no candidate SQL returned")

    final_gate = _deterministic_sql_acceptance_gate(final_sql, ctx=ctx, question=question)
    final_sql = str(final_gate.get("sql") or final_sql).strip()

    final_parse_only_reasons = {str(r) for r in (final_gate.get("reasons") or []) if str(r).strip()}
    if (
        not final_gate.get("accepted", False)
        and final_parse_only_reasons == {"parse_error"}
        and _can_bypass_parse_only_gate(sql=final_sql, ctx=ctx, decision=decision)
    ):
        final_gate = {
            **dict(final_gate or {}),
            "accepted": True,
            "reasons": [],
            "parse_error": "",
        }
        corrections.append("Final acceptance gate: bypassed parse-only false positive for safe family")

    rescue_7b_used = False
    if not final_gate.get("accepted", False) and _is_critical_gate_failure(final_gate):
        _dbg_safety_block("Agent3", f"{final_gate.get('reasons')}")
        rescue = _attempt_last_chance_rescue_7b(
            question=question,
            ctx=ctx,
            current_sql=final_sql,
            gate=final_gate,
            strict_join_key=AGENT_REGEN_PRECHECK_ENABLED,
            router_decision=decision,
            state=state,
        )
        rescue_7b_used = bool(rescue.get("used", False))
        if rescue_7b_used:
            final_sql = str(rescue.get("sql") or final_sql).strip()
            final_gate = rescue.get("gate") or final_gate
            if rescue.get("accepted", False):
                final_model = LARGE_MODEL
                corrections.append("Last-chance rescue succeeded (7B)")
            else:
                corrections.append(
                    f"Last-chance rescue failed: {rescue.get('error')} | reasons={final_gate.get('reasons')}"
                )

    if not final_gate.get("accepted", False) and _is_critical_gate_failure(final_gate):
        final_sql = "SELECT 'FINAL_GATE_BLOCKED' AS error_message;"
        final_model = "gate_blocked"
        corrections.append(f"Fail-closed: blocked unsafe SQL ({final_gate.get('reasons')})")
    elif not final_gate.get("accepted", False):
        corrections.append(f"Final acceptance warnings: {final_gate.get('reasons')}")

    if not final_sql.endswith(";"):
        final_sql += ";"

    _dbg_print_two("[REFINER] final sql:", final_sql)

    # Final validation snapshot for observability in benchmark output.
    final_validation_errors = list(final_gate.get("identifier_errors") or validate_identifiers(final_sql, ctx))
    final_gate_pass = bool(final_gate.get("accepted", False))
    final_gate_reasons = list(final_gate.get("reasons") or [])
    final_gate_reason_tags = _categorize_gate_reasons(final_gate_reasons)

    # END TIMER: Calculate processing time (Model ready → Final SQL complete)
    # This is the core text-to-SQL processing latency for thesis benchmark
    processing_start = state.get("processing_start_time")
    if processing_start:
        processing_time = _time.perf_counter() - processing_start
        state["processing_time"] = processing_time
        print(f"[DEBUG] Processing time: {processing_time:.2f}s")
        draft_inf = float(state.get("draft_inference_time", 0.0) or 0.0)
        postproc_time = max(0.0, processing_time - draft_inf)
        print(
            f"[DEBUG] Time breakdown: draft={draft_inf:.2f}s, "
            f"postprocess={postproc_time:.2f}s"
        )

    # Compute confidence score (schema-agnostic)
    confidence_result = compute_sql_confidence(final_sql, question, ctx)
    _log_confidence_result(confidence_result)
    
    # Confidence-based retry (1x max) if below threshold
    confidence_retry_used = False
    if confidence_result["confidence"] < AGENT_CONFIDENCE_THRESHOLD:
        retry_result = _attempt_confidence_retry(
            question=question,
            ctx=ctx,
            current_sql=final_sql,
            confidence_result=confidence_result,
            make_prompt_fn=make_generation_prompt,
            prompt_kwargs={
                "router_decision": decision,
                "state": state,
            },
            decision=decision,
        )
        confidence_retry_used = retry_result.get("used", False)
        if retry_result.get("accepted", False):
            final_sql = retry_result["sql"]
            confidence_result = retry_result["confidence"]
            final_validation_errors = list(retry_result.get("validation_errors") or final_validation_errors)
            final_gate_pass = bool(retry_result.get("gate_pass", final_gate_pass))
            final_gate_reasons = list(retry_result.get("gate_reasons") or final_gate_reasons)
            final_gate_reason_tags = list(retry_result.get("gate_reason_tags") or _categorize_gate_reasons(final_gate_reasons))
            corrections.append("Confidence retry succeeded (7B)")
            final_model = LARGE_MODEL
        else:
            retry_reason = str(retry_result.get("reason") or "not_accepted").strip()
            corrections.append(f"Confidence retry attempted ({retry_reason})")

    return {
        **state,
        "validation_errors": final_validation_errors,
        "final_sql": final_sql,
        "answer": final_sql,
        "retrieval_context_mode": retrieval_context_mode,
        "retrieval_retry_used": retrieval_retry_used,
        "retrieval_retry_reason": retrieval_retry_reason,
        "retrieval_expanded_table_count": retrieval_expanded_table_count,
        "critic_feedback": " | ".join(corrections),
        "final_model": final_model,
        "final_gate_pass": final_gate_pass,
        "final_gate_reasons": final_gate_reasons,
        "final_gate_reason_tags": final_gate_reason_tags,
        "rescue_7b_used": bool(rescue_7b_used),
        "sql_confidence": confidence_result["confidence"],
        "confidence_signals": confidence_result["signals"],
        "confidence_warnings": confidence_result["warnings"],
        "confidence_low": confidence_result["confidence"] < AGENT_CONFIDENCE_THRESHOLD,
        "confidence_retry_used": confidence_retry_used,
    }
