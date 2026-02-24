# ======================================================================
# ============================ agents.py ===============================
# ======================================================================
# Purpose:
# - Agents 0 --> Guardrail for question validation (hybrid regex + LLM)
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
    OOM_WAIT_TIME,
    OOM_MAX_RETRIES,
    AGENT_IR_ENABLED,
    AGENT_IR_MIN_CONFIDENCE,
    AGENT_IR_DEBUG,
    AGENT_IR_ALLOW_MULTI_TABLE,
    AGENT_IR_MULTI_TABLE_MIN_CONFIDENCE,
    AGENT_IR_MAX_TABLES,
    AGENT_AST_GUARDS_ENABLED,
    AGENT_AST_GUARDS_DEBUG,
    AGENT_REGEN_PRECHECK_ENABLED,
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
    AGENT_ROUTER_ORDINAL_WORDS,
    AGENT_ROUTER_ORDINAL_TARGET_TERMS,
    AGENT_ROUTER_WITHIN_SAME_PATTERN,
    AGENT_ROUTER_BY_TOKEN,
    AGENT_ROUTER_EXPLICIT_SORT_TERMS,
    AGENT_ROUTER_CROSS_TABLE_EXCLUSION_TERMS,
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
)
from .llm_utils import build_llm_json, build_llm_repair
from .state import RAGState
from .utils import _dbg_print_two, try_llm_json, build_router_schema
from .generation import regenerate_with_extra_constraint
from .intent_ir import build_intent_ir, is_ir_plannable
from .sql_planner import build_sql_from_intent, safety_normalize_sql
from .schema_analyzer import (
    get_all_tables,
    get_singular_form,
    get_schema_analyzer,
    get_id_column,
    get_fk_relationships,
    get_transaction_tables,
)
from .helpers_main import (get_single_allowed_table, validate_identifiers, _get_requires_regex, _regex_requirement_ok,
    _regex_requirement_ok, _get_qtype, make_router_prompt, make_generation_prompt, sanitize_notes, normalize_sql_pipeline,
    _find_tables_with_implicit_columns, _extract_fk_from_context)
from .helpers_core import (
    has_join,
    _ast_guardrails,
    _question_requires_word_boundary_regex,
    _question_requires_inner_join,
    _question_wants_include_zero,
    _parse_allowed_identifiers,
    _remove_top_level_order_by_if_not_requested,
    _SQL_KEYWORDS,
)
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
    if any(term in scope for term in AGENT_ROUTER_CROSS_TABLE_EXCLUSION_TERMS):
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
        if re.match(pattern, q_lower, re.IGNORECASE):
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
        print(f"[Guard] LLM judge failed: {e}, allowing question through")
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
    print(f"[Guard] Validating question: '{question[:50]}...'")
    
    # Layer 1: Instant reject obvious garbage
    should_reject, reason = _check_instant_reject(question)
    if should_reject:
        print(f"[Guard] Layer 1 REJECT (regex): {reason[:50]}")
        return False, reason
    
    # Layer 1b: Instant accept definite SQL
    if _check_definite_sql(question):
        print(f"[Guard] ✓ Layer 1 ACCEPT (definite SQL pattern)")
        return True, ""
    
    # Layer 2: LLM judge for borderline cases
    print(f"[Guard] → Layer 2: LLM 1.5B judging...")
    is_valid, reason = _llm_judge_sql_intent(question)
    
    if is_valid:
        print(f"[Guard] ✓ Layer 2 ACCEPT (LLM approved)")
    else:
        print(f"[Guard] Layer 2 REJECT (LLM): {reason[:50]}")
    
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

    # Token-level fallback (skip weak generic tokens)
    stop_tokens = AGENT_COLUMN_STOP_TOKENS
    for token in re.findall(AGENT_COLUMN_TOKEN_PATTERN, spaced):
        if len(token) < 4 or token in stop_tokens:
            continue
        # "ordered by ..." usually indicates sorting intent, not orderdetails semantics
        if (
            AGENT_COLUMN_ORDERED_TOKEN
            and token == AGENT_COLUMN_ORDERED_TOKEN
            and re.search(AGENT_COLUMN_ORDERED_BY_PATTERN, question_lower)
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
def router_agent(state: RAGState) -> RAGState:
    """
    Router agent - determines query complexity and target model.
    Note: Question guard is now handled by separate question_guard_node (runs before retrieve).
    """
    print(f"\n[Router] Evidence-Based Analysis starting...")
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

        # Buat copy pertanyaan khusus untuk scanning tabel ini
        search_scope = question

        if table_singular in ignore_patterns:
            patterns = ignore_patterns[table_singular]
            for pat in patterns:
                search_scope = re.sub(pat, " ", search_scope)

        # Cek 2: Cari kata tabel di search_scope yang SUDAH BERSIH
        if re.search(rf"\b{table_pattern}\b", search_scope):

            # Deteksi referensi ID saja (FK)
            is_just_id_ref = re.search(
                rf"\b{table_pattern}\b\s+(?:{id_terms_pattern})\b",
                search_scope
            )

            if is_just_id_ref:
                if re.search(
                    rf"\b{table_pattern}\b(?!\s+(?:{id_terms_pattern}))",
                    search_scope
                ):
                    relevant_tables.append(tbl)
                    if RETRIEVE_DEBUG:
                        print(f"[Router] ✓ Added {tbl} (with ID ref check)")
            else:
                relevant_tables.append(tbl)
                if RETRIEVE_DEBUG:
                    print(f"[Router] ✓ Added {tbl} (direct match)")
        elif RETRIEVE_DEBUG:
            pattern_str = rf"\b{table_pattern}\b"
            print(f"[Router] ✗ Skipped {tbl} (no match in question)")
            print(f"[Router]   Pattern: {pattern_str}")
            print(f"[Router]   In question? {bool(re.search(pattern_str, search_scope))}")
    
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
                                print(f"[Router] Including {original_tbl} (implicit column mention)")

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
        if src not in whitelisted_lower:
            continue
        for source_col, target_table in (fk_map or {}).items():
            tgt = (target_table or "").lower()
            if tgt not in whitelisted_lower:
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
                print(f"[Router] Evidence table: {tbl} (evidence: {', '.join(evidence)})")

    # STEP 2.8: Remove parent tables that are only referenced by "for each" pattern
    # If a table is ONLY added because of "for each/per/each" pattern (no column mentions),
    # AND another table has the FK column linking to it, skip the parent table.
    # Example: "per customer" + payments.has(customerNumber) → skip customers table
    tables_to_remove = []
    for tbl_lower in list(evidence_tables):
        evidence_list = evidence_details.get(tbl_lower, [])

        # Preserve weak-evidence parent table when it is explicitly the
        # output subject in the question (e.g., "Show product lines ...")
        # and has table+attribute evidence.
        table_entity_pattern = _table_entity_pattern(tbl_lower)
        has_table_attr_evidence = AGENT_ROUTER_EVIDENCE_TAG_TABLE_ATTRIBUTE in evidence_list
        explicit_subject_score = _explicit_subject_match_score(question_lower, table_entity_pattern)
        is_explicit_subject = has_table_attr_evidence and (explicit_subject_score > 0)
        if is_explicit_subject:
            if RETRIEVE_DEBUG:
                print(f"[Router] Keeping {tbl_lower}: explicit subject entity in question")
            continue

        # Preserve weak tables that are explicitly the target of exclusion intent
        # (e.g., "customers with no orders", "never paid").
        if _question_targets_entity_exclusion(question_lower, tbl_lower):
            if RETRIEVE_DEBUG:
                print(f"[Router] Keeping {tbl_lower}: exclusion target entity in question")
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

        # If the entity is explicitly mentioned in non-ID form and evidence comes
        # from FK columns, keep it only when it is a true bridge node between
        # at least two relevant tables (prevents unnecessary joins in single-table
        # aggregates/window queries like payments/customer or employees/offices).
        if has_fk_col_evidence and has_non_id_entity_mention and is_multi_hop_bridge:
            if RETRIEVE_DEBUG:
                print(f"[Router] Keeping {tbl_lower}: explicit entity mention + FK evidence")
            continue

        if is_only_weak and evidence_list:  # Has evidence, but all are weak
            table_singular = get_singular_form(tbl_lower)
            table_id_col = (get_id_column(tbl_lower) or "").lower()

            # Check if any other table has FK column linking to this table
            fk_exists_in_other_table = False
            for other_tbl in relevant_tables:
                if other_tbl.lower() == tbl_lower:
                    continue
                other_cols = _get_table_cols(other_tbl)
                for col in other_cols:
                    # Check if this is an FK column (e.g., customerNumber → customers)
                    col_lower = str(col).lower()
                    # Pattern: <table_singular>(Number|Code|Id|ID)
                    if (table_id_col and col_lower == table_id_col) or _is_conventional_fk_name(table_singular, col_lower):
                        fk_exists_in_other_table = True
                        break
                if fk_exists_in_other_table:
                    break

            if fk_exists_in_other_table:
                # Check if question mentions any non-FK columns from this table
                tbl_cols = sorted(_get_table_cols(tbl_lower))
                # Get FK columns
                fk_cols = set()
                for col in tbl_cols:
                    col_lower = str(col).lower()
                    if (table_id_col and col_lower == table_id_col) or _is_conventional_fk_name(table_singular, col_lower):
                        fk_cols.add(col_lower)

                # Check if any non-FK column is mentioned in question
                non_fk_mentioned = False
                for col in tbl_cols:
                    col_low = str(col).lower()
                    if col_low in fk_cols:
                        continue
                    # Check if column is mentioned (including spaced variant)
                    spaced = re.sub(AGENT_COLUMN_CAMEL_SPLIT_LOWER_PATTERN, r'\1 \2', str(col)).lower()
                    if col_low in question_lower or spaced in question_lower:
                        non_fk_mentioned = True
                        break

                if not non_fk_mentioned:
                    # Safe to remove - only referenced via "for each" and FK exists elsewhere
                    tables_to_remove.append(tbl_lower)
                    if RETRIEVE_DEBUG:
                        print(f"[Router] Skipping {tbl_lower}: only 'for each' ref, FK exists in other table")
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
                                print(f"[Router] Skipping {tbl_lower}: weak-only FK evidence, strong sibling table covers shared key")

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
        pat = _table_entity_pattern(tbl_name.lower())
        if not pat:
            return False
        return bool(re.search(rf"\b{pat}\b", question_text))

    def _question_mentions_table_temporal_column(question_text: str, tbl_name: str) -> bool:
        tbl_l = (tbl_name or "").lower()
        for c in temporal_cols_by_table.get(tbl_l, _get_table_cols(tbl_name)):
            if not _is_temporal_col_name(c):
                continue
            if _question_mentions_column(question_text, str(c)):
                return True
        return False

    tables_to_remove = []
    for tbl in relevant_tables:
        tbl_lower = tbl.lower()

        # Keep if it's an evidence table
        if tbl_lower in evidence_tables:
            if RETRIEVE_DEBUG:
                print(f"[Router] Keeping {tbl}: evidence table")
            continue

        # Keep if it's in the connecting subgraph
        if tbl_lower in tables_in_subgraph:
            if RETRIEVE_DEBUG:
                print(f"[Router] Keeping {tbl}: in connecting subgraph")
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
                    print(f"[Router] Keeping {tbl}: has FK to evidence table + needs non-FK columns")
                continue

        # Temporal recency/year/date questions may need a date-bearing table
        # that is not part of the minimal FK connector among seed tables.
        if has_temporal_intent and _table_has_temporal_column(tbl):
            if (
                _question_mentions_table_entity(question_lower, tbl)
                or _question_mentions_table_temporal_column(question_lower, tbl)
            ):
                if RETRIEVE_DEBUG:
                    print(f"[Router] Keeping {tbl}: temporal intent + date-bearing table")
                continue

        # Mark for removal
        tables_to_remove.append(tbl)
        if RETRIEVE_DEBUG:
            print(f"[Router] Removing {tbl}: not in minimal subgraph")

    # Apply removals
    for tbl in tables_to_remove:
        if tbl in relevant_tables:
            relevant_tables.remove(tbl)

    # SAFETY: Never leave relevant_tables empty - fallback to first whitelisted table
    if not relevant_tables and whitelisted_tables:
        fallback_table = list(whitelisted_tables)[0]
        relevant_tables.append(fallback_table)
        if RETRIEVE_DEBUG:
            print(f"[Router] All tables removed, falling back to: {fallback_table}")

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
                        print(f"[Router] Keeping {original}: bridge table for FK connectivity")

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
    is_ranking_time = any(re.search(rf"\b{re.escape(kw)}\b", question_lower) for kw in time_ranks)
    is_ranking = is_ranking_value or is_ranking_time
    
    # EXPLICIT RANK FUNCTION DETECTION: "a rank of X" / "rank employees" / "ranking"
    # This means user explicitly wants RANK() OVER window function
    # E.g., "return a rank of employees within each job title"
    explicit_rank_patterns = AGENT_ROUTER_EXPLICIT_RANK_PATTERNS
    is_explicit_rank_function = any(re.search(pat, question_lower) for pat in explicit_rank_patterns)
    
    # ORDINAL RANKING DETECTION: "second highest", "third largest", "2nd lowest", etc.
    # These patterns ALWAYS need ROW_NUMBER() with rn = N, much more complex than simple MAX/MIN
    ordinal_words = AGENT_ROUTER_ORDINAL_WORDS
    ordinal_targets = AGENT_ROUTER_ORDINAL_TARGET_TERMS
    ordinal_pattern = rf"\b({'|'.join(re.escape(w) for w in ordinal_words)})\s+({'|'.join(re.escape(w) for w in ordinal_targets)})\b"
    is_ordinal_ranking = bool(re.search(ordinal_pattern, question_lower))
    
    # WINDOW AGGREGATE DETECTION: "within the same X" / "in the same X" pattern
    # This pattern requires COUNT(*) OVER (PARTITION BY X), NOT GROUP BY
    # E.g., "total employees within the same office" → PARTITION BY officeCode
    within_same_pattern = AGENT_ROUTER_WITHIN_SAME_PATTERN
    within_same_match = re.search(within_same_pattern, question_lower)
    is_window_aggregate = bool(within_same_match)
    partition_hint = within_same_match.group(2) if within_same_match else None
    
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
        print(f"[Router] Primary Table: {primary_table} (reason: {primary_reason})")

    # ═══════════════════════════════════════════════════════════════════
    # STEP 5: LOGIKA ESKALASI (Smarter Multi-Table Handling)
    # ═══════════════════════════════════════════════════════════════════
    is_multi_table = len(relevant_tables) > 1
    
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
        print(f"[Router] Ordinal ranking detected ('second highest' etc) -> requires ROW_NUMBER subquery")
    
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
    
    # FIXED: Exclusion detection - only trigger for TRUE exclusion patterns
    # These patterns imply NOT EXISTS / NOT IN / LEFT JOIN ... IS NULL (cross-table exclusion)
    # NOT triggered by: "missing" in single-table context (just IS NULL check)
    has_exclusion = False
    
    # Pattern 1: Cross-table exclusion keywords (always need 7B)
    cross_table_exclusion = AGENT_ROUTER_CROSS_TABLE_EXCLUSION_TERMS
    if any(kw in question_lower for kw in cross_table_exclusion):
        has_exclusion = True
    
    # Pattern 2: "no" + entity noun (dynamic, schema-driven) → implies cross-table
    # Examples: "no orders", "no payments", "no customers"
    if not has_exclusion:
        entity_terms = set()
        for tbl in whitelisted_tables:
            for v in _table_name_variants(tbl):
                entity_terms.add(re.escape(v).replace(r"\ ", r"\s+"))

        if entity_terms:
            entity_pattern = "|".join(sorted(entity_terms, key=len, reverse=True))
            if re.search(rf"\bno\s+(?:{entity_pattern})\b", question_lower):
                has_exclusion = True
    
    # Pattern 3: "missing" or "empty" ONLY if multi-table context
    # Single-table "missing" = just IS NULL (simple, 3B can handle)
    if not has_exclusion and is_multi_table:
        if re.search(AGENT_ROUTER_MISSING_EMPTY_PATTERN, question_lower):
            has_exclusion = True

    # Subquery intent cues (e.g., compare against average/overall baseline, EXISTS wording).
    # This is intentionally lightweight and only used for escalation policy.
    is_subquery_intent = bool(re.search(AGENT_ROUTER_SUBQUERY_INTENT_PATTERN, question_lower))
    has_complex_subquery_cue = bool(
        re.search(AGENT_ROUTER_COMPLEX_SUBQUERY_CUE_PATTERN, question_lower)
    )
    has_exists_keyword = bool(re.search(r"\b(?:exists|not\s+exists)\b", question_lower))

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
        and bool(re.search(AGENT_ROUTER_SIMPLE_SINGLE_TABLE_SUBQUERY_PATTERN, question_lower))
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

    # Multi-fact rollup detector (fanout-risk):
    # customer + orders count + payments total/status in one grouped result.
    # Example: total orders + total payment amount + payment_status.
    relevant_tables_lower = {t.lower() for t in relevant_tables}
    has_dual_fact_rollup = bool(
        is_multi_table
        and "orders" in relevant_tables_lower
        and "payments" in relevant_tables_lower
        and (
            "customers" in relevant_tables_lower
            or bool(re.search(r"\bcustomer", question_lower))
        )
        and bool(
            re.search(
                r"\b(total\s+number\s+of|number\s+of|count\s+of)\s+orders?\b|\btotal_orders?\b",
                question_lower,
                flags=re.I,
            )
        )
        and bool(
            re.search(
                r"\b(total\s+payment(?:\s+amount)?|sum\s+of\s+payments?|total_payments?|total_payment_amount)\b",
                question_lower,
                flags=re.I,
            )
            or re.search(r"\bpayment[_\s]*status\b", question_lower, flags=re.I)
            or ("paid" in question_lower and "unpaid" in question_lower)
        )
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
    if has_dual_fact_rollup:
        escalation_reasons.append("multi_fact_rollup_fanout_risk")

    # keep order, remove duplicates
    escalation_reasons = list(dict.fromkeys(escalation_reasons))
    should_escalate = bool(escalation_reasons)

    if is_multi_table and not should_escalate and AGENT_ROUTER_ALLOW_SIMPLE_MULTI_TABLE_3B:
        print(f"[Router] Multi-table simple pattern detected: {relevant_tables} -> keeping 3B")
    elif escalation_reasons:
        print(f"[Router] Escalating to 7B. reasons={escalation_reasons}")

    chosen_model = LARGE_MODEL if should_escalate else SMALL_MODEL
    complexity = AGENT_ROUTER_COMPLEXITY_COMPLEX if should_escalate else AGENT_ROUTER_COMPLEXITY_SIMPLE
    
    # Tentukan QTYPE
    if is_winner_row or is_explicit_rank_function:
        qtype = AGENT_ROUTER_QTYPE_RANK
    elif has_math or has_grouping:
        qtype = AGENT_ROUTER_QTYPE_AGGREGATE
    else: 
        qtype = AGENT_ROUTER_QTYPE_LOOKUP
    
    print(f"[Router] Detected Tables: {relevant_tables}")
    print(
        f"[Router] Signals -> Math:{has_math}, Sort:{is_sorting}, Rank:{is_ranking}, "
        f"Group:{has_grouping}, WinnerRow:{is_winner_row}, OrdinalRank:{is_ordinal_ranking}, "
        f"Exclusion:{has_exclusion}, WindowAgg:{is_window_aggregate}, "
        f"ExplicitRank:{is_explicit_rank_function}, SubqueryIntent:{is_subquery_intent}, "
        f"SimpleScalarSubquery:{is_simple_scalar_subquery_single_table}, "
        f"SimpleExistsSubquery:{is_simple_exists_subquery}, "
        f"DualFactRollup:{has_dual_fact_rollup}"
    )
    print(f"[Router] Decision: {complexity.upper()} -> {chosen_model}")

    return {
        **state,
        "router_tables": relevant_tables,
        "primary_table": primary_table,
        "primary_table_reason": primary_reason,
        "target_model": chosen_model,
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


def _collect_hard_identifier_errors(errors: List[str]) -> List[str]:
    if not errors:
        return []
    return [
        e for e in errors
        if "not in ALLOWED" in e or "invalid column" in e or "invalid columns" in e
    ]

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
        "ast_fixes": [],
        "ast_warnings": [],
    }

    candidate_sql = (sql or "").strip()
    if not candidate_sql:
        verdict["reasons"].append("empty_sql")
        return verdict

    ast_sql, ast_meta = _ast_guardrails(candidate_sql, ctx=ctx, question=question)
    if ast_meta.get("parse_ok", False):
        if ast_meta.get("fixes"):
            verdict["ast_fixes"] = list(ast_meta.get("fixes") or [])
            candidate_sql = ast_sql
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

        parsed = sqlglot.parse_one(candidate_sql, read="mysql")
        sel = parsed.find(exp.Select)
        if sel is None:
            verdict["reasons"].append("no_select")
            verdict["sql"] = candidate_sql
            return verdict

        def _is_nested_subquery_column(col_expr: "exp.Column") -> bool:
            anc = getattr(col_expr, "parent", None)
            while anc is not None:
                if anc is sel:
                    return False
                if isinstance(anc, exp.Subquery):
                    return True
                anc = getattr(anc, "parent", None)
            return False

        def _norm_expr(expr_in: "exp.Expression") -> str:
            s = expr_in.sql(dialect="mysql")
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

            return False

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
                if has_group_or_agg and len(txn_tables_in_metrics) >= 2:
                    verdict["fanout_risk_tables"] = sorted(txn_tables_in_metrics)
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

    except Exception as e:
        verdict["parse_error"] = f"acceptance_gate_error:{e}"
        verdict["reasons"].append("acceptance_gate_internal_error")

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

    parse_error = gate.get("parse_error") or ""
    if parse_error:
        lines.append(f"- Parse error: {parse_error}")

    return "\n".join(lines)


def _is_critical_gate_failure(gate: Dict[str, Any]) -> bool:
    reasons = {str(r) for r in (gate.get("reasons") or [])}
    critical_reasons = {
        "parse_error",
        "hard_identifier",
        "ambiguous_unqualified",
        "group_projection_mismatch",
        "invalid_join_path",
        "fanout_risk_dual_fact",
    }
    if reasons.intersection(critical_reasons):
        return True
    return bool(gate.get("parse_error"))


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
    _dbg_print_two("[Agent3] Last-chance rescue", "Regenerating with 7B")

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
            print(f"[DRAFT] Model {target_model} still warm (idle {idle_time:.1f}s < keep_alive {keep_alive_sec}s), skip warmup.")
            return True

        """
        Pre-warm model with OOM retry logic.
        Returns True if model is ready, False if failed after all retries.
        """
        for attempt in range(max_retries):
            try:
                print(f"[Model] Loading: {target_model}..." + (f" (attempt {attempt+1})" if attempt > 0 else ""))
                llm.invoke("Say OK")  # Quick call to load model into memory
                print(f"[Model] Ready: {target_model}")
                _last_model_warmup[target_model] = _time.time()
                return True
            except Exception as e:
                error_msg = str(e).lower()
                if AGENT_PREWARM_MEMORY_ERROR_TOKEN in error_msg:
                    print(f"[DRAFT OOM] Memory issue detected. Waiting {OOM_WAIT_TIME}s for previous model to unload...")
                    _time.sleep(OOM_WAIT_TIME)
                    print(f"[DRAFT OOM] Retrying model load...")
                else:
                    print(f"[DRAFT] Pre-warm failed: {e}")
                    return False
        
        print(f"[DRAFT] Failed to load model after {max_retries} OOM retries")
        return False


def draft_sql_agent(state: RAGState) -> RAGState:
    # --- Guard: Skip if question was rejected by router ---
    decision = state.get("router_decision", {})
    if decision.get("complexity") == AGENT_ROUTER_COMPLEXITY_REJECTED or decision.get("is_valid_question") is False:
        print("[DRAFT] Skipped: Question was rejected by router guard")
        return state  # Pass through unchanged

    question = state.get("question", "")
    ctx = state.get("context", "")
    # Route telemetry defaults
    state["ir_used"] = False
    state["ir_confidence"] = 0.0
    state["ir_plannable"] = False
    state["ir_gate_reasons"] = []
    state["ir_table_count"] = 0
    state["ir_context_table_count"] = 0

    # ------------------------------------------------------------------
    # IR-FIRST deterministic planning (schema-agnostic)
    # ------------------------------------------------------------------
    if AGENT_IR_ENABLED:
        intent_ir = build_intent_ir(
            question=question,
            ctx=ctx,
            router_decision=decision,
            primary_table=state.get("primary_table", ""),
        )
        state["intent_ir"] = intent_ir

        mentioned_tables = intent_ir.get("mentioned_tables") or []
        router_tables = state.get("router_tables") or []
        ctx_allowed_ids = _parse_allowed_identifiers(ctx)
        ctx_table_count = len(ctx_allowed_ids)
        table_count = max(len(mentioned_tables), len(router_tables), ctx_table_count)
        dynamic_min_conf = float(AGENT_IR_MIN_CONFIDENCE)
        ir_gate_reasons: List[str] = []
        ir_confidence = float(intent_ir.get("confidence", 0.0) or 0.0)

        if table_count > AGENT_IR_MAX_TABLES:
            ir_gate_reasons.append(
                f"table_count_exceeds_ir_max:{table_count}>{AGENT_IR_MAX_TABLES}"
            )
        if ctx_table_count > AGENT_IR_MAX_TABLES:
            ir_gate_reasons.append(
                f"context_table_count_exceeds_ir_max:{ctx_table_count}>{AGENT_IR_MAX_TABLES}"
            )

        if table_count > 1:
            dynamic_min_conf = max(dynamic_min_conf, float(AGENT_IR_MULTI_TABLE_MIN_CONFIDENCE))
            if not AGENT_IR_ALLOW_MULTI_TABLE:
                ir_gate_reasons.append("multi_table_ir_disabled")

        hard_router_flags = {
            "is_winner_row": "winner_row",
            "has_exclusion": "exclusion",
            "is_window_aggregate": "window_aggregate",
            "is_explicit_rank_function": "explicit_rank",
            "is_ordinal_ranking": "ordinal_rank",
            "is_subquery_intent": "subquery_intent",
        }
        for flag, label in hard_router_flags.items():
            if bool(decision.get(flag, False)):
                ir_gate_reasons.append(f"router_hard:{label}")

        # Percentage intent: allow IR only for supported metric patterns.
        q_low = (question or "").lower()
        if re.search(r"\b(percent|percentage|%\s*of|ratio|rate)\b", q_low):
            ir_metrics = intent_ir.get("metrics") or []
            has_supported_pct_metric = any(
                str(m.get("agg") or "").lower() == "pct_non_null"
                for m in ir_metrics
                if isinstance(m, dict)
            )
            if not has_supported_pct_metric:
                ir_gate_reasons.append("percentage_intent_ir_disabled")

        base_plannable = is_ir_plannable(intent_ir, dynamic_min_conf)
        ir_plannable = base_plannable and not ir_gate_reasons
        state["ir_confidence"] = ir_confidence
        state["ir_plannable"] = bool(ir_plannable)
        state["ir_gate_reasons"] = list(ir_gate_reasons)
        state["ir_table_count"] = int(table_count)
        state["ir_context_table_count"] = int(ctx_table_count)

        if AGENT_IR_DEBUG:
            print(
                "[IR] confidence="
                f"{ir_confidence:.2f}, "
                f"plannable={ir_plannable}, "
                f"table_count={table_count}, "
                f"context_table_count={ctx_table_count}, "
                f"min_conf={dynamic_min_conf:.2f}, "
                f"gate_reasons={ir_gate_reasons or ['none']}"
            )

        if ir_plannable:
            planned_sql, plan_meta = build_sql_from_intent(intent_ir, ctx)
            if planned_sql:
                normalized_sql, normalize_warnings = safety_normalize_sql(planned_sql)
                if AGENT_IR_DEBUG:
                    print(f"[IR] SQL planned successfully. warnings={normalize_warnings}")
                if not normalized_sql.endswith(";"):
                    normalized_sql += ";"
                return {
                    **state,
                    "processing_start_time": _time.perf_counter(),
                    "draft_inference_time": 0.0,
                    "draft_source": "ir_planner",
                    "ir_used": True,
                    "join_plan": plan_meta.get("join_plan", {}),
                    "draft_sql": normalized_sql,
                    "agent_notes": (
                        "Deterministic IR planner used."
                        + (f" warnings={normalize_warnings}" if normalize_warnings else "")
                    ),
                }
            else:
                if AGENT_IR_DEBUG:
                    print(f"[IR] planner fallback -> LLM. errors={plan_meta.get('errors')}")
        elif AGENT_IR_DEBUG and ir_gate_reasons:
            print(f"[IR] gated off -> LLM. reasons={ir_gate_reasons}")

    target_model = state.get("target_model") or LARGE_MODEL
    llm = build_llm_json(model=target_model)

    # ===== PRE-WARM WITH OOM RETRY =====
    # This separates cold-start time from actual inference time
    # Also handles OOM by waiting for previous model to unload
    model_ready = _prewarm_model_with_oom_retry(llm, target_model)

    if not model_ready:
        return {**state, "draft_source": "llm", "draft_sql": "SELECT 'Error: Model OOM after retries';"}

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
    
    # ===== TIMED INFERENCE (model already loaded) =====
    _t0 = _time.perf_counter()
    obj = try_llm_json(llm, prompt, max_attempts=AGENT_DRAFT_TRY_JSON_MAX_ATTEMPTS, oom_retry_wait=OOM_WAIT_TIME)
    _inference_time = _time.perf_counter() - _t0

    # Store inference time in state for debugging (not logged to reduce noise)
    state["draft_inference_time"] = _inference_time

    if obj is None:
        return {**state, "draft_source": "llm", "draft_sql": "SELECT 'Error: LLM failed';"}
        
    draft_sql = (obj.get("sql") or "SELECT 1;").strip()
    reasoning = (obj.get("notes") or "No reasoning provided.").strip()
    print(f"\n[DRAFT] Raw from LLM: {draft_sql}")
    print(f"[DRAFT] LLM Reasoning: {reasoning}")
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
        """Strip DESC/ASC before comma only in SELECT clause, not in ORDER BY."""
        # Find SELECT ... FROM boundary
        match = re.search(AGENT_DRAFT_SELECT_FROM_PATTERN, sql, re.IGNORECASE | re.DOTALL)
        if not match:
            return sql
        select_part = match.group(1)
        # Strip DESC/ASC before comma only in SELECT part
        fixed_select = re.sub(AGENT_DRAFT_STRIP_DESC_ASC_BEFORE_COMMA_PATTERN, " ", select_part, flags=re.IGNORECASE)
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
        has_limit_intent = bool(
            re.search(rf"\b{re.escape(limit_n)}\b", question.lower()) or
            (limit_terms_pattern and re.search(rf"\b(?:{limit_terms_pattern})\b", question.lower()))
        )
        if not has_limit_intent:
            draft_sql = re.sub(AGENT_DRAFT_LIMIT_REMOVE_PATTERN, "", draft_sql, flags=re.IGNORECASE).strip()
    
    if not draft_sql.endswith(";"): draft_sql += ";"
    print(f"[DRAFT] After cleanup: {draft_sql}")
    return {
        **state,
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

    # ------------------------------------------------------------------
    # IR planner path: safety-only normalization + identifier validation.
    # Skip heavy regex rewrite/regeneration to keep deterministic behavior.
    # ------------------------------------------------------------------
    if draft_source == "ir_planner":
        final_sql, warnings = safety_normalize_sql(draft_sql)
        corrections = ["Deterministic IR planner output"]
        gate = _deterministic_sql_acceptance_gate(final_sql, ctx=ctx, question=question)
        current_gate = gate
        final_sql = str(gate.get("sql") or final_sql)
        validation_errors = list(gate.get("identifier_errors") or [])
        if warnings:
            corrections.append(f"Safety warnings: {warnings}")
        if gate.get("ast_fixes"):
            corrections.append(f"AST gate fixes: {gate.get('ast_fixes')}")
        if not gate.get("accepted", False):
            corrections.append(f"Acceptance gate blocked SQL: {gate.get('reasons')}")
        if validation_errors or gate.get("parse_error"):
            if validation_errors:
                corrections.append(f"Identifier validation errors: {validation_errors}")
            # Last-resort fallback to legacy normalizer for recovery.
            final_sql = normalize_sql_pipeline(final_sql, question, ctx)
            gate_after_fallback = _deterministic_sql_acceptance_gate(final_sql, ctx=ctx, question=question)
            current_gate = gate_after_fallback
            final_sql = str(gate_after_fallback.get("sql") or final_sql)
            validation_errors = list(gate_after_fallback.get("identifier_errors") or validation_errors)
            if not gate_after_fallback.get("accepted", False):
                corrections.append(
                    f"Post-fallback acceptance warnings: {gate_after_fallback.get('reasons')}"
                )

        rescue_7b_used = False
        ir_final_model = "ir_planner"
        if not current_gate.get("accepted", False) and _is_critical_gate_failure(current_gate):
            _dbg_print_two("[IR] FINAL GATE FAIL-CLOSED", f"{current_gate.get('reasons')}")
            rescue = _attempt_last_chance_rescue_7b(
                question=question,
                ctx=ctx,
                current_sql=final_sql,
                gate=current_gate,
                strict_join_key=AGENT_REGEN_PRECHECK_ENABLED,
            )
            rescue_7b_used = bool(rescue.get("used", False))
            if rescue_7b_used:
                final_sql = str(rescue.get("sql") or final_sql)
                current_gate = rescue.get("gate") or current_gate
                if rescue.get("accepted", False):
                    ir_final_model = LARGE_MODEL
                    corrections.append("Last-chance rescue succeeded (7B)")
                else:
                    corrections.append(
                        f"Last-chance rescue failed: {rescue.get('error')} | reasons={current_gate.get('reasons')}"
                    )

        if not current_gate.get("accepted", False) and _is_critical_gate_failure(current_gate):
            final_sql = "SELECT 'FINAL_GATE_BLOCKED' AS error_message;"
            ir_final_model = "gate_blocked"
            corrections.append(f"Fail-closed: blocked unsafe SQL ({current_gate.get('reasons')})")

        validation_errors = list(current_gate.get("identifier_errors") or validation_errors)
        if not final_sql.endswith(";"):
            final_sql += ";"

        processing_start = state.get("processing_start_time")
        if processing_start:
            processing_time = _time.perf_counter() - processing_start
            state["processing_time"] = processing_time
            print(f"[DEBUG] Processing time: {processing_time:.2f}s")

        return {
            **state,
            "validation_errors": validation_errors,
            "final_sql": final_sql,
            "answer": final_sql,
            "critic_feedback": " | ".join(corrections),
            "final_model": ir_final_model,
            "final_gate_pass": bool(current_gate.get("accepted", False)),
            "final_gate_reasons": list(current_gate.get("reasons") or []),
            "rescue_7b_used": bool(rescue_7b_used),
        }

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
                _dbg_print_two("[AST] warnings", f"{ast_warnings}")
                corrections.append(f"AST guardrails warnings: {ast_warnings}")
        elif AGENT_AST_GUARDS_DEBUG and ast_meta.get("error"):
            _dbg_print_two("[AST] parse skipped", f"{ast_meta.get('error')}")

    sql_up = final_sql.upper()

    # 2. PREPARE CHECKS (FULL PROTECTION)
    sql_for_tokenizing = final_sql
    sql_for_tokenizing = re.sub(AGENT_REFINE_SINGLE_QUOTED_STRING_PATTERN, " ", sql_for_tokenizing)
    sql_for_tokenizing = re.sub(AGENT_REFINE_DOUBLE_QUOTED_STRING_PATTERN, " ", sql_for_tokenizing)
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
        _dbg_print_two("[Agent3] HARD ERROR - Identifier Validation", f"{hard_identifier_errors}")
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
        ):
            regeneration_model = SMALL_MODEL
        else:
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
                    _dbg_print_two(
                        "[Agent3] HARD ERROR - Hallucination in Main Query", 
                        f"Columns {aliens_in_main} in SELECT but not in FROM table"
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
                        "[Agent3] Skipping hallucination check", 
                        "Alien columns only in subquery (different scope)"
                    )
            else:
                # Can't parse SELECT clause → skip conservatively
                _dbg_print_two("[Agent3] Skipping hallucination check", "Cannot parse SELECT clause")
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
                _dbg_print_two("[Agent3] HARD ERROR - Missing JOIN", f"Columns {alien_cols} belong to {potential_table}")
                extra_instruction = (
                    f"STRICT REQUIREMENT: Columns {alien_cols} do NOT exist in table '{primary_tbl}', but they belong to '{potential_table}'. "
                    f"Detected intent implies multiple tables: {detected_tables}. "
                    "You MUST use explicit JOIN (e.g., JOIN {potential_table} ON ...) to link the data."
                )
                needs_regeneration = True
                regeneration_model = LARGE_MODEL
                
            elif not can_be_solved_by_join:
                _dbg_print_two("[Agent3] HARD ERROR - Hallucination", f"Columns {alien_cols} do not exist in Schema.")
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
    if is_winner_row and not needs_regeneration:
        if re.search(AGENT_REFINE_WINNER_ROW_LEAK_PATTERN, sql_up):
             _dbg_print_two("[Agent3] HARD ERROR - Winner-Row Leak", "rn column found in outer select.")
             extra_instruction = (
                "STRICT REQUIREMENT: Use standard Winner-Row pattern. "
                "Do NOT include the 'rn' (row number) column in the final output."
             )
             needs_regeneration = True
             regeneration_model = LARGE_MODEL  # Force 7B

    # --- KASUS C: ANTI-PATTERN (WHERE IN Subquery) (HARD ERROR) ---
    tuple_in_pattern = re.search(AGENT_REFINE_TUPLE_IN_PATTERN, sql_up)
    if tuple_in_pattern and not needs_regeneration:
        _dbg_print_two("[Agent3] HARD ERROR - Anti-Pattern", "Tuple comparison inside WHERE IN (...)")
        extra_instruction = (
            "STRICT REQUIREMENT: Do NOT use the `WHERE (col1, col2) IN (SELECT ...)` pattern. "
            "It causes Error 1055. INSTEAD, use the Window Function pattern:\n"
            "SELECT ... FROM (SELECT ..., ROW_NUMBER() OVER (...) as rn FROM table) t WHERE rn = 1;"
        )
        needs_regeneration = True
        regeneration_model = LARGE_MODEL  # Force 7B

    # --- KASUS D: MISSING PARTITION KEY IN OUTPUT (HARD ERROR) ---
    if is_winner_row and not needs_regeneration:
        part_match = re.search(AGENT_REFINE_PARTITION_BY_PATTERN, sql_up)
        if part_match:
            part_col = part_match.group(1)
            select_clause_match = re.search(AGENT_REFINE_SELECT_FROM_CLAUSE_PATTERN, sql_up, re.DOTALL)
            if select_clause_match:
                select_clause = select_clause_match.group(1)
                if part_col not in select_clause.upper():
                    _dbg_print_two("[Agent3] HARD ERROR - Missing Partition Key", f"'{part_col}' missing from SELECT")
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
                 _dbg_print_two("[Agent3] HARD ERROR - INNER JOIN Trap", f"Filtering child table {risky_filters}")
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
        
        # STEP 1b: Ambil tokens dari WHERE/HAVING/subquery (kolom yang dipakai sebagai filter)
        # Kolom yang sudah dipakai di WHERE tidak perlu ada di SELECT (bukan "missing")
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

        suffix_synonyms = AGENT_MISSING_COL_SUFFIX_SYNONYMS
        
        # all_valid_cols sudah disiapkan di atas
        for col_raw in all_valid_cols:
            col_lower = col_raw.lower()
            
            # STEP 2: Logic Check yang Lebih Akurat
            # Skip jika kolom sudah ada di SELECT tokens, atau jika ada wildcard (*)
            if col_lower in select_semantic_tokens or AGENT_REFINE_WILDCARD_TOKEN in select_text_chk:
                continue
            
            # Skip jika kolom sudah dipakai di WHERE/HAVING (artinya dipakai sebagai filter, bukan output)
            if col_lower in where_having_tokens:
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

            if re.search(pattern, question, re.IGNORECASE):
                missing_cols.append(col_raw)

        if missing_cols:
            # TIERING LOGIC - Smart escalation based on complexity
            if complexity == AGENT_ROUTER_COMPLEXITY_SIMPLE and qtype == AGENT_ROUTER_QTYPE_LOOKUP:
                # Avoid SQL string surgery. Regenerate so final SQL naturally includes requested columns.
                _dbg_print_two("[Agent3] MEDIUM ERROR - Missing Columns", f"Retry (no auto-append): {missing_cols}")
                extra_instruction = (
                    f"STRICT REQUIREMENT: The user explicitly asked for these columns: {missing_cols}. "
                    f"But they are MISSING from the SELECT clause. "
                    f"You MUST include: {', '.join(missing_cols)} in the SELECT output."
                )
                needs_regeneration = True
                regeneration_model = draft_model if draft_model in {SMALL_MODEL, LARGE_MODEL} else SMALL_MODEL
            
            elif draft_model == SMALL_MODEL and complexity == AGENT_ROUTER_COMPLEXITY_SIMPLE:
                # MEDIUM ERROR: Regenerate dengan 3B (simple query dari 3B yang salah)
                _dbg_print_two("[Agent3] MEDIUM ERROR - Missing Columns", f"Retry with 3B: {missing_cols}")
                extra_instruction = (
                    f"STRICT REQUIREMENT: The user explicitly asked for these columns: {missing_cols}. "
                    f"But they are MISSING from the SELECT clause. "
                    f"You MUST include: {', '.join(missing_cols)} in the output."
                )
                needs_regeneration = True
                regeneration_model = SMALL_MODEL  # Retry dengan 3B
            
            else:
                # HARD ERROR: Complex query atau 7B draft → escalate ke 7B
                _dbg_print_two("[Agent3] HARD ERROR - Critical Omission", f"{missing_cols}")
                extra_instruction = (
                    f"STRICT REQUIREMENT: The user explicitly asked for these columns: {missing_cols}. "
                    f"Please add {missing_cols} to the SELECT clause."
                )
                needs_regeneration = True
                regeneration_model = LARGE_MODEL  # Force 7B

    # --- KASUS G: BROKEN ALIAS SYNTAX (HARD ERROR) ---
    if not needs_regeneration:
        if re.search(AGENT_REFINE_BROKEN_ALIAS_PATTERN, sql_up.lower()):
             _dbg_print_two("[Agent3] HARD ERROR - Syntax Error", "Broken alias syntax `alias.(...)` detected.")
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
             _dbg_print_two("[Agent3] HARD ERROR - Pattern Mismatch", "Question asks for strict pattern (digits), but SQL uses simple LIKE.")
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
            _dbg_print_two("[Agent3] HARD ERROR - Unsafe Month Sorting", "Sorting by Month Name (Alphabetical) detected.")
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
                         _dbg_print_two("[Agent3] HARD ERROR - Aggregation Violation", f"Raw column '{col}' selected but grouped by derived value.")
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

    if not needs_regeneration and not pre_gate.get("accepted", False):
        _dbg_print_two("[Agent3] HARD ERROR - Acceptance Gate", f"{pre_gate.get('reasons')}")
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
            _dbg_print_two("[Agent3] Regenerating with", f"{model_label_local} ({tag})")

            packed_instruction = _regen_constraint_pack(instruction, strict_schema=strict_schema)
            new_sql_local, _ = regenerate_with_extra_constraint(
                llm_regen_local,
                make_generation_prompt,
                ctx,
                question,
                packed_instruction,
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
            retry_instruction = _build_gate_retry_instruction(
                extra_instruction,
                {
                    "reasons": current_attempt.get("gate_reasons") or [],
                    "hard_identifier_errors": current_attempt.get("hard_identifier_errors") or [],
                    "identifier_errors": current_attempt.get("identifier_errors") or [],
                    "ambiguous_unqualified_columns": current_attempt.get("ambiguous_unqualified_columns") or [],
                    "group_projection_violations": current_attempt.get("group_projection_violations") or [],
                    "join_risk_warnings": current_attempt.get("join_risk_warnings") or [],
                    "invalid_join_edges": current_attempt.get("invalid_join_edges") or [],
                    "fanout_risk_tables": current_attempt.get("fanout_risk_tables") or [],
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
        else:
            corrections.append("Regeneration failed: no candidate SQL returned")

    final_gate = _deterministic_sql_acceptance_gate(final_sql, ctx=ctx, question=question)
    final_sql = str(final_gate.get("sql") or final_sql).strip()

    rescue_7b_used = False
    if not final_gate.get("accepted", False) and _is_critical_gate_failure(final_gate):
        _dbg_print_two("[Agent3] FINAL GATE FAIL-CLOSED", f"{final_gate.get('reasons')}")
        rescue = _attempt_last_chance_rescue_7b(
            question=question,
            ctx=ctx,
            current_sql=final_sql,
            gate=final_gate,
            strict_join_key=AGENT_REGEN_PRECHECK_ENABLED,
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

    # Final validation snapshot for observability in benchmark output.
    final_validation_errors = list(final_gate.get("identifier_errors") or validate_identifiers(final_sql, ctx))
    final_gate_pass = bool(final_gate.get("accepted", False))
    final_gate_reasons = list(final_gate.get("reasons") or [])

    # END TIMER: Calculate processing time (Model ready → Final SQL complete)
    # This is the core text-to-SQL processing latency for thesis benchmark
    processing_start = state.get("processing_start_time")
    if processing_start:
        processing_time = _time.perf_counter() - processing_start
        state["processing_time"] = processing_time
        print(f"[DEBUG] Processing time: {processing_time:.2f}s")

    return {
        **state,
        "validation_errors": final_validation_errors,
        "final_sql": final_sql,
        "answer": final_sql,
        "critic_feedback": " | ".join(corrections),
        "final_model": final_model,
        "final_gate_pass": final_gate_pass,
        "final_gate_reasons": final_gate_reasons,
        "rescue_7b_used": bool(rescue_7b_used),
    }
