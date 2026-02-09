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
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple
# ======================================================================
# ======================== LOCAL MODULE IMPORT =========================
# ======================================================================
from .config import ROUTING_MODEL, SMALL_MODEL, LARGE_MODEL, OLLAMA_URL, OLLAMA_KEEP_ALIVE, RETRIEVE_DEBUG
from .llm_utils import build_llm_json, build_llm_repair
from .state import RAGState
from .utils import _dbg_print_two, try_llm_json, build_router_schema
from .generation import regenerate_with_extra_constraint
from .helpers_main import (get_single_allowed_table, validate_identifiers, _get_requires_regex, _regex_requirement_ok,
    _regex_requirement_ok, _get_qtype, make_router_prompt, make_generation_prompt, sanitize_notes, normalize_sql_pipeline,
    _find_tables_with_implicit_columns, _extract_fk_from_context)
from .helpers_core import (has_join, _question_requires_word_boundary_regex, _parse_allowed_identifiers, _remove_top_level_order_by_if_not_requested, _SQL_KEYWORDS)
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
_INSTANT_REJECT_PATTERNS = [
    r"^\s*(hi|hello|hey|halo|hai|hallo)\s*[!.,]?\s*$",
    r"^\s*(good\s*)?(morning|afternoon|evening|night)\s*[!.,]?\s*$",
    r"^\s*(selamat\s+)?(pagi|siang|sore|malam)\s*[!.,]?\s*$",
    r"^\s*(apa\s+kabar|how\s+are\s+you|what'?s\s+up)\s*[?!.,]?\s*$",
    r"^\s*(thanks?|thank\s+you|terima\s*kasih|makasih)\s*[!.,]?\s*$",
    r"^\s*(bye|goodbye|see\s+you|sampai\s+jumpa)\s*[!.,]?\s*$",
    r"^\s*(yes|no|ya|tidak|ok|okay|oke|sip)\s*[!.,]?\s*$",
    r"^\s*(test|testing|tes|coba)\s*[!.,]?\s*$",
    r"^\s*[!?.,:;\-_=+*#@&]+\s*$",  # Just punctuation/symbols
    r"^\s*\d+\s*$",  # Just numbers
    r"^\s*.{0,2}\s*$",  # Too short (0-2 chars)
]

# --- Patterns that BYPASS LLM check (definitely SQL-related) ---
_DEFINITE_SQL_PATTERNS = [
    # Explicit data entity mentions
    r"\b(customer|order|product|employee|payment|office|orderdetail)\s*(s|es)?\b",
    r"\b(pelanggan|pesanan|produk|karyawan|pembayaran|kantor)\b",
    # Explicit query verbs
    r"\b(show|list|display|get|find|retrieve|fetch|count|total|sum|average)\b",
    r"\b(tampilkan|tunjukkan|lihat|ambil|cari|hitung|berapa)\b",
    # SQL keywords
    r"\b(select|from|where|join|group by|order by)\b",
    # Superlatives (ranking intent)
    r"\b(highest|lowest|top|most|least|largest|smallest|latest|earliest)\b",
    r"\b(tertinggi|terendah|terbesar|terkecil|terbaru|terlama)\b",
]

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
            return True, "This doesn't look like a data query. Please ask about your data (e.g., 'Show all customers', 'How many orders in 2023?')."
    
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
    if not text or len(text) < 3:
        return False, ""  # Too short to judge, let other checks handle
    
    t = text.lower().strip()
    
    # IMPORTANT: If text has spaces (likely a sentence), be more lenient
    # Real sentences like "Find products whose names start with..." are valid
    has_spaces = ' ' in t
    word_count = len(t.split())
    
    # If it looks like a real sentence (3+ words), skip gibberish check
    if has_spaces and word_count >= 3:
        return False, ""
    
    alpha_only = re.sub(r'[^a-z]', '', t)
    
    if len(alpha_only) < 3:
        return False, ""  # Not enough letters to analyze
    
    # --- Heuristic 1: Vowel ratio ---
    vowels = set('aiueo')
    vowel_count = sum(1 for c in alpha_only if c in vowels)
    vowel_ratio = vowel_count / len(alpha_only)
    
    # Normal English/Indonesian text has ~35-50% vowels
    # "asdgasdgasd" has ratio ~0.18 (too low)
    # Only apply to single-word inputs (no spaces)
    if vowel_ratio < 0.12 and len(alpha_only) > 5 and not has_spaces:
        return True, "Input appears to be random text. Please enter a valid data query."
    
    # --- Heuristic 2: Consecutive consonants ---
    # Raised threshold to 7 to avoid false positives like "strengths", "start with"
    # "qwrtypsdfgh" still caught (10 consonants)
    # Only apply to single-word inputs
    if not has_spaces:
        consonant_run = re.search(r'[bcdfghjklmnpqrstvwxyz]{7,}', alpha_only)
        if consonant_run and len(alpha_only) > 6:
            return True, "Input appears to be random text. Please enter a valid data query."
    
    # --- Heuristic 3: No spaces in long text ---
    # Normal questions have spaces: "show all customers"
    # Gibberish often doesn't: "asdgasdgasdgasd"
    if len(t) > 15 and ' ' not in t:
        # Exception: could be a single long word like "customerNumber"
        # Check if it looks like camelCase or has some structure
        if not re.search(r'[A-Z]', text) and not re.search(r'[_\-]', text):
            return True, "Input appears to be random text. Please enter a valid data query."
    
    # --- Heuristic 4: Repeated patterns ---
    # "abcabcabc" or "xyxyxyxy"
    # Only for short inputs without spaces
    if len(t) >= 6 and not has_spaces:
        # Check for 2-3 char patterns repeated 3+ times
        for pattern_len in [2, 3]:
            pattern = t[:pattern_len]
            if pattern * 3 in t and len(set(pattern)) < pattern_len:
                return True, "Input appears to be random text. Please enter a valid data query."
    
    # --- Heuristic 5: Keyboard mash detection ---
    # Common keyboard patterns: qwerty, asdf, zxcv
    # Only for inputs without spaces (single gibberish words)
    if not has_spaces:
        keyboard_patterns = ['qwert', 'asdfg', 'zxcvb', 'qazwsx', 'yxcvbn', 'hjkl', 'yuiop']
        for kp in keyboard_patterns:
            if kp in alpha_only:
                return True, "Input appears to be random text. Please enter a valid data query."
    
    return False, ""

def _check_definite_sql(question: str) -> bool:
    """
    Check if question definitely has SQL intent (bypass LLM check).
    """
    q_lower = question.lower()
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
        temperature=0.0,
        format="json",
        keep_alive=OLLAMA_KEEP_ALIVE,
        num_ctx=512,  # Small context for fast inference
        num_predict=64,  # Short response
    )
    
    prompt = f"""You are a query intent classifier for a SQL database system.
Determine if the user's question is asking about DATA or DATABASE QUERIES.

VALID questions (answer "yes"):
- Questions about customers, orders, products, employees, payments, offices
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
            return False, f"This doesn't appear to be a data query. {reason}\n\nPlease ask about your data, for example:\n  - 'Show all customers from USA'\n  - 'How many orders in 2023?'\n  - 'List products with price above 100'"
        
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
    line_sep = "─" * 150
    print(f"\n{line_sep}")
    print(f"{'QUESTION GUARD':^150}")
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
                "complexity": "rejected",
                "qtype": "invalid", 
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
    
    # [FIX] Definisi Pattern yang HARUS DIBUANG sebelum deteksi tabel
    ignore_patterns = {
        'order': [
            r"\border(ed|s)?\s+by\b",
            r"\b(descending|ascending)\s+order\b",
            r"\bin\s+order\b"
        ],
        'group': [r"\bgroup\s+by\b"],
        'sort':  [r"\bsort(ed)?\s+by\b"],
        'rank':  [r"\brank\s+over\b"]
    }
    
    relevant_tables = []
    for tbl in whitelisted_tables:
        if f"- {tbl}:" not in ctx and f"- `{tbl}`:" not in ctx:
            continue

        table_word = tbl.lower().rstrip('s')

        # Buat copy pertanyaan khusus untuk scanning tabel ini
        search_scope = question

        if table_word in ignore_patterns:
            patterns = ignore_patterns[table_word]
            for pat in patterns:
                search_scope = re.sub(pat, " ", search_scope)

        # Cek 2: Cari kata tabel di search_scope yang SUDAH BERSIH
        if re.search(rf"\b{table_word}s?\b", search_scope):

            # Deteksi referensi ID saja (FK)
            is_just_id_ref = re.search(rf"\b{table_word}s?\s+(number|id|code|kode|nomor)\b", search_scope)

            if is_just_id_ref:
                if re.search(rf"\b{table_word}s?\b(?!\s+(number|id|code|kode|nomor))", search_scope):
                    relevant_tables.append(tbl)
                    if RETRIEVE_DEBUG:
                        print(f"[Router] ✓ Added {tbl} (with ID ref check)")
            else:
                relevant_tables.append(tbl)
                if RETRIEVE_DEBUG:
                    print(f"[Router] ✓ Added {tbl} (direct match)")
        elif RETRIEVE_DEBUG:
            pattern_str = rf"\b{table_word}s?\b"
            print(f"[Router] ✗ Skipped {tbl} (no match in question)")
            print(f"[Router]   Pattern: {pattern_str}")
            print(f"[Router]   In question? {bool(re.search(pattern_str, search_scope))}")
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 2: FK-ONLY FILTER (Hapus tabel yang hanya disebutkan untuk FK)
    # ═══════════════════════════════════════════════════════════════════
    filtered_tables = []
    for tbl in relevant_tables:
        table_word = tbl.lower().rstrip('s')
        
        # Pattern: "<table> with <code/id> <angka/list>"
        fk_only_pattern = rf"\b{table_word}s?\s+with\s+(code|number|id|kode|nomor)s?\s+[\d,\sor]+"
        
        if re.search(fk_only_pattern, question):
            temp_scope = re.sub(fk_only_pattern, "", question)
            
            has_other_mention = (
                re.search(rf"\b{table_word}s\b", temp_scope) or
                re.search(rf"\b{table_word}\s+(?!(code|number|id|kode|nomor))", temp_scope)
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
    math_keywords = ['sum', 'count', 'total', 'avg', 'average', 'number of', 'jumlah', 'rata-rata', 
                     'percentage', 'percent', 'ratio', 'proportion', 'persentase', 'rasio']
    has_math = any(re.search(rf"\b{re.escape(kw)}\b", question_lower) for kw in math_keywords)
    
    group_indicators = ["each", "per", "every", "masing-masing", "setiap", "for each", "classify"]
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
    fk_pattern_card = re.compile(r'table:\s*(\w+)\s*\|\s*column:\s*(\w+).*?FK to (\w+)\.(\w+)', re.IGNORECASE)
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
    fk_pattern_dot = re.compile(r'(\w+)\.(\w+).*?FK to (\w+)\.(\w+)', re.IGNORECASE)
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
        table_singular = tbl_lower.rstrip('s')
        for col in tbl_cols:
            col_lower = col.lower()
            col_variants = [col_lower]
            # Add space-separated variant for camelCase
            spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', col).lower()
            if spaced != col_lower:
                col_variants.append(spaced)

            for variant in col_variants:
                if variant in question_lower:
                    # Check if this is an FK column (e.g., customerNumber)
                    fk_pattern = rf'^{re.escape(table_singular)}(number|code|id|id)$'
                    is_fk_col = bool(re.match(fk_pattern, col_lower, re.IGNORECASE))

                    if is_fk_col:
                        # Check if this FK column exists in ANOTHER table
                        fk_exists_elsewhere = False
                        for other_tbl in relevant_tables:
                            if other_tbl.lower() == tbl_lower:
                                continue
                            other_cols = allowed_identifiers.get(other_tbl, [])
                            for other_col in other_cols:
                                if other_col.lower() == col_lower:
                                    fk_exists_elsewhere = True
                                    break
                            if fk_exists_elsewhere:
                                break

                        if fk_exists_elsewhere:
                            # FK column that exists elsewhere → weak evidence
                            evidence.append(f"FK column '{col}'")
                        else:
                            # FK column that ONLY exists here (parent table) → strong evidence
                            evidence.append(f"column '{col}'")
                    else:
                        # Non-FK column → strong evidence
                        evidence.append(f"column '{col}'")
                    break

        # Check 2: "For each X" patterns
        # Handle both plural and singular, with optional words in between (e.g., "for each such customer")
        table_singular = tbl_lower.rstrip('s')
        for_each_patterns = [
            rf"\bfor\s+each\s+(?:\w+\s+)?{tbl_lower}\b",      # "for each customers", "for each such customers"
            rf"\bfor\s+each\s+(?:\w+\s+)?{table_singular}\b",  # "for each customer", "for each such customer"
            rf"\bper\s+(?:\w+\s+)?{tbl_lower}\b",             # "per customers", "per such customers"
            rf"\bper\s+(?:\w+\s+)?{table_singular}\b",         # "per customer", "per such customer"
            rf"\beach\s+(?:\w+\s+)?{tbl_lower}\b",            # "each customers", "each such customers"
            rf"\beach\s+(?:\w+\s+)?{table_singular}\b",        # "each customer", "each such customer"
        ]
        if any(re.search(p, question_lower) for p in for_each_patterns):
            evidence.append("'for each' pattern")

        # Check 3: Table name + attribute patterns
        table_singular = tbl_lower.rstrip('s')
        table_patterns = [
            rf"\b{table_singular}\s+(name|number|id|code)\b",
            rf"\b{tbl_lower}\s+(name|number|id|code)\b",
        ]
        if any(re.search(p, question_lower) for p in table_patterns):
            evidence.append("table+attribute pattern")

        # Check 4: Aggregate keywords with table mention
        if has_math or has_group_word:
            # Patterns like "total orders", "count of customers", "number of orders", etc.
            # More flexible: allows extra words after table name (e.g., "orders placed")
            aggregate_patterns = [
                # "count of orders", "total of orders", "number of orders"
                rf"\b(total|sum|count|average|avg|number)\s+(of\s+)?{tbl_lower}\b",
                # "count of orders placed", "number of orders placed" (with extra words)
                rf"\b(total|sum|count|average|avg|number)\s+(of\s+)?{tbl_lower}\s+\w+",
                # "orders total", "customers count"
                rf"\b{tbl_lower}\s+(total|sum|count|average|number)\b",
            ]
            if any(re.search(p, question_lower) for p in aggregate_patterns):
                evidence.append("aggregate context")

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

        # Check if evidence is ONLY weak patterns
        # Strong evidence: non-FK column mentions, aggregate context
        # Weak evidence: 'for each' pattern, table+attribute pattern, FK column mentions
        weak_evidence = {"'for each' pattern", "table+attribute pattern"}
        # Also check for FK column evidence (format: "FK column 'X'")
        is_only_weak = all(e in weak_evidence or e.startswith("FK column") for e in evidence_list)

        if is_only_weak and evidence_list:  # Has evidence, but all are weak
            table_singular = tbl_lower.rstrip('s')

            # Check if any other table has FK column linking to this table
            fk_exists_in_other_table = False
            for other_tbl in relevant_tables:
                if other_tbl.lower() == tbl_lower:
                    continue
                other_cols = allowed_identifiers.get(other_tbl, [])
                for col in other_cols:
                    # Check if this is an FK column (e.g., customerNumber → customers)
                    col_lower = col.lower()
                    # Pattern: <table_singular>(Number|Code|Id|ID)
                    fk_pattern = rf'^{re.escape(table_singular)}(number|code|id|id)$'
                    if re.match(fk_pattern, col_lower, re.IGNORECASE):
                        fk_exists_in_other_table = True
                        break
                if fk_exists_in_other_table:
                    break

            if fk_exists_in_other_table:
                # Check if question mentions any non-FK columns from this table
                tbl_cols = allowed_identifiers.get(tbl_lower, [])
                # Get FK columns
                fk_cols = set()
                for col in tbl_cols:
                    col_lower = col.lower()
                    fk_pattern = rf'^{re.escape(table_singular)}(number|code|id|id)$'
                    if re.match(fk_pattern, col_lower, re.IGNORECASE):
                        fk_cols.add(col_lower)

                # Check if any non-FK column is mentioned in question
                non_fk_mentioned = False
                for col in tbl_cols:
                    if col.lower() in fk_cols:
                        continue
                    # Check if column is mentioned (including spaced variant)
                    spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', col).lower()
                    if col.lower() in question_lower or spaced in question_lower:
                        non_fk_mentioned = True
                        break

                if not non_fk_mentioned:
                    # Safe to remove - only referenced via "for each" and FK exists elsewhere
                    tables_to_remove.append(tbl_lower)
                    if RETRIEVE_DEBUG:
                        print(f"[Router] Skipping {tbl_lower}: only 'for each' ref, FK exists in other table")

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
            non_fk_cols = [c for c in allowed_identifiers.get(tbl, [])
                          if c.lower() not in fk_columns_by_table.get(tbl_lower, {})]
            needs_non_fk = False
            for col in non_fk_cols:
                col_variants = [col.lower()]
                spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', col).lower()
                if spaced != col.lower():
                    col_variants.append(spaced)
                if any(v in question_lower for v in col_variants):
                    needs_non_fk = True
                    break

            if needs_non_fk:
                if RETRIEVE_DEBUG:
                    print(f"[Router] Keeping {tbl}: has FK to evidence table + needs non-FK columns")
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

    # ═══════════════════════════════════════════════════════════════════
    # STEP 4: DETEKSI SINYAL (Complete Detection)
    # ═══════════════════════════════════════════════════════════════════
    # Note: "rank/ranking" NOT included here - explicit ranking with RANK() OVER 
    # is simpler than winner-row patterns (highest/lowest/top) which need subqueries
    value_ranks = ["highest", "lowest", "top", "bottom", "most", "least", "largest", "smallest", 
                   "terbesar", "terkecil", "paling", "tertinggi", "terendah"]
    time_ranks = ["latest", "earliest", "newest", "oldest", "terbaru", "terlama"]
    is_ranking_value = any(re.search(rf"\b{re.escape(kw)}\b", question_lower) for kw in value_ranks)
    is_ranking_time = any(re.search(rf"\b{re.escape(kw)}\b", question_lower) for kw in time_ranks)
    is_ranking = is_ranking_value or is_ranking_time
    
    # EXPLICIT RANK FUNCTION DETECTION: "a rank of X" / "rank employees" / "ranking"
    # This means user explicitly wants RANK() OVER window function
    # E.g., "return a rank of employees within each job title"
    explicit_rank_patterns = [
        r"\ba\s+rank\s+of\b",           # "a rank of employees"
        r"\brand\s+employees\b",         # "rank employees"
        r"\branking\s+of\b",             # "ranking of X"
        r"\breturn.*\brank\b.*\bwithin\b",  # "return X and rank within"
        r"\brank\s+.*\bbased\s+on\b",    # "rank X based on Y"
    ]
    is_explicit_rank_function = any(re.search(pat, question_lower) for pat in explicit_rank_patterns)
    
    # ORDINAL RANKING DETECTION: "second highest", "third largest", "2nd lowest", etc.
    # These patterns ALWAYS need ROW_NUMBER() with rn = N, much more complex than simple MAX/MIN
    ordinal_words = ["second", "third", "fourth", "fifth", "2nd", "3rd", "4th", "5th",
                     "kedua", "ketiga", "keempat", "kelima"]
    ordinal_pattern = rf"\b({'|'.join(ordinal_words)})\s+(highest|lowest|largest|smallest|biggest|top|bottom|best|worst|most|least|terbesar|terkecil|tertinggi|terendah)\b"
    is_ordinal_ranking = bool(re.search(ordinal_pattern, question_lower))
    
    # WINDOW AGGREGATE DETECTION: "within the same X" / "in the same X" pattern
    # This pattern requires COUNT(*) OVER (PARTITION BY X), NOT GROUP BY
    # E.g., "total employees within the same office" → PARTITION BY officeCode
    within_same_pattern = r"\b(within|in)\s+the\s+same\s+(\w+)"
    within_same_match = re.search(within_same_pattern, question_lower)
    is_window_aggregate = bool(within_same_match)
    partition_hint = within_same_match.group(2) if within_same_match else None
    
    explicit_sort = ["sort by", "ordered by", "sorted by", "order by", "sorted from"]
    is_sorting = any(re.search(rf"\b{re.escape(kw)}\b", question_lower) for kw in explicit_sort) or is_ranking
    
    is_by_logic = ("by " in question_lower and not any(re.search(rf"\b{re.escape(s)}\b", question_lower) for s in explicit_sort) and (has_math or is_ranking))
    has_grouping = has_group_word or is_by_logic

    # ═══════════════════════════════════════════════════════════════════
    # STEP 4.5: PRIMARY TABLE DETERMINATION
    # ═══════════════════════════════════════════════════════════════════
    # Determine which table should be the FROM table (primary table)
    # This provides guidance to LLM for SQL generation
    #
    # Priority rules:
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

        # Rule 1: "For each X" pattern (highest priority)
        # Handle both plural and singular, with optional words in between
        for tbl in relevant_tables:
            tbl_lower = tbl.lower()
            table_singular = tbl_lower.rstrip('s')
            for_each_patterns = [
                rf"\bfor\s+each\s+(?:\w+\s+)?{tbl_lower}\b",
                rf"\bfor\s+each\s+(?:\w+\s+)?{table_singular}\b",
                rf"\bper\s+(?:\w+\s+)?{tbl_lower}\b",
                rf"\bper\s+(?:\w+\s+)?{table_singular}\b",
                rf"\beach\s+(?:\w+\s+)?{tbl_lower}\b",
                rf"\beach\s+(?:\w+\s+)?{table_singular}\b",
            ]
            if any(re.search(p, question_lower) for p in for_each_patterns):
                primary_table = tbl
                primary_reason = "'for each' pattern"
                break

        # Rule 2: Not yet found, check "show X from Y" / "X in Y" patterns
        if not primary_table:
            for tbl in relevant_tables:
                tbl_lower = tbl.lower()
                # Patterns: "customers who placed", "offices in", "products from"
                from_in_patterns = [
                    rf"\b\w+\s+(from|in)\s+{tbl_lower}\b",
                    rf"\b{tbl_lower}\s+who\s+",
                    rf"\b{tbl_lower}\s+that\s+",
                ]
                if any(re.search(p, question_lower) for p in from_in_patterns):
                    primary_table = tbl
                    primary_reason = "from/in pattern"
                    break

        # Rule 3: Not yet found, check "all X that Y" pattern
        if not primary_table:
            for tbl in relevant_tables:
                tbl_lower = tbl.lower()
                if re.search(rf"\ball\s+{tbl_lower}\s+", question_lower):
                    primary_table = tbl
                    primary_reason = "'all X' pattern"
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
        if re.search(r"\b(record|row|entry|item)\s+with\s+(the\s+)?(largest|highest|lowest|smallest|biggest)", question_lower):
            is_winner_row = True
        # "the largest X for each Y" / "largest X per Y"
        if re.search(r"\b(largest|highest|lowest|smallest|biggest)\s+\w+\s+(for\s+each|per)\b", question_lower):
            is_winner_row = True
    
    # Pattern C: Tiebreaker pattern (always complex, needs ROW_NUMBER with multiple ORDER BY)
    # "if multiple X have the same Y, choose the one with..."
    # "if there are ties, pick..."
    has_tiebreaker = bool(re.search(
        r"\bif\s+(multiple|several|there\s+are\s+ties?|same)\b.*\b(choose|pick|select|take)\s+(the\s+)?(one|latest|earliest|first|last)\b",
        question_lower
    ))
    if has_tiebreaker:
        is_winner_row = True
    
    # FIXED: Exclusion detection - only trigger for TRUE exclusion patterns
    # These patterns imply NOT EXISTS / NOT IN / LEFT JOIN ... IS NULL (cross-table exclusion)
    # NOT triggered by: "missing" in single-table context (just IS NULL check)
    has_exclusion = False
    
    # Pattern 1: Cross-table exclusion keywords (always need 7B)
    cross_table_exclusion = ["never ordered", "never paid", "without any", "have no", "has no", 
                             "tidak pernah", "belum pernah", "tidak ada", "yang tidak punya"]
    if any(kw in question_lower for kw in cross_table_exclusion):
        has_exclusion = True
    
    # Pattern 2: "no" + entity noun (no orders, no payments) → implies cross-table
    if not has_exclusion and re.search(r"\bno\s+(orders?|payments?|customers?|products?|employees?)\b", question_lower):
        has_exclusion = True
    
    # Pattern 3: "missing" or "empty" ONLY if multi-table context
    # Single-table "missing" = just IS NULL (simple, 3B can handle)
    if not has_exclusion and is_multi_table:
        if re.search(r"\b(missing|empty)\b", question_lower):
            has_exclusion = True
    
    # ESCALATION LOGIC
    # Escalate to 7B when complexity truly requires it:
    # - Multi-table query (requires JOIN)
    # - Winner-row patterns (requires ROW_NUMBER/RANK subquery)
    # - Cross-table exclusion patterns (NOT IN, NOT EXISTS)
    should_escalate = (
        is_multi_table or   # JOINs need 7B
        is_winner_row or    # ROW_NUMBER subquery patterns need 7B
        has_exclusion       # Cross-table exclusion needs 7B
    )
    
    if is_multi_table:
        print(f"[Router] Multi-table detected: {relevant_tables} -> escalating to 7B")
    if is_winner_row and not is_multi_table:
        print(f"[Router] Winner-row pattern detected (tiebreaker/subquery needed) -> escalating to 7B")
    
    chosen_model = LARGE_MODEL if should_escalate else SMALL_MODEL
    complexity = "complex" if should_escalate else "simple"
    
    # Tentukan QTYPE
    if is_winner_row: 
        qtype = "rank"
    elif has_math or has_grouping: 
        qtype = "aggregate"
    else: 
        qtype = "lookup"
    
    print(f"[Router] Detected Tables: {relevant_tables}")
    print(f"[Router] Signals -> Math:{has_math}, Sort:{is_sorting}, Rank:{is_ranking}, Group:{has_grouping}, WinnerRow:{is_winner_row}, OrdinalRank:{is_ordinal_ranking}, Exclusion:{has_exclusion}, WindowAgg:{is_window_aggregate}, ExplicitRank:{is_explicit_rank_function}")
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
            "is_explicit_rank_function": is_explicit_rank_function
        }
    }
# ======================================================================
# ======================= AGENT 2 - SQL DRAFTER ========================
# ======================================================================
import time as _time  # For inference timing
# Track last warmup time per model
_last_model_warmup = {}
# OOM retry settings
OOM_WAIT_TIME = 35  # Wait 35 seconds for model to unload (keep_alive=30s + buffer)
OOM_MAX_RETRIES = 3  # Max retry attempts on OOM

def _prewarm_model_with_oom_retry(llm, target_model: str, max_retries: int = OOM_MAX_RETRIES) -> bool:
        global _last_model_warmup
        # Get keep_alive in seconds
        import re
        from .config import OLLAMA_KEEP_ALIVE
        m = re.match(r"(\d+)([smh])", str(OLLAMA_KEEP_ALIVE).lower())
        if m:
            val, unit = int(m.group(1)), m.group(2)
            if unit == "s":
                keep_alive_sec = val
            elif unit == "m":
                keep_alive_sec = val * 60
            elif unit == "h":
                keep_alive_sec = val * 3600
            else:
                keep_alive_sec = 60
        else:
            keep_alive_sec = 60

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
                if "memory" in error_msg:
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
    if decision.get("complexity") == "rejected" or decision.get("is_valid_question") == False:
        print("[DRAFT] Skipped: Question was rejected by router guard")
        return state  # Pass through unchanged
    
    target_model = state.get("target_model") or LARGE_MODEL
    llm = build_llm_json(model=target_model)

    # ===== PRE-WARM WITH OOM RETRY =====
    # This separates cold-start time from actual inference time
    # Also handles OOM by waiting for previous model to unload
    model_ready = _prewarm_model_with_oom_retry(llm, target_model)

    if not model_ready:
        return {**state, "draft_sql": "SELECT 'Error: Model OOM after retries';"}

    # START TIMER: Model ready → Final SQL complete (for thesis latency benchmark)
    # This measures the core text-to-SQL processing time, excluding:
    # - Model loading/warmup (one-time cost)
    # - Retrieval (vector store, depends on index size)
    # - SQL execution (depends on database performance)
    state["processing_start_time"] = _time.perf_counter()
    print(f"[DEBUG] processing_start_time set")

    question = state.get("question", "")
    ctx = state.get("context", "")
    
    # Panggil Dynamic Prompt Builder (helpers_main.py)
    # Segala aturan heuristik (Anti-MAX, Anti-1055, Anti-Inner Join) sudah diurus di sana.
    # Pass state for dynamic join context from retrieval results
    prompt = make_generation_prompt(ctx, question, decision, state)
    
    # ===== TIMED INFERENCE (model already loaded) =====
    _t0 = _time.perf_counter()
    obj = try_llm_json(llm, prompt, max_attempts=2, oom_retry_wait=OOM_WAIT_TIME)
    _inference_time = _time.perf_counter() - _t0

    # Store inference time in state for debugging (not logged to reduce noise)
    state["draft_inference_time"] = _inference_time

    if obj is None:
        return {**state, "draft_sql": "SELECT 'Error: LLM failed';"}
        
    draft_sql = (obj.get("sql") or "SELECT 1;").strip()
    reasoning = (obj.get("notes") or "No reasoning provided.").strip()
    print(f"\n[DRAFT] Raw from LLM: {draft_sql}")
    print(f"[DRAFT] LLM Reasoning: {reasoning}")
    draft_sql = _remove_top_level_order_by_if_not_requested(draft_sql, question)
    # 2. [NEW] Hapus syntax error: SELECT col DESC FROM ...
    # Regex ini mencari DESC/ASC sebelum FROM (hanya di SELECT clause, bukan di ORDER BY)
    # Hanya strip DESC/ASC yang langsung sebelum FROM (dalam SELECT list)
    draft_sql = re.sub(r"\s+(DESC|ASC)\s*(?=FROM)", " ", draft_sql, flags=re.IGNORECASE)
    # FIX: Jangan strip DESC/ASC sebelum koma jika ada di dalam ORDER BY atau OVER clause
    # Pattern lama: r"\s+(DESC|ASC)\s*(?=,)" - ini salah strip "ORDER BY amount DESC, date"
    # Pattern baru: hanya strip jika di SELECT clause (sebelum FROM)
    # Gunakan negative lookbehind untuk memastikan bukan dalam ORDER BY context
    def _strip_desc_in_select_only(sql: str) -> str:
        """Strip DESC/ASC before comma only in SELECT clause, not in ORDER BY."""
        # Find SELECT ... FROM boundary
        match = re.search(r"\bSELECT\b(.*?)\bFROM\b", sql, re.IGNORECASE | re.DOTALL)
        if not match:
            return sql
        select_part = match.group(1)
        # Strip DESC/ASC before comma only in SELECT part
        fixed_select = re.sub(r"\s+(DESC|ASC)\s*(?=,)", " ", select_part, flags=re.IGNORECASE)
        if fixed_select != select_part:
            sql = sql[:match.start(1)] + fixed_select + sql[match.end(1):]
        return sql
    draft_sql = _strip_desc_in_select_only(draft_sql)
    # Basic Cleanup (Limit Halusinasi)
    draft_sql = re.sub(r"LIMIT\s+(NULL|0)", "", draft_sql, flags=re.IGNORECASE).strip()
    match = re.search(r"LIMIT\s+(\d+)", draft_sql, re.IGNORECASE)
    if match:
        limit_n = match.group(1)
        if not re.search(rf"\b{limit_n}\b|top|best|highest|lowest|limit", question.lower()):
            draft_sql = re.sub(r"LIMIT\s+\d+\s*;?", "", draft_sql, flags=re.IGNORECASE).strip()
    
    if not draft_sql.endswith(";"): draft_sql += ";"
    print(f"[DRAFT] After cleanup: {draft_sql}")
    return {**state, "draft_sql": draft_sql, "agent_notes": (obj.get("notes") or "").strip()}
# ======================================================================
# ======================= AGENT 3 - SQL REFINER ========================
# ======================================================================
def refine_sql_agent(state: RAGState) -> RAGState:
    # --- Guard: Skip if question was rejected by router ---
    decision = state.get("router_decision", {})
    if decision.get("complexity") == "rejected" or decision.get("is_valid_question") == False:
        print("[REFINE] Skipped: Question was rejected by router guard")
        return state  # Pass through unchanged
    
    # DYNAMIC MODEL SELECTION berdasarkan draft quality
    draft_model = state.get("target_model") or SMALL_MODEL  # Track model asal
    target_model = LARGE_MODEL  # Default untuk regeneration
    
    llm = build_llm_repair(model=target_model)
    question = state.get("question", "")
    ctx = state.get("context", "")
    draft_sql = (state.get("draft_sql") or "").strip()
    
    decision = state.get("router_decision", {})
    detected_tables = state.get("router_tables") or []
    complexity = decision.get("complexity", "complex")
    qtype = decision.get("qtype", "lookup")
    
    # 1. HEURISTIC PIPELINE (Versi Bersih/Safe Mode)
    final_sql = normalize_sql_pipeline(draft_sql, question, ctx)
    sql_up = final_sql.upper()
    corrections = ["Applied Heuristic Pipeline"]

    # 2. PREPARE CHECKS (FULL PROTECTION)
    sql_for_tokenizing = final_sql
    sql_for_tokenizing = re.sub(r"'(?:''|[^'])*'", " ", sql_for_tokenizing)
    sql_for_tokenizing = re.sub(r'"(?:""|[^"])*"', " ", sql_for_tokenizing)
    # FIX 1: Remove function calls (word followed by parenthesis)
    sql_for_tokenizing = re.sub(
        r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(",
        "(",
        sql_for_tokenizing
    )

    # FIX 2: Remove table names from tokenizing
    # This prevents "payments" in "FROM payments" or subquery being treated as column
    allowed_ids = _parse_allowed_identifiers(ctx)
    for tbl in allowed_ids.keys():
        sql_for_tokenizing = re.sub(
            rf"\b{re.escape(tbl)}\b",
            "_TBL_",
            sql_for_tokenizing,
            flags=re.IGNORECASE
        )

    sql_tokens = set(re.findall(r"\b([a-z][a-z0-9_]*)\b", sql_for_tokenizing.lower()))

    # Remove placeholder tokens
    sql_tokens.discard("_tbl_")    
    reserved = _SQL_KEYWORDS.union({
        'year', 'month', 'day', 'date', 'date_format', 'now', 'curdate',
        'concat', 'group_concat', 'coalesce', 'ifnull', 'sum', 'count', 'avg',
        't', 'limit', 'offset', 'as', 'regexp_like', 'str_to_date',
        'case', 'when', 'then', 'else', 'end',
        'union', 'all', 'distinct', 'values', 'exists', 'having',
        'substr',
    })
    
    defined_aliases = set()
    defined_aliases.update(re.findall(r"\bAS\s+([a-zA-Z0-9_]+)\b", sql_up, re.IGNORECASE))
    quoted_aliases = re.findall(r"\bAS\s+['\"`]?([a-zA-Z0-9_]+)['\"`]?", sql_up, re.IGNORECASE)
    defined_aliases.update(quoted_aliases)
    
    # Also capture subquery aliases: FROM (...) alias or FROM (...) AS alias
    subquery_aliases = re.findall(r'\)\s*(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\b', sql_up, re.IGNORECASE)
    for sa in subquery_aliases:
        if sa.lower() not in ('where', 'group', 'order', 'having', 'limit', 'on', 'and', 'or'):
            defined_aliases.add(sa)

    # Also capture table aliases: FROM orders o, JOIN customers c
    # Pattern: FROM table alias or JOIN table alias (with or without AS)
    table_alias_pattern = r'\b(?:FROM|JOIN)\s+[a-zA-Z_][a-zA-Z0-9_]*\s+(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\b'
    for m in re.finditer(table_alias_pattern, sql_up, re.IGNORECASE):
        alias = m.group(1).lower()
        # Skip SQL keywords that might be captured
        if alias not in ('where', 'group', 'order', 'having', 'limit', 'on', 'and', 'or', 'select'):
            defined_aliases.add(alias)

    reserved.update(a.lower() for a in defined_aliases)

    # Fix: For subqueries, FROM (SELECT...) matches "SELECT" - skip that
    # Look for real table name or subquery alias
    primary_match = re.search(r"FROM\s+([A-Za-z0-9_]+)", sql_up)
    primary_tbl = ""
    if primary_match:
        candidate = primary_match.group(1).lower()
        # Skip if it's a keyword (subquery case: FROM (SELECT...))
        if candidate not in ('select', 'with'):
            primary_tbl = candidate
    
    if primary_tbl: reserved.add(primary_tbl)

    primary_cols = {c.lower() for c in allowed_ids.get(primary_tbl, [])}
    alien_cols = [t for t in sql_tokens if t not in reserved and t not in primary_cols and not t.isdigit()]
    
    # --- FLATTEN VALID COLS FOR KASUS J ---
    all_valid_cols = set()
    for cols in allowed_ids.values():
        all_valid_cols.update(cols)

    # 3. DECISION MATRIX
    needs_regeneration = False
    regeneration_model = LARGE_MODEL  # Default 7B untuk regeneration
    extra_instruction = ""

    # --- KASUS A: Schema Validation & Hallucination Check (HARD ERROR) ---
    if alien_cols and "JOIN" not in sql_up:
        # SMART SUBQUERY HANDLING
        # Only skip validation if alien columns are INSIDE subquery (not in main SELECT)
        has_subquery = re.search(r"\(\s*SELECT\b", sql_up, re.IGNORECASE)
        
        if has_subquery:
            # Extract main SELECT clause (before first subquery)
            # Pattern: SELECT ... FROM ... WHERE ... (SELECT ...)
            main_select_match = re.search(r"^SELECT\s+(.*?)\s+FROM", sql_up, re.IGNORECASE | re.DOTALL)
            
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
                        tbl_cols = {c.lower() for c in allowed_ids.get(tbl, [])}
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
                tbl_cols = {c.lower() for c in allowed_ids.get(tbl, [])}
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
        if re.search(r"SELECT\s+.*?\brn\b.*?FROM", sql_up):
             _dbg_print_two("[Agent3] HARD ERROR - Winner-Row Leak", "rn column found in outer select.")
             extra_instruction = (
                "STRICT REQUIREMENT: Use standard Winner-Row pattern. "
                "Do NOT include the 'rn' (row number) column in the final output."
             )
             needs_regeneration = True
             regeneration_model = LARGE_MODEL  # Force 7B

    # --- KASUS C: ANTI-PATTERN (WHERE IN Subquery) (HARD ERROR) ---
    tuple_in_pattern = re.search(r"WHERE\s*\([^)]+,[^)]+\)\s*IN\s*\(SELECT", sql_up)
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
        part_match = re.search(r"PARTITION\s+BY\s+([A-Za-z0-9_]+)", sql_up)
        if part_match:
            part_col = part_match.group(1)
            select_clause_match = re.search(r"^SELECT\s+(.*?)\s+FROM", sql_up, re.DOTALL)
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
    has_groupby = "GROUP BY" in sql_up
    is_left_join = "LEFT JOIN" in sql_up
    if has_groupby and not is_left_join and not needs_regeneration:
        where_match = re.search(r"WHERE\s+(.*?)($|GROUP|ORDER|LIMIT)", sql_up, re.DOTALL)
        group_match = re.search(r"GROUP\s+BY\s+(.*?)($|ORDER|LIMIT|HAVING)", sql_up, re.DOTALL)
        if where_match and group_match:
            where_clause = where_match.group(1).lower()
            group_clause = group_match.group(1).lower()
            where_refs = set(re.findall(r"\b([a-z0-9_]+)\.", where_clause))
            group_refs = set(re.findall(r"\b([a-z0-9_]+)\.", group_clause))
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
        select_match_chk = re.search(r"^SELECT\s+(.*?)\s+FROM", final_sql, re.IGNORECASE | re.DOTALL)
        select_text_chk = select_match_chk.group(1).lower() if select_match_chk else ""
        # Tokenize hanya bagian SELECT
        select_tokens_chk = set(re.findall(r"\b([a-z][a-z0-9_]*)\b", select_text_chk))
        
        # STEP 1b: Ambil tokens dari WHERE/HAVING/subquery (kolom yang dipakai sebagai filter)
        # Kolom yang sudah dipakai di WHERE tidak perlu ada di SELECT (bukan "missing")
        sql_lower = final_sql.lower()
        where_having_tokens = set()
        # Extract WHERE clause
        where_match = re.search(r"\bwhere\s+(.*?)(?:\bgroup\b|\border\b|\blimit\b|\bhaving\b|;|$)", sql_lower, re.DOTALL)
        if where_match:
            where_having_tokens.update(re.findall(r"\b([a-z][a-z0-9_]*)\b", where_match.group(1)))
        # Extract HAVING clause
        having_match = re.search(r"\bhaving\s+(.*?)(?:\border\b|\blimit\b|;|$)", sql_lower, re.DOTALL)
        if having_match:
            where_having_tokens.update(re.findall(r"\b([a-z][a-z0-9_]*)\b", having_match.group(1)))

        suffix_synonyms = {
            "number": r"(number|no|id|code|kode|nomor)",
            "code":   r"(code|id|no|number|kode|nomor)",
            "id":     r"(id|no|code|number|kode|nomor)",
            "date":   r"(date|tanggal|tgl|waktu)",
            "name":   r"(name|nama)",
            "amount": r"(amount|total|jumlah|nilai|harga)",
            "extension": r"(extension|ext)"
        }
        
        # all_valid_cols sudah disiapkan di atas
        for col_raw in all_valid_cols:
            col_lower = col_raw.lower()
            
            # STEP 2: Logic Check yang Lebih Akurat
            # Skip jika kolom sudah ada di SELECT tokens, atau jika ada wildcard (*)
            if col_lower in select_tokens_chk or "*" in select_text_chk: 
                continue
            
            # Skip jika kolom sudah dipakai di WHERE/HAVING (artinya dipakai sebagai filter, bukan output)
            if col_lower in where_having_tokens:
                continue
            
            clean_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', col_raw)
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
            if complexity == "simple" and qtype == "lookup":
                # SOFT ERROR: Auto-fix tanpa regenerate
                _dbg_print_two("[Agent3] SOFT FIX - Missing Columns", f"Auto-appending {missing_cols}")
                
                # STEP 3: Safe Injection (Menangani DISTINCT)
                is_distinct = re.search(r"^SELECT\s+DISTINCT\s+", final_sql, re.IGNORECASE)
                
                if is_distinct:
                    # Pola: SELECT DISTINCT col1 -> SELECT DISTINCT missing, col1
                    final_sql = re.sub(
                        r"^(SELECT\s+DISTINCT\s+)(.*?)(\s+FROM)", 
                        f"\\1{', '.join(missing_cols)}, \\2\\3", 
                        final_sql, 
                        count=1, 
                        flags=re.IGNORECASE | re.DOTALL
                    )
                else:
                    # Pola: SELECT col1 -> SELECT missing, col1
                    final_sql = re.sub(
                        r"^(SELECT\s+)(.*?)(\s+FROM)", 
                        f"\\1{', '.join(missing_cols)}, \\2\\3", 
                        final_sql, 
                        count=1, 
                        flags=re.IGNORECASE | re.DOTALL
                    )
                
                corrections.append(f"Auto-added missing columns: {', '.join(missing_cols)}")
            
            elif draft_model == SMALL_MODEL and complexity == "simple":
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
        if re.search(r"\b[a-z0-9_]+\.\(", sql_up.lower()):
             _dbg_print_two("[Agent3] HARD ERROR - Syntax Error", "Broken alias syntax `alias.(...)` detected.")
             extra_instruction = (
                "SYNTAX ERROR: You generated invalid SQL syntax like `table_alias.(condition)`. "
                "CORRECT: `(table_alias.column IS NULL ...)`."
             )
             needs_regeneration = True
             regeneration_model = LARGE_MODEL  # Force 7B

    # --- KASUS H: REGEX PATTERN MISMATCH (HARD ERROR) ---
    if not needs_regeneration:
        regex_triggers = [
            r"exactly\s+\w+\s+digits",
            r"followed\s+by\s+\w+\s+digits",
            r"\d+\s+digits",
            r"pattern",
            r"format"
        ]
        requires_strict_pattern = any(re.search(p, question, re.IGNORECASE) for p in regex_triggers)
        uses_simple_like = "LIKE" in sql_up and "REGEXP" not in sql_up and "RLIKE" not in sql_up
        
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
    if "MONTH" in sql_up and "ORDER" in sql_up and not needs_regeneration:
        has_month_name_group = re.search(r"GROUP\s+BY\s+.*?DATE_FORMAT", sql_up, re.DOTALL)
        # FIX: Accept sorting by date columns (any alias) as safe - gives chronological order
        # Pattern matches: MONTH(date), YEAR(date), MIN(...), MAX(...), or any *Date column
        has_safe_sort = re.search(
            r"ORDER\s+BY\s+(.*?\.)?(MONTH|YEAR|MIN|MAX|STR_TO_DATE)\(",
            sql_up, re.DOTALL | re.IGNORECASE
        ) or re.search(
            r"ORDER\s+BY\s+.*?\w*Date\b",  # Matches: orderDate, o.orderDate, t.paymentDate, etc.
            sql_up, re.IGNORECASE
        )

        if has_month_name_group and not has_safe_sort:
             _dbg_print_two("[Agent3] HARD ERROR - Unsafe Month Sorting", "Sorting by Month Name (Alphabetical) detected.")
             extra_instruction = (
                "STRICT REQUIREMENT: You are grouping by Month Name, which sorts Alphabetically (Apr, Aug...). "
                "FIX: You MUST Group and Sort by the Month Number as well. "
                "Example: `GROUP BY MONTH(orderDate), month_name ORDER BY MONTH(orderDate) DESC`."
             )
             needs_regeneration = True
             regeneration_model = LARGE_MODEL  # Force 7B

    # --- KASUS K: AGGREGATION VIOLATION (ERROR 1055) (HARD ERROR) ---
    # Menangkap kasus SELECT raw_col FROM... GROUP BY derived_col (Error 1055)
    if "GROUP BY" in sql_up and not needs_regeneration:
        group_clause = re.search(r"GROUP\s+BY\s+(.*?)($|ORDER|LIMIT|HAVING)", sql_up, re.DOTALL)
        if group_clause:
            group_text = group_clause.group(1).lower()
            # Risky cols: kolom yang ada di dalam ekspresi Group By (e.g. CASE WHEN msrp < ...)
            risky_cols = set(re.findall(r"\b([a-z0-9_]+)\s*[<>=]", group_text))
            risky_cols.update(re.findall(r"\((?:.*?\s)?([a-z0-9_]+)(?:\s.*?)?\)", group_text))
            risky_cols = {c for c in risky_cols if c in all_valid_cols}
            
            select_clause = re.search(r"^SELECT\s+(.*?)\s+FROM", sql_up, re.DOTALL).group(1).lower()
            
            for col in risky_cols:
                # Cek apakah 'col' muncul di SELECT tanpa fungsi agregasi
                matches = re.finditer(rf"\b{col}\b", select_clause)
                for m in matches:
                    start, end = m.span()
                    is_func = select_clause[end:end+5].strip().startswith('(')
                    prev_text = select_clause[max(0, start-10):start].upper()
                    is_alias = "AS " in prev_text or "AS" in prev_text
                    
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

    # 4. EXECUTE REGENERATION (dengan dynamic model selection)
    final_model = state.get("target_model", "")  # Default to initial model
    if needs_regeneration:
        # Use dynamic model selection
        llm_regen = build_llm_repair(model=regeneration_model)
        model_label = "7B" if regeneration_model == LARGE_MODEL else "3B"
        _dbg_print_two("[Agent3] Regenerating with", model_label)

        new_sql, _ = regenerate_with_extra_constraint(
            llm_regen,
            make_generation_prompt,
            ctx,
            question,
            extra_instruction
        )
        if new_sql:
            final_sql = normalize_sql_pipeline(new_sql, question, ctx)
            corrections.append(f"Regenerated ({model_label}) & Re-Normalized")
            # Track final model used
            final_model = regeneration_model

    final_sql = final_sql.strip()
    if not final_sql.endswith(";"): final_sql += ";"

    # END TIMER: Calculate processing time (Model ready → Final SQL complete)
    # This is the core text-to-SQL processing latency for thesis benchmark
    processing_start = state.get("processing_start_time")
    if processing_start:
        processing_time = _time.perf_counter() - processing_start
        state["processing_time"] = processing_time
        print(f"[DEBUG] Processing time: {processing_time:.2f}s")

    return {
        **state,
        "final_sql": final_sql,
        "answer": final_sql,
        "critic_feedback": " | ".join(corrections),
        "final_model": final_model,
    }