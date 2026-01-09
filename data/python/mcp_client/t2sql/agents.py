# ======================================================================
# ============================ agents.py ===============================
# ======================================================================
import re
from typing import TYPE_CHECKING, Dict, List, Optional, Set

_SQL_KEYWORDS: Set[str] = {
    "select","from","where","group","by","order","limit","join","left","right","inner",
    "on","and","or","not","as","distinct","count","sum","avg","min","max","having",
    "asc","desc","like","in","between","is","null","case","when","then","else","end",
    "regexp","rlike","over","partition","row_number"
}

from .config import ROUTING_MODEL, SMALL_MODEL, LARGE_MODEL
from .llm_utils import build_llm_json, build_llm_repair
from .state import RAGState
from .utils import _dbg_print_two, try_llm_json, build_router_schema
from .generation import regenerate_with_extra_constraint
from .helpers_main import (get_single_allowed_table, validate_identifiers, _get_requires_regex, _regex_requirement_ok,
    _regex_requirement_ok, _get_qtype, make_router_prompt, make_generation_prompt, sanitize_notes, normalize_sql_pipeline)
from .helpers_core import(has_join, _question_requires_word_boundary_regex, _parse_allowed_identifiers)

# ======================================================================
# ====================== AGENT 1 - MODEL ROUTING =======================
# ======================================================================
def router_agent(state: RAGState) -> RAGState:
    print(f"\n[Router] Evidence-Based Analysis starting...")
    question = state.get("question", "").lower()
    ctx = state.get("context", "")
    
    # 1. IDENTIFIKASI TABEL (Selective Logic)
    allowed_identifiers = _parse_allowed_identifiers(ctx)
    whitelisted_tables = allowed_identifiers.keys()
    
    relevant_tables = []
    for tbl in whitelisted_tables:
        if f"- {tbl}:" in ctx or f"- `{tbl}`:" in ctx:
            table_word = tbl.lower().rstrip('s')
            
            if re.search(rf"\b{table_word}s?\b", question):
                # Deteksi referensi ID saja (FK)
                is_just_id_ref = re.search(rf"\b{table_word}s?\s+(number|id|code|kode|nomor)\b", question)
                if is_just_id_ref:
                    # Ambil tabel jika ada penyebutan entitas di luar ID
                    if re.search(rf"\b{table_word}s?\b(?!\s+(number|id|code|kode|nomor))", question):
                        relevant_tables.append(tbl)
                else:
                    relevant_tables.append(tbl)
    
    # 2. DETEKSI SINYAL
    math_keywords = ['sum', 'count', 'total', 'avg', 'average', 'number of', 'jumlah', 'rata-rata']
    has_math = any(re.search(rf"\b{re.escape(kw)}\b", question) for kw in math_keywords)
    
    value_ranks = ["highest", "lowest", "top", "bottom", "most", "least", "largest", "smallest", "terbesar", "terkecil", "paling"]
    time_ranks = ["latest", "earliest", "newest", "oldest", "terbaru", "terlama"]
    is_ranking_value = any(re.search(rf"\b{re.escape(kw)}\b", question) for kw in value_ranks)
    is_ranking_time = any(re.search(rf"\b{re.escape(kw)}\b", question) for kw in time_ranks)
    is_ranking = is_ranking_value or is_ranking_time
    
    explicit_sort = ["sort by", "ordered by", "sorted by", "order by", "sorted from"]
    is_sorting = any(re.search(rf"\b{re.escape(kw)}\b", question) for kw in explicit_sort) or is_ranking
    
    group_indicators = ["each", "per", "every", "masing-masing", "setiap", "for each", "classify"]
    has_group_word = any(re.search(rf"\b{re.escape(kw)}\b", question) for kw in group_indicators)
    is_by_logic = ("by " in question and not any(re.search(rf"\b{re.escape(s)}\b", question) for s in explicit_sort) and (has_math or is_ranking))
    has_grouping = has_group_word or is_by_logic

    # 3. LOGIKA ESKALASI
    is_multi_table = len(relevant_tables) > 1
    is_winner_row = is_ranking_value and has_grouping
    exclusion_keywords = ["no", "never", "without", "tidak pernah", "belum pernah", "tidak ada", "missing", "empty", "all customers", "all products"]
    has_exclusion = any(re.search(rf"\b{re.escape(kw)}\b", question) for kw in exclusion_keywords)

    # Note: Exclusion tidak lagi memaksa eskalasi agar kasus "missing attribute" (simple) tidak masuk complex
    should_escalate = is_multi_table or is_winner_row
    
    chosen_model = LARGE_MODEL if should_escalate else SMALL_MODEL
    complexity = "complex" if should_escalate else "simple"

    # Tentukan QTYPE untuk Agent 2
    if is_winner_row: qtype = "rank"
    elif has_math or has_grouping: qtype = "aggregate"
    else: qtype = "lookup"

    print(f"[Router] Detected Tables: {relevant_tables}")
    print(f"[Router] Signals -> Math:{has_math}, Sort:{is_sorting}, Rank:{is_ranking}, Group:{has_grouping}, WinnerRow:{is_winner_row}, Exclusion:{has_exclusion}")
    print(f"[Router] Decision: {complexity.upper()} -> {chosen_model}")

    return {
        **state,
        "router_tables": relevant_tables,
        "target_model": chosen_model,
        "router_decision": {
            "complexity": complexity,
            "qtype": qtype,
            "has_grouping": has_grouping,
            "is_ranking": is_ranking,
            "is_winner_row": is_winner_row,
            "has_exclusion": has_exclusion
        }
    }

# ======================================================================
# ======================= AGENT 2 - SQL DRAFTER ========================
# ======================================================================
def draft_sql_agent(state: RAGState) -> RAGState:
    target_model = state.get("target_model") or LARGE_MODEL
    llm = build_llm_json(model=target_model)

    question = state.get("question", "")
    ctx = state.get("context", "")
    decision = state.get("router_decision", {})
    
    # Panggil Dynamic Prompt Builder (helpers_main.py)
    # Segala aturan heuristik (Anti-MAX, Anti-1055, Anti-Inner Join) sudah diurus di sana.
    prompt = make_generation_prompt(ctx, question, decision)
    
    obj = try_llm_json(llm, prompt, max_attempts=2)
    
    if obj is None:
        return {**state, "draft_sql": "SELECT 'Error: LLM failed';"}
        
    draft_sql = (obj.get("sql") or "SELECT 1;").strip()
    print(f"\n[DRAFT] Raw from LLM: {draft_sql}")
    
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
    # GUNAKAN SELALU LARGE_MODEL (7B) untuk perbaikan logika yang kompleks
    target_model = LARGE_MODEL
    
    llm = build_llm_repair(model=target_model)
    question = state.get("question", "")
    ctx = state.get("context", "")
    draft_sql = (state.get("draft_sql") or "").strip()
    
    decision = state.get("router_decision", {})
    detected_tables = state.get("router_tables") or []
    
    # 1. HEURISTIC PIPELINE (Versi Bersih/Safe Mode)
    final_sql = normalize_sql_pipeline(draft_sql, question, ctx)
    sql_up = final_sql.upper()
    corrections = ["Applied Heuristic Pipeline"]

    # 2. PREPARE CHECKS (FULL PROTECTION)
    sql_for_tokenizing = final_sql
    sql_for_tokenizing = re.sub(r"'(?:''|[^'])*'", " ", sql_for_tokenizing)
    sql_for_tokenizing = re.sub(r'"(?:""|[^"])*"', " ", sql_for_tokenizing)
    sql_tokens = set(re.findall(r"\b([a-z][a-z0-9_]*)\b", sql_for_tokenizing.lower()))
    
    allowed_ids = _parse_allowed_identifiers(ctx)
    reserved = _SQL_KEYWORDS.union({
        'year', 'month', 'day', 'date', 'date_format', 'now', 'curdate',
        'concat', 'group_concat', 'coalesce', 'ifnull', 'sum', 'count', 'avg',
        't', 'limit', 'offset', 'as', 'regexp_like', 'str_to_date',
        'case', 'when', 'then', 'else', 'end',
        'union', 'all', 'distinct', 'values', 'exists', 'having'
    })
    
    defined_aliases = set()
    defined_aliases.update(re.findall(r"\bAS\s+([a-zA-Z0-9_]+)\b", sql_up, re.IGNORECASE))
    quoted_aliases = re.findall(r"\bAS\s+['\"`]?([a-zA-Z0-9_]+)['\"`]?", sql_up, re.IGNORECASE)
    defined_aliases.update(quoted_aliases)
    
    reserved.update(a.lower() for a in defined_aliases)

    primary_match = re.search(r"FROM\s+([A-Za-z0-9_]+)", sql_up)
    primary_tbl = primary_match.group(1).lower() if primary_match else ""
    if primary_tbl: reserved.add(primary_tbl)

    primary_cols = {c.lower() for c in allowed_ids.get(primary_tbl, [])}
    alien_cols = [t for t in sql_tokens if t not in reserved and t not in primary_cols and not t.isdigit()]
    
    # --- FLATTEN VALID COLS FOR KASUS J ---
    all_valid_cols = set()
    for cols in allowed_ids.values():
        all_valid_cols.update(cols)

    # 3. DECISION MATRIX
    needs_regeneration = False
    extra_instruction = ""

    # --- KASUS A: Schema Validation & Hallucination Check (SMART) ---
    if alien_cols and "JOIN" not in sql_up:
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
            _dbg_print_two("[Agent3] 🚨 Missing JOIN", f"Columns {alien_cols} belong to {potential_table}")
            extra_instruction = (
                f"STRICT REQUIREMENT: Columns {alien_cols} do NOT exist in table '{primary_tbl}', but they belong to '{potential_table}'. "
                f"Detected intent implies multiple tables: {detected_tables}. "
                "You MUST use explicit JOIN (e.g., JOIN {potential_table} ON ...) to link the data."
            )
            needs_regeneration = True
            
        elif not can_be_solved_by_join:
             _dbg_print_two("[Agent3] 🚨 Hallucination/Typo", f"Columns {alien_cols} do not exist in Schema.")
             extra_instruction = (
                f"STRICT REQUIREMENT: The columns {alien_cols} do NOT exist in the database schema. "
                "1. Check for typos (e.g. `order_date` vs `orderDate`). "
                "2. Use ONLY columns listed in the schema context. "
                "3. Do NOT hallucinate snake_case names if the schema uses camelCase."
             )
             needs_regeneration = True

    # --- KASUS B: Winner-Row Leak Check ---
    is_winner_row = decision.get("is_winner_row", False)
    if is_winner_row and not needs_regeneration:
        if re.search(r"SELECT\s+.*?\brn\b.*?FROM", sql_up):
             _dbg_print_two("[Agent3] 🚨 Winner-Row Leak", "rn column found in outer select.")
             extra_instruction = (
                "STRICT REQUIREMENT: Use standard Winner-Row pattern. "
                "Do NOT include the 'rn' (row number) column in the final output."
             )
             needs_regeneration = True

    # --- KASUS C: ANTI-PATTERN (WHERE IN Subquery) ---
    tuple_in_pattern = re.search(r"WHERE\s*\([^)]+,[^)]+\)\s*IN\s*\(SELECT", sql_up)
    if tuple_in_pattern and not needs_regeneration:
        _dbg_print_two("[Agent3] 🚨 Anti-Pattern Detected", "Tuple comparison inside WHERE IN (...)")
        extra_instruction = (
            "STRICT REQUIREMENT: Do NOT use the `WHERE (col1, col2) IN (SELECT ...)` pattern. "
            "It causes Error 1055. INSTEAD, use the Window Function pattern:\n"
            "SELECT ... FROM (SELECT ..., ROW_NUMBER() OVER (...) as rn FROM table) t WHERE rn = 1;"
        )
        needs_regeneration = True

    # --- KASUS D: MISSING PARTITION KEY IN OUTPUT ---
    if is_winner_row and not needs_regeneration:
        part_match = re.search(r"PARTITION\s+BY\s+([A-Za-z0-9_]+)", sql_up)
        if part_match:
            part_col = part_match.group(1)
            select_clause_match = re.search(r"^SELECT\s+(.*?)\s+FROM", sql_up, re.DOTALL)
            if select_clause_match:
                select_clause = select_clause_match.group(1)
                if part_col not in select_clause.upper():
                    _dbg_print_two("[Agent3] 🚨 Logic Check Failed", f"Partition key '{part_col}' missing from SELECT")
                    extra_instruction = (
                        f"STRICT REQUIREMENT: The query partitions by '{part_col}', but it is MISSING from the final SELECT list. "
                        f"You MUST include '{part_col}' in the output so we know who the record belongs to. "
                        f"Target format: SELECT {part_col}, [other columns] FROM ..."
                    )
                    needs_regeneration = True

    # --- KASUS E: INNER JOIN TRAP (DYNAMIC) ---
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
                 _dbg_print_two("[Agent3] 🚨 Logic Check Failed", f"Filtering child table aliases {risky_filters} in WHERE clause without LEFT JOIN.")
                 extra_instruction = (
                    "STRICT REQUIREMENT: You are aggregating data (GROUP BY), but you are filtering a related table in the WHERE clause. "
                    "This behaves like an INNER JOIN and removes master records with no matches. "
                    "1. You MUST use 'LEFT JOIN'. "
                    "2. You MUST move the filters for the related table from WHERE to the ON clause."
                 )
                 needs_regeneration = True

    # --- KASUS F: CRITICAL COLUMN OMISSION (DYNAMIC) ---
    if not needs_regeneration:
        missing_cols = []
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
            if col_lower in sql_tokens: continue
            
            clean_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', col_raw)
            clean_name = clean_name.replace('_', ' ').lower()
            parts = clean_name.split()
            if not parts: continue

            last_word = parts[-1]
            if last_word in suffix_synonyms:
                prefix = " ".join(parts[:-1])
                pattern_suffix = suffix_synonyms[last_word]
                pattern = rf"\b{re.escape(prefix)}\s*{pattern_suffix}\b"
            else:
                pattern = rf"\b{re.escape(clean_name)}\b"
            
            if re.search(pattern, question, re.IGNORECASE):
                missing_cols.append(col_raw)

        if missing_cols:
             _dbg_print_two("[Agent3] 🚨 Critical Omission", f"User asked for {missing_cols} but not selected.")
             extra_instruction = (
                f"STRICT REQUIREMENT: The user explicitly asked for these columns: {missing_cols}. "
                f"Please add {missing_cols} to the SELECT clause."
             )
             needs_regeneration = True

    # --- KASUS G: BROKEN ALIAS SYNTAX ---
    if not needs_regeneration:
        if re.search(r"\b[a-z0-9_]+\.\(", sql_up.lower()):
             _dbg_print_two("[Agent3] 🚨 Syntax Error", "Broken alias syntax `alias.(...)` detected.")
             extra_instruction = (
                "SYNTAX ERROR: You generated invalid SQL syntax like `table_alias.(condition)`. "
                "CORRECT: `(table_alias.column IS NULL ...)`."
             )
             needs_regeneration = True

    # --- KASUS H: REGEX PATTERN MISMATCH ---
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
             _dbg_print_two("[Agent3] 🚨 Pattern Mismatch", "Question asks for strict pattern (digits), but SQL uses simple LIKE.")
             extra_instruction = (
                "STRICT REQUIREMENT: The user asks for a strict pattern (e.g. 'digits' vs 'characters', or exact length). "
                "Simple 'LIKE' cannot distinguish digits from characters. "
                "1. You MUST use `REGEXP` (or `REGEXP_LIKE`). "
                "2. Use `[0-9]` for digits. "
                "3. Use anchors `^` (start) and `$` (end) if exact length is needed. "
                "4. Example for 'starts with x followed by 3 digits': `REGEXP '^x[0-9]{3}$'`"
             )
             needs_regeneration = True

    # --- KASUS I: UNSAFE MONTH SORTING (NEW) ---
    if "MONTH" in sql_up and "ORDER" in sql_up and not needs_regeneration:
        has_month_name_group = re.search(r"GROUP\s+BY\s+.*?DATE_FORMAT", sql_up, re.DOTALL)
        has_safe_sort = re.search(r"ORDER\s+BY\s+.*?(MONTH|YEAR|MIN|MAX|STR_TO_DATE)\(", sql_up, re.DOTALL)
        
        if has_month_name_group and not has_safe_sort:
             _dbg_print_two("[Agent3] 🚨 Unsafe Month Sorting", "Sorting by Month Name (Alphabetical) detected.")
             extra_instruction = (
                "STRICT REQUIREMENT: You are grouping by Month Name, which sorts Alphabetically (Apr, Aug...). "
                "FIX: You MUST Group and Sort by the Month Number as well. "
                "Example: `GROUP BY MONTH(orderDate), month_name ORDER BY MONTH(orderDate) DESC`."
             )
             needs_regeneration = True

    # --- KASUS J: AGGREGATION VIOLATION (ERROR 1055) ---
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
                         _dbg_print_two("[Agent3] 🚨 Aggregation Violation", f"Raw column '{col}' selected but grouped by derived value.")
                         extra_instruction = (
                            f"STRICT REQUIREMENT: You are grouping by a derived value of '{col}' (e.g. CASE or DATE_FORMAT), "
                            f"but you are selecting the raw '{col}' column. This causes SQL Error 1055. "
                            f"FIX: Remove '{col}' from the SELECT clause. Only select the grouping key and aggregates (COUNT, SUM)."
                         )
                         needs_regeneration = True
                         break
                if needs_regeneration: break

    # 4. EXECUTE REGENERATION
    if needs_regeneration:
        new_sql, _ = regenerate_with_extra_constraint(llm, make_generation_prompt, ctx, question, extra_instruction)
        if new_sql:
            final_sql = normalize_sql_pipeline(new_sql, question, ctx)
            corrections.append("Regenerated & Re-Normalized")

    final_sql = final_sql.strip()
    if not final_sql.endswith(";"): final_sql += ";"

    return {
        **state,
        "final_sql": final_sql,
        "answer": final_sql,
        "critic_feedback": " | ".join(corrections)
    }