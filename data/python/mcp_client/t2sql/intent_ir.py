# ======================================================================
# ============================ intent_ir.py =============================
# ======================================================================
# Purpose:
# - Build schema-grounded Intent IR (intermediate representation) from question.
# - Keep parsing deterministic and metadata-driven (no table hardcode).
# - Provide confidence/plannability gate for IR-first SQL planning.

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .schema_analyzer import (
    get_all_tables,
    get_column_alias_tags,
    get_column_datatype,
    get_column_synonyms,
    get_column_primary_role,
    get_column_description,
    get_column_role,
    get_column_role_tags,
    get_column_stats,
    get_fk_relationships,
    get_id_column,
    get_name_column,
    get_schema_analyzer,
    get_singular_form,
    get_table_synonyms,
    get_table_type,
)
from .config import (
    DOMAIN_HEURISTICS_ENABLED,
    INTENT_IR_BASE_NUMERIC_HINTS,
    INTENT_IR_DOMAIN_NUMERIC_FALLBACK_HINTS,
)
from .rulebook import (
    aggregation_count_distinct_keywords,
    aggregation_count_distinct_phrases,
    any_phrase_in_text,
)
_DATE_HINTS = ("date", "time", "timestamp", "year", "month", "day")
_NUMBER_TOKEN_PATTERN = r"-?\d[\d,]*(?:\.\d+)?"
_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
_MONTH_NAME_TO_NUM = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_FUNCTION_WORD_TOKENS = {
    "the", "a", "an",
    "that", "this", "those", "these",
    "their", "its", "his", "her", "our", "your",
    "of", "to", "for", "with", "without", "and", "or",
    "each", "every", "all", "per",
}
_LEXICAL_MATCH_STOP_TOKENS = _FUNCTION_WORD_TOKENS.union(
    {
        "show", "list", "display", "find", "return", "get",
        "top", "bottom", "first", "last",
        "highest", "lowest", "largest", "smallest",
        "latest", "newest", "earliest", "oldest",
        "sorted", "sort", "ordered", "order",
        "level", "levels",
        "than", "over", "under", "above", "below",
        "more", "less", "least", "most", "minimum", "maximum",
    }
)
_NUMERIC_HINT_TERMS_CACHE: Optional[Set[str]] = None
_SCHEMA_ENTITY_HEAD_TOKEN_CACHE: Optional[Set[str]] = None
_SCHEMA_GROUPABLE_TERM_CACHE: Optional[Set[str]] = None
_GENERIC_IDENTIFIER_TOKENS = {"id", "ids", "number", "numbers", "code", "codes", "key", "keys"}
_GENERIC_GROUPING_TERMS = {
    "status", "type", "category",
    "month", "year", "quarter",
    "country", "city", "state", "territory",
}
_STRUCTURED_IDENTIFIER_ROLE_TERMS = {
    "id", "identifier", "pk", "fk",
    "primary_key", "foreign_key",
    "bridge_key", "fact_key",
}
_STRUCTURED_NAME_ROLE_TERMS = {
    "name", "entity_name", "display_name",
    "label", "title", "descriptor",
}
_STRUCTURED_NUMERIC_ROLE_TERMS = {
    "metric", "measure", "amount", "quantity",
    "monetary", "numeric", "value", "score",
    "ratio", "percentage", "percent", "count", "total",
}
_STRUCTURED_TEMPORAL_ROLE_TERMS = {
    "date", "datetime", "timestamp", "time", "temporal",
    "year", "month", "day", "event_date", "event_time",
}
_NUMERIC_DTYPE_HINTS = ("INT", "DECIMAL", "NUMERIC", "FLOAT", "DOUBLE", "REAL", "MONEY")
_TEMPORAL_DTYPE_HINTS = ("DATE", "TIME", "DATETIME", "TIMESTAMP", "YEAR")
_TEXTUAL_DTYPE_HINTS = ("CHAR", "TEXT", "CLOB", "ENUM", "SET")


def _fallback_singular(table: str) -> str:
    t = (table or "").strip().lower()
    if t.endswith("ies") and len(t) > 3:
        return t[:-3] + "y"
    if t.endswith("s") and not t.endswith("ss") and len(t) > 1:
        return t[:-1]
    return t


def _safe_get_singular_form(table: str) -> str:
    try:
        return get_singular_form(table)
    except Exception:
        return _fallback_singular(table)


def _safe_get_table_synonyms(table: str) -> List[str]:
    try:
        return list(get_table_synonyms(table) or [])
    except Exception:
        return []


def _safe_get_column_synonyms(table: str, column: str) -> List[str]:
    try:
        return list(get_column_synonyms(table, column) or [])
    except Exception:
        return []


def _safe_get_column_primary_role(table: str, column: str) -> str:
    try:
        return str(get_column_primary_role(table, column) or "")
    except Exception:
        return ""


def _safe_get_column_role_tags(table: str, column: str) -> List[str]:
    try:
        return list(get_column_role_tags(table, column) or [])
    except Exception:
        return []


def _safe_get_column_alias_tags(table: str, column: str) -> List[str]:
    try:
        return list(get_column_alias_tags(table, column) or [])
    except Exception:
        return []


def _safe_get_column_stats(table: str, column: str) -> Dict[str, Any]:
    try:
        stats = get_column_stats(table, column) or {}
    except Exception:
        stats = {}
    return dict(stats) if isinstance(stats, dict) else {}


def _safe_get_column_datatype(table: str, column: str) -> str:
    try:
        return str(get_column_datatype(table, column) or "")
    except Exception:
        return ""


def _safe_get_column_description(table: str, column: str) -> str:
    try:
        return str(get_column_description(table, column) or "")
    except Exception:
        return ""


def _safe_get_id_column(table: str) -> Optional[str]:
    try:
        return get_id_column(table)
    except Exception:
        return None


def _safe_get_name_column(table: str) -> Optional[str]:
    try:
        return get_name_column(table)
    except Exception:
        return None


def _safe_get_table_type(table: str) -> str:
    try:
        return get_table_type(table)
    except Exception:
        return "unknown"


def _normalize_categorical_literal_value(
    value: str,
    *,
    column: str = "",
) -> str:
    raw = str(value or "").strip(" \t\r\n'\".,;:!?")
    if not raw:
        return ""

    raw = re.sub(r"^(?:either\s+the|either|the)\s+", "", raw, flags=re.I).strip()
    raw = re.sub(r"\s+(?:territories|territory|regions|region)\s*$", "", raw, flags=re.I).strip()

    ordered_col_words = [
        tok for tok in re.findall(r"[a-z0-9]+", _camel_to_words(column or "").lower())
        if tok
    ]
    col_words = set(ordered_col_words)
    if col_words:
        suffix_terms = set()
        for tok in col_words:
            suffix_terms.add(tok)
            if tok.endswith("y") and len(tok) > 1:
                suffix_terms.add(tok[:-1] + "ies")
            else:
                suffix_terms.add(tok + "s")
        def _norm_term(token: str) -> str:
            t = str(token or "").strip().lower()
            if not t:
                return ""
            if t.endswith("ies") and len(t) > 3:
                return t[:-3] + "y"
            if t.endswith("ses") and len(t) > 3:
                return t[:-2]
            if t.endswith("s") and len(t) > 2 and not t.endswith(("ss", "us")):
                return t[:-1]
            return t
        raw_parts = [part for part in raw.split() if part]
        raw_norm_words = [_norm_term(part) for part in raw_parts]
        lead_count = 0
        for norm_word in raw_norm_words:
            if norm_word and norm_word in col_words:
                lead_count += 1
                continue
            break
        if 0 < lead_count < len(raw_parts):
            raw = " ".join(raw_parts[lead_count:]).strip(" \t\r\n'\".,;:!?")
        raw_words = re.findall(r"[a-z0-9]+", raw.lower())
        if raw_words and raw_words[-1] in suffix_terms and len(raw_words) >= 2:
            kept_words = raw.split()
            raw = " ".join(kept_words[:-1]).strip(" \t\r\n'\".,;:!?")

    return raw


@dataclass
class IRDimension:
    table: str
    column: str
    alias: str = ""
    transform: str = "none"  # none|month_name|month_num|year
    visible: bool = True


@dataclass
class IRMetric:
    agg: str  # count|sum|avg|min|max|pct_non_null
    table: str
    column: str = "*"
    alias: str = ""
    distinct: bool = False


@dataclass
class IRFilter:
    table: str
    column: str
    op: str  # eq|in|not_in|gt|gte|lt|lte|between|year_eq|month_eq|month_between|weekday_eq|like_prefix|like_suffix|like_contains|not_like_contains|not_like_prefix|not_like_suffix|regexp|word_count_eq|len_eq|len_gt|len_gte|len_lt|len_lte|is_null_or_empty|is_not_null_or_empty|col_eq|col_gt|col_gte|col_lt|col_lte
    value: Any = None
    value2: Any = None
    value_column: str = ""


@dataclass
class IRExistence:
    table: str
    mode: str  # at_least_one|none


@dataclass
class IROrder:
    kind: str  # alias|column|expression
    by: str
    direction: str = "ASC"
    table: str = ""


@dataclass
class IRComputedProjection:
    kind: str  # arithmetic
    op: str  # sub|add|mul|div
    left_table: str
    left_column: str
    right_table: str
    right_column: str
    alias: str = ""


@dataclass
class IntentIR:
    subject_entity: str = ""
    dimensions: List[IRDimension] = field(default_factory=list)
    metrics: List[IRMetric] = field(default_factory=list)
    computed_projections: List[IRComputedProjection] = field(default_factory=list)
    filters: List[IRFilter] = field(default_factory=list)
    filter_logic: str = ""
    existence: List[IRExistence] = field(default_factory=list)
    ordering: List[IROrder] = field(default_factory=list)
    order_by_column: str = ""
    order_by_direction: str = ""
    limit: Optional[int] = None
    include_zero: bool = False
    select_distinct: bool = False
    qtype: str = ""
    notes: List[str] = field(default_factory=list)
    confidence: float = 0.0
    planner_compatible: bool = True
    mentioned_tables: List[str] = field(default_factory=list)
    ambiguity_signals: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _parse_allowed_identifiers(ctx: str) -> Dict[str, Set[str]]:
    tables: Dict[str, Set[str]] = {}
    m = re.search(r"### ALLOWED IDENTIFIERS(.*?)(?:\n### |\Z)", ctx or "", flags=re.S)
    if not m:
        return tables

    block = m.group(1)
    for line in block.splitlines():
        s = line.strip()
        if not s.startswith("- "):
            continue
        table_col = s[2:].split(":", 1)
        if len(table_col) != 2:
            continue
        table = table_col[0].strip().strip("`")
        cols = [c.strip().strip("`") for c in table_col[1].split(",") if c.strip()]
        if table:
            tables.setdefault(table.lower(), set()).update(cols)
    return tables


def _load_schema_columns(table: str) -> Set[str]:
    try:
        analyzer = get_schema_analyzer()
        cols = analyzer.get_table_columns(table)
        return {str(c) for c in cols if str(c)}
    except Exception:
        return set()


def _load_allowed_or_schema(ctx: str) -> Dict[str, Set[str]]:
    allowed = _parse_allowed_identifiers(ctx)
    if allowed:
        return allowed

    out: Dict[str, Set[str]] = {}
    try:
        tables = list(get_all_tables())
    except Exception:
        tables = []
    for t in tables:
        cols = _load_schema_columns(t)
        if cols:
            out[t.lower()] = cols
    return out


def _ordered_allowed_tables(allowed: Dict[str, Set[str]]) -> List[Tuple[str, Set[str]]]:
    return [
        (table, allowed[table])
        for table in sorted(
            (
                str(table or "").strip().lower()
                for table in (allowed or {}).keys()
                if str(table or "").strip()
            ),
            key=lambda table: table,
        )
    ]


def _ordered_allowed_columns(table: str, allowed: Dict[str, Set[str]]) -> List[str]:
    table_l = str(table or "").strip().lower()
    return sorted(
        (
            str(col or "").strip()
            for col in (allowed.get(table_l, set()) or set())
            if str(col or "").strip()
        ),
        key=lambda col: (len(col), col.lower()),
    )


def _ordered_variants(values: Iterable[str]) -> Tuple[str, ...]:
    cleaned = {
        str(value or "").strip()
        for value in (values or [])
        if str(value or "").strip()
    }
    return tuple(sorted(cleaned, key=lambda value: (-len(value), value.lower(), value)))


def _camel_to_words(s: str) -> str:
    x = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s or "")
    x = x.replace("_", " ")
    return re.sub(r"\s+", " ", x).strip().lower()


def _phrase_spacing_variants(text: str) -> Set[str]:
    base = re.sub(r"\s+", " ", str(text or "").strip().lower())
    if not base:
        return set()
    variants: Set[str] = {base}
    queue = [base]
    while queue:
        cur = queue.pop()
        candidates = [
            re.sub(r"(?<=[a-z])(?=\d)", " ", cur),
            re.sub(r"(?<=\d)(?=[a-z])", " ", cur),
            re.sub(r"(?<=[a-z])\s+(?=\d)", "", cur),
            re.sub(r"(?<=\d)\s+(?=[a-z])", "", cur),
        ]
        for candidate in candidates:
            normalized = re.sub(r"\s+", " ", str(candidate or "").strip().lower())
            if normalized and normalized not in variants:
                variants.add(normalized)
                queue.append(normalized)
    return variants


def _semantic_token_variants(token: str) -> List[str]:
    tok = str(token or "").strip().lower()
    if not tok:
        return []

    out: List[str] = []

    def _add(value: str) -> None:
        cur = str(value or "").strip().lower()
        if cur and cur not in out:
            out.append(cur)

    _add(tok)

    if tok.endswith("ied") and len(tok) > 4:
        _add(tok[:-3] + "y")
    elif tok.endswith("ed") and len(tok) > 4:
        stem = tok[:-2]
        if len(stem) >= 3:
            _add(stem)
            if len(stem) >= 2 and stem[-1] == stem[-2]:
                _add(stem[:-1])
        if len(tok[:-1]) >= 3:
            _add(tok[:-1])

    if tok.endswith("ies") and len(tok) > 3:
        _add(tok[:-3] + "y")
    elif tok.endswith("uses") and len(tok) > 4:
        _add(tok[:-2])
    elif tok.endswith("sses") and len(tok) > 4:
        _add(tok[:-2])
    elif tok.endswith(("xes", "zes", "ches", "shes")) and len(tok) > 3:
        _add(tok[:-2])
    elif tok.endswith("ses") and len(tok) > 3:
        if tok[:-1].endswith("e"):
            _add(tok[:-1])
        else:
            _add(tok[:-2])
    elif tok.endswith("es") and len(tok) > 3:
        if tok[:-1].endswith("e"):
            _add(tok[:-1])
        else:
            _add(tok[:-2])
    elif tok.endswith("s") and len(tok) > 2 and not tok.endswith(("ss", "us")):
        _add(tok[:-1])

    return out


def _pluralize_last_token(phrase: str) -> str:
    tokens = [t for t in (phrase or "").split() if t]
    if not tokens:
        return phrase
    last = tokens[-1]
    if last.endswith("ies") or last.endswith("s"):
        return phrase
    if last.endswith("y") and len(last) > 1 and last[-2] not in "aeiou":
        tokens[-1] = last[:-1] + "ies"
    else:
        tokens[-1] = last + "s"
    return " ".join(tokens)


def _singularize_last_token(phrase: str) -> str:
    tokens = [t for t in (phrase or "").split() if t]
    if not tokens:
        return phrase
    last = tokens[-1]
    variants = [v for v in _semantic_token_variants(last) if v != last]
    if variants:
        tokens[-1] = variants[0]
    return " ".join(tokens)


def _table_variants(table: str) -> Tuple[str, ...]:
    t = (table or "").strip().lower()
    if not t:
        return tuple()
    out = {t, t.replace("_", " ")}

    singular = _safe_get_singular_form(t) or t
    out.add(singular.lower())
    out.add(singular.lower().replace("_", " "))

    name_col = _safe_get_name_column(t)
    if not name_col:
        singular_compact = singular.lower().replace("_", "").replace(" ", "")
        try:
            analyzer = get_schema_analyzer()
            for candidate in analyzer.get_table_columns(t):
                cand_words = _camel_to_words(candidate)
                cand_compact = cand_words.replace(" ", "")
                if cand_compact and cand_compact == singular_compact:
                    name_col = candidate
                    break
        except Exception:
            name_col = None
    if name_col:
        name_words = _camel_to_words(name_col)
        if name_words:
            out.add(name_words)
            out.add(_pluralize_last_token(name_words))
            out.add(_singularize_last_token(name_words))

    for syn in _safe_get_table_synonyms(t):
        s = str(syn).strip().lower()
        if s:
            out.add(s)
            out.add(s.replace("_", " "))
    return _ordered_variants(x for x in out if x)


def _column_reference_variants(table: str, column: str) -> Tuple[str, ...]:
    t = (table or "").strip().lower()
    c = str(column or "").strip()
    if not c:
        return tuple()

    c_low = c.lower()
    original_c = c
    try:
        analyzer = get_schema_analyzer()
        original_c = str(
            analyzer._column_original_case.get(f"{t}.{c_low}", c)  # type: ignore[attr-defined]
            or c
        )
    except Exception:
        original_c = c
    out = {
        c_low,
        c_low.replace("_", " "),
        _camel_to_words(original_c),
    }
    for syn in _safe_get_column_synonyms(t, c):
        s = str(syn).strip().lower()
        if not s:
            continue
        out.add(s)
        out.add(s.replace("_", " "))
        out.add(s.replace("-", " "))
    for alias in _safe_get_column_alias_tags(t, c):
        a = str(alias).strip().lower()
        if not a:
            continue
        out.add(a)
        out.add(a.replace("_", " "))
        out.add(a.replace("-", " "))
    expanded: Set[str] = set()
    for v in out:
        for vv in _phrase_spacing_variants(v):
            if not vv:
                continue
            expanded.add(vv)
            expanded.add(_pluralize_last_token(vv))
            expanded.add(_singularize_last_token(vv))

    if c_low.endswith("number"):
        stem = c_low[:-6].strip("_")
        if stem:
            expanded.add(f"{stem} numbers".replace("_", " "))
    if c_low.endswith("code"):
        stem = c_low[:-4].strip("_")
        if stem:
            expanded.add(f"{stem} codes".replace("_", " "))

    return _ordered_variants(x for x in expanded if x)


def _column_variants(table: str, column: str) -> Tuple[str, ...]:
    t = (table or "").strip().lower()
    c = str(column or "").strip()
    if not c:
        return tuple()

    out = set(_column_reference_variants(table, column))
    description = re.sub(
        r"\s+",
        " ",
        _safe_get_column_description(t, c).strip().lower().replace("_", " ").replace("-", " "),
    ).strip()
    if description:
        out.add(description)
        for piece in re.split(r"\s*(?:,|;|/)\s*|\s+\bor\b\s+", description, flags=re.I):
            piece_norm = re.sub(r"\s+", " ", str(piece or "").strip(" \t\r\n'\"()[]{}"))
            if piece_norm and len(re.findall(r"[a-z0-9]+", piece_norm)) <= 6:
                out.add(piece_norm)

    expanded: Set[str] = set()
    for v in out:
        for vv in _phrase_spacing_variants(v):
            if vv:
                expanded.add(vv)
    if any("price each" in str(v or "").lower() or "unit price" in str(v or "").lower() for v in expanded):
        expanded.add("price per item")
    return _ordered_variants(x for x in expanded if x)


def _phrase_pos(text: str, phrase: str) -> int:
    if not text or not phrase:
        return -1
    pat = re.escape(phrase).replace(r"\ ", r"\s+")
    m = re.search(rf"\b{pat}\b", text, flags=re.I)
    return m.start() if m else -1


def _original_schema_column_case(table: str, column: str) -> str:
    t = str(table or "").strip().lower()
    c = str(column or "").strip()
    if not t or not c:
        return c
    c_low = c.lower()
    try:
        analyzer = get_schema_analyzer()
        return str(
            analyzer._column_original_case.get(f"{t}.{c_low}", c)  # type: ignore[attr-defined]
            or c
        )
    except Exception:
        return c


def _augment_subject_allowed_columns_from_schema(
    question: str,
    subject: str,
    allowed: Dict[str, Set[str]],
) -> Dict[str, Set[str]]:
    """
    Add explicitly mentioned subject columns that retrieval omitted but schema
    metadata knows. This keeps IR column resolution stable when retrieval surfaces
    a parent/lookup table column while the executable subject table has the same
    field.
    """
    table_l = str(subject or "").strip().lower()
    if not table_l or not allowed:
        return allowed

    q_l = str(question or "").strip().lower()
    if not q_l:
        return allowed

    schema_cols = _load_schema_columns(table_l)
    if not schema_cols:
        return allowed

    def _has_structural_column_intent(col: str) -> bool:
        variants = [
            str(variant or "").strip().lower()
            for variant in _column_variants(table_l, col)
            if str(variant or "").strip()
        ]
        if not variants:
            return False

        for seg in _extract_projection_segments(question):
            seg_l = str(seg or "").strip().lower()
            if any(_phrase_pos(seg_l, variant) >= 0 for variant in variants):
                return True

        for variant in variants:
            pat = re.escape(variant).replace(r"\ ", r"\s+")
            if not pat:
                continue
            if re.search(
                rf"\b(?:for\s+each|per|by|group(?:ed)?\s+by|in\s+each)\s+(?:the\s+)?{pat}\b",
                q_l,
                flags=re.I,
            ):
                return True
            if re.search(
                rf"\b(?:same|within\s+(?:the\s+)?same)\s+(?:the\s+)?{pat}\b",
                q_l,
                flags=re.I,
            ):
                return True
            if re.search(
                rf"\b(?:sort(?:ed)?|order(?:ed)?)\s+by\b[^.]*\b{pat}\b",
                q_l,
                flags=re.I,
            ):
                return True
            if re.search(
                rf"\b(?:sum|total|avg|average|mean|min(?:imum)?|max(?:imum)?|lowest|smallest|highest|largest|latest|earliest)\s+(?:the\s+)?{pat}\b",
                q_l,
                flags=re.I,
            ):
                return True

        return False

    out: Dict[str, Set[str]] = {
        str(t or "").strip().lower(): {
            str(c or "").strip()
            for c in (cols or set())
            if str(c or "").strip()
        }
        for t, cols in allowed.items()
        if str(t or "").strip()
    }
    current_lowers = {
        str(c or "").strip().lower()
        for c in out.get(table_l, set())
        if str(c or "").strip()
    }

    for col in sorted(
        (str(col or "").strip() for col in schema_cols),
        key=lambda col: (len(col), col.lower()),
    ):
        col_s = str(col or "").strip()
        col_l = col_s.lower()
        if not col_s or col_l in current_lowers:
            continue
        explicit_mention = any(
            _phrase_pos(q_l, str(variant or "").strip().lower()) >= 0
            for variant in _column_variants(table_l, col_s)
            if str(variant or "").strip()
        )
        if not explicit_mention:
            continue
        if not _has_structural_column_intent(col_s):
            continue
        out.setdefault(table_l, set()).add(_original_schema_column_case(table_l, col_s))
        current_lowers.add(col_l)

    return out


def _is_numeric_column(col: str, table: str = "") -> bool:
    cl = (col or "").lower()
    if table and _column_has_numeric_semantics(table, col):
        return True
    if _is_date_column(col, table):
        return False
    # ID-like columns (*Number, *Id, *Code) are identifiers, not numeric measures
    if (table and _is_identifier_like_column(table, col)) or cl.endswith("number") or cl.endswith("id") or cl.endswith("code"):
        return False
    if table:
        if _column_has_name_semantics(table, col):
            return False
        dtype = str(_column_schema_semantics(table, col).get("datatype") or "").upper()
        if any(tok in dtype for tok in _TEXTUAL_DTYPE_HINTS):
            return False
    words = _camel_to_words(col or "")
    tokens = {t for t in re.findall(r"[a-z0-9]+", words) if t}
    return any(k in tokens for k in _numeric_hint_terms())


def _is_date_column(col: str, table: str = "") -> bool:
    if table and _column_has_temporal_semantics(table, col):
        return True
    c = (col or "").lower()
    return any(k in c for k in _DATE_HINTS)


def _safe_alias(s: str, fallback: str) -> str:
    raw = re.sub(r"[^A-Za-z0-9_]+", "_", (s or "").strip()).strip("_").lower()
    if not raw:
        raw = fallback
    if raw[0].isdigit():
        raw = f"c_{raw}"
    return raw


def _to_float_token(token: str) -> Optional[float]:
    s = str(token or "").strip()
    if not s:
        return None
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def _to_int_count_token(token: str) -> Optional[int]:
    s = str(token or "").strip().lower()
    if not s:
        return None
    if s.isdigit():
        try:
            return int(s)
        except ValueError:
            return None
    return _NUMBER_WORDS.get(s)


def _question_requests_null_fill_projection(question: str) -> bool:
    q_raw = str(question or "")
    q = q_raw.lower()
    if not re.search(r"\b(?:missing|null|empty|blank)\b", q, flags=re.I):
        return False
    if not re.search(r"\b(?:if|when|whenever)\b", q, flags=re.I):
        return False
    if not re.search(r"\b(?:display|show|return|use)\b", q, flags=re.I):
        return False
    if re.search(r"""["'“”][^"'“”]{1,40}["'“”]""", q_raw):
        return True
    return bool(re.search(r"\b(?:n/?a|none|unknown|zero)\b", q, flags=re.I))


def _derive_filter_logic(question: str, filters: Sequence[IRFilter]) -> str:
    if len(list(filters or [])) < 2:
        return ""
    q = str(question or "").strip().lower()
    if not q:
        return ""

    sentence_scope = re.split(r"[.?!;]", q, maxsplit=1)[0]
    request_trimmed = re.sub(
        r"^\s*(?:(?:can|could|would)\s+you\s+|please\s+|i\s+need\s+to\s+|i\s+want\s+to\s+)?"
        r"(?:show|list|find|get|return|display|include|give|pull)\s+(?:me\s+|us\s+)?",
        "",
        sentence_scope,
        flags=re.I,
    ).strip()
    working_scope = request_trimmed or sentence_scope
    condition_scope = re.split(
        r"\b(?:return|show|list|display|include|give|sort|order(?:ed)?|then\s+show)\b",
        working_scope,
        maxsplit=1,
        flags=re.I,
    )[0]
    if condition_scope:
        q = condition_scope

    sanitized = re.sub(
        r"\bbetween\b[^.]{0,64}\band\b[^.]{0,64}",
        " ",
        q,
        flags=re.I,
    )
    for flt in (filters or []):
        op = str(getattr(flt, "op", "") or "").strip().lower()
        values = getattr(flt, "value", None)
        if op not in {"in", "not_in"} or not isinstance(values, list) or len(values) < 2:
            continue
        clean_values = [
            re.sub(r"\s+", " ", str(v or "").strip().lower())
            for v in values
            if str(v or "").strip()
        ]
        if len(clean_values) < 2:
            continue
        alt_values = sorted(
            {
                re.escape(v).replace(r"\ ", r"\s+")
                for v in clean_values
                if v
            },
            key=len,
            reverse=True,
        )
        if len(alt_values) < 2:
            continue
        value_alt = "|".join(alt_values)
        sanitized = re.sub(
            rf"\b(?:{value_alt})(?:\s*(?:,|/)\s*(?:{value_alt}))*\s*(?:or|and)\s*(?:{value_alt})\b",
            " ",
            sanitized,
            flags=re.I,
        )
    has_but = bool(re.search(r"\bbut\b", sanitized, flags=re.I))
    has_and = bool(re.search(r"\band\b", sanitized, flags=re.I))
    has_or = bool(re.search(r"\bor\b", sanitized, flags=re.I))

    if has_but:
        return "and"
    if has_and and has_or:
        return "mixed"
    if has_and:
        return "and"
    if has_or:
        return "or"
    return "and"


def _extract_projection_fill_literal(question: str) -> str:
    q_raw = str(question or "")
    quoted = re.search(r"""["'“”]([^"'“”]{1,40})["'“”]""", q_raw)
    if quoted:
        return str(quoted.group(1) or "").strip()

    fallback = re.search(r"\b(n/?a|none|unknown|zero)\b", q_raw, flags=re.I)
    if fallback:
        return str(fallback.group(1) or "").strip()
    return ""


def _prune_redundant_filters(filters: Sequence[IRFilter], filter_logic: str = "") -> List[IRFilter]:
    pruned = list(filters or [])
    if len(pruned) < 2:
        return pruned

    drop_indexes: Set[int] = set()
    logic_l = str(filter_logic or "").strip().lower()
    grouped: Dict[Tuple[str, str], List[Tuple[int, IRFilter]]] = {}
    for idx, flt in enumerate(pruned):
        key = (str(flt.table or "").strip().lower(), str(flt.column or "").strip().lower())
        grouped.setdefault(key, []).append((idx, flt))

    for _, entries in grouped.items():
        prefix_values = {
            str(f.value or "")
            for _, f in entries
            if str(f.op or "").strip().lower() == "like_prefix" and f.value is not None
        }
        suffix_values = {
            str(f.value or "")
            for _, f in entries
            if str(f.op or "").strip().lower() == "like_suffix" and f.value is not None
        }
        contains_values = {
            str(f.value or "")
            for _, f in entries
            if str(f.op or "").strip().lower() == "like_contains" and f.value is not None
        }
        regex_values = [
            str(f.value or "")
            for _, f in entries
            if str(f.op or "").strip().lower() == "regexp" and f.value is not None
        ]
        in_values = [
            {
                str(v or "").strip().lower()
                for v in (f.value or [])
                if str(v or "").strip()
            }
            for _, f in entries
            if str(f.op or "").strip().lower() == "in" and isinstance(f.value, list)
        ]
        not_in_values = [
            {
                str(v or "").strip().lower()
                for v in (f.value or [])
                if str(v or "").strip()
            }
            for _, f in entries
            if str(f.op or "").strip().lower() == "not_in" and isinstance(f.value, list)
        ]

        for idx, flt in entries:
            op = str(flt.op or "").strip().lower()
            val = str(flt.value or "")
            val_l = val.strip().lower()
            if op != "eq" or not val_l:
                continue

            if any(val == prefix for prefix in prefix_values):
                drop_indexes.add(idx)
                continue

            if any(val == suffix for suffix in suffix_values):
                drop_indexes.add(idx)
                continue

            if any(val == contained for contained in contains_values):
                drop_indexes.add(idx)
                continue

            escaped = re.escape(val)
            if any(re.match(rf"^\^{escaped}(?:\$|[\[\(A-Za-z0-9\\])", regex_value) for regex_value in regex_values):
                drop_indexes.add(idx)
                continue

            if logic_l == "or" and any(val_l in values for values in in_values):
                drop_indexes.add(idx)
                continue

            if any(val_l in values for values in not_in_values):
                drop_indexes.add(idx)

    return [flt for idx, flt in enumerate(pruned) if idx not in drop_indexes]


def _normalize_match_token(token: str) -> str:
    t = str(token or "").strip().lower()
    if not t:
        return ""
    if t.endswith("ies") and len(t) > 3:
        t = t[:-3] + "y"
    elif t.endswith("ses") and len(t) > 3:
        t = t[:-2]
    elif t.endswith("s") and len(t) > 2 and not t.endswith(("ss", "us")):
        t = t[:-1]
    return t


def _semantic_query_token_variants(tokens: Sequence[str]) -> Set[str]:
    variants: Set[str] = set()
    for token in tokens or []:
        tok = str(token or "").strip().lower()
        if not tok:
            continue
        variants.add(tok)
        if tok.endswith("ied") and len(tok) > 4:
            variants.add(tok[:-3] + "y")
        elif tok.endswith("ing") and len(tok) > 5:
            stem = tok[:-3]
            if len(stem) >= 2 and stem[-1] == stem[-2]:
                stem = stem[:-1]
            if stem:
                variants.add(stem)
        elif tok.endswith("ed") and len(tok) > 4:
            stem = tok[:-2]
            if len(stem) >= 2 and stem[-1] == stem[-2]:
                stem = stem[:-1]
            if stem:
                variants.add(stem)
    return {variant for variant in variants if variant}


def _tokenize_for_match(text: str, stop_tokens: Optional[Set[str]] = None) -> List[str]:
    stop = {str(tok).lower() for tok in (stop_tokens or set()) if str(tok)}
    out: List[str] = []
    seen: Set[str] = set()
    for raw in re.findall(r"[a-z0-9]+", str(text or "").lower()):
        if raw.isdigit():
            continue
        tok = _normalize_match_token(raw)
        if not tok or tok in stop:
            continue
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def _regex_alt_from_terms(terms: Set[str]) -> str:
    pieces = []
    for term in sorted((str(t or "").strip().lower() for t in terms), key=len, reverse=True):
        if not term:
            continue
        pieces.append(re.escape(term).replace(r"\ ", r"\s+"))
    return "|".join(dict.fromkeys(pieces))
def _schema_groupable_terms() -> Set[str]:
    global _SCHEMA_GROUPABLE_TERM_CACHE
    if _SCHEMA_GROUPABLE_TERM_CACHE is not None:
        return set(_SCHEMA_GROUPABLE_TERM_CACHE)

    terms: Set[str] = set(_GENERIC_GROUPING_TERMS)
    try:
        tables = list(get_all_tables() or [])
    except Exception:
        tables = []

    for table in tables:
        table_l = str(table or "").strip().lower()
        if not table_l:
            continue
        candidates = set(_table_variants(table_l) or {table_l})
        candidates.add(_safe_get_singular_form(table_l))
        candidates.update(_safe_get_table_synonyms(table_l))
        for candidate in candidates:
            phrase_tokens = _tokenize_for_match(_camel_to_words(str(candidate or "")), stop_tokens=_FUNCTION_WORD_TOKENS)
            if not phrase_tokens or len(phrase_tokens) > 3:
                continue
            phrase = " ".join(phrase_tokens).strip()
            if phrase and len(phrase) >= 3:
                terms.add(phrase)

    _SCHEMA_GROUPABLE_TERM_CACHE = terms
    return set(terms)


def _normalize_semantic_label(label: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(label or "").strip().lower()).strip("_")


def _role_term_variants(term: str) -> Set[str]:
    term_n = _normalize_semantic_label(term)
    if not term_n:
        return set()
    parts = [part for part in re.split(r"[_\s]+", term_n) if part]
    variants = {term_n}
    variants.update(parts)
    variants.add(term_n.replace("_", ""))
    return {variant for variant in variants if variant}


def _structured_role_matches(role_terms: Set[str], targets: Set[str]) -> bool:
    normalized_targets: Set[str] = set()
    for target in targets:
        normalized_targets.update(_role_term_variants(target))
    for term in role_terms:
        if _role_term_variants(term).intersection(normalized_targets):
            return True
    return False


def _stats_look_numeric(stats: Dict[str, Any]) -> bool:
    if not isinstance(stats, dict) or not stats:
        return False
    for key in ("min", "max", "mean", "stddev", "median", "sum"):
        value = stats.get(key)
        if isinstance(value, (int, float)):
            return True
    return False


def _sample_value_looks_temporal(sample_value: Any) -> bool:
    sample = str(sample_value or "").strip()
    if not sample:
        return False
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", sample):
        return True
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}[ t]\d{2}:\d{2}(?::\d{2})?", sample, flags=re.I):
        return True
    if re.fullmatch(r"\d{2}:\d{2}(?::\d{2})?", sample):
        return True
    return False


@lru_cache(maxsize=4096)
def _column_schema_semantics(table: str, column: str) -> Dict[str, Any]:
    table_l = str(table or "").strip().lower()
    column_s = str(column or "").strip()
    role = str(get_column_role(table_l, column_s) or "").strip().lower() if table_l and column_s else ""
    primary_role = _normalize_semantic_label(_safe_get_column_primary_role(table_l, column_s))
    role_tags = [
        _normalize_semantic_label(tag)
        for tag in _safe_get_column_role_tags(table_l, column_s)
        if _normalize_semantic_label(tag)
    ]
    alias_tags = [
        str(tag or "").strip().lower()
        for tag in _safe_get_column_alias_tags(table_l, column_s)
        if str(tag or "").strip()
    ]
    datatype = _safe_get_column_datatype(table_l, column_s).strip().upper()
    description = _safe_get_column_description(table_l, column_s).strip()
    stats = _safe_get_column_stats(table_l, column_s)

    role_terms = set(role_tags)
    if primary_role:
        role_terms.add(primary_role)
    if role:
        role_terms.add(_normalize_semantic_label(role))

    return {
        "table": table_l,
        "column": column_s,
        "role": role,
        "primary_role": primary_role,
        "role_tags": role_tags,
        "role_terms": role_terms,
        "alias_tags": alias_tags,
        "datatype": datatype,
        "description": description,
        "stats": stats,
    }


def _column_has_identifier_semantics(table: str, column: str) -> bool:
    semantics = _column_schema_semantics(table, column)
    return _structured_role_matches(semantics["role_terms"], _STRUCTURED_IDENTIFIER_ROLE_TERMS)


def _column_has_name_semantics(table: str, column: str) -> bool:
    semantics = _column_schema_semantics(table, column)
    if _structured_role_matches(semantics["role_terms"], _STRUCTURED_NAME_ROLE_TERMS):
        return True
    alias_terms = {
        _normalize_semantic_label(tok)
        for alias in semantics["alias_tags"]
        for tok in re.findall(r"[a-z0-9_]+", alias)
        if _normalize_semantic_label(tok)
    }
    return bool(alias_terms.intersection({"name", "label", "title"}))


def _column_has_temporal_semantics(table: str, column: str) -> bool:
    semantics = _column_schema_semantics(table, column)
    if _structured_role_matches(semantics["role_terms"], _STRUCTURED_TEMPORAL_ROLE_TERMS):
        return True
    if any(tok in semantics["datatype"] for tok in _TEMPORAL_DTYPE_HINTS):
        return True
    return _sample_value_looks_temporal(semantics["stats"].get("sample_value"))


def _column_has_numeric_semantics(table: str, column: str) -> bool:
    semantics = _column_schema_semantics(table, column)
    if _column_has_identifier_semantics(table, column) or _column_has_temporal_semantics(table, column):
        return False
    if _structured_role_matches(semantics["role_terms"], _STRUCTURED_NUMERIC_ROLE_TERMS):
        return True
    if any(tok in semantics["datatype"] for tok in _NUMERIC_DTYPE_HINTS):
        return True
    return _stats_look_numeric(semantics["stats"])


def _column_semantic_tokens(table: str, column: str) -> Set[str]:
    c = str(column or "").strip()
    if not c:
        return set()

    phrases: Set[str] = {_camel_to_words(c)}
    if "_" in c:
        phrases.add(c.lower().replace("_", " "))
    for alias in _column_schema_semantics(table, c)["alias_tags"]:
        alias_s = str(alias or "").strip().lower()
        if alias_s:
            phrases.add(alias_s.replace("_", " "))
            phrases.add(alias_s.replace("-", " "))
    for syn in _safe_get_column_synonyms(table, c):
        s = str(syn).strip().lower()
        if not s:
            continue
        phrases.add(s.replace("_", " "))
        phrases.add(s.replace("-", " "))
    description = re.sub(
        r"\s+",
        " ",
        str(_column_schema_semantics(table, c)["description"] or "").strip().lower().replace("_", " ").replace("-", " "),
    ).strip()
    if description:
        phrases.add(description)
        for piece in re.split(r"\s*(?:,|;|/)\s*|\s+\bor\b\s+", description, flags=re.I):
            piece_norm = re.sub(r"\s+", " ", str(piece or "").strip(" \t\r\n'\"()[]{}"))
            if piece_norm and len(re.findall(r"[a-z0-9]+", piece_norm)) <= 6:
                phrases.add(piece_norm)

    structural_stop_tokens = _FUNCTION_WORD_TOKENS.union({"in", "on", "at", "by"})
    tokens: Set[str] = set()
    for phrase in phrases:
        for spaced_phrase in _phrase_spacing_variants(phrase):
            for tok in _tokenize_for_match(spaced_phrase, stop_tokens=structural_stop_tokens):
                if tok:
                    tokens.add(tok)
        for tok in _tokenize_for_match(phrase, stop_tokens=structural_stop_tokens):
            if tok:
                tokens.add(tok)
    return tokens


def _column_descriptor_tokens(table: str, column: str) -> Set[str]:
    c = str(column or "").strip()
    if not c:
        return set()
    desc = _column_schema_semantics(table, c)["description"]
    if not desc:
        return set()
    desc_norm = _camel_to_words(desc)
    return set(_tokenize_for_match(desc_norm, stop_tokens=_LEXICAL_MATCH_STOP_TOKENS))


_DESCRIPTOR_FAMILY_ROOT_TOKENS = {"description", "detail", "content"}


def _descriptor_family_root_token(token: str) -> str:
    tok = str(token or "").strip().lower()
    if not tok:
        return ""
    for variant in _semantic_token_variants(tok):
        if variant in _DESCRIPTOR_FAMILY_ROOT_TOKENS:
            return variant
    return tok


def _descriptor_family_qualifier_tokens(tokens: Set[str]) -> Set[str]:
    return {
        str(tok or "").strip().lower()
        for tok in (tokens or set())
        if str(tok or "").strip()
        and _descriptor_family_root_token(str(tok or "").strip()) not in _DESCRIPTOR_FAMILY_ROOT_TOKENS
    }


def _descriptor_context_qualifier_tokens(text: str) -> Set[str]:
    text_l = re.sub(r"\s+", " ", str(text or "").strip().lower())
    if not text_l:
        return set()
    contexts: List[str] = []
    for match in re.finditer(r"\b(?:descriptions?|details?|contents?)\b", text_l, flags=re.I):
        start = max(0, match.start() - 48)
        end = min(len(text_l), match.end() + 16)
        if start > 0:
            while start > 0 and not text_l[start - 1].isspace():
                start -= 1
        if end < len(text_l):
            while end < len(text_l) and not text_l[end].isspace():
                end += 1
        context = text_l[start:end].strip()
        if context:
            contexts.append(context)
    if not contexts:
        return set()

    qualifier_tokens: Set[str] = set()
    for context in contexts:
        context_tokens = set(_tokenize_for_match(context, stop_tokens=_LEXICAL_MATCH_STOP_TOKENS))
        qualifier_tokens.update(_descriptor_family_qualifier_tokens(context_tokens))
    return qualifier_tokens


def _is_description_like_column(table: str, column: str) -> bool:
    semantic_tokens = _column_semantic_tokens(table, column)
    descriptor_tokens = _column_descriptor_tokens(table, column)
    merged = semantic_tokens.union(descriptor_tokens)
    return bool(
        {
            _descriptor_family_root_token(tok)
            for tok in merged
            if _descriptor_family_root_token(tok)
        }.intersection(_DESCRIPTOR_FAMILY_ROOT_TOKENS)
    )


def _pick_qualified_description_column(
    text: str,
    table: str,
    allowed: Dict[str, Set[str]],
) -> Optional[str]:
    table_l = str(table or "").strip().lower()
    if not table_l:
        return None
    text_l = re.sub(r"\s+", " ", str(text or "").strip().lower())
    if not text_l or not re.search(r"\b(?:descriptions?|details?|contents?)\b", text_l, flags=re.I):
        return None
    candidates = [
        str(col or "").strip()
        for col in _ordered_allowed_columns(table_l, allowed)
        if str(col or "").strip() and _is_description_like_column(table_l, str(col))
    ]
    if len(candidates) < 2:
        return None

    variant_matches: Dict[str, Tuple[int, int, int]] = {}
    for col in candidates:
        best_col_score: Optional[Tuple[int, int, int]] = None
        for variant in sorted(_column_variants(table_l, col), key=len, reverse=True):
            variant_l = re.sub(r"\s+", " ", str(variant or "").strip().lower())
            if not variant_l:
                continue
            pos = _phrase_pos(text_l, variant_l)
            if pos < 0:
                continue
            variant_tokens = set(_tokenize_for_match(variant_l, stop_tokens=_LEXICAL_MATCH_STOP_TOKENS))
            qualifier_tokens = _descriptor_family_qualifier_tokens(variant_tokens)
            if not qualifier_tokens:
                continue
            score = (len(qualifier_tokens), len(variant_l), -pos)
            if best_col_score is None or score > best_col_score:
                best_col_score = score
        if best_col_score is not None:
            variant_matches[col] = best_col_score
    if len(variant_matches) == 1:
        return next(iter(variant_matches.keys()))
    if len(variant_matches) > 1:
        return None

    query_qualifier_tokens = _descriptor_context_qualifier_tokens(text_l)
    if not query_qualifier_tokens:
        return None

    best_semantic_col: Optional[str] = None
    best_semantic_score: Tuple[int, int] = (-1, -1)
    semantic_tie = False
    for col in candidates:
        column_tokens = _descriptor_family_qualifier_tokens(
            _column_semantic_tokens(table_l, col).union(_column_descriptor_tokens(table_l, col))
        )
        overlap = query_qualifier_tokens.intersection(column_tokens)
        score = (len(overlap), len(column_tokens))
        if score > best_semantic_score:
            best_semantic_col = col
            best_semantic_score = score
            semantic_tie = False
        elif score == best_semantic_score and best_semantic_col and str(best_semantic_col).lower() != str(col).lower():
            semantic_tie = True
    if best_semantic_col and not semantic_tie and best_semantic_score[0] > 0:
        return best_semantic_col
    return None


def _is_identifier_like_column(table: str, column: str) -> bool:
    cl = str(column or "").strip().lower()
    if not cl:
        return False
    semantics = _column_schema_semantics(table, column) if table else None
    if table and _column_has_identifier_semantics(table, column):
        return True
    if cl.endswith(("id", "number", "code", "key")):
        return True
    role = str((semantics or {}).get("role", "") or "").strip().lower()
    return role in {"id", "pk", "fk"}


def _numeric_hint_terms(question_tokens: Optional[Set[str]] = None) -> Set[str]:
    global _NUMERIC_HINT_TERMS_CACHE
    base_hints = (
        set(_NUMERIC_HINT_TERMS_CACHE)
        if _NUMERIC_HINT_TERMS_CACHE is not None
        else None
    )
    if base_hints is None:
        hints: Set[str] = {
            str(tok).lower()
            for tok in INTENT_IR_BASE_NUMERIC_HINTS
            if str(tok).strip()
        }
        try:
            analyzer = get_schema_analyzer()
        except Exception:
            analyzer = None

        try:
            schema_tables = list(get_all_tables() or [])
        except Exception:
            schema_tables = []

        schema_enriched = False
        for table in schema_tables:
            t = str(table or "").strip().lower()
            if not t:
                continue
            try:
                table_columns = list(analyzer.get_table_columns(t)) if analyzer is not None else []
            except Exception:
                table_columns = []
            for col in table_columns:
                c = str(col or "").strip()
                if not c:
                    continue
                c_low = c.lower()
                if c_low.endswith(("id", "number", "code", "key")):
                    continue
                semantic_terms = _column_semantic_tokens(t, c)
                if _column_has_numeric_semantics(t, c) or semantic_terms.intersection(hints):
                    schema_enriched = True
                    hints.update(semantic_terms)

        _NUMERIC_HINT_TERMS_CACHE = set(hints)
        base_hints = set(_NUMERIC_HINT_TERMS_CACHE)

    hints = set(base_hints)
    if question_tokens is not None:
        q_tokens = {str(tok).strip().lower() for tok in question_tokens if str(tok).strip()}
        if not DOMAIN_HEURISTICS_ENABLED:
            return hints
        domain_hits = q_tokens.intersection(INTENT_IR_DOMAIN_NUMERIC_FALLBACK_HINTS)
        schema_or_generic_hits = q_tokens.intersection(hints)
        if domain_hits and not schema_or_generic_hits:
            # Domain-oriented numeric hints remain optional fallbacks; generic
            # numeric cues and schema metadata still drive the main inference path.
            hints.update(domain_hits)
    return hints


def _best_semantic_column_match(
    text: str,
    table: str,
    allowed: Dict[str, Set[str]],
    *,
    numeric_only: bool = False,
    min_score: float = 1.5,
) -> Optional[str]:
    t = (table or "").strip().lower()
    cols = sorted(
        allowed.get(t, set()),
        key=lambda col: (len(str(col or "")), str(col or "").lower()),
    )
    if not cols:
        return None

    query_tokens = _tokenize_for_match(text, stop_tokens=_LEXICAL_MATCH_STOP_TOKENS)
    if not query_tokens:
        return None
    query_token_set = _semantic_query_token_variants(query_tokens)

    best_col = None
    best_score = 0.0
    best_overlap = 0
    for col in cols:
        if numeric_only and not _is_numeric_column(col, t):
            continue
        col_tokens = _column_semantic_tokens(t, col)
        if not col_tokens:
            continue
        overlap = query_token_set.intersection(col_tokens)
        if not overlap:
            continue

        score = float(len(overlap))
        score += (len(overlap) / max(len(col_tokens), 1)) * 0.5
        if any(tok in _numeric_hint_terms(query_token_set) for tok in overlap):
            score += 0.35

        if score > best_score or (score == best_score and best_col is not None and len(col) < len(best_col)):
            best_col = col
            best_score = score
            best_overlap = len(overlap)

    if best_col is None:
        return None
    if best_score < float(min_score) and best_overlap < 2:
        return None
    return best_col


def _best_descriptor_column_match(
    text: str,
    table: str,
    allowed: Dict[str, Set[str]],
    *,
    id_like_only: bool = False,
    min_score: float = 1.0,
) -> Optional[str]:
    t = (table or "").strip().lower()
    cols = sorted(
        allowed.get(t, set()),
        key=lambda col: (len(str(col or "")), str(col or "").lower()),
    )
    if not cols:
        return None

    query_tokens = _tokenize_for_match(text, stop_tokens=_LEXICAL_MATCH_STOP_TOKENS)
    if not query_tokens:
        return None
    query_token_set = _semantic_query_token_variants(query_tokens)

    best_col = None
    best_score = 0.0
    best_overlap = 0
    primary_id = str(_safe_get_id_column(t) or "").strip().lower()
    for col in cols:
        cl = str(col or "").strip().lower()
        is_id_like = _is_identifier_like_column(t, col)
        if id_like_only and not is_id_like:
            continue

        semantic_tokens = _column_semantic_tokens(t, col)
        descriptor_tokens = _column_descriptor_tokens(t, col)
        merged_tokens = semantic_tokens.union(descriptor_tokens)
        overlap = query_token_set.intersection(merged_tokens)
        if not overlap:
            continue

        score = float(len(overlap))
        score += (len(query_token_set.intersection(descriptor_tokens)) * 0.35)
        score += (len(overlap) / max(len(merged_tokens), 1)) * 0.35
        if id_like_only and is_id_like and cl != primary_id:
            score += 0.2

        if score > best_score or (score == best_score and best_col is not None and len(col) < len(best_col)):
            best_col = col
            best_score = score
            best_overlap = len(overlap)

    if best_col is None:
        return None
    if best_score < float(min_score) and best_overlap < 2:
        return None
    return best_col


def _prefer_specific_identifier_match(
    semantic_col: Optional[str],
    descriptor_col: Optional[str],
    table: str,
    text: str = "",
) -> Optional[str]:
    semantic = str(semantic_col or "").strip()
    descriptor = str(descriptor_col or "").strip()
    if descriptor and not semantic:
        return descriptor
    if semantic and not descriptor:
        return semantic
    if not semantic and not descriptor:
        return None
    if semantic.lower() == descriptor.lower():
        return semantic

    primary_id = str(_safe_get_id_column(table) or "").strip().lower()
    semantic_is_primary = bool(primary_id) and semantic.lower() == primary_id
    descriptor_is_primary = bool(primary_id) and descriptor.lower() == primary_id
    if semantic_is_primary and not descriptor_is_primary and _is_identifier_like_column(table, descriptor):
        query_tokens = set(_tokenize_for_match(text, stop_tokens=_LEXICAL_MATCH_STOP_TOKENS))
        descriptor_tokens = _column_semantic_tokens(table, descriptor).union(_column_descriptor_tokens(table, descriptor))
        primary_tokens = _column_semantic_tokens(table, semantic).union(_column_descriptor_tokens(table, semantic))
        # Keep only truly generic identifier words here. Schema-wide head tokens
        # can include relationship cues extracted from metadata-driven table
        # synonyms (e.g. "manager"), which are exactly the words we need to
        # preserve when preferring a specific FK-like identifier over the
        # subject table's primary ID column.
        generic_identifier_tokens = set(_GENERIC_IDENTIFIER_TOKENS)
        specific_relational_tokens = (descriptor_tokens - primary_tokens) - generic_identifier_tokens
        if query_tokens.intersection(specific_relational_tokens):
            return descriptor
    return semantic


def _extract_projection_segments(question: str) -> List[str]:
    q_raw = str(question or "").strip()
    if not q_raw:
        return []

    matches = list(
        re.finditer(
            r"\b(?:show|display|return|list|include)\b\s+(.+?)(?=\b(?:show|display|return|list|include)\b|[.;!?]|$)",
            q_raw,
            flags=re.I,
        )
    )
    if not matches:
        return []

    tail = str(matches[-1].group(1) or "").strip()
    if not tail:
        return []
    tail = re.split(
        r"\b(?:sort(?:ed)?|order(?:ed)?)\s+(?:by|from|alphabetically|chronologically)\b",
        tail,
        maxsplit=1,
        flags=re.I,
    )[0].strip(" \t\r\n,.;:!?")
    if not tail:
        return []
    if "," not in tail and not re.search(r"\band\b", tail, flags=re.I):
        return []

    segments: List[str] = []
    for raw_seg in re.split(r"\s*,\s*|\s+\band\b\s+|\s+\bor\b\s+", tail, flags=re.I):
        seg = str(raw_seg or "").strip(" \t\r\n.,;:!?")
        if not seg:
            continue
        seg = re.sub(
            r"^(?:their|the|all|any|its|his|her|our|your)\s+",
            "",
            seg,
            flags=re.I,
        ).strip()
        seg = re.sub(
            r"^(?:and|or|then)\s+",
            "",
            seg,
            flags=re.I,
        ).strip()
        if seg:
            segments.append(seg)
    return segments


def _choose_subject(
    question: str,
    allowed: Dict[str, Set[str]],
    table_mentions: List[Tuple[str, int]],
    primary_table: str = "",
) -> str:
    p = (primary_table or "").strip().lower()
    if p and p in allowed:
        return p

    if len(table_mentions) >= 2:
        projection_segments = _extract_projection_segments(question)
        if projection_segments:
            projection_scores: List[Tuple[int, int, str]] = []
            for table, pos in table_mentions:
                score = 0
                for seg in projection_segments:
                    if _best_semantic_column_match(seg, table, allowed, min_score=1.0):
                        score += 1
                projection_scores.append((score, -pos, table))
            projection_scores.sort(reverse=True)
            if projection_scores:
                best_score, _, best_table = projection_scores[0]
                next_score = projection_scores[1][0] if len(projection_scores) > 1 else 0
                if best_score >= 2 and best_score > next_score:
                    return best_table

    if table_mentions:
        return sorted(table_mentions, key=lambda x: x[1])[0][0]

    if len(allowed) == 1:
        return next(iter(allowed.keys()))

    # Prefer dimension-like table when nothing explicit is mentioned.
    scored: List[Tuple[int, str]] = []
    for t in allowed:
        typ = _safe_get_table_type(t)
        score = 0
        if typ == "dimension":
            score += 5
        elif typ == "reference":
            score += 2
        if _safe_get_name_column(t):
            score += 1
        if _safe_get_id_column(t):
            score += 1
        scored.append((score, t))
    if scored:
        scored.sort(reverse=True)
        return scored[0][1]

    return ""


def _detect_table_mentions(question: str, tables: Sequence[str]) -> List[Tuple[str, int]]:
    q = (question or "").lower()
    ranked: List[Tuple[str, int, int]] = []
    for t in tables:
        best = -1
        best_len = -1
        for v in _table_variants(t):
            pos = _phrase_pos(q, v)
            cur_len = len(str(v or "").strip())
            if pos >= 0 and (
                best < 0
                or pos < best
                or (pos == best and cur_len > best_len)
            ):
                best = pos
                best_len = cur_len
        if best >= 0:
            ranked.append((t.lower(), best, best_len))
    ranked.sort(key=lambda x: (x[1], -x[2], x[0]))
    return [(t, pos) for (t, pos, _) in ranked]


def _detect_column_mentions(question: str, allowed: Dict[str, Set[str]]) -> List[Tuple[str, str, int]]:
    q = (question or "").lower()
    out: List[Tuple[str, str, int]] = []
    for table, _ in _ordered_allowed_tables(allowed):
        preferred_description_col = _pick_qualified_description_column(question, table, allowed)
        table_variants = {
            re.sub(r"\s+", " ", str(v or "").strip().lower())
            for v in (_table_variants(table) or {table})
            if str(v or "").strip()
        }
        for col in _ordered_allowed_columns(table, allowed):
            if (
                preferred_description_col
                and _is_description_like_column(table, col)
                and str(col or "").strip().lower() != str(preferred_description_col or "").strip().lower()
            ):
                continue
            best = -1
            best_variant = ""
            for v in _column_variants(table, col):
                pos = _phrase_pos(q, v)
                if pos >= 0 and (best < 0 or pos < best):
                    best = pos
                    best_variant = re.sub(r"\s+", " ", str(v or "").strip().lower())
            if best >= 0:
                if (
                    table
                    and _column_has_name_semantics(table, col)
                    and best_variant
                    and best_variant in table_variants
                ):
                    continue
                out.append((table, col, best))
    out.sort(key=lambda x: x[2])
    return out


def _find_best_date_column(table: str, allowed: Dict[str, Set[str]]) -> Optional[str]:
    cols = list(allowed.get((table or "").lower(), set()))
    date_like = [c for c in cols if _is_date_column(c, table)]
    if not date_like:
        return None
    date_like.sort(key=lambda x: (0 if "date" in x.lower() else 1, len(x)))
    return date_like[0]


def _best_id_column(table: str, allowed: Dict[str, Set[str]], question: str = "") -> Optional[str]:
    t = (table or "").lower()
    cols = list(allowed.get(t, set()))
    if not cols:
        return None

    table_root = (_safe_get_singular_form(t) or t).replace("_", "").lower()
    strict_targets = {
        f"{table_root}id",
        f"{table_root}number",
        f"{table_root}code",
    }

    q_low = (question or "").lower()
    _GENERIC_SUFFIXES = {"number", "id", "code", "key", "num"}

    def _mentioned_bonus(col: str) -> int:
        """Prefer columns whose distinguishing camelCase words appear in the question."""
        if not q_low:
            return 0
        tokens = _camel_to_words(col or "").lower().split()
        # Keep only semantically meaningful words (drop generic ID suffixes)
        significant = [w for w in tokens if w and w not in _GENERIC_SUFFIXES]
        if not significant:
            return 0
        return 1 if any(w in q_low for w in significant) else 0

    def score(col: str) -> int:
        cl = (col or "").lower().replace("_", "")
        s = 0
        if cl in strict_targets:
            s += 8
        if table_root and table_root in cl:
            s += 5
        if cl.endswith("number"):
            s += 3
        if cl.endswith("id"):
            s += 2
        if cl.endswith("code"):
            s += 1
        return s

    id_like = [
        c for c in cols
        if c.lower().endswith("id") or c.lower().endswith("number") or c.lower().endswith("code")
    ]
    if id_like:
        id_like.sort(key=lambda c: (score(c), _mentioned_bonus(c), -len(c)), reverse=True)
        return id_like[0]

    id_col = _safe_get_id_column(table)
    if id_col and id_col in cols:
        return id_col
    return None


def _best_name_column(table: str, allowed: Dict[str, Set[str]]) -> Optional[str]:
    name_col = _safe_get_name_column(table)
    if name_col and name_col in allowed.get((table or "").lower(), set()):
        return name_col
    cols = list(allowed.get((table or "").lower(), set()))
    name_like = [c for c in cols if _column_has_name_semantics(table, c)]
    if not name_like:
        name_like = [c for c in cols if c.lower().endswith("name") or "name" in c.lower()]
    if name_like:
        name_like.sort(
            key=lambda c: (
                1 if _column_has_name_semantics(table, c) else 0,
                -len(_column_semantic_tokens(table, c).intersection({"name", "entity", "label", "title"})),
                -len(_column_schema_semantics(table, c)["alias_tags"]),
                -len(str(c or "")),
            ),
            reverse=True,
        )
        return name_like[0]

    def _normalized_identifier(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())

    subject_keys = {
        _normalized_identifier(table),
        _normalized_identifier(_safe_get_singular_form(table)),
    }
    subject_keys.update(
        _normalized_identifier(variant)
        for variant in _table_variants(table)
        if _normalized_identifier(variant)
    )
    subject_keys.discard("")

    subject_aligned = [
        c for c in cols
        if _normalized_identifier(c) in subject_keys
        or _normalized_identifier(_camel_to_words(c)) in subject_keys
    ]
    if subject_aligned:
        subject_aligned.sort(
            key=lambda c: (
                1 if not _is_numeric_column(c, table) and not _is_date_column(c, table) else 0,
                0 if _column_has_identifier_semantics(table, c) else 1,
                -len(str(c or "")),
            ),
            reverse=True,
        )
        return subject_aligned[0]
    return None


def _best_name_component_column_match(
    text: str,
    table: str,
    allowed: Dict[str, Set[str]],
) -> Optional[str]:
    t = str(table or "").strip().lower()
    text_l = re.sub(r"\s+", " ", str(text or "").strip().lower())
    if not t or not text_l:
        return None

    best_col: Optional[str] = None
    best_pos = -1
    best_len = -1
    for col in sorted(allowed.get(t, set()), key=lambda c: (len(str(c or "")), str(c or "").lower())):
        if not _column_has_name_semantics(t, col):
            continue
        for variant in sorted(_column_reference_variants(t, col), key=len, reverse=True):
            variant_l = str(variant or "").strip().lower()
            if not variant_l:
                continue
            pos = _phrase_pos(text_l, variant_l)
            if pos < 0:
                continue
            if pos > best_pos or (pos == best_pos and len(variant_l) > best_len):
                best_col = col
                best_pos = pos
                best_len = len(variant_l)
                break
    return best_col


def _find_best_numeric_column(table: str, allowed: Dict[str, Set[str]]) -> Optional[str]:
    cols = list(allowed.get((table or "").lower(), set()))
    num_like = [c for c in cols if _is_numeric_column(c, table)]
    if not num_like:
        return None
    num_like.sort(key=lambda x: len(x))
    return num_like[0]


def _prefer_numeric_column_from_question(
    question: str,
    subject: str,
    allowed: Dict[str, Set[str]],
) -> Optional[str]:
    return _best_semantic_column_match(
        text=question,
        table=subject,
        allowed=allowed,
        numeric_only=True,
    )


def _table_entity_pattern(table: str) -> str:
    alts = sorted((_table_variants(table) or {table}), key=len, reverse=True)
    return "|".join(re.escape(a).replace(r"\ ", r"\s+") for a in alts if a)


def _wants_distinct_projection(question: str) -> bool:
    q = (question or "").lower()
    if not q:
        return False

    hard_signals = (
        "distinct",
        "no duplicate",
        "without duplicates",
        "tanpa duplikat",
        "tanpa duplicate",
        "hanya yang unik",
        "dedup",
        "de-dup",
    )
    if any(s in q for s in hard_signals):
        return True

    if "unique" in q and any(tok in q for tok in ("row", "rows", "record", "records", "result", "results", "value", "values")):
        return True

    implicit_patterns = (
        r"\bone\s+(?:row|record|result)\s+(?:per|for\s+each)\b",
        r"\beach\s+[a-z0-9_]+\s+only\s+once\b",
        r"\bonly\s+once\s+per\b",
    )
    return any(re.search(pat, q, flags=re.I) for pat in implicit_patterns)


def _wants_count_distinct(question: str) -> bool:
    q = (question or "").lower()
    if not q:
        return False
    if not re.search(r"\b(how\s+many|number\s+of|count|total\s+number)\b", q):
        return False
    if any_phrase_in_text(q, aggregation_count_distinct_phrases()):
        return True
    return any_phrase_in_text(q, aggregation_count_distinct_keywords())


def _detect_metrics(
    question: str,
    qtype: str,
    subject: str,
    allowed: Dict[str, Set[str]],
    table_mentions: List[Tuple[str, int]],
    column_mentions: List[Tuple[str, str, int]],
) -> List[IRMetric]:
    q = (question or "").lower()
    metrics: List[IRMetric] = []

    wants_count = bool(re.search(r"\b(how\s+many|number\s+of|count|total\s+number)\b", q))
    has_explicit_sum_signal = bool(re.search(r"\b(?:sum(?:\s+of)?|jumlah)\b", q))
    wants_avg = bool(re.search(r"\b(avg|average|mean|rata)\b", q))
    has_explicit_max = bool(re.search(r"\bmax(?:imum)?\b", q))
    has_explicit_min = bool(re.search(r"\bmin(?:imum)?\b", q))
    has_upper_superlative = bool(re.search(r"\b(highest|largest)\b", q))
    has_lower_superlative = bool(re.search(r"\b(lowest|smallest)\b", q))
    metric_body_q = re.split(
        r"\b(?:sort(?:ed)?\s+by|order(?:ed)?\s+by)\b",
        q,
        maxsplit=1,
        flags=re.I,
    )[0]
    has_explicit_max_phrase = bool(
        re.search(r"\b(?:max(?:imum)?|highest|largest)\s+[a-z]", metric_body_q, flags=re.I)
    )
    has_explicit_min_phrase = bool(
        re.search(r"\b(?:min(?:imum)?|lowest|smallest)\s+[a-z]", metric_body_q, flags=re.I)
    )
    directional_superlative_sort = bool(
        re.search(
            r"\b(?:order(?:ed)?|sort(?:ed)?)\b[^.]{0,120}\b(?:highest|largest|lowest|smallest)\b",
            q,
            flags=re.I,
        )
        or re.search(
            r"\b(?:highest|largest)\b[^.]{0,40}\b(?:to|-)\b[^.]{0,40}\b(?:lowest|smallest)\b",
            q,
            flags=re.I,
        )
        or re.search(
            r"\b(?:lowest|smallest)\b[^.]{0,40}\b(?:to|-)\b[^.]{0,40}\b(?:highest|largest)\b",
            q,
            flags=re.I,
        )
    )
    wants_max = has_explicit_max or has_explicit_max_phrase or (has_upper_superlative and not directional_superlative_sort)
    wants_min = has_explicit_min or has_explicit_min_phrase or (has_lower_superlative and not directional_superlative_sort)
    wants_percentage = bool(re.search(r"\b(percent|percentage|ratio|rate)\b|%\s*of", q))
    wants_count_distinct = _wants_count_distinct(q)

    temporal_direction_only = bool(
        re.search(
            r"\b(?:ordered|sorted?)\b[^.]{0,80}\b(?:latest|newest|earliest|oldest)\b",
            q,
            flags=re.I,
        )
        or re.search(r"\b(?:latest|newest)\b[^.]{0,24}\b(?:earliest|oldest)\b", q, flags=re.I)
        or re.search(r"\b(?:earliest|oldest)\b[^.]{0,24}\b(?:latest|newest)\b", q, flags=re.I)
    )
    has_temporal_metric_phrase = bool(
        re.search(
            r"\b(?:earliest|oldest|latest|newest|most\s+recent)\s+"
            r"(?:[a-z][a-z0-9_]*\s+){0,3}"
            r"(?:date|time|timestamp|day|month|year)\b",
            q,
            flags=re.I,
        )
    )
    if has_temporal_metric_phrase and not temporal_direction_only:
        if re.search(r"\b(?:earliest|oldest)\b", q, flags=re.I):
            wants_min = True
        if re.search(r"\b(?:latest|newest|most\s+recent)\b", q, flags=re.I):
            wants_max = True

    # Rank-style queries are handled by legacy pipeline; keep IR conservative.
    if (qtype or "").lower() == "rank":
        return metrics

    mentioned_tables = [t for t, _ in table_mentions]
    first_mentioned = mentioned_tables[0] if mentioned_tables else subject

    column_mentioned_tables = [t for t, _, _ in column_mentions]
    metric_target_tables = list(
        dict.fromkeys(mentioned_tables + column_mentioned_tables + [subject])
    )
    if not metric_target_tables:
        metric_target_tables = [subject]

    def _resolve_table_from_phrase(phrase: str) -> Optional[str]:
        p = (phrase or "").lower().strip()
        if not p:
            return None
        phrase_tokens = set(re.findall(r"[a-z0-9]+", p))
        if not phrase_tokens:
            return None

        best_table: Optional[str] = None
        best_score = 0
        candidates = list(dict.fromkeys(mentioned_tables + list(allowed.keys())))
        for t in candidates:
            variants = _table_variants(t) or {t}
            table_best = 0
            for v in variants:
                vv = str(v or "").strip().lower()
                if not vv:
                    continue
                pat = re.escape(vv).replace(r"\ ", r"\s+")
                if re.search(rf"\b{pat}\b", p, flags=re.I):
                    return t
                v_tokens = set(re.findall(r"[a-z0-9]+", vv))
                if not v_tokens:
                    continue
                table_best = max(table_best, len(v_tokens.intersection(phrase_tokens)))
            if table_best > best_score:
                best_score = table_best
                best_table = t
        return best_table if best_score > 0 else None

    def _is_numeric_measure_column(table: str, col: str) -> bool:
        # Numeric aggregates should not bind to temporal columns unless the
        # caller explicitly takes the temporal-extrema path.
        return _is_numeric_column(col, table) and not _is_date_column(col, table)

    def _find_best_numeric_measure_column(table: str) -> Optional[str]:
        cols = list(allowed.get((table or "").lower(), set()))
        num_like = [c for c in cols if _is_numeric_measure_column(table, c)]
        if not num_like:
            return None
        num_like.sort(key=lambda x: len(x))
        return num_like[0]

    # Candidate numeric/date mentions from question.
    mentioned_numeric_cols = [
        (t, c, p) for (t, c, p) in column_mentions if _is_numeric_measure_column(t, c)
    ]
    mentioned_date_cols = [
        (t, c, p) for (t, c, p) in column_mentions if _is_date_column(c, t)
    ]

    def add_metric(m: IRMetric) -> None:
        key = (m.agg, m.table.lower(), m.column.lower(), m.alias.lower(), bool(m.distinct))
        for ex in metrics:
            if (ex.agg, ex.table.lower(), ex.column.lower(), ex.alias.lower(), bool(ex.distinct)) == key:
                return
        metrics.append(m)

    def _resolve_column_from_phrase(phrase: str) -> Optional[Tuple[str, str]]:
        p = (phrase or "").lower().strip()
        if not p:
            return None
        p_tokens = set(_semantic_phrase_tokens(p))
        if not p_tokens:
            return None

        candidates = list(column_mentions or [])
        seen_candidates = {(t, c) for t, c, _ in candidates}
        for t, _ in _ordered_allowed_tables(allowed):
            for c in _ordered_allowed_columns(t, allowed):
                if (t, c) not in seen_candidates:
                    candidates.append((t, c, 10**9))
                    seen_candidates.add((t, c))

        best: Optional[Tuple[str, str]] = None
        best_score = 0
        best_pos = 10**9
        for t, c, pos in candidates:
            score = 0
            for v in _column_variants(t, c):
                v_tokens = set(_semantic_phrase_tokens(v))
                if not v_tokens:
                    continue
                overlap = len(v_tokens.intersection(p_tokens))
                score = max(score, overlap)
            if score <= 0:
                continue
            prefer = 1 if t == subject else 0
            if (
                score > best_score
                or (score == best_score and prefer and best is not None and best[0] != subject)
                or (score == best_score and prefer == (1 if best and best[0] == subject else 0) and pos < best_pos)
            ):
                best = (t, c)
                best_score = score
                best_pos = pos
        return best if best_score > 0 else None

    def _extract_local_metric_phrase(kind: str) -> str:
        """
        Bind each aggregate to its own nearby noun phrase. This avoids using the
        first numeric mention for every metric in questions like
        "total X, average Y".
        """
        if kind == "sum":
            patterns = (
                r"\b(?:sum(?:\s+of)?|total)\s+([a-z][a-z0-9_ ]{1,64}?)(?=\s*(?:,|;|\.|\band\b|\bthen\b|\bfor\b|\bby\b|$))",
            )
        elif kind == "avg":
            patterns = (
                r"\b(?:avg|average|mean)\s+([a-z][a-z0-9_ ]{1,64}?)(?=\s*(?:,|;|\.|\band\b|\bthen\b|\bfor\b|\bby\b|$))",
            )
        elif kind == "max":
            patterns = (
                r"\b(?:max(?:imum)?|highest|largest)\s+([a-z][a-z0-9_ ]{1,64}?)(?=\s*(?:,|;|\.|\band\b|\bthen\b|\bfor\b|\bby\b|$))",
            )
        elif kind == "min":
            patterns = (
                r"\b(?:min(?:imum)?|lowest|smallest)\s+([a-z][a-z0-9_ ]{1,64}?)(?=\s*(?:,|;|\.|\band\b|\bthen\b|\bfor\b|\bby\b|$))",
            )
        else:
            patterns = ()

        for pat in patterns:
            m = re.search(pat, q, flags=re.I)
            if not m:
                continue
            phrase = re.sub(r"\s+", " ", str(m.group(1) or "").strip())
            phrase = re.sub(
                r"^(?:the|their|its|a|an)\s+",
                "",
                phrase,
                flags=re.I,
            ).strip()
            if phrase:
                return phrase
        return ""

    def _resolve_numeric_measure_from_phrase(phrase: str) -> Optional[Tuple[str, str]]:
        p = re.sub(r"\s+", " ", str(phrase or "").strip())
        if not p:
            return None
        p_norm = p.lower()

        explicit_variant_hits: List[Tuple[int, int, str, str]] = []
        for t in metric_target_tables:
            for c in _ordered_allowed_columns(t, allowed):
                if not _is_numeric_measure_column(t, c):
                    continue
                best_variant_score = 0
                for v in _column_variants(t, c):
                    v_norm = re.sub(r"\s+", " ", str(v or "").strip().lower())
                    if not v_norm:
                        continue
                    v_pat = re.escape(v_norm).replace(r"\ ", r"\s+")
                    if p_norm == v_norm:
                        best_variant_score = max(best_variant_score, len(v_norm.split()) + 10)
                    elif re.search(rf"\b{v_pat}\b", p_norm, flags=re.I):
                        best_variant_score = max(best_variant_score, len(v_norm.split()) + 6)
                if best_variant_score > 0:
                    explicit_variant_hits.append(
                        (
                            best_variant_score,
                            1 if t == subject else 0,
                            t,
                            c,
                        )
                    )
        if explicit_variant_hits:
            explicit_variant_hits.sort(key=lambda item: (-item[0], -item[1], item[2].lower(), len(item[3])))
            _, _, best_t, best_c = explicit_variant_hits[0]
            return best_t, best_c

        metric_phrase_stop_tokens = set(_LEXICAL_MATCH_STOP_TOKENS)
        metric_phrase_stop_tokens.discard("list")
        phrase_tokens = {
            tok for tok in _semantic_phrase_tokens(p)
            if tok and tok not in metric_phrase_stop_tokens
        }
        best: Optional[Tuple[str, str]] = None
        best_score = 0.0
        best_pos = 10**9
        for t in metric_target_tables:
            for c in _ordered_allowed_columns(t, allowed):
                if not _is_numeric_measure_column(t, c):
                    continue
                col_tokens = _column_semantic_tokens(t, c)
                if not col_tokens:
                    continue
                overlap = phrase_tokens.intersection(col_tokens)
                if not overlap:
                    continue
                pos = next((p0 for tm, cm, p0 in column_mentions if tm == t and cm == c), 10**9)
                score = float(len(overlap) * 10)
                score += (len(overlap) / max(len(col_tokens), 1)) * 2.0
                score += 2.0 if phrase_tokens.issubset(col_tokens) else 0.0
                score += 1.0 if t == subject else 0.0
                if score > best_score or (score == best_score and pos < best_pos):
                    best = (t, c)
                    best_score = score
                    best_pos = pos

        if best is not None and best_score >= 10.0:
            return best

        resolved = _resolve_column_from_phrase(p)
        if resolved:
            t, c = resolved
            if _is_numeric_measure_column(t, c):
                return t, c

        return None

    def _question_requests_total_metric() -> bool:
        if not re.search(r"\btotal\b", q, flags=re.I):
            return False
        if re.search(r"\btotal\s+(?:number|count)(?:\s+of)?\b", q, flags=re.I):
            return False

        candidate_phrases = [
            str(m.group(1) or "").strip()
            for m in re.finditer(r"\btotal\s+([a-z][a-z0-9_ ]{1,48})\b", q, flags=re.I)
        ]
        candidate_phrases = [p for p in candidate_phrases if p]
        if not candidate_phrases:
            candidate_phrases = [question]

        for phrase in candidate_phrases:
            matched_numeric = _resolve_numeric_measure_from_phrase(phrase)
            if matched_numeric:
                return True

            resolved_table = _resolve_table_from_phrase(phrase)
            resolved_col = _resolve_column_from_phrase(phrase)
            if resolved_table and not (
                resolved_col and _is_numeric_measure_column(resolved_col[0], resolved_col[1])
            ):
                continue

            for table_name in metric_target_tables:
                matched = _best_semantic_column_match(
                    text=phrase,
                    table=table_name,
                    allowed=allowed,
                    numeric_only=True,
                    min_score=1.0,
                )
                if matched:
                    return True

        return False

    def _question_requests_total_row_count() -> Optional[str]:
        """
        Detect `total <entity>` phrases that are better interpreted as row counts
        rather than numeric sums.

        Examples:
        - "total orders"
        - "total payments"

        Returns the best target table when the phrase clearly points to an
        entity/table and does not resolve to a numeric measure column.
        """
        if not re.search(r"\btotal\b", q, flags=re.I):
            return None
        if re.search(r"\btotal\s+(?:number|count)(?:\s+of)?\b", q, flags=re.I):
            return None

        candidate_phrases = [
            str(m.group(1) or "").strip()
            for m in re.finditer(r"\btotal\s+([a-z][a-z0-9_ ]{1,48})\b", q, flags=re.I)
        ]
        candidate_phrases = [p for p in candidate_phrases if p]
        if not candidate_phrases:
            return None

        row_unit_terms = {
            "row", "rows", "record", "records", "entry", "entries",
            "item", "items", "line", "lines", "detail", "details",
            "result", "results",
        }

        for phrase in candidate_phrases:
            phrase_norm = re.sub(r"\s+", " ", str(phrase or "").strip().lower())
            if not phrase_norm:
                continue

            phrase_tokens = [
                tok for tok in re.findall(r"[a-z0-9_'-]+", phrase_norm) if tok
            ]
            if phrase_tokens and all(tok in row_unit_terms for tok in phrase_tokens):
                return subject or first_mentioned

            resolved_numeric = _resolve_numeric_measure_from_phrase(phrase_norm)
            if resolved_numeric:
                continue

            resolved_col = _resolve_column_from_phrase(phrase_norm)
            if resolved_col and _is_numeric_measure_column(resolved_col[0], resolved_col[1]):
                continue

            resolved_table = _resolve_table_from_phrase(phrase_norm)
            if resolved_table:
                return resolved_table

        return None

    def _semantic_phrase_tokens(text: str) -> List[str]:
        out: List[str] = []
        for tok in re.findall(r"[a-z0-9_'-]+", str(text or "").lower()):
            for cur in _semantic_token_variants(tok):
                if cur:
                    out.append(cur)
        return out

    total_row_count_table = _question_requests_total_row_count()
    if total_row_count_table:
        wants_count = True

    def _question_requests_subject_row_count() -> bool:
        if not subject:
            return False

        row_unit_terms = {
            "row", "rows", "record", "records", "entry", "entries",
            "item", "items", "line", "lines", "detail", "details",
        }

        phrase_patterns = (
            r"(?:^|[,.]|\b(?:show|return|include|display|list|and)\b)\s*"
            r"(?:the\s+)?([a-z][a-z0-9_ ]{0,48}?)\s+counts?\b",
            r"\bcounts?\s+(?:of\s+)?([a-z][a-z0-9_ ]{1,48}?)(?=\s*(?:,|;|\.|\band\b|\bthen\b|\bfor\b|\bby\b|$))",
        )
        for pat in phrase_patterns:
            for match in re.finditer(pat, q, flags=re.I):
                phrase = re.sub(r"\s+", " ", str(match.group(1) or "").strip().lower())
                phrase = re.sub(r"^(?:the|a|an|total)\s+", "", phrase, flags=re.I).strip()
                if not phrase:
                    continue

                tokens = [tok for tok in re.findall(r"[a-z0-9_'-]+", phrase) if tok]
                if tokens and all(tok in row_unit_terms for tok in tokens):
                    return True

                resolved_table = _resolve_table_from_phrase(phrase)
                if resolved_table and resolved_table.lower() == subject.lower():
                    return True

        return False

    subject_row_count_requested = _question_requests_subject_row_count()
    if subject_row_count_requested:
        wants_count = True

    wants_sum = bool(has_explicit_sum_signal or _question_requests_total_metric())

    def _count_distinct_container_terms() -> Set[str]:
        """
        Build container nouns for COUNT DISTINCT phrase cleanup.
        Keep "order/orders" as generic only when they are not schema entities.
        """
        generic_terms = {
            "record", "records", "row", "rows", "entry", "entries",
            "result", "results", "dataset", "data", "year", "years",
        }
        schema_entity_tokens: Set[str] = set()
        candidate_tables: Set[str] = {
            str(t or "").strip().lower()
            for t in (mentioned_tables or [])
            if str(t or "").strip()
        }
        candidate_tables.update(
            str(t or "").strip().lower()
            for t in (allowed or {}).keys()
            if str(t or "").strip()
        )
        if not candidate_tables:
            try:
                candidate_tables.update(
                    str(t or "").strip().lower()
                    for t in (get_all_tables() or [])
                    if str(t or "").strip()
                )
            except Exception:
                pass

        for table_name in candidate_tables:
            table_variants = _table_variants(table_name) or {table_name}
            for variant in table_variants:
                schema_entity_tokens.update(
                    tok for tok in re.findall(r"[a-z0-9_'-]+", str(variant or "").lower()) if tok
                )

        terms = set(generic_terms)
        if "order" not in schema_entity_tokens and "orders" not in schema_entity_tokens:
            terms.update({"order", "orders"})
        return terms

    def _count_distinct_phrase_candidates(phrase: str) -> List[str]:
        raw = str(phrase or "").strip()
        if not raw:
            return []

        candidates: List[str] = []

        def _add(text: str) -> None:
            txt = " ".join(str(text or "").strip().split())
            if txt and txt not in candidates:
                candidates.append(txt)

        _add(raw)

        lowered = raw.lower()
        generic_container_terms = _count_distinct_container_terms()
        tail_container_terms = {
            t for t in generic_container_terms
            if t not in {"year", "years"}
        }
        tail_container_alt = "|".join(
            re.escape(t)
            for t in sorted(tail_container_terms, key=len, reverse=True)
        ) or r"records|rows|entries|results|dataset|data"
        stop_patterns = (
            r"\b(?:seen|found|present|appearing|appears|occurring|occurs|recorded)\b.*$",
            r"\b(?:in|within|for|from|during|across|among)\s+"
            r"(?:that|this|those|these|the|all|each|every|same)\b.*$",
            r"\b(?:in|within|for|from|during|across|among)\s+"
            rf"(?:[a-z][a-z0-9_'-]*\s+){{0,4}}(?:{tail_container_alt})\b.*$",
        )
        for pat in stop_patterns:
            trimmed = re.sub(pat, "", lowered, flags=re.I).strip(" ,.;:-")
            if trimmed:
                _add(trimmed)

        tokens = [tok for tok in re.findall(r"[a-z0-9_'-]+", lowered) if tok]
        content_tokens = [tok for tok in tokens if tok not in generic_container_terms]
        if content_tokens:
            _add(" ".join(content_tokens))
            singular_content: List[str] = []
            for tok in content_tokens:
                variants = [v for v in _semantic_token_variants(tok) if v != tok]
                singular_content.append(variants[0] if variants else tok)
            _add(" ".join(singular_content))
        if len(tokens) >= 2:
            _add(" ".join(tokens[:2]))
        if len(tokens) >= 3:
            _add(" ".join(tokens[:3]))

        return candidates

    def _resolve_count_distinct_column_from_phrase(phrase: str) -> Optional[Tuple[str, str]]:
        raw_tokens = _semantic_phrase_tokens(phrase)
        if not raw_tokens:
            return None

        generic_container_terms = _count_distinct_container_terms()
        raw_token_set = set(raw_tokens)
        content_tokens = {
            tok for tok in raw_tokens
            if tok not in generic_container_terms and tok not in _FUNCTION_WORD_TOKENS
        }
        target_tokens = content_tokens or raw_token_set
        if not target_tokens:
            return None

        candidates = column_mentions or []
        if not candidates:
            for t, _ in _ordered_allowed_tables(allowed):
                for c in _ordered_allowed_columns(t, allowed):
                    candidates.append((t, c, 10**9))

        best: Optional[Tuple[str, str]] = None
        best_score = float("-inf")
        best_pos = 10**9
        for t, c, pos in candidates:
            variant_tokens: Set[str] = set()
            for v in _column_variants(t, c):
                variant_tokens.update(_semantic_phrase_tokens(v))
            if not variant_tokens:
                continue

            content_overlap = len(variant_tokens.intersection(target_tokens))
            all_overlap = len(variant_tokens.intersection(raw_token_set))
            if content_overlap <= 0 and all_overlap <= 0:
                continue

            exact_content = 1 if target_tokens.issubset(variant_tokens) else 0
            exact_raw = 1 if raw_token_set.issubset(variant_tokens) else 0
            prefer_subject = 1 if t == subject else 0
            id_like_penalty = 2 if (
                content_overlap == 0
                and str(c or "").lower().endswith(("number", "id", "code"))
            ) else 0

            score = (
                (content_overlap * 10)
                + (all_overlap * 2)
                + (exact_content * 6)
                + (exact_raw * 2)
                + prefer_subject
                - id_like_penalty
            )
            if score > best_score or (
                score == best_score
                and prefer_subject == (1 if best and best[0] == subject else 0)
                and pos < best_pos
            ):
                best = (t, c)
                best_score = score
                best_pos = pos

        return best if best_score > 0 else None

    def _pick_count_distinct_target() -> Optional[Tuple[str, str]]:
        if not wants_count_distinct:
            return None

        m_phrase = re.search(
            r"\b(?:how\s+many|number\s+of|count|total\s+number(?:\s+of)?)\s+(?:the\s+)?"
            r"(?:distinct|unique|different)\s+([a-z][a-z0-9_ ]{1,40})\b",
            q,
            flags=re.I,
        )
        if m_phrase:
            phrase = (m_phrase.group(1) or "").strip()
            for phrase_candidate in _count_distinct_phrase_candidates(phrase):
                resolved_col = _resolve_count_distinct_column_from_phrase(phrase_candidate)
                if not resolved_col:
                    resolved_col = _resolve_column_from_phrase(phrase_candidate)
                if resolved_col:
                    return resolved_col
                resolved_table = _resolve_table_from_phrase(phrase_candidate)
                if resolved_table:
                    id_col = _best_id_column(resolved_table, allowed, question) or "*"
                    return (resolved_table, id_col)

        for t, c, pos in column_mentions:
            window_start = max(0, pos - 48)
            prefix = q[window_start:pos]
            if re.search(r"\b(distinct|unique|different)\b", prefix, flags=re.I):
                return (t, c)

        return None

    def _pick_percentage_nonnull_target() -> Optional[Tuple[str, str]]:
        # Pattern A: "non-null shipped date"
        m_nonnull_after = re.search(
            r"\b(?:non[-\s]?null|not\s+null)\s+([a-z][a-z0-9_ ]{0,40}?)(?:[.,;]|$)",
            q,
            flags=re.I,
        )
        # Pattern B: "shipped date is not null"
        m_nonnull_before = re.search(
            r"\b([a-z][a-z0-9_ ]{0,40}?)\s+is\s+not\s+null\b",
            q,
            flags=re.I,
        )
        phrase = ""
        if m_nonnull_after:
            phrase = (m_nonnull_after.group(1) or "").strip()
        elif m_nonnull_before:
            phrase = (m_nonnull_before.group(1) or "").strip()

        if phrase:
            p_tokens = set(re.findall(r"[a-z0-9]+", phrase.lower()))
            best: Optional[Tuple[str, str]] = None
            best_score = 0
            for t, c, _ in column_mentions:
                if t != subject:
                    continue
                c_tokens = set(re.findall(r"[a-z0-9]+", _camel_to_words(c)))
                score = len(p_tokens.intersection(c_tokens))
                if score > best_score:
                    best_score = score
                    best = (t, c)
            if best is not None and best_score > 0:
                return best

        # Fallback: percentage of non-null usually targets date-like status columns.
        for t, c, _ in column_mentions:
            if t == subject and _is_date_column(c, t):
                return (t, c)
        for c in _ordered_allowed_columns(subject, allowed):
            if _is_date_column(c, subject):
                return (subject, c)
        return None

    # COUNT
    if wants_count:
        distinct_target = _pick_count_distinct_target()
        if distinct_target:
            count_table, count_col = distinct_target
            col_words = _camel_to_words(count_col).replace(" ", "_") or "value"
            alias = _safe_alias(f"distinct_{col_words}_count", "distinct_count")
            add_metric(
                IRMetric(
                    agg="count",
                    table=count_table,
                    column=count_col,
                    alias=alias,
                    distinct=True,
                )
            )
        else:
            count_table = subject if subject_row_count_requested else (total_row_count_table or first_mentioned or subject)
            m_count_of = re.search(
                r"\b(?:number|count|total\s+number)\s+of\s+([a-z][a-z0-9_ ]{1,40})\b",
                q,
                flags=re.I,
            )
            count_col = ""
            if total_row_count_table or subject_row_count_requested:
                count_col = "*"
            if m_count_of:
                count_phrase = re.sub(r"\s+", " ", str(m_count_of.group(1) or "").strip())
                row_unit_count = bool(
                    re.search(
                        r"\b(?:line|lines|row|rows|record|records|entry|entries|item|items|detail|details)\b",
                        count_phrase,
                        flags=re.I,
                    )
                )
                if row_unit_count:
                    count_table = subject or count_table
                    count_col = "*"
                else:
                    resolved_col = _resolve_column_from_phrase(count_phrase)
                    resolved = _resolve_table_from_phrase(count_phrase)
                    if resolved:
                        count_table = resolved
                    elif resolved_col:
                        count_table, count_col = resolved_col
            id_col = count_col or _best_id_column(count_table, allowed, question) or "*"
            alias = _safe_alias(f"{_safe_get_singular_form(count_table)}_count", "row_count")
            add_metric(IRMetric(agg="count", table=count_table, column=id_col, alias=alias))

    # Percentage of non-null values within each group (single-table aggregate).
    # Example: percentage of orders with non-null shippedDate per status.
    if wants_percentage:
        pct_target = _pick_percentage_nonnull_target()
        if pct_target:
            t, c = pct_target
            root = (_camel_to_words(c).split() or ["value"])[0]
            alias = _safe_alias(f"{root}_percentage", "percentage")
            add_metric(IRMetric(agg="pct_non_null", table=t, column=c, alias=alias))

    # SUM / AVG
    def _metric_target_for_numeric(local_phrase: str = "") -> Tuple[str, str]:
        phrase_target = _resolve_numeric_measure_from_phrase(local_phrase)
        if phrase_target:
            return phrase_target
        if mentioned_numeric_cols:
            t, c, _ = mentioned_numeric_cols[0]
            return t, c
        for t in dict.fromkeys([subject] + [tm for tm, _ in table_mentions]):
            preferred = _prefer_numeric_column_from_question(question, t, allowed)
            if preferred and _is_numeric_measure_column(t, preferred):
                return t, preferred
        for t, _ in table_mentions:
            col = _find_best_numeric_measure_column(t)
            if col:
                return t, col
        col = _find_best_numeric_measure_column(subject)
        if col:
            return subject, col
        return subject, "*"

    if wants_sum:
        t, c = _metric_target_for_numeric(_extract_local_metric_phrase("sum"))
        alias = _safe_alias(f"total_{c}", "total_value")
        add_metric(IRMetric(agg="sum", table=t, column=c, alias=alias))

    if wants_avg:
        t, c = _metric_target_for_numeric(_extract_local_metric_phrase("avg"))
        alias = _safe_alias(f"avg_{c}", "avg_value")
        add_metric(IRMetric(agg="avg", table=t, column=c, alias=alias))

    # MAX / MIN in aggregate context only.
    if (qtype or "").lower().startswith("aggregate"):
        prefer_temporal_extrema = bool(
            re.search(r"\b(?:latest|newest|most\s+recent|earliest|oldest)\b", q, flags=re.I)
            or has_temporal_metric_phrase
        )
        if wants_max:
            if prefer_temporal_extrema and mentioned_date_cols:
                t, c, _ = mentioned_date_cols[0]
            else:
                t, c = _metric_target_for_numeric(_extract_local_metric_phrase("max"))
            alias = _safe_alias(f"max_{c}", "max_value")
            add_metric(IRMetric(agg="max", table=t, column=c, alias=alias))

        if wants_min:
            if prefer_temporal_extrema and mentioned_date_cols:
                t, c, _ = mentioned_date_cols[0]
            else:
                t, c = _metric_target_for_numeric(_extract_local_metric_phrase("min"))
            alias = _safe_alias(f"min_{c}", "min_value")
            add_metric(IRMetric(agg="min", table=t, column=c, alias=alias))

    return metrics


def _detect_dimensions(
    question: str,
    qtype: str,
    subject: str,
    allowed: Dict[str, Set[str]],
    column_mentions: List[Tuple[str, str, int]],
    metrics: Optional[List[IRMetric]] = None,
) -> List[IRDimension]:
    q = (question or "").lower()
    dims: List[IRDimension] = []
    seen: Set[Tuple[str, str, str, str]] = set()
    q_group_scope = re.split(
        r"\b(?:sort(?:ed)?\s+by|order(?:ed)?\s+by|sort(?:ed)?\s+from|order(?:ed)?\s+from)\b",
        q,
        maxsplit=1,
        flags=re.I,
    )[0].strip() or q
    dedupe_row_request = bool(
        re.search(r"\bone\s+(?:row|record|result)\s+(?:per|for\s+each)\b", q)
    )
    aggregate_like = bool(
        re.search(
            r"\b(group\s+by|count|number\s+of|sum|avg|average|percentage|percent|ratio|rate)\b",
            q,
        )
        or (re.search(r"\bfor\s+each\b", q) and not dedupe_row_request)
        or (re.search(r"\bin\s+each\b", q) and not dedupe_row_request)
        or (re.search(r"\bper\b", q) and not dedupe_row_request)
    )
    date_day_group_requested = bool(re.search(r"\b(for\s+each|in\s+each|per|by)\s+(?:date|day)\b", q))
    nonnull_percentage_grouping = any(
        str(getattr(m, "agg", "") or "").lower() == "pct_non_null"
        for m in (metrics or [])
    )
    nonnull_cols: Set[str] = set()
    distinct_metric_targets = {
        (str(m.table or "").lower(), str(m.column or "").lower())
        for m in (metrics or [])
        if str(m.agg or "").lower() == "count" and bool(m.distinct)
    }

    def add_dim(d: IRDimension) -> None:
        k = (d.table.lower(), d.column.lower(), d.transform.lower(), d.alias.lower())
        if k in seen:
            return
        seen.add(k)
        dims.append(d)

    def _explicit_group_dim_requested(table: str, col: str) -> bool:
        table_patterns: List[str] = []
        for tv in sorted(_table_variants(table) or {table}, key=len, reverse=True):
            cur = str(tv or "").strip().lower()
            if not cur:
                continue
            table_patterns.append(re.escape(cur).replace(r"\ ", r"\s+"))
        for v in _column_variants(table, col):
            pat = re.escape(v).replace(r"\ ", r"\s+")
            if re.search(rf"\b(?:for\s+each|in\s+each|per|by|group\s+by)\s+{pat}\b", q_group_scope, flags=re.I):
                return True
            for table_pat in table_patterns:
                if re.search(
                    rf"\b(?:for\s+each|in\s+each|per|by|group\s+by)\s+(?:{table_pat})\s+{pat}\b",
                    q_group_scope,
                    flags=re.I,
                ):
                    return True
        return False

    def _subject_column_is_missing_null_predicate_signal(table: str, col: str) -> bool:
        if table != subject:
            return False
        for variant in _column_variants(table, col):
            pat = re.escape(variant).replace(r"\ ", r"\s+")
            if not pat:
                continue
            if re.search(
                rf"\b(?:missing|null|empty|blank|without|lacking|no)\b"
                rf"(?:\s+\w+){{0,4}}\s+{pat}\b",
                q,
                flags=re.I,
            ):
                return True
            if re.search(
                rf"\b(?:don['’]?t|doesn['’]?t|didn['’]?t|do\s+not|does\s+not|did\s+not)\s+have\b"
                rf"(?:\s+\w+){{0,4}}\s+{pat}\b",
                q,
                flags=re.I,
            ):
                return True
            if re.search(
                rf"\b{pat}\b(?:[\s,;:()'\"-]+\w+){{0,3}}[\s,;:()'\"-]+"
                rf"(?:is|are|was|were)\s+(?:missing|null|empty|blank)\b",
                q,
                flags=re.I,
            ):
                return True
            if re.search(
                rf"\b{pat}\b(?:[\s,;:()'\"-]+\w+){{0,3}}[\s,;:()'\"-]+"
                rf"(?:listed|provided|specified|set|filled)\b",
                q,
                flags=re.I,
            ) and re.search(
                r"\b(?:don['’]?t|doesn['’]?t|didn['’]?t|do\s+not|does\s+not|did\s+not|without|missing|null|empty|blank|no)\b",
                q,
                flags=re.I,
            ):
                return True
        return False

    def _subject_column_is_predicate_only_lookup_signal(table: str, col: str) -> bool:
        if (qtype or "").lower() != "lookup":
            return False
        return _subject_column_is_missing_null_predicate_signal(table, col)

    # Time bucket dimensions.
    if re.search(
        r"\b(for\s+each|in\s+each|per|by)\s+month\b|\beach\s+month(?:\s+name)?\b|\bmonth\s+names?\b",
        q,
        flags=re.I,
    ):
        date_col = _find_best_date_column(subject, allowed)
        if date_col:
            add_dim(IRDimension(table=subject, column=date_col, alias="month_name", transform="month_name", visible=True))
            add_dim(IRDimension(table=subject, column=date_col, alias="__month_num", transform="month_num", visible=False))

    if re.search(r"\b(for\s+each|in\s+each|per|by)\s+year\b", q):
        date_col = _find_best_date_column(subject, allowed)
        if date_col:
            add_dim(IRDimension(table=subject, column=date_col, alias="year", transform="year", visible=True))

    # Entity id/name intent.
    subj_pat = _table_entity_pattern(subject) if subject else ""
    subject_id_requested = False
    if subject and subj_pat:
        id_like_token = r"(?:id|ids|code|codes|number(?!\s+of\b)|numbers(?!\s+of\b))"
        direct_subject_pattern = (
            rf"\b(?:{subj_pat})\b(?:[\s,;:()'\"-]+\w+){{0,6}}[\s,;:()'\"-]+"
            rf"{id_like_token}\b"
        )
        direct_subject_id = False
        for direct_match in re.finditer(direct_subject_pattern, q, flags=re.I):
            # Do not let a table-context mention capture an ID/code phrase that
            # belongs to a later projection clause.
            if re.search(r"\b(?:return|show|display|list|include)\b", direct_match.group(0), flags=re.I):
                continue
            direct_subject_id = True
            break
        reverse_subject_id = bool(
            re.search(
                rf"\b{id_like_token}\b(?:[\s,;:()'\"-]+\w+){{0,6}}"
                rf"[\s,;:()'\"-]+(?:of|for)\s+(?:{subj_pat})\b",
                q,
            )
        )
        if direct_subject_id or reverse_subject_id:
            subject_id_requested = True
            id_col = _best_id_column(subject, allowed, question)
            if id_col:
                add_dim(IRDimension(table=subject, column=id_col))
        if re.search(rf"\b(?:{subj_pat})\s+name\b", q):
            name_col = _best_name_column(subject, allowed)
            if name_col:
                add_dim(IRDimension(table=subject, column=name_col))

    # Non-null targets are usually predicate/metric inputs, not grouping dimensions.
    if re.search(r"\b(?:non[-\s]?null|not\s+null)\b", q):
        for t, c, _ in column_mentions:
            if t != subject:
                continue
            c_words = _camel_to_words(c)
            if not c_words:
                continue
            c_pat = re.escape(c_words).replace(r"\ ", r"\s+")
            if re.search(rf"\b(?:non[-\s]?null|not\s+null)\s+{c_pat}\b", q) or re.search(
                rf"\b{c_pat}\s+is\s+not\s+null\b",
                q,
            ):
                nonnull_cols.add(c.lower())

    # Explicit column mentions in question.
    for table, col, _ in column_mentions:
        # Keep dimensions conservative for IR planner:
        # only subject columns and direct parent references (planner-safe subset).
        if table == subject:
            if _subject_column_is_predicate_only_lookup_signal(table, col):
                continue
            if aggregate_like:
                # In aggregate-like intents, keep grouping columns only.
                if (
                    not _explicit_group_dim_requested(table, col)
                    and _subject_column_is_missing_null_predicate_signal(table, col)
                ):
                    continue
                if col.lower() in nonnull_cols:
                    continue
                if nonnull_percentage_grouping and not _explicit_group_dim_requested(table, col):
                    continue
                if (
                    (table.lower(), col.lower()) in distinct_metric_targets
                    and not _explicit_group_dim_requested(table, col)
                ):
                    continue
                cl = col.lower()
                # Check ID-like BEFORE numeric filter: explicitly requested IDs
                # (e.g. "customer number") should survive even if _is_numeric_column
                # would otherwise discard them.
                is_id_like = cl.endswith("id") or cl.endswith("number") or cl.endswith("code")
                if is_id_like and not subject_id_requested and not _explicit_group_dim_requested(table, col):
                    continue
                if not is_id_like and _is_numeric_column(col, table):
                    continue
                if _is_date_column(col, table) and not date_day_group_requested:
                    continue
            add_dim(IRDimension(table=table, column=col))

    if aggregate_like and subject:
        for m in re.finditer(
            r"\b(?:for\s+each|in\s+each|per|by|group\s+by)\s+([a-z][a-z0-9_ ]{1,48}?)"
            r"(?=\s*(?:,|;|\.|\breturn\b|\bshow\b|\bdisplay\b|\blist\b|\binclude\b|\bgive\b|\bkeep\b|\bonly\b|\bsort\b|\border\b|\band\b|$))",
            q,
            flags=re.I,
        ):
            phrase = re.sub(r"\s+", " ", str(m.group(1) or "").strip())
            if not phrase:
                continue
            if phrase in {"month", "year", "date", "day"}:
                continue
            phrase_norm = phrase.lower()
            subject_variants = {
                re.sub(r"\s+", " ", str(v or "").strip().lower())
                for v in (_table_variants(subject) or {subject})
                if str(v or "").strip()
            }
            if phrase_norm in subject_variants:
                id_col = _best_id_column(subject, allowed, question)
                if id_col:
                    add_dim(IRDimension(table=subject, column=id_col))
                continue
            semantic_col = _best_semantic_column_match(
                text=phrase,
                table=subject,
                allowed=allowed,
                min_score=1.0,
            )
            descriptor_col = _best_descriptor_column_match(
                text=phrase,
                table=subject,
                allowed=allowed,
                min_score=1.0,
            )
            resolved_col = _prefer_specific_identifier_match(
                semantic_col=semantic_col,
                descriptor_col=descriptor_col,
                table=subject,
                text=phrase,
            )
            if resolved_col and not _is_numeric_column(resolved_col, subject):
                add_dim(IRDimension(table=subject, column=resolved_col))

    # Projection-list semantic enrichment for lookup questions:
    # use schema-grounded matching on explicit "show/display/return/list" tails.
    if (qtype or "").lower() == "lookup" and subject:
        projection_cols: Set[str] = set()
        best_subject_name_col = str(_best_name_column(subject, allowed) or "").strip()
        for seg in _extract_projection_segments(question):
            seg_l = re.sub(r"\s+", " ", str(seg or "").strip().lower())
            seg_tokens = set(_tokenize_for_match(seg, stop_tokens=_LEXICAL_MATCH_STOP_TOKENS))
            if best_subject_name_col and re.fullmatch(r"(?:their\s+|the\s+)?names?", seg_l):
                projection_cols.add(best_subject_name_col.lower())
                add_dim(IRDimension(table=subject, column=best_subject_name_col))
                continue
            explicit_projection_cols: List[str] = []
            explicit_variants: Dict[str, str] = {}
            for col in _ordered_allowed_columns(subject, allowed):
                for variant in _column_reference_variants(subject, col):
                    pat = re.escape(variant).replace(r"\ ", r"\s+")
                    if pat and re.search(rf"\b{pat}\b", seg, flags=re.I):
                        explicit_projection_cols.append(str(col).strip())
                        explicit_variants.setdefault(str(col).strip(), variant)
                        break
            explicit_projection_cols = list(dict.fromkeys(explicit_projection_cols))
            if len(explicit_projection_cols) == 1:
                resolved_col = explicit_projection_cols[0]
                variant_tokens = set(
                    _tokenize_for_match(
                        explicit_variants.get(resolved_col, ""),
                        stop_tokens=_LEXICAL_MATCH_STOP_TOKENS,
                    )
                )
                extra_tokens = seg_tokens - variant_tokens
                if not extra_tokens:
                    projection_cols.add(str(resolved_col).strip().lower())
                    add_dim(IRDimension(table=subject, column=resolved_col))
                    continue

            semantic_col = _best_semantic_column_match(
                text=seg,
                table=subject,
                allowed=allowed,
                min_score=1.0,
            )
            descriptor_col = _best_descriptor_column_match(
                text=seg,
                table=subject,
                allowed=allowed,
                id_like_only=True,
                min_score=1.0,
            )
            resolved_col = _prefer_specific_identifier_match(
                semantic_col=semantic_col,
                descriptor_col=descriptor_col,
                table=subject,
                text=seg,
            )
            if resolved_col:
                projection_cols.add(str(resolved_col).strip().lower())
                add_dim(IRDimension(table=subject, column=resolved_col))
            for col in explicit_projection_cols:
                projection_cols.add(str(col).strip().lower())
        if projection_cols:
            dims = [
                d for d in dims
                if not (
                    d.table == subject
                    and str(d.column or "").strip().lower() not in projection_cols
                )
            ]
            seen = {
                (d.table.lower(), d.column.lower(), d.transform.lower(), d.alias.lower())
                for d in dims
            }

    # Top-N lookup enrichment: keep an identity column alongside the ranking field.
    if (
        (qtype or "").lower() == "lookup"
        and re.search(r"\b(?:top|bottom|highest|lowest|largest|smallest)\b", q, flags=re.I)
    ):
        ranked_numeric_col = _prefer_numeric_column_from_question(question, subject, allowed)
        has_numeric_dim = any(
            d.table == subject and _is_numeric_column(d.column, d.table)
            for d in dims
        )
        has_identity_dim = any(
            d.table == subject and not _is_numeric_column(d.column, d.table)
            for d in dims
        )
        if ranked_numeric_col and not any(
            d.table == subject and d.column.lower() == ranked_numeric_col.lower()
            for d in dims
        ):
            add_dim(IRDimension(table=subject, column=ranked_numeric_col))
            has_numeric_dim = True
        if has_numeric_dim and not has_identity_dim:
            name_col = _best_name_column(subject, allowed)
            id_col = _best_id_column(subject, allowed, question)
            if name_col:
                add_dim(IRDimension(table=subject, column=name_col))
            elif id_col:
                add_dim(IRDimension(table=subject, column=id_col))
        if has_numeric_dim:
            dims.sort(
                key=lambda d: (
                    1 if d.table == subject and _is_numeric_column(d.column, d.table) else 0,
                    d.table.lower(),
                    d.column.lower(),
                )
            )

    # Generic "name" request without an explicit projection list.
    if subject and not _extract_projection_segments(question) and re.search(r"\bname(s)?\b", q):
        name_col = _best_name_column(subject, allowed)
        if name_col:
            add_dim(IRDimension(table=subject, column=name_col))

    # Lookup fallback.
    if not dims and (qtype or "").lower() == "lookup":
        name_col = _best_name_column(subject, allowed)
        id_col = _best_id_column(subject, allowed, question)
        if name_col:
            add_dim(IRDimension(table=subject, column=name_col))
        elif id_col:
            add_dim(IRDimension(table=subject, column=id_col))

    # Aggregate dimensions become prompt/planner inputs. Canonicalize them by
    # their first textual mention so process-level set/hash ordering cannot
    # change an otherwise identical ablation run.
    if aggregate_like and len(dims) > 1:
        def _aggregate_dimension_order(dimension: IRDimension) -> Tuple[int, str, str, int, str, str]:
            positions = [
                pos
                for variant in _column_variants(dimension.table, dimension.column)
                if (pos := _phrase_pos(q, variant)) >= 0
            ]
            return (
                min(positions) if positions else len(q) + 1,
                dimension.table.lower(),
                dimension.column.lower(),
                0 if dimension.visible else 1,
                dimension.transform.lower(),
                dimension.alias.lower(),
            )

        dims.sort(key=_aggregate_dimension_order)

    return dims


def _detect_filters(
    question: str,
    subject: str,
    allowed: Dict[str, Set[str]],
    column_mentions: List[Tuple[str, str, int]],
    dimensions: List[IRDimension],
    metrics: List[IRMetric],
) -> List[IRFilter]:
    q = (question or "").lower()
    q_raw = (question or "").strip()
    filters: List[IRFilter] = []
    projection_fill_literal = _extract_projection_fill_literal(question)
    projection_null_fill_requested = _question_requests_null_fill_projection(question)
    weekday_name_to_num = {
        "sunday": 1,
        "monday": 2,
        "tuesday": 3,
        "wednesday": 4,
        "thursday": 5,
        "friday": 6,
        "saturday": 7,
    }

    def _question_requests_conditional_null_breakdown() -> bool:
        has_group_or_count = bool(
            re.search(r"\b(?:for\s+each|per|by|group|summarize|summarise|show|return|how\s+many|count)\b", q, flags=re.I)
        )
        has_null_side = bool(
            re.search(r"\b(?:missing|null|blank|empty)\b", q, flags=re.I)
        )
        has_present_side = bool(
            re.search(r"\b(?:filled(?:\s+in)?|present|non[-\s]?null|commented)\b", q, flags=re.I)
        )
        has_pairing = bool(
            re.search(r"\bhow\s+many\b.*\b(?:filled(?:\s+in)?|present|commented)\b.*\bhow\s+many\b.*\b(?:missing|null|blank|empty)\b", q, flags=re.I)
            or re.search(r"\b(?:missing|null|blank|empty)\b.*\b(?:whether|or)\b.*\b(?:blank|null|empty)\b", q, flags=re.I)
        )
        return bool(has_group_or_count and has_null_side and (has_present_side or has_pairing))

    def add_filter(f: IRFilter) -> None:
        for ex in filters:
            if (
                ex.table.lower() == f.table.lower()
                and ex.column.lower() == f.column.lower()
                and ex.op == f.op
                and ex.value == f.value
                and ex.value2 == f.value2
                and ex.value_column == f.value_column
            ):
                return
        filters.append(f)

    aggregate_metric_aggs = {
        str(getattr(m, "agg", "") or "").strip().lower()
        for m in (metrics or [])
    }
    aggregate_query_like = bool(
        aggregate_metric_aggs.intersection({"count", "sum", "avg", "min", "max", "pct_non_null"})
    )
    grouped_aggregate_language = bool(
        re.search(
            r"\b(?:for\s+each|per|by|group(?:ed)?|summari[sz]e|break\s+.*\s+down)\b",
            q,
            flags=re.I,
        )
    )

    def _threshold_has_row_count_tail(tail: str) -> bool:
        scope_terms = (
            list(allowed.keys())
            + [_safe_get_singular_form(t) for t in allowed.keys()]
        )
        return bool(
            re.search(
                r"^\s+(?:\w+\s+){0,4}?(?:rows?|records?|lines?|entries|items?|groups?)\b",
                tail,
                flags=re.I,
            )
            or any(
                re.search(
                    rf"^\s+(?:\w+\s+){{0,4}}?{re.escape(str(term or '').strip().lower())}\b",
                    tail,
                    flags=re.I,
                )
                for term in scope_terms
                if str(term or "").strip()
            )
        )

    def _is_aggregate_bound_threshold(match_obj: re.Match[str]) -> bool:
        if not aggregate_query_like:
            return False

        start = int(match_obj.start())
        end = int(match_obj.end())
        prefix = q[max(0, start - 96):start]
        suffix = q[end:end + 96]

        if _threshold_has_row_count_tail(suffix):
            return True

        if re.search(r"^\s+distinct\b", suffix, flags=re.I):
            return True

        if re.search(r"^\s+times\b", suffix, flags=re.I):
            return True

        if re.search(r"\b(?:appear|appears|appearing|occur|occurs|occurring)\b", prefix, flags=re.I):
            return True

        if re.search(r"\b(?:average|avg|mean)\b", prefix, flags=re.I):
            return True

        if grouped_aggregate_language and re.search(
            r"\b(?:how\s+many|count|number\s+of|total\s+number)\b",
            prefix,
            flags=re.I,
        ):
            return True

        return False

    def _comparison_clause_tail(match_obj: re.Match[str]) -> str:
        suffix = q[match_obj.end():match_obj.end() + 96]
        stop_match = re.search(
            r"[.;:!?]|\s+\b(?:and|or)\b\s+(?:(?:the|a|an)\s+)?[a-z]",
            suffix,
            flags=re.I,
        )
        if stop_match:
            suffix = suffix[:stop_match.start()]
        return suffix

    def _pick_date_target_column() -> Tuple[str, Optional[str]]:
        date_col = None
        date_table = subject
        for t, c, _ in column_mentions:
            if _is_date_column(c, t):
                date_table, date_col = t, c
                break
        if not date_col:
            date_col = _find_best_date_column(subject, allowed)
        if not date_col:
            metric_tables: List[str] = []
            for m in metrics or []:
                t = str(getattr(m, "table", "") or "").strip().lower()
                if t and t not in metric_tables:
                    metric_tables.append(t)
            for mt in metric_tables:
                mt_date_col = _find_best_date_column(mt, allowed)
                if mt_date_col:
                    date_table = mt
                    date_col = mt_date_col
                    break
        return date_table, date_col

    def _normalize_process_tokens(text: str) -> Set[str]:
        out: Set[str] = set()
        for tok in re.findall(r"[a-z]+", str(text or "").lower()):
            if len(tok) <= 2:
                continue
            base = tok
            for suffix in ("ing", "ed", "ment", "tion", "s"):
                if base.endswith(suffix) and len(base) > len(suffix) + 2:
                    base = base[: -len(suffix)]
                    break
            out.add(base)
        return out

    def _extract_negated_process_terms() -> Set[str]:
        patterns = [
            r"\b(?:haven['’]t|have\s+not|hasn['’]t|has\s+not|not)\s+been\s+([a-z][a-z\s-]{1,24}?)(?:\s+yet)?\b",
            r"\bnot\s+([a-z][a-z\s-]{1,24}?)\s+yet\b",
            r"\bpending\s+([a-z][a-z\s-]{1,24})\b",
        ]
        for pat in patterns:
            m = re.search(pat, q, flags=re.I)
            if m:
                terms = _normalize_process_tokens(m.group(1) or "")
                if terms:
                    return terms
        return set()

    def _extract_enum_values_from_description(table: str, column: str) -> List[str]:
        desc = str(get_column_description(table, column) or "").strip()
        if not desc:
            return []

        candidates: List[str] = []
        patterns = [
            r"\(\s*e\.g\.,?\s*([^)]+?)\s*\)",
            r"\be\.g\.,?\s*([^.;]+)",
            r"\bi\.e\.,?\s*([^.;]+)",
        ]
        for pat in patterns:
            m = re.search(pat, desc, flags=re.I)
            if not m:
                continue
            raw = str(m.group(1) or "").strip()
            if not raw:
                continue
            parts = [p.strip(" '\"") for p in raw.split(",")]
            for part in parts:
                clean = re.sub(r"\s+", " ", part).strip()
                if clean and clean not in candidates:
                    candidates.append(clean)
            if candidates:
                break
        return candidates

    def _derive_nonfinal_enum_values(table: str, column: str) -> List[str]:
        enum_values = _extract_enum_values_from_description(table, column)
        if not enum_values:
            return []

        excluded_terms = {
            "shipped",
            "resolved",
            "complete",
            "completed",
            "delivered",
            "closed",
            "fulfilled",
            "finalized",
            "finalised",
        }
        out: List[str] = []
        for value in enum_values:
            value_l = value.lower()
            if any(term in value_l for term in excluded_terms):
                continue
            if value not in out:
                out.append(value)
        return out

    def _question_mentions_enum_values(values: Sequence[str]) -> bool:
        q_text = str(q_raw or q or "").strip()
        if not q_text:
            return False
        for value in values or []:
            clean = re.sub(r"\s+", " ", str(value or "").strip())
            if not clean:
                continue
            pat = r"\b" + r"\s+".join(re.escape(part) for part in clean.split()) + r"\b"
            if re.search(pat, q_text, flags=re.I):
                return True
        return False

    def _find_best_process_proxy_column(table: str, process_terms: Set[str]) -> Optional[str]:
        if not process_terms:
            return None
        best_col = None
        best_score = 0
        for col in _ordered_allowed_columns(table, allowed):
            desc = str(get_column_description(table, col) or "").strip()
            if not desc or not re.search(r"\bnull\s+if\s+not\s+yet\b", desc, flags=re.I):
                continue
            proxy_phrase = re.search(r"\bnull\s+if\s+not\s+yet\s+([^.;]+)", desc, flags=re.I)
            proxy_terms = _normalize_process_tokens((proxy_phrase.group(1) if proxy_phrase else desc))
            score = len(process_terms.intersection(proxy_terms))
            if score > best_score:
                best_score = score
                best_col = col
        return best_col if best_score > 0 else None

    def _find_best_enum_descriptor_column(table: str) -> Optional[str]:
        best_col = None
        best_score = 0
        for col in _ordered_allowed_columns(table, allowed):
            desc = str(get_column_description(table, col) or "").strip()
            if not desc:
                continue
            enum_values = _extract_enum_values_from_description(table, col)
            if len(enum_values) < 2:
                continue
            desc_l = desc.lower()
            score = len(enum_values)
            if any(term in desc_l for term in ("status", "state", "phase", "condition")):
                score += 5
            if score > best_score:
                best_score = score
                best_col = col
        return best_col

    def _supports_categorical_text_filter(table: str, column: str) -> bool:
        if _is_date_column(column, table) or _is_identifier_like_column(table, column):
            return False
        if not _is_numeric_column(column, table):
            return True

        dtype = _column_schema_semantics(table, column)["datatype"]
        if any(tok in dtype for tok in _TEXTUAL_DTYPE_HINTS):
            return True

        desc = _column_schema_semantics(table, column)["description"].lower()
        if re.search(r"\b(?:text|string|label|ratio|category|status|state|type)\b", desc, flags=re.I):
            return True
        if re.search(r"\be\.g\.[^.]*[:/-]", desc, flags=re.I):
            return True

        enum_values = _extract_enum_values_from_description(table, column)
        if enum_values and any(
            not re.fullmatch(_NUMBER_TOKEN_PATTERN, str(v or "").replace(",", "").strip())
            for v in enum_values
        ):
            return True

        return False

    def _pick_text_target_column(anchor_pos: Optional[int] = None) -> Optional[Tuple[str, str]]:
        # Prefer explicit "check number" mention when present.
        if re.search(r"\bcheck\s+number\b", q):
            for c in _ordered_allowed_columns(subject, allowed):
                cl = c.lower()
                if "check" in cl and "number" in cl:
                    return (subject, c)
        if anchor_pos is not None:
            window_start = max(0, anchor_pos - 80)
            window_end = min(len(q), anchor_pos + 24)
            anchor_window = q[window_start:window_end]
            before_anchor = q[window_start:anchor_pos]
            preferred_description_col = _pick_qualified_description_column(anchor_window, subject, allowed)
            if preferred_description_col:
                return (subject, preferred_description_col)
            explicit_name_component = _best_name_component_column_match(before_anchor, subject, allowed)
            if explicit_name_component:
                return (subject, explicit_name_component)
            best_exact: Optional[Tuple[str, str]] = None
            best_exact_score: Tuple[bool, bool, int, int, int] = (False, False, -1, -1, -1)
            for t, _ in _ordered_allowed_tables(allowed):
                table_l = str(t or "").strip().lower()
                if not table_l:
                    continue
                for c in _ordered_allowed_columns(t, allowed):
                    if _is_date_column(c, table_l):
                        continue
                    for variant in sorted(_column_reference_variants(table_l, c), key=len, reverse=True):
                        variant_l = str(variant or "").strip().lower()
                        if not variant_l:
                            continue
                        pos = _phrase_pos(before_anchor, variant_l)
                        if pos < 0:
                            continue
                        variant_tokens = _tokenize_for_match(
                            variant_l,
                            stop_tokens=_LEXICAL_MATCH_STOP_TOKENS,
                        )
                        if not variant_tokens:
                            continue
                        identifier_variant = bool(
                            _is_identifier_like_column(table_l, c)
                            and set(variant_tokens).intersection(_GENERIC_IDENTIFIER_TOKENS)
                        )
                        score_tuple = (
                            table_l == subject,
                            identifier_variant,
                            pos,
                            len(variant_l),
                            -len(str(c or "")),
                        )
                        if score_tuple > best_exact_score:
                            best_exact = (table_l, c)
                            best_exact_score = score_tuple
                            break
            if best_exact is not None:
                return best_exact

            anchor_tokens = set(
                _tokenize_for_match(
                    anchor_window,
                    stop_tokens=_LEXICAL_MATCH_STOP_TOKENS.union(
                        {
                            "start",
                            "starts",
                            "begin",
                            "begins",
                            "followed",
                            "exactly",
                            "digit",
                            "digits",
                            "letter",
                            "letters",
                        }
                    ),
                )
            )

            best_local: Optional[Tuple[str, str]] = None
            best_score = float("-inf")
            best_distance = 10**9
            for t, c, pos in column_mentions:
                if _is_date_column(c, t):
                    continue
                cl = c.lower()
                is_id_like = _is_identifier_like_column(t, c) or cl.endswith(("id", "number", "code", "key"))
                semantic_tokens = _column_semantic_tokens(t, c)
                overlap = anchor_tokens.intersection(semantic_tokens)
                distance = abs(pos - anchor_pos)

                score = float(len(overlap) * 3)
                if pos <= anchor_pos:
                    score += 0.6
                score += max(0.0, 1.5 - (distance / 40.0))
                if not _is_numeric_column(c, t) and not is_id_like:
                    score += 1.0
                elif is_id_like:
                    score -= 1.0

                if overlap and (
                    score > best_score or (score == best_score and distance < best_distance)
                ):
                    best_local = (t, c)
                    best_score = score
                    best_distance = distance

            if best_local is not None:
                return best_local
        # Fallback to first non-numeric/non-date mention.
        for t, c, _ in column_mentions:
            if not _is_numeric_column(c, t) and not _is_date_column(c, t):
                return (t, c)
        name_col = _best_name_column(subject, allowed)
        if name_col:
            return (subject, name_col)
        return None

    def _projected_subject_text_candidates() -> List[str]:
        projection_segments = _extract_projection_segments(question)
        if not projection_segments:
            return []

        name_col = str(_best_name_column(subject, allowed) or "").strip().lower()
        candidates: List[str] = []
        seen: Set[str] = set()
        deferred_name: List[str] = []
        for seg in projection_segments:
            matched = None
            seg_l = str(seg or "").strip().lower()
            best_variant_len = -1
            for column in _ordered_allowed_columns(subject, allowed):
                if not _supports_categorical_text_filter(subject, column):
                    continue
                for variant in sorted(_column_reference_variants(subject, column), key=len, reverse=True):
                    variant_l = str(variant or "").strip().lower()
                    if not variant_l:
                        continue
                    if _phrase_pos(seg_l, variant_l) >= 0 and len(variant_l) > best_variant_len:
                        matched = column
                        best_variant_len = len(variant_l)
                        break
            if not matched:
                matched = _best_semantic_column_match(
                    text=seg,
                    table=subject,
                    allowed=allowed,
                    min_score=1.0,
                )
            if not matched:
                continue
            matched_l = str(matched or "").strip().lower()
            if not matched_l or matched_l in seen:
                continue
            if not _supports_categorical_text_filter(subject, matched):
                continue
            seen.add(matched_l)
            if matched_l == name_col:
                deferred_name.append(matched)
            else:
                candidates.append(matched)
        return candidates + deferred_name

    def _pick_explicit_text_target_column(anchor_pos: int) -> Optional[Tuple[str, str]]:
        prefix = q_raw[max(0, int(anchor_pos) - 96):int(anchor_pos)]
        prefix_l = re.sub(r"\s+", " ", prefix.strip().lower())
        if not prefix_l:
            return None

        preferred_description_col = _pick_qualified_description_column(prefix, subject, allowed)
        if preferred_description_col and _supports_categorical_text_filter(subject, preferred_description_col):
            return (subject, preferred_description_col)

        explicit_name_component = _best_name_component_column_match(prefix_l, subject, allowed)
        if explicit_name_component:
            return (subject, explicit_name_component)

        best_exact: Optional[Tuple[str, str]] = None
        best_exact_len = -1
        best_exact_pos = -1
        for column in sorted(allowed.get(subject, set()), key=lambda c: (len(str(c or "")), str(c or "").lower())):
            if not _supports_categorical_text_filter(subject, column):
                continue
            for variant in sorted(_column_reference_variants(subject, column), key=len, reverse=True):
                variant_l = str(variant or "").strip().lower()
                if not variant_l:
                    continue
                variant_tokens = _tokenize_for_match(
                    variant_l,
                    stop_tokens=_LEXICAL_MATCH_STOP_TOKENS,
                )
                if variant_tokens == ["name"]:
                    continue
                pos = _phrase_pos(prefix_l, variant_l)
                if pos < 0:
                    continue
                if pos > best_exact_pos or (pos == best_exact_pos and len(variant_l) > best_exact_len):
                    best_exact = (subject, column)
                    best_exact_len = len(variant_l)
                    best_exact_pos = pos
                    break

        if best_exact is not None:
            return best_exact

        matched = _best_semantic_column_match(
            text=prefix,
            table=subject,
            allowed=allowed,
            min_score=1.0,
        )
        if matched and _supports_categorical_text_filter(subject, matched):
            return (subject, matched)
        return None

    def _pick_exact_numeric_target_column(anchor_pos: int) -> Optional[Tuple[str, str]]:
        window_start = max(0, anchor_pos - 96)
        anchor_window = q[window_start:anchor_pos]
        identifier_intent = bool(
            re.search(r"\b(?:id|ids|code|codes|number|numbers|key|keys)\b", anchor_window, flags=re.I)
        )
        explicit_self_fk_parent_target = _best_subject_self_fk_parent_column()
        if (
            explicit_self_fk_parent_target
            and _anchor_has_explicit_self_fk_parent_identifier_intent(anchor_window)
        ):
            return explicit_self_fk_parent_target
        best_exact: Optional[Tuple[str, str]] = None
        best_exact_len = -1
        best_exact_pos = -1
        for t, c, _ in column_mentions:
            if t != subject:
                continue
            is_numeric_like = _is_numeric_column(c, t) or (
                identifier_intent and _is_identifier_like_column(t, c)
            )
            if not is_numeric_like:
                continue
            for variant in sorted(_column_reference_variants(t, c), key=len, reverse=True):
                variant_l = str(variant or "").strip().lower()
                if not variant_l:
                    continue
                pos = _phrase_pos(anchor_window, variant_l)
                if pos < 0:
                    continue
                if pos > best_exact_pos or (pos == best_exact_pos and len(variant_l) > best_exact_len):
                    best_exact = (t, c)
                    best_exact_len = len(variant_l)
                    best_exact_pos = pos
                    break
        if best_exact is not None:
            return best_exact

        semantic_match = _best_semantic_column_match(
            text=anchor_window,
            table=subject,
            allowed=allowed,
            numeric_only=not identifier_intent,
            min_score=1.0,
        )
        descriptor_match = (
            _best_descriptor_column_match(
                text=anchor_window,
                table=subject,
                allowed=allowed,
                id_like_only=True,
                min_score=1.0,
            )
            if identifier_intent
            else None
        )
        matched = _prefer_specific_identifier_match(
            semantic_col=semantic_match,
            descriptor_col=descriptor_match,
            table=subject,
            text=anchor_window,
        )
        if matched:
            return (subject, matched)

        best_local: Optional[Tuple[str, str]] = None
        best_score = float("-inf")
        best_distance = 10**9
        for t, c, pos in column_mentions:
            if t != subject:
                continue
            is_numeric_like = _is_numeric_column(c, t) or (
                identifier_intent and _is_identifier_like_column(t, c)
            )
            if not is_numeric_like:
                continue
            distance = abs(pos - anchor_pos)
            score = max(0.0, 2.0 - (distance / 48.0))
            if pos <= anchor_pos:
                score += 0.6
            if score > best_score or (score == best_score and distance < best_distance):
                best_local = (t, c)
                best_score = score
                best_distance = distance
        return best_local

    def _looks_like_alphabetical_sort_phrase(text: str) -> bool:
        clean = re.sub(r"\s+", " ", str(text or "").strip().lower())
        if not clean:
            return False
        return bool(
            re.fullmatch(r"[a-z]\s+to\s+[a-z]", clean)
            or clean in {"a to z", "z to a"}
        )

    def _has_negated_membership_prefix(anchor_pos: int) -> bool:
        prefix = q_raw[max(0, int(anchor_pos) - 28):int(anchor_pos)]
        return bool(
            re.search(
                r"\b(?:not|excluding|except|outside)\s*$"
                r"|\bnot\s+(?:located|based|found|present)\s*$",
                prefix,
                flags=re.I,
            )
        )

    def _question_span_is_column_reference(start_pos: int, end_pos: int) -> bool:
        span_start = int(start_pos)
        span_end = int(end_pos)
        span_phrase = re.sub(r"\s+", " ", q[max(0, span_start):span_end].strip().lower())
        for table, _ in _ordered_allowed_tables(allowed):
            for column in _ordered_allowed_columns(table, allowed):
                for variant in _column_reference_variants(table, column):
                    variant_l = str(variant or "").strip().lower()
                    if not variant_l:
                        continue
                    if span_phrase and span_phrase in variant_l:
                        return True
                    pat = re.escape(variant_l).replace(r"\ ", r"\s+")
                    for m_variant in re.finditer(rf"\b{pat}\b", q, flags=re.I):
                        if m_variant.start() <= span_start < m_variant.end():
                            return True
        return False

    def _looks_like_nonliteral_category_phrase(text: str) -> bool:
        clean = re.sub(r"\s+", " ", str(text or "").strip().lower())
        if not clean:
            return True
        if clean in {
            "value", "values", "field", "fields", "information", "info", "data",
            "count", "counts", "dataset", "result", "results",
        }:
            return True
        if any(clean.startswith(prefix) for prefix in ("each ", "every ", "per ", "all ")):
            return True
        if any(clean.startswith(prefix) for prefix in ("for all ", "for every ", "for each ")):
            return True
        if re.match(r"^(?:the\s+)?(?:same|that|this|those|these|their|its)\b", clean):
            return True
        if re.search(r"\b(?:dataset|data|result|results|count|counts)\b", clean):
            return True
        if re.search(r"\b(?:same\s+\w+|that\s+\w+|this\s+\w+|those\s+\w+|these\s+\w+)\b", clean):
            return True
        if re.search(r"\b(?:missing|null|blank|empty|filled|listed|provided|specified|have|has|had|whether)\b", clean):
            return True
        if re.search(r"\bfor\s+all\b", clean):
            return True
        if re.search(
            r"\b(?:longer|shorter|greater|less|more|fewer)\s+than\b|\bat\s+least\b|\bat\s+most\b",
            clean,
            flags=re.I,
        ):
            return True
        if re.search(
            r"\b(?:characters?|chars?|letters?|digits?|words?|bytes?)\b",
            clean,
            flags=re.I,
        ) and re.search(r"\b\d+\b", clean):
            return True
        if re.fullmatch(r"(?:from\s+)?(?:high|low)\s+to\s+(?:high|low)", clean):
            return True
        if _looks_like_alphabetical_sort_phrase(clean):
            return True
        return False

    def _extract_explicit_categorical_eq() -> Optional[IRFilter]:
        stop_head_terms = (
            "start", "starts", "begin", "begins", "contain", "contains",
            "containing", "end", "ends", "ending", "between", "above", "below",
            "under", "over", "greater", "less", "after", "before", "missing",
            "null", "empty", "blank", "exactly", "followed", "higher", "lower",
            "longer", "shorter", "more", "fewer",
            "equal", "equals", "same", "than", "like", "matching", "match",
            "ascending", "descending", "asc", "desc", "sort", "sorted",
            "order", "ordered",
        )
        clause_stop = (
            r"(?:[.,;!?]|"
            r"\s+(?:return|show|list|display|include|give|sort|order|ordered|sorted|where|having)\b|$)"
        )
        best: Optional[Tuple[int, IRFilter]] = None

        eq_candidates: List[Tuple[str, str]] = []
        seen_eq_candidates: Set[Tuple[str, str]] = set()
        for table, column, _ in column_mentions:
            if table != subject:
                continue
            if not _supports_categorical_text_filter(table, column):
                continue
            key = (table, column)
            if key in seen_eq_candidates:
                continue
            seen_eq_candidates.add(key)
            eq_candidates.append(key)
        for column in _ordered_allowed_columns(subject, allowed):
            if not _supports_categorical_text_filter(subject, column):
                continue
            key = (subject, column)
            if key in seen_eq_candidates:
                continue
            seen_eq_candidates.add(key)
            eq_candidates.append(key)

        known_variant_phrases: Set[str] = set()
        for table, column in eq_candidates:
            for variant in _column_reference_variants(table, column):
                variant_norm = re.sub(r"\s+", " ", str(variant or "").strip().lower())
                if variant_norm:
                    known_variant_phrases.add(variant_norm)

        name_col = _best_name_column(subject, allowed)
        if name_col:
            subject_variants = [
                str(variant or "").strip()
                for variant in _table_variants(subject)
                if str(variant or "").strip()
            ]
            for subject_variant in sorted(subject_variants, key=len, reverse=True):
                subject_pat = re.escape(subject_variant).replace(r"\ ", r"\s+")
                subject_name_match = re.search(
                    rf"\b(?:the\s+)?{subject_pat}\s+name\b\s+"
                    rf"(?:(?:of|is|=|equals?|equal\s+to)\s+)"
                    rf"""["'“”]?(?P<value>[A-Za-z0-9][A-Za-z0-9/&:._'/-]*(?:\s+[A-Za-z0-9/&:._'/-]+){{0,5}}?)["'“”]?(?={clause_stop})""",
                    q_raw,
                    flags=re.I,
                )
                if not subject_name_match:
                    continue
                raw_value = str(subject_name_match.group("value") or "").strip(" \t\r\n'\".,;:!?")
                if raw_value and not _looks_like_nonliteral_category_phrase(raw_value):
                    norm_value = _normalize_categorical_literal_value(raw_value, column=name_col)
                    if norm_value and not _looks_like_nonliteral_category_phrase(norm_value):
                        return IRFilter(table=subject, column=name_col, op="eq", value=norm_value)

            named_match = re.search(
                rf"\b(?:named|called)\s+"
                rf"""["'“”]?(?P<value>[A-Za-z0-9][A-Za-z0-9/&:._'/-]*(?:\s+[A-Za-z0-9/&:._'/-]+){{0,5}}?)["'“”]?(?={clause_stop})""",
                q_raw,
                flags=re.I,
            )
            if named_match:
                raw_value = str(named_match.group("value") or "").strip(" \t\r\n'\".,;:!?")
                if raw_value and not _looks_like_nonliteral_category_phrase(raw_value):
                    norm_value = _normalize_categorical_literal_value(raw_value, column=name_col)
                    if norm_value and not _looks_like_nonliteral_category_phrase(norm_value):
                        return IRFilter(table=subject, column=name_col, op="eq", value=norm_value)

        for table, column in eq_candidates:
            for variant in sorted(_column_reference_variants(table, column), key=len, reverse=True):
                variant_text = str(variant or "").strip()
                if not variant_text:
                    continue
                c_pat = re.escape(variant_text).replace(r"\ ", r"\s+")
                match = re.search(
                    rf"\b{c_pat}\b(?:\s+(?P<link>of|is|=|equals?|equal\s+to))?\s+"
                    rf"""["'“”]?(?P<value>[A-Za-z0-9][A-Za-z0-9/&:._'/-]*(?:\s+[A-Za-z0-9/&:._'/-]+){{0,5}}?)["'“”]?(?={clause_stop})""",
                    q_raw,
                    flags=re.I,
                )
                if not match:
                    continue
                prefix_local = q_raw[max(0, int(match.start()) - 40):int(match.start())]
                if re.search(r"\b(?:not\s+in|excluding|except(?:\s+for)?|outside)\s+$", prefix_local, flags=re.I):
                    continue

                raw_value = str(match.group("value") or "").strip(" \t\r\n'\".,;:!?")
                if not raw_value:
                    continue
                has_explicit_link = bool(str(match.groupdict().get("link") or "").strip())
                if not has_explicit_link and raw_value.lower() in {"is", "are", "was", "were"}:
                    continue
                variant_norm = re.sub(r"\s+", " ", variant_text.strip().lower())
                raw_value_norm = re.sub(r"\s+", " ", raw_value.strip().lower())
                combined_phrase = f"{variant_norm} {raw_value_norm}".strip()
                if (
                    not has_explicit_link
                    and raw_value_norm
                    and combined_phrase != variant_norm
                    and combined_phrase in known_variant_phrases
                ):
                    continue
                if (
                    projection_null_fill_requested
                    and projection_fill_literal
                    and raw_value.lower() == projection_fill_literal.lower()
                ):
                    continue
                raw_value_l = raw_value.lower()
                if raw_value_l.split() and raw_value_l.split()[0] in stop_head_terms:
                    continue
                scope_terms = {
                    str(subject or "").strip().lower(),
                    str(_safe_get_singular_form(subject) or "").strip().lower(),
                }
                if (
                    aggregate_query_like
                    and raw_value_l in scope_terms
                    and re.search(
                        r"\b(?:total|commented|filled|present|missing(?:-[a-z]+)?|count(?:ed)?|number\s+of)\b",
                        prefix_local,
                        flags=re.I,
                    )
                ):
                    continue
                if _looks_like_nonliteral_category_phrase(raw_value):
                    continue
                if re.search(r"\b(?:or|and)\b", raw_value_l):
                    continue

                norm_value = _normalize_categorical_literal_value(raw_value, column=column)
                if not norm_value:
                    continue
                if _looks_like_nonliteral_category_phrase(norm_value):
                    continue

                enum_values = _extract_enum_values_from_description(table, column)
                if enum_values:
                    matched_enum = next(
                        (
                            enum_value
                            for enum_value in enum_values
                            if norm_value.lower() == str(enum_value or "").strip().lower()
                        ),
                        None,
                    )
                    if matched_enum:
                        norm_value = matched_enum

                candidate = IRFilter(table=table, column=column, op="eq", value=norm_value)
                if best is None or int(match.start()) < best[0]:
                    best = (int(match.start()), candidate)

        if best is not None:
            return best[1]

        candidate_matches = []
        for match in re.finditer(r"""["'“”]([^"'“”]{1,40})["'“”]""", q_raw, flags=re.I):
            candidate_matches.append((match.start(), match.end(), str(match.group(1) or "").strip()))
        for match in re.finditer(r"\b([A-Za-z0-9]+(?:[:/.-][A-Za-z0-9]+){1,4})\b", q_raw, flags=re.I):
            candidate_matches.append((match.start(), match.end(), str(match.group(1) or "").strip()))
        candidate_matches.sort(key=lambda item: item[0])

        for start_pos, end_pos, raw_value in candidate_matches:
            if not raw_value:
                continue
            if (
                projection_null_fill_requested
                and projection_fill_literal
                and raw_value.lower() == projection_fill_literal.lower()
            ):
                continue
            prefix_local = q_raw[max(0, start_pos - 20):start_pos]
            tail_local = q_raw[end_pos:end_pos + 40]
            if re.search(r"\beither\b", prefix_local, flags=re.I):
                continue
            if re.search(
                r"""\b(?:or|and)\b\s*(?:["'“”][^"'“”]+["'“”]|[A-Za-z0-9]+(?:[:/.-][A-Za-z0-9]+){1,4})""",
                tail_local,
                flags=re.I,
            ):
                continue

            prefix = q_raw[max(0, start_pos - 80):start_pos]
            matched = _best_semantic_column_match(
                text=prefix,
                table=subject,
                allowed=allowed,
                min_score=1.0,
            )
            if not matched or not _supports_categorical_text_filter(subject, matched):
                continue

            raw_value_l = raw_value.lower()
            if raw_value_l.split() and raw_value_l.split()[0] in stop_head_terms:
                continue
            if _looks_like_nonliteral_category_phrase(raw_value):
                continue

            norm_value = _normalize_categorical_literal_value(raw_value, column=matched)
            if not norm_value:
                continue
            if _looks_like_nonliteral_category_phrase(norm_value):
                continue
            return IRFilter(table=subject, column=matched, op="eq", value=norm_value)

        return None

    def _extract_explicit_categorical_in() -> Optional[IRFilter]:
        clause_stop = (
            r"(?:[.,;!?]|"
            r"\s+(?:return|show|list|display|include|give|sort|order|ordered|sorted|where|having)\b|$)"
        )
        best: Optional[Tuple[int, IRFilter]] = None

        for table, column, _ in column_mentions:
            if table != subject:
                continue
            if not _supports_categorical_text_filter(table, column):
                continue

            for variant in sorted(_column_variants(table, column), key=len, reverse=True):
                variant_text = str(variant or "").strip()
                if not variant_text:
                    continue
                c_pat = re.escape(variant_text).replace(r"\ ", r"\s+")
                match = re.search(
                    rf"\b{c_pat}\b(?:\s+of)?\s+(?:either\s+)?(?P<tail>[^.;!?]{{0,96}}?)(?={clause_stop})",
                    q_raw,
                    flags=re.I,
                )
                if not match:
                    continue

                raw_tail = str(match.group("tail") or "").strip()
                if not raw_tail or not re.search(r"\b(?:or|and)\b|,", raw_tail, flags=re.I):
                    continue

                candidate_values = re.findall(r"""["'“”]([^"'“”]+)["'“”]""", raw_tail)
                if len(candidate_values) < 2:
                    enum_values = _extract_enum_values_from_description(table, column)
                    matched_enum_values: List[str] = []
                    seen_enum_values: Set[str] = set()
                    for enum_value in enum_values:
                        clean_enum = str(enum_value or "").strip()
                        if not clean_enum:
                            continue
                        pat = r"\b" + r"\s+".join(re.escape(part) for part in clean_enum.split()) + r"\b"
                        if not re.search(pat, raw_tail, flags=re.I):
                            continue
                        enum_key = clean_enum.lower()
                        if enum_key in seen_enum_values:
                            continue
                        seen_enum_values.add(enum_key)
                        matched_enum_values.append(clean_enum)
                    candidate_values = matched_enum_values
                if len(candidate_values) < 2:
                    continue

                normalized_values: List[str] = []
                seen_values: Set[str] = set()
                for raw_value in candidate_values:
                    clean = str(raw_value or "").strip()
                    if not clean or _looks_like_nonliteral_category_phrase(clean):
                        continue
                    norm_value = _normalize_categorical_literal_value(clean, column=column)
                    if not norm_value:
                        continue
                    if _looks_like_nonliteral_category_phrase(norm_value):
                        continue
                    key = norm_value.lower()
                    if key in seen_values:
                        continue
                    seen_values.add(key)
                    normalized_values.append(norm_value)
                if len(normalized_values) < 2:
                    continue

                op = "not_in" if _has_negated_membership_prefix(match.start()) else "in"
                candidate = IRFilter(table=table, column=column, op=op, value=normalized_values)
                if best is None or int(match.start()) < best[0]:
                    best = (int(match.start()), candidate)

        if best is not None:
            return best[1]

        quoted_list_match = re.search(
            r"""(?:either\s+)?["'“”][^"'“”]+["'“”](?:\s*,\s*["'“”][^"'“”]+["'“”])*(?:\s*,?\s*(?:or|and)\s+)["'“”][^"'“”]+["'“”]""",
            q_raw,
            flags=re.I,
        )
        if not quoted_list_match:
            return None

        prefix = q_raw[max(0, quoted_list_match.start() - 80):quoted_list_match.start()]
        matched = _best_semantic_column_match(
            text=prefix,
            table=subject,
            allowed=allowed,
            min_score=1.0,
        )
        if not matched or not _supports_categorical_text_filter(subject, matched):
            return None

        quoted_values = re.findall(r"""["'“”]([^"'“”]+)["'“”]""", quoted_list_match.group(0) or "")
        normalized_values: List[str] = []
        seen_values: Set[str] = set()
        for raw_value in quoted_values:
            clean = str(raw_value or "").strip()
            if not clean or _looks_like_nonliteral_category_phrase(clean):
                continue
            norm_value = _normalize_categorical_literal_value(clean, column=matched)
            if not norm_value:
                continue
            if _looks_like_nonliteral_category_phrase(norm_value):
                continue
            key = norm_value.lower()
            if key in seen_values:
                continue
            seen_values.add(key)
            normalized_values.append(norm_value)
        if len(normalized_values) < 2:
            return None

        op = "not_in" if _has_negated_membership_prefix(quoted_list_match.start()) else "in"
        return IRFilter(table=subject, column=matched, op=op, value=normalized_values)

    def _extract_contains_filter() -> Optional[IRFilter]:
        clause_stop = (
            r"(?:[.,;:!?]|"
            r"\s+(?:but|except|excluding|return|show|list|display|include|give|sort|order|ordered|sorted|where|having)\b|$)"
        )
        match = re.search(
            rf"\bcontain(?:s|ing)?\b\s+(?:the\s+)?(?:word|words|letter|letters|text|string|phrase)?\s*"
            rf"""["'“”]?([A-Za-z0-9][A-Za-z0-9/&:' -]{{0,40}}?)["'“”]?(?={clause_stop})""",
            q_raw,
            flags=re.I,
        )
        if not match:
            return None

        neg_prefix = q_raw[max(0, match.start() - 20):match.start()]
        if re.search(r"\b(?:doesn['’]?t|does\s+not|do\s+not|not)\b", neg_prefix, flags=re.I):
            return None

        raw_value = str(match.group(1) or "").strip(" \t\r\n'\".,;:!?")
        if not raw_value:
            return None
        if _looks_like_nonliteral_category_phrase(raw_value):
            return None

        target = _pick_explicit_text_target_column(match.start())
        name_contains_intent = bool(
            re.search(
                r"\bnames?\b(?:[\s,;:()'\"-]+\w+){0,4}[\s,;:()'\"-]+contain(?:s|ing)?\b",
                q,
                flags=re.I,
            )
            or re.search(
                r"\bcontain(?:s|ing)?\b(?:[\s,;:()'\"-]+\w+){0,4}[\s,;:()'\"-]+\bnames?\b",
                q,
                flags=re.I,
            )
        )
        if not target and name_contains_intent:
            name_col = _best_name_column(subject, allowed)
            if name_col:
                target = (subject, name_col)
        if not target:
            target = _pick_text_target_column(anchor_pos=match.start())
        if not target:
            return None

        return IRFilter(
            table=target[0],
            column=target[1],
            op="like_contains",
            value=raw_value,
        )

    def _extract_not_like_contains_filter() -> Optional[IRFilter]:
        clause_stop = (
            r"(?:[.,;:!?]|"
            r"\s+(?:but|except|excluding|return|show|list|display|include|give|sort|order|ordered|sorted|where|having)\b|$)"
        )
        match = re.search(
            rf"\b(?:doesn['’]?t|does\s+not|do\s+not|not)\s+contain(?:s|ing)?\b"
            rf"(?:\s+(?:the\s+)?(?:word|words|letter|letters|text|string|phrase))?\s*"
            rf"""["'“”]?([A-Za-z0-9][A-Za-z0-9/&:' -]{{0,40}}?)["'“”]?(?={clause_stop})""",
            q_raw,
            flags=re.I,
        )
        if not match:
            match = re.search(
                rf"\bwithout\b\s+(?:the\s+)?(?:word|words|letter|letters|text|string|phrase)\s*"
                rf"""["'“”]?([A-Za-z0-9][A-Za-z0-9/&:' -]{{0,40}}?)["'“”]?(?={clause_stop})""",
                q_raw,
                flags=re.I,
            )
        if not match:
            return None

        raw_value = str(match.group(1) or "").strip(" \t\r\n'\".,;:!?")
        if not raw_value:
            return None
        if _looks_like_nonliteral_category_phrase(raw_value):
            return None

        target = _pick_explicit_text_target_column(match.start())
        name_contains_intent = bool(
            re.search(
                r"\bnames?\b(?:[\s,;:()'\"-]+\w+){0,6}[\s,;:()'\"-]+contain(?:s|ing)?\b",
                q,
                flags=re.I,
            )
            or re.search(
                r"\bcontain(?:s|ing)?\b(?:[\s,;:()'\"-]+\w+){0,6}[\s,;:()'\"-]+\bnames?\b",
                q,
                flags=re.I,
            )
            or re.search(
                r"\bnames?\b(?:[\s,;:()'\"-]+\w+){0,6}[\s,;:()'\"-]+\bwithout\b",
                q,
                flags=re.I,
            )
            or re.search(
                r"\bwithout\b(?:[\s,;:()'\"-]+\w+){0,6}[\s,;:()'\"-]+\bnames?\b",
                q,
                flags=re.I,
            )
        )
        if not target and name_contains_intent:
            name_col = _best_name_column(subject, allowed)
            if name_col:
                target = (subject, name_col)
        if not target:
            target = _pick_text_target_column(anchor_pos=match.start())
        if not target:
            return None

        return IRFilter(
            table=target[0],
            column=target[1],
            op="not_like_contains",
            value=raw_value,
        )

    def _extract_not_like_suffix_filter() -> Optional[IRFilter]:
        clause_stop = (
            r"(?:[.,;:!?]|"
            r"\s+(?:but|except|excluding|return|show|list|display|include|give|sort|order|ordered|sorted|where|having)\b|$)"
        )
        match = re.search(
            rf"\b(?:doesn['’]?t|does\s+not|do\s+not|not)\s+end(?:s)?\s+with\b"
            rf"(?:\s+(?:the\s+)?(?:word|words|letter|letters|text|string|phrase))?\s*"
            rf"""["'“”]?([A-Za-z0-9][A-Za-z0-9/&:' -]{{0,20}}?)["'“”]?(?={clause_stop})""",
            q_raw,
            flags=re.I,
        )
        if not match:
            return None

        raw_value = str(match.group(1) or "").strip(" \t\r\n'\".,;:!?")
        if not raw_value or _looks_like_nonliteral_category_phrase(raw_value):
            return None

        target = _pick_explicit_text_target_column(match.start())
        if not target:
            prefix = q_raw[max(0, match.start() - 96):match.start()]
            prefix_l = re.sub(r"\s+", " ", prefix.strip().lower())
            best_exact = None
            best_exact_len = -1
            best_exact_pos = -1
            for column in _ordered_allowed_columns(subject, allowed):
                if not _supports_categorical_text_filter(subject, column):
                    continue
                for variant in sorted(_column_reference_variants(subject, column), key=len, reverse=True):
                    variant_l = str(variant or "").strip().lower()
                    if not variant_l:
                        continue
                    pos = _phrase_pos(prefix_l, variant_l)
                    if pos < 0:
                        continue
                    if pos > best_exact_pos or (pos == best_exact_pos and len(variant_l) > best_exact_len):
                        best_exact = column
                        best_exact_len = len(variant_l)
                        best_exact_pos = pos
                        break
            if best_exact:
                target = (subject, best_exact)

            matched = _best_semantic_column_match(
                text=prefix,
                table=subject,
                allowed=allowed,
                min_score=1.0,
            )
            if not target and matched and _supports_categorical_text_filter(subject, matched):
                target = (subject, matched)
        if not target:
            target = _pick_text_target_column(anchor_pos=match.start())
        if not target:
            return None

        return IRFilter(
            table=target[0],
            column=target[1],
            op="not_like_suffix",
            value=raw_value,
        )

    def _extract_like_suffix_filter() -> Optional[IRFilter]:
        clause_stop = (
            r"(?:[.,;:!?]|"
            r"\s+(?:but|except|excluding|return|show|list|display|include|give|sort|order|ordered|sorted|where|having)\b|$)"
        )
        match = re.search(
            rf"\bend(?:s)?\s+with\b"
            rf"(?:\s+(?:the\s+)?(?:word|words|letter|letters|text|string|phrase))?\s*"
            rf"""["'“”]?([A-Za-z0-9][A-Za-z0-9/&:' -]{{0,20}}?)["'“”]?(?={clause_stop})""",
            q_raw,
            flags=re.I,
        )
        if not match:
            return None

        neg_prefix = q_raw[max(0, match.start() - 16):match.start()]
        if re.search(r"\b(?:doesn['’]?t|does\s+not|do\s+not|not)\b", neg_prefix, flags=re.I):
            return None

        raw_value = str(match.group(1) or "").strip(" \t\r\n'\".,;:!?")
        if not raw_value or _looks_like_nonliteral_category_phrase(raw_value):
            return None

        target = _pick_explicit_text_target_column(match.start())
        name_suffix_intent = bool(
            re.search(r"\bnames?\b(?:[\s,;:()'\"-]+\w+){0,4}[\s,;:()'\"-]+\bend(?:s)?\b", q, flags=re.I)
            or re.search(r"\bend(?:s)?\b(?:[\s,;:()'\"-]+\w+){0,4}[\s,;:()'\"-]+\bnames?\b", q, flags=re.I)
        )
        if not target and name_suffix_intent:
            name_col = _best_name_column(subject, allowed)
            if name_col:
                target = (subject, name_col)
        if not target:
            target = _pick_text_target_column(anchor_pos=match.start())
        if not target:
            return None

        return IRFilter(
            table=target[0],
            column=target[1],
            op="like_suffix",
            value=raw_value,
        )

    def _extract_not_like_prefix_filter() -> Optional[IRFilter]:
        clause_stop = (
            r"(?:[.,;:!?]|"
            r"\s+(?:but|except|excluding|return|show|list|display|include|give|sort|order|ordered|sorted|where|having)\b|$)"
        )
        match = re.search(
            rf"\b(?:doesn['’]?t|does\s+not|do\s+not|not)\s+(?:start(?:s)?|begin(?:s)?)\s+with\b"
            rf"(?:\s+(?:the\s+)?(?:word|words|letter|letters|text|string|phrase))?\s*"
            rf"""["'“”]?([A-Za-z0-9][A-Za-z0-9/&:' -]{{0,20}}?)["'“”]?(?={clause_stop})""",
            q_raw,
            flags=re.I,
        )
        if not match:
            return None

        raw_value = str(match.group(1) or "").strip(" \t\r\n'\".,;:!?")
        if not raw_value or _looks_like_nonliteral_category_phrase(raw_value):
            return None

        target = _pick_explicit_text_target_column(match.start())
        name_prefix_intent = bool(
            re.search(r"\bnames?\b(?:[\s,;:()'\"-]+\w+){0,4}[\s,;:()'\"-]+(?:start(?:s)?|begin(?:s)?)\b", q, flags=re.I)
            or re.search(r"\b(?:start(?:s)?|begin(?:s)?)\b(?:[\s,;:()'\"-]+\w+){0,4}[\s,;:()'\"-]+\bnames?\b", q, flags=re.I)
        )
        if not target and name_prefix_intent:
            name_col = _best_name_column(subject, allowed)
            if name_col:
                target = (subject, name_col)
        if not target:
            target = _pick_text_target_column(anchor_pos=match.start())
        if not target:
            return None

        return IRFilter(
            table=target[0],
            column=target[1],
            op="not_like_prefix",
            value=raw_value,
        )

    def _find_null_like_target_column() -> Optional[Tuple[str, str]]:
        subject_mentions = [(t, c, p) for (t, c, p) in column_mentions if t == subject]
        other_mentions = [(t, c, p) for (t, c, p) in column_mentions if t != subject]
        for t, c, _ in subject_mentions + other_mentions:
            for variant in _column_reference_variants(t, c):
                c_pat = re.escape(variant).replace(r"\ ", r"\s+")
                if re.search(
                    rf"\b{c_pat}\b(?:[\s,;:()'\"-]+\w+){{0,3}}[\s,;:()'\"-]+"
                    rf"(?:is|are|was|were)\s+(?:missing|null|empty|blank)\b",
                    q,
                    flags=re.I,
                ):
                    return (t, c)
                if re.search(
                    rf"\b(?:missing|null|empty|blank|without|lacking|no)\b"
                    rf"(?:\s+\w+){{0,4}}\s+{c_pat}\b",
                    q,
                    flags=re.I,
                ):
                    return (t, c)
                if re.search(
                    rf"\b(?:don['’]?t|doesn['’]?t|didn['’]?t|do\s+not|does\s+not|did\s+not)\s+have\b"
                    rf"(?:\s+\w+){{0,4}}\s+{c_pat}\b",
                    q,
                    flags=re.I,
                ):
                    return (t, c)
                if re.search(
                    rf"\b{c_pat}\b(?:[\s,;:()'\"-]+\w+){{0,3}}[\s,;:()'\"-]+"
                    rf"(?:listed|provided|specified|set|filled)\b",
                    q,
                    flags=re.I,
                ) and re.search(
                    r"\b(?:don['’]?t|doesn['’]?t|didn['’]?t|do\s+not|does\s+not|did\s+not|without|missing|null|empty|blank|no)\b",
                    q,
                    flags=re.I,
                ):
                    return (t, c)
        for c in _ordered_allowed_columns(subject, allowed):
            for variant in _column_reference_variants(subject, c):
                c_pat = re.escape(variant).replace(r"\ ", r"\s+")
                if re.search(
                    rf"\b{c_pat}\b(?:[\s,;:()'\"-]+\w+){{0,3}}[\s,;:()'\"-]+"
                    rf"(?:is|are|was|were)\s+(?:missing|null|empty|blank)\b",
                    q,
                    flags=re.I,
                ):
                    return (subject, c)
                if re.search(
                    rf"\b(?:missing|null|empty|blank|without|lacking|no)\b"
                    rf"(?:\s+\w+){{0,4}}\s+{c_pat}\b",
                    q,
                    flags=re.I,
                ):
                    return (subject, c)
                if re.search(
                    rf"\b(?:don['’]?t|doesn['’]?t|didn['’]?t|do\s+not|does\s+not|did\s+not)\s+have\b"
                    rf"(?:\s+\w+){{0,4}}\s+{c_pat}\b",
                    q,
                    flags=re.I,
                ):
                    return (subject, c)
        return None

    def _find_non_null_like_target_column() -> Optional[Tuple[str, str]]:
        subject_mentions = [(t, c, p) for (t, c, p) in column_mentions if t == subject]
        other_mentions = [(t, c, p) for (t, c, p) in column_mentions if t != subject]
        candidate_pairs: List[Tuple[str, str]] = []
        seen_pairs: Set[Tuple[str, str]] = set()
        for t, c, _ in subject_mentions:
            key = (t, c)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            candidate_pairs.append(key)
        for c in _ordered_allowed_columns(subject, allowed):
            key = (subject, c)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            candidate_pairs.append(key)
        for t, c, _ in other_mentions:
            key = (t, c)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            candidate_pairs.append(key)
        clause_stop = r"(?:[.,;:!?]|\s+(?:and|or|but|return|show|list|display|include|give|sort|order|ordered|sorted|where|having)\b|$)"
        for t, c in candidate_pairs:
            for variant in _column_reference_variants(t, c):
                c_pat = re.escape(variant).replace(r"\ ", r"\s+")
                if re.search(
                    rf"\b(?:have|has)\b(?:[\s,;:()'\"-]+\w+){{0,4}}[\s,;:()'\"-]+{c_pat}\b(?={clause_stop})",
                    q,
                    flags=re.I,
                ):
                    local_span = re.search(
                        rf"\b(?:have|has)\b(?:[\s,;:()'\"-]+\w+){{0,4}}[\s,;:()'\"-]+{c_pat}\b(?={clause_stop})",
                        q,
                        flags=re.I,
                    )
                    local_text = q[max(0, local_span.start() - 16):min(len(q), local_span.end() + 24)] if local_span else ""
                    if not re.search(
                        r"\b(?:along\s+with|together\s+with|highest|lowest|largest|smallest|top|bottom|"
                        r"average|avg|mean|above|below|under|over|between|at\s+least|at\s+most|"
                        r"sort(?:ed)?|order(?:ed)?\s+by|ascending|descending)\b",
                        local_text,
                        flags=re.I,
                    ):
                        return (t, c)
                if re.search(
                    rf"\bwith\b(?:[\s,;:()'\"-]+\w+){{0,2}}[\s,;:()'\"-]+{c_pat}\b(?={clause_stop})",
                    q,
                    flags=re.I,
                ):
                    local_span = re.search(
                        rf"\bwith\b(?:[\s,;:()'\"-]+\w+){{0,2}}[\s,;:()'\"-]+{c_pat}\b(?={clause_stop})",
                        q,
                        flags=re.I,
                    )
                    local_text = q[max(0, local_span.start() - 16):min(len(q), local_span.end() + 24)] if local_span else ""
                    if not re.search(
                        r"\b(?:along\s+with|together\s+with|highest|lowest|largest|smallest|top|bottom|"
                        r"average|avg|mean|above|below|under|over|between|at\s+least|at\s+most|"
                        r"sort(?:ed)?|order(?:ed)?\s+by|ascending|descending)\b",
                        local_text,
                        flags=re.I,
                    ):
                        return (t, c)
                if re.search(
                    rf"\b{c_pat}\b(?:[\s,;:()'\"-]+\w+){{0,3}}[\s,;:()'\"-]+"
                    rf"(?:filled(?:\s+in)?|present|provided|available)\b",
                    q,
                    flags=re.I,
                ):
                    return (t, c)
        if not _question_requests_conditional_null_breakdown():
            assigned_match = re.search(
                r"\bassigned\b(?P<tail>(?:[\s,;:()'\"-]+\w+){0,4})",
                q,
                flags=re.I,
            )
            if assigned_match and not re.search(r"\bunassigned\b", q, flags=re.I):
                assigned_phrase = re.sub(
                    r"\s+",
                    " ",
                    str(assigned_match.group(0) or "").strip(),
                )
                matched = _best_semantic_column_match(
                    assigned_phrase,
                    subject,
                    allowed,
                    min_score=1.0,
                )
                if matched and _is_identifier_like_column(subject, matched):
                    return (subject, matched)
        return None

    def _best_subject_self_fk_parent_column() -> Optional[Tuple[str, str]]:
        q_l = str(q or "").strip().lower()
        if not q_l or not subject:
            return None

        try:
            fk = get_fk_relationships() or {}
        except Exception:
            fk = {}

        subject_fk_map = None
        for src, fmap in (fk or {}).items():
            if str(src or "").strip().lower() == str(subject).strip().lower():
                subject_fk_map = fmap or {}
                break
        if not subject_fk_map:
            return None

        candidate_cols: List[str] = []
        allowed_cols = {
            str(c or "").strip().lower(): str(c or "").strip()
            for c in _ordered_allowed_columns(subject, allowed)
        }
        for fk_col, tgt_table in subject_fk_map.items():
            fk_col_l = str(fk_col or "").strip().lower()
            tgt_l = str(tgt_table or "").strip().lower()
            if not fk_col_l or tgt_l != str(subject).strip().lower():
                continue
            canonical = allowed_cols.get(fk_col_l, str(fk_col or "").strip())
            if canonical:
                candidate_cols.append(canonical)
        if not candidate_cols:
            return None

        q_tokens = set(_tokenize_for_match(q_l, stop_tokens=_LEXICAL_MATCH_STOP_TOKENS))
        best_col = ""
        best_score = -1
        for col in candidate_cols:
            role_tokens = _column_semantic_tokens(subject, col).union(_column_descriptor_tokens(subject, col))
            score = len(q_tokens.intersection(role_tokens))
            if score > best_score:
                best_score = score
                best_col = col

        if best_col and best_score > 0:
            return (subject, best_col)
        if len(candidate_cols) == 1:
            return (subject, candidate_cols[0])
        return None

    def _anchor_has_explicit_self_fk_parent_identifier_intent(text: str) -> bool:
        text_l = re.sub(r"\s+", " ", str(text or "").strip().lower())
        if not text_l:
            return False
        parent_role_pat = r"(?:manager|supervisor|parent|leader|boss|mentor|owner)"
        if re.search(r"\breport(?:s|ing)?\s+to\b", text_l, flags=re.I):
            return True
        if re.search(
            rf"\b{parent_role_pat}\b(?:[\s,;:()'\"-]+\w+){{0,3}}[\s,;:()'\"-]+\b(?:number|id|code)\b",
            text_l,
            flags=re.I,
        ):
            return True
        if re.search(
            rf"\b{parent_role_pat}\s+employee\s+(?:number|id|code)\b",
            text_l,
            flags=re.I,
        ):
            return True
        return False

    def _find_required_self_fk_parent_column() -> Optional[Tuple[str, str]]:
        q_l = str(q or "").strip().lower()
        if not q_l or not subject:
            return None

        parent_role_terms = {
            "manager", "supervisor", "parent", "leader", "boss", "mentor", "owner",
        }
        negative_parent_intent = bool(
            re.search(
                r"\b(?:do(?:es)?\s+not|doesn['’]?t|did(?:\s+not)?|didn['’]?t|not)\s+report(?:s|ing)?\s+to\b"
                r"|\breport(?:s|ing)?\s+to\s+anyone\b"
                r"|\b(?:without|no)\s+(?:a\s+)?(?:manager|supervisor|parent|leader|boss|mentor|owner)\b"
                r"|\b(?:do(?:es)?\s+not|doesn['’]?t|not)\s+have\b[^.]{0,32}\b(?:manager|supervisor|parent|leader|boss|mentor|owner)\b",
                q_l,
                flags=re.I,
            )
        )
        if negative_parent_intent:
            return None
        if not (
            re.search(r"\breport(?:s|ing)?\s+to\b", q_l, flags=re.I)
            or (
                re.search(r"\b(?:with|has|have|having|under|managed\s+by)\b", q_l, flags=re.I)
                and any(re.search(rf"\b{re.escape(term)}\b", q_l, flags=re.I) for term in parent_role_terms)
            )
        ):
            return None
        return _best_subject_self_fk_parent_column()

    def _find_missing_self_fk_parent_column() -> Optional[Tuple[str, str]]:
        q_l = str(q or "").strip().lower()
        if not q_l or not subject:
            return None
        if not re.search(
            r"\b(?:do(?:es)?\s+not|doesn['’]?t|did(?:\s+not)?|didn['’]?t|not)\s+report(?:s|ing)?\s+to\b"
            r"|\breport(?:s|ing)?\s+to\s+anyone\b"
            r"|\b(?:without|no)\s+(?:a\s+)?(?:manager|supervisor|parent|leader|boss|mentor|owner)\b"
            r"|\b(?:do(?:es)?\s+not|doesn['’]?t|not)\s+have\b[^.]{0,32}\b(?:manager|supervisor|parent|leader|boss|mentor|owner)\b",
            q_l,
            flags=re.I,
        ):
            return None
        return _best_subject_self_fk_parent_column()

    def _subject_projection_comparable_columns() -> List[str]:
        projected_cols: List[str] = []
        seen: Set[str] = set()
        for dim in dimensions or []:
            table = str(getattr(dim, "table", "") or "").strip().lower()
            column = str(getattr(dim, "column", "") or "").strip()
            if table != subject or not column:
                continue
            if not (_is_numeric_column(column, table) or _is_date_column(column, table)):
                continue
            key = column.lower()
            if key in seen:
                continue
            seen.add(key)
            projected_cols.append(column)
        return projected_cols

    def _resolve_subject_comparison_column(phrase: str) -> Optional[str]:
        phrase_text = re.sub(r"\s+", " ", str(phrase or "").strip().lower())
        if not phrase_text:
            return None

        projected_cols = _subject_projection_comparable_columns()
        candidate_cols = projected_cols or [
            c for c in _ordered_allowed_columns(subject, allowed)
            if _is_numeric_column(c, subject) or _is_date_column(c, subject)
        ]
        if not candidate_cols:
            return None

        best_exact = None
        best_exact_len = -1
        for column in candidate_cols:
            for variant in sorted(_column_reference_variants(subject, column), key=len, reverse=True):
                variant_l = str(variant or "").strip().lower()
                if variant_l and _phrase_pos(phrase_text, variant_l) >= 0 and len(variant_l) > best_exact_len:
                    best_exact = column
                    best_exact_len = len(variant_l)
                    break
        if best_exact:
            return best_exact

        semantic_match = _best_semantic_column_match(
            text=phrase_text,
            table=subject,
            allowed=allowed,
            min_score=1.0,
        )
        if semantic_match and semantic_match in candidate_cols:
            return semantic_match

        phrase_tokens = _semantic_query_token_variants(
            _tokenize_for_match(phrase_text, stop_tokens=_LEXICAL_MATCH_STOP_TOKENS)
        )
        if not phrase_tokens:
            return None
        best_col = None
        best_score = 0.0
        best_overlap = 0
        for column in candidate_cols:
            semantic_tokens = _column_semantic_tokens(subject, column).union(_column_descriptor_tokens(subject, column))
            overlap = phrase_tokens.intersection(semantic_tokens)
            if not overlap:
                continue
            score = float(len(overlap))
            score += (len(overlap) / max(len(semantic_tokens), 1)) * 0.5
            if projected_cols and column in projected_cols:
                score += 0.25
            if score > best_score or (score == best_score and best_col is not None and len(column) < len(best_col)):
                best_col = column
                best_score = score
                best_overlap = len(overlap)

        if best_col is not None and (best_score >= 1.0 or best_overlap >= 2):
            return best_col
        return None

    # Numeric comparisons.
    numeric_col = None
    numeric_mentions = [(t, c, p) for (t, c, p) in column_mentions if _is_numeric_column(c, t)]

    def _pick_comparison_numeric_target(anchor_pos: int) -> Optional[Tuple[str, str]]:
        numeric_candidate_tables = [subject]
        for t, _, _ in column_mentions:
            if t not in numeric_candidate_tables:
                numeric_candidate_tables.append(t)

        window_start = max(0, int(anchor_pos) - 64)
        anchor_window = q[window_start:int(anchor_pos)]
        anchor_focus = re.split(r"\b(?:and|but|or|while)\b", anchor_window, flags=re.I)[-1].strip()
        anchor_tokens = set(
            _tokenize_for_match(
                anchor_window,
                stop_tokens=_LEXICAL_MATCH_STOP_TOKENS.union(
                    {
                        "least", "most", "minimum", "maximum", "greater", "less",
                        "more", "above", "below", "under", "over", "between",
                        "than", "equal", "equals", "exactly",
                    }
                ),
            )
        )
        local_candidate_tables: List[str] = []
        local_table_distance: Dict[str, int] = {}
        for t, _, pos in column_mentions:
            if pos < window_start or pos > int(anchor_pos):
                continue
            if t not in local_candidate_tables:
                local_candidate_tables.append(t)
            local_table_distance[t] = min(local_table_distance.get(t, 10**9), abs(pos - int(anchor_pos)))
        if not local_candidate_tables:
            local_candidate_tables = list(numeric_candidate_tables)
        elif subject not in local_candidate_tables:
            local_candidate_tables.append(subject)

        if anchor_focus and anchor_focus != anchor_window.strip():
            best_focus_target: Optional[Tuple[str, str]] = None
            best_focus_score = 0.0
            for table in local_candidate_tables:
                matched = _best_semantic_column_match(
                    text=anchor_focus,
                    table=table,
                    allowed=allowed,
                    numeric_only=True,
                    min_score=1.0,
                )
                if not matched:
                    continue
                overlap = _semantic_query_token_variants(
                    _tokenize_for_match(anchor_focus, stop_tokens=_LEXICAL_MATCH_STOP_TOKENS)
                ).intersection(_column_semantic_tokens(table, matched))
                score = float(len(overlap))
                if table == subject:
                    score += 0.25
                if score > best_focus_score:
                    best_focus_score = score
                    best_focus_target = (table, matched)
            if best_focus_target is not None and best_focus_score >= 1.0:
                return best_focus_target

        best_semantic_target: Optional[Tuple[str, str]] = None
        best_semantic_score = 0.0
        for table in local_candidate_tables:
            matched = _best_semantic_column_match(
                text=anchor_window,
                table=table,
                allowed=allowed,
                numeric_only=True,
                min_score=1.0,
            )
            if not matched:
                continue
            overlap = anchor_tokens.intersection(_column_semantic_tokens(table, matched))
            score = float(len(overlap))
            nearest_distance = int(local_table_distance.get(table, 10**9))
            score += max(0.0, 1.25 - (nearest_distance / 48.0))
            if table == subject:
                score += 0.25
            if score > best_semantic_score:
                best_semantic_score = score
                best_semantic_target = (table, matched)

        best_local: Optional[Tuple[str, str]] = None
        best_score = float("-inf")
        best_distance = 10**9
        for t, c, pos in numeric_mentions:
            semantic_tokens = _column_semantic_tokens(t, c)
            overlap = anchor_tokens.intersection(semantic_tokens)
            distance = abs(pos - int(anchor_pos))
            score = float(len(overlap) * 3)
            best_variant_len = 0
            for variant in sorted(_column_reference_variants(t, c), key=len, reverse=True):
                variant_l = str(variant or "").strip().lower()
                if variant_l and _phrase_pos(anchor_window, variant_l) >= 0:
                    best_variant_len = len(variant_l)
                    break
            if best_variant_len > 0:
                score += 4.0 + (best_variant_len / 32.0)
            if pos <= int(anchor_pos):
                score += 0.6
            score += max(0.0, 1.5 - (distance / 48.0))
            if t == subject:
                score += 0.35
            if (overlap or best_variant_len > 0) and (
                score > best_score or (score == best_score and distance < best_distance)
            ):
                best_local = (t, c)
                best_score = score
                best_distance = distance

        if best_local is not None and best_distance <= 32:
            return best_local
        if best_semantic_target is not None:
            return best_semantic_target
        if best_local is not None:
            return best_local

        if not numeric_mentions:
            return None

        nearest_before = [
            (t, c, pos) for (t, c, pos) in numeric_mentions
            if pos <= int(anchor_pos)
        ]
        if nearest_before:
            t, c, _ = min(nearest_before, key=lambda x: abs(x[2] - int(anchor_pos)))
            return (t, c)

        t, c, _ = min(numeric_mentions, key=lambda x: abs(x[2] - int(anchor_pos)))
        return (t, c)

    cond_pos_candidates: List[int] = []
    for pat in (
        r"\bbetween\b",
        r"\bat\s+least\b",
        r"\bat\s+most\b",
        r"\bminimum\b",
        r"\bmaximum\b",
        r"\bgreater\s+than\b",
        r"\bmore\s+than\b",
        r"\babove\b",
        r"\bover\b",
        r"\bless\s+than\b",
        r"\bunder\b",
        r"\bbelow\b",
    ):
        m_cond = re.search(pat, q)
        if m_cond:
            cond_pos_candidates.append(m_cond.start())
    cond_pos = min(cond_pos_candidates) if cond_pos_candidates else -1

    if numeric_mentions:
        if cond_pos >= 0:
            t, c, _ = min(numeric_mentions, key=lambda x: abs(x[2] - cond_pos))
            numeric_col = (t, c)
        else:
            t, c, _ = numeric_mentions[0]
            numeric_col = (t, c)

    if not numeric_col:
        preferred = _prefer_numeric_column_from_question(q, subject, allowed)
        if preferred:
            numeric_col = (subject, preferred)

    if not numeric_col and metrics:
        m0 = metrics[0]
        m0_agg = str(m0.agg or "").strip().lower()
        if m0.column != "*" and (_is_numeric_column(m0.column, m0.table) or m0_agg == "count"):
            numeric_col = (m0.table, m0.column)
    if not numeric_col:
        c0 = _find_best_numeric_column(subject, allowed)
        if c0:
            numeric_col = (subject, c0)

    m_between = re.search(
        rf"\bbetween\s+({_NUMBER_TOKEN_PATTERN})\s+and\s+({_NUMBER_TOKEN_PATTERN})\b",
        q,
    )
    m_range = re.search(
        rf"\bfrom\s+({_NUMBER_TOKEN_PATTERN})\s+(?:through|thru|to)\s+({_NUMBER_TOKEN_PATTERN})(?:\s+inclusive)?\b",
        q,
    )
    range_match = m_between or m_range
    range_target = _pick_exact_numeric_target_column(range_match.start()) if range_match else None
    if range_match and not range_target:
        range_target = _pick_comparison_numeric_target(range_match.start())
    if range_match and (range_target or numeric_col):
        v1 = _to_float_token(range_match.group(1))
        v2 = _to_float_token(range_match.group(2))
        t, c = range_target or numeric_col
        if v1 is not None and v2 is not None:
            add_filter(IRFilter(table=t, column=c, op="between", value=v1, value2=v2))

    m_gte = re.search(rf"\b(?:at\s+least|minimum\s+of)\s+({_NUMBER_TOKEN_PATTERN})\b", q)
    gte_target = _pick_comparison_numeric_target(m_gte.start()) if m_gte else None
    if m_gte and (gte_target or numeric_col):
        threshold_tail = _comparison_clause_tail(m_gte)
        row_count_threshold = _threshold_has_row_count_tail(threshold_tail)
        if not row_count_threshold and not _is_aggregate_bound_threshold(m_gte):
            v = _to_float_token(m_gte.group(1))
            t, c = gte_target or numeric_col
            if v is not None:
                add_filter(IRFilter(table=t, column=c, op="gte", value=v))

    m_lte = re.search(rf"\b(?:at\s+most|maximum\s+of)\s+({_NUMBER_TOKEN_PATTERN})\b", q)
    lte_target = _pick_comparison_numeric_target(m_lte.start()) if m_lte else None
    if m_lte and (lte_target or numeric_col) and not _is_aggregate_bound_threshold(m_lte):
        v = _to_float_token(m_lte.group(1))
        t, c = lte_target or numeric_col
        if v is not None:
            add_filter(IRFilter(table=t, column=c, op="lte", value=v))

    m_gt = re.search(rf"\b(?:greater\s+than|more\s+than|above|over)\s+({_NUMBER_TOKEN_PATTERN})\b", q)
    gt_target = _pick_comparison_numeric_target(m_gt.start()) if m_gt else None
    if m_gt and (gt_target or numeric_col) and not _is_aggregate_bound_threshold(m_gt):
        v = _to_float_token(m_gt.group(1))
        t, c = gt_target or numeric_col
        if v is not None:
            add_filter(IRFilter(table=t, column=c, op="gt", value=v))

    m_lt = re.search(rf"\b(?:less\s+than|under|below)\s+({_NUMBER_TOKEN_PATTERN})\b", q)
    lt_target = _pick_comparison_numeric_target(m_lt.start()) if m_lt else None
    if m_lt and (lt_target or numeric_col) and not _is_aggregate_bound_threshold(m_lt):
        v = _to_float_token(m_lt.group(1))
        t, c = lt_target or numeric_col
        if v is not None:
            add_filter(IRFilter(table=t, column=c, op="lt", value=v))

    m_eq_num = re.search(
        rf"\b(?:is|equals?|equal\s+to|=)\s+({_NUMBER_TOKEN_PATTERN})(?![:/-]\d)\b",
        q,
        flags=re.I,
    )
    if m_eq_num:
        v = _to_float_token(m_eq_num.group(1))
        target = _pick_exact_numeric_target_column(m_eq_num.start())
        if v is not None and target:
            add_filter(IRFilter(table=target[0], column=target[1], op="eq", value=v))

    # Explicit full-date filters take priority over generic year-only fallback.
    full_date_between_match = re.search(
        r"\bbetween\s+((?:19|20)\d{2}-\d{2}-\d{2})\s+and\s+((?:19|20)\d{2}-\d{2}-\d{2})\b",
        q,
        flags=re.I,
    )
    full_date_from_to_match = re.search(
        r"\bfrom\s+((?:19|20)\d{2}-\d{2}-\d{2})\s+(?:through|thru|to)\s+((?:19|20)\d{2}-\d{2}-\d{2})\b",
        q,
        flags=re.I,
    )
    full_date_range_match = full_date_between_match or full_date_from_to_match
    full_date_match = re.search(r"\b((?:19|20)\d{2}-\d{2}-\d{2})\b", q)
    if full_date_range_match:
        date_table, date_col = _pick_date_target_column()
        if date_col:
            start_date = str(full_date_range_match.group(1) or "").strip()
            end_date = str(full_date_range_match.group(2) or "").strip()
            if start_date and end_date:
                add_filter(IRFilter(table=date_table, column=date_col, op="between", value=start_date, value2=end_date))

    if full_date_match:
        date_literal = full_date_match.group(1)
        date_table, date_col = _pick_date_target_column()
        if date_col:
            date_prefix = q[:full_date_match.start()]
            if re.search(r"\bon\s+or\s+after\b", date_prefix):
                add_filter(IRFilter(table=date_table, column=date_col, op="gte", value=date_literal))
            elif re.search(r"\bon\s+or\s+before\b", date_prefix):
                add_filter(IRFilter(table=date_table, column=date_col, op="lte", value=date_literal))
            elif re.search(r"\b(?:after|later\s+than)\b", date_prefix):
                add_filter(IRFilter(table=date_table, column=date_col, op="gt", value=date_literal))
            elif re.search(r"\b(?:before|earlier\s+than)\b", date_prefix):
                add_filter(IRFilter(table=date_table, column=date_col, op="lt", value=date_literal))

    weekday_match = re.search(
        r"\b(?:on\s+(?:a\s+)?)?(sunday|monday|tuesday|wednesday|thursday|friday|saturday)\b",
        q,
        flags=re.I,
    )
    if weekday_match and re.search(r"\b(?:placed|made|occurr?ed|happened|on)\b", q, flags=re.I):
        date_table, date_col = _pick_date_target_column()
        weekday_name = str(weekday_match.group(1) or "").strip().lower()
        weekday_num = weekday_name_to_num.get(weekday_name)
        if date_col and weekday_num is not None:
            add_filter(IRFilter(table=date_table, column=date_col, op="weekday_eq", value=weekday_num))

    quarter_map = {
        "first": (1, 3),
        "1st": (1, 3),
        "q1": (1, 3),
        "second": (4, 6),
        "2nd": (4, 6),
        "q2": (4, 6),
        "third": (7, 9),
        "3rd": (7, 9),
        "q3": (7, 9),
        "fourth": (10, 12),
        "4th": (10, 12),
        "q4": (10, 12),
    }

    def _question_uses_quarter_as_bucket() -> bool:
        return bool(
            re.search(
                r"\b(?:group\s+the|group(?:ed)?\s+by|by|per|for\s+each)\s+"
                r"(?:the\s+)?(?:calendar\s+)?quarter\b",
                q,
                flags=re.I,
            )
            or re.search(
                r"\bquarter\s+number\b",
                q,
                flags=re.I,
            )
        )

    def _quarter_match_is_filter_context(match_obj: re.Match[str]) -> bool:
        token = str(match_obj.group(1) or match_obj.group(2) or "").strip().lower()
        prefix = q[max(0, match_obj.start() - 40):match_obj.start()]
        suffix = q[match_obj.end():match_obj.end() + 32]
        if token.startswith("q"):
            if re.search(r"\b(?:sort|sorted|order|ordered)\s+from\s*$", prefix, flags=re.I):
                return False
            if re.match(r"\s+to\s+q[1-4]\b", suffix, flags=re.I):
                return False
            if re.match(r"\s+to\s+quarter\s+[1-4]\b", suffix, flags=re.I):
                return False
            if _question_uses_quarter_as_bucket():
                return False
        return True

    quarter_match = re.search(
        r"\b(?:(first|1st|second|2nd|third|3rd|fourth|4th)\s+quarter|(q[1-4]))\b",
        q,
        flags=re.I,
    )
    month_range_match = re.search(
        r"\b(" + "|".join(sorted(_MONTH_NAME_TO_NUM.keys(), key=len, reverse=True)) + r")\s+(?:through|thru|to|and)\s+("
        + "|".join(sorted(_MONTH_NAME_TO_NUM.keys(), key=len, reverse=True)) + r")\b",
        q,
        flags=re.I,
    )
    date_table, date_col = _pick_date_target_column()
    if quarter_match and date_col and _quarter_match_is_filter_context(quarter_match):
        quarter_token = str(quarter_match.group(1) or quarter_match.group(2) or "").strip().lower()
        bounds = quarter_map.get(quarter_token)
        if bounds:
            add_filter(IRFilter(table=date_table, column=date_col, op="month_between", value=int(bounds[0]), value2=int(bounds[1])))
    elif month_range_match and date_col:
        start_month = _MONTH_NAME_TO_NUM.get(str(month_range_match.group(1) or "").strip().lower())
        end_month = _MONTH_NAME_TO_NUM.get(str(month_range_match.group(2) or "").strip().lower())
        if start_month is not None and end_month is not None:
            lo, hi = sorted((int(start_month), int(end_month)))
            add_filter(IRFilter(table=date_table, column=date_col, op="month_between", value=lo, value2=hi))

    month_match = re.search(
        r"\b(?:month\s+of\s+)?(" + "|".join(sorted(_MONTH_NAME_TO_NUM.keys(), key=len, reverse=True)) + r")\b",
        q,
        flags=re.I,
    )
    if month_match and not quarter_match and not month_range_match:
        month_num = _MONTH_NAME_TO_NUM.get(str(month_match.group(1) or "").strip().lower())
        if month_num is not None and date_col:
            add_filter(IRFilter(table=date_table, column=date_col, op="month_eq", value=int(month_num)))

    # Year filters.
    years = re.findall(r"\b((?:19|20)\d{2})\b", q)
    if years and not full_date_match:
        date_table, date_col = _pick_date_target_column()
        if date_col:
            add_filter(IRFilter(table=date_table, column=date_col, op="year_eq", value=int(years[0])))

    # Same-table column-to-column comparison (conservative):
    # Prefer subject-table semantic resolution from the comparison phrase itself.
    # This keeps detail-row comparisons on the subject table instead of drifting
    # to similarly described columns in related tables.
    if not metrics:
        compare_phrase_specs = (
            (r"\b(?:is\s+)?equal\s+to\b", "col_eq"),
            (r"\bequals\b", "col_eq"),
            (r"\b(?:is\s+)?(?:the\s+)?same\s+as\b", "col_eq"),
            (r"\bon\s+or\s+after\b", "col_gte"),
            (r"\bon\s+or\s+before\b", "col_lte"),
            (r"\b(?:later\s+than|(?<!on or )after)\b", "col_gt"),
            (r"\b(?:earlier\s+than|(?<!on or )before)\b", "col_lt"),
            (r"\b(?:is\s+)?(?:higher|greater)\s+than\b", "col_gt"),
            (r"\b(?:is\s+)?more\s+than\b", "col_gt"),
            (r"\b(?:is\s+)?(?:lower|less)\s+than\b", "col_lt"),
        )
        for pat, matched_op in compare_phrase_specs:
            for m_compare in re.finditer(pat, q, flags=re.I):
                left_window = q[max(0, m_compare.start() - 80):m_compare.start()]
                left_phrase = re.split(
                    r"\b(?:where|whose|with|that|which|who|were|was|are|is)\b",
                    left_window,
                    maxsplit=0,
                    flags=re.I,
                )[-1].strip()
                right_window = q[m_compare.end():m_compare.end() + 80]
                right_phrase = re.split(
                    r"[.,;:!?]|\b(?:return|show|list|display|include|give|sort|order|ordered|sorted)\b",
                    right_window,
                    maxsplit=1,
                    flags=re.I,
                )[0].strip()
                left_col = _resolve_subject_comparison_column(left_phrase)
                right_col = _resolve_subject_comparison_column(right_phrase)
                comparable_pair = bool(
                    left_col and right_col and left_col.lower() != right_col.lower() and (
                        (_is_date_column(left_col, subject) and _is_date_column(right_col, subject))
                        or (_is_numeric_column(left_col, subject) and _is_numeric_column(right_col, subject))
                    )
                )
                if comparable_pair:
                    add_filter(
                        IRFilter(
                            table=subject,
                            column=left_col,
                            op=matched_op,
                            value_column=right_col,
                        )
                    )

        # Conservative fallback: explicit same-table date-like column mentions
        # with a short span between them.
        compare_specs = (
            (r"\b(?:is\s+)?equal\s+to\b", "col_eq"),
            (r"\bequals\b", "col_eq"),
            (r"\b(?:is\s+)?(?:the\s+)?same\s+as\b", "col_eq"),
            (r"\bon\s+or\s+after\b", "col_gte"),
            (r"\bon\s+or\s+before\b", "col_lte"),
            (r"\b(?:later\s+than|(?<!on or )after)\b", "col_gt"),
            (r"\b(?:earlier\s+than|(?<!on or )before)\b", "col_lt"),
        )
        date_mentions = [
            (t, c, p) for (t, c, p) in column_mentions if _is_date_column(c, t)
        ]
        best_compare = None
        best_gap = 10**9
        for i, (t1, c1, p1) in enumerate(date_mentions):
            for t2, c2, p2 in date_mentions[i + 1:]:
                if t1 != t2 or c1.lower() == c2.lower():
                    continue
                if p1 <= p2:
                    left_table, left_col, left_pos = t1, c1, p1
                    right_col, right_pos = c2, p2
                else:
                    left_table, left_col, left_pos = t2, c2, p2
                    right_col, right_pos = c1, p1
                gap = right_pos - left_pos
                if gap <= 0 or gap > 96:
                    continue
                between = q[left_pos:right_pos]
                if re.search(r"\b(?:most\s+recent|latest|newest|earliest|oldest)\b", between, flags=re.I):
                    continue
                matched_op = ""
                for pat, op in compare_specs:
                    if re.search(pat, between, flags=re.I):
                        matched_op = op
                        break
                if not matched_op:
                    continue
                if gap < best_gap:
                    best_gap = gap
                    best_compare = (left_table, left_col, matched_op, right_col)

        if best_compare is not None:
            left_table, left_col, matched_op, right_col = best_compare
            add_filter(
                IRFilter(
                    table=left_table,
                    column=left_col,
                    op=matched_op,
                    value_column=right_col,
                )
            )

    # Country/city style location filter:
    # "customers in Japan", "customers from USA".
    def _pick_categorical_target_column(text_hint: str = "") -> Optional[str]:
        hint = str(text_hint or "").strip()
        if hint:
            matched = _best_semantic_column_match(
                text=hint,
                table=subject,
                allowed=allowed,
                min_score=1.0,
            )
            if matched and not _is_date_column(matched, subject) and not _is_numeric_column(matched, subject):
                return matched
        for preferred in ("country", "territory", "state", "city"):
            for c in _ordered_allowed_columns(subject, allowed):
                if c.lower() == preferred:
                    return c
        projected_candidates = _projected_subject_text_candidates()
        if projected_candidates:
            return projected_candidates[0]
        return None

    def _membership_intro_is_negative(match_obj: Optional[re.Match]) -> bool:
        intro = str(match_obj.group("intro") or "").strip().lower() if match_obj else ""
        return bool(intro and intro.startswith(("outside", "excluding", "except")))

    location_match = re.search(
        r"\b(?P<intro>in|from|outside|excluding|except(?:\s+for)?)\s+(?P<value>[A-Za-z][A-Za-z\s-]{1,40}?)"
        r"(?=\s+(?:ordered|sorted|grouped|having|where|with|without|show|return|list|display|include|give)\b"
        r"|\s+and\s+(?:show|return|list|display|include|give)\b|\s+(?:whose|that|which)\b|[.,;:!?]|\s*$)",
        q_raw,
        flags=re.I,
    )
    location_list_match = re.search(
        r"\b(?P<intro>in|from|outside|excluding|except(?:\s+for)?)\s+(?P<value>[A-Za-z][A-Za-z\s-]{1,80}?(?:\s*,\s*[A-Za-z][A-Za-z\s-]{1,40})*(?:\s*,?\s*(?:or|and)\s+[A-Za-z][A-Za-z\s-]{1,40})+)"
        r"(?=\s+(?:ordered|sorted|grouped|having|where|with|without|show|return|list|display|include|give)\b"
        r"|\s+and\s+(?:show|return|list|display|include|give)\b|\s+(?:whose|that|which)\b|[.,;:!?]|\s*$)",
        q_raw,
        flags=re.I,
    )
    if location_list_match and not _question_span_is_column_reference(location_list_match.start(), location_list_match.end()):
        raw_list = str(location_list_match.group("value") or "").strip().strip(".,;:!?")
        raw_list = re.split(
            r"\b(?:show|return|list|display|include|give)\b",
            raw_list,
            maxsplit=1,
            flags=re.I,
        )[0].strip(" \t\r\n'\".,;:!?")
        raw_list = re.sub(r"\s+(?:or|and)\s+", ", ", raw_list, flags=re.I)
        values = [
            part.strip(" \t\r\n'\".,;:!?")
            for part in raw_list.split(",")
            if part.strip(" \t\r\n'\".,;:!?")
        ]
        if _looks_like_nonliteral_category_phrase(raw_list):
            values = []
        values = list(dict.fromkeys(values))
        if len(values) == 1 and _looks_like_alphabetical_sort_phrase(values[0]):
            values = []
        if len(values) >= 2:
            target_col = _pick_categorical_target_column(raw_list)
            if target_col:
                values = [
                    _normalize_categorical_literal_value(v, column=target_col)
                    for v in values
                    if _normalize_categorical_literal_value(v, column=target_col)
                ]
                op = "not_in" if (
                    _membership_intro_is_negative(location_list_match)
                    or _has_negated_membership_prefix(location_list_match.start())
                ) else "in"
                add_filter(IRFilter(table=subject, column=target_col, op=op, value=values))
                location_match = None
    if location_match and not _question_span_is_column_reference(location_match.start(), location_match.end()):
        loc = str(location_match.group("value") or "").strip().strip(".,;:!?")
        loc_low = loc.lower()
        stop_loc = {
            "the", "a", "an", "that", "this", "those", "these",
            "latest", "earliest", "highest", "lowest",
            "year", "month", "week", "day",
            "a to z", "z to a",
        }
        quantifier_starts = ("each ", "every ", "all ", "per ")
        is_deictic_ref = bool(re.search(r"^(?:that|this|those|these|their|its)\b", loc_low))
        loc_tokens = [t for t in re.findall(r"[a-z0-9]+", loc_low) if t]
        content_tokens = [t for t in loc_tokens if t not in _FUNCTION_WORD_TOKENS]

        # Schema-aware check: phrases composed only of schema terms
        # (e.g., "that territory", "this status") are references, not literal values.
        schema_terms: Set[str] = set()
        for c in _ordered_allowed_columns(subject, allowed):
            c_words = [w for w in re.findall(r"[a-z0-9]+", _camel_to_words(c)) if w]
            schema_terms.update(c_words)
            for syn in _safe_get_column_synonyms(subject, c):
                schema_terms.update(w for w in re.findall(r"[a-z0-9]+", str(syn).lower()) if w)
        is_schema_reference_phrase = bool(content_tokens) and all(t in schema_terms for t in content_tokens)
        if (
            loc_low
            and loc_low not in stop_loc
            and not re.search(r"\d", loc_low)
            and not is_deictic_ref
            and not any(loc_low.startswith(prefix) for prefix in quantifier_starts)
            and not is_schema_reference_phrase
            and not _looks_like_nonliteral_category_phrase(loc_low)
            and not _looks_like_alphabetical_sort_phrase(loc_low)
        ):
            target_col = _pick_categorical_target_column(loc)
            if target_col:
                norm_loc = _normalize_categorical_literal_value(loc, column=target_col)
                if norm_loc:
                    if (
                        _membership_intro_is_negative(location_match)
                        or _has_negated_membership_prefix(location_match.start())
                    ):
                        add_filter(IRFilter(table=subject, column=target_col, op="not_in", value=[norm_loc]))
                    else:
                        add_filter(IRFilter(table=subject, column=target_col, op="eq", value=norm_loc))

    explicit_categorical_in = _extract_explicit_categorical_in()
    if explicit_categorical_in is not None:
        add_filter(explicit_categorical_in)

    explicit_categorical_eq = _extract_explicit_categorical_eq()
    if explicit_categorical_eq is not None:
        add_filter(explicit_categorical_eq)

    not_like_contains_filter = _extract_not_like_contains_filter()
    if not_like_contains_filter is not None:
        add_filter(not_like_contains_filter)

    contains_filter = _extract_contains_filter()
    if contains_filter is not None:
        add_filter(contains_filter)

    not_like_suffix_filter = _extract_not_like_suffix_filter()
    if not_like_suffix_filter is not None:
        add_filter(not_like_suffix_filter)

    like_suffix_filter = _extract_like_suffix_filter()
    if like_suffix_filter is not None:
        add_filter(like_suffix_filter)

    not_like_prefix_filter = _extract_not_like_prefix_filter()
    if not_like_prefix_filter is not None:
        add_filter(not_like_prefix_filter)

    # Status IN ('a','b',...)
    quoted_values = re.findall(r"""["'“”]([^"'“”]+)["'“”]""", question or "")
    if quoted_values and re.search(r"\bstatus\b", q):
        status_col = None
        status_table = subject
        for c in _ordered_allowed_columns(subject, allowed):
            if "status" in c.lower():
                status_col = c
                break
        if status_col:
            if len(quoted_values) >= 2:
                add_filter(IRFilter(table=status_table, column=status_col, op="in", value=[v.strip() for v in quoted_values]))
            else:
                add_filter(IRFilter(table=status_table, column=status_col, op="eq", value=quoted_values[0].strip()))

    negated_process_terms = _extract_negated_process_terms()
    if negated_process_terms:
        proxy_col = _find_best_process_proxy_column(subject, negated_process_terms)
        enum_col = _find_best_enum_descriptor_column(subject)
        if proxy_col:
            add_filter(IRFilter(table=subject, column=proxy_col, op="is_null_or_empty"))
        if proxy_col and enum_col:
            candidate_values = _derive_nonfinal_enum_values(subject, enum_col)
            if candidate_values and _question_mentions_enum_values(candidate_values):
                if len(candidate_values) >= 2:
                    add_filter(IRFilter(table=subject, column=enum_col, op="in", value=candidate_values))
                else:
                    add_filter(IRFilter(table=subject, column=enum_col, op="eq", value=candidate_values[0]))

    # Missing/empty (skip when intent is explicit non-null/not-null).
    if (
        re.search(r"\b(missing|null|empty|blank)\b", q)
        and not _question_requests_conditional_null_breakdown()
        and not re.search(r"\b(?:non[-\s]?null|not\s+null)\b", q)
        and not _question_requests_null_fill_projection(question)
    ):
        target = None
        for t, c, _ in column_mentions:
            if "status" in c.lower() or "state" in c.lower() or "name" in c.lower():
                target = (t, c)
                break
        if not target:
            for c in _ordered_allowed_columns(subject, allowed):
                if "state" in c.lower():
                    target = (subject, c)
                    break
        if target:
            add_filter(IRFilter(table=target[0], column=target[1], op="is_null_or_empty"))

    if (
        not _question_requests_conditional_null_breakdown()
        and
        not re.search(r"\b(?:non[-\s]?null|not\s+null)\b", q)
        and not _question_requests_null_fill_projection(question)
    ):
        null_like_target = _find_null_like_target_column()
        if null_like_target:
            add_filter(IRFilter(table=null_like_target[0], column=null_like_target[1], op="is_null_or_empty"))

        missing_self_fk_parent_target = _find_missing_self_fk_parent_column()
        if missing_self_fk_parent_target:
            add_filter(
                IRFilter(
                    table=missing_self_fk_parent_target[0],
                    column=missing_self_fk_parent_target[1],
                    op="is_null_or_empty",
                )
            )

        self_fk_parent_target = None if missing_self_fk_parent_target else _find_required_self_fk_parent_column()
        if self_fk_parent_target:
            has_specific_parent_filter = any(
                str(getattr(flt, "table", "") or "").strip().lower() == str(self_fk_parent_target[0] or "").strip().lower()
                and str(getattr(flt, "column", "") or "").strip().lower() == str(self_fk_parent_target[1] or "").strip().lower()
                and str(getattr(flt, "op", "") or "").strip().lower() not in {"is_null_or_empty", "is_not_null_or_empty"}
                for flt in (filters or [])
            )
            if not has_specific_parent_filter:
                add_filter(
                    IRFilter(
                        table=self_fk_parent_target[0],
                        column=self_fk_parent_target[1],
                        op="is_not_null_or_empty",
                    )
                )

        non_null_like_target = _find_non_null_like_target_column()
        if non_null_like_target:
            add_filter(
                IRFilter(
                    table=non_null_like_target[0],
                    column=non_null_like_target[1],
                    op="is_not_null_or_empty",
                )
            )

    # Prefix and regex pattern.
    strict_regex_applied = False

    # Pattern: "starts with two uppercase letters, followed by exactly 5 digits,
    # and ends with any single digit" (single-table deterministic regex).
    m_upper_start = re.search(
        r"\bstart(?:s)?\s+with\s+(?:exactly\s+)?(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+uppercase\s+letters?\b",
        q,
        flags=re.I,
    )
    m_exact_digits = re.search(
        r"\bfollowed\s+by\s+exactly\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+digits?\b",
        q,
        flags=re.I,
    )
    if m_upper_start and m_exact_digits:
        n_letters = _to_int_count_token(m_upper_start.group(1))
        n_digits = _to_int_count_token(m_exact_digits.group(1))
        has_end_single_digit = bool(
            re.search(r"\bend(?:s)?\s+with\s+(?:any\s+)?(?:single|one)\s+digit\b", q, flags=re.I)
        )
        if n_letters is not None and n_digits is not None and n_letters >= 0 and n_digits >= 0:
            if has_end_single_digit:
                regex = f"^[A-Z]{{{n_letters}}}[0-9]{{{n_digits}}}[0-9]$"
            else:
                regex = f"^[A-Z]{{{n_letters}}}[0-9]{{{n_digits}}}$"
            target = _pick_text_target_column(anchor_pos=m_upper_start.start())
            if target:
                add_filter(IRFilter(table=target[0], column=target[1], op="regexp", value=regex))
                strict_regex_applied = True

    if not strict_regex_applied:
        m_letter_then_digits = re.search(
            r"\b(?:exactly\s+)?"
            r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
            r"(?:(uppercase|lowercase)\s+)?letters?\s+followed\s+by\s+(?:exactly\s+)?"
            r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+digits?\b",
            q,
            flags=re.I,
        )
        if m_letter_then_digits:
            n_letters = _to_int_count_token(m_letter_then_digits.group(1))
            letter_case = str(m_letter_then_digits.group(2) or "").strip().lower()
            n_digits = _to_int_count_token(m_letter_then_digits.group(3))
            if n_letters is not None and n_digits is not None and n_letters >= 0 and n_digits >= 0:
                target = _pick_text_target_column(anchor_pos=m_letter_then_digits.start())
                if target:
                    if letter_case == "uppercase":
                        letter_class = "[A-Z]"
                    elif letter_case == "lowercase":
                        letter_class = "[a-z]"
                    elif _is_identifier_like_column(target[0], target[1]):
                        # Identifier/code-like columns generally use canonical
                        # uppercase alpha prefixes unless the question asks
                        # for lowercase explicitly.
                        letter_class = "[A-Z]"
                    else:
                        letter_class = "[A-Za-z]"
                    regex = f"^{letter_class}{{{n_letters}}}[0-9]{{{n_digits}}}$"
                    add_filter(IRFilter(table=target[0], column=target[1], op="regexp", value=regex))
                    strict_regex_applied = True

    if not strict_regex_applied:
        m_literal_prefix_digits = re.search(
            r"""\bmatch(?:es|ing)?\s+(?:the\s+letter\s+)?["'“”]?([A-Za-z0-9]+)["'“”]?\s+followed\s+by\s+exactly\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+digits?\b""",
            q,
            flags=re.I,
        )
        if m_literal_prefix_digits:
            token = str(m_literal_prefix_digits.group(1) or "").strip()
            n_digits = _to_int_count_token(m_literal_prefix_digits.group(2))
            target = _pick_text_target_column(anchor_pos=m_literal_prefix_digits.start())
            if target and token and n_digits is not None and n_digits >= 0:
                regex = f"^{re.escape(token)}[0-9]{{{n_digits}}}$"
                add_filter(IRFilter(table=target[0], column=target[1], op="regexp", value=regex))
                strict_regex_applied = True

    if not strict_regex_applied:
        m_prefix = re.search(
            r"""\b(?:start(?:s)?\s+with|begins?\s+with)\b(?:\s+the\s+letter)?\s*["'“”]?([+A-Za-z0-9][A-Za-z0-9+:/.-]*)""",
            q_raw,
            flags=re.I,
        )
        if m_prefix:
            token = str(m_prefix.group(1) or "").strip(" \t\r\n'\".,;:!?")
            token_low = str(token or "").strip().lower()
            # Guard against quantifier capture on structural regex phrases like
            # "starts with exactly two uppercase letters ...", where the
            # fallback prefix parser would otherwise treat "exactly" / "two"
            # as a literal prefix.
            if token_low in {
                "exactly",
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
                "ten",
                "eleven",
                "twelve",
            }:
                token = ""
            target = _pick_explicit_text_target_column(m_prefix.start())
            name_prefix_intent = bool(
                re.search(r"\bnames?\b(?:[\s,;:()'\"-]+\w+){0,4}[\s,;:()'\"-]+(?:start(?:s)?|begin(?:s)?)\b", q)
                or re.search(r"\b(?:start(?:s)?|begin(?:s)?)\b(?:[\s,;:()'\"-]+\w+){0,4}[\s,;:()'\"-]+\bnames?\b", q)
            )
            if not target and name_prefix_intent:
                name_col = _best_name_column(subject, allowed)
                if name_col:
                    target = (subject, name_col)
            if not target:
                target = _pick_text_target_column(anchor_pos=m_prefix.start())
            if target and token:
                exact_digits = re.search(
                    r"\bexactly\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+digits?\b",
                    q,
                )
                if exact_digits:
                    n = _to_int_count_token(exact_digits.group(1))
                    if n is not None and n >= 0:
                        regex = f"^{re.escape(token)}[0-9]{{{n}}}$"
                        add_filter(IRFilter(table=target[0], column=target[1], op="regexp", value=regex))
                else:
                    add_filter(IRFilter(table=target[0], column=target[1], op="like_prefix", value=token))

    m_exact_word_count = re.search(
        r"\b(?:has|have|with|having|is|are|contains?|contain|consists?\s+of)\s+exactly\s+"
        r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+words?\b",
        q,
        flags=re.I,
    )
    if m_exact_word_count:
        n_words = _to_int_count_token(m_exact_word_count.group(1))
        if n_words is not None and n_words >= 1:
            target = _pick_text_target_column(anchor_pos=m_exact_word_count.start())
            if target:
                add_filter(IRFilter(table=target[0], column=target[1], op="word_count_eq", value=n_words))

    length_unit_pat = r"(?:characters?|chars?|letters?)"
    length_comparison_patterns = [
        (
            rf"\b(?:longer|greater|more)\s+than\s+"
            rf"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+{length_unit_pat}\b",
            "len_gt",
        ),
        (
            rf"\b(?:shorter|less)\s+than\s+"
            rf"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+{length_unit_pat}\b",
            "len_lt",
        ),
        (
            rf"\bat\s+least\s+"
            rf"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+{length_unit_pat}\b",
            "len_gte",
        ),
        (
            rf"\bat\s+most\s+"
            rf"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+{length_unit_pat}\b",
            "len_lte",
        ),
        (
            rf"\bexactly\s+"
            rf"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+{length_unit_pat}\b",
            "len_eq",
        ),
    ]
    for pattern, op in length_comparison_patterns:
        m_length_cmp = re.search(pattern, q, flags=re.I)
        if not m_length_cmp:
            continue
        matched_length_phrase = str(m_length_cmp.group(0) or "")
        if "letter" in matched_length_phrase.lower():
            structural_prefix = q[max(0, m_length_cmp.start() - 24):m_length_cmp.end()]
            structural_suffix = q[m_length_cmp.end():m_length_cmp.end() + 40]
            if re.search(r"\b(?:uppercase|lowercase)\b", structural_prefix, flags=re.I) or re.search(
                r"\bfollowed\s+by\b",
                structural_suffix,
                flags=re.I,
            ):
                continue
        n_chars = _to_int_count_token(m_length_cmp.group(1))
        if n_chars is None or n_chars < 0:
            continue
        target = _pick_text_target_column(anchor_pos=m_length_cmp.start())
        if target:
            add_filter(IRFilter(table=target[0], column=target[1], op=op, value=n_chars))
        break

    # IN list from ids/codes phrasing.
    # Pattern: "<entity> with codes 1,2,3" (e.g., offices with codes 1,2,3).
    in_applied = False
    m_entity_code_list = re.search(
        r"\b([a-z][a-z_]{2,})\s+(?:with\s+)?(?:id|ids|code|codes|number|numbers)\b(?P<tail>[^.]{0,80})",
        q,
    )
    if m_entity_code_list and not range_match:
        ent_raw = (m_entity_code_list.group(1) or "").lower()
        tail = m_entity_code_list.group("tail") or ""
        enumerated_entity_id_list = re.search(
            r"\b\d+\b(?:\s*,\s*\d+\b)+(?:\s*,?\s*(?:or|and)\s+\d+\b)?",
            tail,
        )
        conjunctive_entity_id_pair = re.search(
            r"\b\d+\b\s*(?:or|and)\s*\d+\b",
            tail,
        )
        comma_list_text = str(enumerated_entity_id_list.group(0) or "").strip() if enumerated_entity_id_list else ""
        comma_list_is_thousands_literal = bool(
            comma_list_text
            and re.fullmatch(r"\d{1,3}(?:\s*,\s*\d{3})+", comma_list_text)
        )
        has_entity_id_list = bool(
            (
                enumerated_entity_id_list is not None
                and not comma_list_is_thousands_literal
            )
            or conjunctive_entity_id_pair is not None
        )
        nums = re.findall(r"\b\d+\b", tail)
        if has_entity_id_list and len(nums) >= 2:
            ent = _safe_get_singular_form(ent_raw).replace("_", "").lower()
            subj_cols = _ordered_allowed_columns(subject, allowed)
            target_col = None
            id_like = [
                c for c in subj_cols
                if c.lower().endswith("id") or c.lower().endswith("code") or c.lower().endswith("number")
            ]
            if id_like:
                ent_hits = [c for c in id_like if ent and ent in c.lower().replace("_", "")]
                if ent_hits:
                    ent_hits.sort(key=len)
                    target_col = ent_hits[0]
                else:
                    id_like.sort(key=len)
                    target_col = id_like[0]
            if target_col:
                add_filter(IRFilter(table=subject, column=target_col, op="in", value=nums))
                in_applied = True

    # Pattern: bare entity + enumerated code list
    # Example: "employees work in offices 1, 2, or 3"
    m_entity_bare_list = re.search(
        r"\b([a-z][a-z_]{2,})\s+(?P<tail>\d+(?:\s*,\s*\d+)*(?:\s*,?\s*(?:or|and)\s+\d+)+)\b",
        q,
    )
    if m_entity_bare_list and not in_applied and not range_match:
        ent_raw = (m_entity_bare_list.group(1) or "").lower()
        tail = m_entity_bare_list.group("tail") or ""
        nums = re.findall(r"\b\d+\b", tail)
        if len(nums) >= 2:
            ent = _safe_get_singular_form(ent_raw).replace("_", "").lower()
            subj_cols = _ordered_allowed_columns(subject, allowed)
            id_like = [
                c for c in subj_cols
                if c.lower().endswith("id") or c.lower().endswith("code") or c.lower().endswith("number")
            ]
            ent_hits = [c for c in id_like if ent and ent in c.lower().replace("_", "")]
            if ent_hits:
                ent_hits.sort(key=len)
                add_filter(IRFilter(table=subject, column=ent_hits[0], op="in", value=nums))
                in_applied = True

    id_context = re.search(r"\b(?:id|ids|code|codes|number|numbers)\b(?P<tail>[^.]{0,80})", q)
    if id_context and not in_applied and not range_match:
        tail = id_context.group("tail") or ""
        if re.search(r"\bbetween\b", q[max(0, id_context.start() - 24):id_context.end() + 48], flags=re.I):
            tail = ""
        comma_list_match = re.search(r"\b\d+(?:\s*,\s*\d+)+\b", tail)
        comma_list_text = str(comma_list_match.group(0) or "").strip() if comma_list_match else ""
        comma_list_is_thousands_literal = bool(
            comma_list_text
            and re.fullmatch(r"\d{1,3}(?:\s*,\s*\d{3})+", comma_list_text)
        )
        enumerated_id_list = re.search(
            r"\b\d+\b(?:\s*,\s*\d+\b)+(?:\s*,?\s*(?:or|and)\s+\d+\b)?",
            tail,
        )
        conjunctive_id_pair = re.search(
            r"\b\d+\b\s*(?:or|and)\s*\d+\b",
            tail,
        )
        has_list_signal = bool(
            (
                enumerated_id_list is not None
                and not comma_list_is_thousands_literal
            )
            or conjunctive_id_pair is not None
        )
        nums_raw = re.findall(rf"\b{_NUMBER_TOKEN_PATTERN}\b", tail)
        nums = [n.replace(",", "") for n in nums_raw if n and not re.match(r"^\d{1,3}(,\d{3})+$", n)]
        if has_list_signal and len(nums) >= 2:
            target = None
            id_anchor = id_context.start()
            id_like_mentions = []
            for t, c, pos in column_mentions:
                cl = c.lower()
                if cl.endswith("id") or cl.endswith("code") or cl.endswith("number"):
                    id_like_mentions.append((t, c, pos))
            if id_like_mentions:
                t, c, _ = min(id_like_mentions, key=lambda x: abs(x[2] - id_anchor))
                target = (t, c)
            if not target:
                id_col = _best_id_column(subject, allowed, question)
                if id_col:
                    target = (subject, id_col)
            if target:
                add_filter(IRFilter(table=target[0], column=target[1], op="in", value=nums))

    m_entity_single_id = re.search(r"\b([a-z][a-z_]{2,})\s+(\d+)\b", q)
    if m_entity_single_id and not in_applied:
        ent_raw = str(m_entity_single_id.group(1) or "").strip().lower()
        num = str(m_entity_single_id.group(2) or "").strip()
        prefix_local = q[max(0, m_entity_single_id.start() - 24):m_entity_single_id.start()]
        if num and not re.search(r"\b(?:between|from|through|thru|to)\s*$", prefix_local, flags=re.I):
            ent = _safe_get_singular_form(ent_raw).replace("_", "").lower()
            subj_cols = _ordered_allowed_columns(subject, allowed)
            id_like = [
                c for c in subj_cols
                if c.lower().endswith("id") or c.lower().endswith("code") or c.lower().endswith("number")
            ]
            ent_hits = [c for c in id_like if ent and ent in c.lower().replace("_", "")]
            if ent_hits:
                ent_hits.sort(key=len)
                add_filter(IRFilter(table=subject, column=ent_hits[0], op="eq", value=num))

    filter_logic = _derive_filter_logic(question, filters)
    return _prune_redundant_filters(filters, filter_logic)


def _detect_existence(
    question: str,
    subject: str,
    table_mentions: List[Tuple[str, int]],
    metrics: Optional[List[IRMetric]] = None,
) -> List[IRExistence]:
    q = (question or "").lower()
    ex: List[IRExistence] = []
    seen = set()

    for table, _ in table_mentions:
        if table == subject:
            continue
        pat = _table_entity_pattern(table)
        if not pat:
            continue

        # Guard against FK / attribute mentions such as "customer number" or
        # "office code", which refer to a subject-table column rather than an
        # existence request over related rows.
        entity_attr_ref = bool(
            re.search(
                rf"\b(?:{pat})\b(?:[\s,;:()'\"-]+\w+){{0,2}}[\s,;:()'\"-]+"
                r"(?:id|ids|code|codes|number|numbers|name|names|date|dates|status)\b",
                q,
                flags=re.I,
            )
        )
        if entity_attr_ref:
            continue

        has_none = bool(
            re.search(rf"\b(?:no|without(?:\s+any)?|never|not)\b(?:[\s,;:()'\"-]+\w+){{0,8}}[\s,;:()'\"-]+(?:{pat})\b", q)
        )
        has_some = bool(
            re.search(rf"\b(?:with|have|has|had|at\s+least\s+one|one\s+or\s+more|any)\b(?:[\s,;:()'\"-]+\w+){{0,8}}[\s,;:()'\"-]+(?:{pat})\b", q)
        )

        if has_none and (table, "none") not in seen:
            ex.append(IRExistence(table=table, mode="none"))
            seen.add((table, "none"))
        elif has_some and (table, "at_least_one") not in seen:
            ex.append(IRExistence(table=table, mode="at_least_one"))
            seen.add((table, "at_least_one"))

    # Fallback for implicit existence phrasing where the related table is not
    # explicitly named in the question (e.g., "products that have been ordered
    # at least once").
    has_implicit_positive_existence = bool(
        re.search(r"\bat\s+least\s+once\b", q, flags=re.I)
        or re.search(r"\b(?:have|has|had)\s+been\s+[a-z]+(?:ed|en)\b", q, flags=re.I)
    )
    if has_implicit_positive_existence:
        metric_tables = []
        for m in (metrics or []):
            t = str(getattr(m, "table", "") or "").strip().lower()
            if t and t != subject:
                metric_tables.append(t)
        for t in dict.fromkeys(metric_tables):
            if (t, "at_least_one") not in seen:
                ex.append(IRExistence(table=t, mode="at_least_one"))
                seen.add((t, "at_least_one"))

    return ex


def _detect_ordering_and_limit(
    question: str,
    qtype: str,
    subject: str,
    dimensions: List[IRDimension],
    metrics: List[IRMetric],
    column_mentions: List[Tuple[str, str, int]],
) -> Tuple[List[IROrder], Optional[int]]:
    q = (question or "").lower()
    qtype_l = str(qtype or "").strip().lower()
    subject_l = str(subject or "").strip().lower()
    lookup_order_enrichment = qtype_l == "lookup"
    value_rank_desc_pat = (
        r"highest|largest|latest|newest|"
        r"most\s+expensive|more\s+expensive|costliest|priciest|"
        r"highest(?:\s+|-)?priced"
    )
    value_rank_asc_pat = (
        r"lowest|smallest|earliest|oldest|"
        r"least\s+expensive|less\s+expensive|cheapest|"
        r"lowest(?:\s+|-)?priced"
    )
    has_value_rank_lookup = bool(
        lookup_order_enrichment
        and re.search(rf"\b(?:{value_rank_desc_pat}|{value_rank_asc_pat})\b", q, flags=re.I)
    )
    ordering: List[IROrder] = []
    limit: Optional[int] = None

    def _looks_like_non_row_count_context(match_obj: re.Match[str]) -> bool:
        tail = q[match_obj.end():match_obj.end() + 48]
        return bool(
            re.match(
                r"\s+(?:character|characters|char|chars|letter|letters|digit|digits|word|words|byte|bytes)\b",
                tail,
                flags=re.I,
            )
        )

    def add_order(item: IROrder) -> None:
        for ex in ordering:
            if (
                ex.kind == item.kind
                and ex.by.lower() == item.by.lower()
                and ex.table.lower() == item.table.lower()
                and ex.direction.upper() == item.direction.upper()
            ):
                return
        ordering.append(item)

    def _segment_mentions_column(seg: str, table: str, column: str) -> bool:
        seg_text = re.sub(r"\s+", " ", str(seg or "").strip().lower())
        t = str(table or "").strip().lower()
        c = str(column or "").strip()
        if not seg_text or not t or not c:
            return False
        for variant in sorted(_column_variants(t, c), key=len, reverse=True):
            if _phrase_pos(seg_text, str(variant or "").strip().lower()) >= 0:
                return True
        return False

    def _segment_column_match_score(seg: str, table: str, column: str) -> Tuple[int, int]:
        seg_text = re.sub(r"\s+", " ", str(seg or "").strip().lower())
        t = str(table or "").strip().lower()
        c = str(column or "").strip()
        if not seg_text or not t or not c:
            return (0, 0)
        best = (0, 0)
        for variant in _column_variants(t, c):
            variant_text = re.sub(r"\s+", " ", str(variant or "").strip().lower())
            if not variant_text or _phrase_pos(seg_text, variant_text) < 0:
                continue
            token_count = len(re.findall(r"[a-z0-9]+", variant_text))
            score = (token_count, len(variant_text))
            if score > best:
                best = score
        if best > (0, 0):
            return best
        semantic_tokens = _semantic_query_token_variants(
            _tokenize_for_match(seg_text, stop_tokens=_LEXICAL_MATCH_STOP_TOKENS)
        )
        overlap = semantic_tokens.intersection(
            _column_semantic_tokens(t, c).union(_column_descriptor_tokens(t, c))
        )
        if len(overlap) >= 2:
            return (1, len(overlap))
        return best

    def _preferred_lookup_order_dimension() -> Optional[IRDimension]:
        visible_dims = [d for d in dimensions if bool(getattr(d, "visible", True))]
        if not visible_dims:
            return None
        subject_dims = [
            d for d in visible_dims
            if str(d.table or "").strip().lower() == subject_l
        ]
        candidate_dims = subject_dims or visible_dims
        non_identifier_dims = [
            d for d in candidate_dims
            if not _is_identifier_like_column(d.table, d.column)
        ]
        return non_identifier_dims[0] if non_identifier_dims else candidate_dims[0]

    def _metric_order_terms(m: IRMetric) -> Set[str]:
        alias = (m.alias or "").lower().strip()
        terms: Set[str] = set()
        generic_metric_tokens = {
            "count", "percentage", "percent", "ratio", "rate", "share",
            "average", "avg", "mean", "sum", "total",
            "min", "minimum", "max", "maximum",
            "distinct", "value", "values", "calculated",
        }

        def _token_variants(tok: str) -> Set[str]:
            t = (tok or "").strip().lower()
            if not t:
                return set()
            out = {t}
            if t.endswith("ies") and len(t) > 3:
                out.add(t[:-3] + "y")
            elif t.endswith("es") and len(t) > 2 and not t.endswith(("ses", "xes", "zes")):
                out.add(t[:-2])
            elif t.endswith("s") and len(t) > 1 and not t.endswith(("ss", "us")):
                out.add(t[:-1])
            return {x for x in out if x}

        def _column_bases(col: str) -> List[str]:
            words = _camel_to_words(col or "")
            if not words:
                return []
            toks = [t for t in words.split() if t]
            bases: List[str] = []
            full = " ".join(toks)
            if full:
                bases.append(full)
            if len(toks) >= 2 and toks[-1] in {"date", "time", "timestamp", "number", "code", "id"}:
                trimmed = " ".join(toks[:-1]).strip()
                if trimmed:
                    bases.append(trimmed)
            if toks:
                bases.append(toks[0])
            # preserve order, drop duplicates
            seen_bases: Set[str] = set()
            ordered: List[str] = []
            for b in bases:
                if b and b not in seen_bases:
                    seen_bases.add(b)
                    ordered.append(b)
            return ordered

        alias_bases: List[str] = []
        if alias:
            alias_words = alias.replace("_", " ").strip()
            if alias_words:
                terms.add(alias_words)
                alias_tokens = [t for t in alias_words.split() if t]
                for tok in alias_tokens:
                    if tok in generic_metric_tokens:
                        terms.add(tok)
                base_tokens = [t for t in alias_tokens if t not in generic_metric_tokens]
                if base_tokens:
                    alias_bases.append(" ".join(base_tokens))
                    alias_bases.extend(base_tokens)

        column_bases = _column_bases(str(getattr(m, "column", "") or ""))
        bases: List[str] = []
        for b in alias_bases + column_bases:
            for v in _token_variants(b) if " " not in b else {b}:
                if v and v not in bases:
                    bases.append(v)

        agg = str(getattr(m, "agg", "") or "").lower()
        if agg == "pct_non_null":
            terms.update(
                {
                    "percentage", "percent", "ratio", "rate", "share",
                    "calculated percentage", "calculated percent",
                    "calculated ratio", "calculated rate", "calculated share",
                }
            )
            for base in bases:
                terms.update(
                    {
                        f"{base} percentage",
                        f"{base} percent",
                        f"{base} ratio",
                        f"{base} rate",
                        f"{base} share",
                    }
                )
        elif agg == "count":
            terms.update({"count", "total count"})
            if bool(getattr(m, "distinct", False)):
                terms.update({"distinct count", "unique count", "different count"})
            for base in bases:
                terms.update(
                    {
                        f"{base} count",
                        f"count of {base}",
                        f"number of {base}",
                    }
                )
                if bool(getattr(m, "distinct", False)):
                    terms.update(
                        {
                            f"distinct {base} count",
                            f"unique {base} count",
                            f"different {base} count",
                        }
                    )
        elif agg == "avg":
            terms.update({"average", "avg", "mean", "calculated average"})
            for base in bases:
                terms.update(
                    {
                        f"{base} average",
                        f"average {base}",
                        f"{base} avg",
                        f"avg {base}",
                        f"mean {base}",
                    }
                )
        elif agg == "sum":
            terms.update({"sum", "total", "total value"})
            for base in bases:
                terms.update(
                    {
                        f"{base} total",
                        f"total {base}",
                        f"sum of {base}",
                    }
                )
        elif agg == "min":
            terms.update({"min", "minimum"})
            for base in bases:
                terms.update({f"minimum {base}", f"{base} minimum"})
        elif agg == "max":
            terms.update({"max", "maximum"})
            for base in bases:
                terms.update({f"maximum {base}", f"{base} maximum"})

        return {t for t in terms if t}

    m_lim = re.search(r"\b(?:top|bottom|first|last)\s+(\d+)\b", q)
    if m_lim and not _looks_like_non_row_count_context(m_lim):
        limit = int(m_lim.group(1))
    elif re.search(r"\b(?:highest|lowest|largest|smallest|newest|oldest|latest|earliest)\b", q) or has_value_rank_lookup:
        # Handles forms like "show the 10 products with the highest ..."
        m_ranked_n = re.search(
            r"\b(?:show|list|display|find|return|get)\s+(?:the\s+)?(\d{1,3})\b",
            q,
        )
        if not m_ranked_n:
            m_ranked_n = re.search(
                rf"\bthe\s+(\d{{1,3}})\s+\w+(?:\s+\w+){{0,4}}\s+"
                rf"(?:with\s+(?:the\s+)?(?:highest|lowest|largest|smallest|newest|oldest|latest|earliest)"
                rf"|(?:{value_rank_desc_pat}|{value_rank_asc_pat}))\b",
                q,
            )
        if not m_ranked_n and has_value_rank_lookup:
            m_ranked_n = re.search(
                rf"\b(?:show|list|display|find|return|get)\s+(?:the\s+)?(\d{{1,3}})\s+"
                rf"(?:[a-z][a-z0-9_'-]*\s+){{0,4}}(?:{value_rank_desc_pat}|{value_rank_asc_pat})\b",
                q,
                flags=re.I,
            )
        if m_ranked_n:
            n = int(m_ranked_n.group(1))
            if 0 < n <= 500:
                limit = n

    # Direction clues.
    if re.search(
        r"\b(highest|largest|latest|newest)\s+(?:to|-)\s+(lowest|smallest|earliest|oldest)\b",
        q,
    ):
        direction = "DESC"
    elif re.search(
        r"\b(lowest|smallest|earliest|oldest)\s+(?:to|-)\s+(highest|largest|latest|newest)\b",
        q,
    ):
        direction = "ASC"
    else:
        desc = bool(re.search(rf"\b(desc|descending|highest|largest|latest|newest|top|{value_rank_desc_pat})\b", q))
        asc = bool(re.search(rf"\b(asc|ascending|lowest|smallest|earliest|oldest|bottom|{value_rank_asc_pat})\b", q))
        if desc and not asc:
            direction = "DESC"
        elif asc and not desc:
            direction = "ASC"
        elif desc and asc:
            # Tie-break defaults to descending for value-ranking language.
            direction = "DESC"
        else:
            direction = "ASC"

    def _segment_direction(seg: str) -> str:
        if lookup_order_enrichment and re.search(r"\b(reverse\s+alphabetical|reversed\s+alphabetical|z\s+to\s+a)\b", seg, flags=re.I):
            return "DESC"
        if lookup_order_enrichment and re.search(r"\b(a\s+to\s+z|a-z)\b", seg, flags=re.I):
            return "ASC"
        if re.search(
            rf"\b(highest|largest|latest|newest|desc|descending|{value_rank_desc_pat})\b",
            seg,
            flags=re.I,
        ):
            return "DESC"
        if re.search(
            rf"\b(lowest|smallest|earliest|oldest|asc|ascending|alphabetical(?:ly)?|alphabetic(?:ally)?|{value_rank_asc_pat})\b",
            seg,
            flags=re.I,
        ):
            return "ASC"
        return direction

    # Explicit ordering phrase.
    # Also support more natural directional wording such as:
    #   "ordered from highest percentage to lowest"
    #   "sorted from latest to earliest"
    m_order = re.search(
        r"\b(?:order(?:ed)?\s+by|sort(?:ed)?\s+by|order(?:ed)?\s+from|sort(?:ed)?\s+from)\s+(.+?)(?:[.;]|$)",
        q,
    )
    if m_order:
        phrase = m_order.group(1).strip()
        segments = [
            seg.strip(" ,")
            for seg in re.split(r"\s*,\s*|\s+\bthen\b\s+|\s+\band\b\s+", phrase, flags=re.I)
            if seg.strip(" ,")
        ]
        if not segments:
            segments = [phrase]

        for seg in segments:
            seg_dir = _segment_direction(seg)
            matched = False

            # Metric aliases first.
            for m in metrics:
                order_terms = _metric_order_terms(m)
                if any(term and re.search(rf"\b{re.escape(term)}\b", seg, flags=re.I) for term in order_terms):
                    add_order(IROrder(kind="alias", by=m.alias, direction=seg_dir))
                    matched = True
                    break
            if matched:
                continue

            # Dimension aliases.
            for d in dimensions:
                alias = (d.alias or "").lower()
                alias_words = alias.replace("_", " ").strip()
                if alias_words and _phrase_pos(seg, alias_words) >= 0:
                    add_order(IROrder(kind="alias", by=d.alias, direction=seg_dir))
                    matched = True
                    break
            if not matched and lookup_order_enrichment:
                best_dim: Optional[Tuple[Tuple[int, int], IRDimension]] = None
                for d in dimensions:
                    score = _segment_column_match_score(seg, d.table, d.column)
                    if score <= (0, 0):
                        continue
                    if best_dim is None or score > best_dim[0]:
                        best_dim = (score, d)
                if best_dim is not None:
                    add_order(
                        IROrder(
                            kind="column",
                            by=best_dim[1].column,
                            table=best_dim[1].table,
                            direction=seg_dir,
                        )
                    )
                    matched = True
            if matched:
                continue

            # Column mentions fallback.
            if lookup_order_enrichment:
                best_col: Optional[Tuple[Tuple[int, int], str, str]] = None
                for t, c, _ in column_mentions:
                    score = _segment_column_match_score(seg, t, c)
                    if score <= (0, 0):
                        continue
                    if best_col is None or score > best_col[0]:
                        best_col = (score, t, c)
                if best_col is not None:
                    add_order(
                        IROrder(
                            kind="column",
                            by=best_col[2],
                            table=best_col[1],
                            direction=seg_dir,
                        )
                    )
                    matched = True
            if matched:
                continue

            if (
                lookup_order_enrichment
                and not metrics
                and re.search(
                    r"\b(?:alphabetical(?:ly)?|alphabetic(?:ally)?|a\s+to\s+z|z\s+to\s+a)\b",
                    seg,
                    flags=re.I,
                )
            ):
                preferred_dim = _preferred_lookup_order_dimension()
                if preferred_dim is not None:
                    add_order(
                        IROrder(
                            kind="column",
                            by=preferred_dim.column,
                            table=preferred_dim.table,
                            direction=seg_dir,
                        )
                    )

        if ordering:
            return ordering, limit

    # "sorted alphabetically" / "in alphabetical order" on simple lookup:
    # default to the first visible dimension.
    if (
        not ordering
        and not metrics
        and re.search(r"\b(?:alphabetical(?:ly)?|alphabetic(?:ally)?)\b", q)
    ):
        visible_dims = [d for d in dimensions if bool(getattr(d, "visible", True))]
        if len(visible_dims) == 1:
            d0 = visible_dims[0]
            # Check for reverse alphabetical
            direction = "DESC" if re.search(r"\b(?:reverse\s+alphabetical|reversed\s+alphabetical|z\s+to\s+a)\b", q, flags=re.I) else "ASC"
            ordering.append(IROrder(kind="column", by=d0.column, table=d0.table, direction=direction))
            return ordering, limit

    # Month special ordering.
    if any(d.alias == "__month_num" for d in dimensions):
        d = "DESC" if re.search(r"\b(latest|newest|descending)\b", q) else "ASC"
        ordering.append(IROrder(kind="expression", by="MONTH", direction=d))
        return ordering, limit

    # Temporal lookup fallback:
    # "starting from the latest order date", "latest payment date first", etc.
    temporal_desc = bool(re.search(r"\b(?:latest|newest|most\s+recent)\b", q, flags=re.I))
    temporal_asc = bool(re.search(r"\b(?:earliest|oldest)\b", q, flags=re.I))
    if not ordering and not metrics and (temporal_desc or temporal_asc):
        date_mentions = [(t, c, p) for (t, c, p) in column_mentions if _is_date_column(c, t)]
        if date_mentions:
            temporal_anchor = min(
                (
                    m.start()
                    for m in re.finditer(
                        r"\b(?:latest|newest|most\s+recent|earliest|oldest)\b",
                        q,
                        flags=re.I,
                    )
                ),
                default=-1,
            )
            if temporal_anchor >= 0:
                t, c, _ = min(date_mentions, key=lambda x: abs(x[2] - temporal_anchor))
            else:
                t, c, _ = date_mentions[0]
            add_order(
                IROrder(
                    kind="column",
                    by=c,
                    table=t,
                    direction="DESC" if temporal_desc and not temporal_asc else "ASC",
                )
            )

    # Ranking-style fallback: if question implies extremum and no explicit ORDER BY phrase,
    # sort by first numeric column mention.
    if (
        not ordering
        and not metrics
        and (
            re.search(r"\b(highest|lowest|largest|smallest|newest|oldest|latest|earliest|top|bottom)\b", q)
            or has_value_rank_lookup
        )
    ):
        numeric_mentions = [(t, c, p) for (t, c, p) in column_mentions if _is_numeric_column(c, t)]
        subject_numeric_mentions = [
            item for item in numeric_mentions
            if subject_l and str(item[0] or "").strip().lower() == subject_l
        ]
        numeric_dims = [
            d for d in dimensions
            if _is_numeric_column(d.column, d.table)
        ]
        subject_numeric_dims = [
            d for d in numeric_dims
            if subject_l and str(d.table or "").strip().lower() == subject_l
        ]

        if subject_numeric_mentions:
            t, c, _ = subject_numeric_mentions[0]
            add_order(IROrder(kind="column", by=c, table=t, direction=direction))
        elif subject_numeric_dims:
            d0 = subject_numeric_dims[0]
            add_order(IROrder(kind="column", by=d0.column, table=d0.table, direction=direction))
        elif numeric_mentions:
            t, c, _ = numeric_mentions[0]
            add_order(IROrder(kind="column", by=c, table=t, direction=direction))
        elif numeric_dims:
            d0 = numeric_dims[0]
            add_order(IROrder(kind="column", by=d0.column, table=d0.table, direction=direction))

    # Top-N fallback by first metric.
    if limit and metrics:
        add_order(IROrder(kind="alias", by=metrics[0].alias, direction="DESC" if "bottom" not in q else "ASC"))
    elif limit and not ordering:
        numeric_mentions = [(t, c, p) for (t, c, p) in column_mentions if _is_numeric_column(c, t)]
        subject_numeric_mentions = [
            item for item in numeric_mentions
            if subject_l and str(item[0] or "").strip().lower() == subject_l
        ]
        numeric_dims = [
            d for d in dimensions
            if _is_numeric_column(d.column, d.table)
        ]
        subject_numeric_dims = [
            d for d in numeric_dims
            if subject_l and str(d.table or "").strip().lower() == subject_l
        ]
        if subject_numeric_mentions:
            t, c, _ = subject_numeric_mentions[0]
            add_order(IROrder(kind="column", by=c, table=t, direction=direction))
        elif subject_numeric_dims:
            d0 = subject_numeric_dims[0]
            add_order(IROrder(kind="column", by=d0.column, table=d0.table, direction=direction))
        elif numeric_mentions:
            t, c, _ = numeric_mentions[0]
            add_order(IROrder(kind="column", by=c, table=t, direction=direction))
        elif numeric_dims:
            d0 = numeric_dims[0]
            add_order(IROrder(kind="column", by=d0.column, table=d0.table, direction=direction))

    return ordering, limit


def _derive_primary_order_fields(ordering: List[IROrder]) -> Tuple[str, str]:
    if not ordering:
        return "", ""

    first = ordering[0]
    if not isinstance(first, IROrder):
        return "", ""

    by = str(first.by or "").strip()
    if not by:
        return "", ""

    sign = str(first.direction or "ASC").strip().upper() or "ASC"
    if sign not in {"ASC", "DESC"}:
        sign = "ASC"

    if str(first.kind or "").strip().lower() == "column" and str(first.table or "").strip():
        return f"{str(first.table).strip()}.{by}", sign
    return by, sign


def _detect_include_zero(question: str, subject: str, metrics: List[IRMetric], existence: List[IRExistence]) -> bool:
    q = (question or "").lower()
    if not metrics:
        return False

    if any(e.mode == "at_least_one" for e in existence):
        return False

    subj_pat = _table_entity_pattern(subject) if subject else ""
    if subj_pat and re.search(rf"\b(?:all|every|each)\s+(?:{subj_pat})\b", q):
        return True

    if re.search(r"\bincluding\b.*\bwith\s+no\b", q):
        return True

    return False


def _compute_confidence(
    qtype: str,
    subject: str,
    dimensions: List[IRDimension],
    metrics: List[IRMetric],
    filters: List[IRFilter],
    existence: List[IRExistence],
    mentioned_tables: List[str],
) -> Tuple[float, bool, List[str]]:
    notes: List[str] = []
    score = 0.35

    if subject:
        score += 0.2
    else:
        notes.append("subject_missing")

    if dimensions or metrics:
        score += 0.2
    else:
        notes.append("projection_missing")

    if (
        (qtype or "").lower() == "lookup"
        and subject
        and dimensions
        and not metrics
        and not existence
        and len(mentioned_tables) <= 1
    ):
        score += 0.1

    if filters:
        score += 0.1
    if existence:
        score += 0.1
    if any(str(m.agg or "").lower() == "pct_non_null" for m in metrics):
        score += 0.1

    # Structured aggregate shape bonus:
    # grouped aggregate with explicit dimension+metric is usually planner-safe
    # even without literal filters (e.g., count per territory).
    metric_aggs = {str(m.agg or "").lower() for m in metrics}
    if (
        (qtype or "").lower() == "aggregate"
        and dimensions
        and metrics
        and metric_aggs.issubset({"count", "sum", "avg", "min", "max", "pct_non_null"})
    ):
        score += 0.1

    # Reward simple grouped earliest/latest style aggregates:
    # e.g., "for each status, count orders and earliest order date".
    if (
        (qtype or "").lower() == "aggregate"
        and dimensions
        and metrics
        and bool(metric_aggs.intersection({"min", "max"}))
    ):
        score += 0.1

    planner_compatible = True
    if (qtype or "").lower() == "rank":
        planner_compatible = False
        score -= 0.25
        notes.append("rank_query_fallback")

    if len(mentioned_tables) >= 4 and not metrics and not existence:
        score -= 0.1
        notes.append("high_table_ambiguity")

    # Check disconnected mentioned tables (schema graph).
    try:
        fk = get_fk_relationships() or {}
        adj: Dict[str, Set[str]] = {}
        for src, fmap in fk.items():
            s = str(src).lower()
            adj.setdefault(s, set())
            for _, tgt in (fmap or {}).items():
                t = str(tgt).lower()
                adj.setdefault(t, set())
                adj[s].add(t)
                adj[t].add(s)
        mts = [t.lower() for t in mentioned_tables]
        if len(mts) >= 2:
            seen = {mts[0]}
            queue = [mts[0]]
            while queue:
                cur = queue.pop(0)
                for nxt in adj.get(cur, set()):
                    if nxt not in seen:
                        seen.add(nxt)
                        queue.append(nxt)
            if any(t not in seen for t in mts[1:]):
                score -= 0.1
                notes.append("disconnected_mentions")
    except Exception:
        pass

    score = max(0.0, min(1.0, score))
    return score, planner_compatible, notes


def _detect_row_level_computed_projections(
    question: str,
    subject: str,
    allowed: Dict[str, Set[str]],
    column_mentions: List[Tuple[str, str, int]],
) -> List[IRComputedProjection]:
    q_raw = str(question or "").strip()
    q = q_raw.lower()
    subject_l = str(subject or "").strip().lower()
    if not q or not subject_l:
        return []

    if not re.search(r"\b(?:show|return|list|display|find|get|include|give)\b", q, flags=re.I):
        return []
    if re.search(r"\b(?:group(?:ed)?\s+by|group\s+by|summari[sz]e|break\s+.*\s+down)\b", q, flags=re.I):
        return []
    if re.search(
        r"\b(?:count|sum|avg|average|mean|min(?:imum)?|max(?:imum)?|total\s+number|how\s+many|percentage|percent)\b",
        q,
        flags=re.I,
    ):
        return []

    def _is_numeric_measure_column(col: str) -> bool:
        return _is_numeric_column(col, subject_l) and not _is_date_column(col, subject_l)

    numeric_subject_cols = [
        str(col or "").strip()
        for col in _ordered_allowed_columns(subject_l, allowed)
        if str(col or "").strip() and _is_numeric_measure_column(str(col or "").strip())
    ]
    if len(numeric_subject_cols) < 2:
        return []

    def _clean_operand_phrase(text: str) -> str:
        phrase = re.sub(r"\s+", " ", str(text or "").strip().lower())
        phrase = re.sub(r"^(?:the|a|an|their|its|our|your)\s+", "", phrase, flags=re.I)
        phrase = phrase.strip(" ,;:.!?()[]{}'\"")
        return phrase

    def _resolve_operand(
        phrase: str,
        *,
        exclude: Optional[Set[str]] = None,
    ) -> Optional[str]:
        phrase_norm = _clean_operand_phrase(phrase)
        if not phrase_norm:
            return None

        best_exact: Optional[str] = None
        best_exact_len = -1
        for col in numeric_subject_cols:
            if exclude and col.lower() in exclude:
                continue
            for variant in sorted(_column_variants(subject_l, col), key=len, reverse=True):
                variant_norm = re.sub(r"\s+", " ", str(variant or "").strip().lower())
                if not variant_norm:
                    continue
                variant_pat = re.escape(variant_norm).replace(r"\ ", r"\s+")
                if re.search(rf"\b{variant_pat}\b", phrase_norm, flags=re.I) and len(variant_norm) > best_exact_len:
                    best_exact = col
                    best_exact_len = len(variant_norm)
                    break
        if best_exact:
            return best_exact

        narrowed_allowed = {subject_l: set(numeric_subject_cols)}
        matched = _best_semantic_column_match(
            text=phrase_norm,
            table=subject_l,
            allowed=narrowed_allowed,
            numeric_only=True,
            min_score=1.0,
        )
        if matched and not (exclude and matched.lower() in exclude):
            return matched
        return None

    def _alias_for_projection(op: str, left_col: str, right_col: str) -> str:
        q_low = q
        if op == "sub":
            if re.search(r"\bmarkup\b", q_low, flags=re.I):
                return "markup"
            if re.search(r"\bprofit\s+margins?\b", q_low, flags=re.I):
                return "profit_margin"
            if re.search(r"\bmargins?\b", q_low, flags=re.I):
                return "margin"
            if re.search(r"\bdifference\b", q_low, flags=re.I):
                return "difference"
        op_word = {
            "sub": "minus",
            "add": "plus",
            "mul": "times",
            "div": "divided_by",
        }.get(op, "computed")
        return _safe_alias(f"{left_col}_{op_word}_{right_col}", "computed_value")

    boundary_terms = (
        "for\\s+each|per|return|show|display|list|include|give|sort|order|ordered|sorted|where|having"
    )
    patterns = [
        (
            "sub",
            re.compile(
                rf"\bdifference\s+between\s+"
                rf"(?P<left>[a-z][a-z0-9_ ]{{1,60}}?)\s+and\s+"
                rf"(?P<right>[a-z][a-z0-9_ ]{{1,60}}?)(?=[.,;!?]|$|\s+(?:{boundary_terms})\b)",
                flags=re.I,
            ),
        ),
        (
            "sub",
            re.compile(
                rf"\b(?P<left>[a-z][a-z0-9_ ]{{1,60}}?)\s+minus\s+"
                rf"(?P<right>[a-z][a-z0-9_ ]{{1,60}}?)(?=[.,;!?]|$|\s+(?:{boundary_terms})\b)",
                flags=re.I,
            ),
        ),
        (
            "add",
            re.compile(
                rf"\b(?P<left>[a-z][a-z0-9_ ]{{1,60}}?)\s+plus\s+"
                rf"(?P<right>[a-z][a-z0-9_ ]{{1,60}}?)(?=[.,;!?]|$|\s+(?:{boundary_terms})\b)",
                flags=re.I,
            ),
        ),
        (
            "mul",
            re.compile(
                rf"\b(?P<left>[a-z][a-z0-9_ ]{{1,60}}?)\s+"
                rf"(?:multiplied\s+by|times)\s+"
                rf"(?P<right>[a-z][a-z0-9_ ]{{1,60}}?)(?=[.,;!?]|$|\s+(?:{boundary_terms})\b)",
                flags=re.I,
            ),
        ),
        (
            "div",
            re.compile(
                rf"\b(?P<left>[a-z][a-z0-9_ ]{{1,60}}?)\s+"
                rf"(?:divided\s+by|over)\s+"
                rf"(?P<right>[a-z][a-z0-9_ ]{{1,60}}?)(?=[.,;!?]|$|\s+(?:{boundary_terms})\b)",
                flags=re.I,
            ),
        ),
    ]

    projections: List[IRComputedProjection] = []
    seen: Set[Tuple[str, str, str]] = set()
    for op, pat in patterns:
        for match in pat.finditer(q):
            left_phrase = _clean_operand_phrase(match.group("left") or "")
            right_phrase = _clean_operand_phrase(match.group("right") or "")
            if not (left_phrase and right_phrase):
                continue
            left_col = _resolve_operand(left_phrase)
            if not left_col:
                continue
            right_col = _resolve_operand(right_phrase, exclude={left_col.lower()})
            if not right_col or left_col.lower() == right_col.lower():
                continue
            key = (op, left_col.lower(), right_col.lower())
            if key in seen:
                continue
            seen.add(key)
            projections.append(
                IRComputedProjection(
                    kind="arithmetic",
                    op=op,
                    left_table=subject,
                    left_column=left_col,
                    right_table=subject,
                    right_column=right_col,
                    alias=_alias_for_projection(op, left_col, right_col),
                )
            )

    return projections


def _should_reclassify_pseudo_aggregate_lookup(
    question: str,
    qtype: str,
    dimensions: List[IRDimension],
    metrics: List[IRMetric],
    filters: List[IRFilter],
    existence: List[IRExistence],
) -> bool:
    qtype_l = str(qtype or "").strip().lower()
    if qtype_l not in {"aggregate", "rank"}:
        return False
    if metrics or existence or not filters:
        return False

    q = str(question or "").lower()
    if not q:
        return False

    has_row_return_language = bool(
        re.search(
            r"\b(?:show|return|list|display|find|get)\b",
            q,
            flags=re.I,
        )
    )
    if not has_row_return_language:
        return False

    grouping_terms_alt = _regex_alt_from_terms(_schema_groupable_terms())
    has_grouping_language = bool(
        re.search(
            r"\b(?:for\s+each|group(?:ed)?\s+by|group\s+by|summari[sz]e|break\s+.*\s+down)\b",
            q,
            flags=re.I,
        )
        or (
            grouping_terms_alt
            and re.search(rf"\bper\s+(?:{grouping_terms_alt})\b", q, flags=re.I)
        )
    )
    if has_grouping_language:
        return False

    has_aggregate_lexemes = bool(
        re.search(
            r"\b(?:count|sum|avg|average|mean|min(?:imum)?|max(?:imum)?|total(?:\s+number)?|"
            r"percentage|percent|ratio|rate|how\s+many)\b",
            q,
            flags=re.I,
        )
    )
    if has_aggregate_lexemes:
        return False

    has_rank_lexemes = bool(
        re.search(
            r"\b(?:rank|ranking|row\s+number|dense\s+rank|position|top|bottom|highest|lowest|largest|smallest|"
            r"earliest|latest|newest|oldest|most\s+recent)\b",
            q,
            flags=re.I,
        )
    )
    if qtype_l == "rank" and has_rank_lexemes:
        return False

    has_projection_shape = bool(dimensions) or bool(_extract_projection_segments(question))
    if not has_projection_shape:
        return False

    return True


def _detect_ambiguity_signals(
    question: str,
    subject: str,
    table_mentions: List[Tuple[str, int]],
    notes: List[str],
    column_mentions: List[Tuple[str, str, int]],
) -> List[str]:
    signals: List[str] = []
    q = str(question or "").strip().lower()
    subject_l = str(subject or "").strip().lower()
    table_mention_names = [
        str(t or "").strip().lower()
        for (t, _) in (table_mentions or [])
        if str(t or "").strip()
    ]
    unique_tables = list(dict.fromkeys(table_mention_names))
    table_positions: Dict[str, List[int]] = {}
    for t, pos in (table_mentions or []):
        tl = str(t or "").strip().lower()
        if not tl:
            continue
        table_positions.setdefault(tl, []).append(int(pos))

    if not subject_l and len(unique_tables) >= 2:
        signals.append("unresolved_subject_multi_table")
    if len(unique_tables) >= 4:
        signals.append("high_table_ambiguity")
    if "disconnected_mentions" in {str(n or "").strip().lower() for n in (notes or [])}:
        signals.append("disconnected_table_mentions")

    column_tables = {
        str(t or "").strip().lower()
        for (t, _, _) in (column_mentions or [])
        if str(t or "").strip()
    }
    if not subject_l and len(column_tables) >= 3:
        signals.append("column_mentions_span_many_tables")

    # Explicit column-link ambiguity:
    # the same column token maps to multiple tables without clear table qualification.
    # This is a schema-agnostic guard to avoid overconfident wrong links.
    if q and len(unique_tables) >= 2:
        col_to_mentions: Dict[str, List[Tuple[str, int]]] = {}
        for table, col, pos in (column_mentions or []):
            tl = str(table or "").strip().lower()
            cl = str(col or "").strip().lower()
            if not tl or not cl:
                continue
            col_to_mentions.setdefault(cl, []).append((tl, int(pos)))

        proximity_chars = 40
        for col_key, mentions in col_to_mentions.items():
            candidate_tables = {t for t, _ in mentions}
            if len(candidate_tables) < 2:
                continue

            has_explicit_qualifier = False
            for t, col_pos in mentions:
                near_positions = table_positions.get(t, [])
                if any(abs(col_pos - tp) <= proximity_chars for tp in near_positions):
                    has_explicit_qualifier = True
                    break

            if not has_explicit_qualifier:
                signals.append("ambiguous_column_link")
                break

    return list(dict.fromkeys(signals))


def build_intent_ir(
    question: str,
    ctx: str,
    router_decision: Optional[Dict[str, Any]] = None,
    primary_table: str = "",
) -> Dict[str, Any]:
    router_decision = router_decision or {}
    qtype = str(router_decision.get("qtype") or "").strip().lower()

    allowed = _load_allowed_or_schema(ctx)
    tables = sorted(allowed.keys())

    table_mentions = _detect_table_mentions(question, tables)
    subject = _choose_subject(
        question=question,
        allowed=allowed,
        table_mentions=table_mentions,
        primary_table=primary_table,
    )
    allowed = _augment_subject_allowed_columns_from_schema(question, subject, allowed)

    column_mentions = _detect_column_mentions(question, allowed)
    computed_projections = _detect_row_level_computed_projections(
        question=question,
        subject=subject,
        allowed=allowed,
        column_mentions=column_mentions,
    )
    if computed_projections and (not qtype or qtype == "aggregate"):
        qtype = "lookup"
    metrics = _detect_metrics(
        question=question,
        qtype=qtype,
        subject=subject,
        allowed=allowed,
        table_mentions=table_mentions,
        column_mentions=column_mentions,
    )
    if not qtype:
        qtype = "aggregate" if metrics else "lookup"
    dimensions = _detect_dimensions(
        question=question,
        qtype=qtype,
        subject=subject,
        allowed=allowed,
        column_mentions=column_mentions,
        metrics=metrics,
    )
    filters = _detect_filters(
        question=question,
        subject=subject,
        allowed=allowed,
        column_mentions=column_mentions,
        dimensions=dimensions,
        metrics=metrics,
    )
    existence = _detect_existence(
        question=question,
        subject=subject,
        table_mentions=table_mentions,
        metrics=metrics,
    )
    ordering, limit = _detect_ordering_and_limit(
        question=question,
        qtype=qtype,
        subject=subject,
        dimensions=dimensions,
        metrics=metrics,
        column_mentions=column_mentions,
    )
    if _should_reclassify_pseudo_aggregate_lookup(
        question=question,
        qtype=qtype,
        dimensions=dimensions,
        metrics=metrics,
        filters=filters,
        existence=existence,
    ):
        qtype = "lookup"
        dimensions = _detect_dimensions(
            question=question,
            qtype=qtype,
            subject=subject,
            allowed=allowed,
            column_mentions=column_mentions,
            metrics=metrics,
        )
        ordering, limit = _detect_ordering_and_limit(
            question=question,
            qtype=qtype,
            subject=subject,
            dimensions=dimensions,
            metrics=metrics,
            column_mentions=column_mentions,
        )
    order_by_column, order_by_direction = (
        _derive_primary_order_fields(ordering)
        if qtype == "lookup"
        else ("", "")
    )
    include_zero = _detect_include_zero(
        question=question,
        subject=subject,
        metrics=metrics,
        existence=existence,
    )
    filter_logic = _derive_filter_logic(question, filters)
    select_distinct = _wants_distinct_projection(question)
    if (
        qtype != "rank"
        and select_distinct
        and dimensions
        and not metrics
        and not existence
    ):
        qtype = "lookup"

    mentioned_tables = [t for t, _ in table_mentions]
    confidence, planner_compatible, notes = _compute_confidence(
        qtype=qtype,
        subject=subject,
        dimensions=dimensions,
        metrics=metrics,
        filters=filters,
        existence=existence,
        mentioned_tables=mentioned_tables,
    )
    ambiguity_signals = _detect_ambiguity_signals(
        question=question,
        subject=subject,
        table_mentions=table_mentions,
        notes=notes,
        column_mentions=column_mentions,
    )

    ir = IntentIR(
        subject_entity=subject,
        dimensions=dimensions,
        metrics=metrics,
        computed_projections=computed_projections,
        filters=filters,
        filter_logic=filter_logic,
        existence=existence,
        ordering=ordering,
        order_by_column=order_by_column,
        order_by_direction=order_by_direction,
        limit=limit,
        include_zero=include_zero,
        select_distinct=select_distinct,
        qtype=qtype,
        notes=notes,
        confidence=confidence,
        planner_compatible=planner_compatible,
        mentioned_tables=mentioned_tables,
        ambiguity_signals=ambiguity_signals,
    )
    return ir.to_dict()


def is_ir_plannable(intent_ir: Dict[str, Any], min_confidence: float = 0.65) -> bool:
    if not intent_ir:
        return False
    if not intent_ir.get("planner_compatible", True):
        return False
    if float(intent_ir.get("confidence", 0.0) or 0.0) < float(min_confidence):
        return False
    if not str(intent_ir.get("subject_entity") or "").strip():
        return False
    if not (
        intent_ir.get("dimensions")
        or intent_ir.get("metrics")
        or intent_ir.get("computed_projections")
        or intent_ir.get("filters")
        or intent_ir.get("existence")
    ):
        return False
    return True
