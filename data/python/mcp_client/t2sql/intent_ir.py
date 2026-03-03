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
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .schema_analyzer import (
    get_all_tables,
    get_column_synonyms,
    get_fk_relationships,
    get_id_column,
    get_name_column,
    get_schema_analyzer,
    get_singular_form,
    get_table_synonyms,
    get_table_type,
)


_NUMERIC_HINTS = (
    "amount",
    "total",
    "price",
    "quantity",
    "stock",
    "msrp",
    "revenue",
    "sales",
    "cost",
    "rate",
    "count",
    "number",
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
_FUNCTION_WORD_TOKENS = {
    "the", "a", "an",
    "that", "this", "those", "these",
    "their", "its", "his", "her", "our", "your",
    "of", "to", "for", "with", "without", "and", "or",
}


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
    op: str  # eq|in|gt|gte|lt|lte|between|year_eq|like_prefix|regexp|is_null_or_empty
    value: Any = None
    value2: Any = None


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
class IntentIR:
    subject_entity: str = ""
    dimensions: List[IRDimension] = field(default_factory=list)
    metrics: List[IRMetric] = field(default_factory=list)
    filters: List[IRFilter] = field(default_factory=list)
    existence: List[IRExistence] = field(default_factory=list)
    ordering: List[IROrder] = field(default_factory=list)
    limit: Optional[int] = None
    include_zero: bool = False
    select_distinct: bool = False
    qtype: str = ""
    notes: List[str] = field(default_factory=list)
    confidence: float = 0.0
    planner_compatible: bool = True
    mentioned_tables: List[str] = field(default_factory=list)

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


def _camel_to_words(s: str) -> str:
    x = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s or "")
    x = x.replace("_", " ")
    return re.sub(r"\s+", " ", x).strip().lower()


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
    if last.endswith("ies") and len(last) > 3:
        tokens[-1] = last[:-3] + "y"
    elif last.endswith("s") and not last.endswith("ss") and len(last) > 1:
        tokens[-1] = last[:-1]
    return " ".join(tokens)


def _table_variants(table: str) -> Set[str]:
    t = (table or "").strip().lower()
    if not t:
        return set()
    out = {t, t.replace("_", " ")}

    singular = _safe_get_singular_form(t) or t
    out.add(singular.lower())
    out.add(singular.lower().replace("_", " "))

    for syn in _safe_get_table_synonyms(t):
        s = str(syn).strip().lower()
        if s:
            out.add(s)
            out.add(s.replace("_", " "))
    return {x for x in out if x}


def _column_variants(table: str, column: str) -> Set[str]:
    t = (table or "").strip().lower()
    c = str(column or "").strip()
    if not c:
        return set()

    c_low = c.lower()
    out = {
        c_low,
        c_low.replace("_", " "),
        _camel_to_words(c),
    }
    for syn in _safe_get_column_synonyms(t, c):
        s = str(syn).strip().lower()
        if not s:
            continue
        out.add(s)
        out.add(s.replace("_", " "))
        out.add(s.replace("-", " "))

    expanded: Set[str] = set()
    for v in out:
        vv = re.sub(r"\s+", " ", (v or "").strip().lower())
        if not vv:
            continue
        expanded.add(vv)
        expanded.add(_pluralize_last_token(vv))
        expanded.add(_singularize_last_token(vv))

    # Domain-friendly reorder variants for common business phrasings.
    if "quantity" in c_low and "stock" in c_low:
        expanded.update({"stock quantity", "stock quantities", "quantity stock"})
    if "amount" in c_low and "payment" in c_low:
        expanded.update({"payment amount", "payment amounts", "amount paid"})
    if c_low.endswith("number"):
        stem = c_low[:-6].strip("_")
        if stem:
            expanded.add(f"{stem} numbers".replace("_", " "))
    if c_low.endswith("code"):
        stem = c_low[:-4].strip("_")
        if stem:
            expanded.add(f"{stem} codes".replace("_", " "))

    return {x for x in expanded if x}


def _phrase_pos(text: str, phrase: str) -> int:
    if not text or not phrase:
        return -1
    pat = re.escape(phrase).replace(r"\ ", r"\s+")
    m = re.search(rf"\b{pat}\b", text, flags=re.I)
    return m.start() if m else -1


def _is_numeric_column(col: str) -> bool:
    words = _camel_to_words(col or "")
    tokens = {t for t in re.findall(r"[a-z0-9]+", words) if t}
    if any(k in tokens for k in _NUMERIC_HINTS):
        return True
    if "credit limit" in words:
        return True
    return False


def _is_date_column(col: str) -> bool:
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


def _choose_subject(
    question: str,
    allowed: Dict[str, Set[str]],
    table_mentions: List[Tuple[str, int]],
    primary_table: str = "",
) -> str:
    p = (primary_table or "").strip().lower()
    if p and p in allowed:
        return p

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
    out: List[Tuple[str, int]] = []
    for t in tables:
        best = -1
        for v in _table_variants(t):
            pos = _phrase_pos(q, v)
            if pos >= 0 and (best < 0 or pos < best):
                best = pos
        if best >= 0:
            out.append((t.lower(), best))
    out.sort(key=lambda x: x[1])
    return out


def _detect_column_mentions(question: str, allowed: Dict[str, Set[str]]) -> List[Tuple[str, str, int]]:
    q = (question or "").lower()
    out: List[Tuple[str, str, int]] = []
    for table, cols in allowed.items():
        for col in cols:
            best = -1
            for v in _column_variants(table, col):
                pos = _phrase_pos(q, v)
                if pos >= 0 and (best < 0 or pos < best):
                    best = pos
            if best >= 0:
                out.append((table, col, best))
    out.sort(key=lambda x: x[2])
    return out


def _find_best_date_column(table: str, allowed: Dict[str, Set[str]]) -> Optional[str]:
    cols = list(allowed.get((table or "").lower(), set()))
    date_like = [c for c in cols if _is_date_column(c)]
    if not date_like:
        return None
    date_like.sort(key=lambda x: (0 if "date" in x.lower() else 1, len(x)))
    return date_like[0]


def _best_id_column(table: str, allowed: Dict[str, Set[str]]) -> Optional[str]:
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
        id_like.sort(key=lambda c: (score(c), -len(c)), reverse=True)
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
    name_like = [c for c in cols if c.lower().endswith("name") or "name" in c.lower()]
    if name_like:
        name_like.sort(key=len)
        return name_like[0]
    return None


def _find_best_numeric_column(table: str, allowed: Dict[str, Set[str]]) -> Optional[str]:
    cols = list(allowed.get((table or "").lower(), set()))
    num_like = [c for c in cols if _is_numeric_column(c)]
    if not num_like:
        return None
    num_like.sort(key=lambda x: len(x))
    return num_like[0]


def _prefer_numeric_column_from_question(
    question: str,
    subject: str,
    allowed: Dict[str, Set[str]],
) -> Optional[str]:
    q = (question or "").lower()
    cols = list(allowed.get((subject or "").lower(), set()))
    if not cols:
        return None

    def choose(col_terms: Tuple[str, ...]) -> Optional[str]:
        hits = [c for c in cols if any(k in c.lower() for k in col_terms)]
        if not hits:
            return None
        hits.sort(key=len)
        return hits[0]

    if "stock" in q:
        c = choose(("stock", "quantity"))
        if c:
            return c
    if any(k in q for k in ("payment amount", "amount paid", "amount")):
        c = choose(("amount", "payment"))
        if c:
            return c
    if "credit" in q:
        c = choose(("credit",))
        if c:
            return c
    if any(k in q for k in ("buy price", "msrp", "price", "cost")):
        c = choose(("price", "msrp", "cost"))
        if c:
            return c
    return None


def _table_entity_pattern(table: str) -> str:
    alts = sorted((_table_variants(table) or {table}), key=len, reverse=True)
    return "|".join(re.escape(a).replace(r"\ ", r"\s+") for a in alts if a)


def _question_mentions_entity(question: str, table: str) -> bool:
    q = (question or "").lower()
    if not q:
        return False
    pat = _table_entity_pattern(table)
    if not pat:
        return False
    return bool(re.search(rf"\b(?:{pat})\b", q, flags=re.I))


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
    return bool(re.search(r"\b(distinct|unique|different)\b", q))


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
    wants_sum = bool(
        re.search(
            r"\b(sum|total\s+(?:amount|paid|payment|payments|quantity|value|sales?|revenue|cost|price|stock)|jumlah)\b",
            q,
        )
    )
    wants_avg = bool(re.search(r"\b(avg|average|mean|rata)\b", q))
    wants_max = bool(re.search(r"\b(max(?:imum)?|highest|largest)\b", q))
    wants_min = bool(re.search(r"\b(min(?:imum)?|lowest|smallest)\b", q))
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

    # Candidate numeric/date mentions from question.
    mentioned_numeric_cols = [
        (t, c, p) for (t, c, p) in column_mentions if _is_numeric_column(c)
    ]
    mentioned_date_cols = [
        (t, c, p) for (t, c, p) in column_mentions if _is_date_column(c)
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
        p_tokens = set(re.findall(r"[a-z0-9]+", p))
        if not p_tokens:
            return None

        candidates = column_mentions or []
        if not candidates:
            for t, cols in allowed.items():
                for c in cols:
                    candidates.append((t, c, 10**9))

        best: Optional[Tuple[str, str]] = None
        best_score = 0
        best_pos = 10**9
        for t, c, pos in candidates:
            score = 0
            for v in _column_variants(t, c):
                v_tokens = set(re.findall(r"[a-z0-9]+", v))
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

    def _pick_count_distinct_target() -> Optional[Tuple[str, str]]:
        if not wants_count_distinct:
            return None

        for t, c, pos in column_mentions:
            window_start = max(0, pos - 48)
            prefix = q[window_start:pos]
            if re.search(r"\b(distinct|unique|different)\b", prefix, flags=re.I):
                return (t, c)

        m_phrase = re.search(
            r"\b(?:how\s+many|number\s+of|count|total\s+number(?:\s+of)?)\s+"
            r"(?:distinct|unique|different)\s+([a-z][a-z0-9_ ]{1,40})\b",
            q,
            flags=re.I,
        )
        if m_phrase:
            phrase = (m_phrase.group(1) or "").strip()
            resolved_col = _resolve_column_from_phrase(phrase)
            if resolved_col:
                return resolved_col
            resolved_table = _resolve_table_from_phrase(phrase)
            if resolved_table:
                id_col = _best_id_column(resolved_table, allowed) or "*"
                return (resolved_table, id_col)

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
            if t == subject and _is_date_column(c):
                return (t, c)
        for c in allowed.get(subject, set()):
            if _is_date_column(c):
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
            count_table = first_mentioned or subject
            m_count_of = re.search(
                r"\b(?:number|count|total\s+number)\s+of\s+([a-z][a-z0-9_ ]{1,40})\b",
                q,
                flags=re.I,
            )
            if m_count_of:
                resolved = _resolve_table_from_phrase(m_count_of.group(1))
                if resolved:
                    count_table = resolved
            id_col = _best_id_column(count_table, allowed) or "*"
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
    def _metric_target_for_numeric() -> Tuple[str, str]:
        if mentioned_numeric_cols:
            t, c, _ = mentioned_numeric_cols[0]
            return t, c
        for t, _ in table_mentions:
            col = _find_best_numeric_column(t, allowed)
            if col:
                return t, col
        col = _find_best_numeric_column(subject, allowed)
        if col:
            return subject, col
        return subject, "*"

    if wants_sum:
        t, c = _metric_target_for_numeric()
        alias = _safe_alias(f"total_{c}", "total_value")
        add_metric(IRMetric(agg="sum", table=t, column=c, alias=alias))

    if wants_avg:
        t, c = _metric_target_for_numeric()
        alias = _safe_alias(f"avg_{c}", "avg_value")
        add_metric(IRMetric(agg="avg", table=t, column=c, alias=alias))

    # MAX / MIN in aggregate context only.
    if (qtype or "").lower().startswith("aggregate"):
        if wants_max:
            if mentioned_date_cols:
                t, c, _ = mentioned_date_cols[0]
            else:
                t, c = _metric_target_for_numeric()
            alias = _safe_alias(f"max_{c}", "max_value")
            add_metric(IRMetric(agg="max", table=t, column=c, alias=alias))

        if wants_min:
            if mentioned_date_cols:
                t, c, _ = mentioned_date_cols[0]
            else:
                t, c = _metric_target_for_numeric()
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
    dedupe_row_request = bool(
        re.search(r"\bone\s+(?:row|record|result)\s+(?:per|for\s+each)\b", q)
    )
    aggregate_like = bool(
        re.search(
            r"\b(group\s+by|count|number\s+of|sum|avg|average|percentage|percent|ratio|rate)\b",
            q,
        )
        or (re.search(r"\bfor\s+each\b", q) and not dedupe_row_request)
        or (re.search(r"\bper\b", q) and not dedupe_row_request)
    )
    date_day_group_requested = bool(re.search(r"\b(for\s+each|per|by)\s+(?:date|day)\b", q))
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
        for v in _column_variants(table, col):
            pat = re.escape(v).replace(r"\ ", r"\s+")
            if re.search(rf"\b(?:for\s+each|per|by|group\s+by)\s+{pat}\b", q, flags=re.I):
                return True
        return False

    # Time bucket dimensions.
    if re.search(r"\b(for\s+each|per|by)\s+month\b", q):
        date_col = _find_best_date_column(subject, allowed)
        if date_col:
            add_dim(IRDimension(table=subject, column=date_col, alias="month_name", transform="month_name", visible=True))
            add_dim(IRDimension(table=subject, column=date_col, alias="__month_num", transform="month_num", visible=False))

    if re.search(r"\b(for\s+each|per|by)\s+year\b", q):
        date_col = _find_best_date_column(subject, allowed)
        if date_col:
            add_dim(IRDimension(table=subject, column=date_col, alias="year", transform="year", visible=True))

    # Entity id/name intent.
    subj_pat = _table_entity_pattern(subject) if subject else ""
    subject_id_requested = False
    if subject and subj_pat:
        id_like_token = r"(?:id|ids|code|codes|number(?!\s+of\b)|numbers(?!\s+of\b))"
        direct_subject_id = bool(
            re.search(
                rf"\b(?:{subj_pat})\b(?:[\s,;:()'\"-]+\w+){{0,6}}[\s,;:()'\"-]+"
                rf"{id_like_token}\b",
                q,
            )
        )
        reverse_subject_id = bool(
            re.search(
                rf"\b{id_like_token}\b(?:[\s,;:()'\"-]+\w+){{0,6}}"
                rf"[\s,;:()'\"-]+(?:of|for)\s+(?:{subj_pat})\b",
                q,
            )
        )
        if direct_subject_id or reverse_subject_id:
            subject_id_requested = True
            id_col = _best_id_column(subject, allowed)
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
            if aggregate_like:
                # In aggregate-like intents, keep grouping columns only.
                if col.lower() in nonnull_cols:
                    continue
                if (
                    (table.lower(), col.lower()) in distinct_metric_targets
                    and not _explicit_group_dim_requested(table, col)
                ):
                    continue
                if _is_numeric_column(col):
                    continue
                cl = col.lower()
                is_id_like = cl.endswith("id") or cl.endswith("number") or cl.endswith("code")
                if is_id_like and not subject_id_requested:
                    continue
                if _is_date_column(col) and not date_day_group_requested:
                    continue
            add_dim(IRDimension(table=table, column=col))

    # Generic "name" request without explicit entity token.
    if subject and re.search(r"\bname(s)?\b", q):
        name_col = _best_name_column(subject, allowed)
        if name_col:
            add_dim(IRDimension(table=subject, column=name_col))

    # Lookup fallback.
    if not dims and (qtype or "").lower() == "lookup":
        name_col = _best_name_column(subject, allowed)
        id_col = _best_id_column(subject, allowed)
        if name_col:
            add_dim(IRDimension(table=subject, column=name_col))
        elif id_col:
            add_dim(IRDimension(table=subject, column=id_col))

    return dims


def _detect_filters(
    question: str,
    subject: str,
    allowed: Dict[str, Set[str]],
    column_mentions: List[Tuple[str, str, int]],
    metrics: List[IRMetric],
) -> List[IRFilter]:
    q = (question or "").lower()
    q_raw = (question or "").strip()
    filters: List[IRFilter] = []

    def add_filter(f: IRFilter) -> None:
        for ex in filters:
            if (
                ex.table.lower() == f.table.lower()
                and ex.column.lower() == f.column.lower()
                and ex.op == f.op
                and ex.value == f.value
                and ex.value2 == f.value2
            ):
                return
        filters.append(f)

    def _pick_text_target_column() -> Optional[Tuple[str, str]]:
        # Prefer explicit "check number" mention when present.
        if re.search(r"\bcheck\s+number\b", q):
            for c in allowed.get(subject, set()):
                cl = c.lower()
                if "check" in cl and "number" in cl:
                    return (subject, c)
        # Fallback to first non-numeric/non-date mention.
        for t, c, _ in column_mentions:
            if not _is_numeric_column(c) and not _is_date_column(c):
                return (t, c)
        name_col = _best_name_column(subject, allowed)
        if name_col:
            return (subject, name_col)
        return None

    # Numeric comparisons.
    numeric_col = None
    numeric_mentions = [(t, c, p) for (t, c, p) in column_mentions if _is_numeric_column(c)]
    cond_pos_candidates: List[int] = []
    for pat in (
        r"\bbetween\b",
        r"\bat\s+least\b",
        r"\bat\s+most\b",
        r"\bminimum\b",
        r"\bmaximum\b",
        r"\bgreater\s+than\b",
        r"\bmore\s+than\b",
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
        if m0.column != "*" and (_is_numeric_column(m0.column) or m0_agg == "count"):
            numeric_col = (m0.table, m0.column)
    if not numeric_col:
        c0 = _find_best_numeric_column(subject, allowed)
        if c0:
            numeric_col = (subject, c0)

    m_between = re.search(
        rf"\bbetween\s+({_NUMBER_TOKEN_PATTERN})\s+and\s+({_NUMBER_TOKEN_PATTERN})\b",
        q,
    )
    if m_between and numeric_col:
        v1 = _to_float_token(m_between.group(1))
        v2 = _to_float_token(m_between.group(2))
        t, c = numeric_col
        if v1 is not None and v2 is not None:
            add_filter(IRFilter(table=t, column=c, op="between", value=v1, value2=v2))

    m_gte = re.search(rf"\b(?:at\s+least|minimum\s+of)\s+({_NUMBER_TOKEN_PATTERN})\b", q)
    if m_gte and numeric_col:
        v = _to_float_token(m_gte.group(1))
        t, c = numeric_col
        if v is not None:
            add_filter(IRFilter(table=t, column=c, op="gte", value=v))

    m_lte = re.search(rf"\b(?:at\s+most|maximum\s+of)\s+({_NUMBER_TOKEN_PATTERN})\b", q)
    if m_lte and numeric_col:
        v = _to_float_token(m_lte.group(1))
        t, c = numeric_col
        if v is not None:
            add_filter(IRFilter(table=t, column=c, op="lte", value=v))

    m_gt = re.search(rf"\b(?:greater\s+than|more\s+than|over)\s+({_NUMBER_TOKEN_PATTERN})\b", q)
    if m_gt and numeric_col:
        v = _to_float_token(m_gt.group(1))
        t, c = numeric_col
        if v is not None:
            add_filter(IRFilter(table=t, column=c, op="gt", value=v))

    m_lt = re.search(rf"\b(?:less\s+than|under|below)\s+({_NUMBER_TOKEN_PATTERN})\b", q)
    if m_lt and numeric_col:
        v = _to_float_token(m_lt.group(1))
        t, c = numeric_col
        if v is not None:
            add_filter(IRFilter(table=t, column=c, op="lt", value=v))

    # Year filters.
    years = re.findall(r"\b((?:19|20)\d{2})\b", q)
    if years:
        date_col = None
        date_table = subject
        for t, c, _ in column_mentions:
            if _is_date_column(c):
                date_table, date_col = t, c
                break
        if not date_col:
            date_col = _find_best_date_column(subject, allowed)
        if not date_col:
            # Fallback: use metric table date when subject table has no date column
            # (e.g., subject customers, metric from payments, filter year 2004).
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
        if date_col:
            add_filter(IRFilter(table=date_table, column=date_col, op="year_eq", value=int(years[0])))

    # Country/city style location filter:
    # "customers in Japan", "customers from USA".
    location_match = re.search(
        r"\b(?:in|from)\s+([A-Za-z][A-Za-z\s-]{1,40}?)"
        r"(?=\s+(?:ordered|sorted|grouped|having|where|with|without|show|return|list)\b|[.,;:!?]|\s*$)",
        q_raw,
        flags=re.I,
    )
    if location_match:
        loc = location_match.group(1).strip().strip(".,;:!?")
        loc_low = loc.lower()
        stop_loc = {
            "the", "a", "an", "that", "this", "those", "these",
            "latest", "earliest", "highest", "lowest",
            "year", "month", "week", "day",
        }
        is_deictic_ref = bool(re.search(r"^(?:that|this|those|these|their|its)\b", loc_low))
        loc_tokens = [t for t in re.findall(r"[a-z0-9]+", loc_low) if t]
        content_tokens = [t for t in loc_tokens if t not in _FUNCTION_WORD_TOKENS]

        # Schema-aware check: phrases composed only of schema terms
        # (e.g., "that territory", "this status") are references, not literal values.
        schema_terms: Set[str] = set()
        for c in allowed.get(subject, set()):
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
            and not is_schema_reference_phrase
        ):
            # Prefer country column, fallback city.
            target_col = None
            for c in allowed.get(subject, set()):
                if c.lower() == "country":
                    target_col = c
                    break
            if not target_col:
                for c in allowed.get(subject, set()):
                    if c.lower() == "city":
                        target_col = c
                        break
            if target_col:
                add_filter(IRFilter(table=subject, column=target_col, op="eq", value=loc))

    # Status IN ('a','b',...)
    quoted_values = re.findall(r"""["'“”]([^"'“”]+)["'“”]""", question or "")
    if quoted_values and re.search(r"\bstatus\b", q):
        status_col = None
        status_table = subject
        for c in allowed.get(subject, set()):
            if "status" in c.lower():
                status_col = c
                break
        if status_col:
            if len(quoted_values) >= 2:
                add_filter(IRFilter(table=status_table, column=status_col, op="in", value=[v.strip() for v in quoted_values]))
            else:
                add_filter(IRFilter(table=status_table, column=status_col, op="eq", value=quoted_values[0].strip()))

    # Missing/empty (skip when intent is explicit non-null/not-null).
    if re.search(r"\b(missing|null|empty|blank)\b", q) and not re.search(
        r"\b(?:non[-\s]?null|not\s+null)\b",
        q,
    ):
        target = None
        for t, c, _ in column_mentions:
            if "status" in c.lower() or "state" in c.lower() or "name" in c.lower():
                target = (t, c)
                break
        if not target:
            for c in allowed.get(subject, set()):
                if "state" in c.lower():
                    target = (subject, c)
                    break
        if target:
            add_filter(IRFilter(table=target[0], column=target[1], op="is_null_or_empty"))

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
            target = _pick_text_target_column()
            if target:
                add_filter(IRFilter(table=target[0], column=target[1], op="regexp", value=regex))
                strict_regex_applied = True

    if not strict_regex_applied:
        m_prefix = re.search(
            r"""\b(?:start(?:s)?\s+with|begins?\s+with)\b(?:\s+the\s+letter)?\s*["'“”]?([A-Za-z0-9]+)""",
            q,
            flags=re.I,
        )
        if m_prefix:
            token = m_prefix.group(1)
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
            target = None
            name_prefix_intent = bool(
                re.search(r"\bnames?\b(?:[\s,;:()'\"-]+\w+){0,4}[\s,;:()'\"-]+(?:start(?:s)?|begin(?:s)?)\b", q)
                or re.search(r"\b(?:start(?:s)?|begin(?:s)?)\b(?:[\s,;:()'\"-]+\w+){0,4}[\s,;:()'\"-]+\bnames?\b", q)
            )
            if name_prefix_intent:
                name_col = _best_name_column(subject, allowed)
                if name_col:
                    target = (subject, name_col)
            if not target:
                target = _pick_text_target_column()
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

    # IN list from ids/codes phrasing.
    # Pattern: "<entity> with codes 1,2,3" (e.g., offices with codes 1,2,3).
    in_applied = False
    m_entity_code_list = re.search(
        r"\b([a-z][a-z_]{2,})\s+(?:with\s+)?(?:id|ids|code|codes|number|numbers)\b(?P<tail>[^.]{0,80})",
        q,
    )
    if m_entity_code_list:
        ent_raw = (m_entity_code_list.group(1) or "").lower()
        tail = m_entity_code_list.group("tail") or ""
        nums = re.findall(r"\b\d+\b", tail)
        if len(nums) >= 2:
            ent = _safe_get_singular_form(ent_raw).replace("_", "").lower()
            subj_cols = list(allowed.get(subject, set()))
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

    id_context = re.search(r"\b(?:id|ids|code|codes|number|numbers)\b(?P<tail>[^.]{0,80})", q)
    if id_context and not in_applied:
        tail = id_context.group("tail") or ""
        has_list_signal = bool(
            re.search(r"\b(or|and)\b", tail)
            or re.search(r"\b\d+\s*,\s+\d+\b", tail)
            or re.search(r"\b\d+,\d+,\d+\b", tail)
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
                id_col = _best_id_column(subject, allowed)
                if id_col:
                    target = (subject, id_col)
            if target:
                add_filter(IRFilter(table=target[0], column=target[1], op="in", value=nums))

    return filters


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
    dimensions: List[IRDimension],
    metrics: List[IRMetric],
    column_mentions: List[Tuple[str, str, int]],
) -> Tuple[List[IROrder], Optional[int]]:
    q = (question or "").lower()
    ordering: List[IROrder] = []
    limit: Optional[int] = None

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
    if m_lim:
        limit = int(m_lim.group(1))
    elif re.search(r"\b(?:highest|lowest|largest|smallest|newest|oldest|latest|earliest)\b", q):
        # Handles forms like "show the 10 products with the highest ..."
        m_ranked_n = re.search(
            r"\b(?:show|list|display|find|return|get)\s+(?:the\s+)?(\d{1,3})\b",
            q,
        )
        if not m_ranked_n:
            m_ranked_n = re.search(
                r"\bthe\s+(\d{1,3})\s+\w+(?:\s+\w+){0,4}\s+with\s+(?:the\s+)?(?:highest|lowest|largest|smallest|newest|oldest|latest|earliest)\b",
                q,
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
        desc = bool(re.search(r"\b(desc|descending|highest|largest|latest|newest|top)\b", q))
        asc = bool(re.search(r"\b(asc|ascending|lowest|smallest|earliest|oldest|bottom)\b", q))
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
        if re.search(
            r"\b(highest|largest|latest|newest|desc|descending)\b",
            seg,
            flags=re.I,
        ):
            return "DESC"
        if re.search(
            r"\b(lowest|smallest|earliest|oldest|asc|ascending|alphabetical(?:ly)?|alphabetic(?:ally)?)\b",
            seg,
            flags=re.I,
        ):
            return "ASC"
        return direction

    # Explicit ORDER BY phrase.
    m_order = re.search(r"\b(?:order(?:ed)?\s+by|sort(?:ed)?\s+by)\s+(.+?)(?:[.;]|$)", q)
    if m_order:
        phrase = m_order.group(1).strip()
        segments = [
            seg.strip(" ,")
            for seg in re.split(r"\s*,\s*|\s+\bthen\b\s+", phrase, flags=re.I)
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
                if alias_words and alias_words in seg:
                    add_order(IROrder(kind="alias", by=d.alias, direction=seg_dir))
                    matched = True
                    break
            if matched:
                continue

            # Column mentions fallback.
            for t, c, _ in column_mentions:
                c_words = _camel_to_words(c)
                if c_words and c_words in seg:
                    add_order(IROrder(kind="column", by=c, table=t, direction=seg_dir))
                    matched = True
                    break

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
            ordering.append(IROrder(kind="column", by=d0.column, table=d0.table, direction="ASC"))
            return ordering, limit

    # Month special ordering.
    if any(d.alias == "__month_num" for d in dimensions):
        d = "DESC" if re.search(r"\b(latest|newest|descending)\b", q) else "ASC"
        ordering.append(IROrder(kind="expression", by="MONTH", direction=d))
        return ordering, limit

    # Ranking-style fallback: if question implies extremum and no explicit ORDER BY phrase,
    # sort by first numeric column mention.
    if not ordering and re.search(r"\b(highest|lowest|largest|smallest|newest|oldest|latest|earliest|top|bottom)\b", q):
        numeric_mentions = [(t, c, p) for (t, c, p) in column_mentions if _is_numeric_column(c)]
        if numeric_mentions:
            t, c, _ = numeric_mentions[0]
            add_order(IROrder(kind="column", by=c, table=t, direction=direction))

    # Top-N fallback by first metric.
    if limit and metrics:
        add_order(IROrder(kind="alias", by=metrics[0].alias, direction="DESC" if "bottom" not in q else "ASC"))
    elif limit and not ordering:
        numeric_mentions = [(t, c, p) for (t, c, p) in column_mentions if _is_numeric_column(c)]
        if numeric_mentions:
            t, c, _ = numeric_mentions[0]
            add_order(IROrder(kind="column", by=c, table=t, direction=direction))

    return ordering, limit


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

    column_mentions = _detect_column_mentions(question, allowed)
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
        dimensions=dimensions,
        metrics=metrics,
        column_mentions=column_mentions,
    )
    include_zero = _detect_include_zero(
        question=question,
        subject=subject,
        metrics=metrics,
        existence=existence,
    )
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

    ir = IntentIR(
        subject_entity=subject,
        dimensions=dimensions,
        metrics=metrics,
        filters=filters,
        existence=existence,
        ordering=ordering,
        limit=limit,
        include_zero=include_zero,
        select_distinct=select_distinct,
        qtype=qtype,
        notes=notes,
        confidence=confidence,
        planner_compatible=planner_compatible,
        mentioned_tables=mentioned_tables,
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
        or intent_ir.get("filters")
        or intent_ir.get("existence")
    ):
        return False
    return True
