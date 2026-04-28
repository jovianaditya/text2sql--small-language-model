import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Set, Tuple


def _default_rules_path() -> Path:
    return Path(__file__).with_name("t2sql_rules.yml")


def _coerce_non_empty_list(val: Any) -> list[str]:
    if not isinstance(val, list):
        return []
    out: list[str] = []
    for item in val:
        s = str(item).strip()
        if s:
            out.append(s)
    return out


@lru_cache(maxsize=1)
def _load_rules() -> Dict[str, Any]:
    path_raw = os.getenv("T2SQL_RULES_PATH", "").strip()
    path = Path(path_raw).expanduser() if path_raw else _default_rules_path()
    if not path.is_absolute() and not path_raw:
        # Default path is absolute via __file__. Keep this branch for safety.
        path = _default_rules_path()

    if not path.exists():
        return {}

    try:
        import yaml  # type: ignore
    except Exception:
        return {}

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if not isinstance(data, dict):
        return {}
    return data


def reset_rulebook_cache() -> None:
    _load_rules.cache_clear()


def rule_value(path: Sequence[str], default: Any = None) -> Any:
    cur: Any = _load_rules()
    for key in path:
        if not isinstance(cur, dict):
            return default
        if key not in cur:
            return default
        cur = cur.get(key)
    return cur


def rule_list(path: Sequence[str], default: Optional[Iterable[str]] = None) -> list[str]:
    val = rule_value(path, None)
    if val is None:
        return [str(x).strip() for x in (default or []) if str(x).strip()]
    out = _coerce_non_empty_list(val)
    if out:
        return out
    return [str(x).strip() for x in (default or []) if str(x).strip()]


def rule_dict(path: Sequence[str], default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    val = rule_value(path, None)
    if isinstance(val, dict):
        return val
    return dict(default or {})


def normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


@lru_cache(maxsize=4096)
def _compile_phrase_pattern(phrase_norm: str) -> re.Pattern[str]:
    p = re.escape(phrase_norm).replace(r"\ ", r"\s+")
    starts_alnum = bool(re.match(r"^[a-z0-9_]", phrase_norm))
    ends_alnum = bool(re.search(r"[a-z0-9_]$", phrase_norm))
    left = r"\b" if starts_alnum else ""
    right = r"\b" if ends_alnum else ""
    return re.compile(rf"{left}{p}{right}", flags=re.I)


def phrase_in_text(text: str, phrase: str) -> bool:
    t = normalized_text(text)
    p = normalized_text(phrase)
    if not t or not p:
        return False
    return bool(_compile_phrase_pattern(p).search(t))


def any_phrase_in_text(text: str, phrases: Iterable[str]) -> bool:
    t = normalized_text(text)
    if not t:
        return False
    for phrase in phrases:
        p = normalized_text(str(phrase or ""))
        if not p:
            continue
        if _compile_phrase_pattern(p).search(t):
            return True
    return False


def _find_phrase_spans(text: str, phrases: Iterable[str]) -> list[Tuple[int, int]]:
    spans: list[Tuple[int, int]] = []
    t = normalized_text(text)
    if not t:
        return spans
    for phrase in phrases:
        p = normalized_text(str(phrase or ""))
        if not p:
            continue
        pat = _compile_phrase_pattern(p)
        for m in pat.finditer(t):
            spans.append((m.start(), m.end()))
    return spans


def _has_proximity_between_phrase_sets(
    text: str,
    left_phrases: Iterable[str],
    right_phrases: Iterable[str],
    max_gap_chars: int,
) -> bool:
    left = _find_phrase_spans(text, left_phrases)
    right = _find_phrase_spans(text, right_phrases)
    if not left or not right:
        return False
    gap = max(0, int(max_gap_chars))
    for ls, le in left:
        for rs, re_ in right:
            if rs <= le:
                d = ls - re_
            else:
                d = rs - le
            if abs(d) <= gap:
                return True
    return False


@lru_cache(maxsize=1)
def name_component_lexical_tokens() -> Set[str]:
    defaults = [
        "name",
        "fullname",
        "full_name",
        "first",
        "last",
        "given",
        "family",
        "surname",
        "forename",
        "middle",
    ]
    vals = rule_list(("name_intent", "name_component_column_hints", "lexical_tokens"), defaults)
    return {normalized_text(v).replace(" ", "_") for v in vals if normalized_text(v)}


@lru_cache(maxsize=1)
def name_component_synonym_hints() -> Set[str]:
    defaults = [
        "full name",
        "first name",
        "last name",
        "given name",
        "family name",
        "surname",
    ]
    vals = rule_list(("name_intent", "name_component_column_hints", "synonym_hints"), defaults)
    return {normalized_text(v) for v in vals if normalized_text(v)}


@lru_cache(maxsize=1)
def name_component_priority_groups() -> Dict[str, Set[str]]:
    raw = rule_dict(
        ("name_intent", "name_component_column_hints", "ordering_priority"),
        default={},
    )
    groups: Dict[str, Set[str]] = {}
    for k, v in raw.items():
        items = _coerce_non_empty_list(v)
        groups[str(k).strip().lower()] = {
            normalized_text(x).replace(" ", "_")
            for x in items
            if normalized_text(x)
        }
    return groups


@lru_cache(maxsize=1)
def identifier_id_like_terms() -> list[str]:
    defaults = [
        "id",
        "ids",
        "identifier",
        "identifiers",
        "code",
        "codes",
        "number",
        "numbers",
        "nomor",
        "nomors",
        "no",
        "kode",
    ]
    return rule_list(("identifier_intent", "id_like_terms"), defaults)


@lru_cache(maxsize=1)
def name_request_terms() -> list[str]:
    defaults = ["name", "names", "nama", "full name", "full names", "complete name", "nama lengkap"]
    split_terms = rule_list(("name_intent", "split_components", "require_any"), ["name", "names", "nama"])
    full_phrases = rule_list(("name_intent", "combined_full_name", "any_phrases"), ["full name", "complete name"])
    merged = list(dict.fromkeys(split_terms + full_phrases + defaults))
    return [normalized_text(x) for x in merged if normalized_text(x)]


@lru_cache(maxsize=1)
def retrieval_extra_stopwords() -> list[str]:
    defaults = [
        "show", "all", "list", "their", "that", "have", "no",
        "include", "includes", "including", "sort", "sorted", "sorting",
        "result", "results", "record", "records", "row", "rows",
        "display", "return", "get", "give", "find", "fetch", "please", "kindly",
        "alphabetical", "alphabetically", "alphabetized", "alphabetize",
        "reverse alphabetical", "reversed alphabetical", "reverse",
        "ascending", "descending", "distinct",
        "yang", "dan", "atau", "ke", "dari", "di", "itu", "ini",
        "tampilkan", "daftar",
    ]
    return rule_list(("retrieval_intent", "extra_stopwords"), defaults)


@lru_cache(maxsize=1)
def retrieval_generic_mention_terms() -> list[str]:
    defaults = [
        "name", "names", "id", "ids", "code", "codes", "number", "numbers",
        "date", "dates", "time", "status", "type", "types", "value", "values",
        "amount", "amounts", "total", "totals", "count", "counts", "description",
        "descriptions", "detail", "details", "record", "records", "row", "rows",
        "address", "city", "country", "state", "phone", "email", "contact",
    ]
    return rule_list(("retrieval_intent", "generic_mention_terms"), defaults)


@lru_cache(maxsize=1)
def retrieval_join_signals() -> list[str]:
    defaults = [
        "join", "gabung", "combine", "combine with", "along with", "together with",
        "and their", "with their", "beserta", "bersama", "dengan",
        "for each", "per ", "setiap",
    ]
    return rule_list(("retrieval_intent", "join_signals"), defaults)


def question_wants_split_name_components(question: str) -> bool:
    q = normalized_text(question)
    if not q:
        return False
    require_any = rule_list(
        ("name_intent", "split_components", "require_any"),
        ["name", "names", "nama"],
    )
    require_g1 = rule_list(
        ("name_intent", "split_components", "require_group_1"),
        ["first", "given", "forename"],
    )
    require_g2 = rule_list(
        ("name_intent", "split_components", "require_group_2"),
        ["last", "family", "surname"],
    )
    forbid = rule_list(
        ("name_intent", "split_components", "forbid_phrases"),
        ["full name", "complete name", "nama lengkap"],
    )
    if not any_phrase_in_text(q, require_any):
        return False
    if not any_phrase_in_text(q, require_g1):
        return False
    if not any_phrase_in_text(q, require_g2):
        return False
    if any_phrase_in_text(q, forbid):
        return False
    return True


def question_wants_combined_full_name(question: str) -> bool:
    q = normalized_text(question)
    if not q:
        return False
    any_phrases = rule_list(
        ("name_intent", "combined_full_name", "any_phrases"),
        ["full name", "complete name", "nama lengkap"],
    )
    if any_phrase_in_text(q, any_phrases):
        return True

    verbs = rule_list(
        ("name_intent", "combined_full_name", "any_verbs"),
        [
            "combine",
            "combined",
            "combining",
            "concatenate",
            "concatenated",
            "concatenating",
            "concat",
            "join",
            "joined",
            "joining",
            "gabung",
            "gabungkan",
        ],
    )
    if not verbs:
        return False

    name_terms = rule_list(
        ("name_intent", "combined_full_name", "require_any"),
        rule_list(("name_intent", "split_components", "require_any"), ["name", "names", "nama"]),
    )
    if not any_phrase_in_text(q, name_terms):
        return False

    gap_chars = rule_value(("name_intent", "combined_full_name", "verb_name_window_chars"), 48)
    try:
        gap = int(gap_chars)
    except Exception:
        gap = 48
    return _has_proximity_between_phrase_sets(q, verbs, name_terms, gap)


def rulebook_debug_snapshot() -> Dict[str, Any]:
    return {
        "path": os.getenv("T2SQL_RULES_PATH", "").strip() or str(_default_rules_path()),
        "loaded": bool(_load_rules()),
        "name_component_lexical_tokens": sorted(name_component_lexical_tokens()),
        "name_component_synonym_hints": sorted(name_component_synonym_hints()),
        "retrieval_extra_stopwords_count": len(retrieval_extra_stopwords()),
        "retrieval_generic_mention_terms_count": len(retrieval_generic_mention_terms()),
        "retrieval_join_signals_count": len(retrieval_join_signals()),
    }
