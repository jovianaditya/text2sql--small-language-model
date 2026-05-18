import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Set, Tuple


def _default_rules_path() -> Path:
    return Path(__file__).with_name("t2sql_rules.yml")


def _default_config_path() -> Path:
    return Path(__file__).with_name("t2sql_config.yml")


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
    _load_config.cache_clear()


@lru_cache(maxsize=1)
def _load_config() -> Dict[str, Any]:
    path_raw = (
        os.getenv("T2SQL_CONFIG_PATH", "").strip()
        or os.getenv("T2SQL_CONFIG_DEFAULTS_PATH", "").strip()
    )
    path = Path(path_raw).expanduser() if path_raw else _default_config_path()
    if not path.is_absolute() and not path_raw:
        path = _default_config_path()

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


def config_default_list(key: str, default: Optional[Iterable[str]] = None) -> list[str]:
    """
    Read config lexical defaults from rulebook path:
      config.lists.<key>
    """
    k = str(key or "").strip()
    if not k:
        return [str(x).strip() for x in (default or []) if str(x).strip()]
    from_overlay = rule_value(("config", "lists", k), None)
    if from_overlay is None:
        # Backward compatibility for older rulebook structure
        from_overlay = rule_value(("config_defaults", "lists", k), None)
    if isinstance(from_overlay, list):
        return _coerce_non_empty_list(from_overlay)
    if from_overlay is None:
        overlay_root = _load_config()
        for root_key in ("config", "config_defaults"):
            cur = overlay_root
            for part in (root_key, "lists", k):
                if not isinstance(cur, dict) or part not in cur:
                    cur = None
                    break
                cur = cur.get(part)
            if isinstance(cur, list):
                return _coerce_non_empty_list(cur)
    return rule_list(("config", "lists", k), rule_list(("config_defaults", "lists", k), default))


def config_default_dict(key: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Read config lexical dict defaults from rulebook path:
      config.dicts.<key>
    """
    k = str(key or "").strip()
    if not k:
        return dict(default or {})
    from_overlay = rule_value(("config", "dicts", k), None)
    if from_overlay is None:
        # Backward compatibility for older rulebook structure
        from_overlay = rule_value(("config_defaults", "dicts", k), None)
    if isinstance(from_overlay, dict):
        return dict(from_overlay)
    if from_overlay is None:
        overlay_root = _load_config()
        for root_key in ("config", "config_defaults"):
            cur = overlay_root
            for part in (root_key, "dicts", k):
                if not isinstance(cur, dict) or part not in cur:
                    cur = None
                    break
                cur = cur.get(part)
            if isinstance(cur, dict):
                return dict(cur)
    return rule_dict(("config", "dicts", k), rule_dict(("config_defaults", "dicts", k), default))


def config_default_scalar(key: str, default: Any = None) -> Any:
    """
    Read config scalar defaults from rulebook path:
      config.scalars.<key>
    """
    k = str(key or "").strip()
    if not k:
        return default
    from_overlay = rule_value(("config", "scalars", k), None)
    if from_overlay is None:
        # Backward compatibility for older rulebook structure
        from_overlay = rule_value(("config_defaults", "scalars", k), None)
    if from_overlay is not None and not isinstance(from_overlay, (list, dict)):
        return from_overlay
    if from_overlay is None:
        overlay_root = _load_config()
        for root_key in ("config", "config_defaults"):
            cur: Any = overlay_root
            for part in (root_key, "scalars", k):
                if not isinstance(cur, dict) or part not in cur:
                    cur = None
                    break
                cur = cur.get(part)
            if cur is not None and not isinstance(cur, (list, dict)):
                return cur
    return default


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


@lru_cache(maxsize=1)
def retrieval_probe_id_like_terms() -> list[str]:
    defaults = ["id", "identifier", "nomor", "no", "code", "kode"]
    return rule_list(("retrieval_intent", "probe", "id_like_terms"), defaults)


@lru_cache(maxsize=1)
def retrieval_probe_semantic_stopwords() -> Set[str]:
    defaults = [
        "a", "an", "the", "of", "for", "to", "in", "on", "at", "by", "with",
        "and", "or", "from", "per", "each", "every", "all", "any", "this",
        "that", "these", "those", "show", "list", "return", "display", "give",
        "find", "fetch", "get", "table", "column", "data", "value", "values",
    ]
    vals = rule_list(("retrieval_intent", "probe", "semantic_stopwords"), defaults)
    return {normalized_text(v) for v in vals if normalized_text(v)}


@lru_cache(maxsize=1)
def retrieval_probe_count_terms() -> list[str]:
    defaults = ["count", "berapa banyak", "how many"]
    return rule_list(("retrieval_intent", "probe", "count_terms"), defaults)


@lru_cache(maxsize=1)
def retrieval_probe_temporal_terms() -> list[str]:
    defaults = [
        "today", "yesterday", "last", "this", "bulan", "minggu", "tahun",
        "quarter", "month", "week", "year", "tanggal", "hari",
        "mingguan", "bulanan",
    ]
    return rule_list(("retrieval_intent", "probe", "temporal_terms"), defaults)


@lru_cache(maxsize=1)
def retrieval_probe_metric_fallback_terms() -> list[str]:
    defaults = ["amount", "price"]
    return rule_list(("retrieval_intent", "probe", "metric_fallback_terms"), defaults)


@lru_cache(maxsize=1)
def retrieval_probe_aggregate_temporal_fallback_terms() -> list[str]:
    defaults = ["date", "year", "month"]
    return rule_list(("retrieval_intent", "probe", "aggregate_temporal_fallback_terms"), defaults)


@lru_cache(maxsize=1)
def retrieval_probe_generic_temporal_fallback_terms() -> list[str]:
    defaults = ["date"]
    return rule_list(("retrieval_intent", "probe", "generic_temporal_fallback_terms"), defaults)


@lru_cache(maxsize=1)
def retrieval_probe_short_query_fallback_terms() -> list[str]:
    defaults = ["amount", "date"]
    return rule_list(("retrieval_intent", "probe", "short_query_fallback_terms"), defaults)


@lru_cache(maxsize=1)
def retrieval_probe_geo_country_terms() -> list[str]:
    defaults = ["country", "negara"]
    return rule_list(("retrieval_intent", "probe", "geo_keywords", "country"), defaults)


@lru_cache(maxsize=1)
def retrieval_probe_geo_city_terms() -> list[str]:
    defaults = ["city", "kota"]
    return rule_list(("retrieval_intent", "probe", "geo_keywords", "city"), defaults)


@lru_cache(maxsize=1)
def retrieval_probe_geo_region_terms() -> list[str]:
    defaults = ["region", "wilayah", "territory", "territories"]
    return rule_list(("retrieval_intent", "probe", "geo_keywords", "region"), defaults)


@lru_cache(maxsize=1)
def retrieval_probe_geo_country_targets() -> Set[str]:
    defaults = ["country", "nation"]
    vals = rule_list(("retrieval_intent", "probe", "geo_variant_targets", "country"), defaults)
    return {normalized_text(v) for v in vals if normalized_text(v)}


@lru_cache(maxsize=1)
def retrieval_probe_geo_city_targets() -> Set[str]:
    defaults = ["city", "municipality", "town"]
    vals = rule_list(("retrieval_intent", "probe", "geo_variant_targets", "city"), defaults)
    return {normalized_text(v) for v in vals if normalized_text(v)}


@lru_cache(maxsize=1)
def query_classification_ordinal_terms() -> list[str]:
    defaults = ["second", "third", "fourth", "fifth", "2nd", "3rd", "4th", "5th", "kedua", "ketiga", "keempat", "kelima"]
    return rule_list(("query_classification", "ordinal_terms"), defaults)


@lru_cache(maxsize=1)
def query_classification_ordinal_superlative_terms() -> list[str]:
    defaults = [
        "highest", "lowest", "largest", "smallest", "biggest", "top", "bottom",
        "best", "worst", "most", "least", "terbesar", "terkecil", "tertinggi", "terendah",
    ]
    return rule_list(("query_classification", "ordinal_superlative_terms"), defaults)


@lru_cache(maxsize=1)
def query_classification_row_container_terms() -> list[str]:
    defaults = ["row", "record", "entry"]
    return rule_list(("query_classification", "row_container_terms"), defaults)


@lru_cache(maxsize=1)
def query_classification_sort_terms() -> list[str]:
    defaults = ["sorted", "ordered"]
    return rule_list(("query_classification", "sort_terms"), defaults)


@lru_cache(maxsize=1)
def query_classification_group_terms() -> list[str]:
    defaults = ["per", "each", "every", "within", "partition"]
    return rule_list(("query_classification", "group_terms"), defaults)


@lru_cache(maxsize=1)
def query_classification_aggregate_terms() -> list[str]:
    defaults = ["count", "sum", "total", "average", "avg", "number of"]
    return rule_list(("query_classification", "aggregate_terms"), defaults)


@lru_cache(maxsize=1)
def query_classification_aggregate_shape_terms() -> list[str]:
    defaults = ["classify", "band", "bands", "grouping", "distribution"]
    return rule_list(("query_classification", "aggregate_shape_terms"), defaults)


@lru_cache(maxsize=1)
def aggregation_count_distinct_keywords() -> list[str]:
    defaults = ["distinct", "unique", "different", "unik", "berbeda"]
    return rule_list(("aggregation_intent", "count_distinct_keywords"), defaults)


@lru_cache(maxsize=1)
def aggregation_count_distinct_phrases() -> list[str]:
    defaults = ["count distinct", "distinct count", "how many distinct", "number of unique", "jumlah unik"]
    return rule_list(("aggregation_intent", "count_distinct_phrases"), defaults)


@lru_cache(maxsize=1)
def ordering_direction_desc_terms() -> list[str]:
    defaults = [
        "descending", "desc", "highest", "largest", "latest", "newest",
        "most", "reverse alphabetical", "reversed alphabetical", "reverse",
    ]
    return rule_list(("ordering_intent", "direction_terms", "desc"), defaults)


@lru_cache(maxsize=1)
def ordering_direction_asc_terms() -> list[str]:
    defaults = [
        "ascending", "asc", "a-z", "alphabetical", "chronological",
        "chronologically", "earliest", "oldest", "smallest", "lowest",
    ]
    return rule_list(("ordering_intent", "direction_terms", "asc"), defaults)


@lru_cache(maxsize=1)
def ordering_temporal_desc_terms() -> list[str]:
    defaults = ["latest", "newest", "most recent", "descending", "desc"]
    return rule_list(("ordering_intent", "temporal_terms", "desc"), defaults)


@lru_cache(maxsize=1)
def ordering_temporal_asc_terms() -> list[str]:
    defaults = ["earliest", "oldest", "ascending", "asc", "chronological", "chronologically"]
    return rule_list(("ordering_intent", "temporal_terms", "asc"), defaults)


@lru_cache(maxsize=1)
def ordering_temporal_intent_terms() -> list[str]:
    defaults = [
        "most recent", "least recent", "latest", "earliest",
        "newest", "oldest", "chronological", "chronologically",
    ]
    return rule_list(("ordering_intent", "temporal_intent_terms"), defaults)


@lru_cache(maxsize=1)
def temporal_year_column_terms() -> list[str]:
    defaults = ["year", "yr", "tahun"]
    return rule_list(("temporal_intent", "year_column_terms"), defaults)


@lru_cache(maxsize=1)
def final_order_metric_desc_cues() -> list[str]:
    defaults = [
        "more", "greater", "higher", "larger", "most",
        "greater number", "higher number", "larger number",
    ]
    return rule_list(("heuristic_repairs", "final_order", "metric_desc_cues"), defaults)


@lru_cache(maxsize=1)
def final_order_first_position_terms() -> list[str]:
    defaults = ["first", "appear first", "come first", "show first"]
    return rule_list(("heuristic_repairs", "final_order", "first_position_terms"), defaults)


@lru_cache(maxsize=1)
def grouped_self_fk_independence_cues() -> list[str]:
    defaults = [
        "not necessarily",
        "not required to be",
        "regardless of whether",
        "independent of",
        "may or may not be",
    ]
    return rule_list(("heuristic_repairs", "grouped_self_fk_existence", "independence_cues"), defaults)


@lru_cache(maxsize=1)
def existence_count_metric_phrases() -> list[str]:
    defaults = [
        "ever ordered",
        "been ordered",
        "ever been ordered",
        "ordered at least once",
    ]
    return rule_list(("heuristic_repairs", "existence_count_metric", "phrases"), defaults)


@lru_cache(maxsize=1)
def guidance_fact_support_phrases() -> list[str]:
    defaults = [
        "order line",
        "order lines",
        "ordered products",
        "total quantity",
        "total revenue",
    ]
    return rule_list(("prompt_guidance", "fact_support", "phrases"), defaults)


@lru_cache(maxsize=1)
def group_average_difference_window_phrases() -> list[str]:
    defaults = [
        "gap between",
        "difference between",
        "gap from average",
        "difference from average",
    ]
    return rule_list(("heuristic_repairs", "group_average_difference_window", "phrases"), defaults)


@lru_cache(maxsize=1)
def ordering_superlative_keywords() -> list[str]:
    defaults = [
        "top", "bottom", "most", "least", "most expensive", "least expensive",
        "cheapest", "highest", "lowest", "largest", "smallest", "newest",
        "latest", "earliest", "oldest", "termahal", "termurah",
    ]
    return rule_list(("ordering_intent", "superlative_keywords"), defaults)


@lru_cache(maxsize=1)
def ordering_window_context_regex_patterns() -> list[str]:
    defaults = [
        r"\brank\b.*\b(ascending|descending)\s+order\b",
        r"\brow[_\s]?number\b.*\b(ascending|descending)\s+order\b",
        r"\bdense[_\s]?rank\b.*\b(ascending|descending)\s+order\b",
        r"\bpartition\s+by\b.*\b(ascending|descending)\b",
        r"\bbased\s+on\b.*\b(ascending|descending)\s+order\b",
        r"\b(cumulative|running)\b.*\bordered\s+by\b",
        r"\b(second|third|fourth|fifth|\d+(?:st|nd|rd|th))\b.*\b(sorted|ordered)\s+by\b",
        r"\b(sorted|ordered)\s+by\b.*\b(second|third|fourth|fifth|\d+(?:st|nd|rd|th))\b",
        r"\brank\b[^.]{0,96}\bby\b[^.]{0,64}\b(?:ascending|descending|asc|desc)\b",
        r"\brow[_\s]?number\b[^.]{0,96}\bby\b[^.]{0,64}\b(?:ascending|descending|asc|desc)\b",
        r"\bdense[_\s]?rank\b[^.]{0,96}\bby\b[^.]{0,64}\b(?:ascending|descending|asc|desc)\b",
        r"\bwithin\s+(?:each|the\s+same)\b[^.]{0,96}\bby\b[^.]{0,64}\b(?:ascending|descending|asc|desc)\b",
        r"\brank\b[^.]{0,96}\bordered\s+by\b",
        r"\brow[_\s]?number\b[^.]{0,96}\bordered\s+by\b",
        r"\bdense[_\s]?rank\b[^.]{0,96}\bordered\s+by\b",
    ]
    return rule_list(("ordering_intent", "window_context_regex"), defaults)


@lru_cache(maxsize=1)
def ordering_phrase_regex_patterns() -> list[str]:
    defaults = [
        r"\bfrom\s+(the\s+)?(highest|lowest|latest|earliest|newest|oldest)",
        r"\bfrom\s+(a\s+to\s+z|z\s+to\s+a)",
        r"\b(sorted|ordered)\s+from\b",
        r"\b(?:start|starting|begin|beginning)\s+(?:with|from)\s+(?:the\s+)?(?:highest|lowest|latest|earliest|newest|oldest|most\s+recent|least\s+recent)",
        r"\b(?:most\s+recent|least\s+recent|latest|earliest|newest|oldest)\s+first\b",
        r"\b(?:largest|highest|greatest|biggest|lowest|smallest|least)\b[^.]{0,48}\b(?:at|on)\s+the\s+(?:top|bottom)\b",
        r"\bsort\s+from\b",
        r"\b(?:order|sort|arrange)(?:ed)?\s+(?:the\s+)?(?:[a-z0-9_-]+\s+){0,6}chronologically\b",
        r"\bin\s+chronological\s+order\b",
    ]
    return rule_list(("ordering_intent", "ordering_phrase_regex"), defaults)


@lru_cache(maxsize=1)
def ordering_final_output_sort_regex_patterns() -> list[str]:
    defaults = [
        r"\b(?:final\s+)?(?:result|results|output|rows?)\s+(?:should\s+be\s+)?(?:sorted|ordered|arranged)\s+by\b",
        r"\bsort(?:ed)?\s+(?:the\s+)?(?:final\s+)?(?:result|results|output|rows?)\s+by\b",
        r"\border(?:ed)?\s+(?:the\s+)?(?:final\s+)?(?:result|results|output|rows?)\s+by\b",
        r"\barrange\s+(?:the\s+)?(?:final\s+)?(?:result|results|output|rows?)\s+by\b",
    ]
    return rule_list(("ordering_intent", "final_output_sort_regex"), defaults)


@lru_cache(maxsize=1)
def ordering_explicit_sort_command_regex_patterns() -> list[str]:
    defaults = [
        r"(?:^|[.;!?]\s*)(?:sort|sorted|order|ordered|arrange|arranged)\s+by\b",
        r"(?:^|[.;!?]\s*|,\s*then\s+)(?:sort|order|arrange)\s+by\b",
    ]
    return rule_list(("ordering_intent", "explicit_sort_command_regex"), defaults)


@lru_cache(maxsize=1)
def ordering_first_last_row_selection_regex_patterns() -> list[str]:
    defaults = [
        r"\b(?:first|last)\s+\d+\b(?!\s*(?:characters?|chars?|letters?|digits?|words?|months?|years?|days?|quarters?))",
        r"\b(?:first|last)\s+(?:row|rows|record|records|entry|entries|result|results|item|items|tuple|tuples)\b",
        r"\b(?:the\s+)?(?:first|last)\s+page\b",
    ]
    return rule_list(("ordering_intent", "first_last_row_selection_regex"), defaults)


@lru_cache(maxsize=1)
def winner_row_rank_terms() -> list[str]:
    defaults = [
        "largest", "highest", "greatest", "maximum", "max", "biggest",
        "smallest", "lowest", "least", "minimum", "min",
        "best", "worst", "top", "bottom",
        "best-selling", "best selling", "top-selling", "top selling",
        "terbesar", "tertinggi", "maksimum",
        "terkecil", "terendah", "minimum",
        "paling besar", "paling kecil",
    ]
    return rule_list(("winner_row_intent", "rank_terms"), defaults)


@lru_cache(maxsize=1)
def winner_row_grouped_report_terms() -> list[str]:
    defaults = ["summarize", "summary", "group", "grouped", "aggregate", "break down", "roll up"]
    return rule_list(("winner_row_intent", "grouped_report_terms"), defaults)


@lru_cache(maxsize=1)
def winner_row_row_object_terms() -> list[str]:
    defaults = ["row", "record", "id", "code", "name", "date", "number", "who", "which"]
    return rule_list(("winner_row_intent", "row_object_terms"), defaults)


@lru_cache(maxsize=1)
def winner_row_reporting_verbs() -> list[str]:
    defaults = ["show", "list", "display", "find", "return"]
    return rule_list(("winner_row_intent", "reporting_verbs"), defaults)


@lru_cache(maxsize=1)
def winner_row_same_row_terms() -> list[str]:
    defaults = [
        "and its", "and their", "and the", "along with",
        "dan", "beserta", "serta",
        "which", "that", "who",
        "yang mana", "yang",
    ]
    return rule_list(("winner_row_intent", "same_row_terms"), defaults)


@lru_cache(maxsize=1)
def winner_row_context_terms() -> list[str]:
    defaults = [
        "when", "date when", "occurred", "on what date", "at what date",
        "what date", "which date",
        "kapan", "tanggal", "pada tanggal", "waktu", "tanggalnya", "kapan terjadinya",
    ]
    return rule_list(("winner_row_intent", "context_terms"), defaults)


@lru_cache(maxsize=1)
def winner_row_multi_attr_connectors() -> list[str]:
    defaults = ["and", "with", "beserta", "dan"]
    return rule_list(("winner_row_intent", "multi_attr_connectors"), defaults)


@lru_cache(maxsize=1)
def winner_row_tie_terms() -> list[str]:
    defaults = [
        "if multiple", "if there are multiple", "if there is still a tie",
        "tie", "ties", "tied", "choose the one", "pick the one",
    ]
    return rule_list(("winner_row_intent", "tie_terms"), defaults)


@lru_cache(maxsize=1)
def winner_row_aggregate_metric_terms() -> list[str]:
    defaults = ["total", "sum", "average", "avg", "count", "number of", "quantity"]
    return rule_list(("winner_row_intent", "aggregate_metric_terms"), defaults)


@lru_cache(maxsize=1)
def winner_row_aggregate_guard_terms() -> list[str]:
    defaults = ["total", "sum of", "count", "berapa banyak", "jumlah total"]
    return rule_list(("winner_row_intent", "aggregate_guard_terms"), defaults)


@lru_cache(maxsize=1)
def winner_row_grouped_report_metric_terms() -> list[str]:
    defaults = [
        "how many", "count", "number of", "total",
        "average", "avg", "sum",
        "highest", "lowest", "largest", "smallest",
        "maximum", "minimum", "max", "min",
    ]
    return rule_list(("winner_row_intent", "grouped_report_metric_terms"), defaults)


@lru_cache(maxsize=1)
def threshold_count_object_stop_terms() -> list[str]:
    defaults = ["and", "or", "but", "where", "with", "having", "filter", "sort", "order"]
    return rule_list(("heuristic_repairs", "threshold_count_object", "local_stop_terms"), defaults)


@lru_cache(maxsize=1)
def aggregate_threshold_intent_aggregate_terms() -> list[str]:
    defaults = ["count", "sum", "total", "avg", "average", "mean", "number of"]
    return rule_list(("heuristic_repairs", "aggregate_threshold_intent", "aggregate_terms"), defaults)


@lru_cache(maxsize=1)
def aggregate_threshold_intent_comparison_terms() -> list[str]:
    defaults = ["more than", "over", "greater than", "at least", "less than", "under", "below", "exceeds"]
    return rule_list(("heuristic_repairs", "aggregate_threshold_intent", "comparison_terms"), defaults)


@lru_cache(maxsize=1)
def aggregate_threshold_intent_clause_terms() -> list[str]:
    defaults = ["where", "whose", "having", "with"]
    return rule_list(("heuristic_repairs", "aggregate_threshold_intent", "clause_terms"), defaults)


@lru_cache(maxsize=1)
def ordered_quantity_repair_order_terms() -> list[str]:
    defaults = ["order", "ordered"]
    return rule_list(("heuristic_repairs", "ordered_quantity_stock_aggregate", "order_terms"), defaults)


@lru_cache(maxsize=1)
def ordered_quantity_repair_quantity_terms() -> list[str]:
    defaults = ["quantity", "qty", "unit", "units"]
    return rule_list(("heuristic_repairs", "ordered_quantity_stock_aggregate", "quantity_terms"), defaults)


@lru_cache(maxsize=1)
def ordered_quantity_repair_inventory_terms() -> list[str]:
    defaults = ["stock", "stocked", "inventory", "in stock", "on hand"]
    return rule_list(("heuristic_repairs", "ordered_quantity_stock_aggregate", "inventory_terms"), defaults)


@lru_cache(maxsize=1)
def ordered_quantity_repair_non_quantity_metric_terms() -> list[str]:
    defaults = ["price", "cost", "amount", "value", "rate"]
    return rule_list(("heuristic_repairs", "ordered_quantity_stock_aggregate", "non_quantity_metric_terms"), defaults)


@lru_cache(maxsize=1)
def positive_row_filter_keep_phrases() -> list[str]:
    defaults = [
        "in stock",
        "available",
        "availability",
        "with stock",
        "stocked",
        "non-zero",
        "positive",
        "greater than 0",
        "more than 0",
        "at least 1",
        "minimum 1",
    ]
    return rule_list(("heuristic_repairs", "unrequested_positive_row_filters", "keep_phrases"), defaults)


@lru_cache(maxsize=1)
def latest_year_scope_aggregate_terms() -> list[str]:
    defaults = [
        "sum",
        "total",
        "jumlah",
        "aggregate",
        "aggregated",
        "average",
        "avg",
        "mean",
        "count",
        "amount",
        "quantity",
        "value",
        "price",
        "cost",
        "rate",
        "ratio",
        "percent",
    ]
    return rule_list(("heuristic_repairs", "latest_year_same_year_aggregate", "aggregate_terms"), defaults)


@lru_cache(maxsize=1)
def semantic_question_token_stopwords() -> Set[str]:
    defaults = [
        "the", "and", "for", "with", "from", "that", "this", "those",
        "these", "their", "there", "have", "has", "had", "into", "onto",
        "than", "then", "show", "list", "find", "give", "return", "display",
    ]
    vals = rule_list(("agent_semantics", "question_token_stopwords"), defaults)
    return {normalized_text(v) for v in vals if normalized_text(v)}


@lru_cache(maxsize=1)
def semantic_contract_count_how_many_phrases() -> list[str]:
    defaults = ["how many"]
    return rule_list(("agent_semantics", "semantic_contract", "count_how_many_phrases"), defaults)


@lru_cache(maxsize=1)
def semantic_contract_count_terms() -> list[str]:
    defaults = ["count", "count of"]
    return rule_list(("agent_semantics", "semantic_contract", "count_terms"), defaults)


@lru_cache(maxsize=1)
def semantic_contract_count_number_of_prefixes() -> list[str]:
    defaults = ["number of", "total number of"]
    return rule_list(("agent_semantics", "semantic_contract", "count_number_of_prefixes"), defaults)


@lru_cache(maxsize=1)
def semantic_contract_ordinal_terms() -> list[str]:
    defaults = ["first", "second", "third", "fourth", "fifth", "1st", "2nd", "3rd", "4th", "5th"]
    return rule_list(("agent_semantics", "semantic_contract", "ordinal_terms"), defaults)


@lru_cache(maxsize=1)
def semantic_contract_per_group_terms() -> list[str]:
    defaults = ["for each", "per", "within each", "for every"]
    return rule_list(("agent_semantics", "semantic_contract", "per_group_terms"), defaults)


@lru_cache(maxsize=1)
def semantic_contract_identifier_like_targets() -> Set[str]:
    defaults = [
        "id", "ids", "identifier", "identifiers", "number", "numbers",
        "code", "codes", "name", "names", "date", "dates", "status", "statuses",
    ]
    vals = rule_list(("agent_semantics", "semantic_contract", "identifier_like_targets"), defaults)
    return {normalized_text(v) for v in vals if normalized_text(v)}


@lru_cache(maxsize=1)
def semantic_contract_average_terms() -> list[str]:
    defaults = ["average", "avg", "mean"]
    return rule_list(("agent_semantics", "semantic_contract", "average_terms"), defaults)


@lru_cache(maxsize=1)
def semantic_contract_ratio_definition_terms() -> list[str]:
    defaults = ["defined as", "where"]
    return rule_list(("agent_semantics", "semantic_contract", "ratio_definition_terms"), defaults)


@lru_cache(maxsize=1)
def semantic_contract_ratio_count_reference_phrases() -> list[str]:
    defaults = ["number of", "count of", "total number of"]
    return rule_list(("agent_semantics", "semantic_contract", "ratio_count_reference_phrases"), defaults)


@lru_cache(maxsize=1)
def semantic_contract_ratio_operator_terms() -> list[str]:
    defaults = ["divided by", "over", "per"]
    return rule_list(("agent_semantics", "semantic_contract", "ratio_operator_terms"), defaults)


@lru_cache(maxsize=1)
def semantic_contract_sum_terms() -> list[str]:
    defaults = ["sum", "sum of"]
    return rule_list(("agent_semantics", "semantic_contract", "sum_terms"), defaults)


@lru_cache(maxsize=1)
def semantic_contract_sum_total_prefix_terms() -> list[str]:
    defaults = ["total"]
    return rule_list(("agent_semantics", "semantic_contract", "sum_total_prefix_terms"), defaults)


@lru_cache(maxsize=1)
def semantic_contract_sum_total_count_phrases() -> list[str]:
    defaults = ["total number", "total count", "total number of"]
    return rule_list(("agent_semantics", "semantic_contract", "sum_total_count_phrases"), defaults)


@lru_cache(maxsize=1)
def semantic_phrase_token_stopwords() -> Set[str]:
    defaults = [
        "a", "an", "the", "that", "this", "those", "these", "same",
        "within", "in", "of", "for", "to", "by", "on", "from",
        "including", "include", "includes", "excluded", "excluding",
        "except", "counting", "count", "itself", "themselves",
    ]
    vals = rule_list(("agent_semantics", "phrase_token_stopwords"), defaults)
    return {normalized_text(v) for v in vals if normalized_text(v)}


@lru_cache(maxsize=1)
def projection_null_fill_missing_terms() -> list[str]:
    defaults = ["missing", "null", "empty", "blank"]
    return rule_list(("agent_semantics", "projection_null_fill", "missing_terms"), defaults)


@lru_cache(maxsize=1)
def projection_null_fill_conditional_terms() -> list[str]:
    defaults = ["if", "when", "whenever"]
    return rule_list(("agent_semantics", "projection_null_fill", "conditional_terms"), defaults)


@lru_cache(maxsize=1)
def projection_null_fill_action_terms() -> list[str]:
    defaults = ["display", "show", "return", "use"]
    return rule_list(("agent_semantics", "projection_null_fill", "action_terms"), defaults)


@lru_cache(maxsize=1)
def projection_null_fill_fallback_terms() -> list[str]:
    defaults = ["n/a", "none", "unknown", "zero"]
    return rule_list(("agent_semantics", "projection_null_fill", "fallback_terms"), defaults)


@lru_cache(maxsize=1)
def numeric_threshold_comparison_terms() -> list[str]:
    defaults = [
        "greater than", "more than", "over", "above", "less than", "under", "below",
        "at least", "at most", "minimum", "minimum of", "maximum", "maximum of",
    ]
    return rule_list(("agent_semantics", "numeric_threshold", "comparison_terms"), defaults)


@lru_cache(maxsize=1)
def numeric_threshold_number_words() -> list[str]:
    defaults = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]
    return rule_list(("agent_semantics", "numeric_threshold", "number_words"), defaults)


@lru_cache(maxsize=1)
def global_avg_baseline_average_terms() -> list[str]:
    defaults = ["avg", "average", "mean"]
    return rule_list(("agent_semantics", "global_avg_baseline_compare", "average_terms"), defaults)


@lru_cache(maxsize=1)
def global_avg_baseline_terms() -> list[str]:
    defaults = ["overall", "global", "across all", "entire", "across the whole", "across the entire"]
    return rule_list(("agent_semantics", "global_avg_baseline_compare", "baseline_terms"), defaults)


@lru_cache(maxsize=1)
def global_avg_comparison_terms() -> list[str]:
    defaults = ["higher", "greater", "more", "above", "over", "lower", "less", "below", "under"]
    return rule_list(("agent_semantics", "global_avg_baseline_compare", "comparison_terms"), defaults)


@lru_cache(maxsize=1)
def global_avg_baseline_reference_terms() -> list[str]:
    defaults = ["overall", "global", "average", "mean"]
    return rule_list(("agent_semantics", "global_avg_baseline_compare", "baseline_reference_terms"), defaults)


@lru_cache(maxsize=1)
def offset_window_sequence_terms() -> list[str]:
    defaults = ["previous", "prior", "next", "following"]
    return rule_list(("agent_semantics", "offset_window_sequence", "offset_terms"), defaults)


@lru_cache(maxsize=1)
def offset_window_ordering_context_regex_patterns() -> list[str]:
    defaults = [
        r"\bordered\s+by\b",
        r"\bbased\s+on\b[^.]{0,80}\border\b",
        r"\b(?:date|time|timestamp)[-\s]+order\b",
    ]
    return rule_list(("agent_semantics", "offset_window_sequence", "ordering_context_regex"), defaults)


@lru_cache(maxsize=1)
def case_bucket_language_regex_patterns() -> list[str]:
    defaults = [
        r"\b(?:bucket|classify|categori[sz]e|group(?:ed|ing)?\s+into|bands?|ranges?|distribution)\b",
        r"\bgroup(?:ed|ing)?\b[^.]{0,80}\bby\s+whether\b",
        r"\bby\s+whether\b",
    ]
    return rule_list(("agent_semantics", "case_bucketing", "bucket_language_regex"), defaults)


@lru_cache(maxsize=1)
def case_bucket_numeric_threshold_regex_patterns() -> list[str]:
    defaults = [
        r"[<>]=?\s*\d",
        r"\bbetween\s+\d+(?:\.\d+)?\s+(?:and|-|to)\s+\d+(?:\.\d+)?\b",
    ]
    return rule_list(("agent_semantics", "case_bucketing", "numeric_threshold_regex"), defaults)


@lru_cache(maxsize=1)
def case_bucket_numeric_terms() -> list[str]:
    defaults = ["low", "mid", "medium", "high"]
    return rule_list(("agent_semantics", "case_bucketing", "numeric_bucket_terms"), defaults)


@lru_cache(maxsize=1)
def case_bucket_conditional_terms() -> list[str]:
    defaults = ["if", "when", "whenever", "otherwise", "else", "remaining", "rest"]
    return rule_list(("agent_semantics", "case_bucketing", "conditional_terms"), defaults)


@lru_cache(maxsize=1)
def case_bucket_missing_terms() -> list[str]:
    defaults = ["null", "missing", "blank", "empty", "not available", "n/a"]
    return rule_list(("agent_semantics", "case_bucketing", "missing_terms"), defaults)


@lru_cache(maxsize=1)
def case_bucket_comparison_terms() -> list[str]:
    defaults = ["later", "earlier", "before", "after", "greater than", "less than", "above", "below"]
    return rule_list(("agent_semantics", "case_bucketing", "comparison_terms"), defaults)


@lru_cache(maxsize=1)
def case_bucket_temporal_terms() -> list[str]:
    defaults = ["date", "time", "timestamp"]
    return rule_list(("agent_semantics", "case_bucketing", "temporal_terms"), defaults)


@lru_cache(maxsize=1)
def case_bucket_pattern_terms() -> list[str]:
    defaults = ["match", "matches", "matching", "pattern", "regex", "regexp", "followed by", "digit", "digits", "letter", "letters"]
    return rule_list(("agent_semantics", "case_bucketing", "pattern_terms"), defaults)


@lru_cache(maxsize=1)
def case_bucket_aggregate_request_terms() -> list[str]:
    defaults = ["how many", "distribution"]
    return rule_list(("agent_semantics", "case_bucketing", "aggregate_request_terms"), defaults)


@lru_cache(maxsize=1)
def case_bucket_aggregate_request_regex_patterns() -> list[str]:
    defaults = [
        r"\b(?:count|counts|number\s+of|total\s+(?:number\s+of)?|sum|average|avg|mean|min(?:imum)?|max(?:imum)?|distinct)\b",
    ]
    return rule_list(("agent_semantics", "case_bucketing", "aggregate_request_regex"), defaults)


@lru_cache(maxsize=1)
def order_term_token_stopwords() -> Set[str]:
    defaults = [
        "the", "a", "an", "by", "from", "to", "then", "and", "with",
        "first", "final", "result", "results", "output", "rows", "row",
        "sort", "sorted", "ordered", "arrange", "arranged",
        "ascending", "descending", "asc", "desc", "highest", "lowest",
        "largest", "smallest", "latest", "earliest", "newest", "oldest",
    ]
    vals = rule_list(("agent_semantics", "order_term_stopwords"), defaults)
    return {normalized_text(v) for v in vals if normalized_text(v)}


@lru_cache(maxsize=1)
def existence_anti_phrases() -> list[str]:
    defaults = [
        "with no", "without any", "who have no", "who has no",
        "that have no", "that has no", "never", "have never",
        "has never", "without", "tanpa",
    ]
    return rule_list(("existence_intent", "anti_phrases"), defaults)


@lru_cache(maxsize=1)
def existence_positive_phrases() -> list[str]:
    defaults = [
        "who have", "who has", "that have", "that has",
        "with at least one", "having at least one",
        "have at least one", "has at least one",
        "with any", "have any", "has any",
    ]
    return rule_list(("existence_intent", "positive_phrases"), defaults)


@lru_cache(maxsize=1)
def set_operation_explicit_union_terms() -> list[str]:
    defaults = ["union", "union all"]
    return rule_list(("set_operation_intent", "explicit_union_terms"), defaults)


@lru_cache(maxsize=1)
def set_operation_union_all_terms() -> list[str]:
    defaults = ["union all"]
    return rule_list(("set_operation_intent", "union_all_terms"), defaults)


@lru_cache(maxsize=1)
def set_operation_forbid_union_phrases() -> list[str]:
    defaults = ["do not use union", "don't use union", "dont use union"]
    return rule_list(("set_operation_intent", "forbid_union_phrases"), defaults)


@lru_cache(maxsize=1)
def set_operation_action_terms() -> list[str]:
    defaults = [
        "combine", "combined", "combining", "merge", "merged", "merging",
        "append", "appended", "appending", "stack", "stacked", "stacking",
        "concatenate", "concatenated", "concatenating",
        "put", "putting",
    ]
    return rule_list(("set_operation_intent", "action_terms"), defaults)


@lru_cache(maxsize=1)
def set_operation_combined_result_phrases() -> list[str]:
    defaults = [
        "in one result",
        "into one result",
        "in one result set",
        "into one result set",
        "as a single result set",
        "in a single result set",
        "into a single result set",
        "one combined result",
        "one combined list",
    ]
    return rule_list(("set_operation_intent", "combined_result_phrases"), defaults)


@lru_cache(maxsize=1)
def set_operation_result_noun_terms() -> list[str]:
    defaults = ["result set", "result sets", "query", "queries", "list", "lists"]
    return rule_list(("set_operation_intent", "result_noun_terms"), defaults)


@lru_cache(maxsize=1)
def set_operation_keywords() -> list[str]:
    defaults = ["union", "union all", "combine results", "gabungkan hasil"]
    return rule_list(("set_operation_intent", "set_op_keywords"), defaults)


@lru_cache(maxsize=1)
def set_operation_distinct_union_cues() -> list[str]:
    defaults = [
        "remove duplicate rows",
        "remove duplicates",
        "remove duplicate",
        "without duplicates",
        "no duplicate",
        "distinct",
        "deduplicate",
        "deduplicated",
        "deduped",
        "dedup",
        "de-duplicate",
        "de-duplicated",
        "de-duped",
        "de-dup",
        "de duplicate",
        "de duplicated",
        "de duped",
        "de dup",
    ]
    return rule_list(("set_operation_intent", "distinct_union_cues"), defaults)


@lru_cache(maxsize=1)
def set_operation_duplicate_object_terms() -> list[str]:
    defaults = ["duplicate", "duplicates", "duplicate rows", "duplicated rows"]
    return rule_list(("set_operation_intent", "duplicate_object_terms"), defaults)


@lru_cache(maxsize=1)
def set_operation_duplicate_preserve_verbs() -> list[str]:
    defaults = ["keep", "preserve", "allow", "include", "including"]
    return rule_list(("set_operation_intent", "duplicate_preserve_verbs"), defaults)


@lru_cache(maxsize=1)
def set_operation_duplicate_passive_verbs() -> list[str]:
    defaults = ["allowed", "kept", "preserved", "included"]
    return rule_list(("set_operation_intent", "duplicate_passive_verbs"), defaults)


@lru_cache(maxsize=1)
def set_operation_duplicate_no_remove_phrases() -> list[str]:
    defaults = ["do not remove", "don't remove", "dont remove", "without removing"]
    return rule_list(("set_operation_intent", "duplicate_no_remove_phrases"), defaults)


@lru_cache(maxsize=1)
def set_operation_branch_label_hints() -> list[str]:
    defaults = ["label", "year label", "category label"]
    return rule_list(("set_operation_intent", "branch_label_hints"), defaults)


@lru_cache(maxsize=1)
def set_operation_branch_label_connector_terms() -> list[str]:
    defaults = ["or"]
    return rule_list(("set_operation_intent", "branch_label_connector_terms"), defaults)


@lru_cache(maxsize=1)
def missing_value_phrases() -> list[str]:
    defaults = [
        "missing", "is missing", "missing information", "empty", "blank",
        "not provided", "unknown", "null or empty", "null/empty",
        "kosong", "tidak ada", "belum diisi",
    ]
    return rule_list(("missing_value_intent", "missing_phrases"), defaults)


@lru_cache(maxsize=1)
def present_value_phrases() -> list[str]:
    defaults = [
        "not empty", "non-empty", "filled", "provided",
        "has value", "have value", "with value",
        "has text", "have text", "with text",
        "has comments", "have comments", "with comments",
        "has comment", "have comment", "with comment",
        "has description", "have description", "with description",
        "has note", "have note", "with note",
        "has notes", "have notes", "with notes",
    ]
    return rule_list(("missing_value_intent", "present_phrases"), defaults)


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
        "ordering_desc_terms_count": len(ordering_direction_desc_terms()),
        "ordering_asc_terms_count": len(ordering_direction_asc_terms()),
        "semantic_phrase_stopwords_count": len(semantic_phrase_token_stopwords()),
        "order_term_stopwords_count": len(order_term_token_stopwords()),
    }
