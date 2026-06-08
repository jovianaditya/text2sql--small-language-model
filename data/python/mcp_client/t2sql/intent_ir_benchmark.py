#!/usr/bin/env python3
"""
Extract and compare Intent IR hint summaries from evaluation logs.

Examples:
  python -m t2sql.intent_ir_benchmark extract --difficulty easy --log evaluation_easy_p1.log
  python -m t2sql.intent_ir_benchmark compare --difficulty medium --log evaluation_medium_p1.log
  python -m t2sql.intent_ir_benchmark gold --difficulty easy
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from itertools import zip_longest
from pathlib import Path
from typing import Dict, Iterable

from .schema_analyzer import get_all_tables, get_id_column


DEFAULT_LOG_BY_DIFFICULTY = {
    "easy": "evaluation_easy_p1.log",
    "easytest": "evaluation_test_easy_p1.log",
    "easyshadow": "evaluation_shadow_easy_p1.log",
    "medium": "evaluation_medium_p1.log",
    "mediumshadow": "evaluation_shadow_medium_p1.log",
    "hard": "evaluation_hard_p1.log",
}

QUESTION_PREFIX_BY_DIFFICULTY = {
    "easy": "E",
    "easytest": "E",
    "easyshadow": "SE",
    "medium": "M",
    "mediumshadow": "SM",
    "hard": "H",
}
EASY_LIKE_DIFFICULTIES = {"easy", "easytest", "easyshadow"}
NO_GUIDANCE_DIFFICULTIES = EASY_LIKE_DIFFICULTIES.union({"medium", "mediumshadow"})

CASE_HEADER_RE = re.compile(r"^\[(?P<qid>[A-Z]+?\d+)\]\s.*$")
SUMMARY_START_RE = re.compile(r"^\[CANDIDATE INTENT SUMMARY\]")
SUMMARY_STOP_RE = re.compile(r"^Use this intent only as ")
SUMMARY_CONFIDENCE_RE = re.compile(
    r"(\[CANDIDATE INTENT SUMMARY\]\s*\(confidence=)(\d+(?:\.\d+)?)(\):)",
    flags=re.I,
)
SUMMARY_OUTPUT_SHAPE_RE = re.compile(r"^\s*-\s*Output shape:\s*.*$", flags=re.I)
SUMMARY_EXECUTION_SHAPE_RE = re.compile(r"^\s*-\s*Execution shape:\s*.*$", flags=re.I)
SUMMARY_ORDER_INSENSITIVE_LIST_RE = re.compile(
    r"^(?P<prefix>\s*-\s*(?P<label>Dimensions|Metrics):\s*)(?P<body>.+?)\s*$"
)
COUNT_METRIC_RE = re.compile(
    r"""(?is)^\s*COUNT\s*\(\s*(?P<inner>`?[A-Za-z_][A-Za-z0-9_]*`?(?:\s*\.\s*`?[A-Za-z_][A-Za-z0-9_]*`?)?)\s*\)\s*$"""
)
COUNT_EXPR_RE = re.compile(
    r"""(?is)COUNT\s*\(\s*(?P<inner>`?[A-Za-z_][A-Za-z0-9_]*`?(?:\s*\.\s*`?[A-Za-z_][A-Za-z0-9_]*`?)?)\s*\)"""
)
QUALIFIED_IDENTIFIER_RE = re.compile(
    r"""(?P<table>`?[A-Za-z_][A-Za-z0-9_]*`?)\s*\.\s*(?P<column>`?[A-Za-z_][A-Za-z0-9_]*`?)"""
)
TABLE_ORDER_GUIDANCE_RE = re.compile(
    r"^\[PROMPT DEBUG\] Adding hint: (?P<guidance>TABLE ORDER GUIDANCE: .*?)\.\s+Use\b"
)
RUN_END_RE = re.compile(r"^RUN END:\s*(?P<value>.+?)\s*$")
NUMBER_RE = re.compile(r"\d+")
COMPACT_DIFFICULTY_FLAG_RE = re.compile(
    r"^(--(?P<difficulty>easyshadow|easytest|easy|mediumshadow|medium|hard))(?P<suffix>\d.*)$"
)
JAKARTA_TZ = timezone(timedelta(hours=7), name="UTC+07:00")

_EMBEDDED_EASYSHADOW_GOLD_BLOB = (
    "H4sIADRLJmoC/81XbU/jRhD+K6NIpyRS6jpAeFOplBIjoULokaATaqtqsZeLi73r864PWSf+e2dfHL9TUBvp+BDs2Z3ZmXlm5xl/"
    "G1AicrEhAX8enMK3wcqb6v9fMipkyNlfUcgoSga/48qfMFLbx7Da8GfwMyF5TFMBzxsuKPicSeJLeAxTIYGRmAJlAa6GcgNyQyGi"
    "UtIUiAO3VGYp00JtxXGcwQQGIotjkub6uPP5cnG5mK89uFyuveUaVnfX1/Pbe/QBD3oMA8p8euY6J7Px6R8M4AdYZQ9/UzyfMhnK"
    "/LT0zyx/zGiag8wTegoR509ZYuSLMKZMYKiiouIUT0sMY1KVmyAvVIzttYzJNK+Kkg1n1JxzEUYYff2QhjG4uvzVg+EHMjQqN2lA"
    "05B9flVnvjqf9Diu1gYvmFjEzu2H1d3iehGyALg6FQIqSRgpbGlKNVJfMqIzazbQAEIBD1Q+U8pg3wWCqjN3AiHzo0yEXzE15D/g"
    "etyLqz7euvd2aKtajn5ZZvEDTSf1lSTlQebLcx7QxkoR/o2JvqUX+tQj/qYB9ug1G/CLt/7keUtMn+MCJgYT6Lhj/TTqNg8/wdRV"
    "m5oV8uo5C08VSV8KKmWy11sle/UiKS8/z6RA1HSJ3K3mug7OCSMBKfoCehBKiMIYfys1M3PxD7d/x5ffVsgb2oKO8UqF2NERmjXR"
    "qbctBp0XWw9T17zYmmhZhuXNGnMEoyGmfjiBocn8cDx+pYNUzlSFYaHf74V+v973aZxEPKdU931IacJTCZIDgRgP/6w6PJbAM0+f"
    "sBcAf3zE0oUpVh8c1jo/2tlJf9j693botypO8VRAX65EpOj4pcwEZ5pFKX0syaEUmjyJNW8WQ8cWuFxpYJd3V1cW+a4zNfDTCRy2"
    "0e7crpmiL9RKEzjorYSDBlNo2wX/J1xIEuEYgEfhE0ZiqP+ohrqtB1/nzEfA8BdreSdNwPr3DpIwCjVcC5kJry4z/m/fCv4vBIhw"
    "GkreagBtk5b5jz60mL9jb4nUrBepWf3O6m4vIImIj3TAGVxzFpC8jozmfVZlRQiIVDyOWOPpMhMTELtBSvv3Tjbv4nErW2i/rcS4"
    "3sBgMb9fzq+9UVNrDGcwNNkZdnJsZW8FicNeJA47piuBCScSSJRSEuSwIV8ptk+xCZME4VFJ1zknOHkxLlUbNTEoYvVpFNFAIfe9"
    "I1GUTPFqwqtBs01l59hUmChZbhs/MlxlTKoZr3XPHgir2ysceNSL4lH9PiUkxxRgh4sJXkjEZ891DyqzMolVL1DTDsIcoaK0tN76"
    "9Il3woGFf2/HsdBwjOuTUtCchsqFDfWfWlL70AVq4wz4+awYdiyY9978dtRlSd1KleIWnk2TZtLtMlHF+bgX5+P6bbWfAwXFqY9a"
    "YcjNcNv0ZFaD1O639Fa8KbXJTm5r4d87cLYa9S+dptRML03pyidRVfyQ5b+pD5MGzl3GLMFhvtoU17m/bK0nvWCdNEjOjiOqdQah"
    "QLLLNSiCYjaxmwZBSoUAZQAINpUf5+rGMjCfJXEoBLpj4NzNh+v/OY7YaK4wmL1/GUj6Ror6aDh4eXn5BxleNo4PEgAA"
)

_EMBEDDED_EASY_GOLD_BLOB = (
    "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"
    "4Pcu+DXvtQeKATf/nyX9A0t8t6SQAA"
)

_EMBEDDED_MEDIUM_GOLD_BLOB = (
    "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"
    "9RVl237ba0p8PembHJz7y+fv0fLYgz5Aw7AAA="
)

_EMBEDDED_EASYTEST_GOLD_BLOB = (
    "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"
)

_EMBEDDED_MEDIUMSHADOW_GOLD_BLOB = (
    "H4sIAONBIWoC/8VVbW/aMBD+K6d8gimNaCvUDq2VGNCuUqFboe2mbZpMcoDXxGaOU5RV/Pdd4iSEllC1WzU+gLkX++55/JzvrQA9"
    "HgXhjHlyYbXurWF/N/n5FWGouRQ/fC7Qallfyf4daia6DqdKRnOYszhAoUMYx/kaYmTKgUvUkRKgZ7jmsIt/royEtkFLzXxggeM4"
    "lm2FURAwFSfHddqD7lm3PerB2WDUG4xgeNXvty+/UA2uFBPuoXDxqOG8bdZb3wTADgyj8U90NdDuXMetojjj/RShikHHc2wBm04V"
    "TplG47qI9DzSQBAkzmnSGXqQ1QILrmcgBYKSC5ijSvsADxW/o6iJkkFxkpMtusXWXU7/Q4IxbG2J6qNW3KWQzsXVYFR7U7eTbmtF"
    "AgsSsMjab39+ZDVbGD5OuK9R0Uab0uEYmo3kYy1tYnmvguW9EsvDFAP+GwGDuS9jxJRqOZlwF4lCD9eYLtntIiNnekbwvZDlXafR"
    "qGK5KOxJmstcFFmOqblDJVdykaC+SshXgygYo8qoWrkVzqXS4UjWzc3ptIc9uPnQGxjU9ytQ3y+hfsKFZ1BTnABPt2EafGShhj1w"
    "o1DLgGiGxUyGCOwOFZsSzoryKYwH9M2JKPQJ8sNG4xW0VdTwLNSLLMd0F1dC3r4+rZWi087Ok8YyOjb7Nkoh3xSOjwg86nzL5vAu"
    "wStXSLOCq2aJqxsihwtA5s6K9oAMufhsIjKg6UI0gYqE4GKazTw5KWahEShIRXp6KVeHfz8HnxpW9sqat2pEUHbM0L19ZDUNPmB7"
    "04wywB9UAH9QAv595N6iLomBCy1pCil8oIQxE14ItYGEjjFPpILB1fk5KVeTjykvMf1XkTzzBcofH5+RxrfKbHW3K6RmAD+sAPzw"
    "8YuvpBe55sVPRv6EqzAZTZrEllzpxHaHwiOMBQvWn4fMzgXXnPmvcdOz4v413mtNrr/82YlOtrhOW9wgpy1xFeOvSOkPLz9uHm1z"
    "GeqdvDU6KCl4Jx2tqRDCmfQ9a7lc/gFWR2no5QkAAA=="
)

def _numeric_qid_key(qid: str) -> tuple[str, int]:
    match = NUMBER_RE.search(qid)
    prefix = re.match(r"^[A-Z]+", qid or "")
    return (prefix.group(0) if prefix else qid[:1]), int(match.group()) if match else 0


def _normalize_summary(summary: str) -> str:
    return "\n".join(line.rstrip() for line in summary.strip().splitlines())


@lru_cache(maxsize=1)
def _primary_key_columns_by_table() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    try:
        for table in get_all_tables() or []:
            table_l = str(table or "").strip().lower()
            if not table_l:
                continue
            pk_col = str(get_id_column(table_l) or "").strip().lower()
            if pk_col:
                mapping[table_l] = pk_col
    except Exception:
        return {}
    return mapping


def _normalize_metric_item(item: str) -> str:
    text = str(item or "").strip()
    if not text:
        return text
    return _normalize_identifier_case(_normalize_count_pk_equivalence(text))


def _normalize_identifier_case(text: str) -> str:
    raw = str(text or "")
    if not raw:
        return raw

    def repl(match: re.Match[str]) -> str:
        table = str(match.group("table") or "").replace("`", "").strip().lower()
        column = str(match.group("column") or "").replace("`", "").strip().lower()
        if not table or not column:
            return match.group(0)
        return f"{table}.{column}"

    return QUALIFIED_IDENTIFIER_RE.sub(repl, raw)


def _normalize_count_pk_equivalence(text: str) -> str:
    raw = str(text or "")
    if not raw:
        return raw

    def repl(match: re.Match[str]) -> str:
        inner = re.sub(r"\s+", "", str(match.group("inner") or "")).replace("`", "")
        if "." not in inner:
            return match.group(0)

        table_name, col_name = [part.strip().lower() for part in inner.split(".", 1)]
        if not table_name or not col_name:
            return match.group(0)

        pk_map = _primary_key_columns_by_table()
        if pk_map.get(table_name) == col_name:
            return "COUNT(*)"
        return match.group(0)

    normalized = COUNT_EXPR_RE.sub(repl, raw)
    if re.match(r"(?is)^\s*COUNT\s*\(\s*\*\s*\)\s*$", normalized.strip()):
        return "COUNT(*)"
    return normalized


def _normalize_order_insensitive_summary_lines(summary: str) -> str:
    normalized_lines: list[str] = []
    for raw_line in _normalize_summary(summary).splitlines():
        raw_line = _normalize_identifier_case(_normalize_count_pk_equivalence(raw_line))
        match = SUMMARY_ORDER_INSENSITIVE_LIST_RE.match(raw_line)
        if not match:
            normalized_lines.append(raw_line)
            continue
        items = [
            (_normalize_metric_item(item) if str(match.group("label") or "").strip() == "Metrics" else item).strip()
            for item in str(match.group("body") or "").split(",")
            if item.strip()
        ]
        items.sort(key=lambda item: item.lower())
        normalized_lines.append(f"{match.group('prefix')}{', '.join(items)}")
    return "\n".join(normalized_lines)


def _summary_compare_key(summary: str) -> str:
    normalized = _normalize_order_insensitive_summary_lines(summary)
    return SUMMARY_CONFIDENCE_RE.sub(r"\1<ignored>\3", normalized)


def _normalize_order_insensitive_line(line: str) -> str:
    normalized = _normalize_summary(str(line or ""))
    return _normalize_order_insensitive_summary_lines(normalized)


def parse_difficulty_filter(filter_spec: str | None) -> set[int] | None:
    """Mirror testing_evaluation.py optional ID/range filter parsing."""
    if not filter_spec:
        return None

    filter_spec = filter_spec.strip()

    if "-" in filter_spec:
        try:
            start, end = filter_spec.split("-", 1)
            return set(range(int(start), int(end) + 1))
        except ValueError:
            return None

    if "," in filter_spec:
        try:
            return set(int(x.strip()) for x in filter_spec.split(","))
        except ValueError:
            return None

    try:
        return {int(filter_spec)}
    except ValueError:
        return None


def _load_embedded_gold_blob(blob: str) -> Dict[str, Dict[str, Dict[str, str]]]:
    raw = gzip.decompress(base64.b64decode(blob)).decode("utf-8")
    return json.loads(raw)


def _load_embedded_gold() -> Dict[str, Dict[str, Dict[str, str]]]:
    payload: Dict[str, Dict[str, Dict[str, str]]] = {}
    payload.update(_load_embedded_gold_blob(_EMBEDDED_EASY_GOLD_BLOB))
    payload.update(_load_embedded_gold_blob(_EMBEDDED_MEDIUM_GOLD_BLOB))
    payload.update(_load_embedded_gold_blob(_EMBEDDED_EASYTEST_GOLD_BLOB))
    payload.update(_load_embedded_gold_blob(_EMBEDDED_EASYSHADOW_GOLD_BLOB))
    payload.update(_load_embedded_gold_blob(_EMBEDDED_MEDIUMSHADOW_GOLD_BLOB))
    return payload


def _strip_easy_ignored_shape_lines(summary: str) -> str:
    kept_lines = [
        line
        for line in _normalize_summary(summary).splitlines()
        if not SUMMARY_OUTPUT_SHAPE_RE.match(line)
        and not SUMMARY_EXECUTION_SHAPE_RE.match(line)
    ]
    return "\n".join(kept_lines).strip()


def _normalize_summary_for_difficulty(summary: str, difficulty: str) -> str:
    normalized = _normalize_summary(summary)
    if difficulty in EASY_LIKE_DIFFICULTIES:
        return _strip_easy_ignored_shape_lines(normalized)
    return normalized


def _normalize_gold_benchmarks(
    gold_benchmarks: Dict[str, Dict[str, Dict[str, str]]],
) -> Dict[str, Dict[str, Dict[str, str]]]:
    normalized: Dict[str, Dict[str, Dict[str, str]]] = {}
    for difficulty, cases in gold_benchmarks.items():
        normalized_cases: Dict[str, Dict[str, str]] = {}
        for qid, payload in cases.items():
            next_payload = dict(payload)
            summary = str(next_payload.get("summary") or "")
            if summary and difficulty in EASY_LIKE_DIFFICULTIES:
                next_payload["summary"] = _strip_easy_ignored_shape_lines(summary)
            normalized_cases[qid] = next_payload
        normalized[difficulty] = normalized_cases
    return normalized


HARD_GOLD_BENCHMARK = {
    "H2": {
        "question_line": "[H2] (hard) Only include customers who have placed orders, and for each such customer return...",
        "summary": "[CANDIDATE INTENT SUMMARY] (confidence=0.95):\n  - Query type: aggregate\n  - Dimensions: customers.customerName, customers.customerNumber\n  - Metrics: COUNT(orders.orderNumber)",
        "table_order_guidance": "TABLE ORDER GUIDANCE: Prefer starting with FROM customers, then INNER JOIN orders",
    },
    "H3": {
        "question_line": "[H3] (hard) List only products that have been ordered at least once, and for each such produ...",
        "summary": "[CANDIDATE INTENT SUMMARY] (confidence=0.95):\n  - Query type: aggregate\n  - Dimensions: products.productName, products.productCode\n  - Metrics: SUM(orderdetails.quantityOrdered)",
        "table_order_guidance": "TABLE ORDER GUIDANCE: Prefer starting with FROM products, then INNER JOIN orderdetails",
    },
    "H11": {
        "question_line": "[H11] (hard) Return customers who have made at least one payment but have never placed any or...",
        "summary": "[CANDIDATE INTENT SUMMARY] (confidence=0.95):\n  - Query type: aggregate\n  - Dimensions: customers.customerNumber, customers.customerName\n  - Metrics: SUM(payments.amount)",
        "table_order_guidance": "TABLE ORDER GUIDANCE: Prefer starting with FROM customers, then INNER JOIN payments, LEFT JOIN orders",
    },
    "H15": {
        "question_line": "[H15] (hard) Find products that were ordered in 2004 but never ordered in 2005. For each such...",
        "summary": "[CANDIDATE INTENT SUMMARY] (confidence=0.85):\n  - Query type: aggregate\n  - Dimensions: products.productCode, products.productName\n  - Metrics: SUM(orderdetails.quantityOrdered)",
        "table_order_guidance": "TABLE ORDER GUIDANCE: Prefer starting with FROM products, then INNER JOIN orderdetails, INNER JOIN orders",
    },
}


GOLD_BENCHMARKS = _normalize_gold_benchmarks(_load_embedded_gold())
GOLD_BENCHMARKS["hard"] = HARD_GOLD_BENCHMARK


def _resolve_log_path(difficulty: str, log_arg: str | None) -> Path:
    if log_arg:
        return Path(log_arg).expanduser().resolve()
    default_name = DEFAULT_LOG_BY_DIFFICULTY.get(difficulty)
    if not default_name:
        raise ValueError(f"No default log configured for difficulty '{difficulty}'.")
    return (Path.cwd() / default_name).resolve()


def parse_log(log_path: Path, difficulty: str) -> Dict[str, Dict[str, str]]:
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    expected_prefix = QUESTION_PREFIX_BY_DIFFICULTY[difficulty]
    current_qid: str | None = None
    current_question_line: str | None = None
    current_table_order_guidance: str | None = None
    capture: list[str] | None = None
    cases: Dict[str, Dict[str, str]] = {}

    for raw_line in log_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        header_match = CASE_HEADER_RE.match(line)
        if header_match:
            current_qid = header_match.group("qid")
            current_question_line = line
            current_table_order_guidance = None

        guidance_match = TABLE_ORDER_GUIDANCE_RE.match(line)
        if guidance_match:
            current_table_order_guidance = guidance_match.group("guidance").rstrip(".")

        if SUMMARY_START_RE.match(line):
            capture = [line]
            continue

        if capture is None:
            continue

        if SUMMARY_STOP_RE.match(line):
            if current_qid and current_qid.startswith(expected_prefix):
                payload = {
                    "question_line": current_question_line or current_qid,
                    "summary": _normalize_summary("\n".join(capture)),
                }
                if current_table_order_guidance:
                    payload["table_order_guidance"] = current_table_order_guidance
                cases[current_qid] = payload
            capture = None
            continue

        capture.append(line)

    return dict(sorted(cases.items(), key=lambda item: _numeric_qid_key(item[0])))


def filter_cases_by_ids(
    cases: Dict[str, Dict[str, str]],
    difficulty: str,
    selected_ids: set[int] | None,
) -> Dict[str, Dict[str, str]]:
    if selected_ids is None:
        return cases

    prefix = QUESTION_PREFIX_BY_DIFFICULTY[difficulty]
    wanted_qids = {f"{prefix}{case_id}" for case_id in selected_ids}
    return {
        qid: payload
        for qid, payload in _iter_case_rows(cases)
        if qid in wanted_qids
    }


def _iter_case_rows(cases: Dict[str, Dict[str, str]]) -> Iterable[tuple[str, Dict[str, str]]]:
    return sorted(cases.items(), key=lambda item: _numeric_qid_key(item[0]))


def print_cases_text(cases: Dict[str, Dict[str, str]]) -> None:
    for _, payload in _iter_case_rows(cases):
        print(payload["question_line"])
        if payload.get("table_order_guidance"):
            print(payload["table_order_guidance"])
        print(payload["summary"])
        print()


def print_cases_json(difficulty: str, source: str, cases: Dict[str, Dict[str, str]]) -> None:
    payload = {
        "difficulty": difficulty,
        "source": source,
        "count": len(cases),
        "cases": cases,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=True))


def _print_indented_block(title: str, content: str) -> None:
    print(title)
    for line in content.splitlines():
        print(f"  {line}")


def _print_changed_lines(expected_text: str, actual_text: str, difficulty: str = "") -> None:
    print("Changed lines:")
    if difficulty in EASY_LIKE_DIFFICULTIES:
        expected_text = _normalize_summary_for_difficulty(expected_text, difficulty)
        actual_text = _normalize_summary_for_difficulty(actual_text, difficulty)
    changed_count = 0
    for expected_line, actual_line in zip_longest(
        expected_text.splitlines(),
        actual_text.splitlines(),
        fillvalue="<missing>",
    ):
        if expected_line == actual_line:
            continue
        if _normalize_order_insensitive_line(expected_line) == _normalize_order_insensitive_line(actual_line):
            continue
        changed_count += 1
        print(f"  - Gold     : {expected_line}")
        print(f"  - Predicted: {actual_line}")

    if changed_count == 0:
        print("  - No line-level differences detected after normalization.")


def _print_checked_timestamp() -> None:
    checked_at = datetime.now(JAKARTA_TZ).strftime("%Y-%m-%d %H:%M:%S UTC+07:00")
    print(f"Checked at: {checked_at}")


def _extract_run_end(log_path: Path) -> str:
    run_end = ""
    try:
        for line in log_path.read_text(encoding="utf-8").splitlines():
            match = RUN_END_RE.match(line.rstrip())
            if match:
                run_end = match.group("value").strip()
    except FileNotFoundError:
        return ""
    return run_end


def _print_log_footer(run_end: str = "") -> None:
    print(f"Run end: {run_end}")
    _print_checked_timestamp()


def _mismatch_label(summary_changed: bool, guidance_changed: bool) -> str:
    if summary_changed and guidance_changed:
        return "Guidance + Summary mismatch"
    if guidance_changed:
        return "Guidance mismatch"
    return "Summary mismatch"


def _unexpected_guidance_cases(
    difficulty: str,
    actual_cases: Dict[str, Dict[str, str]],
) -> list[tuple[str, Dict[str, str]]]:
    if difficulty not in NO_GUIDANCE_DIFFICULTIES:
        return []

    return [
        (qid, payload)
        for qid, payload in _iter_case_rows(actual_cases)
        if str(payload.get("table_order_guidance") or "").strip()
    ]


def compare_cases(
    difficulty: str,
    actual_cases: Dict[str, Dict[str, str]],
    selected_ids: set[int] | None = None,
) -> tuple[list[str], list[str], list[dict[str, str]]]:
    gold_cases = filter_cases_by_ids(GOLD_BENCHMARKS[difficulty], difficulty, selected_ids)
    gold_ids = set(gold_cases)
    actual_ids = set(actual_cases)

    missing = sorted(gold_ids - actual_ids, key=_numeric_qid_key)
    unexpected = sorted(actual_ids - gold_ids, key=_numeric_qid_key)

    mismatches: list[dict[str, str]] = []
    for qid in sorted(gold_ids & actual_ids, key=_numeric_qid_key):
        expected_summary = _normalize_summary_for_difficulty(gold_cases[qid]["summary"], difficulty)
        actual_summary = _normalize_summary_for_difficulty(actual_cases[qid]["summary"], difficulty)
        expected_summary_key = _summary_compare_key(expected_summary)
        actual_summary_key = _summary_compare_key(actual_summary)
        expected_guidance = str(gold_cases[qid].get("table_order_guidance") or "").strip()
        actual_guidance = str(actual_cases[qid].get("table_order_guidance") or "").strip()
        if expected_summary_key != actual_summary_key or expected_guidance != actual_guidance:
            mismatches.append(
                {
                    "qid": qid,
                    "expected_summary": expected_summary,
                    "actual_summary": actual_summary,
                    "expected_summary_key": expected_summary_key,
                    "actual_summary_key": actual_summary_key,
                    "expected_guidance": expected_guidance,
                    "actual_guidance": actual_guidance,
                }
            )

    return missing, unexpected, mismatches


def print_compare_report(
    difficulty: str,
    log_path: Path,
    actual_cases: Dict[str, Dict[str, str]],
    selected_ids: set[int] | None = None,
) -> int:
    is_easy_or_medium = difficulty in NO_GUIDANCE_DIFFICULTIES
    total_gold_cases = len(GOLD_BENCHMARKS[difficulty])
    gold_cases = filter_cases_by_ids(GOLD_BENCHMARKS[difficulty], difficulty, selected_ids)
    missing, unexpected, mismatches = compare_cases(difficulty, actual_cases, selected_ids)
    unexpected_guidance = _unexpected_guidance_cases(difficulty, actual_cases)
    guidance_mismatch_count = sum(
        1
        for mismatch in mismatches
        if mismatch["expected_guidance"] != mismatch["actual_guidance"]
    )
    summary_mismatch_count = sum(
        1
        for mismatch in mismatches
        if mismatch["expected_summary_key"] != mismatch["actual_summary_key"]
    )

    print(f"Difficulty: {difficulty}")
    print(f"Log: {log_path}")
    print(f"Gold benchmark cases: {total_gold_cases}")
    print(f"Extracted cases: {len(actual_cases)}")
    print(f"Mismatched cases: {len(mismatches)}")
    if not is_easy_or_medium:
        print(f"Guidance mismatches: {guidance_mismatch_count}")
    print(f"Summary IR-HINT mismatches: {summary_mismatch_count}")
    print()

    if mismatches:
        print("Case mismatches:")
        total_mismatches = len(mismatches)
        separator = "=" * 80
        inner_separator = "-" * 80
        for index, mismatch in enumerate(mismatches, start=1):
            qid = mismatch["qid"]
            expected_summary = mismatch["expected_summary"]
            actual_summary = mismatch["actual_summary"]
            expected_summary_key = mismatch["expected_summary_key"]
            actual_summary_key = mismatch["actual_summary_key"]
            expected_guidance = mismatch["expected_guidance"]
            actual_guidance = mismatch["actual_guidance"]
            guidance_changed = expected_guidance != actual_guidance
            summary_changed = expected_summary_key != actual_summary_key
            print(separator)
            print(f"[Mismatch {index}/{total_mismatches}] {qid}")
            print(f"Type: {_mismatch_label(summary_changed, guidance_changed)}")
            print(f"Question: {gold_cases[qid]['question_line']}")
            print(inner_separator)
            if expected_guidance or actual_guidance:
                _print_indented_block(
                    "Gold table order guidance:",
                    expected_guidance or "<missing>",
                )
                print()
                _print_indented_block(
                    "Predicted table order guidance:",
                    actual_guidance or "<missing>",
                )
                print()
                if guidance_changed:
                    _print_changed_lines(expected_guidance or "<missing>", actual_guidance or "<missing>", difficulty)
                    print()
            if summary_changed or difficulty != "hard":
                _print_indented_block("Gold summary:", expected_summary)
                print()
                _print_indented_block("Predicted summary:", actual_summary)
                print()
                if summary_changed:
                    _print_changed_lines(expected_summary, actual_summary, difficulty)
                else:
                    print("Changed lines:")
                    print("  - No summary differences.")
        print()

    if unexpected_guidance:
        print("Unexpected TABLE ORDER GUIDANCE logs:")
        separator = "=" * 80
        inner_separator = "-" * 80
        total_unexpected_guidance = len(unexpected_guidance)
        for index, (qid, payload) in enumerate(unexpected_guidance, start=1):
            print(separator)
            print(f"[Unexpected Guidance {index}/{total_unexpected_guidance}] {qid}")
            print(f"Question: {payload['question_line']}")
            print(inner_separator)
            _print_indented_block(
                "Logged table order guidance:",
                str(payload.get("table_order_guidance") or "<missing>"),
            )
        print()
    elif difficulty in NO_GUIDANCE_DIFFICULTIES:
        print("TABLE ORDER GUIDANCE check:")
        print("  - Not found in this log.")
        print()

    if not mismatches:
        print("Benchmark match: PASS")
        _print_log_footer(_extract_run_end(log_path))
        return 0

    print("Benchmark match: FAIL")
    _print_log_footer(_extract_run_end(log_path))
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract and compare Intent IR hint summaries from evaluation logs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command_name in ("extract", "compare", "gold"):
        subparser = subparsers.add_parser(command_name)
        subparser.add_argument(
            "--difficulty",
            choices=sorted(GOLD_BENCHMARKS),
            help="Backward-compatible difficulty selector. Prefer --easy / --medium.",
        )
        subparser.add_argument(
            "--easy",
            nargs="?",
            const="",
            help="Only use easy questions. Optionally specify ID/range: --easy 4, --easy 4-7, --easy 1,3,5",
        )
        subparser.add_argument(
            "--easytest",
            nargs="?",
            const="",
            help="Only use embedded easy test questions. Optionally specify ID/range: --easytest 4, --easytest 4-7, --easytest 1,3,5",
        )
        subparser.add_argument(
            "--easyshadow",
            nargs="?",
            const="",
            help="Only use embedded shadow easy questions. Optionally specify ID/range: --easyshadow 4, --easyshadow 4-7, --easyshadow 1,3,5",
        )
        subparser.add_argument(
            "--medium",
            nargs="?",
            const="",
            help="Only use medium questions. Optionally specify ID/range: --medium 2, --medium 4-7, --medium 1,3,5",
        )
        subparser.add_argument(
            "--mediumshadow",
            nargs="?",
            const="",
            help="Only use embedded shadow medium questions. Optionally specify ID/range: --mediumshadow 2, --mediumshadow 4-7, --mediumshadow 1,3,5",
        )
        subparser.add_argument(
            "--hard",
            nargs="?",
            const="",
            help="Only use hard questions. Optionally specify ID/range: --hard 2, --hard 2-5, --hard 2,11,15",
        )
        if command_name in {"extract", "compare"}:
            subparser.add_argument(
                "--log",
                help="Path to the evaluation log. Defaults to the P1 log for the selected difficulty.",
            )
        if command_name in {"extract", "gold"}:
            subparser.add_argument(
                "--json",
                action="store_true",
                help="Print JSON instead of plain text.",
            )

    return parser


def resolve_requested_difficulty(args: argparse.Namespace) -> tuple[str, set[int] | None]:
    selections = []
    for difficulty_name in ("easy", "easytest", "easyshadow", "medium", "mediumshadow", "hard"):
        arg_value = getattr(args, difficulty_name)
        if arg_value is not None:
            selections.append((difficulty_name, parse_difficulty_filter(arg_value)))

    if args.difficulty:
        selections.append((args.difficulty, None))

    if not selections:
        raise ValueError(
            "Choose one difficulty selector: --easy, --easytest, --easyshadow, --medium, --mediumshadow, --hard, or --difficulty."
        )

    unique_difficulties = {difficulty for difficulty, _ in selections}
    if len(unique_difficulties) != 1:
        raise ValueError("Use only one difficulty at a time.")

    difficulty, selected_ids = selections[0]
    if difficulty not in GOLD_BENCHMARKS:
        available = ", ".join(sorted(GOLD_BENCHMARKS))
        raise ValueError(
            f"No embedded gold benchmark for difficulty '{difficulty}'. Available: {available}."
        )

    return difficulty, selected_ids


def normalize_compact_difficulty_args(argv: list[str]) -> list[str]:
    """Support compact forms like --easy4 and --medium2-5."""
    normalized: list[str] = []
    for token in argv:
        match = COMPACT_DIFFICULTY_FLAG_RE.match(token)
        if match:
            normalized.append(f"--{match.group('difficulty')}")
            normalized.append(match.group("suffix"))
            continue
        normalized.append(token)
    return normalized


def main() -> int:
    parser = build_parser()
    args = parser.parse_args(normalize_compact_difficulty_args(sys.argv[1:]))
    try:
        difficulty, selected_ids = resolve_requested_difficulty(args)
    except ValueError as exc:
        parser.error(str(exc))

    if args.command == "gold":
        cases = filter_cases_by_ids(GOLD_BENCHMARKS[difficulty], difficulty, selected_ids)
        if args.json:
            print_cases_json(difficulty, "embedded-gold", cases)
        else:
            print_cases_text(cases)
            print(f"Embedded gold cases: {len(cases)}")
        _print_checked_timestamp()
        return 0

    log_path = _resolve_log_path(difficulty, getattr(args, "log", None))
    cases = filter_cases_by_ids(parse_log(log_path, difficulty), difficulty, selected_ids)

    if args.command == "extract":
        if args.json:
            print_cases_json(difficulty, str(log_path), cases)
        else:
            print_cases_text(cases)
            print(f"Extracted cases: {len(cases)}")
        _print_log_footer(_extract_run_end(log_path))
        return 0

    return print_compare_report(difficulty, log_path, cases, selected_ids)


if __name__ == "__main__":
    sys.exit(main())
