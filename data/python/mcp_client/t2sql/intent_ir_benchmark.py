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
    "easyshadow": "evaluation_shadow_easy_p1.log",
    "medium": "evaluation_medium_p1.log",
    "mediumshadow": "evaluation_shadow_medium_p1.log",
    "hard": "evaluation_hard_p1.log",
}

QUESTION_PREFIX_BY_DIFFICULTY = {
    "easy": "E",
    "easyshadow": "SE",
    "medium": "M",
    "mediumshadow": "SM",
    "hard": "H",
}
EASY_LIKE_DIFFICULTIES = {"easy", "easyshadow"}
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
    r"^(--(?P<difficulty>easyshadow|easy|mediumshadow|medium|hard))(?P<suffix>\d.*)$"
)
JAKARTA_TZ = timezone(timedelta(hours=7), name="UTC+07:00")

_EMBEDDED_GOLD_BLOB = (
    "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"
)

_EMBEDDED_EASYSHADOW_GOLD_BLOB = (
    "H4sIABM9IWoC/82X3W/bOAzA/xUiL12BnJe0Tb+Aewj6AQxYs93SPRy2IVBtttbVllxJXs4Y+r8fZdmJ5Y9de7gA60NqU6JE8ieS"
    "8o8RMl3omEVyPTr/MVpeTe2/pxy14VKsEi5wdD76QvJv8MbO3YdlLNcQ5trIFJWGdSw1QiiFYaGBe660AcFSBBQRjXITg4kREjQG"
    "FbAAPqHJlSiF5SpBEIzGI52nKVOF3exivrh8dzm/vYJ3i9urxS0sP9/czD/9SRbQNvc8QhHi75PgbLZ//lUA/AbL/O4vpN1RGG6K"
    "8611bviPHFUBpsjwHBIpH/PMyS95ikKTn7qhEtRPC3Ji3JQ7F6+th92xXBhVNEVZLAW6fa55Qr77m7QWg4Q/4krn9/f8b2BO7YOK"
    "UHHx8FO9+fJiPGC8HRs9jwnqwQDUgw3Uay6iBlSZG01xLiF9Xs6B0egFEyxiNW+FETdkdUq/XMMdmjWigNmE/mj6Lws1T+9QjV+C"
    "u/TwvXWwh3QL7JtePT8qwaQM43TiXvaBIuIpupVBSLPiAr7sUeD3xrDn4r73bf8np6Kx5+VVTf1wgPqhn8qYZoksEMtUBoWZVAaM"
    "BAYp7fxgk5bMXkv1CGSWpBMaIkxBKjj2kpnW+c/UTwepb6x7OfWNSlA/1dS3Iwmrk3grc65dyMiT3m/zfSt0UdK3sn0OeqZQeqws"
    "U5EnyUqqFU0xRUW/b3Mo4U8t+uMe6r0qZRUY8ntbB44GTsSRXwfcwnVpz6Q2LKEKT/vQE/nkqvqJR786F2EZvZDQ0S8d6B3Ugcq6"
    "l5+HSsHjW8ucc77MWb95q0t7LSDSihvZqQHdJV1RzxTaon7SBtkzfwNqNgBq5qeutItpyBIWImETcCNFxAofjJ0DosoA9xYxQ/7a"
    "tKatTa7HoHcBqrTuFZzK+UH5b9G0t5JdllZXEmd4G0FrNlD1faSArPCpCk0HQltjg+B4AMFxK1ccAhMzAyxRyKICYvYdqXzqmGcZ"
    "cbHRLoPNFNrqbhPcmW97aohJgpFF9msjqLzpgdCl1K6K3vRtg9t4T2XO1cPuXv3VcwBjU2/bB08GSJ74yZSxggJB1S1lrggfTCZH"
    "VANRuXsQS20lsJcdQp2Qoqn6eudGm+6gD9bWvZxlrRE4w8dbQfsytB2IMXzsSKuHPrCtPeCBAj/z7jd9q0CBTNmctBHusGyvaUH2"
    "W9NgfDrA+NTP1kzJKA9N3drsd4p2Tc31tOnZzMNZza/aWv1m1cY7yNbaulcwrjSC6uGiaedG6m4vbekyZElTfJcXHxV1pBbjvsW8x"
    "kYx6yDs09lU1rMBVmet5lbdQmzljLimJleUTDRSOKmYRpFCrcHqA9OweDu3ySrAfZCkXGuyxdHcQT7+r7eQypf35MrBv9xDhm4R/"
    "mXQBXo6GfqYn/Q0MYjQMJ7oRsl7ylnprptArazxpXfoPmdmkzHVyjDJNf9u7xS7CLXdvDLula2s0uppaPWIlzjeSO38B+d7R4/Cf"
    "cXCuLfXDazRCF5Qh29TKfuXh8TYT0aa1dvzhjZyRXMoBu6IPD//A7SYttX/EQAA"
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
    payload = _load_embedded_gold_blob(_EMBEDDED_GOLD_BLOB)
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
    for difficulty_name in ("easy", "easyshadow", "medium", "mediumshadow", "hard"):
        arg_value = getattr(args, difficulty_name)
        if arg_value is not None:
            selections.append((difficulty_name, parse_difficulty_filter(arg_value)))

    if args.difficulty:
        selections.append((args.difficulty, None))

    if not selections:
        raise ValueError(
            "Choose one difficulty selector: --easy, --easyshadow, --medium, --mediumshadow, --hard, or --difficulty."
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
