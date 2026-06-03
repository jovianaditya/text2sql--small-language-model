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
    "medium": "evaluation_medium_p1.log",
    "hard": "evaluation_hard_p1.log",
}

QUESTION_PREFIX_BY_DIFFICULTY = {
    "easy": "E",
    "medium": "M",
    "hard": "H",
}

CASE_HEADER_RE = re.compile(r"^\[(?P<qid>[A-Z]\d+)\]\s.*$")
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
TABLE_ORDER_GUIDANCE_RE = re.compile(
    r"^\[PROMPT DEBUG\] Adding hint: (?P<guidance>TABLE ORDER GUIDANCE: .*?)\.\s+Use\b"
)
RUN_END_RE = re.compile(r"^RUN END:\s*(?P<value>.+?)\s*$")
NUMBER_RE = re.compile(r"\d+")
COMPACT_DIFFICULTY_FLAG_RE = re.compile(r"^(--(?P<difficulty>easy|medium|hard))(?P<suffix>\d.*)$")
JAKARTA_TZ = timezone(timedelta(hours=7), name="UTC+07:00")

_EMBEDDED_GOLD_BLOB = (
    "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"
)


def _numeric_qid_key(qid: str) -> tuple[str, int]:
    match = NUMBER_RE.search(qid)
    return qid[:1], int(match.group()) if match else 0


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
    if re.match(r"(?is)^\s*COUNT\s*\(\s*\*\s*\)\s*$", text):
        return "COUNT(*)"

    match = COUNT_METRIC_RE.match(text)
    if not match:
        return text

    inner = re.sub(r"\s+", "", str(match.group("inner") or "")).replace("`", "")
    if "." not in inner:
        return text

    table_name, col_name = [part.strip().lower() for part in inner.split(".", 1)]
    if not table_name or not col_name:
        return text

    pk_map = _primary_key_columns_by_table()
    if pk_map.get(table_name) == col_name:
        return "COUNT(*)"
    return text


def _normalize_order_insensitive_summary_lines(summary: str) -> str:
    normalized_lines: list[str] = []
    for raw_line in _normalize_summary(summary).splitlines():
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


def _load_embedded_gold() -> Dict[str, Dict[str, Dict[str, str]]]:
    raw = gzip.decompress(base64.b64decode(_EMBEDDED_GOLD_BLOB)).decode("utf-8")
    return json.loads(raw)


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
    if difficulty == "easy":
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
            if summary and difficulty == "easy":
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
    if difficulty == "easy":
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
    if difficulty not in {"easy", "medium"}:
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
    is_easy_or_medium = difficulty in {"easy", "medium"}
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
    elif difficulty in {"easy", "medium"}:
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
            "--medium",
            nargs="?",
            const="",
            help="Only use medium questions. Optionally specify ID/range: --medium 2, --medium 4-7, --medium 1,3,5",
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
    for difficulty_name in ("easy", "medium", "hard"):
        arg_value = getattr(args, difficulty_name)
        if arg_value is not None:
            selections.append((difficulty_name, parse_difficulty_filter(arg_value)))

    if args.difficulty:
        selections.append((args.difficulty, None))

    if not selections:
        raise ValueError("Choose one difficulty selector: --easy, --medium, --hard, or --difficulty.")

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
