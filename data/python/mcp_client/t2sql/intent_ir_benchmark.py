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
COUNT_EXPR_RE = re.compile(
    r"""(?is)COUNT\s*\(\s*(?P<inner>`?[A-Za-z_][A-Za-z0-9_]*`?(?:\s*\.\s*`?[A-Za-z_][A-Za-z0-9_]*`?)?)\s*\)"""
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
    "Yz29TB7+77//B9MFBZuYrwAA"
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
    return _normalize_count_pk_equivalence(text)


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
        raw_line = _normalize_count_pk_equivalence(raw_line)
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
