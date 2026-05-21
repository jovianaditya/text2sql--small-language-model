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
from itertools import zip_longest
from pathlib import Path
from typing import Dict, Iterable


DEFAULT_LOG_BY_DIFFICULTY = {
    "easy": "evaluation_easy_p1.log",
    "medium": "evaluation_medium_p1.log",
    "hard": "evaluation_hard_p1.log",
}

QUESTION_PREFIX_BY_DIFFICULTY = {
    "easy": "E",
    "medium": "M",
}

CASE_HEADER_RE = re.compile(r"^\[(?P<qid>[A-Z]\d+)\]\s.*$")
SUMMARY_START_RE = re.compile(r"^\[CANDIDATE INTENT SUMMARY\]")
SUMMARY_STOP_PREFIX = "Use this intent only as a reference."
NUMBER_RE = re.compile(r"\d+")
COMPACT_DIFFICULTY_FLAG_RE = re.compile(r"^(--(?P<difficulty>easy|medium|hard))(?P<suffix>\d.*)$")

_EMBEDDED_GOLD_BLOB = (
    "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"
    "I6zl6a8AAA=="
)


def _numeric_qid_key(qid: str) -> tuple[str, int]:
    match = NUMBER_RE.search(qid)
    return qid[:1], int(match.group()) if match else 0


def _normalize_summary(summary: str) -> str:
    return "\n".join(line.rstrip() for line in summary.strip().splitlines())


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


GOLD_BENCHMARKS = _load_embedded_gold()


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
    capture: list[str] | None = None
    cases: Dict[str, Dict[str, str]] = {}

    for raw_line in log_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        header_match = CASE_HEADER_RE.match(line)
        if header_match:
            current_qid = header_match.group("qid")
            current_question_line = line

        if SUMMARY_START_RE.match(line):
            capture = [line]
            continue

        if capture is None:
            continue

        if line.startswith(SUMMARY_STOP_PREFIX):
            if current_qid and current_qid.startswith(expected_prefix):
                cases[current_qid] = {
                    "question_line": current_question_line or current_qid,
                    "summary": _normalize_summary("\n".join(capture)),
                }
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


def _print_changed_lines(expected_summary: str, actual_summary: str) -> None:
    print("Changed lines:")
    changed_count = 0
    for expected_line, actual_line in zip_longest(
        expected_summary.splitlines(),
        actual_summary.splitlines(),
        fillvalue="<missing>",
    ):
        if expected_line == actual_line:
            continue
        changed_count += 1
        print(f"  - Gold     : {expected_line}")
        print(f"  - Predicted: {actual_line}")

    if changed_count == 0:
        print("  - No line-level differences detected after normalization.")


def compare_cases(
    difficulty: str,
    actual_cases: Dict[str, Dict[str, str]],
    selected_ids: set[int] | None = None,
) -> tuple[list[str], list[str], list[tuple[str, str, str]]]:
    gold_cases = filter_cases_by_ids(GOLD_BENCHMARKS[difficulty], difficulty, selected_ids)
    gold_ids = set(gold_cases)
    actual_ids = set(actual_cases)

    missing = sorted(gold_ids - actual_ids, key=_numeric_qid_key)
    unexpected = sorted(actual_ids - gold_ids, key=_numeric_qid_key)

    mismatches: list[tuple[str, str, str]] = []
    for qid in sorted(gold_ids & actual_ids, key=_numeric_qid_key):
        expected_summary = _normalize_summary(gold_cases[qid]["summary"])
        actual_summary = _normalize_summary(actual_cases[qid]["summary"])
        if expected_summary != actual_summary:
            mismatches.append((qid, expected_summary, actual_summary))

    return missing, unexpected, mismatches


def print_compare_report(
    difficulty: str,
    log_path: Path,
    actual_cases: Dict[str, Dict[str, str]],
    selected_ids: set[int] | None = None,
) -> int:
    gold_cases = filter_cases_by_ids(GOLD_BENCHMARKS[difficulty], difficulty, selected_ids)
    missing, unexpected, mismatches = compare_cases(difficulty, actual_cases, selected_ids)

    print(f"Difficulty: {difficulty}")
    print(f"Log: {log_path}")
    print(f"Gold benchmark cases: {len(gold_cases)}")
    print(f"Extracted cases: {len(actual_cases)}")
    print(f"Missing cases: {len(missing)}")
    print(f"Unexpected cases: {len(unexpected)}")
    print(f"Mismatched summaries: {len(mismatches)}")
    print()

    if missing:
        print("Missing cases from log:")
        for qid in missing:
            print(f"  - {qid}: {gold_cases[qid]['question_line']}")
        print()

    if unexpected:
        print("Unexpected cases in log:")
        for qid in unexpected:
            print(f"  - {qid}: {actual_cases[qid]['question_line']}")
        print()

    if mismatches:
        print("Summary mismatches:")
        total_mismatches = len(mismatches)
        separator = "=" * 80
        inner_separator = "-" * 80
        for index, (qid, expected_summary, actual_summary) in enumerate(mismatches, start=1):
            print(separator)
            print(f"[Mismatch {index}/{total_mismatches}] {qid}")
            print(f"Question: {gold_cases[qid]['question_line']}")
            print(inner_separator)
            _print_indented_block("Gold summary:", expected_summary)
            print()
            _print_indented_block("Predicted summary:", actual_summary)
            print()
            _print_changed_lines(expected_summary, actual_summary)
        print()

    if not missing and not unexpected and not mismatches:
        print("Benchmark match: PASS")
        return 0

    print("Benchmark match: FAIL")
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
            help="Reserved for hard questions. Gold benchmark is not embedded yet.",
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
        return 0

    log_path = _resolve_log_path(difficulty, getattr(args, "log", None))
    cases = filter_cases_by_ids(parse_log(log_path, difficulty), difficulty, selected_ids)

    if args.command == "extract":
        if args.json:
            print_cases_json(difficulty, str(log_path), cases)
        else:
            print_cases_text(cases)
            print(f"Extracted cases: {len(cases)}")
        return 0

    return print_compare_report(difficulty, log_path, cases, selected_ids)


if __name__ == "__main__":
    sys.exit(main())
