from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

from .helpers import classify_query

if TYPE_CHECKING:
    from .state import RAGState


def _now() -> float:
    return perf_counter()


def _elapsed_ms(t0: float) -> int:
    return int((perf_counter() - t0) * 1000)


def _log_total(label: str, t0: float):
    ms = _elapsed_ms(t0)
    print(f"\n[TOTAL] {label}: {ms} ms (~{ms/1000:.3f}s)")


def _print_banner(title: str, width: int = 150) -> str:
    line = "─" * width
    print(f"\n{line}")
    print(title.center(width))
    print(line)
    return line

def _get_requires_regex(state: "RAGState") -> bool:
    """
    Hitung satu kali apakah pertanyaan butuh REGEXP,
    lalu simpan ke state['requires_regex'].
    """
    val = state.get("requires_regex")
    if isinstance(val, bool):
        return val

    q = state.get("question", "")
    needs = _question_requires_regex(q)
    state["requires_regex"] = needs
    return needs

def _get_qtype(state: "RAGState") -> str:
    q = (state.get("question") or "").strip()
    if not q:
        return "lookup"

    qtype = state.get("qtype") or ""
    if qtype:
        return qtype

    qtype = classify_query(q)
    state["qtype"] = qtype
    return qtype