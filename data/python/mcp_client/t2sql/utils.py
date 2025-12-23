from __future__ import annotations
import glob
import json
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

from .config import SQL_DEBUG
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

def _dbg_print_two(label: str, sql: str):
    if SQL_DEBUG:
        print(f"\n{label}")
        print(sql)

def resolve_jsonl_paths(spec: str) -> List[str]:
    paths: List[str] = []
    if not spec:
        return paths
    p = Path(spec)
    if any(ch in spec for ch in "*?[]"):
        paths.extend(glob.glob(spec))
    elif p.is_dir():
        paths.extend(str(x) for x in p.glob("*.jsonl"))
    elif p.is_file():
        paths.append(str(p))
    else:
        paths.extend(glob.glob(spec))
    out, seen = [], set()
    for x in paths:
        xp = str(Path(x))
        if xp.lower().endswith(".jsonl") and xp not in seen:
            seen.add(xp)
            out.append(xp)
    return sorted(out)

def read_jsonl(path: str) -> Iterable[tuple[str, Optional[Dict[str, Any]]]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                obj = None
            yield s, obj

def _meta_table(md: Dict[str, Any]) -> str:
    return str(md.get("table") or md.get("tableName") or "").strip()

def _meta_column(md: Dict[str, Any]) -> str:
    return str(md.get("column") or md.get("columnName") or "").strip()