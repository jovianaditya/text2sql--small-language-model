# utils_timing.py
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Optional, Dict


@contextmanager
def timed(label: str, store: Optional[Dict[str, float]] = None, print_each: bool = True):
    """
    Context manager untuk ngukur durasi blok kode.
    - label: nama stage
    - store: dict untuk simpan ms per label (opsional)
    - print_each: kalau False, tidak print per stage (buat ringkasan di akhir saja)
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if isinstance(store, dict):
            store[label] = dt_ms
        if print_each:
            print(f"[TIMING] {label}: {dt_ms:,.1f} ms")