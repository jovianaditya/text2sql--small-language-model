#!/usr/bin/env python3
"""Repo-root wrapper for the packaged intent IR benchmark utility."""

from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent / "data" / "python" / "mcp_client"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from t2sql.intent_ir_benchmark import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
