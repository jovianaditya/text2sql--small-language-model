# ======================================================================
# =============================== cli.py ===============================
# ======================================================================
# Purpose:
# - Command-line interface for the T2SQL REPL.
# - Loads environment variables and initializes the REPL with JSONL input files.
# - Handles asynchronous execution and compatibility with existing event loops.
# - Provides user feedback when no input files are found.

# ======================================================================
# ====================== STANDARD LIBRARY IMPORT =======================
# ======================================================================
import asyncio
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=False)
# ======================================================================
# ======================== LOCAL MODULE IMPORT =========================
# ======================================================================
from .config import JSON_INPUT
from .utils import resolve_jsonl_paths
from .repl import repl

def main():
    paths = resolve_jsonl_paths(JSON_INPUT)
    if not paths:
        raise RuntimeError(
            "No JSONL inputs found.\n"
            "Set JSON_INPUT or EXPORT_JSONL_OUTPUT to one of:\n"
            "  • a single file: ./exports/schema_export.jsonl\n"
            "  • a directory:   ./datahub_metadata\n"
            "  • a glob:        ./datahub_metadata/*.jsonl\n"
        )

    try:
        asyncio.run(repl(paths))
    except RuntimeError as e:
        if "asyncio.run() cannot be called" in str(e):
            import nest_asyncio
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            loop.run_until_complete(repl(paths))
        else:
            raise

if __name__ == "__main__":
    main()
