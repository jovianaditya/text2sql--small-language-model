# Config Boundary Guide

This project intentionally splits configuration into four layers:

1. `.env` (runtime knobs + secrets)
- Use for deployment-specific values:
  - endpoints, model names, timeouts, debug flags, paths, feature toggles
- Keep values short and operational.
- Avoid storing long phrase lists/regex lexicons here.

2. `t2sql_rules.yml` (intent/rule content)
- Use for long and mostly-static language lists:
  - phrase lexicons, intent keyword lists, matching rules
- This keeps `.env` clean and makes rule changes auditable.

3. `t2sql_config.yml` (config lexical defaults)
- Loaded by `rulebook.py` and consumed by `config.py` when env overrides are absent.
- Treat this as static/default config data, not per-environment runtime settings.

4. Python code (`config.py`, `rulebook.py`)
- `config.py`:
  - parse env values
  - map aliases
  - expose typed constants and runtime flags
- `rulebook.py`:
  - load `t2sql_rules.yml` and `t2sql_config.yml`
  - provide typed list/dict accessors

## Quick Decision Rule
- If value changes per environment or is secret: put in `.env`.
- If value is intent/semantic rulebook-like: put in `t2sql_rules.yml`.
- If value is lexical default for config constants: put in `t2sql_config.yml`.
- If value is parsing/compat behavior: put in code.

## Backward-Compatible Aliases
Legacy env aliases are still accepted (for migration only), but preferred keys should be used in new setups.
Examples:
- `OLLAMA_BASE_URL` -> `OLLAMA_URL`
- `T2SQL_MCP_TIMEOUT` -> `MCP_CLIENT_TIMEOUT`
- `CHROMA_PATH` -> `CHROMA_DIR`
- `CHROMA_COLLECTION` -> `COLLECTION_NAME`

## Current Recommended Workflow
1. Keep `.env` minimal (runtime knobs + secrets only).
2. Keep lexical defaults in `t2sql_config.yml`.
3. Keep semantic/intent rules in `t2sql_rules.yml`.
4. Run env bloat checker when editing `.env`:
   - `python -m t2sql.scripts.check_env_bloat --env ../.env`

## Runtime Safety Note
- Removing `t2sql.scripts.*` helper scripts does not break runtime pipeline.
- Runtime depends on:
  - `.env` (optional overrides)
  - `t2sql_rules.yml`
  - `t2sql_config.yml`
  - `config.py` and `rulebook.py`