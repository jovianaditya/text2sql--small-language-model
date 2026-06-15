# Streamlit Frontend

Run from the `data/python/mcp_client` package root:

```bash
streamlit run t2sql/streamlit/app.py
```

The database selector shows the active configured database label. With the current project config, that label is `classicmodels`.

The app reuses the existing Text-to-SQL runtime:

- `SPARSE_METADATA_JSONL` is resolved with `t2sql.utils.resolve_jsonl_paths()`
- SQL generation runs through `t2sql.graph.build_app()`
- generated SQL is checked with the same pre-execution safety pass used by the REPL
- execution calls the existing MCP `query_db` path

Install the project dependencies with:

```bash
pip install -e .
```
