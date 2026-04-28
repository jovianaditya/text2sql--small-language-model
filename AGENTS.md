# AGENTS.md

## Project Overview
This is a Text to SQL system that converts natural language questions into SQL queries.
Built with Python, LangGraph, Model Context Protocol (MCP), Ollama, and MySQL

## YOU ARE A SENIOR MASTER ENGINEER
- Patch bugs **AGNOSTICALLY** — never hardcode values as a fix
- Never change or break existing logic that is already valid
- Always analyze the full issue before touching anything
- Minimal diffs — change only what is necessary

## Before Starting ANY Task
1. **Analyze** — fully understand the issue or request first
2. **Read** the relevant hand-off doc from the table below
3. **Plan** — write what you intend to change and why before writing code
4. Only then write code

## File Map — Read This Before Touching Any File

### `config.py` — Single Source of Truth for All Settings
All tunable constants for the entire system live here. Loaded from `.env` with
sensible defaults. Covers model names, retrieval scoring weights, LLM budgets,
regex patterns, agent behavior knobs, and domain heuristics profile.
- **Technical role**: this module does more than static constants. It contains
  env parsing helpers such as `_env_json_list()` and `_env_json_dict()`, profile
  selection via `DOMAIN_HEURISTICS_PROFILE`, collection-name inference from
  JSONL metadata filenames, and path inference for dense/sparse/schema metadata
  defaults. It is therefore the place where runtime behavior, data-source
  discovery, and fallback policy are centralized.
- **Touch this when**: adding a new env var, tuning a threshold, changing model
  runtime defaults, adjusting metadata path inference, or externalizing a magic
  value from another module
- **Never**: hardcode a value anywhere else — always add it here first
- **Key sections**: helper env parsers → collection/path inference → APP/GENERAL
  → RETRIEVAL → MCP → DEBUG → AGENT_* / HELPERS_* / INTENT_* constants
- **High-impact outputs**: `COLLECTION_NAME`, `DENSE_METADATA_JSONL`,
  `SPARSE_METADATA_JSONL`, `SCHEMA_METADATA_JSONL`, model names, retrieval
  scoring knobs, timeout budgets, HITL toggles

### `schema_analyzer.py` — Schema Metadata Engine (Foundation Layer)
Parses `metadata_jsonl` JSONL files to dynamically build the entire schema
understanding: table types (transaction/dimension/bridge), FK relationships,
column roles, synonyms, and datatypes. Everything schema-related flows from here.
- **Technical flow**: `SchemaAnalyzer(jsonl_path)` parses dense metadata JSONL,
  records table/column inventories, merges metadata descriptions with inline
  `Description:` text, extracts synonyms and role hints, tracks original column
  case, records PK/FK hints, then runs `_infer_table_types()` to classify tables
  as `transaction`, `dimension`, `bridge`, or `reference`. The module-level
  singleton accessors wrap this class and expose a schema API used everywhere
  else: `get_fk_relationships()`, `get_id_column()`, `get_name_column()`,
  `is_numeric_column()`, `get_columns_by_role()`, `infer_best_order_column()`,
  `infer_best_aggregate_column()`, and more.
- **Touch this when**: schema parsing is wrong, a new metadata field needs
  extraction, FK inference breaks, table typing is off, ID/name column selection
  is unstable, or semantic column classification should improve
- **Never**: hardcode table/column names anywhere — query this module instead
- **Key APIs**: `get_schema_analyzer()`, `get_all_tables()`, `get_fk_relationships()`,
  `get_id_column()`, `get_name_column()`, `is_numeric_column()`, `is_date_column()`,
  `get_aggregatable_columns()`, `get_filterable_date_columns()`
- **Depended on by**: `helpers_core.py`, `helpers_main.py`, `intent_ir.py`,
  `retrieval.py`, `confidence.py`, `agents.py`

### `intent_ir.py` — Natural Language → Intent Intermediate Representation
Converts the user's natural language question into a structured IR (dataclass)
before SQL is generated. Detects entities, filters, aggregations, and joins
using schema metadata. Acts as a plannability gate — high-confidence IR goes
direct to SQL planning, low-confidence falls back to LLM.
- **Technical flow**: defines the typed IR dataclasses `IRDimension`,
  `IRMetric`, `IRFilter`, `IRExistence`, `IROrder`, and the top-level
  `IntentIR`. `build_intent_ir()` parses the question against runtime allowed
  identifiers plus schema metadata, detects subject table, dimensions, metrics,
  filters, existence semantics, ordering, limit, distinctness, and include-zero
  behavior, then computes a confidence score and planner-compatibility flag.
  `is_ir_plannable()` is the hard gate used by drafting to decide whether the
  deterministic IR lane is safe enough to bypass LLM-first generation.
- **Touch this when**: intent parsing is wrong, a new question pattern is not
  detected, IR fields need extending, confidence/plannability gates are too loose
  or too strict, or schema-grounded column matching misses the right column
- **Reads from**: `schema_analyzer.py`, `config.py`
- **Key functions**: `build_intent_ir()`, `is_ir_plannable()`
- **Key outputs**: `subject_entity`, `dimensions`, `metrics`, `filters`,
  `existence`, `ordering`, `limit`, `select_distinct`, `include_zero`,
  `confidence`, `planner_compatible`, `mentioned_tables`
- **Feeds into**: SQL planning / agent draft phase via `agents.py`

### `helpers_core.py` — SQL Heuristic Rules Phase 0–7 (Low-level)
Core heuristic functions shared across all pipeline phases. Handles SQL
canonicalization (backticks, aliases, keywords), regex caching, and
guardrail logic for GROUP BY, COUNT, JOIN type decisions.
Organized by phase: Phase 0 = syntax/canonicalization → Phase 7 = final guards.
- **Technical flow**: this is the large shared heuristic library behind SQL
  cleanup and deterministic repair. Early phases normalize syntax, backticks,
  aliases, identifiers, and join conditions. Middle phases repair projection,
  GROUP BY/HAVING consistency, existence rewrites, winner-row/window patterns,
  regex predicates, null semantics, set operations, and distinct handling.
  Late phases add AST safety checks, window-function cleanup, order/limit
  enforcement, and final pre-execution guards. The regex helpers are cached to
  avoid repeated `re.compile()` cost across the normalization pipeline.
- **Touch this when**: a SQL syntax fix is wrong, a shared heuristic is
  misfiring, JOIN/GROUP BY/window rewrites are wrong, regex patterns need
  updating, or a repair belongs in the reusable low-level layer instead of a
  single agent path
- **Never**: add pipeline-orchestration logic here — keep this file generic,
  reusable, and phase-scoped
- **Key technical areas**: regex cache utilities, identifier parsing,
  canonicalization, join/existence rewrites, aggregate/group-by guards, regex
  normalization, null/distinct/set-op fixes, AST guardrails, final safety pass
- **Reads from**: `schema_analyzer.py`, `config.py`, `rulebook`

### `helpers_main.py` — SQL Postprocess & Normalization Pipeline (High-level)
Orchestrates the full SQL postprocessing pipeline. Handles whitelist validation
(allowed identifiers), prompt building for LLM generation, and the main
normalize/postprocess chain that calls into `helpers_core.py` phase by phase.
- **Technical flow**: this module is the high-level coordinator that turns the
  low-level helper library into an ordered SQL cleanup pipeline. It validates SQL
  identifiers against the runtime `### ALLOWED IDENTIFIERS` block, classifies
  question/query type, builds retrieval probes via `build_dynamic_probe()`,
  sanitizes notes, repairs alias mismatches, and drives the full
  `normalize_sql_pipeline()` pass over raw draft SQL. It is also where cross-cut
  prompt-building and postprocess policy live, rather than in the low-level
  helper file.
- **Touch this when**: the overall postprocess pipeline is wrong, prompt
  templates need changing, identifier whitelist validation breaks, retrieval
  probes are poor, query classification is weak, or phase ordering in SQL
  normalization should change
- **Reads from**: `helpers_core.py`, `schema_analyzer.py`, `config.py`, `utils.py`
- **Key functions**: `validate_identifiers()`, `classify_query()`, `_get_qtype()`,
  `build_dynamic_probe()`, `normalize_sql_pipeline()`
- **This is the main entry point** for SQL postprocessing — start here when debugging output quality

### `chroma_embedding.py` — Dense Vector Store & Embedding Lifecycle
Owns the dense retrieval storage layer backed by persistent ChromaDB. The main
entry point is `build_vector_store()`, which creates or reuses the singleton
collection, computes a source signature from `DENSE_METADATA_JSONL`, compares it
with the saved stamp file, and decides whether to keep, auto-ingest, or fully
rebuild the collection. `ingest_dense_data()` performs JSONL batch upserts and
flattens nested metadata so Chroma can store it safely. `OllamaEmbeddingFunction`
is the embedding adapter that calls Ollama `/api/embeddings` and provides the
vector function used by the collection.
- **Touch this when**: dense retrieval returns stale rows, collection counts are wrong, dense metadata changes are not triggering rebuilds, batch upsert metadata is rejected by Chroma, or Ollama embedding responses/timeouts need adjustment
- **Reads from**: `config.py` (`CHROMA_DIR`, `COLLECTION_NAME`, `OLLAMA_URL`, `DENSE_METADATA_JSONL`)
- **Writes/side effects**: persistent Chroma collection files and the dense-source stamp file under `CHROMA_DIR`
- **Key functions**: `build_vector_store()`, `ingest_dense_data()`, `OllamaEmbeddingFunction`
- **Depended on by**: `ingestion.py`, `retrieval.py`

### `ingestion.py` — LangGraph Ingestion Node
Thin orchestration node for the graph ingestion stage. Delegates the real work
to `chroma_embedding.py`, then records collection counts and prints the
ingestion report that appears during runtime.
- **Technical flow**: `ingest_node(state)` calls `build_vector_store()`, reads
  `collection.count()`, and updates the shared `RAGState` counters:
  `added`, `updated`, and `skipped`. In the current implementation this node is
  intentionally thin and treats existing indexed docs as skipped while the real
  ingest/rebuild decision stays centralized in `chroma_embedding.py`.
- **Touch this when**: graph-stage ingestion telemetry is wrong, the printed
  report is misleading, node ordering around ingest is broken, or state counters
  need to reflect a different ingestion policy
- **Reads from**: `state.py`, `chroma_embedding.py`, `config.py`
- **Writes to state**: `added`, `updated`, `skipped`
- **Key function**: `ingest_node()`
- **Feeds into**: `graph.py`

### `retrieval.py` — Hybrid Retrieval, Reranking, and Context Builder
Implements the RAG retrieval layer: sparse BM25 search over lexical JSONL,
dense Chroma retrieval, reranking/boosting, table scoring, FK path discovery,
and construction of the final LLM context blocks and allowed identifiers.
- **Technical flow**: `retrieve_node(state)` first opens the dense Chroma
  collection, derives `qtype` via `helpers_main._get_qtype()`, optionally
  detects a primary entity table for lookup-style questions, filters the input
  JSONL list down to sparse metadata files, builds a BM25 query from the user
  question, and runs sparse search + dense vector search in parallel conceptually.
  It then reranks candidates using distance, sparse hits, table/column role
  boosts, query-intent heuristics, and FK/bridge-table reasoning before building
  the final context text. That context includes `### ALLOWED IDENTIFIERS`,
  metadata cards, stats summaries, and join-relevant evidence for downstream SQL generation.
- **Touch this when**: the wrong tables/columns are surfaced, dense/sparse
  balance is off, lookup questions over-select tables, bridge/FK inference is
  wrong, allowed identifiers are missing columns, or the context window is too noisy
- **Reads from**: `config.py`, `state.py`, `chroma_embedding.py`, `schema_analyzer.py`, `helpers_main.py`, `utils.py`
- **Writes to state**: primarily `context`, and retrieval-derived hints used later by routing/drafting
- **Key functions**: `retrieve_node()`, `build_sparse_query_from_question()`, `jsonl_sparse_search()`, `score_tables_from_reranked()`, `_build_allowed_block()`, `_build_cards()`
- **Feeds into**: `agents.py` via `state["context"]`

### `agents.py` — Question Guard, Router, Drafter, and Refiner
This is the main multi-agent decision layer. It validates whether a question is
in-scope, routes complexity/model choice, drafts SQL, applies deterministic and
LLM-assisted repairs, and performs the final refinement/acceptance flow.
- **Technical flow**:
  `question_guard_node()` runs first and can terminate the graph early by
  setting `router_decision.is_valid_question=False`. `router_agent()` reads the
  retrieval context, resolves relevant tables/primary table, estimates query
  type/complexity, and chooses the target model lane. `draft_sql_agent()` is
  IR-first: it tries `build_intent_ir()`, evaluates whether the IR is plannable,
  records IR telemetry (`ir_used`, `ir_confidence`, `ir_gate_reasons`, etc.),
  and only falls back to LLM drafting when deterministic planning is not safe
  enough. `refine_sql_agent()` runs the postprocess/heuristic normalization,
  semantic-contract enforcement, AST guardrails, confidence scoring, retry
  logic, and final SQL acceptance path.
- **Touch this when**: valid questions are rejected, router complexity/model
  choice is wrong, primary table detection is unstable, IR planning should have
  been used but was skipped, draft SQL is malformed, semantic repair changes good
  SQL, or final acceptance gates are letting bad SQL through
- **Reads from**: `config.py`, `llm_utils.py`, `generation.py`, `intent_ir.py`, `confidence.py`, `helpers_main.py`, `state.py`, `utils.py`
- **Writes to state**: `router_decision`, `router_tables`, `primary_table`,
  `target_model`, `final_model`, `intent_ir`, `draft_sql`, `draft_source`,
  `final_sql`, `answer`, `agent_notes`, `validation_errors`, `critic_feedback`,
  confidence telemetry, and IR telemetry fields
- **Key functions**: `question_guard_node()`, `router_agent()`, `draft_sql_agent()`, `refine_sql_agent()`
- **This is the main entry point** for agent behavior — start here when debugging end-to-end SQL generation decisions

### `generation.py` — LLM Output Extraction & Constrained Regeneration
Utility helpers for parsing imperfect LLM outputs back into structured JSON and
for rerunning generation with extra constraints. Keeps repair flows robust when
models return code fences, wrapper text, or malformed JSON.
- **Technical flow**: `_strip_code_fences()` and `_extract_json_object()` try to
  recover a valid JSON object from noisy model text by removing code fences and
  performing brace-matching extraction. `_validate_sql_notes()` enforces the
  expected minimal contract (`sql` and/or `notes`). `regenerate_with_extra_constraint()`
  builds a stricter prompt, calls `try_llm_json()`, and returns normalized
  `(sql, notes)` for retry/repair paths.
- **Touch this when**: model responses contain wrappers that are not parsed,
  retries lose the SQL payload, or stricter regeneration prompts need a more
  reliable extraction contract
- **Reads from**: `utils.py`
- **Key functions**: `_extract_json_object()`, `_validate_sql_notes()`, `regenerate_with_extra_constraint()`
- **Depended on by**: `agents.py`

### `confidence.py` — SQL Confidence Scoring Layer
Computes a schema-agnostic confidence score for generated SQL using multiple
signals: syntax validity, identifier validity, entity coverage, join-path
plausibility, and structural sanity checks.
- **Technical flow**: `compute_sql_confidence(sql, question, ctx)` combines five
  weighted signals: SQL parse validity via `sqlglot`, identifier validation
  against the runtime `### ALLOWED IDENTIFIERS` block, entity coverage between
  question and SQL tables, join-path validation using schema FK metadata plus
  fallback adjacency heuristics, and structural plausibility checks such as
  aggregate/order-column sanity. The result includes an overall score, per-signal
  scores, warning strings, and detailed diagnostics used by retry logic.
- **Touch this when**: low-confidence warnings are noisy, the weighting between
  syntax/schema/semantic checks is wrong, aggregate/order checks create false
  positives, or retries should be triggered by different evidence
- **Reads from**: `schema_analyzer.py`
- **Key functions**: `compute_sql_confidence()`, `format_confidence_report()`
- **Depended on by**: `agents.py`

### `llm_utils.py` — Central LLM Builders
Provides the shared `ChatOllama` builders used by the agent pipeline. Splits
JSON-mode generation from free-text repair mode and centralizes keep-alive,
timeout, and token-budget behavior.
- **Technical flow**: `build_llm_json()` configures `ChatOllama` with
  `format="json"` for agents that must emit machine-readable objects, while
  `build_llm_repair()` uses text mode for refinement/repair. Both builders share
  the base URL, keep-alive behavior, context window, and timeout settings pulled
  from configuration so model switching stays centralized.
- **Touch this when**: Ollama connection settings are wrong, the JSON agent is
  emitting non-JSON, keep-alive/model switching is inefficient, timeouts are too
  short, or token budgets should be tuned globally
- **Reads from**: `config.py`
- **Key functions**: `build_llm_json()`, `build_llm_repair()`
- **Depended on by**: `agents.py`

### `graph.py` — LangGraph Workflow Assembly
Defines the application graph and the stage order between guard, ingestion,
retrieval, router, drafter, and refiner. This is where feature flags determine
which branches execute and where the compiled app object is built.
- **Technical flow**: `build_app()` constructs a `StateGraph(RAGState)` with the
  node sequence `guard -> ingest -> retrieve -> router -> draft_sql -> refine_sql`.
  The helper branch functions (`_after_guard`, `_after_ingest`, `_after_retrieve`,
  `_after_draft`) read `scenario_config` and `router_decision` to decide whether
  the run exits early, skips RAG, skips multi-agent routing, or continues through
  refinement before `END`.
- **Touch this when**: stage order is wrong, a branch is taken unexpectedly,
  `scenario_config` flags are not respected, or a new node/branch must be added
- **Reads from**: `state.py`, `ingestion.py`, `retrieval.py`, `agents.py`
- **Key function**: `build_app()`
- **This is the main entry point** for the workflow graph — start here when debugging stage transitions

### `mcp_execution.py` — MCP SQL Execution Client
Wraps the MCP tool calls that actually execute SQL against the database. It
normalizes payloads, applies timeout handling, and exposes both preview-mode
execution and full-result execution for export/evaluation flows.
- **Technical flow**: `_mcp_call()` wraps `client.call_tool()` in
  `asyncio.wait_for()`. `query_db()` sends SQL to the MCP `query_db` tool and is
  intended for preview-mode execution, while `query_db_full()` calls
  `query_db_full` to fetch the full result set for evaluation/export. Both paths
  strip trailing semicolons, optionally inject a database override, normalize the
  returned payload with `_extract_payload()` / `_ensure_json()`, and can either
  reuse an existing `Client` or create their own transport session.
- **Touch this when**: MCP tool calls fail, payload normalization is wrong,
  timeout behavior is incorrect, preview/full execution diverge, or result parsing
  breaks because the MCP server returns a different shape
- **Reads from**: `config.py`, `utils.py`
- **Key functions**: `query_db()`, `query_db_full()`
- **Depended on by**: `repl.py`

### `repl.py` — Interactive Runtime Loop
Runs the interactive question-answering session. It builds the graph app,
opens the MCP client transport, executes the pipeline for each question,
prints latency/debug output, handles HITL review flow, and supports CSV export
for full query results.
- **Technical flow**: `repl(paths)` builds the LangGraph app once, opens a
  shared MCP transport session, then loops through user questions. For each
  question `_run_question_with_optional_review()` creates `make_state(paths, question)`,
  invokes the graph, runs `_exec_via_mcp()` on the final SQL after the safety
  pre-pass, logs end-to-end timing, and optionally triggers HITL review. Success
  review can export full results to CSV using `query_db_full()`. Failure review
  records structured audit logs including SQL, router output, confidence, and
  user feedback.
- **Touch this when**: interactive runs fail, graph execution exceptions are not
  surfaced well, latency reporting is misleading, SQL execution safety behavior
  is wrong, HITL review logging breaks, or CSV export flow needs changes
- **Reads from**: `graph.py`, `state.py`, `mcp_execution.py`, `helpers_core.py`, `config.py`, `utils.py`
- **Key functions**: `repl()`, `_run_question_with_optional_review()`, `_exec_via_mcp()`, `_prompt_hitl_review()`, `_prompt_success_review()`
- **This is the main runtime shell** for manual testing — start here when debugging REPL behavior

### `cli.py` — Command-Line Bootstrap
Small startup wrapper for launching the REPL from the command line. Loads `.env`,
resolves sparse JSONL inputs, and handles async event-loop compatibility before
delegating to `repl.py`.
- **Technical flow**: `main()` resolves `SPARSE_METADATA_JSONL` into actual path
  inputs using `resolve_jsonl_paths()`, fails fast if no sparse metadata is
  available, then launches `asyncio.run(repl(paths))`. It includes a fallback for
  environments that already have an active event loop by applying `nest_asyncio`
  and rerunning the coroutine on the current loop.
- **Touch this when**: startup fails before the REPL opens, `.env` loading is
  wrong, sparse metadata paths are not resolved correctly, or notebook/embedded
  environments need different event-loop handling
- **Reads from**: `config.py`, `utils.py`, `repl.py`
- **Key function**: `main()`
- **This is the CLI entry point** for local runs

### `rulebook.py` — YAML Rule Loader & Phrase-Matching Utilities
Loads `t2sql_rules.yml` as a cached rulebook and exposes reusable helpers for
normalized text matching, lexical phrase detection, proximity rules, and
name/identifier intent heuristics.
- **Technical flow**: `_load_rules()` lazily loads and caches the YAML rulebook,
  optionally overridden by `T2SQL_RULES_PATH`. The module then exposes typed
  accessors such as `rule_value()`, `rule_list()`, and `rule_dict()`, plus
  normalized phrase-matching helpers like `phrase_in_text()` and proximity-based
  checks. Higher-level helpers such as `question_wants_split_name_components()`
  and `question_wants_combined_full_name()` convert YAML-configured phrase sets
  into concrete intent decisions used by downstream heuristics.
- **Touch this when**: YAML-driven phrase logic is wrong, lexical rules need
  extending, cache invalidation/reload behavior is incorrect, or intent helpers
  should be data-driven instead of hardcoded
- **Reads from**: `t2sql_rules.yml`
- **Key functions**: `_load_rules()`, `rule_value()`, `rule_list()`, `phrase_in_text()`, `question_wants_split_name_components()`, `question_wants_combined_full_name()`
- **Depended on by**: heuristic and intent layers that need configurable phrase rules

### `state.py` — Shared Pipeline State Contract
Defines the `RAGState` TypedDict shared across the LangGraph workflow and the
`make_state()` initializer that seeds every run with consistent defaults for
retrieval, routing, SQL generation, telemetry, confidence, and HITL fields.
- **Technical contract**: `RAGState` is the single shared state object passed
  through LangGraph nodes. It holds input paths/question, retrieval context,
  router outputs, primary-table decisions, model selection, draft/final SQL,
  IR telemetry, confidence telemetry, validation errors, HITL audit fields, and
  benchmark timing fields. `make_state()` is the authoritative initializer for
  those defaults, so new cross-node data should be added here first.
- **Touch this when**: a new field must flow between nodes, defaults are missing,
  state ownership between stages is unclear, or one node is assuming a field that
  another node never initializes
- **Key type/function**: `RAGState`, `make_state()`
- **Depended on by**: `graph.py`, `retrieval.py`, `agents.py`, `ingestion.py`, `repl.py`

## Development Questions
### `qa_classicmodels_evaluation_easy.yml` — Development questions for easy difficulty
Contains only easy difficulty questions with mix natural language style (semi-formal to casual).
Single table only, filter-based queries. Covers: WHERE, IN, NOT IN, BETWEEN, IS NULL,
LIKE, REGEXP, ORDER BY, LIMIT, date functions (YEAR, MONTH, DAYOFWEEK),
string functions (UPPER, LEFT, LENGTH), arithmetic columns, DISTINCT, COALESCE,
multi-condition (AND, OR), and column comparison.
- **Total**: 50 questions
- **Technical structure**: YAML list under `cases`, where each case has `id`,
  `difficulty`, `question`, and `groundtruth_sql`
- **Primary use**: validates the single-table/filter lane, lexical retrieval,
  identifier selection, predicate normalization, and simple ordering/limit behavior
- **Run results**: You can check the logs for easy difficulty question on this file: `evaluation_easy,log` 

### `qa_classicmodels_evaluation_medium.yml` — Development questions for medium difficulty
Contains only medium difficulty questions with mix natural language style (semi-formal to casual).
Single table only, but includes aggregation and advanced single-table operations.
Covers: GROUP BY, HAVING, aggregate functions (COUNT, SUM, AVG, MIN, MAX),
window functions (RANK, DENSE_RANK, ROW_NUMBER, LEAD, LAG, running total),
subqueries (single-table), and CASE WHEN.
- **Total**: 50 questions
- **Technical structure**: YAML list under `cases`, where each case has `id`,
  `difficulty`, `question`, and `groundtruth_sql`
- **Primary use**: validates aggregate planning, grouped output repair, HAVING
  semantics, window-function rewriting, and single-table subquery handling
- **Run results**: You can check the logs for easy difficulty question on this file: `evaluation_medium.log` 

### `qa_classicmodels_evaluation_hard.yml` — Development questions for hard difficulty
Contains only hard difficulty questions with mix natural language style (semi-formal to casual).
Multi-table queries involving JOIN (up to 3 tables). Combines patterns from easy and medium:
JOIN + aggregation, JOIN + window functions, JOIN + subquery, JOIN + HAVING,
JOIN + CASE WHEN, self-join, LEFT JOIN + IS NULL anti-pattern, NOT EXISTS,
correlated subqueries, and pivot-style CASE WHEN aggregation.
- **Total**: 30 questions
- **Technical structure**: YAML list under `cases`, where each case has `id`,
  `difficulty`, `question`, and `groundtruth_sql`
- **Primary use**: validates routing to multi-table complexity, join-path
  selection, anti-join/exclusion semantics, correlated-subquery rewrites, and
  cross-table aggregation/window patterns
- **Run results**: You can check the logs for easy difficulty question on this file: `evaluation_hard.log` 

## Coding Rules
- Never hardcode values; use schema-driven approach — use metadata files on data/python/mcp_client/metadata_jsonl`
- Do not refactor working logic while fixing a bug — isolate the fix
- SQL generation must be agnostic to table/column names (read from schema, not hardcoded)

## Tech Stack
- **Orchestration**: LangGraph
- **LLM Runtime**: Ollama (qwen2.5:3b, qwen2.5:7b)
- **Protocol**: Model Context Protocol (MCP)
- **Database**: MySQL
- **Language**: Python

## When Fixing Bugs
1. Read the relevant hand-off doc first
2. Identify the root cause — do not patch symptoms
3. State your fix plan before writing code
4. Make the smallest possible change that fixes the issue
5. Confirm no existing logic was broken
