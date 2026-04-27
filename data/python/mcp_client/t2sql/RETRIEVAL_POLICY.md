# Retrieval Policy

This document explains how retrieval works in the `t2sql` Text-to-SQL pipeline.
It is written to be presentation-ready, including for readers who do not already
know how retrieval, embeddings, BM25, or RAG systems work.

The implementation is mainly in `retrieval.py`. Configuration values are defined
in `config.py`, so retrieval behavior can be tuned without hardcoding values in
the pipeline.

## 1. Why Retrieval Exists

The Text-to-SQL system receives a natural-language question, for example:

```text
Show customer names and their order dates.
```

The SQL generator cannot safely use the whole database schema blindly. If it sees
too many tables and columns, it may:

- use the wrong table,
- invent a column that does not exist,
- join tables incorrectly,
- choose an ID column when the user wanted a name column,
- use a numeric metric column that was not requested.

Retrieval solves this by selecting a small, relevant schema context before SQL is
generated.

In simple terms:

```text
User question
  -> retrieval finds the most relevant tables and columns
  -> SQL generator receives only that controlled context
  -> SQL is generated with fewer hallucinated identifiers
```

The final retrieval output is not SQL. It is a compact schema context that guides
the SQL generator.

## 2. Key Terms

| Term | Meaning |
| --- | --- |
| User question | The natural-language input from the user. |
| Schema | The database structure: tables, columns, data types, keys, and relationships. |
| Metadata JSONL | A JSONL representation of schema knowledge used by retrieval. |
| Dense retrieval | Semantic search using embeddings. It finds columns that are conceptually similar to the question. |
| Sparse retrieval | Lexical search using BM25. It finds columns that match exact words, aliases, IDs, codes, or sample values. |
| Candidate | A table/column document returned by dense or sparse retrieval. |
| Reranking | The process of sorting candidates again using combined evidence. |
| Context | The final information passed to the SQL generator. |
| Allowed identifiers | The whitelist of tables and columns the SQL generator may use. |
| Cards | Short metadata descriptions for retrieved columns. |
| Alias map | A mapping from aliases/synonyms to real column names. |

## 3. High-Level Flow

The retrieval node follows this flow:

```text
1. Receive user question
2. Detect query type, for example lookup, aggregate, or rank
3. Build sparse query for BM25
4. Run sparse retrieval over JSONL metadata
5. Build dense embedding probe
6. Run dense retrieval over ChromaDB
7. Rerank dense candidates with sparse and metadata signals
8. Keep only final top candidates
9. Score and select relevant tables
10. Build final context:
    - ALLOWED IDENTIFIERS
    - CARDS
    - _INTERNAL_HINT.alias_map
```

The system uses hybrid retrieval because dense and sparse retrieval are good at
different things.

## 4. Metadata Inputs

Retrieval uses schema metadata stored as JSONL files.

There are two main JSONL styles:

| JSONL Type | Typical Filename | Used For |
| --- | --- | --- |
| Dense JSONL | `*_dense.jsonl` | ChromaDB semantic embedding search |
| Sparse JSONL | `*_sparse.jsonl` | BM25 lexical search |

Each JSONL item represents a searchable schema document, usually a table/column
entry. It may include:

- document ID,
- table name,
- column name,
- column description,
- aliases and synonyms,
- column roles, such as ID, name, metric, or date,
- datatype,
- statistics, such as samples or min/max values,
- FK relationship hints.

The retrieval policy is schema-driven. It does not hardcode dataset-specific
table or column names. Instead, it relies on metadata and `schema_analyzer.py`.

## 5. Ingestion and Vector Store

Dense retrieval uses ChromaDB. The Chroma collection is created by
`build_vector_store()`.

If the collection is empty, the system automatically ingests available JSONL
files. It prefers dense JSONL files because those are designed for semantic
embedding retrieval.

Each JSONL item is stored in Chroma as:

| Field | Purpose |
| --- | --- |
| `id` | Unique document or column ID |
| `text` | Text that will be embedded |
| `metadata` | Table/column metadata used later for scoring and context |

The default embedding model is:

```text
snowflake-arctic-embed2:568m
```

Chroma uses cosine distance. A smaller distance means the query and candidate are
semantically closer.

## 6. Sparse Retrieval

Sparse retrieval is the lexical search path. It is implemented by
`jsonl_sparse_search()`.

### 6.1 What Sparse Retrieval Is

Sparse retrieval searches by word overlap. In this system, it uses BM25.

BM25 is useful when the question contains words that should match schema
metadata directly, such as:

- exact column names,
- aliases,
- ID/code terms,
- sample values,
- words inside descriptions,
- business terms stored in `metadata.keywords`.

Example:

```text
Question: Show customers from USA
```

Sparse retrieval can match `USA` if that value appears in sample values or
keywords, even if embedding search alone does not prioritize it.

### 6.2 Sparse File Selection

The node reads `state["jsonl_paths"]` and keeps files whose path contains:

```text
RETRIEVAL_SPARSE_FILENAME_MARKER = "_sparse"
```

If no `_sparse` file is found, it falls back to all JSONL paths for backward
compatibility.

### 6.3 Sparse Query Construction

The sparse query is built by `build_sparse_query_from_question()`.

The process:

1. Convert the question to lowercase.
2. Normalize thousand separators, for example `1,000` becomes `1000`.
3. Normalize possessives, for example `customer's` becomes `customer`.
4. Replace non-alphanumeric characters with spaces.
5. Remove stopwords, while preserving configured exceptions like `name` and
   `names`.
6. Apply light plural stemming:
   - `ies` becomes `y`,
   - trailing `s` can be removed with configured exceptions.
7. Add schema-derived phrase aliases. For example, if metadata says a phrase is
   an alias for a column, the real column token can be added to the query.

The goal is to make the user question easier for BM25 to match against schema
metadata.

### 6.4 Sparse Corpus Construction

For every JSONL item, the system builds BM25 tokens from the best available
source:

1. `metadata.keywords`, if present. This is preferred because it can contain
   aliases, sample values, role tags, and retrieval-specific terms.
2. `text`, if present.
3. Fallback fields:
   - table name,
   - column name,
   - description,
   - aliases,
   - roles,
   - table synonyms,
   - column synonyms.

The BM25 index is cached using file path, modification time, and file size. If
the files do not change, the BM25 index does not need to be rebuilt.

### 6.5 Sparse Score and Top-K

Sparse retrieval uses:

```text
rank_bm25.BM25Okapi
```

If all BM25 scores are zero or negative, sparse returns no hits.

Otherwise, results are sorted by BM25 score descending and capped by:

```text
SPARSE_MAX_HITS = 20
```

So sparse retrieval returns at most 20 document IDs by default.

Important: these 20 sparse hits are not directly sent to the LLM as the final
context. They are evidence used during reranking, filtering, and table selection.

## 7. Dense Retrieval

Dense retrieval is the semantic search path. It is performed with ChromaDB and
Ollama embeddings.

### 7.1 What Dense Retrieval Is

Dense retrieval converts text into numerical vectors called embeddings. Two texts
with similar meaning should have nearby vectors.

This helps when the user does not use the exact database column name.

Example:

```text
Question: Which clients have high spending limits?
```

The schema may not have columns named `clients` or `spending limits`. Dense
retrieval can still find semantically related schema items such as customer
tables or credit-limit columns, depending on metadata text and synonyms.

### 7.2 Dense Query Probe

Dense retrieval does not always embed only the raw question. It builds a compact
probe:

```python
build_dynamic_probe(question, qtype)
```

The probe starts with the original question and may add schema-grounded hints,
such as:

- mentioned table labels,
- ID or display-name columns depending on intent,
- metric columns for aggregate questions,
- date/time columns for temporal questions,
- geography hints such as `country` or `city`.

The probe is intentionally short. By default, it adds at most 16 extra tokens.

This improves embedding search while keeping the original question dominant.

### 7.3 Dense Candidate Count

Dense retrieval calls:

```python
col.query(query_texts=[query_text], n_results=dense_k)
```

The default dense count is computed as:

```text
base_k = max(RETRIEVAL_BASE_K_FLOOR, TOP_K)
```

Default values:

```text
TOP_K = 5
RETRIEVAL_BASE_K_FLOOR = 12
```

Because the floor is 12, normal dense retrieval returns at least 12 candidates,
even though `TOP_K` is 5.

There is one higher-recall case for lookup-name questions. If the system thinks
the user wants an entity display-name column, but sparse evidence has not already
found the preferred name column, dense retrieval expands to:

```text
RETRIEVAL_DENSE_K_FLOOR = 28
```

So dense retrieval returns:

- 12 candidates in normal cases,
- 28 candidates in the special lookup-name recall case.

## 8. Dense and Sparse Together

Dense and sparse retrieval solve different problems.

| Retrieval Type | Strength | Weakness |
| --- | --- | --- |
| Dense | Understands semantic similarity | Can miss exact IDs, codes, or literal sample values |
| Sparse | Matches exact words and aliases | Can miss semantic meaning when words differ |

The system combines them like this:

1. Dense retrieval creates the main candidate pool.
2. Sparse retrieval provides lexical evidence.
3. Dense candidates that also appear in sparse results receive a score bonus.
4. Metadata-based rules adjust the ranking further.

This gives the pipeline both semantic recall and lexical precision.

## 9. Fusion and Reranking

Reranking is the step where candidates are sorted into the final order.

### 9.1 Main Formula

For each dense candidate:

```text
base = -distance
boosted = base + sparse_bonus + column_priority_boost
```

Explanation:

- `distance` comes from Chroma.
- Smaller distance means better semantic match.
- The system uses `-distance`, so better semantic matches receive higher base
  scores.
- `sparse_bonus` is added if the dense candidate was also found by BM25.
- `column_priority_boost` is added or subtracted based on schema metadata and
  question intent.

### 9.2 Sparse Bonus

If a dense candidate matches one of the sparse hit IDs, it receives:

```text
SPARSE_BONUS = 0.75
```

Supported sparse match formats include:

- `table.column`,
- `field_urn`,
- `urn`,
- legacy suffixes like `:table::column`.

This means exact lexical evidence can push a candidate higher, even if it was
not the closest semantic match.

### 9.3 Column Priority Boost

`apply_column_priority_boost()` adjusts scores using metadata and question
signals.

It checks:

- whether the question asks for a name,
- whether the question asks for ID/code/number,
- whether a table is explicitly mentioned,
- the schema-derived table type, such as dimension, transaction, bridge, or
  unknown,
- column roles, such as name, ID, metric, or measure,
- generic column-name patterns, such as suffixes `id`, `code`, `number`, and
  token `name`,
- contact-name cues, so the system does not confuse a company/customer/product
  name with a contact person's name.

Important defaults:

```text
NAME_QUERY_BONUS = 0.3
ID_QUERY_BONUS = 0.4
COL_BOOST_DIM_NAME_EXPLICIT = 0.9
COL_BOOST_DIM_CONTACT_PENALTY = -0.8
COL_BOOST_DIM_NAME_DEFAULT = 0.6
COL_BOOST_DIM_ID_PENALTY = -0.35
COL_BOOST_OTHER_NAME_DEFAULT = 0.5
COL_BOOST_OTHER_ID_PENALTY = -0.3
```

### 9.4 Preferred Name Auto-Injection

There is a very limited auto-injection mechanism for lookup-name questions.

It runs only when:

- the question appears to ask for a display name,
- the preferred name column is known from schema metadata,
- sparse retrieval found that preferred name column,
- dense retrieval did not include it.

Then the preferred name column is fetched from Chroma and inserted into the
scored candidate list with a small score advantage:

```text
RETRIEVAL_AUTO_INJECT_BONUS_DELTA = 0.05
```

This is still schema-driven. The preferred name column is not hardcoded.

### 9.5 Final Reranked Count

After scoring, candidates are sorted by `boosted` score descending:

```python
reranked = sorted(scored, key=lambda x: x["boosted"], reverse=True)
```

Then the list is capped by:

```text
MAX_CTX_DOCS = 12
```

This is the main final retrieval limit.

Only the top 12 reranked candidates become the main evidence for:

- `ranked_cols`,
- table selection,
- `ALLOWED IDENTIFIERS`,
- `CARDS`.

Important nuance:

- Sparse can return 20 hits.
- Dense can return 12 or 28 hits.
- But the final main evidence is capped at 12 reranked candidates.
- The LLM does not receive all sparse and dense raw candidates.

## 10. Column Filtering

After reranking, the system still does not blindly trust every candidate.

Before a column enters `ALLOWED IDENTIFIERS` or `CARDS`, it is checked by:

```python
_is_column_relevant_to_query()
```

The main rules:

- Primary keys and identifying columns are kept.
- Columns explicitly mentioned in the question are kept.
- Column descriptions can help if specific description terms match the question.
- For count queries, unrelated metric/numeric columns can be removed.
- For sum, average, or total queries, relevant metric columns are allowed.
- For lookup queries, display and filter columns are favored, especially when
  their table is mentioned.

This step is important because retrieval can bring back useful but still noisy
columns. Filtering prevents noisy columns from becoming available to the SQL
generator.

## 11. Table Scoring and Table Selection

After candidate columns are reranked, the system scores tables.

The function is:

```python
score_tables_from_reranked()
```

Table scoring decides which tables should appear in the final context.

Signals include:

```text
TABLE_SCORE_SPARSE_HIT = 0.35
TABLE_SCORE_TABLE_MENTION = 1.3
TABLE_SCORE_DESC_TOKEN = 0.3
TABLE_SCORE_MENTION_DIM = 0.9
TABLE_SCORE_MENTION_TXN = 0.7
TABLE_SCORE_MENTION_OTHER = 0.8
TABLE_SCORE_BRIDGE_PENALTY = -0.5
TABLE_SCORE_BRIDGE_NUMERIC_BOOST = 0.3
```

Table mentions are detected from:

- table name,
- singular table form,
- table-name variants,
- schema metadata synonyms.

The base number of top scoring tables is:

```text
MAX_TOP_TABLES = 3
```

Tables must exceed:

```text
RETRIEVAL_TOP_TABLE_MIN_SCORE = 0.0
```

For count-only aggregate questions, the score floor can become stricter to avoid
extra noisy tables. For metric aggregate questions, it can be more permissive so
detail or measure tables are not removed too aggressively.

## 12. Single-Table Lock

Some questions clearly need only one table.

Example:

```text
List customer names from Germany.
```

If the question appears single-table and has no join intent, retrieval can lock
the context to the primary table.

The lock is high-confidence when:

```text
RETRIEVAL_SINGLE_LOCK_MIN_SCORE_GAP = 0.6
```

and:

```text
RETRIEVAL_SINGLE_LOCK_MAX_NON_PRIMARY_SPARSE_HITS = 4
```

Meaning:

- the primary table must lead the second-best table clearly,
- and there must not be too many sparse hits from other tables.

If the lock is low-confidence, retrieval falls back to the primary table plus top
table candidates:

```text
RETRIEVAL_SINGLE_LOCK_FALLBACK_TOPK = 1
```

## 13. Bridge Table Inference

Some questions need multiple tables connected through foreign keys.

Example:

```text
Show customers and their order dates.
```

If selected tables need a connecting table, retrieval can infer bridge tables
from FK metadata.

Bridge inference runs when at least:

```text
RETRIEVAL_BRIDGE_MIN_TABLE_COUNT = 2
```

tables are selected.

FK paths are searched with BFS up to:

```text
RETRIEVAL_FK_PATH_MAX_DEPTH = 3
```

The system extracts FK hints from metadata descriptions. This allows it to add
connecting tables needed for joins without hardcoding table names.

## 14. Final Context Construction

The final retrieval output is stored in:

```python
state["context"]
```

It has three main sections:

```text
### ALLOWED IDENTIFIERS
- table_a: col1, col2, col3
- table_b: col1, col2

### CARDS
- table: table_a | column: col1 | desc: ... | STATS: ...
- table: table_a | column: col2 | desc: ...

### _INTERNAL_HINT
{"alias_map": {...}}
```

These sections are explained below.

### 14.1 `### ALLOWED IDENTIFIERS`

This is the strict whitelist of database identifiers.

It tells the SQL generator:

```text
You may only use these tables and columns.
```

Example:

```text
### ALLOWED IDENTIFIERS
- customers: customerNumber, customerName, country
- orders: orderNumber, customerNumber, orderDate
```

This means the generator may use:

```text
customers.customerNumber
customers.customerName
customers.country
orders.orderNumber
orders.customerNumber
orders.orderDate
```

It should not invent columns outside the whitelist.

This block is built after:

- reranking,
- column relevance filtering,
- table scoring,
- single-table lock, if applicable,
- bridge table inference, if applicable,
- ID/name preservation for simple lookup contexts.

Why it matters:

`ALLOWED IDENTIFIERS` is the main guardrail against invalid SQL identifiers.

### 14.2 `### CARDS`

Cards explain what the retrieved columns mean.

A card looks like this:

```text
- table: customers | column: customerName | desc: Customer full name | STATS: Sample='Atelier graphique'
```

Each card can include:

- table name,
- column name,
- short description,
- optional statistics.

Statistics can include:

- sample values,
- min/max range,
- mean,
- distinct count.

The description is chosen by priority:

1. `metadata.description`,
2. parsed `Description:` text from the dense document,
3. fallback humanized column name.

Default card limits:

```text
RETRIEVAL_CARDS_MAX_DOCS = 15
RETRIEVAL_CARDS_MAX_INJECT = 12
RETRIEVAL_CARDS_DESC_MAX_LEN = 100
RETRIEVAL_STATS_MAX_SAMPLES = 5
```

Important nuance:

The source reranked list is already capped by:

```text
MAX_CTX_DOCS = 12
```

So even though `RETRIEVAL_CARDS_MAX_DOCS = 15`, the normal maximum number of
cards coming directly from reranked retrieval is 12.

Then the system can inject simple cards for allowed columns that are missing from
the card list, up to:

```text
RETRIEVAL_CARDS_MAX_INJECT = 12
```

Why cards matter:

`ALLOWED IDENTIFIERS` tells the generator what columns exist. `CARDS` explain
what those columns mean.

### 14.3 `_INTERNAL_HINT.alias_map`

The alias map is an internal mapping from metadata aliases to real column names.

Example:

```json
{"alias_map": {"customers": {"client name": "customerName"}}}
```

Meaning:

```text
If the user says "client name" for the customers table,
use the real column customers.customerName.
```

Why it matters:

Users often say words that do not exactly match the database column name. The
alias map helps downstream generation and normalization resolve those phrases to
valid schema identifiers.

## 15. Concrete End-to-End Example

Assume the user asks:

```text
Show customer names and their order dates.
```

The retrieval process behaves conceptually like this:

1. The system detects that this is a lookup-style question.
2. Sparse retrieval searches exact words like `customer`, `name`, `order`, and
   `date` over metadata keywords.
3. Dense retrieval embeds a probe based on the question and schema hints.
4. Dense may return candidates such as:
   - `customers.customerName`,
   - `customers.customerNumber`,
   - `orders.orderDate`,
   - `orders.customerNumber`.
5. Sparse evidence boosts candidates that also lexically match.
6. Table scoring keeps relevant tables such as `customers` and `orders`.
7. FK/bridge logic helps preserve join-capable tables if needed.
8. Final context may look like:

```text
### ALLOWED IDENTIFIERS
- customers: customerNumber, customerName
- orders: orderNumber, customerNumber, orderDate

### CARDS
- table: customers | column: customerName | desc: Customer full name
- table: customers | column: customerNumber | desc: Customer identifier
- table: orders | column: orderDate | desc: Date when the order was placed
- table: orders | column: customerNumber | desc: FK to customers.customerNumber

### _INTERNAL_HINT
{"alias_map": {"customers": {"customer name": "customerName"}}}
```

The SQL generator then has enough information to produce a valid join without
seeing unrelated tables and columns.

## 16. Final Retrieval Counts

The default counts are:

| Stage | Default | What It Means |
| --- | ---: | --- |
| Sparse BM25 hits | 20 | Maximum lexical hits from sparse retrieval. |
| Dense candidates, normal case | 12 | Minimum dense candidates because `RETRIEVAL_BASE_K_FLOOR = 12`. |
| Dense candidates, lookup-name case | 28 | Used when the system needs more recall for a preferred display-name column. |
| Final reranked candidates | 12 | Main final evidence used for context. |
| Base top tables | 3 | Default number of high-scoring tables before bridge/temporal additions. |
| Cards from reranked retrieval | Up to 12 in practice | Reranked input is capped by `MAX_CTX_DOCS = 12`. |
| Injected cards | 12 | Additional simple cards for allowed columns missing from cards. |

The most important presentation point:

```text
Sparse and dense retrieval collect candidates.
Reranking chooses the final evidence.
Only the top 12 reranked candidates are the main context source.
```

## 17. Configuration Summary

| Config | Default | Role |
| --- | ---: | --- |
| `TOP_K` | 5 | General top-k knob. Dense still uses the base floor. |
| `RETRIEVAL_BASE_K_FLOOR` | 12 | Minimum dense candidates in normal retrieval. |
| `RETRIEVAL_DENSE_K_FLOOR` | 28 | Dense candidate count for lookup-name recall case. |
| `SPARSE_MAX_HITS` | 20 | Maximum sparse BM25 hits. |
| `SPARSE_BONUS` | 0.75 | Score added when a dense candidate also appears in sparse hits. |
| `MAX_CTX_DOCS` | 12 | Final reranked candidate cap. |
| `MAX_TOP_TABLES` | 3 | Base number of top scoring tables. |
| `RETRIEVAL_CARDS_MAX_DOCS` | 15 | Maximum card source candidates, though reranked input is normally only 12. |
| `RETRIEVAL_CARDS_MAX_INJECT` | 12 | Maximum simple injected cards. |
| `RETRIEVAL_CARDS_DESC_MAX_LEN` | 100 | Maximum description length in cards. |
| `RETRIEVAL_STATS_MAX_SAMPLES` | 5 | Maximum sample values included in stats text. |

## 18. Debugging Output

Debugging is controlled by:

```text
RETRIEVE_DEBUG = 1
SPARSE_DEBUG = 0
```

When `RETRIEVE_DEBUG` is enabled, the system prints:

- original question,
- dense query/probe,
- sparse query,
- sparse tokens,
- query type,
- sparse candidates,
- dense candidates,
- reranked results,
- table scores,
- selected tables,
- final context table/column counts.

When `SPARSE_DEBUG` is enabled, the system prints BM25 top hits and scores.

## 19. What to Emphasize in a Presentation

For a non-technical audience, the clearest explanation is:

```text
The system does not ask the LLM to guess from the whole database.
It first retrieves the most relevant schema pieces.
Dense retrieval understands meaning.
Sparse retrieval catches exact words and values.
Reranking combines both.
Then the SQL generator receives a whitelist and compact column explanations.
```

For a technical audience, emphasize:

- retrieval is hybrid: Chroma dense search plus BM25 sparse search,
- reranking is score-based: `-distance + sparse bonus + metadata boost`,
- final evidence is capped by `MAX_CTX_DOCS = 12`,
- schema context is controlled by `ALLOWED IDENTIFIERS`,
- metadata descriptions and stats are supplied through `CARDS`,
- aliases are passed through `_INTERNAL_HINT.alias_map`,
- behavior is configurable from `config.py`, not hardcoded in retrieval logic.

## 20. Design Intuition

Dense retrieval provides semantic recall. It helps when the user describes a
concept without using the exact database column name.

Sparse retrieval provides lexical precision. It helps with exact column names,
aliases, IDs, codes, sample values, and keywords.

Reranking combines both:

- dense distance gives semantic relevance,
- sparse bonus gives lexical confirmation,
- metadata boosts align ID/name/metric columns with user intent,
- table scoring keeps only relevant tables,
- column filtering prevents noisy columns from entering the whitelist.

The final result is a small, schema-grounded context that helps the SQL generator
produce valid SQL while reducing hallucinated tables and columns.
