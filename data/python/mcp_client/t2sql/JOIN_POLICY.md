# Join Policy

This document describes the current runtime join-selection behavior for the Text-to-SQL pipeline.
Primary target: keep join decisions auditable, schema-driven, and stable when new questions are added.

Assumptions for this document:

1. This file reflects the current default behavior in `data/python/mcp_client/t2sql/agents.py`, `helpers_main.py`, and `config.py`.
2. The policy below describes runtime logic, not an aspirational design.
3. If environment flags are overridden, runtime behavior can differ from this document.

## 1. Scope

This document explains:

1. How the system determines the `FROM` table and join order.
2. How the system chooses `INNER JOIN` vs `LEFT JOIN`.
3. How connector-table and FK-path guidance is added.
4. When join hints are sent to the LLM.
5. What `simple` vs `non-simple` means in relation to join hints.

## 2. High-Level Flow

Actual runtime order:

1. Retrieval identifies relevant tables and evidence.
2. The router determines `primary_table` and `primary_table_reason`.
3. The router computes `complexity` (`simple` or `complex`) and stores `router_decision`.
4. If the runtime uses the LLM drafting path, `make_generation_prompt(...)` builds the LLM prompt.
5. During prompt build, join guidance may be appended as prompt constraints:
   - `TABLE ORDER (CRITICAL)`
   - `JOIN TYPE (CRITICAL)`
   - `CONNECTOR PATH RULE (CRITICAL)`
   - `JOIN KEY HINTS (CRITICAL)`
   - `JOIN PATH RULE (CRITICAL)`

Important distinction:

1. Join hints are prompt-time guidance for the LLM.
2. They do not directly rewrite SQL by themselves.
3. They are only relevant when the system actually goes through the LLM prompt path.

## 3. Primary Table Determination

When more than one table is relevant, the router picks a `primary_table` using this priority order:

1. Explicit lead subject in the question.
   - Example shape: `Show customers ...`
   - Goal: anchor the query on the entity the user is explicitly asking to list.
2. `For each X` pattern.
   - Example shape: `For each customer ...`
   - Goal: use the grouping subject as the `FROM` anchor.
3. `X from Y` / `X in Y` pattern.
   - Goal: anchor on the container/entity named by the question.
4. `All X that ...` pattern.
   - Goal: anchor on the entity being filtered.
5. Table with the most direct evidence from retrieval.
6. First relevant table as fallback.

Single-table queries are trivial:

1. If only one relevant table is present, that table becomes `primary_table`.

Operational effect:

1. `primary_table` becomes the starting table in `TABLE ORDER (CRITICAL)`.
2. `primary_table_reason` is appended to the hint for log visibility.

## 4. What Counts as `Non-Simple`

In current runtime, `non-simple` effectively means:

1. `router_decision.complexity == "complex"`
2. This happens when `should_escalate = bool(escalation_reasons)`

Important nuance:

1. Multi-table does not automatically mean `complex`.
2. Simple multi-table queries are allowed by default.
3. Current default config allows simple multi-table up to `AGENT_ROUTER_SIMPLE_MULTI_TABLE_MAX_TABLES=3`.

Queries become `complex` when at least one escalation reason is present, including:

1. `multi_table_forced`
2. `multi_table_too_wide:<n>><max>`
3. `winner_row_subquery_multi_table`
4. `cross_table_exclusion`
5. `window_aggregate_multi_table`
6. `explicit_rank_function_multi_table`
7. `subquery_intent_complex`
8. `multi_table_distinct_threshold`
9. `set_operation`
10. `multi_fact_rollup_fanout_risk`

This matters because the current `TABLE ORDER` hint is attached only for `complexity != "simple"` plus valid state and `primary_table`.

## 5. When the LLM Receives Join Hints

The LLM receives join hints only on the LLM prompt path.

Actual runtime behavior:

1. `agents.py` calls `make_generation_prompt(ctx, question, decision, state)` before LLM inference.
2. `make_generation_prompt(...)` decides whether join guidance should be appended.

### 5.1 `TABLE ORDER (CRITICAL)`

This hint is added only when all below are true:

1. `complexity != "simple"`
2. `state` exists
3. `primary_table` exists

The resulting hint shape is:

```text
TABLE ORDER (CRITICAL): Start your query with FROM <primary_table>, then <JOIN table1>, <JOIN table2> ...
```

Notes:

1. If there are no `other_tables`, the hint still anchors the query on `FROM <primary_table>`.
2. The hint only describes table order and join type.
3. It does not tell the LLM to use subqueries, pre-aggregation, or derived tables.

### 5.2 `JOIN TYPE (CRITICAL)`

This guidance is appended immediately after `TABLE ORDER` and explains why the chosen join type matters.

It is only added inside the same non-simple + state + `primary_table` block.

### 5.3 `CONNECTOR PATH RULE (CRITICAL)`

This guidance is added when a target table is not directly adjacent to the `primary_table` in the FK graph.

Example policy:

1. If the shortest FK path from `customers` to `orderdetails` is `customers -> orders -> orderdetails`,
2. The prompt tells the LLM to reach `orderdetails` via `orders`,
3. And explicitly warns not to invent a direct join between `customers` and `orderdetails`.

### 5.4 `JOIN KEY HINTS (CRITICAL)` and `JOIN PATH RULE (CRITICAL)`

These are added when FK-consistent join edges can be built from runtime state.

Purpose:

1. Restrict the LLM to known FK join predicates.
2. Prevent invented join conditions.
3. Force the listed FK edges to be used as one connected path.

## 6. How Table Order Is Built

Once `primary_table` is chosen:

1. `other_tables` are taken from `state["router_tables"]`
2. `primary_table` is removed from that list
3. The remaining tables are ordered by FK distance from `primary_table`

This means the hint is schema-aware:

1. The system prefers a table order that follows actual FK connectivity.
2. It does not preserve retrieval order blindly.

## 7. How Join Type Is Chosen

Join type is decided per table, not by one global rule only.

The current priority is:

1. Include-zero + exclusion target:
   - If the question wants to preserve primary rows even when related rows do not exist,
   - and a table is the exclusion target,
   - use `LEFT JOIN` for that target.

2. Forced required inner chain:
   - If `_should_force_required_inner_join_chain(...)` returns true,
   - use `INNER JOIN` for every supporting table in the path.

3. Mixed intent on the same table:
   - If a table is simultaneously treated as required and excluded in different slices,
   - use `INNER JOIN` and keep exclusion logic separate with anti-join semantics.

4. Pure exclusion target:
   - Use `LEFT JOIN` for the exclusion-target table.

5. Existence target:
   - If the question explicitly requires existence such as `with`, `having`, `placed`, `paid`, `at least one`,
   - use `INNER JOIN` for that target table.

6. General filtering intent:
   - If the question clearly filters to entities that must have matching related rows,
   - use `INNER JOIN`.

7. Exclusion + aggregate positive slice:
   - In some aggregate exclusion questions, non-target support tables use `INNER JOIN`.

8. Default:
   - Use `LEFT JOIN` to preserve rows from the primary table.

Default fallback rule:

1. If the system chooses `LEFT JOIN`, filters on the joined table should stay in the `ON` clause, not the `WHERE` clause.

## 8. Forced Required Inner Chain

The current helper `_should_force_required_inner_join_chain(...)` returns `True` only for a narrow set of patterns.

It returns `False` when:

1. There is only one table.
2. The question wants include-zero semantics.

It returns `True` when one of these is true:

1. Router escalation reason contains `multi_table_distinct_threshold`
2. The question asks for a latest-year scope with aggregation

Practical effect:

1. These patterns are more fragile if the LLM weakens the join chain to `LEFT JOIN`.
2. The prompt therefore forces an `INNER JOIN` chain for the metric path.

## 9. Connector Path Policy

The connector-path rule is based on the shortest FK path in the schema graph.

Behavior:

1. Build an undirected FK graph from schema relationships.
2. Find the shortest path from `primary_table` to each ordered target table.
3. If the path length is greater than 2, then there is an intermediate connector table.
4. Add a prompt rule telling the LLM to reach the target through that connector.

This prevents mistakes like:

1. Skipping `orders` and joining `customers` directly to `orderdetails`
2. Inventing direct joins that do not exist in the schema

## 10. Join Hint Philosophy

Current policy for join hints is intentionally limited:

1. Tell the LLM the correct table order.
2. Tell the LLM the correct `INNER JOIN` vs `LEFT JOIN` shape.
3. Tell the LLM the allowed FK path and connector tables.
4. Do not tell the LLM to use specific SQL structures such as:
   - pre-aggregate first
   - use a derived table
   - use a subquery before joining

Rationale:

1. Table-order and join-type hints are generally stable and robust.
2. Over-specifying implementation details in the hint can push the LLM into brittle SQL shapes.

## 11. Observability in Logs

The following log lines make join policy auditable:

1. `[ROUTER] Primary Table: ...`
2. `[ROUTER] Multi-table simple pattern detected: ... -> keeping 3B`
3. `[ROUTER] Escalating to 7B. reasons=[...]`
4. `[PROMPT DEBUG] complexity=..., state provided, primary_table=...`
5. `[PROMPT DEBUG] Adding hint: TABLE ORDER (CRITICAL): ...`

Interpretation:

1. If there is no LLM prompt path, you will not see `TABLE ORDER (CRITICAL)`.
2. If the query remains `simple`, current runtime does not append this table-order hint.
3. If the query is `complex` and `primary_table` exists, you should expect join guidance in the prompt.

## 12. Canonical Summary

Short version:

1. The router first decides relevant tables and the `primary_table`.
2. The router then decides whether the query is `simple` or `complex`.
3. If the system goes through the LLM path and the query is `complex`, prompt-time join hints are added.
4. Those hints are schema-driven:
   - `FROM` anchor = `primary_table`
   - table order = FK-distance-aware order
   - join type = derived from existence, exclusion, include-zero, and special hard patterns
   - path safety = connector path + FK join-key hints
5. The hint is meant to constrain join behavior, not to prescribe full SQL implementation details.
