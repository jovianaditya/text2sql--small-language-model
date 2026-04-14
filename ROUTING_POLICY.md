# Routing Policy

This document describes the current runtime routing behavior for the Text-to-SQL pipeline.
Primary target: keep behavior stable when new questions are added, without patch-per-question drift.

## 1. Scope and Goal

1. Maximize VA/EX stability first, speed second.
2. Keep deterministic IR limited to the current safe subset.
3. Let the router compute an LLM fallback candidate (`3B` or `7B`) without committing to an LLM call yet.
4. Let the draft phase try IR first, and select the actual LLM model only when IR is not used.
5. Keep decisions auditable from logs.

Assumptions for this document:

1. This file reflects the current default runtime behavior from `agents.py` and `config.py`.
2. The examples and route map are aligned to the current default code path; a checked-in `run_full.log` can lag until the benchmark is rerun.
3. If env knobs are overridden, runtime behavior can differ from this document.

## 2. Key Terms

1. `IR`: Intermediate Representation lane (`intent_ir` + `sql_planner`) without free-form SQL drafting.
2. `LLM fallback candidate`: the router-computed fallback model before the IR gate resolves (`qwen2.5:3b-instruct` or `qwen2.5:7b-instruct`).
3. `final lane`: the actual lane that produced final SQL (`IR`, `3B`, or `7B`).
4. `router hard flag`: a router signal that can block IR; it is broader than the subset of conditions that escalate to `7B`.
5. `7B escalation reason`: an entry in `router_decision.escalation_reasons`; any non-empty list forces the `7B` lane.
6. `context_table_count`: number of tables present in retrieval context cards.
7. `table_count`: routed/mentioned table scope used by the IR gate.
8. `critical gate failure`: parse/identifier/join/fanout failure that must not be executed.

## 3. Routing Algorithm (Actual Runtime Order)

1. Question Guard rejects non-SQL questions.
2. Retrieval builds context and allowed identifiers.
3. Router detects tables, computes signals, and stores an LLM fallback candidate (`3B` or `7B`).
4. The draft phase then builds `intent_ir` and runs the IR gate before any LLM model is selected for execution.
5. If IR is enabled, eligible, plannable, and the planner returns SQL, the draft source becomes `IR` and no LLM model is invoked in the draft phase.
6. If IR is gated off or the planner fails, the draft phase selects the actual LLM model from the router fallback candidate and runs the LLM lane.
7. Refiner + acceptance gate always run before execution.
8. On the IR path, a whitelisted non-critical semantic mismatch triggers bounded semantic recovery in this order:
   - deterministic IR semantic repair
   - LLM semantic fallback using the router fallback candidate
   - one optional semantic escalation to `7B` if the router candidate was `3B`
9. On the LLM path, a whitelisted non-critical semantic mismatch at the final pre-exec acceptance gate triggers bounded semantic retry instead of immediate execution.
10. If the gate has a critical failure, runtime does bounded rescue.
11. If the query still has a critical failure, runtime fail-closes (`FINAL_GATE_BLOCKED`) and does not execute unsafe SQL.

Observable logging behavior:

1. Logs show the router fallback candidate first.
2. Logs then show the IR gate result.
3. If IR is not used, the drafter logs the actual fallback selection before LLM inference.
4. If IR draft SQL hits the semantic-recovery path, logs show `Semantic repair` and, when needed, `Semantic fallback`.
5. On the LLM path, logs show `[LLM] Semantic retry: not needed`, `eligible`, or `skipped (non-whitelisted)` at the final pre-exec acceptance gate.
6. Final logs may therefore show `LLM Candidate: 3b -> Final: ir_planner` for clean IR, or a `3B`/`7B` final model if bounded semantic fallback or bounded LLM semantic retry replaced the original draft SQL.

## 4. IR Gate Policy (Current Runtime)

Default runtime settings:

1. `AGENT_IR_ENABLED=1`
2. `AGENT_IR_MIN_CONFIDENCE=0.80`
3. `AGENT_IR_MULTI_TABLE_MIN_CONFIDENCE=0.90`
4. `AGENT_IR_MAX_TABLES=2`
5. `AGENT_IR_ALLOW_MULTI_TABLE=1`
6. `AGENT_IR_ALLOW_SIMPLE_HAVING=1`

Primary routing principle:

1. Route by supported query family first, not by raw retrieval breadth alone.
2. `context_table_count` is a secondary signal, not the primary family classifier.
3. The IR gate is conservative: tightly resolved single-table top-N is in the active safe subset, and top-N resolution must come from schema-grounded lexical matching rather than phrase-specific rules; same-table date-column comparison remains stricter and still prefers the LLM lane unless a safe compare filter is actually formed.

IR-preferred families:

1. Low-risk single-table lookup/filter/distinct lookup inside the planner-safe subset.
2. Single-table lookup with schema-grounded numeric column resolution, deterministic top-N ordering, and `LIMIT`.
3. Experimental single-table lookup with resolved same-table date-column comparison only when a safe compare filter is actually formed.
4. Single-table aggregate/group/having patterns inside the planner-safe subset.
5. Compact planner-safe aggregate families where effective scope is `<= AGENT_IR_MAX_TABLES`.
6. Deterministic regex templates only.
7. Supported percentage intent only when the IR metric is `pct_non_null`.
8. Simple 2-table deterministic aggregate families that remain planner-safe.

LLM-preferred families (IR is gated off by current runtime):

1. Winner-row and ordinal-rank families.
2. Window aggregate / window rank families.
3. Any subquery intent, including simple single-table scalar subquery and simple EXISTS/NOT EXISTS wording.
4. Set-operation intent.
5. Exclusion / anti-join semantics.
6. Dual-fact rollups and other fanout-risk patterns.
7. Multi-table `COUNT(DISTINCT ...)` threshold patterns.
8. Projection null-fill lookup patterns (for example, `show ... state as 'N/A' when missing`).
9. Unresolved column-to-column comparison lookups where no safe same-table IR compare filter is formed.
10. Scalar extrema-date lookup patterns (for example, `on the latest date` lookup shape).
11. Offset-window phrasing (for example, `previous`, `next`, `prior`, `following` with `ordered by`).
12. Unresolved top-N lookup patterns where `LIMIT` is found but no high-confidence schema-grounded numeric ordering is formed.
13. Open / broad regex semantics outside the deterministic regex template family.

IR gate mechanics:

1. Effective `qtype` is usually the router `qtype`.
2. Exception: for planner-safe distinct lookup, IR may keep `lookup` from `intent_ir`.
3. `table_count` is computed as:
   - `1` for low-risk single-table lookup
   - `max(len(mentioned_tables), len(router_tables), 1)` for compact IR-safe families
   - `max(len(mentioned_tables), len(router_tables), 1)` for lookup-subject-focused families
   - otherwise `max(len(mentioned_tables), len(router_tables), context_table_count)`
4. `dynamic_min_conf` starts at `0.80`.
5. If `table_count > 1`, `dynamic_min_conf` becomes `max(current, 0.90)`.
6. If the IR gate passes but `build_sql_from_intent(...)` fails, runtime falls back to the router-computed LLM fallback candidate.
7. If the IR draft passes planning but later hits a whitelisted semantic-only acceptance mismatch, runtime can still hand off to bounded semantic recovery before execution.

Exact IR gate reason labels currently emitted by runtime:

1. `unsupported_qtype_for_ir:<qtype>`
2. `open_regex_prefers_llm`
3. `table_count_exceeds_ir_max:<n>><max>`
4. `context_table_count_exceeds_ir_max:<n>><max>`
5. `multi_table_ir_disabled`
6. `router_hard:dual_fact_rollup`
7. `router_hard:winner_row`
8. `router_hard:exclusion`
9. `router_hard:window_aggregate`
10. `router_hard:explicit_rank`
11. `router_hard:ordinal_rank`
12. `router_hard:subquery_intent`
13. `router_hard:simple_scalar_subquery`
14. `having_like_threshold_unresolved`
15. `having_like_threshold_intent_ir_disabled`
16. `unsupported_simple_having_shape` (or a helper-specific reason string for unsupported simple-HAVING shape)
17. `percentage_intent_ir_disabled`
18. `projection_null_fill_ir_unsupported`
19. `column_to_column_compare_ir_unsupported`
20. `scalar_extrema_date_subquery_ir_unsupported`
21. `offset_window_ir_unsupported`
22. `top_n_lookup_ir_unresolved`
23. `set_operation_ir_unsupported`
24. `unsupported_ir_filter_op:<op>`
25. `unsupported_ir_metric:<agg>`

Important interpretation:

1. IR is intentionally not universal.
2. Many advanced single-table questions are still LLM-routed by design.
3. A router `SIMPLE -> 3B` fallback candidate does not force an LLM call; IR can still terminate the flow before any LLM is selected.
4. IR hard flags are broader than `7B` escalation rules.

## 5. 3B vs 7B Policy (Current Runtime)

### 5.1 Use 3B

Use `3B` when all below are true:

1. IR is not selected.
2. `router_decision.escalation_reasons` is empty.
3. The actual LLM `target_model` is assigned in `draft_sql_agent` only after the IR gate is skipped or fails.
4. This includes:
   - ordinary single-table lookup/filter/aggregate cases
   - single-table winner-row / ordinal-rank cases
   - single-table explicit-rank and window-aggregate cases
   - simple single-table scalar subquery cases
   - simple single-table set-operation cases
   - simple EXISTS/NOT EXISTS cases when table scope is `<= 2`
   - simple multi-table cases within `AGENT_ROUTER_SIMPLE_MULTI_TABLE_MAX_TABLES` when no explicit escalation reason fires

### 5.2 Use 7B

Use `7B` when `router_decision.escalation_reasons` is non-empty.

Exact current escalation labels:

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

Operational nuance:

1. `7B` escalation is driven by explicit escalation reasons, not by router hard flags alone.
2. Single-table `WinnerRow=True`, `OrdinalRank=True`, `ExplicitRank=True`, or `WindowAgg=True` can still remain on `3B`.
3. Cross-table exclusion reaches `7B` only when the current exclusion detector actually sets `has_exclusion=True`.
4. Simple multi-table patterns remain on `3B` and log `Multi-table simple pattern detected: ... -> keeping 3B`.

## 6. Decision Matrix

|                       Condition                       |        Final lane       |
|-------------------------------------------------------|-------------------------|
| Guard reject                                          | reject                  |
| IR gate passes and planner returns SQL                | IR                      |
| IR not selected and `escalation_reasons` is empty     | 3B                      |
| IR not selected and `escalation_reasons` is non-empty | 7B                      |
| Gate critical fail after draft/refine                 | rescue then fail-closed |

## 7. Canonical Examples

### Example A: Router `3B` Fallback Candidate, Final `IR`

Question:
`Find employees whose extension starts with x and has exactly three digits.`

Expected route:

1. Router computes `SIMPLE -> 3B` as the LLM fallback candidate.
2. IR gate passes before any LLM model is selected for execution.
3. Final lane becomes `IR`.

Expected SQL shape:

```sql
SELECT employeeNumber, firstName, lastName, extension
FROM employees
WHERE extension REGEXP '^x[0-9]{3}$';
```

### Example B: Top-N Lookup Now Uses `IR`

Question:
`Show the 10 products with the highest stock levels.`

Expected route:

1. Router computes `SIMPLE -> 3B` as the LLM fallback candidate.
2. IR resolves the numeric sort key from schema-grounded lexical matching, then applies `LIMIT` deterministically.
3. Final lane becomes `IR` before any LLM model is selected.

Expected SQL shape:

```sql
SELECT productCode, productName, quantityInStock
FROM products
ORDER BY quantityInStock DESC
LIMIT 10;
```

### Example C: Single-Table Winner-Row Still Stays `3B`

Question:
`For each customer id, return the payment record with the largest payment amount.`

Expected route:

1. Router sets `WinnerRow=True`.
2. Table scope remains single-table.
3. IR is blocked.
4. Final lane stays `3B` because no `7B` escalation reason fires.

### Example D: Multi-Table Hard Pattern Escalates to `7B`

Question:
`For each customer, return the order with the highest total order amount.`

Expected route:

1. Multi-table winner-row pattern.
2. Router emits `winner_row_subquery_multi_table`.
3. Final lane is `7B`.

Expected SQL shape:

```sql
SELECT customerNumber, orderNumber, order_total
FROM (
  SELECT o.customerNumber, o.orderNumber,
         SUM(od.quantityOrdered * od.priceEach) AS order_total,
         ROW_NUMBER() OVER (
           PARTITION BY o.customerNumber
           ORDER BY SUM(od.quantityOrdered * od.priceEach) DESC
         ) AS rn
  FROM orders o
  JOIN orderdetails od ON o.orderNumber = od.orderNumber
  GROUP BY o.customerNumber, o.orderNumber
) t
WHERE rn = 1;
```

## 8. Acceptance Gate Policy

Acceptance gate runs for both IR and LLM outputs.

Critical reasons (must block or rescue):

1. `parse_error`
2. `hard_identifier`
3. `ambiguous_unqualified`
4. `group_projection_mismatch`
5. `invalid_join_path`
6. `fanout_risk_dual_fact`

Non-critical reasons:

1. `semantic_contract_mismatch`

Policy:

1. Critical failure triggers retry/rescue logic.
2. `missing_order_by` is evaluated against final-result ordering intent, not raw sort words alone. Window-ranking phrasing such as `rank ... in descending order` does not by itself require a top-level `ORDER BY`, a top-level `ORDER BY` on a set-operation root (for example, `UNION ... ORDER BY`) satisfies the contract, and projection phrases such as `first names` / `last names` do not count as ordering intent.
3. On the LLM path, `semantic_contract_mismatch` does not immediately force execution when all semantic failures are inside the current safe whitelist:
   - `missing_order_by`
   - `missing_limit`
   - `unparsed_limit`
   - `limit_mismatch:*`
   - `missing_agg_count`
   - `missing_agg_sum`
   - `missing_agg_avg`
4. LLM semantic retry order is:
   - bounded regeneration starting from the current draft model (`3B` or `7B`)
   - one targeted retry with the same model if the first regenerated SQL still fails the gate
   - one optional escalation to `7B` only when the original retry model was `3B`
5. On the IR path, `semantic_contract_mismatch` does not immediately force execution. Runtime first tries bounded semantic recovery when all semantic failures are inside the current safe whitelist:
   - `missing_order_by`
   - `missing_limit`
   - `limit_mismatch:*`
   - `missing_agg_count`
   - `missing_agg_sum`
   - `missing_agg_avg`
6. IR semantic recovery order is:
   - deterministic rebuild from `intent_ir`
   - LLM semantic fallback using the router fallback candidate
   - one optional semantic escalation to `7B` if the router fallback candidate was `3B`
7. If bounded semantic recovery or bounded LLM semantic retry still does not clear the mismatch and the failure remains non-critical, runtime can continue with warning.

## 9. Operational Knobs

IR knobs:

1. `AGENT_IR_ENABLED`
2. `AGENT_IR_MIN_CONFIDENCE`
3. `AGENT_IR_MAX_TABLES`
4. `AGENT_IR_ALLOW_MULTI_TABLE`
5. `AGENT_IR_MULTI_TABLE_MIN_CONFIDENCE`
6. `AGENT_IR_ALLOW_SIMPLE_HAVING`
7. `AGENT_IR_DEBUG`

Router knobs:

1. `AGENT_ROUTER_ALLOW_SIMPLE_MULTI_TABLE_3B`
2. `AGENT_ROUTER_SIMPLE_MULTI_TABLE_MAX_TABLES`
3. `AGENT_ROUTER_ESCALATE_ON_MULTI_TABLE`
4. `AGENT_ROUTER_ESCALATE_ON_WINNER_ROW`
5. `AGENT_ROUTER_ESCALATE_ON_EXCLUSION`
6. `AGENT_ROUTER_ESCALATE_ON_WINDOW_AGGREGATE`
7. `AGENT_ROUTER_ESCALATE_ON_EXPLICIT_RANK`
8. `AGENT_ROUTER_ESCALATE_ON_SUBQUERY_INTENT`
9. `AGENT_ROUTER_ALLOW_SIMPLE_SINGLE_TABLE_SUBQUERY_3B`

## 10. Route Map (Current Default Runtime for the 50-Case Dev Benchmark)

This map reflects the current default final-lane allocation for the 50-case dev benchmark.

Notes:

1. The checked-in `run_full.log` can lag this policy until the benchmark is rerun. The latest `run_easy.log` confirms `E2 -> IR` while `E13` still remains on `3B`.
2. `IR` means final lane is deterministic planner, even if the router fallback candidate was `3B`.
3. The easy, medium, and hard tables below reflect the current code path.

### 10.1 Easy (E1-E13)

| ID  | Current Lane |                                               Note                                             |
|-----|--------------|------------------------------------------------------------------------------------------------|
| E1  | 3B           | Single-table equality filter + simple sort                                                     |
| E2  | 3B           | Top-N lookup uses IR when schema-grounded numeric ordering + `LIMIT` resolve deterministically |
| E3  | 3B           | Single-table year filter + descending date sort                                                |
| E4  | 3B           | Single-table numeric threshold + explicit sort                                                 |
| E5  | 3B           | Single-table null / empty filter                                                               |
| E6  | 3B           | Single-table prefix pattern (`LIKE 'prefix%'`)                                                 |
| E7  | 3B           | Single-table `IN (...)` filter on FK-like code column                                          |
| E8  | 3B           | Single-table deterministic regex pattern                                                       |
| E9  | 3B           | Single-table categorical `IN (...)` filter                                                     |
| E10 | 3B           | Single-table `BETWEEN` numeric filter                                                          |
| E11 | 3B           | Single-table lookup with top-level `SELECT DISTINCT`                                           |
| E12 | 3B           | Projection null-fill lookup is outside current IR shape                                        |
| E13 | 3B           | Current benchmark phrasing still falls back to LLM (`column_to_column_compare_ir_unsupported`) |

### 10.2 Medium (M1-M20)

| ID  | Current Lane |                          Note                            |
|-----|--------------|----------------------------------------------------------|
| M1  | 3B           | Single-table aggregate + month grouping                  |
| M2  | 3B           | CASE-band aggregation                                    |
| M3  | 3B           | Winner-row single-table (payments)                       |
| M4  | 3B           | Single-table group + min/count                           |
| M5  | 3B           | Group + HAVING aggregate filter                          |
| M6  | 3B           | Simple scalar subquery single-table                      |
| M7  | 3B           | Supported percentage metric pattern                      |
| M8  | 3B           | Group + HAVING count                                     |
| M9  | 3B           | Single-table group + order-by aggregate                  |
| M10 | 3B           | Window rank single-table                                 |
| M11 | 3B           | Second-highest single-table winner-row                   |
| M12 | 3B           | Window cumulative single-table                           |
| M13 | 3B           | Window count partition single-table                      |
| M14 | 3B           | Window avg partition single-table                        |
| M15 | 3B           | Window percentage-over-total single-table                |
| M16 | 3B           | Simple single-table set operation stays in LLM lane      |
| M17 | 3B           | Single-table `COUNT(DISTINCT ...)` grouped by order year |
| M18 | 3B           | Offset window (`LAG`) single-table                       |
| M19 | 3B           | Scalar extrema-date subquery single-table                |
| M20 | 3B           | Single-table `COUNT(DISTINCT ...)` + HAVING              |

### 10.3 Hard (H1-H17)

| ID  | Current Lane |                                 Note                                       |
|-----|--------------|----------------------------------------------------------------------------|
| H1  | 3B           | Multi-table aggregate, planner-safe family                                 |
| H2  | 3B           | Simple 2-table existence + grouped count                                   |
| H3  | 3B           | Simple 2-table join + grouped sum                                          |
| H4  | 3B           | Simple multi-join + grouped sum                                            |
| H5  | 3B           | Null-preserving multi-table rollup stays in LLM lane but does not escalate |
| H6  | 7B           | Explicit ranking in multi-table context                                    |
| H7  | 7B           | Multi-table winner-row pattern                                             |
| H8  | 7B           | Multi-table winner-row / derived-subquery pattern                          |
| H9  | 7B           | Complex subquery intent                                                    |
| H10 | 7B           | Multi-table winner-row pattern                                             |
| H11 | 3B           | Current phrasing stays in 3B (no router escalation reason fired)           |
| H12 | 7B           | Multi-table complex subquery intent                                        |
| H13 | 7B           | Multi-table winner-row + exclusion                                         |
| H14 | 7B           | Dual-fact rollup fanout-risk pattern                                       |
| H15 | 7B           | Cross-table exclusion                                                      |
| H16 | 7B           | Multi-table `COUNT(DISTINCT ...)` threshold                                |
| H17 | 7B           | Multi-table winner-row per group                                           |

### 10.4 Current Expected Split (50-Case Dev Benchmark)

1. Final `IR`: 21/50
2. Final `3B`: 18/50
3. Final `7B`: 11/50
4. Router fallback candidate `SIMPLE`: 39/50
5. Router fallback candidate `COMPLEX`: 11/50

## 11. Benchmark and Change Protocol

When changing routing behavior:

1. Update this file first.
2. Add targeted regression cases for changed rules.
3. Run `easy`, `medium`, `hard`, then the full set.
4. Report:
   - VA and EX
   - final-lane split (`IR`, `3B`, `7B`)
   - router fallback-candidate split (`SIMPLE`, `COMPLEX`)
   - gate hard-error rate
   - average latency

Acceptance criteria for a routing patch:

1. No regression on the benchmark.
2. Reduced hard-error frequency on new questions.
3. No uncontrolled increase in `7B` usage or latency.
4. If the 50-case dev route mix moves materially away from the current baseline (`IR=21`, `3B=18`, `7B=11`), treat that as a deliberate policy change and document why.
