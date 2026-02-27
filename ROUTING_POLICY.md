# Routing Policy

This document defines the routing policy for the Text-to-SQL pipeline.
Primary target: keep behavior stable when new questions are added, without patch-per-question drift.

## 1. Scope and Goal

1. Maximize VA/EX stability first, speed second.
2. Use deterministic IR only in clearly safe scope.
3. Use 3B as default LLM lane for non-hard cases.
4. Use 7B only for hard semantic shapes.
5. Keep decisions auditable from logs.

Assumptions for this document:

1. This document is normative for `IR Safe Subset v1` policy.
2. Runtime behavior can temporarily differ until all knobs/patches are fully aligned.
3. If env knobs are overridden, runtime behavior can differ from this document.

## 2. Key Terms

1. `IR`: Intermediate Representation lane (`intent_ir` + `sql_planner`) without free-form SQL drafting.
2. `IR confidence`: score from intent extraction completeness and grounding quality.
3. `context_table_count`: number of tables present in retrieval context cards.
4. `table_count`: routed/mentioned table scope used by planner gate.
5. `hard pattern`: query intent with high error risk for small model or deterministic lane.
6. `critical gate failure`: parse/identifier/join/fanout failure that must not be executed.

## 3. Routing Algorithm (Deterministic Order)

1. Question Guard rejects non-SQL questions.
2. Retrieval builds context and allowed identifiers.
3. Router detects tables and hard-pattern signals.
4. IR gate runs first.
5. If IR is eligible, plannable, and planner can build SQL, use IR SQL.
6. If IR is not eligible or planner fails to build SQL, use LLM lane.
7. LLM lane selection:
   - 3B for non-hard patterns.
   - 7B for hard patterns.
8. Refiner + acceptance gate always run before execution.
9. If gate has critical failure, do bounded rescue.
10. If still critical, fail-closed (`FINAL_GATE_BLOCKED`), do not run unsafe SQL.

## 4. IR Eligibility Policy (IR Safe Subset v1)

Default target settings:

1. `AGENT_IR_ENABLED=1`
2. `AGENT_IR_MIN_CONFIDENCE=0.80`
3. `AGENT_IR_MULTI_TABLE_MIN_CONFIDENCE=0.90`
4. `AGENT_IR_MAX_TABLES=2`
5. `AGENT_IR_ALLOW_MULTI_TABLE=1`

IR is allowed only when all conditions pass:

1. `qtype` is `lookup` or `aggregate`.
2. `confidence >= dynamic_min_conf`.
3. `table_count <= AGENT_IR_MAX_TABLES`.
4. `context_table_count <= AGENT_IR_MAX_TABLES`.
5. No hard router flags.
6. No dual-fact rollup fanout risk.
7. Percentage intent is allowed only when IR metric supports it (`pct_non_null`).
8. IR operators/metrics stay inside planner-safe subset:
   - filter ops: `eq|in|gt|gte|lt|lte|between|year_eq|like_prefix|regexp|is_null_or_empty`
   - metric ops: `count|sum|avg|min|max|pct_non_null`
9. If table scope is 2 tables, FK path must be simple and plannable.

Implementation notes:

1. `table_count = max(len(mentioned_tables), len(router_tables), context_table_count)`.
2. `dynamic_min_conf`:
   - `0.80` for single-table scope.
   - `0.90` for 2-table scope.
3. IR hard router flags are:
   - `is_winner_row`
   - `has_exclusion`
   - `is_window_aggregate`
   - `is_explicit_rank_function`
   - `is_ordinal_ranking`
   - `is_subquery_intent`
4. If planner cannot produce valid SQL, route falls back to LLM lane.
5. Retrieval single-table lock should prefer preserving single-table context when confidence is acceptable.

IR is gated off when any condition fails.

Important interpretation:

1. IR is intentionally not universal; it covers safe deterministic subset only.
2. Hard semantic shapes remain in LLM lane by design.
3. Policy target keeps IR coverage moderate so LLM contribution remains measurable.

## 5. 3B vs 7B Policy

### 5.1 Use 3B

Use 3B when all below are true:

1. IR not selected.
2. No hard flags.
3. Query is one of:
   - single-table lookup/filter/aggregate/pattern
   - simple join up to `AGENT_ROUTER_SIMPLE_MULTI_TABLE_MAX_TABLES`
   - simple single-table scalar subquery
   - simple EXISTS/NOT EXISTS without nested complexity and table scope `<= 2`

### 5.2 Use 7B

Use 7B when any hard flag is true (with corresponding `AGENT_ROUTER_ESCALATE_ON_*` knob enabled, default `1`):

1. Winner-row + multi-table.
2. Cross-table exclusion semantics.
3. Window aggregate/rank in multi-table context.
4. Explicit ranking intent in multi-table context.
5. Complex subquery intent.
6. Dual-fact rollup fanout risk (`orders` + `payments` style metrics mixed in one grouped output).

## 6. Decision Matrix

|                             Condition                                 |          Lane           |
|-----------------------------------------------------------------------|-------------------------|
| Guard reject                                                          | reject                  |
| IR eligible (safe subset v1) + plannable + planner returns SQL        | IR                      |
| IR eligible but planner fails / planner safety rejects                | LLM fallback            |
| IR not eligible + no hard flag                                        | 3B                      |
| IR not eligible + hard flag                                           | 7B                      |
| Gate critical fail after draft/refine                                 | rescue then fail-closed |

## 7. Canonical Examples

### Example A: IR path

Question:
`Find employees whose extension starts with x and has exactly three digits.`

Expected route:

1. Single table (`employees`)
2. High IR confidence
3. IR selected

Expected SQL shape:

```sql
SELECT employeeNumber, firstName, lastName, extension
FROM employees
WHERE extension REGEXP '^x[0-9]{3}$';
```

### Example B: 3B path (simple multi-table)

Question:
`Show each customer and their total number of orders.`

Expected route:

1. Multi-table but simple join + aggregate
2. No window rank, no complex subquery
3. 3B lane

Expected SQL shape:

```sql
SELECT c.customerNumber, COUNT(o.orderNumber) AS total_orders
FROM customers c
LEFT JOIN orders o ON c.customerNumber = o.customerNumber
GROUP BY c.customerNumber;
```

### Example C: 7B path (hard pattern)

Question:
`For each customer, return the order with the highest total order amount.`

Expected route:

1. Multi-table join (`orders -> orderdetails`)
2. Winner-row + window ranking intent
3. 7B lane (or 3B only if policy explicitly allows this pattern)

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

Non-critical reasons (warning only):

1. `semantic_contract_mismatch`

Policy:

1. Critical failure triggers retry/rescue logic.
2. Non-critical-only failure can continue with warning.

## 9. Operational Knobs

IR knobs:

1. `AGENT_IR_ENABLED`
2. `AGENT_IR_MIN_CONFIDENCE`
3. `AGENT_IR_MAX_TABLES`
4. `AGENT_IR_ALLOW_MULTI_TABLE`
5. `AGENT_IR_MULTI_TABLE_MIN_CONFIDENCE`
6. `AGENT_IR_DEBUG`

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

## 10. Route Map 40Q&A (Target)

This map is the expected lane allocation for `IR Safe Subset v1`.
It is a policy target map used to keep IR/LLM balance explicit and auditable.

Notes:

1. This is a policy expectation, not a strict guarantee for every paraphrase.
2. If retrieval scope or env knobs change, route may shift.
3. IR target coverage is intentionally moderate to preserve measurable LLM contribution.

### 10.1 Easy (E1-E10)

| ID  | Expected lane |                  Note                        |
|-----|---------------|----------------------------------------------|
| E1  |      IR       | Single-table, IR-safe filter + sort          |
| E2  |      3B       | Top-N pattern; IR confidence below threshold |
| E3  |      IR       | Single-table date filter + sort              |
| E4  |      3B       | Routed as non-IR in current baseline context |
| E5  |      IR       | Single-table null/missing filter             |
| E6  |      IR       | Single-table prefix pattern                  |
| E7  |      3B       | Context scope multi-table; IR gated off      |
| E8  |      IR       | Single-table regex pattern                   |
| E9  |      IR       | Single-table categorical filter              |
| E10 |      IR       | Single-table range filter                    |

### 10.2 Medium (M1-M15)

| ID  | Expected lane |                  Note                    |
|-----|---------------|------------------------------------------|
| M1  |      IR       | Single-table aggregate + month grouping  |
| M2  |      3B       | CASE-band aggregation                    |
| M3  |      3B       | Winner-row single-table (payments)       |
| M4  |      IR       | Single-table group + min/count           |
| M5  |      3B       | Group + HAVING aggregate filter          |
| M6  |      3B       | Simple scalar subquery single-table      |
| M7  |      IR       | Supported percentage metric pattern      |
| M8  |      3B       | Group + HAVING count                     |
| M9  |      IR       | Single-table group + order-by aggregate  |
| M10 |      3B       | Window rank single-table                 |
| M11 |      3B       | Second-highest single-table winner-row   |
| M12 |      3B       | Window cumulative single-table           |
| M13 |      3B       | Window count partition single-table      |
| M14 |      3B       | Window avg partition single-table        |
| M15 |      3B       | Window percentage-over-total single-table|

### 10.3 Hard (H1-H15)

| ID  | Expected lane |                  Note                             |
|-----|---------------|---------------------------------------------------|
| H1  |      3B       | Multi-table aggregate, non-hard pattern           |
| H2  |      IR       | Simple 2-table existence + grouped count          |
| H3  |      IR       | Simple 2-table join + grouped sum                 |
| H4  |      3B       | Simple multi-join + grouped sum                   |
| H5  |      3B       | Left-join rollup with null-preserving intent      |
| H6  |      7B       | Explicit ranking in multi-table context           |
| H7  |      7B       | Latest-year per-group + re-aggregate              |
| H8  |      7B       | Multi-table latest-year derived subquery          |
| H9  |      7B       | Complex subquery + grouped ordering               |
| H10 |      7B       | Multi-table + subquery/exists + having            |
| H11 |      3B       | Exclusion aggregate pattern solved in 3B lane     |
| H12 |      7B       | Multi-table aggregate + scalar subquery in HAVING |
| H13 |      7B       | Multi-table winner-row with tie-break ordering    |
| H14 |      7B       | Dual-fact rollup fanout-risk pattern              |
| H15 |      7B       | Cross-year exclusion + grouped ordering           |

### 10.4 Target Route Split

1. IR: 13/40
2. 3B: 18/40
3. 7B: 9/40

## 11. Benchmark and Change Protocol

When changing routing behavior:

1. Update this file first.
2. Add targeted regression cases for changed rule.
3. Run `easy`, `medium`, `hard`, then full set.
4. Report:
   - VA and EX
   - route split (IR vs fallback)
   - 3B vs 7B final model usage
   - gate hard-error rate
   - average latency

Acceptance criteria for a routing patch:

1. No regression on existing benchmark.
2. Reduced hard-error frequency on new questions.
3. No uncontrolled increase in 7B usage or latency.
4. IR share stays in moderate band (roughly 20%-40% on current 40-case devset).