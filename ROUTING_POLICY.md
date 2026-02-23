# Routing Policy

This document defines the production routing policy for the Text-to-SQL system.
Goal: keep the system stable when new questions are added, without frequent manual patching.

## 1. Objective

1. Maximize correctness and stability.
2. Keep inference efficient (use deterministic IR when safe).
3. Escalate to stronger model only for clearly hard patterns.
4. Make behavior auditable and reproducible.

## 2. End-to-End Decision Flow

1. Question Guard checks whether question is SQL-relevant.
2. Retrieval builds context (`ALLOWED IDENTIFIERS` + cards).
3. Router classifies complexity and chooses initial model lane.
4. IR gate evaluates whether deterministic planner can safely handle the query.
5. If IR passes gate, system uses deterministic planner SQL.
6. If IR fails gate, system uses LLM drafting (3B or 7B based on router policy).
7. Refiner validates SQL and may regenerate with stricter constraints.

## 3. IR Policy (Deterministic Lane)

IR is enabled only when confidence and scope are safe.

Default policy:

1. `AGENT_IR_ENABLED=1`
2. `AGENT_IR_MIN_CONFIDENCE=0.80`
3. `AGENT_IR_MAX_TABLES=1`
4. `AGENT_IR_ALLOW_MULTI_TABLE=0`

IR is gated off if any of these hold:

1. Table count exceeds IR max table policy.
2. Context table count exceeds IR max table policy.
3. Router detects hard flags:
   - winner-row
   - cross-table exclusion
   - window aggregate
   - explicit rank function
   - ordinal ranking
   - subquery intent (complex)
4. Percentage intent is detected but IR has no supported percent metric.

## 4. LLM Routing Policy

### 4.1 3B Lane (default fallback lane)

Use 3B when:

1. IR is gated off and query is not hard-pattern.
2. Query is single-table (lookup/aggregate/filter/pattern).
3. Query is simple multi-table join and router sees no hard pattern.
4. Query is simple scalar subquery on single table.
5. Query is simple EXISTS/NOT EXISTS pattern with limited table fanout.

### 4.2 7B Hard Gate

Escalate directly to 7B when any hard condition appears:

1. Winner-row subquery pattern in multi-table scenario.
2. Cross-table exclusion semantics (`never`, `no X`, `without any`, etc.).
3. Window aggregate across multi-table context.
4. Explicit rank/window function intent across multi-table context.
5. Complex subquery intent.
6. Multi-fact rollup fanout risk:
   - customers + orders + payments
   - order count + payment total/payment status in one output.

## 5. Refine/Recovery Policy

1. Always run identifier validation before execution.
2. If identifier errors occur:
   - stay in 3B regeneration for simple multi-table candidate;
   - escalate regeneration to 7B for complex/hard candidate.
3. Keep deterministic rewrites for known high-risk semantic shapes:
   - fanout-safe rollups;
   - missing join alias on child metric usage.

## 6. Canonical Examples

1. H5-like question (employee -> customer -> payments total):
   - Router: 3B lane.
   - Recovery: deterministic rollup rewrite ensures missing payment join alias is fixed.
2. H11-like question (customers paid but never ordered):
   - Router: 3B lane (simple exclusion aggregate pattern).
   - Validation: must avoid false invalid-column from sparse context.
3. H14-like question (customer + total orders + total payments + payment_status):
   - Router: direct 7B via `multi_fact_rollup_fanout_risk`.

## 7. Config Knobs (Operational)

IR:

1. `AGENT_IR_ENABLED`
2. `AGENT_IR_MIN_CONFIDENCE`
3. `AGENT_IR_MAX_TABLES`
4. `AGENT_IR_ALLOW_MULTI_TABLE`
5. `AGENT_IR_MULTI_TABLE_MIN_CONFIDENCE`
6. `AGENT_IR_DEBUG`

Router:

1. `AGENT_ROUTER_ALLOW_SIMPLE_MULTI_TABLE_3B`
2. `AGENT_ROUTER_SIMPLE_MULTI_TABLE_MAX_TABLES`
3. `AGENT_ROUTER_ESCALATE_ON_WINNER_ROW`
4. `AGENT_ROUTER_ESCALATE_ON_EXCLUSION`
5. `AGENT_ROUTER_ESCALATE_ON_WINDOW_AGGREGATE`
6. `AGENT_ROUTER_ESCALATE_ON_EXPLICIT_RANK`
7. `AGENT_ROUTER_ESCALATE_ON_SUBQUERY_INTENT`

## 8. Stability Rules for Future Changes

When adding/changing routing rules:

1. Add rule in this policy file first.
2. Add targeted test cases for the rule.
3. Run at least:
   - easy
   - medium
   - hard subset covering the changed path
4. Record route stats:
   - IR coverage
   - 3B usage
   - 7B usage
   - regeneration rate
   - validator hard-error rate

## 9. Success Criteria

A routing change is accepted only if:

1. Existing benchmark does not regress.
2. New questions increase coverage without brittle rule explosion.
3. Hard-error regeneration frequency decreases over time.
