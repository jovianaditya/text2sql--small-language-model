# INTENT IR Policy

## Purpose
This document defines the policy for `Intent IR` in this system:

- what `Intent IR` is responsible for
- what each field means
- how qtype should be assigned
- what should be shown in `IR-HINT`
- what should stay out of `IR-HINT`
- how to keep the system agnostic and avoid benchmark overfitting

This is a policy document, not a code listing. The implementation lives in:

- [intent_ir.py](/home/home/thesis/data/python/mcp_client/t2sql/intent_ir.py)
- [helpers_main.py](/home/home/thesis/data/python/mcp_client/t2sql/helpers_main.py)
- [agents.py](/home/home/thesis/data/python/mcp_client/t2sql/agents.py)
- [config.py](/home/home/thesis/data/python/mcp_client/t2sql/config.py)

## Core Principles

### 1. Intent IR is backbone-first
Intent IR should represent the backbone semantic structure of the question:

- subject table
- visible output backbone columns
- core metric family
- real requested filters
- ordering intent

It should not try to fully encode the final SQL plan in the hint.

### 2. Intent IR must stay agnostic
All logic must be:

- schema-driven
- phrase-pattern driven
- reusable on unseen questions

Do not:

- hardcode benchmark question IDs
- hardcode one-off fixes for a single case
- inject exact SQL formulas into the hint unless the pattern is very stable and generic

### 3. Hinting should guide, not over-steer
The IR-HINT is only a soft recommendation for the LLM.

It should:

- provide enough structure to reduce drift
- avoid brittle SQL-shaped detail
- prefer semantic wording over exact implementation wording when possible

### 4. Prefer omission over misleading detail
If a detail cannot be represented generically and safely, it is better to omit it than to show a misleading hint.

This rule is especially important for:

- set-operation branch filters
- CASE bucket internals
- threshold/HAVING scaffolding
- branch-specific or execution-only helper columns

## Intent IR Responsibilities

Intent IR has two roles:

1. Structured semantic extraction
2. Planner and hint input

The same IR can be used in two lanes:

- deterministic planner lane when the IR is plannable and safe
- soft IR-HINT lane when the IR is useful but not safe enough to plan directly

## Qtype Policy

## Important separation: router qtype vs raw IR qtype vs IR-HINT qtype
This system does not use one single qtype concept everywhere.

There are three different layers:

1. `Router qtype`
2. `Raw Intent IR qtype`
3. `IR-HINT displayed qtype`

These layers have different jobs and should not be mixed together.

### 1. Router qtype
This is the coarse family used by:

- retrieval bias
- router decisions
- some planner gates
- some downstream refiner heuristics

Current router qtype family:

- `lookup`
- `aggregate`
- `rank`

This layer should stay conservative and stable.

### 2. Raw Intent IR qtype
This is the qtype stored inside the built `Intent IR` payload.

Its job is to support:

- IR confidence
- IR plannability
- dimension / metric / ordering detection behavior
- planner-safe family checks

By policy, raw IR qtype should stay close to the coarse router family and should not be expanded casually.

Current intended raw IR family:

- `lookup`
- `aggregate`
- `rank`

Policy:

- `window_aggregate` is not required to become a raw IR qtype
- `set_operation` is not required to become a raw IR qtype
- those can remain derived display families for IR-HINT if that is safer for retrieval and planner stability

### 3. IR-HINT displayed qtype
This is the qtype label shown inside the soft hint passed to the LLM.

Its job is explanatory, not architectural.

It can be more descriptive than the raw IR qtype when that helps the LLM and does not destabilize the rest of the pipeline.

Current intended IR-HINT displayed qtypes:

- `lookup`
- `aggregate`
- `window_aggregate`
- `set_operation`

Important note:

- `rank` belongs to router/raw-IR family policy
- but `rank` is not currently a normal IR-HINT displayed qtype
- rank-like questions are usually suppressed from IR-HINT or handled through other hard-router logic because they are high-risk and easy to over-steer

### Practical matrix

|          Layer          |                Main purpose                 |                Allowed / intended values                  |
|-------------------------|---------------------------------------------|                                                           |
| Router qtype            | retrieval + routing + coarse family control | `lookup`, `aggregate`, `rank`                             |
| Raw Intent IR qtype     | IR parsing + confidence + plannability      | `lookup`, `aggregate`, `rank`                             |
| IR-HINT displayed qtype | LLM-facing explanation                      | `lookup`, `aggregate`, `window_aggregate` `set_operation` |

### Policy summary

- Do not assume router qtype and IR-HINT qtype are the same thing.
- Do not force every displayed family into the raw IR qtype layer.
- Use the raw IR qtype for stable system behavior.
- Use the IR-HINT qtype for better LLM explanation when safe.
- `rank` is an internal family first, not a preferred IR-HINT display label.

## Qtype definitions

### `lookup`
Use `lookup` when the result is fundamentally row-preserving and there is no grouped collapse of rows.

Typical properties:

- returns base rows or distinct rows
- may use support metrics for filtering
- may use scalar extrema support metrics like `MAX(orderDate)` for latest-date lookup
- may still include `Metrics:` in the hint if those metrics are support semantics, not grouped output

Examples:

- rows on the most recent order date
- list distinct countries
- orders with latest date

### `aggregate`
Use `aggregate` when the result is grouped or collapsed.

Typical properties:

- `GROUP BY`
- grouped metrics such as `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`
- one row per group or one collapsed summary row

Examples:

- count orders per status
- average amount by month
- vendors with at least 2 products

### `window_aggregate`
Use `window_aggregate` when the result is row-preserving but depends on same-group aggregate context or window semantics.

Typical properties:

- visible output remains row-level
- comparison against group average, group count, previous/next row, rank, running total, etc.
- often implemented with `OVER (PARTITION BY ...)`
- may also be implementable by correlated subquery, but the family is still row-preserving aggregate context

Examples:

- products whose MSRP is above the average MSRP in the same product line
- products whose buy price is above the average buy price in the same vendor
- previous payment amount per customer
- employee count within the same office

Policy note:

- `window_aggregate` is not the same as grouped `aggregate`
- `window_aggregate` is also not plain `lookup`
- it is the right family when the row survives but the condition or output depends on same-group aggregate semantics
- in current policy, this is mainly an `IR-HINT displayed qtype`, not necessarily a raw IR qtype

### `rank`
Use `rank` when the question asks for ranking semantics explicitly enough that the rank family is the main shape.

Typical properties:

- winner-row logic
- ordinal ranking
- explicit `RANK`, `DENSE_RANK`, or `ROW_NUMBER` style semantics

Policy note:

- `rank` is part of router and raw IR coarse-family policy
- `rank` is not currently a standard IR-HINT displayed qtype
- most rank-like shapes are better suppressed from IR-HINT than shown with a thin or misleading hint

### `set_operation`
Use `set_operation` as a displayed family when the question explicitly requires combining multiple result branches with `UNION` or `UNION ALL`.

Typical properties:

- multiple branches
- same column count across branches
- compatible branch datatypes
- duplicates either removed or preserved depending on question intent

Examples:

- append two product result sets and keep duplicates
- combine orders from 2003 and 2004 into one result set

Policy note:

- `set_operation` is best treated as a derived display family
- it should not automatically replace the coarse retrieval family everywhere
- branch-specific filters should not be flattened into one generic `Filters:` line
- in current policy, this is mainly an `IR-HINT displayed qtype`, not necessarily a raw IR qtype

## Field Policy

## Subject entity
`Subject entity` should identify the main executable table or semantic row source.

It should be:

- singular in purpose, even if retrieval context contains extra neighbor tables
- schema-grounded
- the table that best matches the visible result rows

## Dimensions
`Dimensions` represent backbone visible columns or grouping-support backbone columns.

### For lookup
Use visible row columns and distinct backbone columns.

Examples:

- `orders.orderNumber`
- `customers.country`
- `orders.orderDate`

### For aggregate
Use true grouping backbone columns.

Examples:

- `employees.jobTitle`
- `payments.paymentDate` as the source backbone for quarter grouping

### For transformed grouping
When grouping uses a derived expression, `Dimensions` should usually show the source backbone column rather than the exact SQL expression.

Examples:

- quarter of `paymentDate` -> `payments.paymentDate`
- month grouping on `paymentDate` -> `payments.paymentDate`
- first 2 characters of `checkNumber` -> `payments.checkNumber`
- CASE buckets on `creditLimit` -> `customers.creditLimit`

### For bucket and classification queries
Show the raw source column that drives the bucket, not the entire CASE logic.

Examples:

- CASE on `MSRP` -> `products.MSRP`
- CASE on `creditLimit` -> `customers.creditLimit`

### For window aggregate
Use the visible row backbone columns.

Examples:

- `productCode`, `productName`, `productLine`, `MSRP`
- `employeeNumber`, `jobTitle`

Do not force helper-only partition columns into `Dimensions:` unless they are part of the visible row backbone or clearly the source backbone for a transformed grouping concept.

### What not to put in `Dimensions`
Do not include:

- helper columns used only inside a metric condition
- leaked IDs that are not part of output or grouping backbone
- branch-only columns from set-operation internals
- exact CASE expressions
- exact temporal transformation expressions when the source backbone is enough

## Metrics
`Metrics` represent aggregate or support-metric intent.

### Use metrics for:

- grouped measures
- support metrics required by latest/earliest semantics
- support metrics required by same-group comparisons
- conditional aggregate family when clearly requested

### Metric wording policy

#### Normal aggregate metrics
Use canonical aggregate wording when possible:

- `COUNT(*)`
- `SUM(table.column)`
- `AVG(table.column)`
- `MIN(table.column)`
- `MAX(table.column)`

#### DISTINCT
Do not put `DISTINCT` inside `Metrics:` when it is projection semantics.

Instead use:

- `Projection: DISTINCT ...`

#### Conditional aggregate metrics
Prefer abstract conditional wording rather than exact brittle SQL formulas.

Preferred style:

- `SUM(table.column) with CASE WHEN`
- `AVG(table.column) with CASE WHEN`

This keeps the aggregate family visible without overfitting to one exact formula.

#### Percentage metrics
Prefer semantic wording rather than exact SQL arithmetic expressions.

Preferred style:

- `percentage of rows with non-null orders.shippedDate`
- `percentage of products.quantityInStock relative to the total stock across all rows`

Avoid:

- raw SQL-shaped expressions like `100 * x / SUM(x) OVER ()`

### Support metrics on lookup
Support metrics are allowed on `lookup` when they are required to explain the filter semantics.

Example:

- latest-date lookup can show `MAX(orderDate)` while remaining `lookup`

### What not to put in `Metrics`
Do not include:

- fake extrema caused only by ordering language
- SQL implementation detail that is not stable across equivalent solutions
- helper expressions that are not semantically meaningful to the user question

## Filters
`Filters` should represent real requested row-filter semantics only.

### Good filter hint behavior

- categorical equality
- explicit `IN` / `NOT IN`
- numeric thresholds
- date equality/range
- `is_null_or_empty`
- `is_not_null_or_empty`
- `like_prefix`
- `like_suffix`
- `like_contains`
- `not_like_prefix`
- `not_like_suffix`
- `not_like_contains`
- `regexp`

### Regex rendering policy
For hint display, regex filters should stay backbone-level:

- `Filters: table.column with REGEXP`

Do not render exact regex text in the hint unless there is a very strong reason.

### Set-operation filter policy
Do not flatten branch-specific filters from a set-operation question into one global `Filters:` line.

Reason:

- branch filters belong to different result branches
- a flat filter section can mislead the LLM into combining branch logic incorrectly

### Bucket/classification filter policy
Do not expose bucket-internal predicates as `Filters:` when they are really classification logic rather than row restriction.

### False-positive policy
Filter extraction must avoid producing:

- output-label words as filters
- aggregate threshold text as row filters
- explanatory phrases as categorical literals
- bucket labels as literal values

## Ordering
`Ordering` should be shown when it is explicit and stable.

Best fit:

- lookup
- set-operation result ordering
- visible-row ordering where the order target is clear

Avoid adding speculative order guidance when the order target is unclear.

## Projection
Use `Projection:` for output-shape semantics such as `DISTINCT`.

Example:

- `Projection: DISTINCT customers.country`

Do not overload `Metrics:` with projection behavior.

## Execution Shape
`Execution shape` should stay very conservative.

Current best use:

- safe single-table lookup only

Preferred wording:

- `direct single-table SELECT from the subject table; use simple WHERE filtering only if the question asks for it; avoid subqueries, set operations, and self-joins unless the question explicitly requires them`

Do not use `Execution shape` for complex medium/hard aggregate or window families.

When an otherwise simple lookup is gated into an unsupported family where this
wording would become misleading, suppress the section instead of replacing it
with a more specialized mini-plan.

Examples of suppression-first cases:

- scalar-extrema lookup families that need a same-table scalar subquery
- other unsupported lookup subfamilies where the backbone lookup wording would
  incorrectly discourage the required shape

Policy intent:

- keep `IR-HINT` backbone-only
- prefer omission over specialized fallback prose
- avoid turning `IR-HINT` into an implementation-shaped execution recipe
- reduce false positives from over-eager structural advice

## Sections intentionally avoided or minimized
To reduce noise and over-steering, the following sections should generally be avoided or removed unless there is a strong generic reason:

- `Threshold shape`
- `Derived grouping`
- `Comparison shape`
- `Aggregation shape`
- `Window shape`

These sections often become implementation-shaped and can increase prompt noise.

## Confidence and Planner Policy

## Confidence
Intent IR has a confidence score.

The score should reflect:

- presence of subject
- structural fields found
- filter and metric clarity
- family stability
- table ambiguity

Low-confidence IR should not be passed as a planner path.

## Planner compatibility
Intent IR should be planner-compatible only for safe families.

Good planner candidates:

- simple lookup
- compact aggregate
- safe regex family where explicitly supported

LLM-first families:

- rank
- set operation
- many window aggregate cases
- unsupported percentage shapes
- unsupported subquery intent
- ambiguous multi-table scope

## Hint gating
IR-HINT should be emitted only when:

- IR exists
- confidence is high enough
- payload is not empty
- context scope is safe enough
- the hint is likely to help more than hurt

IR-HINT should be suppressed when:

- the payload is too weak
- the context is too broad
- the router marks a hard unsupported family and the hint would be misleading

## Anti-overfitting Rules

### Good patches
Good patches are:

- phrase-family based
- schema-driven
- conservative
- minimal

Examples:

- support `does not contain` -> `not_like_contains`
- render regex generically as `with REGEXP`
- keep source backbone columns for transformed grouping

### Bad patches
Bad patches are:

- keyed to one benchmark ID
- keyed to one exact numeric literal
- keyed to one table name as a special exception
- keyed to one exact question string

If a fix only works because the benchmark wording is known, it should not be merged.

## Practical Guidance for Future Changes

Before changing `Intent IR`, ask:

1. Is this semantic structure reusable on unseen questions?
2. Is this field backbone-level or implementation-level?
3. Will this hint reduce drift, or over-steer the model?
4. Is omission safer than a misleading detail?
5. Does this preserve current valid logic?

## Recommended interpretation by layer

### Router qtype / raw IR qtype

- `lookup`
  - row-preserving
  - no grouped collapse
  - support metrics allowed

- `aggregate`
  - grouped or collapsed output
  - `GROUP BY` family

- `rank`
  - ranking or winner-row family

### IR-HINT displayed qtype

- `lookup`
  - row-preserving
  - may still include support metrics

- `aggregate`
  - grouped or collapsed output

- `window_aggregate`
  - row-preserving output
  - same-group aggregate context or window semantics required

- `set_operation`
  - multiple compatible branches combined by `UNION` or `UNION ALL`

### Notable exception

- `rank` exists in router/raw-IR policy
- but is not a normal IR-HINT displayed qtype under the current conservative hint strategy

## Examples

### Example 1: latest-date lookup
Question:

- orders on the most recent order date

Recommended hint shape:

- `Query type: lookup`
- `Dimensions: orders.orderNumber, orders.orderDate, orders.status`
- `Metrics: MAX(orders.orderDate)`

### Example 2: same-product-line average comparison
Question:

- products whose MSRP is higher than the average MSRP in the same product line

Recommended hint shape:

- `Query type: window_aggregate`
- `Dimensions: products.productCode, products.productName, products.productLine, products.MSRP`
- `Metrics: AVG(products.MSRP)`

### Example 3: same-vendor average comparison
Question:

- products whose buyPrice is higher than the average buyPrice in the same vendor

Recommended hint shape:

- `Query type: window_aggregate`
- `Dimensions: products.productCode, products.productVendor, products.buyPrice`
- `Metrics: AVG(products.buyPrice)`

### Example 4: set operation
Question:

- append two product result sets and keep duplicates

Recommended hint shape:

- `Query type: set_operation`
- `Dimensions: products.productCode, products.productName`
- `Ordering: ...`

Do not flatten branch filters into one `Filters:` line.

## Final Policy
Intent IR should be:

- semantically helpful
- structurally conservative
- schema-grounded
- agnostic
- minimal when detail would overfit

If a detail cannot be expressed generically, prefer not to show it in the hint.

## Questions on Development Set that supposed getting Intent IR HINT based on current threshold rules:
Easy:
E1-E50
Medium:
E1-E2 ; E5-E8 ; E10