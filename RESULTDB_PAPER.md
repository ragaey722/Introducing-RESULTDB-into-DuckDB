# Experimental paper ResultDB optimizer

`resultdb_optimizer='legacy'` remains the default. Ordinary `decompose`, ordinary
SQL join planning retain their existing behavior. All direct strategies now share
one `ResultDBOptimizer` with graph-capture and execution-planning phases.
The previous simplified root/fold enumerator has been removed. TDRoot, TDFold-NoTVC,
TDFold and AUTO enumeration require paper mode and injected cardinalities.
Outside paper mode, `auto` selects ordinary decompose without enumeration; explicit
root/fold strategies report a configuration error. The `legacy` setting remains
for compatibility and the baseline paths; it no longer selects an old enumerator.

## Configuration

```sql
SET resultdb_cardinality_file = '/absolute/path/query_cardinalities.json';
SET resultdb_cardinality_dataset = 'job';
SET resultdb_optimizer = 'paper';
SET resultdb_strategy = 'tdfold';
SELECT RESULTDB a.id, b.id FROM a, b WHERE a.id = b.id;
```

Paper strategies are `tdroot` (acyclic input), `tdfold_no_tvc`, `tdfold`, and `auto`
(a DuckDB adaptation of TDResultDB, described below). Explicit `semijoin` remains the legacy heuristic. Explicit
`decompose` uses DuckDB's native join planner, as does ordinary plain SQL.
The former `resultdb_join_planner='cout'` option has been removed. The paper
enumerator still uses C_out to choose joins inside direct ResultDB folds.

Historical artifact labels `matched_plain` and `matched_decompose` refer to
these injected-C_out comparisons. They do not represent exact statistics
through DuckDB's native optimizer. The separate row-cardinality experiment
uses `optimizer_statistics_mode='oracle'`;
see `RESULTDB_NATIVE_ORACLE.md`.

## Current AUTO and optimized-input cardinalities

AUTO now compares TDFold against **native-oracle decompose**. It no longer
compares against the historical injected-C_out decompose candidate. Configure:

```sql
SET resultdb_optimizer='paper';
SET resultdb_enumeration_limit=3000000;
SET optimizer_statistics_file='/absolute/path/optimized_input_oracle.json';
SET optimizer_statistics_mode='oracle';
SET resultdb_strategy='auto';
```

The single ResultDB optimizer captures the original graph, attaches the inputs
after expression/filter propagation, and retains a TDFold candidate. The main
query then goes through native DuckDB join ordering with exact row counts. An
AUTO-only third phase costs that actual join tree and selects one candidate.
The native tree's join cost is the configured join weight times the sum of
intermediate join bags. Its decomposition cost is the full join bag times the
number of requested relation occurrences times the decomposition weight. Ties
choose native decompose. The winning native tree continues through ordinary
downstream optimization; the winning direct candidate retains its costed folds
and TDRoot schedule. Native join search rules and limits are unchanged.

The native oracle profile also supplies TDFold's joins and existential reductions,
identified by the actual optimized local predicates and original occurrence
bindings. Original equality edges remain authoritative for direct enumeration.
Reduction requests count left-bag survivors with EXISTS, preserving multiplicity;
they do not count DISTINCT public outputs. Diagnostic COUNT/EXISTS SQL is only an
offline request identity and is never executed by native planning or execution.
No Mutable-format cardinality file is needed for this configuration. Explicit
tdroot/tdfold_no_tvc/tdfold can use the same native oracle mode and input semantics.

Missing counts fail preparation with the exact key, stage and count request.
Discovery alone permits estimates and cannot execute. Profiles require separately
generated coverage; old native-decompose profiles generally lack reduction
requests. Empty-result proofs still come from logical rules, never injected zeros.

Mutable-format files remain available for paper-cost fixtures and explicit direct
strategies. Their original local-filter scope must not be presented as validated
optimized-input counts. Historical AUTO measurements use
the previous competitor and cannot be pooled with this adaptation.

The file setting parses and validates the entire file once, atomically, at SET
time. Failed SETs retain the previous snapshot. RESET releases the connection's
snapshot; existing prepared plans retain theirs. Use one query-specific file and
an explicit dataset key. Aliases are occurrence identities, including self-joins
and relations that are only used for filtering. File aliases must match the
bound query aliases. File contents are not refreshed automatically.

Mutable's schema is accepted directly:

```json
{"job": [
  {"relations": ["a"], "size": 100,
   "reductions": [{"right_relations": ["b"], "size": 5}]},
  {"relations": ["b"], "size": 10,
   "reductions": [{"right_relations": ["a"], "size": 7}]},
  {"relations": ["a", "b"], "size": 12}
]}
```

Sizes describe filtered base bags, intermediate join bags, or surviving left bag
rows of a semijoin. They are not counts of distinct public output projections.
Every requested estimate must be present. Missing estimates, malformed JSON,
conflicting duplicate entries, negative/nonfinite sizes, overlapping reduction
sets, and reductions exceeding their unreduced size are errors. The optimizer
never runs counting SQL and never uses estimates to prove a result empty.

## Algorithms and costs

The specification is *Query Optimization for Database-Returning Queries*,
Sections 4–6, pp. 6–17. Source comparison was against Mutable QO revision
`ac6d9b63a831eac782e9c3b75de2e4b91a46bc9d`.

TDRoot caches `(parent,node)` states across all roots. Children are ordered by
their independently estimated effect on the parent. Bottom-up estimates carry
the union of reducing occurrence sets. Fully reduced estimates are direct
lookups using the complement of the node's original occurrence set. Each parent
charges outgoing top-down work only for output-required subtrees. This includes
the global root. The selected orders are passed to the native execution program.

TDFold constructs cyclic biconnected blocks and block/articulation assignments.
SC1F joins a whole BEP; SC2F examines connected complements only when the BEP is
the whole block. SCTVC greedily contracts nonintersecting connected vertex cuts,
then recursively solves the resulting blocks while retaining excluded cut
vertices and original occurrence identities. Algorithm 3 is a local fold
selection heuristic, including its top-down term. The final direct score uses
the selected folds' join/decomposition costs plus TDRoot, not that heuristic.

Native oracle mode also supplies counts during internal relation/fold optimization,
including projected working-tuple DISTINCT outputs. Scoped adapters resolve new
projection bindings back to original occurrences. Paper-selected fold join trees
remain fixed; heuristic folds retain native join reordering. Degree/index heuristic
root and fold decisions remain independent of estimates. Historical campaigns
predating this internal-plan extension do not measure these new semantics.

Join plans use connected-subset dynamic programming with C_out and preserve the
chosen binary splits through native logical construction. The common optimizer
captures the original graph before expression/filter rewrites, then attaches
optimized relation inputs after whole-query expression rewriting, filter
pullup/pushdown and regex-range derivation. This includes local filters propagated
through equalities to other occurrences. It replaces the original inputs rather
than adding duplicate copies of predicates. Inferred cross-table equalities do
not become additional edges in the original graph.

The heuristic strategy also uses this captured graph. Its fold and root choices use highest degree with original-index ties, without cardinality estimates. Internal DuckDB planning uses ordinary estimates by default, or exact row counts when native oracle mode is enabled. Its former equality-forest
canonicalization has been removed: an original triangle remains a triangle and
may require a heuristic fold. Existing historical heuristic measurements describe
the previous implementation and must not be relabeled as results for this one.
The `legacy`/`paper` setting remains a compatibility/configuration gate for
enumerated strategies; it no longer denotes a separate optimizer object or phase.

Mutable-format cardinalities retain their original local-filter scope; they do
not establish exact optimized-input counts. The separate native oracle profile
now supplies those counts for paper direct strategies and AUTO as described above.
No transformed-input counts are manufactured from old estimates, and no runtime
counting is performed.

Subplan optimization may simplify expressions and choose physical operators;
join re-enumeration is disabled for selected folded join trees. Build/probe
orientation is still DuckDB's physical choice; C_out is symmetric in that choice.

Default coefficients are:

| Setting | Default | Term |
|---|---:|---|
| `resultdb_cost_join` | 1 | Intermediate join output rows |
| `resultdb_cost_probe` | 1 | Semijoin target rows |
| `resultdb_cost_build` | 3 | Semijoin source rows |
| `resultdb_cost_decompose` | 1 | Fully reduced fold rows × requested occurrences |

Singleton decomposition is zero. Algorithm 3 counts the fold's rows, weighted by
the probe coefficient, in both its loops. All coefficients must be positive and
finite. These are abstract costs, not predicted milliseconds.

Explicit TDRoot/TDFold strategies cost their direct candidate only. The legacy
diagnostic field for the unenumerated decompose candidate is zero in those
strategies; that zero must not be interpreted as a computed decompose cost.
AUTO computes both complete candidates before selecting one.

Important distinctions from Mutable's source:

- The published Algorithm 3 top-down term is included; the reference's
  `WeakCardinalityHeuristic` omits it.
- The global root's outgoing top-down costs are included.
- Mutable's `1.9` multiplier on the decompose candidate's join cost is not copied.
  The inspected source did not establish how that constant was calibrated.
- Requested occurrence count replaces the legacy DuckDB payload-byte-width term.
- Missing estimates fail; Mutable's Cartesian-product/no-reduction fallbacks are
  not enabled.

Connected TVC pairs match the reference implementation's candidate restriction;
nonadjacent Cartesian-product folds are not enumerated. The implementation uses
simple graph searches for TVCs and connected complements, not the asymptotically
optimal TVC routine discussed in the paper. Work is explicitly bounded by
`resultdb_enumeration_limit` (default 200000); there are at most 62 original
occurrences and 20 occurrences per enumerated join. Exceeding limits fails strict
planning, including `auto`, rather than silently selecting another optimizer.

Native folded working relations retain their existing early DISTINCT; final
public outputs retain independent duplicate elimination and `QueryResult::next`.
Injected join-bag cardinalities therefore need not equal physical working-row
counts after that DISTINCT. This is an execution/cost-model difference to measure,
not a reason to change ResultDB semantics or the baseline.

## Validation and experiments

Focused paper tests are tagged `[resultdb_paper]`; baseline ResultDB tests remain
available. Tests cover strict loading, snapshot lifetime, aliases, zero
estimates, graph preservation, schedules, exhaustive small-tree root choices,
connected fold partitions, and recursive TVCs.

Experiment tools and data are local-only under `.codex_tmp/resultdb_bench`.
Reference cardinalities and newly generated primary33 cardinalities are separate
tracks. Supplied files are planning fixtures until the corresponding prefiltered
data is verified. The reference q23a acyclic file contains missing numeric values.
No supplied file is repaired in place.

Diagnostics retain the snapshot SHA-256, dataset key, weights, chosen join trees,
reduction schedule, cost components, lookup count, and enumeration work. Cardinality
loading is timed separately from Prepare. Enumeration time is included in Prepare,
not added to total a second time. Local result retrieval is neither network
transfer nor client-side post-join reconstruction.

Calibration is a separate experiment: normalize probe weight to one and evaluate
join `{0.5,1,2}`, build `{1,2,3,4}`, and decomposition `{0.5,1,2}`. Deduplicate
identical selected plans, validate before timing, and group related JOB templates
within deterministic five-fold validation. Report held-out measurements separately
from the untuned profile. Existing T1/T6 artifacts do not validate this implementation.
