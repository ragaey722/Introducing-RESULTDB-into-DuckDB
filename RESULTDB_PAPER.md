# Experimental paper ResultDB optimizer

One `ResultDBOptimizer` captures the original graph, attaches optimized local
inputs after filter propagation, and dispatches strategy selection. Use the
single `resultdb_strategy` selector below; the old `resultdb_optimizer` gate has
been removed. Its former AUTO-to-decompose fallback is no longer available.

| Strategy | SQL | Statistics |
|---|---|---|
| `Plain` | ordinary `SELECT` | estimated or oracle |
| `Decompose` (default) | `SELECT RESULTDB` | estimated or oracle |
| `TD_heuristic` | `SELECT RESULTDB` | estimated or oracle for internal native plans |
| `TD_root` | `SELECT RESULTDB` over an acyclic original graph | oracle |
| `TD_fold_no_tvc` | `SELECT RESULTDB` | oracle |
| `TD_fold` | `SELECT RESULTDB` | oracle |
| `Auto` | `SELECT RESULTDB` | oracle |

Ordinary SELECT always retains its flat output, regardless of the selector.
`Plain` with `SELECT RESULTDB` is rejected. The heuristic retains degree/index
root and fold decisions in both statistics modes. Discovery may use estimates
only to collect offline requests; discovery plans cannot execute.

## Configuration

```sql
SET optimizer_statistics_file = '/absolute/path/native_oracle.json';
SET optimizer_statistics_mode = 'oracle';
SET resultdb_strategy = 'TD_fold';
SELECT RESULTDB a.id, b.id FROM a, b WHERE a.id = b.id;
```

Canonical strategy names are case-insensitive. Historical input spellings
`semijoin`, `tdroot`, `tdfold`, `tdfold_no_tvc`, `tdfold-no-tvc`, and `tdresultdb`
remain aliases. Settings and new diagnostics emit the canonical names.
`Decompose` and ordinary SQL use DuckDB's native join planner.
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
SET resultdb_enumeration_limit=3000000;
SET optimizer_statistics_file='/absolute/path/optimized_input_oracle.json';
SET optimizer_statistics_mode='oracle';
SET resultdb_strategy='Auto';
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
All enumerated strategies use this native oracle mode and input semantics.
The older Mutable-format loader and its separate file/dataset settings have been removed.

Missing counts fail preparation with the exact key, stage and count request.
Discovery alone permits estimates and cannot execute. Profiles require separately
generated coverage; old native-decompose profiles generally lack reduction
requests. Empty-result proofs still come from logical rules, never injected zeros.

The native profile is loaded and validated atomically at SET time. Failed SETs
retain the previous snapshot. RESET releases the connection's snapshot; existing
prepared statements retain theirs, including on rebind. The profile identifies
requests by query, schema and count-request fingerprints and validates the frozen
database identity. See [RESULTDB_NATIVE_ORACLE.md](RESULTDB_NATIVE_ORACLE.md) for
the native profile and discovery workflow.

Counts describe raw/filtered bags, join bags, surviving left-bag rows after
existential reductions, or internal working-tuple DISTINCT outputs. These are
different quantities. Missing requested counts fail strict preparation; no
runtime counting or estimated fallback occurs. Injected zeros never prove a
result empty. Historical Mutable-format files and measurements remain local
artifacts; they are no longer accepted as current optimizer inputs.

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

Native oracle mode also supplies bag counts during internal relation/fold optimization.
Scoped adapters resolve new
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
All strategies share these phases; only root, fold and reduction-schedule
selection differs. AUTO additionally compares against the native decompose tree.

The native oracle supplies counts for the actual optimized inputs, including
propagated filters. Historical Mutable-format counts have a different scope and
are not substituted for these requests. No runtime counting is performed.

Subplan optimization may simplify expressions and choose physical operators;
join re-enumeration is disabled for selected folded join trees. Build/probe
orientation is still DuckDB's physical choice; C_out is symmetric in that choice.

Optional advanced cost settings retain these fixed default coefficients:

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
`resultdb_enumeration_limit` (default 3000000); there are at most 62 original
occurrences and 20 occurrences per enumerated join. Exceeding limits fails strict
planning, including `auto`, rather than silently selecting another optimizer.

Native folded working relations preserve join-bag multiplicities through the
semijoin reductions. Final public outputs retain independent duplicate elimination
and `QueryResult::next`. The former early fold DISTINCT is no longer inserted;
historical campaigns that used it remain measurements of their frozen builds.

## Validation and experiments

Focused paper tests are tagged `[resultdb_paper]`; baseline ResultDB tests remain
available. Tests cover strict loading, snapshot lifetime, aliases, zero
estimates, graph preservation, schedules, exhaustive small-tree root choices,
connected fold partitions, and recursive TVCs.

Current experiment tools and data are local-only under `resultdb_benchmarks/`;
older scratch artifacts remain in `.codex_tmp`.
Reference cardinalities and newly generated primary33 cardinalities are separate
tracks. Supplied files are planning fixtures until the corresponding prefiltered
data is verified. The reference q23a acyclic file contains missing numeric values.
No supplied file is repaired in place.

Diagnostics retain the native profile SHA-256, input-scope label, weights, chosen join trees,
reduction schedule, cost components, lookup count, and enumeration work. Cardinality
loading is timed separately from Prepare. Enumeration time is included in Prepare,
not added to total a second time. Local result retrieval is neither network
transfer nor client-side post-join reconstruction.

Calibration is a separate experiment: normalize probe weight to one and evaluate
join `{0.5,1,2}`, build `{1,2,3,4}`, and decomposition `{0.5,1,2}`. Deduplicate
identical selected plans, validate before timing, and group related JOB templates
within deterministic five-fold validation. Report held-out measurements separately
from the untuned profile. Existing T1/T6 artifacts do not validate this implementation.

New diagnostics distinguish `tvc_enabled` (search allowed) from `tvc_used`
(the selected direct plan actually uses a TVC). AUTO selecting Decompose reports
`tvc_used=false`. Historical enabled-only artifacts cannot establish actual usage.
