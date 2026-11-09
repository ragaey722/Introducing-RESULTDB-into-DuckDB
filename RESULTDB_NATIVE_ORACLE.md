# Native row-cardinality oracle experiment

This experiment asks how exact row counts affect **plain SPJ** and **ResultDB decompose** when both retain DuckDB's native optimizer. It does not add exact column statistics or replace DuckDB's join search or cost formulas.

The four comparisons are plain estimated, plain oracle, decompose estimated, and decompose oracle. Historical `matched_plain` and `matched_decompose` artifacts used the now-removed injected-C_out join planner; they are not native oracle results. Preserve their raw labels for provenance, but describe them as **injected-C_out** comparisons in analysis.

## Configuration

```sql
SET resultdb_strategy='decompose';
SET optimizer_statistics_file='/absolute/path/oracle.json';
SET optimizer_statistics_mode='oracle';
```

`optimizer_statistics_mode` defaults to `estimated`. The independently configured file is loaded and validated at SET time, including a full database SHA256 check. Prepared statements retain the immutable file and mode, including when they must rebind. A later SET affects subsequent preparations.

The version-1 experiment supports deterministic, parameter-free inner SPJ over ordinary tables in one frozen, read-only database. Database file size, modification time, WAL absence, schema, query hash, relation occurrences and actual planning predicates identify requests. Self-joins have distinct occurrence identities. Unsupported shapes fail explicitly. Plain/decompose always retain DuckDB's native join planner. Paper-mode tdroot/tdfold_no_tvc/tdfold and AUTO can consume optimized-input counts through this profile; AUTO's native decompose comparison is described in `RESULTDB_PAPER.md`.

Each profile supplies exact integer bag row counts for raw scans, filtered relations and the candidate joins requested by the native search. The native row estimate is replaced, while column statistics, search rules and cost formulas stay in place. Native constant-list MARK joins introduced by IN rewriting preserve the left bag; their boolean marker is described with the equivalent IN predicate for offline counting. The generated constant-list scan is not an original database occurrence; its native estimate is retained. In this checkout that helper can display a zero estimate despite a nonempty literal list. The oracle scope is the database relations and their join/filter bags, rather than every internal helper operator.

Counts remain advisory planning information. They do not set maximum-cardinality proofs, and a supplied zero cannot independently remove execution. Actual database statistics may still prove a predicate empty through DuckDB's ordinary rules.

## Coverage and offline generation

Strict oracle preparation fails on an absent request, invalid profile or database mismatch. `SET optimizer_statistics_discovery=true` is a separate offline preparation mode that exports requests with estimated fallback. Those prepared statements are forbidden from executing. Repeated discovery with newly counted requests captures candidates exposed by the changed estimates; a final strict preparation seals coverage.

Diagnostic COUNT SQL in an audit describes the requested row bag. The native optimizer neither parses nor executes that SQL, and creates no temporary tables. Only the local offline generator runs counting SQL. Its exact weighted factor implementation preserves duplicates and checks supported numeric conversions; unsupported accelerated shapes use direct COUNT where applicable. Failed or timed-out counts are recorded, never replaced with estimated numbers.

The heuristic semijoin strategy uses highest degree and original-index ties for
both folding and root selection. Roots still prefer requested output relations.
These graph decisions consume no injected cardinalities. Oracle mode also applies
to its internal scans, local inputs and native fold join ordering; estimated mode
remains available. It does not request `resultdb_heuristic_input` counts.

Paper direct strategies add `resultdb_join_bag` and `resultdb_reduction` requests.
The latter use correlated EXISTS over the accumulated reducing occurrences and
retain all left-bag duplicates. Their local predicates include filters propagated
before ResultDB phase two. AUTO additionally requests exact counts for the native
join tree selected before its third-phase decision. The same profile fingerprints
and strict coverage rules apply. Existing four-baseline profiles are preserved;
new reduction counts belong in fresh artifacts. Each internal working plan has a
scoped binding adapter, sharing the immutable profile and preparation audit while
keeping its own operator pins. Projection/cast lineage resolves to original
occurrences. Fold inputs preserve join-bag multiplicities; duplicate elimination
occurs at the public outputs. Existing DISTINCT-description support and historical
profile entries are retained, but fold construction no longer requests an early
working-tuple DISTINCT. Generated projection IDs are
not request identities. Paper fold join trees remain fixed; oracle rows reach
later native optimization even when join ordering is skipped. Counts are refreshed
before build/probe decisions and physical planning and retained through statistics
propagation. Internal request stages include `internal[working_relation]` and the
offline preparation diagnostics include the corresponding physical plans. Strict
normal preparation does not render plan text; the coverage harness captures it
after preparation, outside benchmark timers. Ordinary helper constant lists
and runtime reduction state are outside this row-oracle extension.

Current experimental code and data live under local-only `resultdb_benchmarks/`; older scratch artifacts remain in `.codex_tmp`. The native profile is the sole production cardinality-file format, shared by native planning and ResultDB enumeration/internal planning. It includes database/query fingerprints and counting provenance. Counts may be reused only for identical SQL semantics on the same verified database. Historical Mutable-format files remain unchanged as artifacts; their loader and separate configuration have been removed.

## Reading the implementation

1. `src/include/duckdb/optimizer/native_row_oracle.hpp` defines the profile, per-preparation audit and graph adapter. `src/optimizer/native_row_oracle.cpp` implements them.
2. `src/optimizer/join_order/cardinality_estimator.cpp` and `relation_statistics_helper.cpp` are the row-count injection points. The adjacent native cost model and enumerator retain their formulas and search limits.
3. `test/api/test_native_row_oracle.cpp` checks exact consumed counts, independent small outputs, immutable snapshots and advisory zeros. Existing ResultDB tests cover output chains and metadata.

Cardinality entries accept exact nonnegative signed-64-bit COUNT(*) values, up to 2^63-1. The profile and audit retain the integer without rounding. DuckDB's existing double-precision cost arithmetic can round large values when consuming them; its formulas and enumeration rules remain unchanged. Negative, fractional and out-of-range counts still fail. This range includes CE candidate joins exceeding the earlier 2^53-1 profile limit.

The experiment keeps DuckDB's native enumeration safeguards. The separate paper experiment uses a three-million-work cap and records its work count; that cap is not a native join-search setting.

## Measurement and stopping rule

Use the same binary, database, threads and sampling for all four variants. Validate outputs before timing. Report preparation, materialized execution, local result retrieval and their total separately; exclude benchmark payload accounting. Report profile loading and offline counting separately. Do not merge incompatible historical binaries or query transformations into a controlled comparison.

After verified coverage, correctness checks and a reproducible benchmark analysis, freeze this extension. Slower oracle plans are experimental findings. Calibration, additional optimizer changes and a full Mutable execution comparison are future work.
