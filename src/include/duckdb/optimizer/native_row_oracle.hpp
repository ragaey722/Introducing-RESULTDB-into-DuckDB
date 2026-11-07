// Experimental row-cardinality input for DuckDB's existing join optimizer.
#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/common/unordered_set.hpp"
#include "duckdb/common/vector.hpp"

namespace duckdb {
class ClientContext;
class LogicalOperator;
class LogicalGet;
class Expression;
class QueryGraphManager;
class JoinRelationSet;

class NativeRowOracleFile {
public:
	struct Entry {
		string query_sha256, schema, sql;
		idx_t cardinality;
	};
	string path, sha256, database_path, database_sha256, generator;
	idx_t database_size = 0;
	int64_t database_mtime = 0;
	unordered_map<string, Entry> entries;
	DUCKDB_API static shared_ptr<const NativeRowOracleFile> Parse(const string &json, const string &path);
	DUCKDB_API static shared_ptr<const NativeRowOracleFile> Load(ClientContext &context, const string &path);
	DUCKDB_API static string Hash(const string &text);
	DUCKDB_API static string Key(const string &query_hash, const string &schema, const string &sql);
};

class NativeRowOracleGraph;

//! One preparation, with an immutable input file and an audit of unique requests.
//! Discovery may use estimates but its prepared statements cannot execute.
class NativeRowOracle {
public:
	struct Request {
		string key, schema, sql, stage;
		idx_t cardinality = 0;
		bool found = false;
		idx_t calls = 0;
	};
	struct Description {
		uint64_t relations = 0;
		vector<string> predicates;
	};
	shared_ptr<const NativeRowOracleFile> file;
	string query_sha256, database_path;
	bool discovery;
	idx_t missing = 0, native_join_searches = 0;
	unordered_map<string, Request> requests;
	unordered_map<const LogicalOperator *, idx_t> pinned;

	DUCKDB_API NativeRowOracle(ClientContext &context, LogicalOperator &plan, const string &query,
	                          shared_ptr<const NativeRowOracleFile> file, bool discovery);
	DUCKDB_API string ToJSON() const;
	shared_ptr<NativeRowOracleGraph> BindGraph(QueryGraphManager &graph);
	idx_t Pin(LogicalOperator &op);
	idx_t Raw(LogicalGet &get, idx_t fallback);
	Description Describe(LogicalOperator &op);
	string Render(const Expression &expression) const;
	idx_t Lookup(Description description, idx_t fallback, const string &stage);
	//! Exact bag survivors of left after an existential reduction by right.
	//! SQL is an offline request identity only; planning never executes it.
	idx_t LookupReduction(Description left, Description right, idx_t fallback, const string &stage);

private:
	struct Scan {
		LogicalGet *get;
		string from, schema;
		idx_t ordinal;
	};
	unordered_map<idx_t, Scan> scans;
	vector<idx_t> scan_order;
	//! IN-clause MARK columns describe boolean predicates, not new base occurrences.
	unordered_map<idx_t, unique_ptr<Expression>> constant_marks;
	void CollectScans(ClientContext &context, LogicalOperator &op);
};

class NativeRowOracleGraph {
public:
	NativeRowOracle &oracle;
	vector<NativeRowOracle::Description> relations;
	vector<pair<uint64_t, string>> predicates;
	unordered_map<string, idx_t> cache;
	explicit NativeRowOracleGraph(NativeRowOracle &oracle) : oracle(oracle) {
	}
	idx_t Cardinality(JoinRelationSet &set, idx_t fallback);
};
} // namespace duckdb
