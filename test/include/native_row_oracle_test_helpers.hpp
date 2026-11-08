// Shared offline counting helpers for small native-oracle correctness fixtures.
#pragma once
#include "catch.hpp"
#include "test_helpers.hpp"
#include "duckdb/main/client_config.hpp"
#include "duckdb/main/prepared_statement_data.hpp"
#include "duckdb/optimizer/native_row_oracle.hpp"
#include <fstream>
#include <sstream>
#include <set>

using namespace duckdb;

namespace {
inline string NativeQuote(const string &text) {
	string out = "\"";
	for (auto c : text) {
		if (c == '\\' || c == '"') { out += '\\'; }
		if (c == '\n') { out += "\\n"; } else if (c == '\r') { out += "\\r"; } else { out += c; }
	}
	return out + "\"";
}
inline string NativeRead(const string &path) {
	std::ifstream in(path, std::ios::binary);
	std::ostringstream result;
	result << in.rdbuf();
	return result.str();
}
inline string NativeProfile(const string &database, const vector<shared_ptr<NativeRowOracle>> &audits,
                     Connection &counter, bool zeros = false, const string &excluded_key = "") {
	string result = "{\"version\":1,\"database_path\":" + NativeQuote(audits.front()->database_path) +
	                ",\"database_sha256\":" + NativeQuote(NativeRowOracleFile::Hash(NativeRead(database))) +
	                ",\"database_size\":" + std::to_string(NativeRead(database).size()) +
	                ",\"generator\":\"independent-test-count-star\",\"entries\":[";
	unordered_set<string> seen;
	for (auto &audit : audits) {
		for (auto &item : audit->requests) {
			if (item.first == excluded_key) { continue; }
			if (!seen.insert(item.first).second) { continue; }
			auto &request = item.second;
			auto count = counter.Query(request.sql);
			REQUIRE_NO_FAIL(*count);
			if (result.back() != '[') { result += ","; }
			result += "{\"key\":" + NativeQuote(item.first) + ",\"query_sha256\":" + NativeQuote(audit->query_sha256) +
			          ",\"schema\":" + NativeQuote(request.schema) + ",\"sql\":" + NativeQuote(request.sql) +
			          ",\"cardinality\":" + (zeros ? "0" : count->GetValue(0, 0).ToString()) + "}";
		}
	}
	return result + "]}";
}
inline std::multiset<string> NativeRows(QueryResult &result) {
 std::multiset<string> rows;
 while (auto chunk = result.Fetch()) {
  for (idx_t row=0; row<chunk->size(); row++) {
   string key;
   for (idx_t col=0; col<chunk->ColumnCount(); col++) {
    auto value = chunk->GetValue(col,row);
    auto text = value.IsNull() ? string("NULL") : value.ToSQLString();
    key += std::to_string(text.size()) + ":" + text;
   }
   rows.insert(key);
  }
 }
 return rows;
}
inline void NativeCompare(QueryResult &actual, QueryResult &expected) {
 auto a = &actual; auto e = &expected;
 while (a && e) {
  REQUIRE(a->names == e->names);
  REQUIRE(a->types == e->types);
  REQUIRE(a->properties.resultdb.tables.size() == e->properties.resultdb.tables.size());
  for (idx_t i = 0; i < a->properties.resultdb.tables.size(); i++) {
   REQUIRE(a->properties.resultdb.tables[i].name == e->properties.resultdb.tables[i].name);
  }
  REQUIRE(NativeRows(*a) == NativeRows(*e));
  a = a->next.get(); e = e->next.get();
 }
 REQUIRE(a == nullptr); REQUIRE(e == nullptr);
}
inline void NativeWrite(const string &path, const string &text) {
	std::ofstream file(path, std::ios::binary);
	file << text;
}

inline void SealResultDBOracle(Connection &con, Connection &counter, const string &database, const string &profile,
                       const vector<string> &queries, bool zeros = false,
                       const vector<string> &strategies = {"auto", "tdfold", "tdfold_no_tvc", "decompose"}) {
	vector<shared_ptr<NativeRowOracle>> audits;
	REQUIRE_NO_FAIL(con.Query("SET resultdb_optimizer='paper'; "
	                          "SET resultdb_enumeration_limit=3000000; SET optimizer_statistics_discovery=true"));
	bool complete = false;
	for (idx_t round = 0; round < 12; round++) {
		idx_t missing = 0;
		for (auto &strategy : strategies) {
			REQUIRE_NO_FAIL(con.Query(string("SET resultdb_strategy='") + strategy + "'"));
			for (auto &query : queries) {
				auto prepared = con.Prepare(query);
				INFO((prepared->HasError() ? prepared->GetError() : query));
				REQUIRE(!prepared->HasError());
				audits.push_back(prepared->data->properties.native_row_oracle);
				missing += audits.back()->missing;
			}
		}
		NativeWrite(profile, NativeProfile(database, audits, counter, zeros));
		REQUIRE_NO_FAIL(con.Query("SET optimizer_statistics_file='" + profile + "'"));
		if (!missing && round > 0) { complete = true; break; }
	}
	REQUIRE(complete);
	REQUIRE_NO_FAIL(con.Query("SET optimizer_statistics_discovery=false; SET optimizer_statistics_mode='oracle'"));
}
} // namespace
