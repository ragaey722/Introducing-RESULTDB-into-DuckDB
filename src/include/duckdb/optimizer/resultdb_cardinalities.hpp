#pragma once

#include "duckdb/optimizer/resultdb_plan_enumerator.hpp"

namespace duckdb {
class ClientContext;

//! Parsed once at SET time; immutable after publication to the connection.
class ResultDBCardinalityFile {
public:
	struct Entry {
		vector<string> relations;
		vector<string> reduced_by;
		double cardinality;
	};
	static shared_ptr<const ResultDBCardinalityFile> Parse(const string &json, const string &path = "<memory>");
	static shared_ptr<const ResultDBCardinalityFile> Load(ClientContext &context, const string &path);
	string path;
	string sha256;
	unordered_map<string, vector<Entry>> datasets;
};

class ResultDBInjectedCardinalities : public ResultDBEnumerationCostProvider {
public:
	ResultDBInjectedCardinalities(shared_ptr<const ResultDBCardinalityFile> file, const string &dataset,
	                              vector<string> aliases);
	double Cardinality(const vector<idx_t> &nodes) override;
	double PayloadWidth(const vector<idx_t> &) override;
	double ReducedCardinality(uint64_t relations, uint64_t reduced_by) override;
	idx_t lookups = 0;
	const vector<string> aliases;

private:
	string Describe(uint64_t mask) const;
	shared_ptr<const ResultDBCardinalityFile> file;
	unordered_map<uint64_t, unordered_map<uint64_t, double>> values;
};
} // namespace duckdb
