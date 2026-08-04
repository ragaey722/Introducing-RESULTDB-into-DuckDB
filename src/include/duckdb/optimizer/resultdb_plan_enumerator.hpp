//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/resultdb_plan_enumerator.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"

namespace duckdb {

//! Cost input used by the ResultDB enumerators. Node indexes refer to the
//! original relation occurrences in ResultDBEnumerationInput::nodes.
class ResultDBEnumerationCostProvider {
public:
	virtual ~ResultDBEnumerationCostProvider() = default;

	virtual double Cardinality(const vector<idx_t> &nodes) = 0;
	virtual double PayloadWidth(const vector<idx_t> &nodes) = 0;
};

struct ResultDBEnumerationNode {
	idx_t stable_id = DConstants::INVALID_INDEX;
	double cardinality = 0;
	bool requested = false;
};

struct ResultDBEnumerationEdge {
	idx_t left = DConstants::INVALID_INDEX;
	idx_t right = DConstants::INVALID_INDEX;
};

struct ResultDBEnumerationInput {
	vector<ResultDBEnumerationNode> nodes;
	vector<ResultDBEnumerationEdge> edges;
};

struct ResultDBRootEnumerationResult {
	bool valid = false;
	string error;
	double cost = 0;
	idx_t root = DConstants::INVALID_INDEX;
	vector<idx_t> parent;
	vector<idx_t> parent_edge;
	vector<idx_t> order;
	//! Explicit sibling orders. The child indexes refer to the contracted graph.
	vector<vector<idx_t>> bottom_up_children;
	vector<vector<idx_t>> top_down_children;
	vector<double> bottom_up_cardinality;
	vector<double> fully_reduced_cardinality;
};

struct ResultDBFoldEnumerationResult : public ResultDBRootEnumerationResult {
	//! Disjoint groups of original node indexes. Singleton groups are omitted.
	vector<vector<idx_t>> folds;
	idx_t candidate_count = 0;
	vector<idx_t> block_sizes;
	bool used_tvc = false;
	double fold_cost = 0;
};

class ResultDBPlanEnumerator {
public:
	//! Paper-style root enumeration on an already acyclic graph.
	static ResultDBRootEnumerationResult EnumerateRoot(const ResultDBEnumerationInput &input,
	                                                    ResultDBEnumerationCostProvider &costs);

	//! Enumerates SC1F/SC2F candidates, optionally applies the paper's greedy
	//! non-overlapping two-vertex-cut transformation, and runs TDRoot on every
	//! resulting contracted tree.
	static ResultDBFoldEnumerationResult EnumerateFolds(const ResultDBEnumerationInput &input,
	                                                     ResultDBEnumerationCostProvider &costs, bool use_tvc,
	                                                     idx_t work_limit = 200000);

	//! Exact connected-subgraph DP join cost used by TDResultDB's decompose candidate.
	static double EstimateJoinCost(const ResultDBEnumerationInput &input, ResultDBEnumerationCostProvider &costs,
	                               const vector<idx_t> &nodes);
};

} // namespace duckdb
