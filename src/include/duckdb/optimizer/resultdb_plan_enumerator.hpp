//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/resultdb_plan_enumerator.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"

namespace duckdb {

//! Cost input used by the paper ResultDB enumerator. Node indexes refer to the
//! original relation occurrences in ResultDBEnumerationInput::nodes.
class ResultDBEnumerationCostProvider {
public:
	virtual ~ResultDBEnumerationCostProvider() = default;

	virtual double Cardinality(const vector<idx_t> &nodes) = 0;
	virtual double PayloadWidth(const vector<idx_t> &nodes) = 0;
	//! Paper mode: original occurrence masks, never contracted node indexes.
	virtual double ReducedCardinality(uint64_t relations, uint64_t reduced_by) = 0;
};

struct ResultDBEnumerationNode {
	idx_t stable_id = DConstants::INVALID_INDEX;
	double cardinality = 0;
	bool requested = false;
	uint64_t relations = 0;
};

struct ResultDBCostWeights {
	double join = 1;
	double probe = 1;
	double build = 3;
	double decompose = 1;
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

struct ResultDBPaperPlan : public ResultDBFoldEnumerationResult {
	//! All entries use original occurrence masks. A singleton has no split.
	unordered_map<uint64_t, uint64_t> join_splits;
	unordered_map<uint64_t, double> join_costs;
	double join_cost = 0;
	double decomposition_cost = 0;
	double reduction_cost = 0;
	idx_t work = 0;
};

class ResultDBPlanEnumerator {
public:
	static ResultDBRootEnumerationResult EnumeratePaperRoot(const ResultDBEnumerationInput &input,
	                                                        ResultDBEnumerationCostProvider &costs,
	                                                        const ResultDBCostWeights &weights);
	static ResultDBPaperPlan EnumeratePaper(const ResultDBEnumerationInput &input,
	                                        ResultDBEnumerationCostProvider &costs, const ResultDBCostWeights &weights,
	                                        bool use_tvc, idx_t work_limit = 200000);
};

} // namespace duckdb
