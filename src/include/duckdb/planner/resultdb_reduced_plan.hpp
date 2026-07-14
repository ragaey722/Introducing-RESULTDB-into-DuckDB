//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/planner/resultdb_reduced_plan.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/types.hpp"

namespace duckdb {

class LogicalOperator;
class PhysicalPlan;

struct ResultDBYannakakisSourceColumn {
	idx_t table_index = DConstants::INVALID_INDEX;
	idx_t source_column_index = DConstants::INVALID_INDEX;
};

struct ResultDBYannakakisRelation {
	//! Bound relation occurrence represented by this working relation, or INVALID_INDEX for a folded relation.
	idx_t table_index = DConstants::INVALID_INDEX;
	//! Bound relation occurrences represented by this working relation.
	vector<idx_t> table_indices;
	//! Source columns stored in this working relation, in working-column order.
	vector<ResultDBYannakakisSourceColumn> source_columns;
	//! Source column indexes for single-table working relations.
	vector<idx_t> source_column_indices;
	//! Working relation output names/types.
	vector<string> names;
	vector<LogicalType> types;
	//! Logical plan that materializes the filtered, projected, duplicate-free base relation.
	unique_ptr<LogicalOperator> base_plan;
};

struct ResultDBYannakakisJoinColumn {
	idx_t left_column_index = DConstants::INVALID_INDEX;
	idx_t right_column_index = DConstants::INVALID_INDEX;
};

struct ResultDBYannakakisEdge {
	idx_t left_relation = DConstants::INVALID_INDEX;
	idx_t right_relation = DConstants::INVALID_INDEX;
	vector<ResultDBYannakakisJoinColumn> columns;
};

struct ResultDBYannakakisOutputColumn {
	idx_t working_column_index = DConstants::INVALID_INDEX;
	string name;
	LogicalType type;
};

struct ResultDBYannakakisOutputTable {
	idx_t relation = DConstants::INVALID_INDEX;
	idx_t table_metadata_index = DConstants::INVALID_INDEX;
	vector<ResultDBYannakakisOutputColumn> columns;
};

struct ResultDBYannakakisProgram {
	idx_t root_relation = DConstants::INVALID_INDEX;
	vector<idx_t> parent;
	vector<idx_t> parent_edge;
	vector<idx_t> order;
	vector<ResultDBYannakakisRelation> relations;
	vector<ResultDBYannakakisEdge> edges;
	vector<ResultDBYannakakisOutputTable> outputs;
};

struct PreparedResultDBYannakakisProgram {
	idx_t root_relation = DConstants::INVALID_INDEX;
	vector<idx_t> parent;
	vector<idx_t> parent_edge;
	vector<idx_t> order;
	vector<ResultDBYannakakisRelation> relations;
	vector<ResultDBYannakakisEdge> edges;
	vector<ResultDBYannakakisOutputTable> outputs;
	//! Physical plans that materialize each base/folded relation once before in-memory Yannakakis reduction.
	vector<unique_ptr<PhysicalPlan>> base_plans;
};

} // namespace duckdb
