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
	//! Optional comparison type for a casted join key. INVALID means materialize the source type unchanged.
	LogicalType cast_type = LogicalType::INVALID;
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
	//! Logical plan that produces the filtered/projected working relation. Folded plans retain DISTINCT.
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

struct PreparedResultDBYannakakisReductionStep {
	idx_t target_relation = DConstants::INVALID_INDEX;
	idx_t source_relation = DConstants::INVALID_INDEX;
	vector<idx_t> target_columns;
	vector<idx_t> source_columns;
};

struct PreparedResultDBYannakakisRelationPhase {
	//! Index into bottom_up_steps for this relation -> parent, or INVALID for the root.
	idx_t bottom_up_to_parent_step = DConstants::INVALID_INDEX;
	//! Indexes into bottom_up_steps for child -> this relation, in prepared execution order.
	vector<idx_t> bottom_up_from_children_steps;
	//! Index into top_down_steps for parent -> this relation, or INVALID when no top-down reduction is needed.
	idx_t top_down_from_parent_step = DConstants::INVALID_INDEX;
	//! Indexes into top_down_steps for this relation -> required children, in prepared execution order.
	vector<idx_t> top_down_to_children_steps;
	//! Indexes into PreparedResultDBYannakakisProgram::outputs produced by this relation.
	vector<idx_t> output_indexes;
	//! Whether bottom-up survivors must remain live until this relation's top-down phase.
	bool retain_bottom_up = false;
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
	//! Every child-to-parent reduction, in execution order.
	vector<PreparedResultDBYannakakisReductionStep> bottom_up_steps;
	//! Only parent-to-child reductions needed to fully reduce requested output relations.
	vector<PreparedResultDBYannakakisReductionStep> top_down_steps;
	//! Whether a relation lies on a root-to-output path and must remain live through top-down reduction.
	vector<uint8_t> required_for_output;
	//! Per-relation roles and references into the prepared bottom-up/top-down schedules.
	vector<PreparedResultDBYannakakisRelationPhase> relation_phases;

	//! Validates the tree program and compiles oriented, output-directed reduction steps.
	void BuildReductionSchedule();
};

} // namespace duckdb
