//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/resultdb_yannakakis_optimizer.cpp
//
//
//===----------------------------------------------------------------------===//

#include "duckdb/optimizer/resultdb_yannakakis_optimizer.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_cross_product.hpp"
#include "duckdb/planner/operator/logical_distinct.hpp"
#include "duckdb/planner/operator/logical_dummy_scan.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_join.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/resultdb_reduced_plan.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace duckdb {

struct ResultDBSemijoinEdge {
	TableIndex left_table;
	TableIndex right_table;
	vector<JoinCondition> conditions;
};

struct ResultDBSemijoinAnalysis {
	vector<TableIndex> relations;
	unordered_map<idx_t, idx_t> relation_map;
	unordered_map<idx_t, unique_ptr<LogicalOperator>> relation_plans;
	vector<idx_t> estimated_cardinalities;
	vector<ResultDBSemijoinEdge> edges;
	string unsupported_reason;
};

static bool TryGetSingleResultDBTable(LogicalOperator &op, TableIndex &table_index) {
	unordered_set<TableIndex> bindings;
	LogicalJoin::GetTableReferences(op, bindings);
	if (bindings.size() != 1) {
		return false;
	}
	table_index = *bindings.begin();
	return table_index.IsValid();
}

static void PushResultDBFilter(unique_ptr<LogicalOperator> &root, unique_ptr<Expression> filter_expr) {
	auto filter = make_uniq<LogicalFilter>(std::move(filter_expr));
	filter->AddChild(std::move(root));
	root = std::move(filter);
}

static bool AddResultDBRelation(ResultDBSemijoinAnalysis &analysis, LogicalOperator &op, ClientContext &context) {
	TableIndex table_index;
	if (!TryGetSingleResultDBTable(op, table_index)) {
		analysis.unsupported_reason =
		    "semijoin currently supports only binary inner equijoins over relation occurrences";
		return false;
	}
	if (analysis.relation_map.find(table_index.index) != analysis.relation_map.end()) {
		return true;
	}
	auto relation_position = analysis.relations.size();
	analysis.relations.push_back(table_index);
	analysis.relation_map[table_index.index] = relation_position;
	analysis.estimated_cardinalities.push_back(op.EstimateCardinality(context));
	analysis.relation_plans[table_index.index] = op.Copy(context);
	return true;
}

static ResultDBSemijoinEdge &GetResultDBEdge(ResultDBSemijoinAnalysis &analysis, TableIndex left_table,
                                             TableIndex right_table) {
	for (auto &edge : analysis.edges) {
		if ((edge.left_table == left_table && edge.right_table == right_table) ||
		    (edge.left_table == right_table && edge.right_table == left_table)) {
			return edge;
		}
	}
	ResultDBSemijoinEdge edge;
	edge.left_table = left_table;
	edge.right_table = right_table;
	analysis.edges.push_back(std::move(edge));
	return analysis.edges.back();
}

static bool AddResultDBJoinCondition(ResultDBSemijoinAnalysis &analysis, const JoinCondition &condition) {
	if (!condition.IsComparison() || condition.GetComparisonType() != ExpressionType::COMPARE_EQUAL) {
		analysis.unsupported_reason = "semijoin currently supports only equality join predicates";
		return false;
	}
	if (condition.GetLHS().GetExpressionType() != ExpressionType::BOUND_COLUMN_REF ||
	    condition.GetRHS().GetExpressionType() != ExpressionType::BOUND_COLUMN_REF) {
		analysis.unsupported_reason = "semijoin join predicates must compare source columns directly";
		return false;
	}
	auto &left_colref = condition.GetLHS().Cast<BoundColumnRefExpression>();
	auto &right_colref = condition.GetRHS().Cast<BoundColumnRefExpression>();
	if (left_colref.depth > 0 || right_colref.depth > 0) {
		analysis.unsupported_reason = "semijoin does not support correlated join predicates";
		return false;
	}
	auto left_table = left_colref.binding.table_index;
	auto right_table = right_colref.binding.table_index;
	if (left_table == right_table) {
		analysis.unsupported_reason = "semijoin join predicates must connect two relation occurrences";
		return false;
	}

	auto &edge = GetResultDBEdge(analysis, left_table, right_table);
	auto copied_condition = condition.Copy();
	if (edge.left_table == right_table && edge.right_table == left_table) {
		copied_condition.Swap();
	}
	edge.conditions.push_back(std::move(copied_condition));
	return true;
}

static bool AddResultDBJoinCondition(ResultDBSemijoinAnalysis &analysis, Expression &expr) {
	if (!BoundComparisonExpression::IsComparison(expr.GetExpressionType())) {
		analysis.unsupported_reason = "semijoin direct output does not support cross-table non-comparison predicates";
		return false;
	}
	if (expr.GetExpressionType() != ExpressionType::COMPARE_EQUAL) {
		analysis.unsupported_reason = "semijoin currently supports only equality join predicates";
		return false;
	}
	auto &comparison = expr.Cast<BoundFunctionExpression>();
	JoinCondition condition(BoundComparisonExpression::Left(comparison).Copy(),
	                        BoundComparisonExpression::Right(comparison).Copy(), expr.GetExpressionType());
	return AddResultDBJoinCondition(analysis, condition);
}

static bool AnalyzeResultDBSemijoin(LogicalOperator &op, ResultDBSemijoinAnalysis &analysis,
                                    ClientContext &context);

static bool AnalyzeResultDBFilter(LogicalFilter &filter, ResultDBSemijoinAnalysis &analysis, ClientContext &context) {
	if (filter.children.size() != 1) {
		analysis.unsupported_reason = "semijoin direct output does not support filters with multiple children";
		return false;
	}
	if (!AnalyzeResultDBSemijoin(*filter.children[0], analysis, context)) {
		return false;
	}

	vector<unique_ptr<Expression>> predicates;
	for (auto &expr : filter.expressions) {
		predicates.push_back(expr->Copy());
	}
	LogicalFilter::SplitPredicates(predicates);

	for (auto &predicate : predicates) {
		if (predicate->HasSubquery()) {
			analysis.unsupported_reason = "semijoin direct output does not support WHERE predicates with subqueries";
			return false;
		}

		unordered_set<TableIndex> bindings;
		LogicalJoin::GetExpressionBindings(*predicate, bindings);
		if (bindings.empty()) {
			if (!predicate->IsFoldable()) {
				analysis.unsupported_reason =
				    "semijoin direct output does not support non-foldable WHERE predicates without table references";
				return false;
			}
			for (auto &entry : analysis.relation_plans) {
				PushResultDBFilter(entry.second, predicate->Copy());
			}
			continue;
		}
		if (bindings.size() == 1) {
			auto table_index = bindings.begin()->index;
			auto entry = analysis.relation_plans.find(table_index);
			if (entry == analysis.relation_plans.end()) {
				analysis.unsupported_reason = "semijoin WHERE predicate references an unknown relation occurrence";
				return false;
			}
			PushResultDBFilter(entry->second, std::move(predicate));
			continue;
		}
		if (!AddResultDBJoinCondition(analysis, *predicate)) {
			return false;
		}
	}
	return true;
}

static bool AnalyzeResultDBSemijoin(LogicalOperator &op, ResultDBSemijoinAnalysis &analysis,
                                    ClientContext &context) {
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_PROJECTION:
	case LogicalOperatorType::LOGICAL_DISTINCT:
		if (op.children.size() == 1) {
			return AnalyzeResultDBSemijoin(*op.children[0], analysis, context);
		}
		analysis.unsupported_reason =
		    StringUtil::Format("semijoin direct output does not support logical operator %s with multiple children",
		                       LogicalOperatorToString(op.type));
		return false;
	case LogicalOperatorType::LOGICAL_FILTER:
		return AnalyzeResultDBFilter(op.Cast<LogicalFilter>(), analysis, context);
	case LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
		if (op.children.size() != 2) {
			analysis.unsupported_reason = "semijoin currently supports only binary joins";
			return false;
		}
		return AnalyzeResultDBSemijoin(*op.children[0], analysis, context) &&
		       AnalyzeResultDBSemijoin(*op.children[1], analysis, context);
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN: {
		auto &join = op.Cast<LogicalComparisonJoin>();
		if (join.join_type != JoinType::INNER) {
			analysis.unsupported_reason = "semijoin currently supports only INNER joins";
			return false;
		}
		if (join.children.size() != 2) {
			analysis.unsupported_reason = "semijoin currently supports only binary joins";
			return false;
		}
		if (!AnalyzeResultDBSemijoin(*join.children[0], analysis, context) ||
		    !AnalyzeResultDBSemijoin(*join.children[1], analysis, context)) {
			return false;
		}
		if (join.conditions.empty()) {
			analysis.unsupported_reason = "semijoin requires equality join predicates";
			return false;
		}
		for (auto &condition : join.conditions) {
			if (!AddResultDBJoinCondition(analysis, condition)) {
				return false;
			}
		}
		return true;
	}
	case LogicalOperatorType::LOGICAL_GET:
		return AddResultDBRelation(analysis, op, context);
	default:
		analysis.unsupported_reason =
		    StringUtil::Format("semijoin direct output does not support logical operator %s",
		                       LogicalOperatorToString(op.type));
		return false;
	}
}

static void StoreResultDBJoinMetadata(ResultDBProperties &properties, const ResultDBSemijoinAnalysis &analysis) {
	properties.join_edges.clear();
	for (auto &edge : analysis.edges) {
		ResultDBJoinEdgeMetadata edge_metadata;
		edge_metadata.left_table_index = edge.left_table.index;
		edge_metadata.right_table_index = edge.right_table.index;
		for (auto &condition : edge.conditions) {
			auto &left_colref = condition.GetLHS().Cast<BoundColumnRefExpression>();
			auto &right_colref = condition.GetRHS().Cast<BoundColumnRefExpression>();
			ResultDBJoinColumnMetadata column_metadata;
			column_metadata.left_column_index = left_colref.binding.column_index.GetIndex();
			column_metadata.right_column_index = right_colref.binding.column_index.GetIndex();
			edge_metadata.columns.push_back(column_metadata);
		}
		properties.join_edges.push_back(std::move(edge_metadata));
	}
}

static bool BuildResultDBJoinTree(ResultDBSemijoinAnalysis &analysis, idx_t root_relation,
                                  vector<vector<std::pair<idx_t, idx_t>>> &adjacency, vector<idx_t> &parent,
                                  vector<idx_t> &parent_edge, vector<idx_t> &order) {
	auto relation_count = analysis.relations.size();
	if (relation_count < 2) {
		analysis.unsupported_reason = "semijoin requires at least two joined relation occurrences";
		return false;
	}
	if (analysis.edges.size() != relation_count - 1) {
		analysis.unsupported_reason = "semijoin currently supports only acyclic join graphs";
		return false;
	}

	adjacency.clear();
	adjacency.resize(relation_count);
	for (idx_t edge_idx = 0; edge_idx < analysis.edges.size(); edge_idx++) {
		auto &edge = analysis.edges[edge_idx];
		auto left_entry = analysis.relation_map.find(edge.left_table.index);
		auto right_entry = analysis.relation_map.find(edge.right_table.index);
		if (left_entry == analysis.relation_map.end() || right_entry == analysis.relation_map.end()) {
			analysis.unsupported_reason = "semijoin join graph references an unknown relation occurrence";
			return false;
		}
		adjacency[left_entry->second].emplace_back(right_entry->second, edge_idx);
		adjacency[right_entry->second].emplace_back(left_entry->second, edge_idx);
	}

	parent.assign(relation_count, DConstants::INVALID_INDEX);
	parent_edge.assign(relation_count, DConstants::INVALID_INDEX);
	vector<bool> visited(relation_count, false);
	order.clear();
	order.push_back(root_relation);
	visited[root_relation] = true;

	for (idx_t order_idx = 0; order_idx < order.size(); order_idx++) {
		auto current = order[order_idx];
		for (auto &entry : adjacency[current]) {
			auto next = entry.first;
			if (visited[next]) {
				continue;
			}
			visited[next] = true;
			parent[next] = current;
			parent_edge[next] = entry.second;
			order.push_back(next);
		}
	}

	if (order.size() != relation_count) {
		analysis.unsupported_reason = "semijoin currently supports only connected acyclic join graphs";
		return false;
	}
	return true;
}

static double ComputeResultDBRootCost(const ResultDBSemijoinAnalysis &analysis, const vector<idx_t> &parent,
                                      const vector<idx_t> &order) {
	vector<idx_t> depth(analysis.relations.size(), 0);
	for (auto relation : order) {
		if (parent[relation] != DConstants::INVALID_INDEX) {
			depth[relation] = depth[parent[relation]] + 1;
		}
	}

	double cost = 0;
	for (idx_t relation_idx = 0; relation_idx < analysis.relations.size(); relation_idx++) {
		auto cardinality = analysis.estimated_cardinalities[relation_idx];
		cost += static_cast<double>(cardinality) * static_cast<double>(depth[relation_idx]);
	}
	return cost;
}

static bool ChooseResultDBRootRelation(ResultDBSemijoinAnalysis &analysis, vector<idx_t> &best_parent,
                                       vector<idx_t> &best_parent_edge,
                                       vector<idx_t> &best_order, idx_t &best_root) {
	bool found = false;
	double best_cost = std::numeric_limits<double>::infinity();

	for (idx_t root = 0; root < analysis.relations.size(); root++) {
		vector<vector<std::pair<idx_t, idx_t>>> adjacency;
		vector<idx_t> parent;
		vector<idx_t> parent_edge;
		vector<idx_t> order;
		if (!BuildResultDBJoinTree(analysis, root, adjacency, parent, parent_edge, order)) {
			return false;
		}

		auto cost = ComputeResultDBRootCost(analysis, parent, order);
		if (!found || cost < best_cost ||
		    (cost == best_cost && analysis.relations[root].index < analysis.relations[best_root].index)) {
			found = true;
			best_cost = cost;
			best_root = root;
			best_parent = std::move(parent);
			best_parent_edge = std::move(parent_edge);
			best_order = std::move(order);
		}
	}
	return found;
}

static vector<unique_ptr<Expression>> BuildResultDBDistinctTargets(const vector<string> &names,
                                                                   const vector<LogicalType> &types,
                                                                   TableIndex projection_index) {
	vector<unique_ptr<Expression>> distinct_targets;
	for (idx_t column_idx = 0; column_idx < types.size(); column_idx++) {
		distinct_targets.push_back(make_uniq<BoundColumnRefExpression>(
		    names[column_idx], types[column_idx], ColumnBinding(projection_index, ProjectionIndex(column_idx))));
	}
	return distinct_targets;
}

static void AddResultDBWorkingColumn(vector<vector<idx_t>> &needed_columns, idx_t relation_pos, idx_t source_idx) {
	auto &columns = needed_columns[relation_pos];
	if (std::find(columns.begin(), columns.end(), source_idx) == columns.end()) {
		columns.push_back(source_idx);
	}
}

static idx_t GetResultDBWorkingColumnIndex(const vector<idx_t> &source_column_indices, idx_t source_idx) {
	auto entry = std::find(source_column_indices.begin(), source_column_indices.end(), source_idx);
	if (entry == source_column_indices.end()) {
		throw InternalException("ResultDB Yannakakis working relation is missing a required source column");
	}
	return NumericCast<idx_t>(entry - source_column_indices.begin());
}

static unique_ptr<LogicalOperator> BuildResultDBBaseMaterializationPlan(Binder &binder, TableIndex table_index,
                                                                        unique_ptr<LogicalOperator> base_plan,
                                                                        ResultDBYannakakisRelation &relation) {
	base_plan->ResolveOperatorTypes();
	vector<unique_ptr<Expression>> projection_expressions;
	for (idx_t working_idx = 0; working_idx < relation.source_column_indices.size(); working_idx++) {
		auto source_idx = relation.source_column_indices[working_idx];
		if (source_idx >= base_plan->types.size()) {
			throw InternalException("ResultDB Yannakakis source column index is out of range");
		}
		auto type = base_plan->types[source_idx];
		auto name = StringUtil::Format("resultdb_%llu_%llu", table_index.index, source_idx);
		relation.names.push_back(name);
		relation.types.push_back(type);
		projection_expressions.push_back(make_uniq<BoundColumnRefExpression>(
		    name, type, ColumnBinding(table_index, ProjectionIndex(source_idx))));
	}

	auto projection_index = binder.GenerateTableIndex();
	auto projection = make_uniq<LogicalProjection>(projection_index, std::move(projection_expressions));
	projection->AddChild(std::move(base_plan));

	auto distinct = make_uniq<LogicalDistinct>(
	    BuildResultDBDistinctTargets(relation.names, relation.types, projection_index), DistinctType::DISTINCT);
	distinct->AddChild(std::move(projection));
	return std::move(distinct);
}

static unique_ptr<ResultDBYannakakisProgram>
BuildResultDBYannakakisProgram(Binder &binder, ResultDBSemijoinAnalysis &analysis, idx_t root_relation,
                               vector<idx_t> parent, vector<idx_t> parent_edge, vector<idx_t> order) {
	auto &properties = binder.GetStatementProperties();
	auto program = make_uniq<ResultDBYannakakisProgram>();
	program->root_relation = root_relation;
	program->parent = std::move(parent);
	program->parent_edge = std::move(parent_edge);
	program->order = std::move(order);
	program->relations.resize(analysis.relations.size());

	vector<vector<idx_t>> needed_columns(analysis.relations.size());
	for (auto &edge : analysis.edges) {
		auto left_relation = analysis.relation_map.find(edge.left_table.index);
		auto right_relation = analysis.relation_map.find(edge.right_table.index);
		D_ASSERT(left_relation != analysis.relation_map.end());
		D_ASSERT(right_relation != analysis.relation_map.end());
		for (auto &condition : edge.conditions) {
			auto &left_colref = condition.GetLHS().Cast<BoundColumnRefExpression>();
			auto &right_colref = condition.GetRHS().Cast<BoundColumnRefExpression>();
			AddResultDBWorkingColumn(needed_columns, left_relation->second,
			                         left_colref.binding.column_index.GetIndex());
			AddResultDBWorkingColumn(needed_columns, right_relation->second,
			                         right_colref.binding.column_index.GetIndex());
		}
	}

	for (idx_t table_metadata_idx = 0; table_metadata_idx < properties.resultdb.tables.size(); table_metadata_idx++) {
		auto &table = properties.resultdb.tables[table_metadata_idx];
		auto relation_entry = analysis.relation_map.find(table.table_index);
		if (relation_entry == analysis.relation_map.end()) {
			throw InternalException("ResultDB semijoin output table does not map to a Yannakakis relation");
		}
		for (auto &column : table.columns) {
			if (column.source_column_index == DConstants::INVALID_INDEX) {
				throw InternalException("ResultDB semijoin column metadata is missing source_column_index");
			}
			AddResultDBWorkingColumn(needed_columns, relation_entry->second, column.source_column_index);
		}
	}

	for (idx_t relation_pos = 0; relation_pos < analysis.relations.size(); relation_pos++) {
		auto relation_table = analysis.relations[relation_pos];
		auto &relation = program->relations[relation_pos];
		relation.table_index = relation_table.index;
		auto &source_column_indices = needed_columns[relation_pos];
		std::sort(source_column_indices.begin(), source_column_indices.end());
		relation.source_column_indices = std::move(source_column_indices);

		auto relation_plan = analysis.relation_plans.find(relation_table.index);
		D_ASSERT(relation_plan != analysis.relation_plans.end());
		relation.base_plan =
		    BuildResultDBBaseMaterializationPlan(binder, relation_table, std::move(relation_plan->second), relation);
	}

	program->edges.reserve(analysis.edges.size());
	for (auto &edge : analysis.edges) {
		ResultDBYannakakisEdge program_edge;
		auto left_relation = analysis.relation_map.find(edge.left_table.index);
		auto right_relation = analysis.relation_map.find(edge.right_table.index);
		D_ASSERT(left_relation != analysis.relation_map.end());
		D_ASSERT(right_relation != analysis.relation_map.end());
		program_edge.left_relation = left_relation->second;
		program_edge.right_relation = right_relation->second;
		for (auto &condition : edge.conditions) {
			auto &left_colref = condition.GetLHS().Cast<BoundColumnRefExpression>();
			auto &right_colref = condition.GetRHS().Cast<BoundColumnRefExpression>();
			ResultDBYannakakisJoinColumn column;
			column.left_column_index =
			    GetResultDBWorkingColumnIndex(program->relations[program_edge.left_relation].source_column_indices,
			                                  left_colref.binding.column_index.GetIndex());
			column.right_column_index =
			    GetResultDBWorkingColumnIndex(program->relations[program_edge.right_relation].source_column_indices,
			                                  right_colref.binding.column_index.GetIndex());
			program_edge.columns.push_back(column);
		}
		program->edges.push_back(std::move(program_edge));
	}

	program->outputs.reserve(properties.resultdb.tables.size());
	for (idx_t table_metadata_idx = 0; table_metadata_idx < properties.resultdb.tables.size(); table_metadata_idx++) {
		auto &table = properties.resultdb.tables[table_metadata_idx];
		auto relation_entry = analysis.relation_map.find(table.table_index);
		D_ASSERT(relation_entry != analysis.relation_map.end());
		ResultDBYannakakisOutputTable output_table;
		output_table.relation = relation_entry->second;
		output_table.table_metadata_index = table_metadata_idx;
		for (auto &column : table.columns) {
			ResultDBYannakakisOutputColumn output_column;
			output_column.working_column_index =
			    GetResultDBWorkingColumnIndex(program->relations[output_table.relation].source_column_indices,
			                                  column.source_column_index);
			output_column.name = column.name;
			output_column.type = column.type;
			output_table.columns.push_back(std::move(output_column));
		}
		program->outputs.push_back(std::move(output_table));
	}
	return program;
}

ResultDBYannakakisOptimizer::ResultDBYannakakisOptimizer(Binder &binder, ClientContext &context)
    : binder(binder), context(context) {
}

unique_ptr<LogicalOperator>
ResultDBYannakakisOptimizer::Optimize(unique_ptr<LogicalOperator> plan,
                                      unique_ptr<ResultDBYannakakisProgram> &resultdb_yannakakis_program) {
	auto &properties = binder.GetStatementProperties();
	resultdb_yannakakis_program.reset();
	if (!properties.resultdb.enabled) {
		return plan;
	}
	properties.resultdb.execution_strategy = ResultDBStrategy::DECOMPOSE;
	properties.resultdb.join_edges.clear();
	if (properties.resultdb.requested_strategy == ResultDBStrategy::DECOMPOSE) {
		return plan;
	}

	ResultDBSemijoinAnalysis analysis;
	if (!AnalyzeResultDBSemijoin(*plan, analysis, context)) {
		if (properties.resultdb.requested_strategy == ResultDBStrategy::AUTO) {
			return plan;
		}
		throw BinderException("RESULTDB semijoin strategy does not support this query: %s",
		                      analysis.unsupported_reason);
	}

	vector<idx_t> parent;
	vector<idx_t> parent_edge;
	vector<idx_t> order;
	idx_t root_relation = DConstants::INVALID_INDEX;
	if (!ChooseResultDBRootRelation(analysis, parent, parent_edge, order, root_relation)) {
		if (properties.resultdb.requested_strategy == ResultDBStrategy::AUTO) {
			return plan;
		}
		throw BinderException("RESULTDB semijoin strategy does not support this query: %s",
		                      analysis.unsupported_reason);
	}

	resultdb_yannakakis_program =
	    BuildResultDBYannakakisProgram(binder, analysis, root_relation, std::move(parent), std::move(parent_edge),
	                                   std::move(order));
	properties.resultdb.execution_strategy = ResultDBStrategy::SEMIJOIN;
	StoreResultDBJoinMetadata(properties.resultdb, analysis);
	return make_uniq_base<LogicalOperator, LogicalDummyScan>(binder.GenerateTableIndex());
}

} // namespace duckdb
