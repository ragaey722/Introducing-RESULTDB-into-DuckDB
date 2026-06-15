#include "duckdb/planner/binder.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/logical_operator_deep_copy.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_distinct.hpp"
#include "duckdb/planner/operator/list.hpp"
#include "duckdb/planner/operator/logical_dummy_scan.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_join.hpp"
#include "duckdb/planner/operator/logical_limit.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/query_node/bound_select_node.hpp"

#include <utility>

namespace duckdb {

unique_ptr<LogicalOperator> Binder::PlanFilter(unique_ptr<Expression> condition, unique_ptr<LogicalOperator> root) {
	PlanSubqueries(condition, root);
	auto filter = make_uniq<LogicalFilter>(std::move(condition));
	filter->AddChild(std::move(root));
	return std::move(filter);
}

struct ResultDBSemijoinEdge {
	TableIndex left_table;
	TableIndex right_table;
	vector<JoinCondition> conditions;
};

struct ResultDBSemijoinAnalysis {
	vector<TableIndex> relations;
	unordered_map<idx_t, idx_t> relation_map;
	unordered_map<idx_t, unique_ptr<LogicalOperator>> relation_plans;
	vector<ResultDBSemijoinEdge> edges;
	string unsupported_reason;
};

struct ResultDBPlanWithBinding {
	unique_ptr<LogicalOperator> plan;
	TableIndex table_index;
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

static bool AnalyzeResultDBSemijoin(LogicalOperator &op, ResultDBSemijoinAnalysis &analysis,
                                    ClientContext &context) {
	if (op.type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		return AddResultDBRelation(analysis, op, context);
	}

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

static vector<JoinCondition> CopyResultDBEdgeConditions(const ResultDBSemijoinEdge &edge, TableIndex left_table,
                                                        TableIndex right_table) {
	vector<JoinCondition> result;
	for (auto &condition : edge.conditions) {
		auto copied_condition = condition.Copy();
		if (edge.left_table == left_table && edge.right_table == right_table) {
			result.push_back(std::move(copied_condition));
		} else if (edge.left_table == right_table && edge.right_table == left_table) {
			copied_condition.Swap();
			result.push_back(std::move(copied_condition));
		} else {
			throw InternalException("ResultDB semijoin edge does not contain requested relation pair");
		}
	}
	return result;
}

static void ReplaceResultDBConditionTable(Expression &expr, TableIndex old_table, TableIndex new_table) {
	if (expr.GetExpressionType() != ExpressionType::BOUND_COLUMN_REF) {
		throw InternalException("ResultDB semijoin condition is not a bound column reference");
	}
	auto &colref = expr.Cast<BoundColumnRefExpression>();
	if (colref.binding.table_index == old_table) {
		colref.binding.table_index = new_table;
	}
}

static void ReplaceResultDBConditionTable(JoinCondition &condition, TableIndex old_table, TableIndex new_table) {
	ReplaceResultDBConditionTable(condition.GetLHS(), old_table, new_table);
	ReplaceResultDBConditionTable(condition.GetRHS(), old_table, new_table);
}

static vector<JoinCondition> CopyResultDBEdgeConditions(const ResultDBSemijoinEdge &edge, TableIndex left_table,
                                                        TableIndex right_table, TableIndex left_binding,
                                                        TableIndex right_binding) {
	auto conditions = CopyResultDBEdgeConditions(edge, left_table, right_table);
	for (auto &condition : conditions) {
		ReplaceResultDBConditionTable(condition, left_table, left_binding);
		ReplaceResultDBConditionTable(condition, right_table, right_binding);
	}
	return conditions;
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

static bool SupportsResultDBDirectSemijoinSelectNode(BoundSelectNode &statement, string &unsupported_reason) {
	if (statement.sample_options) {
		unsupported_reason = "semijoin direct output does not support SAMPLE";
		return false;
	}
	if (!statement.aggregates.empty() || !statement.groups.group_expressions.empty() ||
	    !statement.groups.grouping_sets.empty()) {
		unsupported_reason = "semijoin direct output does not support aggregates or GROUP BY";
		return false;
	}
	if (statement.having) {
		unsupported_reason = "semijoin direct output does not support HAVING";
		return false;
	}
	if (!statement.windows.empty()) {
		unsupported_reason = "semijoin direct output does not support window functions";
		return false;
	}
	if (statement.qualify) {
		unsupported_reason = "semijoin direct output does not support QUALIFY";
		return false;
	}
	if (!statement.unnests.empty()) {
		unsupported_reason = "semijoin direct output does not support UNNEST";
		return false;
	}
	if (!statement.modifiers.empty()) {
		unsupported_reason = "semijoin direct output does not support query modifiers";
		return false;
	}
	return true;
}

static ResultDBPlanWithBinding CopyResultDBRelationPlan(Binder &binder, ResultDBSemijoinAnalysis &analysis,
                                                        idx_t relation_pos, bool fresh_binding) {
	auto relation = analysis.relations[relation_pos];
	auto entry = analysis.relation_plans.find(relation.index);
	D_ASSERT(entry != analysis.relation_plans.end());
	ResultDBPlanWithBinding result;
	if (fresh_binding) {
		LogicalOperatorDeepCopy deep_copy(binder, nullptr);
		result.plan = deep_copy.DeepCopy(entry->second);
		if (!TryGetSingleResultDBTable(*result.plan, result.table_index)) {
			throw InternalException("ResultDB semijoin relation copy lost its single output binding");
		}
	} else {
		result.plan = entry->second->Copy(binder.context);
		result.table_index = relation;
	}
	return result;
}

static ResultDBPlanWithBinding
BuildResultDBSemijoinMessage(Binder &binder, ResultDBSemijoinAnalysis &analysis,
                             const vector<vector<std::pair<idx_t, idx_t>>> &adjacency, idx_t sender_pos,
                             idx_t recipient_pos) {
	// V1 keeps the recursive message builder simple. It can copy plan fragments
	// quadratically across the join tree; replacing that is future work.
	auto sender_table = analysis.relations[sender_pos];
	auto result = CopyResultDBRelationPlan(binder, analysis, sender_pos, true);
	for (auto &entry : adjacency[sender_pos]) {
		auto neighbor_pos = entry.first;
		if (neighbor_pos == recipient_pos) {
			continue;
		}
		auto neighbor_table = analysis.relations[neighbor_pos];
		auto message = BuildResultDBSemijoinMessage(binder, analysis, adjacency, neighbor_pos, sender_pos);
		auto conditions = CopyResultDBEdgeConditions(analysis.edges[entry.second], sender_table, neighbor_table,
		                                             result.table_index, message.table_index);
		result.plan = LogicalComparisonJoin::CreateJoin(JoinType::SEMI, JoinRefType::REGULAR, std::move(result.plan),
		                                                std::move(message.plan), std::move(conditions));
	}
	return result;
}

static void PushResultDBFilter(unique_ptr<LogicalOperator> &root, unique_ptr<Expression> filter_expr) {
	auto filter = make_uniq<LogicalFilter>(std::move(filter_expr));
	filter->AddChild(std::move(root));
	root = std::move(filter);
}

static bool ApplyResultDBWhereFilters(ResultDBSemijoinAnalysis &analysis, const unique_ptr<Expression> &where_clause) {
	if (!where_clause) {
		return true;
	}

	vector<unique_ptr<Expression>> predicates;
	predicates.push_back(where_clause->Copy());
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
		if (bindings.size() > 1) {
			analysis.unsupported_reason = "semijoin direct output does not support cross-table WHERE predicates";
			return false;
		}

		auto table_index = bindings.begin()->index;
		auto entry = analysis.relation_plans.find(table_index);
		if (entry == analysis.relation_plans.end()) {
			analysis.unsupported_reason = "semijoin WHERE predicate references an unknown relation occurrence";
			return false;
		}
		PushResultDBFilter(entry->second, std::move(predicate));
	}
	return true;
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

static idx_t GetResultDBDeterministicRootRelation(const ResultDBSemijoinAnalysis &analysis) {
	D_ASSERT(!analysis.relations.empty());
	idx_t root = 0;
	for (idx_t relation_pos = 1; relation_pos < analysis.relations.size(); relation_pos++) {
		if (analysis.relations[relation_pos].index < analysis.relations[root].index) {
			root = relation_pos;
		}
	}
	return root;
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

static unique_ptr<ResultDBYannakakisProgram> BuildResultDBYannakakisProgram(Binder &binder,
                                                                            ResultDBSemijoinAnalysis &analysis,
                                                                            idx_t root_relation) {
	auto &properties = binder.GetStatementProperties();
	auto program = make_uniq<ResultDBYannakakisProgram>();
	program->root_relation = root_relation;
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

unique_ptr<LogicalOperator> Binder::PlanResultDBSemijoin(unique_ptr<LogicalOperator> root, BoundSelectNode &statement) {
	auto &properties = GetStatementProperties();
	if (!properties.resultdb.enabled) {
		return root;
	}
	properties.resultdb.execution_strategy = ResultDBStrategy::DECOMPOSE;
	properties.resultdb.join_edges.clear();
	statement.resultdb_yannakakis_program.reset();
	if (properties.resultdb.requested_strategy == ResultDBStrategy::DECOMPOSE) {
		return root;
	}

	string unsupported_reason;
	if (!SupportsResultDBDirectSemijoinSelectNode(statement, unsupported_reason)) {
		if (properties.resultdb.requested_strategy == ResultDBStrategy::AUTO) {
			return root;
		}
		throw BinderException("RESULTDB semijoin strategy does not support this query: %s", unsupported_reason);
	}

	ResultDBSemijoinAnalysis analysis;
	if (!AnalyzeResultDBSemijoin(*root, analysis, context)) {
		if (properties.resultdb.requested_strategy == ResultDBStrategy::AUTO) {
			return root;
		}
		throw BinderException("RESULTDB semijoin strategy does not support this query: %s",
		                      analysis.unsupported_reason);
	}
	if (!ApplyResultDBWhereFilters(analysis, statement.where_clause)) {
		if (properties.resultdb.requested_strategy == ResultDBStrategy::AUTO) {
			return root;
		}
		throw BinderException("RESULTDB semijoin strategy does not support this query: %s",
		                      analysis.unsupported_reason);
	}

	vector<idx_t> parent;
	vector<idx_t> parent_edge;
	vector<idx_t> order;
	vector<vector<std::pair<idx_t, idx_t>>> adjacency;
	auto root_relation = GetResultDBDeterministicRootRelation(analysis);
	if (!BuildResultDBJoinTree(analysis, root_relation, adjacency, parent, parent_edge, order)) {
		if (properties.resultdb.requested_strategy == ResultDBStrategy::AUTO) {
			return root;
		}
		throw BinderException("RESULTDB semijoin strategy does not support this query: %s",
		                      analysis.unsupported_reason);
	}

	statement.resultdb_yannakakis_program = BuildResultDBYannakakisProgram(*this, analysis, root_relation);
	properties.resultdb.execution_strategy = ResultDBStrategy::SEMIJOIN;
	StoreResultDBJoinMetadata(properties.resultdb, analysis);
	return make_uniq_base<LogicalOperator, LogicalDummyScan>(GenerateTableIndex());
}

unique_ptr<LogicalOperator> Binder::CreatePlan(BoundSelectNode &statement) {
	D_ASSERT(statement.from_table.plan);
	auto root = std::move(statement.from_table.plan);
	root = PlanResultDBSemijoin(std::move(root), statement);
	if (GetStatementProperties().resultdb.enabled &&
	    GetStatementProperties().resultdb.execution_strategy == ResultDBStrategy::SEMIJOIN) {
		return root;
	}

	// plan the sample clause
	if (statement.sample_options) {
		root = make_uniq<LogicalSample>(std::move(statement.sample_options), std::move(root));
	}

	if (statement.where_clause) {
		root = PlanFilter(std::move(statement.where_clause), std::move(root));
	}

	if (!statement.aggregates.empty() || !statement.groups.group_expressions.empty() || statement.having) {
		if (!statement.groups.group_expressions.empty()) {
			// visit the groups
			for (auto &group : statement.groups.group_expressions) {
				PlanSubqueries(group, root);
			}
		}
		// now visit all aggregate expressions
		for (auto &expr : statement.aggregates) {
			PlanSubqueries(expr, root);
		}
		// finally create the aggregate node with the group_index and aggregate_index as obtained from the binder
		auto aggregate = make_uniq<LogicalAggregate>(statement.group_index, statement.aggregate_index,
		                                             std::move(statement.aggregates));
		aggregate->groups = std::move(statement.groups.group_expressions);
		aggregate->groupings_index = statement.groupings_index;
		aggregate->grouping_sets = std::move(statement.groups.grouping_sets);
		aggregate->grouping_functions = std::move(statement.grouping_functions);

		aggregate->AddChild(std::move(root));
		root = std::move(aggregate);
	} else if (!statement.groups.grouping_sets.empty()) {
		// edge case: we have grouping sets but no groups or aggregates
		// this can only happen if we have e.g. select 1 from tbl group by ();
		// just output a dummy scan
		root = make_uniq_base<LogicalOperator, LogicalDummyScan>(statement.group_index);
	}

	if (statement.having) {
		PlanSubqueries(statement.having, root);
		auto having = make_uniq<LogicalFilter>(std::move(statement.having));

		having->AddChild(std::move(root));
		root = std::move(having);
	}

	if (!statement.windows.empty()) {
		auto win = make_uniq<LogicalWindow>(statement.window_index);
		win->expressions = std::move(statement.windows);
		// visit the window expressions
		for (auto &expr : win->expressions) {
			PlanSubqueries(expr, root);
		}
		D_ASSERT(!win->expressions.empty());
		win->AddChild(std::move(root));
		root = std::move(win);
	}

	if (statement.qualify) {
		PlanSubqueries(statement.qualify, root);
		auto qualify = make_uniq<LogicalFilter>(std::move(statement.qualify));

		qualify->AddChild(std::move(root));
		root = std::move(qualify);
	}

	for (idx_t i = statement.unnests.size(); i > 0; i--) {
		auto unnest_level = i - 1;
		auto entry = statement.unnests.find(unnest_level);
		if (entry == statement.unnests.end()) {
			throw InternalException("unnests specified at level %d but none were found", unnest_level);
		}
		auto &unnest_node = entry->second;
		auto unnest = make_uniq<LogicalUnnest>(unnest_node.index);
		unnest->expressions = std::move(unnest_node.expressions);
		// visit the unnest expressions
		for (auto &expr : unnest->expressions) {
			PlanSubqueries(expr, root);
		}
		D_ASSERT(!unnest->expressions.empty());
		unnest->AddChild(std::move(root));
		root = std::move(unnest);
	}

	for (auto &expr : statement.select_list) {
		PlanSubqueries(expr, root);
	}

	auto proj = make_uniq<LogicalProjection>(statement.projection_index, std::move(statement.select_list));
	auto &projection = *proj;
	proj->AddChild(std::move(root));
	root = std::move(proj);

	// finish the plan by handling the elements of the QueryNode
	root = VisitQueryNode(statement, std::move(root));

	// add a prune node if necessary
	if (statement.need_prune) {
		D_ASSERT(root);
		vector<unique_ptr<Expression>> prune_expressions;
		for (idx_t i = 0; i < statement.column_count; i++) {
			prune_expressions.push_back(
			    make_uniq<BoundColumnRefExpression>(projection.expressions[i]->GetReturnType(),
			                                        ColumnBinding(statement.projection_index, ProjectionIndex(i))));
		}
		auto prune = make_uniq<LogicalProjection>(statement.prune_index, std::move(prune_expressions));
		prune->AddChild(std::move(root));
		root = std::move(prune);
	}
	return root;
}

} // namespace duckdb
