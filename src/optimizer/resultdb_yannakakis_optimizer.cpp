//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/resultdb_yannakakis_optimizer.cpp
//
//
//===----------------------------------------------------------------------===//

#include "duckdb/optimizer/resultdb_yannakakis_optimizer.hpp"
#include "duckdb/optimizer/resultdb_plan_enumerator.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/optimizer/statistics_propagator.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_cross_product.hpp"
#include "duckdb/planner/operator/logical_distinct.hpp"
#include "duckdb/planner/operator/logical_dummy_scan.hpp"
#include "duckdb/planner/operator/logical_empty_result.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_join.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/resultdb_reduced_plan.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

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

struct ResultDBSemijoinGraphNode {
	vector<TableIndex> tables;
	idx_t estimated_cardinality = 0;
	bool folded = false;
	bool enumerated_fold = false;
	idx_t stable_id = DConstants::INVALID_INDEX;
};

struct ResultDBSemijoinGraphEdge {
	idx_t left_node = DConstants::INVALID_INDEX;
	idx_t right_node = DConstants::INVALID_INDEX;
	vector<JoinCondition> conditions;
};

struct ResultDBSemijoinGraph {
	vector<ResultDBSemijoinGraphNode> nodes;
	vector<ResultDBSemijoinGraphEdge> edges;
	unordered_map<idx_t, idx_t> node_map;
	string unsupported_reason;
};

struct ResultDBColumnKey {
	idx_t table_index = DConstants::INVALID_INDEX;
	idx_t column_index = DConstants::INVALID_INDEX;
	LogicalType comparison_type = LogicalType::INVALID;
};

struct ResultDBCanonicalJoinCondition {
	const JoinCondition *condition;
	idx_t left_column = DConstants::INVALID_INDEX;
	idx_t right_column = DConstants::INVALID_INDEX;
};

struct ResultDBJoinColumnRef {
	const BoundColumnRefExpression *column = nullptr;
	LogicalType comparison_type = LogicalType::INVALID;
	bool requires_cast = false;
};

static bool TryGetResultDBJoinColumnRef(const Expression &expr, ResultDBJoinColumnRef &result) {
	result = ResultDBJoinColumnRef();
	if (expr.GetExpressionType() == ExpressionType::BOUND_COLUMN_REF) {
		auto &column = expr.Cast<BoundColumnRefExpression>();
		result.column = &column;
		result.comparison_type = column.GetReturnType();
		return true;
	}
	if (expr.GetExpressionClass() != ExpressionClass::BOUND_CAST) {
		return false;
	}
	auto &cast = expr.Cast<BoundCastExpression>();
	if (cast.try_cast || cast.child->GetExpressionType() != ExpressionType::BOUND_COLUMN_REF) {
		return false;
	}
	result.column = &cast.child->Cast<BoundColumnRefExpression>();
	result.comparison_type = cast.GetReturnType();
	result.requires_cast = result.column->GetReturnType() != result.comparison_type;
	return true;
}

static LogicalType GetResultDBJoinColumnCastType(const ResultDBJoinColumnRef &column) {
	return column.requires_cast ? column.comparison_type : LogicalType::INVALID;
}

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
	ResultDBJoinColumnRef left;
	ResultDBJoinColumnRef right;
	if (!TryGetResultDBJoinColumnRef(condition.GetLHS(), left) ||
	    !TryGetResultDBJoinColumnRef(condition.GetRHS(), right)) {
		analysis.unsupported_reason =
		    "semijoin join predicates must compare source columns with optional direct casts";
		return false;
	}
	if (left.comparison_type != right.comparison_type) {
		analysis.unsupported_reason = "semijoin join predicate comparison types do not match";
		return false;
	}
	auto &left_colref = *left.column;
	auto &right_colref = *right.column;
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
			ResultDBJoinColumnRef left;
			ResultDBJoinColumnRef right;
			if (!TryGetResultDBJoinColumnRef(condition.GetLHS(), left) ||
			    !TryGetResultDBJoinColumnRef(condition.GetRHS(), right)) {
				throw InternalException("ResultDB join metadata contains an invalid join column");
			}
			ResultDBJoinColumnMetadata column_metadata;
			column_metadata.left_column_index = left.column->binding.column_index.GetIndex();
			column_metadata.right_column_index = right.column->binding.column_index.GetIndex();
			edge_metadata.columns.push_back(column_metadata);
		}
		properties.join_edges.push_back(std::move(edge_metadata));
	}
}

static bool SameResultDBColumnKey(const ResultDBColumnKey &left, const ResultDBColumnKey &right) {
	return left.table_index == right.table_index && left.column_index == right.column_index &&
	       left.comparison_type == right.comparison_type;
}

static idx_t GetOrCreateResultDBColumnKey(vector<ResultDBColumnKey> &columns, const ResultDBJoinColumnRef &column) {
	ResultDBColumnKey key;
	key.table_index = column.column->binding.table_index.index;
	key.column_index = column.column->binding.column_index.GetIndex();
	key.comparison_type = column.comparison_type;
	auto entry = std::find_if(columns.begin(), columns.end(), [&](const ResultDBColumnKey &candidate) {
		return SameResultDBColumnKey(candidate, key);
	});
	if (entry != columns.end()) {
		return NumericCast<idx_t>(entry - columns.begin());
	}
	auto result = columns.size();
	columns.push_back(key);
	return result;
}

static idx_t FindResultDBUnionParent(vector<idx_t> &parents, idx_t entry) {
	if (parents[entry] == entry) {
		return entry;
	}
	parents[entry] = FindResultDBUnionParent(parents, parents[entry]);
	return parents[entry];
}

static bool UnionResultDBColumns(vector<idx_t> &parents, idx_t left, idx_t right) {
	auto left_parent = FindResultDBUnionParent(parents, left);
	auto right_parent = FindResultDBUnionParent(parents, right);
	if (left_parent == right_parent) {
		return false;
	}
	if (left_parent > right_parent) {
		std::swap(left_parent, right_parent);
	}
	parents[right_parent] = left_parent;
	return true;
}

static void CanonicalizeResultDBJoinConditions(ResultDBSemijoinAnalysis &analysis) {
	vector<ResultDBColumnKey> columns;
	vector<ResultDBCanonicalJoinCondition> conditions;
	for (auto &edge : analysis.edges) {
		for (auto &condition : edge.conditions) {
			ResultDBJoinColumnRef left;
			ResultDBJoinColumnRef right;
			if (!TryGetResultDBJoinColumnRef(condition.GetLHS(), left) ||
			    !TryGetResultDBJoinColumnRef(condition.GetRHS(), right)) {
				throw InternalException("ResultDB canonicalization contains an invalid join column");
			}
			ResultDBCanonicalJoinCondition entry;
			entry.condition = &condition;
			entry.left_column = GetOrCreateResultDBColumnKey(columns, left);
			entry.right_column = GetOrCreateResultDBColumnKey(columns, right);
			conditions.push_back(entry);
		}
	}

	vector<idx_t> parents;
	parents.reserve(columns.size());
	for (idx_t column_idx = 0; column_idx < columns.size(); column_idx++) {
		parents.push_back(column_idx);
	}

	ResultDBSemijoinAnalysis canonical;
	for (auto &condition : conditions) {
		if (!UnionResultDBColumns(parents, condition.left_column, condition.right_column)) {
			continue;
		}
		if (!AddResultDBJoinCondition(canonical, *condition.condition)) {
			throw InternalException("ResultDB canonical join condition became unsupported: %s",
			                        canonical.unsupported_reason);
		}
	}
	analysis.edges = std::move(canonical.edges);
}

static bool ResultDBNodeContainsTable(const ResultDBSemijoinGraphNode &node, idx_t table_index) {
	for (auto table : node.tables) {
		if (table.index == table_index) {
			return true;
		}
	}
	return false;
}

static idx_t ResultDBNodeStableId(const ResultDBSemijoinGraphNode &node) {
	idx_t stable_id = DConstants::INVALID_INDEX;
	for (auto table : node.tables) {
		if (stable_id == DConstants::INVALID_INDEX || table.index < stable_id) {
			stable_id = table.index;
		}
	}
	return stable_id;
}

static void SortResultDBNodeTables(vector<TableIndex> &tables) {
	std::sort(tables.begin(), tables.end(), [](const TableIndex &left, const TableIndex &right) {
		return left.index < right.index;
	});
	tables.erase(std::unique(tables.begin(), tables.end(), [](const TableIndex &left, const TableIndex &right) {
		             return left.index == right.index;
	             }),
	             tables.end());
}

static ResultDBSemijoinGraphEdge &GetResultDBGraphEdge(ResultDBSemijoinGraph &graph, idx_t left_node,
                                                       idx_t right_node) {
	for (auto &edge : graph.edges) {
		if ((edge.left_node == left_node && edge.right_node == right_node) ||
		    (edge.left_node == right_node && edge.right_node == left_node)) {
			return edge;
		}
	}
	ResultDBSemijoinGraphEdge edge;
	edge.left_node = left_node;
	edge.right_node = right_node;
	graph.edges.push_back(std::move(edge));
	return graph.edges.back();
}

static bool AddResultDBGraphCondition(ResultDBSemijoinGraph &graph, ResultDBSemijoinGraphEdge &edge,
                                      const JoinCondition &condition) {
	auto copied_condition = condition.Copy();
	ResultDBJoinColumnRef left;
	ResultDBJoinColumnRef right;
	if (!TryGetResultDBJoinColumnRef(copied_condition.GetLHS(), left) ||
	    !TryGetResultDBJoinColumnRef(copied_condition.GetRHS(), right)) {
		graph.unsupported_reason = "semijoin join graph contains an invalid join column";
		return false;
	}
	auto left_table = left.column->binding.table_index.index;
	auto right_table = right.column->binding.table_index.index;
	bool lhs_in_left = ResultDBNodeContainsTable(graph.nodes[edge.left_node], left_table);
	bool rhs_in_right = ResultDBNodeContainsTable(graph.nodes[edge.right_node], right_table);
	bool lhs_in_right = ResultDBNodeContainsTable(graph.nodes[edge.right_node], left_table);
	bool rhs_in_left = ResultDBNodeContainsTable(graph.nodes[edge.left_node], right_table);
	if (lhs_in_left && rhs_in_right) {
		edge.conditions.push_back(std::move(copied_condition));
		return true;
	}
	if (lhs_in_right && rhs_in_left) {
		copied_condition.Swap();
		edge.conditions.push_back(std::move(copied_condition));
		return true;
	}
	graph.unsupported_reason = "semijoin join graph condition does not match folded graph endpoints";
	return false;
}

static bool RebuildResultDBGraphEdges(const ResultDBSemijoinAnalysis &analysis, ResultDBSemijoinGraph &graph) {
	graph.node_map.clear();
	for (idx_t node_idx = 0; node_idx < graph.nodes.size(); node_idx++) {
		for (auto table : graph.nodes[node_idx].tables) {
			graph.node_map[table.index] = node_idx;
		}
	}

	graph.edges.clear();
	for (auto &analysis_edge : analysis.edges) {
		auto left_entry = graph.node_map.find(analysis_edge.left_table.index);
		auto right_entry = graph.node_map.find(analysis_edge.right_table.index);
		if (left_entry == graph.node_map.end() || right_entry == graph.node_map.end()) {
			graph.unsupported_reason = "semijoin join graph references an unknown relation occurrence";
			return false;
		}
		if (left_entry->second == right_entry->second) {
			continue;
		}
		auto &graph_edge = GetResultDBGraphEdge(graph, left_entry->second, right_entry->second);
		for (auto &condition : analysis_edge.conditions) {
			if (!AddResultDBGraphCondition(graph, graph_edge, condition)) {
				return false;
			}
		}
	}
	return true;
}

static vector<vector<std::pair<idx_t, idx_t>>> BuildResultDBGraphAdjacency(const ResultDBSemijoinGraph &graph) {
	vector<vector<std::pair<idx_t, idx_t>>> adjacency;
	adjacency.resize(graph.nodes.size());
	for (idx_t edge_idx = 0; edge_idx < graph.edges.size(); edge_idx++) {
		auto &edge = graph.edges[edge_idx];
		adjacency[edge.left_node].emplace_back(edge.right_node, edge_idx);
		adjacency[edge.right_node].emplace_back(edge.left_node, edge_idx);
	}
	return adjacency;
}

static bool ResultDBGraphIsConnected(const ResultDBSemijoinGraph &graph,
                                     const vector<vector<std::pair<idx_t, idx_t>>> &adjacency) {
	if (graph.nodes.empty()) {
		return false;
	}
	vector<bool> visited(graph.nodes.size(), false);
	vector<idx_t> order;
	order.push_back(0);
	visited[0] = true;
	for (idx_t order_idx = 0; order_idx < order.size(); order_idx++) {
		auto current = order[order_idx];
		for (auto &entry : adjacency[current]) {
			auto next = entry.first;
			if (visited[next]) {
				continue;
			}
			visited[next] = true;
			order.push_back(next);
		}
	}
	return order.size() == graph.nodes.size();
}

static bool IsBetterResultDBGraphNodeChoice(const ResultDBSemijoinGraph &graph,
                                            const vector<vector<std::pair<idx_t, idx_t>>> &adjacency,
                                            idx_t candidate, idx_t current_best) {
	if (current_best == DConstants::INVALID_INDEX) {
		return true;
	}
	auto candidate_degree = adjacency[candidate].size();
	auto best_degree = adjacency[current_best].size();
	if (candidate_degree != best_degree) {
		return candidate_degree > best_degree;
	}
	auto candidate_cardinality = graph.nodes[candidate].estimated_cardinality;
	auto best_cardinality = graph.nodes[current_best].estimated_cardinality;
	if (candidate_cardinality != best_cardinality) {
		return candidate_cardinality < best_cardinality;
	}
	return graph.nodes[candidate].stable_id < graph.nodes[current_best].stable_id;
}

static idx_t ChooseResultDBFoldNode(const ResultDBSemijoinGraph &graph,
                                    const vector<vector<std::pair<idx_t, idx_t>>> &adjacency) {
	idx_t best_node = DConstants::INVALID_INDEX;
	for (idx_t node_idx = 0; node_idx < graph.nodes.size(); node_idx++) {
		if (adjacency[node_idx].empty()) {
			continue;
		}
		if (IsBetterResultDBGraphNodeChoice(graph, adjacency, node_idx, best_node)) {
			best_node = node_idx;
		}
	}
	return best_node;
}

static idx_t ChooseResultDBFoldNeighbor(const ResultDBSemijoinGraph &graph,
                                        const vector<vector<std::pair<idx_t, idx_t>>> &adjacency, idx_t node_idx) {
	idx_t best_neighbor = DConstants::INVALID_INDEX;
	for (auto &entry : adjacency[node_idx]) {
		auto candidate = entry.first;
		if (IsBetterResultDBGraphNodeChoice(graph, adjacency, candidate, best_neighbor)) {
			best_neighbor = candidate;
		}
	}
	return best_neighbor;
}

static bool FoldResultDBGraphOnce(const ResultDBSemijoinAnalysis &analysis, ResultDBSemijoinGraph &graph,
                                  const vector<vector<std::pair<idx_t, idx_t>>> &adjacency) {
	auto left_node = ChooseResultDBFoldNode(graph, adjacency);
	if (left_node == DConstants::INVALID_INDEX) {
		graph.unsupported_reason = "semijoin could not choose a cyclic graph node to fold";
		return false;
	}
	auto right_node = ChooseResultDBFoldNeighbor(graph, adjacency, left_node);
	if (right_node == DConstants::INVALID_INDEX) {
		graph.unsupported_reason = "semijoin could not choose a cyclic graph neighbor to fold";
		return false;
	}

	ResultDBSemijoinGraphNode folded_node;
	folded_node.folded = true;
	folded_node.tables = graph.nodes[left_node].tables;
	folded_node.tables.insert(folded_node.tables.end(), graph.nodes[right_node].tables.begin(),
	                          graph.nodes[right_node].tables.end());
	SortResultDBNodeTables(folded_node.tables);
	folded_node.stable_id = ResultDBNodeStableId(folded_node);
	folded_node.estimated_cardinality =
	    std::max(graph.nodes[left_node].estimated_cardinality, graph.nodes[right_node].estimated_cardinality);

	vector<ResultDBSemijoinGraphNode> new_nodes;
	new_nodes.reserve(graph.nodes.size() - 1);
	for (idx_t node_idx = 0; node_idx < graph.nodes.size(); node_idx++) {
		if (node_idx == left_node || node_idx == right_node) {
			continue;
		}
		new_nodes.push_back(std::move(graph.nodes[node_idx]));
	}
	new_nodes.push_back(std::move(folded_node));
	graph.nodes = std::move(new_nodes);
	return RebuildResultDBGraphEdges(analysis, graph);
}

static bool BuildResultDBInitialGraph(const ResultDBSemijoinAnalysis &analysis, ResultDBSemijoinGraph &graph) {
	if (analysis.relations.size() < 2) {
		graph.unsupported_reason = "semijoin requires at least two joined relation occurrences";
		return false;
	}

	graph.nodes.clear();
	graph.nodes.reserve(analysis.relations.size());
	for (idx_t relation_idx = 0; relation_idx < analysis.relations.size(); relation_idx++) {
		ResultDBSemijoinGraphNode node;
		node.tables.push_back(analysis.relations[relation_idx]);
		node.estimated_cardinality = analysis.estimated_cardinalities[relation_idx];
		node.stable_id = analysis.relations[relation_idx].index;
		graph.nodes.push_back(std::move(node));
	}
	if (!RebuildResultDBGraphEdges(analysis, graph)) {
		return false;
	}
	auto adjacency = BuildResultDBGraphAdjacency(graph);
	if (!ResultDBGraphIsConnected(graph, adjacency)) {
		graph.unsupported_reason = "semijoin currently supports only connected join graphs";
		return false;
	}
	return true;
}

static bool FoldResultDBGraphHeuristically(const ResultDBSemijoinAnalysis &analysis,
	                                       ResultDBSemijoinGraph &graph) {

	while (true) {
		auto adjacency = BuildResultDBGraphAdjacency(graph);
		if (!ResultDBGraphIsConnected(graph, adjacency)) {
			graph.unsupported_reason = "semijoin currently supports only connected join graphs";
			return false;
		}
		if (graph.edges.size() < graph.nodes.size() - 1) {
			graph.unsupported_reason = "semijoin currently supports only connected join graphs";
			return false;
		}
		if (graph.edges.size() == graph.nodes.size() - 1) {
			return true;
		}
		if (!FoldResultDBGraphOnce(analysis, graph, adjacency)) {
			return false;
		}
	}
}

static bool BuildResultDBSemijoinGraph(const ResultDBSemijoinAnalysis &analysis, ResultDBSemijoinGraph &graph) {
	return BuildResultDBInitialGraph(analysis, graph) && FoldResultDBGraphHeuristically(analysis, graph);
}

static bool BuildResultDBJoinTree(ResultDBSemijoinGraph &graph, idx_t root_relation,
                                  vector<vector<std::pair<idx_t, idx_t>>> &adjacency, vector<idx_t> &parent,
                                  vector<idx_t> &parent_edge, vector<idx_t> &order) {
	auto relation_count = graph.nodes.size();
	if (relation_count < 2) {
		graph.unsupported_reason = "semijoin requires at least two joined relation occurrences";
		return false;
	}
	if (graph.edges.size() != relation_count - 1) {
		graph.unsupported_reason = "semijoin could not fold the join graph into an acyclic graph";
		return false;
	}

	adjacency = BuildResultDBGraphAdjacency(graph);
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
		graph.unsupported_reason = "semijoin currently supports only connected acyclic join graphs";
		return false;
	}
	return true;
}

static bool ResultDBNodeContainsOutputTable(const ResultDBSemijoinGraphNode &node,
                                            const ResultDBProperties &properties) {
	for (auto &table : properties.tables) {
		if (ResultDBNodeContainsTable(node, table.table_index)) {
			return true;
		}
	}
	return false;
}

static bool ChooseResultDBRootRelation(ResultDBSemijoinGraph &graph, const ResultDBProperties &properties,
                                       vector<idx_t> &best_parent, vector<idx_t> &best_parent_edge,
                                       vector<idx_t> &best_order, idx_t &best_root) {
	auto adjacency = BuildResultDBGraphAdjacency(graph);
	bool has_output_candidate = false;
	for (auto &node : graph.nodes) {
		if (ResultDBNodeContainsOutputTable(node, properties)) {
			has_output_candidate = true;
			break;
		}
	}

	best_root = DConstants::INVALID_INDEX;
	for (idx_t node_idx = 0; node_idx < graph.nodes.size(); node_idx++) {
		if (has_output_candidate && !ResultDBNodeContainsOutputTable(graph.nodes[node_idx], properties)) {
			continue;
		}
		if (IsBetterResultDBGraphNodeChoice(graph, adjacency, node_idx, best_root)) {
			best_root = node_idx;
		}
	}
	if (best_root == DConstants::INVALID_INDEX) {
		graph.unsupported_reason = "semijoin could not choose a Yannakakis root";
		return false;
	}

	vector<vector<std::pair<idx_t, idx_t>>> tree_adjacency;
	return BuildResultDBJoinTree(graph, best_root, tree_adjacency, best_parent, best_parent_edge, best_order);
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

static bool SameResultDBSourceColumn(const ResultDBYannakakisSourceColumn &left,
                                     const ResultDBYannakakisSourceColumn &right) {
	return left.table_index == right.table_index && left.source_column_index == right.source_column_index &&
	       left.cast_type == right.cast_type;
}

static void AddResultDBWorkingColumn(vector<vector<ResultDBYannakakisSourceColumn>> &needed_columns, idx_t relation_pos,
                                     idx_t table_index, idx_t source_idx,
                                     const LogicalType &cast_type = LogicalType::INVALID) {
	ResultDBYannakakisSourceColumn source_column;
	source_column.table_index = table_index;
	source_column.source_column_index = source_idx;
	source_column.cast_type = cast_type;
	auto &columns = needed_columns[relation_pos];
	auto entry = std::find_if(columns.begin(), columns.end(), [&](const ResultDBYannakakisSourceColumn &current) {
		return SameResultDBSourceColumn(current, source_column);
	});
	if (entry == columns.end()) {
		columns.push_back(source_column);
	}
}

static void SortResultDBSourceColumns(vector<ResultDBYannakakisSourceColumn> &columns) {
	std::sort(columns.begin(), columns.end(),
	          [](const ResultDBYannakakisSourceColumn &left, const ResultDBYannakakisSourceColumn &right) {
		          if (left.table_index != right.table_index) {
			          return left.table_index < right.table_index;
		          }
		          if (left.source_column_index != right.source_column_index) {
			          return left.source_column_index < right.source_column_index;
		          }
		          return left.cast_type.ToString() < right.cast_type.ToString();
	          });
	columns.erase(std::unique(columns.begin(), columns.end(), SameResultDBSourceColumn), columns.end());
}

static idx_t GetResultDBWorkingColumnIndex(const vector<ResultDBYannakakisSourceColumn> &source_columns,
                                           idx_t table_index, idx_t source_idx,
                                           const LogicalType &cast_type = LogicalType::INVALID) {
	auto entry = std::find_if(source_columns.begin(), source_columns.end(),
	                          [&](const ResultDBYannakakisSourceColumn &source_column) {
		                          return source_column.table_index == table_index &&
		                                 source_column.source_column_index == source_idx &&
		                                 source_column.cast_type == cast_type;
	                          });
	if (entry == source_columns.end()) {
		throw InternalException("ResultDB Yannakakis working relation is missing a required source column");
	}
	return NumericCast<idx_t>(entry - source_columns.begin());
}

static unique_ptr<LogicalOperator> TakeResultDBRelationPlan(ResultDBSemijoinAnalysis &analysis, idx_t table_index) {
	auto entry = analysis.relation_plans.find(table_index);
	if (entry == analysis.relation_plans.end() || !entry->second) {
		throw InternalException("ResultDB Yannakakis relation plan is missing");
	}
	return std::move(entry->second);
}

static LogicalType GetResultDBSourceColumnType(ResultDBSemijoinAnalysis &analysis, idx_t table_index,
                                               idx_t source_idx) {
	auto entry = analysis.relation_plans.find(table_index);
	if (entry == analysis.relation_plans.end() || !entry->second) {
		throw InternalException("ResultDB Yannakakis relation plan is missing");
	}
	entry->second->ResolveOperatorTypes();
	if (source_idx >= entry->second->types.size()) {
		throw InternalException("ResultDB Yannakakis source column index is out of range");
	}
	return entry->second->types[source_idx];
}

static bool ResultDBTableSetContains(const unordered_set<idx_t> &tables, idx_t table_index) {
	return tables.find(table_index) != tables.end();
}

static vector<JoinCondition>
CollectResultDBInternalJoinConditions(const ResultDBSemijoinAnalysis &analysis, const unordered_set<idx_t> &left_tables,
                                      idx_t right_table) {
	vector<JoinCondition> conditions;
	for (auto &edge : analysis.edges) {
		for (auto &condition : edge.conditions) {
			auto copied_condition = condition.Copy();
			ResultDBJoinColumnRef left;
			ResultDBJoinColumnRef right;
			if (!TryGetResultDBJoinColumnRef(copied_condition.GetLHS(), left) ||
			    !TryGetResultDBJoinColumnRef(copied_condition.GetRHS(), right)) {
				throw InternalException("ResultDB folded join contains an invalid join column");
			}
			auto left_table = left.column->binding.table_index.index;
			auto condition_right_table = right.column->binding.table_index.index;
			if (ResultDBTableSetContains(left_tables, left_table) && condition_right_table == right_table) {
				conditions.push_back(std::move(copied_condition));
				continue;
			}
			if (left_table == right_table && ResultDBTableSetContains(left_tables, condition_right_table)) {
				copied_condition.Swap();
				conditions.push_back(std::move(copied_condition));
			}
		}
	}
	return conditions;
}

class DuckDBResultDBEnumerationCostProvider : public ResultDBEnumerationCostProvider {
public:
	DuckDBResultDBEnumerationCostProvider(Binder &binder_p, ClientContext &context_p,
	                                    ResultDBSemijoinAnalysis &analysis_p,
	                                    const ResultDBSemijoinGraph &graph_p,
	                                    const ResultDBProperties &properties_p)
	    : binder(binder_p), context(context_p), analysis(analysis_p), graph(graph_p), properties(properties_p) {
	}

	double Cardinality(const vector<idx_t> &nodes) override {
		vector<idx_t> table_indexes;
		for (auto node_idx : nodes) {
			if (node_idx >= graph.nodes.size()) {
				throw InternalException("ResultDB cost provider received an invalid graph node");
			}
			for (auto table : graph.nodes[node_idx].tables) {
				table_indexes.push_back(table.index);
			}
		}
		std::sort(table_indexes.begin(), table_indexes.end());
		table_indexes.erase(std::unique(table_indexes.begin(), table_indexes.end()), table_indexes.end());
		string key;
		for (auto table_index : table_indexes) {
			key += std::to_string(table_index) + ",";
		}
		auto cached = cardinalities.find(key);
		if (cached != cardinalities.end()) {
			return cached->second;
		}
		double result = 0;
		for (auto table_index : table_indexes) {
			auto relation_entry = analysis.relation_map.find(table_index);
			if (relation_entry == analysis.relation_map.end()) {
				throw InternalException("ResultDB cost provider could not map a relation occurrence");
			}
			// LogicalOperator::EstimateCardinality is DuckDB's available estimate at
			// this pre-join-order stage. For a connected fold it uses the maximum
			// child estimate, matching the estimate attached to the logical join.
			result = std::max(result, static_cast<double>(analysis.estimated_cardinalities[relation_entry->second]));
		}
		if (table_indexes.size() > 1) {
			auto root_entry = analysis.relation_plans.find(table_indexes[0]);
			if (root_entry == analysis.relation_plans.end() || !root_entry->second) {
				throw InternalException("ResultDB cost provider is missing a relation plan");
			}
			auto root = root_entry->second->Copy(context);
			unordered_set<idx_t> visited {table_indexes[0]};
			while (visited.size() < table_indexes.size()) {
				bool added = false;
				for (auto table_index : table_indexes) {
					if (visited.find(table_index) != visited.end()) {
						continue;
					}
					auto conditions = CollectResultDBInternalJoinConditions(analysis, visited, table_index);
					if (conditions.empty()) {
						continue;
					}
					auto right_entry = analysis.relation_plans.find(table_index);
					if (right_entry == analysis.relation_plans.end() || !right_entry->second) {
						throw InternalException("ResultDB cost provider is missing a relation plan");
					}
					root = LogicalComparisonJoin::CreateJoin(JoinType::INNER, JoinRefType::REGULAR, std::move(root),
					                                         right_entry->second->Copy(context), std::move(conditions));
					visited.insert(table_index);
					added = true;
					break;
				}
				if (!added) {
					throw InternalException("ResultDB cost provider received a disconnected relation subset");
				}
			}
			Optimizer optimizer(binder, context);
			StatisticsPropagator propagator(optimizer, *root);
			auto node_statistics = propagator.PropagateStatistics(root);
			if (node_statistics && node_statistics->has_estimated_cardinality) {
				result = static_cast<double>(node_statistics->estimated_cardinality);
			}
		}
		cardinalities[key] = result;
		return result;
	}

	double PayloadWidth(const vector<idx_t> &nodes) override {
		unordered_set<idx_t> table_indexes;
		for (auto node_idx : nodes) {
			for (auto table : graph.nodes[node_idx].tables) {
				table_indexes.insert(table.index);
			}
		}
		double width = 0;
		for (auto &table : properties.tables) {
			if (table_indexes.find(table.table_index) == table_indexes.end()) {
				continue;
			}
			for (auto &column : table.columns) {
				width += static_cast<double>(GetTypeIdSize(column.type.InternalType()));
			}
		}
		return std::max(width, 1.0);
	}

private:
	Binder &binder;
	ClientContext &context;
	ResultDBSemijoinAnalysis &analysis;
	const ResultDBSemijoinGraph &graph;
	const ResultDBProperties &properties;
	unordered_map<string, double> cardinalities;
};

static ResultDBEnumerationInput BuildResultDBEnumerationInput(const ResultDBSemijoinGraph &graph,
	                                                          const ResultDBProperties &properties) {
	ResultDBEnumerationInput result;
	for (auto &graph_node : graph.nodes) {
		ResultDBEnumerationNode node;
		node.stable_id = graph_node.stable_id;
		node.cardinality = static_cast<double>(graph_node.estimated_cardinality);
		node.requested = ResultDBNodeContainsOutputTable(graph_node, properties);
		result.nodes.push_back(node);
	}
	for (auto &graph_edge : graph.edges) {
		result.edges.push_back({graph_edge.left_node, graph_edge.right_node});
	}
	return result;
}

static bool ApplyResultDBFoldPartition(const ResultDBSemijoinAnalysis &analysis, ResultDBSemijoinGraph &graph,
	                                   const vector<vector<idx_t>> &folds,
	                                   ResultDBEnumerationCostProvider &costs) {
	vector<uint8_t> assigned(graph.nodes.size(), 0);
	vector<vector<idx_t>> partition;
	for (auto fold : folds) {
		if (fold.size() < 2) {
			continue;
		}
		if (fold.size() > 20) {
			graph.unsupported_reason = "exact ResultDB fold join enumeration supports at most 20 relations per fold";
			return false;
		}
		std::sort(fold.begin(), fold.end());
		for (auto node_idx : fold) {
			if (node_idx >= graph.nodes.size() || assigned[node_idx]) {
				graph.unsupported_reason = "TDFold produced overlapping or invalid folds";
				return false;
			}
			assigned[node_idx] = 1;
		}
		partition.push_back(std::move(fold));
	}
	for (idx_t node_idx = 0; node_idx < graph.nodes.size(); node_idx++) {
		if (!assigned[node_idx]) {
			partition.push_back({node_idx});
		}
	}
	std::sort(partition.begin(), partition.end(), [](const auto &left, const auto &right) {
		return left.front() < right.front();
	});

	vector<ResultDBSemijoinGraphNode> contracted_nodes;
	for (auto &group : partition) {
		ResultDBSemijoinGraphNode node;
		for (auto node_idx : group) {
			node.tables.insert(node.tables.end(), graph.nodes[node_idx].tables.begin(), graph.nodes[node_idx].tables.end());
		}
		SortResultDBNodeTables(node.tables);
		node.folded = node.tables.size() > 1;
		node.enumerated_fold = node.folded;
		node.stable_id = ResultDBNodeStableId(node);
		node.estimated_cardinality = static_cast<idx_t>(std::ceil(costs.Cardinality(group)));
		contracted_nodes.push_back(std::move(node));
	}
	graph.nodes = std::move(contracted_nodes);
	return RebuildResultDBGraphEdges(analysis, graph);
}

static unique_ptr<LogicalOperator> BuildResultDBFoldJoinPlan(ResultDBSemijoinAnalysis &analysis,
                                                             const ResultDBSemijoinGraphNode &node) {
	if (node.tables.empty()) {
		throw InternalException("ResultDB Yannakakis graph node has no source tables");
	}
	auto tables = node.tables;
	SortResultDBNodeTables(tables);
	if (node.enumerated_fold) {
		if (tables.size() > 20) {
			throw InvalidInputException("Exact ResultDB fold join enumeration supports at most 20 relations per fold");
		}
		auto full = (uint64_t(1) << tables.size()) - 1;
		vector<double> cardinality(full + 1, 0);
		vector<double> cost(full + 1, std::numeric_limits<double>::infinity());
		vector<uint64_t> split(full + 1, 0);
		for (idx_t pos = 0; pos < tables.size(); pos++) {
			auto relation_pos = analysis.relation_map.at(tables[pos].index);
			cardinality[uint64_t(1) << pos] = analysis.estimated_cardinalities[relation_pos];
			cost[uint64_t(1) << pos] = 0;
		}
		for (uint64_t subset = 1; subset <= full; subset++) {
			if ((subset & (subset - 1)) == 0) {
				continue;
			}
			for (idx_t pos = 0; pos < tables.size(); pos++) {
				if ((subset >> pos) & 1) {
					cardinality[subset] = std::max(cardinality[subset],
					                               cardinality[uint64_t(1) << pos]);
				}
			}
			for (uint64_t left = (subset - 1) & subset; left; left = (left - 1) & subset) {
				auto right = subset ^ left;
				if (!right || left > right || !std::isfinite(cost[left]) || !std::isfinite(cost[right])) {
					continue;
				}
				bool connected = false;
				for (auto &edge : analysis.edges) {
					auto left_pos = std::find_if(tables.begin(), tables.end(), [&](TableIndex table) {
						return table == edge.left_table;
					});
					auto right_pos = std::find_if(tables.begin(), tables.end(), [&](TableIndex table) {
						return table == edge.right_table;
					});
					if (left_pos == tables.end() || right_pos == tables.end()) {
						continue;
					}
					auto left_bit = uint64_t(1) << NumericCast<idx_t>(left_pos - tables.begin());
					auto right_bit = uint64_t(1) << NumericCast<idx_t>(right_pos - tables.begin());
					if ((subset & left_bit) && (subset & right_bit) &&
					    static_cast<bool>(left & left_bit) != static_cast<bool>(left & right_bit)) {
						connected = true;
						break;
					}
				}
				if (!connected) {
					continue;
				}
				auto candidate = cost[left] + cost[right] + cardinality[subset];
				if (candidate < cost[subset] || (candidate == cost[subset] && left < split[subset])) {
					cost[subset] = candidate;
					split[subset] = left;
				}
			}
		}
		if (!std::isfinite(cost[full])) {
			throw InternalException("ResultDB exact fold join enumeration found no connected plan");
		}
		std::function<unique_ptr<LogicalOperator>(uint64_t)> build = [&](uint64_t subset) {
			if ((subset & (subset - 1)) == 0) {
				idx_t pos = 0;
				while (((subset >> pos) & 1) == 0) {
					pos++;
				}
				return TakeResultDBRelationPlan(analysis, tables[pos].index);
			}
			auto left_mask = split[subset];
			auto right_mask = subset ^ left_mask;
			unordered_set<idx_t> left_tables;
			for (idx_t pos = 0; pos < tables.size(); pos++) {
				if ((left_mask >> pos) & 1) {
					left_tables.insert(tables[pos].index);
				}
			}
			vector<JoinCondition> conditions;
			for (auto &edge : analysis.edges) {
				auto edge_left_pos = std::find_if(tables.begin(), tables.end(), [&](TableIndex table) {
					return table == edge.left_table;
				});
				auto edge_right_pos = std::find_if(tables.begin(), tables.end(), [&](TableIndex table) {
					return table == edge.right_table;
				});
				if (edge_left_pos == tables.end() || edge_right_pos == tables.end()) {
					continue;
				}
				auto edge_left_bit = uint64_t(1) << NumericCast<idx_t>(edge_left_pos - tables.begin());
				auto edge_right_bit = uint64_t(1) << NumericCast<idx_t>(edge_right_pos - tables.begin());
				if (!(subset & edge_left_bit) || !(subset & edge_right_bit)) {
					continue;
				}
				bool edge_left_in_left = left_tables.find(edge.left_table.index) != left_tables.end();
				bool edge_right_in_left = left_tables.find(edge.right_table.index) != left_tables.end();
				if (edge_left_in_left == edge_right_in_left) {
					continue;
				}
				for (auto &condition : edge.conditions) {
					auto copied = condition.Copy();
					if (!edge_left_in_left) {
						copied.Swap();
					}
					conditions.push_back(std::move(copied));
				}
			}
			if (conditions.empty()) {
				throw InternalException("ResultDB exact fold join split has no join condition");
			}
			return LogicalComparisonJoin::CreateJoin(JoinType::INNER, JoinRefType::REGULAR, build(left_mask),
			                                         build(right_mask), std::move(conditions));
		};
		return build(full);
	}
	auto root_table = tables[0].index;
	auto root = TakeResultDBRelationPlan(analysis, root_table);
	unordered_set<idx_t> visited_tables;
	visited_tables.insert(root_table);

	while (visited_tables.size() < tables.size()) {
		bool added_table = false;
		for (auto candidate_table : tables) {
			if (ResultDBTableSetContains(visited_tables, candidate_table.index)) {
				continue;
			}
			auto conditions = CollectResultDBInternalJoinConditions(analysis, visited_tables, candidate_table.index);
			if (conditions.empty()) {
				continue;
			}
			auto right = TakeResultDBRelationPlan(analysis, candidate_table.index);
			root = LogicalComparisonJoin::CreateJoin(JoinType::INNER, JoinRefType::REGULAR, std::move(root),
			                                         std::move(right), std::move(conditions));
			visited_tables.insert(candidate_table.index);
			added_table = true;
			break;
		}
		if (!added_table) {
			throw InternalException("ResultDB Yannakakis folded relation is internally disconnected");
		}
	}
	return root;
}

static unique_ptr<LogicalOperator> BuildResultDBNodeMaterializationPlan(Binder &binder,
                                                                        ResultDBSemijoinAnalysis &analysis,
                                                                        const ResultDBSemijoinGraphNode &node,
                                                                        ResultDBYannakakisRelation &relation) {
	vector<unique_ptr<Expression>> projection_expressions;
	for (idx_t working_idx = 0; working_idx < relation.source_columns.size(); working_idx++) {
		auto &source_column = relation.source_columns[working_idx];
		auto source_type =
		    GetResultDBSourceColumnType(analysis, source_column.table_index, source_column.source_column_index);
		auto type = source_column.cast_type == LogicalType::INVALID ? source_type : source_column.cast_type;
		auto name = StringUtil::Format("resultdb_%llu_%llu_%llu", source_column.table_index,
		                               source_column.source_column_index, working_idx);
		relation.names.push_back(name);
		relation.types.push_back(type);
		unique_ptr<Expression> expression = make_uniq<BoundColumnRefExpression>(
		    name, source_type, ColumnBinding(TableIndex(source_column.table_index),
		                                     ProjectionIndex(source_column.source_column_index)));
		if (source_column.cast_type != LogicalType::INVALID) {
			expression = BoundCastExpression::AddDefaultCastToType(std::move(expression), source_column.cast_type);
		}
		projection_expressions.push_back(std::move(expression));
	}

	auto base_plan = BuildResultDBFoldJoinPlan(analysis, node);
	relation.preserve_join_order = node.enumerated_fold;
	auto projection_index = binder.GenerateTableIndex();
	auto projection = make_uniq<LogicalProjection>(projection_index, std::move(projection_expressions));
	projection->AddChild(std::move(base_plan));
	if (!node.folded && node.tables.size() == 1) {
		return std::move(projection);
	}

	auto distinct = make_uniq<LogicalDistinct>(
	    BuildResultDBDistinctTargets(relation.names, relation.types, projection_index), DistinctType::DISTINCT);
	distinct->AddChild(std::move(projection));
	return std::move(distinct);
}

static unique_ptr<ResultDBYannakakisProgram>
BuildResultDBYannakakisProgram(Binder &binder, ResultDBSemijoinAnalysis &analysis, ResultDBSemijoinGraph &graph,
                               idx_t root_relation,
	                           vector<idx_t> parent, vector<idx_t> parent_edge, vector<idx_t> order,
	                           vector<vector<idx_t>> bottom_up_children = {},
	                           vector<vector<idx_t>> top_down_children = {}) {
	auto &properties = binder.GetStatementProperties();
	auto program = make_uniq<ResultDBYannakakisProgram>();
	program->root_relation = root_relation;
	program->parent = std::move(parent);
	program->parent_edge = std::move(parent_edge);
	program->order = std::move(order);
	program->bottom_up_children = std::move(bottom_up_children);
	program->top_down_children = std::move(top_down_children);
	program->relations.resize(graph.nodes.size());

	vector<vector<ResultDBYannakakisSourceColumn>> needed_columns(graph.nodes.size());
	for (auto &edge : graph.edges) {
		for (auto &condition : edge.conditions) {
			ResultDBJoinColumnRef left;
			ResultDBJoinColumnRef right;
			if (!TryGetResultDBJoinColumnRef(condition.GetLHS(), left) ||
			    !TryGetResultDBJoinColumnRef(condition.GetRHS(), right)) {
				throw InternalException("ResultDB Yannakakis edge contains an invalid join column");
			}
			AddResultDBWorkingColumn(needed_columns, edge.left_node, left.column->binding.table_index.index,
			                         left.column->binding.column_index.GetIndex(),
			                         GetResultDBJoinColumnCastType(left));
			AddResultDBWorkingColumn(needed_columns, edge.right_node, right.column->binding.table_index.index,
			                         right.column->binding.column_index.GetIndex(),
			                         GetResultDBJoinColumnCastType(right));
		}
	}

	for (idx_t table_metadata_idx = 0; table_metadata_idx < properties.resultdb.tables.size(); table_metadata_idx++) {
		auto &table = properties.resultdb.tables[table_metadata_idx];
		auto relation_entry = graph.node_map.find(table.table_index);
		if (relation_entry == graph.node_map.end()) {
			throw InternalException("ResultDB semijoin output table does not map to a Yannakakis relation");
		}
		for (auto &column : table.columns) {
			if (column.source_column_index == DConstants::INVALID_INDEX) {
				throw InternalException("ResultDB semijoin column metadata is missing source_column_index");
			}
			AddResultDBWorkingColumn(needed_columns, relation_entry->second, table.table_index,
			                         column.source_column_index);
		}
	}

	for (idx_t relation_pos = 0; relation_pos < graph.nodes.size(); relation_pos++) {
		auto &node = graph.nodes[relation_pos];
		auto &relation = program->relations[relation_pos];
		if (node.tables.size() == 1) {
			relation.table_index = node.tables[0].index;
		}
		for (auto table : node.tables) {
			relation.table_indices.push_back(table.index);
		}
		auto &source_column_indices = needed_columns[relation_pos];
		SortResultDBSourceColumns(source_column_indices);
		relation.source_columns = std::move(source_column_indices);
		if (relation.table_index != DConstants::INVALID_INDEX) {
			for (auto &source_column : relation.source_columns) {
				relation.source_column_indices.push_back(source_column.source_column_index);
			}
		}

		relation.base_plan = BuildResultDBNodeMaterializationPlan(binder, analysis, node, relation);
	}

	program->edges.reserve(graph.edges.size());
	for (auto &edge : graph.edges) {
		ResultDBYannakakisEdge program_edge;
		program_edge.left_relation = edge.left_node;
		program_edge.right_relation = edge.right_node;
		for (auto &condition : edge.conditions) {
			ResultDBJoinColumnRef left;
			ResultDBJoinColumnRef right;
			if (!TryGetResultDBJoinColumnRef(condition.GetLHS(), left) ||
			    !TryGetResultDBJoinColumnRef(condition.GetRHS(), right)) {
				throw InternalException("ResultDB Yannakakis edge contains an invalid join column");
			}
			ResultDBYannakakisJoinColumn column;
			column.left_column_index =
			    GetResultDBWorkingColumnIndex(program->relations[program_edge.left_relation].source_columns,
			                                  left.column->binding.table_index.index,
			                                  left.column->binding.column_index.GetIndex(),
			                                  GetResultDBJoinColumnCastType(left));
			column.right_column_index =
			    GetResultDBWorkingColumnIndex(program->relations[program_edge.right_relation].source_columns,
			                                  right.column->binding.table_index.index,
			                                  right.column->binding.column_index.GetIndex(),
			                                  GetResultDBJoinColumnCastType(right));
			program_edge.columns.push_back(column);
		}
		program->edges.push_back(std::move(program_edge));
	}

	program->outputs.reserve(properties.resultdb.tables.size());
	for (idx_t table_metadata_idx = 0; table_metadata_idx < properties.resultdb.tables.size(); table_metadata_idx++) {
		auto &table = properties.resultdb.tables[table_metadata_idx];
		auto relation_entry = graph.node_map.find(table.table_index);
		D_ASSERT(relation_entry != graph.node_map.end());
		ResultDBYannakakisOutputTable output_table;
		output_table.relation = relation_entry->second;
		output_table.table_metadata_index = table_metadata_idx;
		for (auto &column : table.columns) {
			ResultDBYannakakisOutputColumn output_column;
			output_column.working_column_index =
			    GetResultDBWorkingColumnIndex(program->relations[output_table.relation].source_columns,
			                                  table.table_index,
			                                  column.source_column_index);
			output_column.name = column.name;
			output_column.type = column.type;
			output_table.columns.push_back(std::move(output_column));
		}
		program->outputs.push_back(std::move(output_table));
	}
	return program;
}

static LogicalType GetResultDBEmptySourceColumnType(const ResultDBTableMetadata &table, idx_t source_idx) {
	for (auto &column : table.columns) {
		if (column.source_column_index == source_idx) {
			return column.type;
		}
	}
	throw InternalException("ResultDB empty relation is missing output column type");
}

static unique_ptr<LogicalOperator> BuildResultDBEmptyRelationPlan(Binder &binder,
                                                                  const ResultDBYannakakisRelation &relation) {
	auto table_index = binder.GenerateTableIndex();
	vector<ColumnBinding> bindings;
	bindings.reserve(relation.types.size());
	for (idx_t column_idx = 0; column_idx < relation.types.size(); column_idx++) {
		bindings.emplace_back(table_index, ProjectionIndex(column_idx));
	}
	return make_uniq<LogicalEmptyResult>(relation.types, std::move(bindings));
}

static unique_ptr<ResultDBYannakakisProgram> BuildResultDBEmptyYannakakisProgram(Binder &binder) {
	auto &properties = binder.GetStatementProperties();
	auto program = make_uniq<ResultDBYannakakisProgram>();
	auto relation_count = properties.resultdb.tables.size();
	program->root_relation = relation_count == 0 ? DConstants::INVALID_INDEX : 0;
	program->parent.assign(relation_count, DConstants::INVALID_INDEX);
	program->parent_edge.assign(relation_count, DConstants::INVALID_INDEX);
	program->relations.resize(relation_count);

	vector<vector<ResultDBYannakakisSourceColumn>> needed_columns(relation_count);
	for (idx_t table_metadata_idx = 0; table_metadata_idx < relation_count; table_metadata_idx++) {
		auto &table = properties.resultdb.tables[table_metadata_idx];
		for (auto &column : table.columns) {
			if (column.source_column_index == DConstants::INVALID_INDEX) {
				throw InternalException("ResultDB semijoin column metadata is missing source_column_index");
			}
			AddResultDBWorkingColumn(needed_columns, table_metadata_idx, table.table_index,
			                         column.source_column_index);
		}
	}

	for (idx_t table_metadata_idx = 0; table_metadata_idx < relation_count; table_metadata_idx++) {
		auto &table = properties.resultdb.tables[table_metadata_idx];
		auto &relation = program->relations[table_metadata_idx];
		relation.table_index = table.table_index;
		relation.table_indices.push_back(table.table_index);
		SortResultDBSourceColumns(needed_columns[table_metadata_idx]);
		relation.source_columns = std::move(needed_columns[table_metadata_idx]);
		for (auto &source_column : relation.source_columns) {
			relation.source_column_indices.push_back(source_column.source_column_index);
			relation.names.push_back(StringUtil::Format("resultdb_empty_%llu_%llu", source_column.table_index,
			                                            source_column.source_column_index));
			relation.types.push_back(GetResultDBEmptySourceColumnType(table, source_column.source_column_index));
		}
		relation.base_plan = BuildResultDBEmptyRelationPlan(binder, relation);
		program->order.push_back(table_metadata_idx);
	}

	program->outputs.reserve(relation_count);
	for (idx_t table_metadata_idx = 0; table_metadata_idx < relation_count; table_metadata_idx++) {
		auto &table = properties.resultdb.tables[table_metadata_idx];
		ResultDBYannakakisOutputTable output_table;
		output_table.relation = table_metadata_idx;
		output_table.table_metadata_index = table_metadata_idx;
		for (auto &column : table.columns) {
			ResultDBYannakakisOutputColumn output_column;
			output_column.working_column_index =
			    GetResultDBWorkingColumnIndex(program->relations[output_table.relation].source_columns,
			                                  table.table_index, column.source_column_index);
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

static void ResetResultDBPlanningDiagnostics(ResultDBProperties &properties) {
	properties.selected_root_table_index = DConstants::INVALID_INDEX;
	properties.selected_folds.clear();
	properties.estimated_semijoin_cost = 0;
	properties.estimated_decompose_cost = 0;
	properties.enumeration_time_ms = 0;
	properties.fold_candidate_count = 0;
	properties.block_sizes.clear();
	properties.tvc_enabled = false;
	properties.planning_reason.clear();
}

static void StoreResultDBSelectedGraph(ResultDBProperties &properties, const ResultDBSemijoinGraph &graph,
	                                   idx_t root_relation) {
	if (root_relation < graph.nodes.size()) {
		properties.selected_root_table_index = graph.nodes[root_relation].stable_id;
	}
	properties.selected_folds.clear();
	for (auto &node : graph.nodes) {
		if (node.tables.size() < 2) {
			continue;
		}
		vector<idx_t> fold;
		for (auto table : node.tables) {
			fold.push_back(table.index);
		}
		properties.selected_folds.push_back(std::move(fold));
	}
}

static vector<idx_t> AllResultDBEnumerationNodes(const ResultDBEnumerationInput &input) {
	vector<idx_t> result;
	for (idx_t node_idx = 0; node_idx < input.nodes.size(); node_idx++) {
		result.push_back(node_idx);
	}
	return result;
}

unique_ptr<LogicalOperator>
ResultDBYannakakisOptimizer::Optimize(unique_ptr<LogicalOperator> plan,
                                      unique_ptr<ResultDBYannakakisProgram> &resultdb_yannakakis_program) {
	auto &properties = binder.GetStatementProperties();
	resultdb_yannakakis_program.reset();
	if (!properties.resultdb.enabled) {
		return plan;
	}
	properties.resultdb.execution_strategy = ResultDBExecutionStrategy::DECOMPOSE;
	properties.resultdb.join_edges.clear();
	ResetResultDBPlanningDiagnostics(properties.resultdb);
	if (properties.resultdb.requested_strategy == ResultDBStrategy::DECOMPOSE) {
		properties.resultdb.planning_reason = "explicit decompose strategy";
		return plan;
	}

	if (plan->type == LogicalOperatorType::LOGICAL_EMPTY_RESULT) {
		if (properties.resultdb.requested_strategy == ResultDBStrategy::AUTO) {
			properties.resultdb.planning_reason = "TDResultDB selected decompose for a statically empty query";
			return plan;
		}
		resultdb_yannakakis_program = BuildResultDBEmptyYannakakisProgram(binder);
		properties.resultdb.execution_strategy = ResultDBExecutionStrategy::SEMIJOIN;
		properties.resultdb.planning_reason = "explicit direct strategy for a statically empty query";
		return make_uniq_base<LogicalOperator, LogicalDummyScan>(binder.GenerateTableIndex());
	}

	ResultDBSemijoinAnalysis analysis;
	if (!AnalyzeResultDBSemijoin(*plan, analysis, context)) {
		if (properties.resultdb.requested_strategy == ResultDBStrategy::AUTO) {
			return plan;
		}
		throw BinderException("RESULTDB semijoin strategy does not support this query: %s",
		                      analysis.unsupported_reason);
	}
	CanonicalizeResultDBJoinConditions(analysis);
	StoreResultDBJoinMetadata(properties.resultdb, analysis);

	vector<idx_t> parent;
	vector<idx_t> parent_edge;
	vector<idx_t> order;
	idx_t root_relation = DConstants::INVALID_INDEX;
	ResultDBSemijoinGraph graph;
	auto planning_start = std::chrono::steady_clock::now();
	auto requested_strategy = properties.resultdb.requested_strategy;
	bool use_heuristic_folds = requested_strategy == ResultDBStrategy::SEMIJOIN ||
	                           requested_strategy == ResultDBStrategy::TDROOT;
	bool graph_valid = use_heuristic_folds ? BuildResultDBSemijoinGraph(analysis, graph)
	                                     : BuildResultDBInitialGraph(analysis, graph);
	if (!graph_valid) {
		if (properties.resultdb.requested_strategy == ResultDBStrategy::AUTO) {
			properties.resultdb.planning_reason = graph.unsupported_reason;
			return plan;
		}
		throw BinderException("RESULTDB semijoin strategy does not support this query: %s",
		                      graph.unsupported_reason);
	}

	vector<vector<idx_t>> bottom_up_children;
	vector<vector<idx_t>> top_down_children;
	if (requested_strategy == ResultDBStrategy::SEMIJOIN) {
		if (!ChooseResultDBRootRelation(graph, properties.resultdb, parent, parent_edge, order, root_relation)) {
			throw BinderException("RESULTDB semijoin strategy does not support this query: %s",
			                      graph.unsupported_reason);
		}
		properties.resultdb.planning_reason = "existing degree/cardinality heuristic";
	} else if (requested_strategy == ResultDBStrategy::TDROOT) {
		DuckDBResultDBEnumerationCostProvider cost_provider(binder, context, analysis, graph, properties.resultdb);
		auto input = BuildResultDBEnumerationInput(graph, properties.resultdb);
		auto root_plan = ResultDBPlanEnumerator::EnumerateRoot(input, cost_provider);
		if (!root_plan.valid) {
			throw BinderException("RESULTDB TDRoot strategy does not support this query: %s", root_plan.error);
		}
		root_relation = root_plan.root;
		parent = std::move(root_plan.parent);
		parent_edge = std::move(root_plan.parent_edge);
		order = std::move(root_plan.order);
		bottom_up_children = std::move(root_plan.bottom_up_children);
		top_down_children = std::move(root_plan.top_down_children);
		properties.resultdb.estimated_semijoin_cost = root_plan.cost;
		properties.resultdb.planning_reason = "TDRoot over the heuristic fold tree";
	} else {
		DuckDBResultDBEnumerationCostProvider cost_provider(binder, context, analysis, graph, properties.resultdb);
		auto input = BuildResultDBEnumerationInput(graph, properties.resultdb);
		auto all_nodes = AllResultDBEnumerationNodes(input);
		auto join_cost = ResultDBPlanEnumerator::EstimateJoinCost(input, cost_provider, all_nodes);
		properties.resultdb.estimated_decompose_cost =
		    join_cost + cost_provider.Cardinality(all_nodes) * cost_provider.PayloadWidth(all_nodes);
		bool use_tvc = requested_strategy != ResultDBStrategy::TDFOLD_NO_TVC;
		properties.resultdb.tvc_enabled = use_tvc;
		ResultDBFoldEnumerationResult fold_plan;
		try {
			fold_plan = ResultDBPlanEnumerator::EnumerateFolds(input, cost_provider, use_tvc);
		} catch (Exception &ex) {
			if (requested_strategy == ResultDBStrategy::AUTO) {
				properties.resultdb.planning_reason = StringUtil::Format("TDResultDB selected decompose: %s", ex.what());
				return plan;
			}
			throw;
		}
		properties.resultdb.fold_candidate_count = fold_plan.candidate_count;
		properties.resultdb.block_sizes = fold_plan.block_sizes;
		properties.resultdb.estimated_semijoin_cost = fold_plan.cost;
		if (!fold_plan.valid) {
			if (requested_strategy == ResultDBStrategy::AUTO) {
				properties.resultdb.planning_reason = "TDResultDB selected decompose because TDFold found no plan";
				return plan;
			}
			throw BinderException("RESULTDB TDFold strategy does not support this query: %s", fold_plan.error);
		}
		if (requested_strategy == ResultDBStrategy::AUTO &&
		    properties.resultdb.estimated_decompose_cost <= properties.resultdb.estimated_semijoin_cost) {
			properties.resultdb.planning_reason = "TDResultDB cost comparison selected decompose";
			auto planning_end = std::chrono::steady_clock::now();
			properties.resultdb.enumeration_time_ms =
			    std::chrono::duration<double, std::milli>(planning_end - planning_start).count();
			return plan;
		}
		if (!ApplyResultDBFoldPartition(analysis, graph, fold_plan.folds, cost_provider)) {
			if (requested_strategy == ResultDBStrategy::AUTO) {
				properties.resultdb.planning_reason = graph.unsupported_reason;
				return plan;
			}
			throw BinderException("RESULTDB TDFold produced an invalid plan: %s", graph.unsupported_reason);
		}
		DuckDBResultDBEnumerationCostProvider contracted_cost_provider(binder, context, analysis, graph,
		                                                             properties.resultdb);
		auto contracted_input = BuildResultDBEnumerationInput(graph, properties.resultdb);
		auto root_plan = ResultDBPlanEnumerator::EnumerateRoot(contracted_input, contracted_cost_provider);
		if (!root_plan.valid) {
			throw BinderException("RESULTDB TDFold produced an invalid contracted tree: %s", root_plan.error);
		}
		root_relation = root_plan.root;
		parent = std::move(root_plan.parent);
		parent_edge = std::move(root_plan.parent_edge);
		order = std::move(root_plan.order);
		bottom_up_children = std::move(root_plan.bottom_up_children);
		top_down_children = std::move(root_plan.top_down_children);
		properties.resultdb.planning_reason = requested_strategy == ResultDBStrategy::AUTO
		                                          ? "TDResultDB cost comparison selected TDFold+TDRoot"
		                                          : (use_tvc ? "TDFold with TVCs followed by TDRoot"
		                                                     : "TDFold without TVCs followed by TDRoot");
	}

	resultdb_yannakakis_program = BuildResultDBYannakakisProgram(
	    binder, analysis, graph, root_relation, std::move(parent), std::move(parent_edge), std::move(order),
	    std::move(bottom_up_children), std::move(top_down_children));
	properties.resultdb.execution_strategy = ResultDBExecutionStrategy::SEMIJOIN;
	StoreResultDBSelectedGraph(properties.resultdb, graph, root_relation);
	auto planning_end = std::chrono::steady_clock::now();
	properties.resultdb.enumeration_time_ms =
	    std::chrono::duration<double, std::milli>(planning_end - planning_start).count();
	return make_uniq_base<LogicalOperator, LogicalDummyScan>(binder.GenerateTableIndex());
}

} // namespace duckdb
