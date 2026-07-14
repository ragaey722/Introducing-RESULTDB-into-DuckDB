#include "duckdb/execution/operator/helper/physical_resultdb_direct_collector.hpp"

#include "duckdb/common/enum_util.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/aggregate_hashtable.hpp"
#include "duckdb/execution/operator/join/physical_hash_join.hpp"
#include "duckdb/execution/join_hashtable.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/materialized_query_result.hpp"
#include "duckdb/main/prepared_statement_data.hpp"
#include "duckdb/parallel/meta_pipeline.hpp"
#include "duckdb/parallel/pipeline.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/storage/buffer_manager.hpp"

namespace duckdb {

static PreparedResultDBYannakakisProgram &GetPreparedResultDBYannakakisProgram(PreparedStatementData &data) {
	if (!data.resultdb_yannakakis_program) {
		throw InternalException("ResultDB direct collector expected a prepared Yannakakis program");
	}
	return *data.resultdb_yannakakis_program;
}

PhysicalResultDBDirectCollector::PhysicalResultDBDirectCollector(PhysicalPlan &physical_plan,
                                                                 PreparedStatementData &data)
    : PhysicalResultCollector(physical_plan, data), program(GetPreparedResultDBYannakakisProgram(data)) {
	for (auto &base_plan : program.base_plans) {
		base_plans.push_back(base_plan->Root());
	}
}

class ResultDBDirectCollectorGlobalState : public GlobalSinkState {
public:
	mutex combine_lock;
	vector<unique_ptr<ColumnDataCollection>> relation_collections;
	shared_ptr<ClientContext> context;
};

class ResultDBDirectCollectorLocalState : public LocalSinkState {
public:
	idx_t relation_idx = DConstants::INVALID_INDEX;
	unique_ptr<ColumnDataCollection> collection;
	ColumnDataAppendState append_state;
};

static vector<LogicalType> GetResultDBTableTypes(const ResultDBTableMetadata &table) {
	vector<LogicalType> types;
	for (auto &column : table.columns) {
		types.push_back(column.type);
	}
	return types;
}

static vector<string> GetResultDBTableNames(const ResultDBTableMetadata &table) {
	vector<string> names;
	for (auto &column : table.columns) {
		names.push_back(column.name);
	}
	return names;
}

static unique_ptr<ColumnDataCollection> CreateResultDBDirectCollection(ClientContext &context,
                                                                       QueryResultMemoryType memory_type,
                                                                       const vector<LogicalType> &types) {
	switch (memory_type) {
	case QueryResultMemoryType::IN_MEMORY:
		return make_uniq<ColumnDataCollection>(Allocator::DefaultAllocator(), types);
	case QueryResultMemoryType::BUFFER_MANAGED:
		return make_uniq<ColumnDataCollection>(BufferManager::GetBufferManager(*context.db), types,
		                                       ColumnDataCollectionLifetime::THROW_ERROR_AFTER_DATABASE_CLOSES);
	default:
		throw NotImplementedException("CreateResultDBDirectCollection for %s", EnumUtil::ToString(memory_type));
	}
}

static StatementProperties GetResultDBSingleTableProperties(StatementProperties properties, idx_t table_idx) {
	auto table_metadata = properties.resultdb.tables[table_idx];
	for (idx_t column_idx = 0; column_idx < table_metadata.columns.size(); column_idx++) {
		table_metadata.columns[column_idx].flat_column_index = column_idx;
	}
	properties.resultdb.tables.clear();
	properties.resultdb.tables.push_back(std::move(table_metadata));
	return properties;
}

static idx_t GetResultDBDirectEdgeColumn(const ResultDBYannakakisEdge &edge, idx_t relation_idx,
                                         const ResultDBYannakakisJoinColumn &column) {
	if (edge.left_relation == relation_idx) {
		return column.left_column_index;
	}
	if (edge.right_relation == relation_idx) {
		return column.right_column_index;
	}
	throw InternalException("ResultDB Yannakakis edge does not contain relation");
}

static vector<idx_t> GetResultDBDirectEdgeColumns(const ResultDBYannakakisEdge &edge, idx_t relation_idx) {
	vector<idx_t> result;
	result.reserve(edge.columns.size());
	for (auto &column : edge.columns) {
		result.push_back(GetResultDBDirectEdgeColumn(edge, relation_idx, column));
	}
	return result;
}

static vector<LogicalType> GetResultDBDirectColumnTypes(const vector<LogicalType> &input_types,
                                                        const vector<idx_t> &columns) {
	vector<LogicalType> result;
	result.reserve(columns.size());
	for (auto column_idx : columns) {
		result.push_back(input_types[column_idx]);
	}
	return result;
}

static vector<column_t> ToResultDBDirectColumnIds(const vector<idx_t> &columns) {
	vector<column_t> result;
	result.reserve(columns.size());
	for (auto column_idx : columns) {
		result.push_back(column_t(column_idx));
	}
	return result;
}

static vector<JoinCondition> BuildResultDBDirectJoinConditions(const vector<LogicalType> &key_types) {
	vector<JoinCondition> conditions;
	conditions.reserve(key_types.size());
	for (idx_t key_idx = 0; key_idx < key_types.size(); key_idx++) {
		auto lhs = make_uniq<BoundReferenceExpression>(key_types[key_idx], key_idx);
		auto rhs = make_uniq<BoundReferenceExpression>(key_types[key_idx], key_idx);
		conditions.emplace_back(std::move(lhs), std::move(rhs), ExpressionType::COMPARE_EQUAL);
	}
	return conditions;
}

static vector<idx_t> BuildResultDBDirectIdentityProjection(idx_t column_count) {
	vector<idx_t> result;
	result.reserve(column_count);
	for (idx_t column_idx = 0; column_idx < column_count; column_idx++) {
		result.push_back(column_idx);
	}
	return result;
}

struct ResultDBDirectHashTableState {
	//! JoinHashTable keeps references to these vectors; keep them alive with the table.
	vector<JoinCondition> conditions;
	vector<idx_t> output_columns;
	vector<idx_t> output_in_probe;
	unique_ptr<JoinHashTable> table;
};

static unique_ptr<ResultDBDirectHashTableState>
BuildResultDBDirectHashTable(ClientContext &context, const PhysicalOperator &op, const ColumnDataCollection &source,
                             const vector<idx_t> &source_columns, idx_t target_column_count) {
	auto key_types = GetResultDBDirectColumnTypes(source.Types(), source_columns);
	vector<LogicalType> payload_types;

	auto state = make_uniq<ResultDBDirectHashTableState>();
	state->conditions = BuildResultDBDirectJoinConditions(key_types);
	state->output_in_probe = BuildResultDBDirectIdentityProjection(target_column_count);
	state->table = make_uniq<JoinHashTable>(context, op, state->conditions, payload_types, JoinType::SEMI, 0U,
	                                        state->output_columns, nullptr, nullptr, state->output_in_probe);

	auto source_column_ids = ToResultDBDirectColumnIds(source_columns);
	PartitionedTupleDataAppendState append_state;
	ColumnDataScanState source_scan;
	DataChunk source_chunk;
	DataChunk source_keys;
	DataChunk empty_payload;
	source.InitializeScan(source_scan, source_column_ids);
	source.InitializeScanChunk(source_scan, source_chunk);
	source_keys.Initialize(Allocator::Get(context), key_types);
	state->table->GetSinkCollection().InitializeAppendState(append_state);
	while (source.Scan(source_scan, source_chunk)) {
		source_keys.Reset();
		for (idx_t key_idx = 0; key_idx < source_columns.size(); key_idx++) {
			source_keys.data[key_idx].Reference(source_chunk.data[key_idx]);
		}
		source_keys.SetCardinality(source_chunk);
		empty_payload.SetCardinality(source_chunk);
		state->table->Build(append_state, source_keys, empty_payload);
	}

	state->table->Unpartition();
	state->table->AllocatePointerTable();
	state->table->InitializePointerTable(0U, state->table->capacity);
	if (state->table->Count() > 0) {
		state->table->Finalize(0U, state->table->GetDataCollection().ChunkCount(), false);
	}
	state->table->finalized = true;
	return state;
}

static unique_ptr<ColumnDataCollection>
ApplyResultDBDirectSemijoin(ClientContext &context, QueryResultMemoryType memory_type, const PhysicalOperator &op,
                            const ColumnDataCollection &target, const ColumnDataCollection &source,
                            const vector<idx_t> &target_columns, const vector<idx_t> &source_columns) {
	auto key_types = GetResultDBDirectColumnTypes(target.Types(), target_columns);
	auto hash_table_state = BuildResultDBDirectHashTable(context, op, source, source_columns, target.Types().size());
	auto &hash_table = *hash_table_state->table;
	auto output = CreateResultDBDirectCollection(context, memory_type, target.Types());
	ColumnDataAppendState append_state;
	output->InitializeAppend(append_state);

	if (hash_table.Count() == 0) {
		return output;
	}

	ColumnDataScanState target_scan;
	DataChunk target_chunk;
	DataChunk target_keys;
	DataChunk probe_data;
	DataChunk result_chunk;
	target.InitializeScan(target_scan);
	target.InitializeScanChunk(target_scan, target_chunk);
	target_keys.Initialize(Allocator::Get(context), key_types);
	probe_data.Initialize(Allocator::Get(context), target.Types());
	result_chunk.Initialize(Allocator::Get(context), target.Types());

	TupleDataChunkState key_state;
	TupleDataCollection::InitializeChunkState(key_state, key_types);
	JoinHashTable::ScanStructure scan_structure(hash_table, key_state);
	JoinHashTable::ProbeState probe_state;

	while (target.Scan(target_scan, target_chunk)) {
		target_keys.Reset();
		for (idx_t key_idx = 0; key_idx < target_columns.size(); key_idx++) {
			target_keys.data[key_idx].Reference(target_chunk.data[target_columns[key_idx]]);
		}
		target_keys.SetCardinality(target_chunk);
		probe_data.Reference(target_chunk);

		hash_table.Probe(scan_structure, target_keys, key_state, probe_state);
		do {
			result_chunk.Reset();
			scan_structure.Next(target_keys, probe_data, result_chunk);
			if (result_chunk.size() > 0) {
				output->Append(append_state, result_chunk);
			}
		} while (!scan_structure.PointersExhausted() && result_chunk.size() > 0);
	}
	return output;
}

static void ExecuteResultDBDirectReduction(ClientContext &context, QueryResultMemoryType memory_type,
                                           const PhysicalOperator &op,
                                           const PreparedResultDBYannakakisProgram &program,
                                           vector<unique_ptr<ColumnDataCollection>> &relations) {
	if (program.order.empty()) {
		throw InternalException("ResultDB Yannakakis program is missing relation order");
	}

	auto reduce = [&](idx_t target, idx_t source, idx_t edge_idx) {
		auto &edge = program.edges[edge_idx];
		auto target_columns = GetResultDBDirectEdgeColumns(edge, target);
		auto source_columns = GetResultDBDirectEdgeColumns(edge, source);
		relations[target] = ApplyResultDBDirectSemijoin(context, memory_type, op, *relations[target],
		                                                *relations[source], target_columns, source_columns);
	};

	for (idx_t order_offset = program.order.size(); order_offset > 0; order_offset--) {
		auto child = program.order[order_offset - 1];
		if (child != program.root_relation) {
			reduce(program.parent[child], child, program.parent_edge[child]);
		}
	}

	for (idx_t order_idx = 1; order_idx < program.order.size(); order_idx++) {
		auto child = program.order[order_idx];
		reduce(child, program.parent[child], program.parent_edge[child]);
	}
}

static unique_ptr<ColumnDataCollection>
BuildResultDBDirectOutputCollection(ClientContext &context, QueryResultMemoryType memory_type,
                                    const ResultDBYannakakisOutputTable &output,
                                    const ColumnDataCollection &relation) {
	vector<LogicalType> types;
	types.reserve(output.columns.size());
	vector<column_t> projection_column_ids;
	projection_column_ids.reserve(output.columns.size());
	for (auto &column : output.columns) {
		types.push_back(column.type);
		projection_column_ids.push_back(column_t(column.working_column_index));
	}

	auto &allocator = Allocator::Get(context);
	GroupedAggregateHashTable distinct_table(context, allocator, types);
	DataChunk projected_chunk;
	DataChunk empty_payload;
	unsafe_vector<idx_t> filter;

	ColumnDataScanState relation_scan_state;
	relation.InitializeScan(relation_scan_state, projection_column_ids);
	relation.InitializeScanChunk(relation_scan_state, projected_chunk);
	while (relation.Scan(relation_scan_state, projected_chunk)) {
		distinct_table.AddChunk(projected_chunk, empty_payload, filter);
	}

	auto collection = CreateResultDBDirectCollection(context, memory_type, types);
	ColumnDataAppendState append_state;
	collection->InitializeAppend(append_state);

	AggregateHTScanState distinct_scan_state;
	DataChunk distinct_rows;
	DataChunk payload_rows;
	distinct_rows.Initialize(allocator, types);
	distinct_table.InitializeScan(distinct_scan_state);
	while (true) {
		distinct_rows.Reset();
		payload_rows.Reset();
		if (!distinct_table.Scan(distinct_scan_state, distinct_rows, payload_rows)) {
			break;
		}
		if (distinct_rows.size() > 0) {
			collection->Append(append_state, distinct_rows);
		}
	}
	return collection;
}

unique_ptr<GlobalSinkState> PhysicalResultDBDirectCollector::GetGlobalSinkState(ClientContext &context) const {
	if (program.relations.empty() || program.base_plans.size() != program.relations.size()) {
		throw InternalException("ResultDB direct collector expected prepared base relation plans");
	}

	auto state = make_uniq<ResultDBDirectCollectorGlobalState>();
	state->context = context.shared_from_this();
	for (auto &relation : program.relations) {
		state->relation_collections.push_back(CreateResultDBDirectCollection(context, memory_type, relation.types));
	}
	return std::move(state);
}

unique_ptr<LocalSinkState> PhysicalResultDBDirectCollector::GetLocalSinkState(ExecutionContext &context) const {
	if (!context.pipeline) {
		throw InternalException("ResultDB direct collector local sink state requires a pipeline");
	}
	auto entry = pipeline_relation_map.find(context.pipeline.get());
	if (entry == pipeline_relation_map.end()) {
		throw InternalException("ResultDB direct collector could not map pipeline to Yannakakis relation");
	}

	auto state = make_uniq<ResultDBDirectCollectorLocalState>();
	state->relation_idx = entry->second;
	state->collection =
	    make_uniq<ColumnDataCollection>(context.client, base_plans[state->relation_idx].get().GetTypes());
	state->collection->InitializeAppend(state->append_state);
	return std::move(state);
}

SinkResultType PhysicalResultDBDirectCollector::Sink(ExecutionContext &context, DataChunk &chunk,
                                                     OperatorSinkInput &input) const {
	auto &lstate = input.local_state.Cast<ResultDBDirectCollectorLocalState>();
	lstate.collection->Append(lstate.append_state, chunk);
	return SinkResultType::NEED_MORE_INPUT;
}

SinkCombineResultType PhysicalResultDBDirectCollector::Combine(ExecutionContext &context,
                                                               OperatorSinkCombineInput &input) const {
	auto &gstate = input.global_state.Cast<ResultDBDirectCollectorGlobalState>();
	auto &lstate = input.local_state.Cast<ResultDBDirectCollectorLocalState>();
	if (lstate.collection->Count() == 0) {
		return SinkCombineResultType::FINISHED;
	}

	lock_guard<mutex> guard(gstate.combine_lock);
	gstate.relation_collections[lstate.relation_idx]->Combine(*lstate.collection);
	return SinkCombineResultType::FINISHED;
}

unique_ptr<QueryResult> PhysicalResultDBDirectCollector::GetResult(GlobalSinkState &state) const {
	auto &gstate = state.Cast<ResultDBDirectCollectorGlobalState>();
	if (properties.resultdb.tables.empty()) {
		throw InternalException("RESULTDB query has no output tables");
	}

	ExecuteResultDBDirectReduction(*gstate.context, memory_type, *this, program, gstate.relation_collections);

	vector<unique_ptr<ColumnDataCollection>> output_collections;
	output_collections.resize(properties.resultdb.tables.size());
	for (auto &output : program.outputs) {
		output_collections[output.table_metadata_index] =
		    BuildResultDBDirectOutputCollection(*gstate.context, memory_type, output,
		                                        *gstate.relation_collections[output.relation]);
	}

	unique_ptr<QueryResult> first_result;
	QueryResult *current_result = nullptr;
	for (idx_t table_idx = 0; table_idx < properties.resultdb.tables.size(); table_idx++) {
		auto &table = properties.resultdb.tables[table_idx];
		if (!output_collections[table_idx]) {
			throw InternalException("ResultDB direct output collection is missing");
		}
		auto table_result = make_uniq<MaterializedQueryResult>(
		    statement_type, GetResultDBSingleTableProperties(properties, table_idx), GetResultDBTableNames(table),
		    std::move(output_collections[table_idx]), gstate.context->GetClientProperties());
		auto table_result_ptr = table_result.get();
		if (!first_result) {
			first_result = std::move(table_result);
		} else {
			current_result->next = std::move(table_result);
		}
		current_result = table_result_ptr;
	}
	return first_result;
}

vector<const_reference<PhysicalOperator>> PhysicalResultDBDirectCollector::GetChildren() const {
	vector<const_reference<PhysicalOperator>> result;
	for (auto &base_plan : base_plans) {
		result.push_back(base_plan.get());
	}
	return result;
}

void PhysicalResultDBDirectCollector::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	sink_state.reset();
	pipeline_relation_map.clear();

	auto &state = meta_pipeline.GetState();
	state.SetPipelineSource(current, *this);

	for (idx_t relation_idx = 0; relation_idx < base_plans.size(); relation_idx++) {
		auto &child_meta_pipeline = meta_pipeline.CreateChildMetaPipeline(current, *this);
		child_meta_pipeline.Build(base_plans[relation_idx].get());

		vector<shared_ptr<Pipeline>> child_pipelines;
		child_meta_pipeline.GetPipelines(child_pipelines, false);
		for (auto &pipeline : child_pipelines) {
			pipeline_relation_map[pipeline.get()] = relation_idx;
		}
	}
}

bool PhysicalResultDBDirectCollector::ParallelSink() const {
	return true;
}

bool PhysicalResultDBDirectCollector::SinkOrderDependent() const {
	return false;
}

} // namespace duckdb
