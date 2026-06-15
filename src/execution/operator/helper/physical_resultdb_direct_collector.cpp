#include "duckdb/execution/operator/helper/physical_resultdb_direct_collector.hpp"

#include "duckdb/common/enum_util.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/materialized_query_result.hpp"
#include "duckdb/main/prepared_statement_data.hpp"
#include "duckdb/parallel/meta_pipeline.hpp"
#include "duckdb/parallel/pipeline.hpp"
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
	for (auto &phase : program.phases) {
		phase_plans.push_back(phase.plan->Root());
	}
}

class ResultDBDirectCollectorGlobalState : public GlobalSinkState {
public:
	mutex glock;
	vector<unique_ptr<ColumnDataCollection>> output_collections;
	shared_ptr<ClientContext> context;
};

class ResultDBDirectCollectorLocalState : public LocalSinkState {
public:
	idx_t phase_idx = DConstants::INVALID_INDEX;
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

unique_ptr<GlobalSinkState> PhysicalResultDBDirectCollector::GetGlobalSinkState(ClientContext &context) const {
	auto state = make_uniq<ResultDBDirectCollectorGlobalState>();
	state->context = context.shared_from_this();
	for (auto &collection : program.collections) {
		collection->Reset();
	}
	for (auto &table : properties.resultdb.tables) {
		state->output_collections.push_back(
		    CreateResultDBDirectCollection(context, memory_type, GetResultDBTableTypes(table)));
	}
	return std::move(state);
}

unique_ptr<LocalSinkState> PhysicalResultDBDirectCollector::GetLocalSinkState(ExecutionContext &context) const {
	if (!context.pipeline) {
		throw InternalException("ResultDB direct collector local sink state requires a pipeline");
	}
	auto entry = pipeline_phase_map.find(context.pipeline.get());
	if (entry == pipeline_phase_map.end()) {
		throw InternalException("ResultDB direct collector could not map pipeline to Yannakakis phase");
	}

	auto state = make_uniq<ResultDBDirectCollectorLocalState>();
	state->phase_idx = entry->second;
	state->collection =
	    make_uniq<ColumnDataCollection>(context.client, phase_plans[state->phase_idx].get().GetTypes());
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

	lock_guard<mutex> l(gstate.glock);
	auto &phase = program.phases[lstate.phase_idx];
	switch (phase.type) {
	case ResultDBYannakakisPhaseType::BASE:
	case ResultDBYannakakisPhaseType::SEMIJOIN:
		program.collections[phase.target_collection]->Combine(*lstate.collection);
		break;
	case ResultDBYannakakisPhaseType::OUTPUT:
		gstate.output_collections[phase.output_table_index]->Combine(*lstate.collection);
		break;
	default:
		throw InternalException("Unknown ResultDB Yannakakis phase type");
	}
	return SinkCombineResultType::FINISHED;
}

unique_ptr<QueryResult> PhysicalResultDBDirectCollector::GetResult(GlobalSinkState &state) const {
	auto &gstate = state.Cast<ResultDBDirectCollectorGlobalState>();
	if (properties.resultdb.tables.empty()) {
		throw InternalException("RESULTDB query has no output tables");
	}

	unique_ptr<QueryResult> first_result;
	QueryResult *current_result = nullptr;
	for (idx_t table_idx = 0; table_idx < properties.resultdb.tables.size(); table_idx++) {
		auto &table = properties.resultdb.tables[table_idx];
		auto table_result = make_uniq<MaterializedQueryResult>(
		    statement_type, GetResultDBSingleTableProperties(properties, table_idx), GetResultDBTableNames(table),
		    std::move(gstate.output_collections[table_idx]), gstate.context->GetClientProperties());
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
	for (auto &phase_plan : phase_plans) {
		result.push_back(phase_plan.get());
	}
	return result;
}

void PhysicalResultDBDirectCollector::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	sink_state.reset();
	pipeline_phase_map.clear();

	auto &state = meta_pipeline.GetState();
	state.SetPipelineSource(current, *this);

	vector<vector<shared_ptr<Pipeline>>> phase_all_pipelines;
	phase_all_pipelines.resize(phase_plans.size());

	for (idx_t phase_idx = 0; phase_idx < phase_plans.size(); phase_idx++) {
		auto &child_meta_pipeline = meta_pipeline.CreateChildMetaPipeline(current, *this);
		child_meta_pipeline.Build(phase_plans[phase_idx].get());

		vector<shared_ptr<Pipeline>> child_pipelines;
		child_meta_pipeline.GetPipelines(child_pipelines, false);
		for (auto &pipeline : child_pipelines) {
			pipeline_phase_map[pipeline.get()] = phase_idx;
		}
		child_meta_pipeline.GetPipelines(phase_all_pipelines[phase_idx], true);
	}

	for (idx_t phase_idx = 0; phase_idx < program.phases.size(); phase_idx++) {
		for (auto dependency_idx : program.phases[phase_idx].dependencies) {
			for (auto &pipeline : phase_all_pipelines[phase_idx]) {
				for (auto &dependency : phase_all_pipelines[dependency_idx]) {
					pipeline->AddDependency(dependency);
				}
			}
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
