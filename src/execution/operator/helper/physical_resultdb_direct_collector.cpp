#include "duckdb/execution/operator/helper/physical_resultdb_direct_collector.hpp"

#include "duckdb/common/enum_util.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/aggregate_hashtable.hpp"
#include "duckdb/execution/operator/helper/physical_resultdb_yannakakis_phase.hpp"
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

static vector<LogicalType> GetResultDBDirectOutputTypes(const ResultDBYannakakisOutputTable &output) {
	vector<LogicalType> types;
	types.reserve(output.columns.size());
	for (auto &column : output.columns) {
		types.push_back(column.type);
	}
	return types;
}

static unique_ptr<ColumnDataCollection>
MaterializeResultDBDirectOutput(ClientContext &context, QueryResultMemoryType memory_type,
                                const vector<LogicalType> &types, GroupedAggregateHashTable &distinct_table) {
	auto collection = CreateResultDBDirectCollection(context, memory_type, types);
	ColumnDataAppendState append_state;
	collection->InitializeAppend(append_state);

	AggregateHTScanState distinct_scan_state;
	DataChunk distinct_rows;
	DataChunk payload_rows;
	auto &allocator = Allocator::Get(context);
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

PhysicalResultDBDirectCollector::PhysicalResultDBDirectCollector(PhysicalPlan &physical_plan,
                                                                 PreparedStatementData &data)
    : PhysicalResultCollector(physical_plan, data), program(GetPreparedResultDBYannakakisProgram(data)) {
	if (program.relations.empty() || program.base_plans.size() != program.relations.size() ||
	    program.relation_phases.size() != program.relations.size() ||
	    program.required_for_output.size() != program.relations.size() ||
	    program.root_relation >= program.relations.size()) {
		throw InternalException("ResultDB direct collector expected a prepared phase program");
	}
	if (!program.edges.empty() &&
	    (program.parent.size() != program.relations.size() ||
	     program.order.size() != program.relations.size())) {
		throw InternalException("ResultDB direct collector expected a complete rooted phase program");
	}
	if (program.outputs.size() != properties.resultdb.tables.size()) {
		throw InternalException("ResultDB direct output metadata does not match the prepared program");
	}
	vector<uint8_t> seen_output_metadata(properties.resultdb.tables.size(), 0);
	for (auto &output : program.outputs) {
		if (output.table_metadata_index >= seen_output_metadata.size() ||
		    seen_output_metadata[output.table_metadata_index]) {
			throw InternalException("ResultDB direct output metadata indexes are not a permutation");
		}
		seen_output_metadata[output.table_metadata_index] = 1;
		auto &table = properties.resultdb.tables[output.table_metadata_index];
		if (output.columns.size() != table.columns.size()) {
			throw InternalException("ResultDB direct output column count does not match its public metadata");
		}
		for (idx_t column_idx = 0; column_idx < output.columns.size(); column_idx++) {
			if (output.columns[column_idx].name != table.columns[column_idx].name ||
			    output.columns[column_idx].type != table.columns[column_idx].type) {
				throw InternalException("ResultDB direct output column does not match its public metadata");
			}
		}
	}
	for (auto &base_plan : program.base_plans) {
		base_plans.push_back(base_plan->Root());
	}

	auto relation_count = program.relations.size();
	bottom_up_phases.resize(relation_count);
	top_down_phases.resize(relation_count);
	output_distinct_indexes.assign(program.outputs.size(), DConstants::INVALID_INDEX);
	root_bottom_up_phase = make_uniq<PhysicalResultDBYannakakisPhase>(
	    physical_plan, base_plans[program.root_relation], program.root_relation,
	    program.relations[program.root_relation].types, memory_type, false, "RESULTDB_ROOT_FULL");
	// A multi-relation program without edges is the existing statically-empty
	// special case. Its root stream is sufficient to initialize the collector;
	// GetResult constructs every typed empty output directly.
	if (program.edges.empty() && relation_count > 1) {
		return;
	}
	for (idx_t relation_idx = 0; relation_idx < relation_count; relation_idx++) {
		if (relation_idx == program.root_relation) {
			continue;
		}
		bottom_up_phases[relation_idx] = make_uniq<PhysicalResultDBYannakakisPhase>(
		    physical_plan, base_plans[relation_idx], relation_idx, program.relations[relation_idx].types, memory_type,
		    program.relation_phases[relation_idx].retain_bottom_up, "RESULTDB_BOTTOM_UP");
	}

	// Every non-root relation builds exactly one child-to-parent key set.
	for (idx_t relation_idx = 0; relation_idx < relation_count; relation_idx++) {
		if (relation_idx == program.root_relation) {
			continue;
		}
		auto step_idx = program.relation_phases[relation_idx].bottom_up_to_parent_step;
		if (step_idx >= program.bottom_up_steps.size()) {
			throw InternalException("ResultDB bottom-up phase is missing its parent reduction step");
		}
		auto &step = program.bottom_up_steps[step_idx];
		if (step.source_relation != relation_idx) {
			throw InternalException("ResultDB bottom-up phase has an invalid parent reduction step");
		}
		auto output_idx = bottom_up_phases[relation_idx]->AddOutputKey(
		    step.source_columns, program.relations[step.target_relation].types, step.target_columns);
		D_ASSERT(output_idx == 0);
	}

	// A parent probes every finalized child key set in the canonical prepared
	// bottom-up order. The root delegates its sink interface to root_bottom_up_phase.
	for (idx_t relation_idx = 0; relation_idx < relation_count; relation_idx++) {
		auto &consumer = relation_idx == program.root_relation ? *root_bottom_up_phase
		                                                       : *bottom_up_phases[relation_idx];
		for (auto step_idx : program.relation_phases[relation_idx].bottom_up_from_children_steps) {
			if (step_idx >= program.bottom_up_steps.size()) {
				throw InternalException("ResultDB bottom-up phase references an invalid child step");
			}
			auto child_relation = program.bottom_up_steps[step_idx].source_relation;
			if (child_relation == program.root_relation || !bottom_up_phases[child_relation]) {
				throw InternalException("ResultDB bottom-up phase references an invalid child producer");
			}
			consumer.AddInputKey(
			    ResultDBYannakakisPhaseHandle(*bottom_up_phases[child_relation],
			                                  *bottom_up_phases[child_relation]),
			    0);
		}
	}

	auto configure_outputs = [&](PhysicalResultDBYannakakisPhase &phase, idx_t relation_idx) {
		for (auto output_idx : program.relation_phases[relation_idx].output_indexes) {
			if (output_idx >= program.outputs.size() ||
			    program.outputs[output_idx].relation != relation_idx ||
			    output_distinct_indexes[output_idx] != DConstants::INVALID_INDEX) {
				throw InternalException("ResultDB full phase references an invalid output");
			}
			auto &output = program.outputs[output_idx];
			vector<idx_t> projection_columns;
			projection_columns.reserve(output.columns.size());
			for (auto &column : output.columns) {
				projection_columns.push_back(column.working_column_index);
			}
			output_distinct_indexes[output_idx] =
			    phase.AddOutputDistinct(std::move(projection_columns),
			                            GetResultDBDirectOutputTypes(output));
		}
	};

	vector<idx_t> downward_key_indexes(program.top_down_steps.size(), DConstants::INVALID_INDEX);
	auto configure_downward_keys = [&](PhysicalResultDBYannakakisPhase &phase, idx_t relation_idx) {
		for (auto step_idx : program.relation_phases[relation_idx].top_down_to_children_steps) {
			if (step_idx >= program.top_down_steps.size() ||
			    downward_key_indexes[step_idx] != DConstants::INVALID_INDEX) {
				throw InternalException("ResultDB full phase references an invalid downward reduction");
			}
			auto &step = program.top_down_steps[step_idx];
			if (step.source_relation != relation_idx) {
				throw InternalException("ResultDB downward key is attached to the wrong producer");
			}
			downward_key_indexes[step_idx] =
			    phase.AddOutputKey(step.source_columns, program.relations[step.target_relation].types,
			                       step.target_columns);
		}
	};

	configure_downward_keys(*root_bottom_up_phase, program.root_relation);
	configure_outputs(*root_bottom_up_phase, program.root_relation);

	// Create top-down phases in root-to-leaf order. Each phase owns a runtime
	// scan of its retained bottom-up survivors, probes the already-finalized
	// parent downward key, and simultaneously produces outputs and child keys.
	for (idx_t order_idx = 1; order_idx < program.order.size(); order_idx++) {
		auto relation_idx = program.order[order_idx];
		if (!program.required_for_output[relation_idx]) {
			continue;
		}
		auto step_idx = program.relation_phases[relation_idx].top_down_from_parent_step;
		if (step_idx >= program.top_down_steps.size()) {
			throw InternalException("ResultDB top-down phase is missing its parent reduction");
		}
		auto &step = program.top_down_steps[step_idx];
		if (step.target_relation != relation_idx || step.source_relation != program.parent[relation_idx] ||
		    downward_key_indexes[step_idx] == DConstants::INVALID_INDEX) {
			throw InternalException("ResultDB top-down phase has an invalid parent reduction");
		}
		auto parent_relation = step.source_relation;
		PhysicalResultDBYannakakisPhase *parent_phase;
		PhysicalOperator *parent_state_owner;
		if (parent_relation == program.root_relation) {
			parent_phase = root_bottom_up_phase.get();
			parent_state_owner = this;
		} else {
			if (!top_down_phases[parent_relation]) {
				throw InternalException("ResultDB top-down parent phase was not prepared first");
			}
			parent_phase = top_down_phases[parent_relation].get();
			parent_state_owner = parent_phase;
		}
		if (!bottom_up_phases[relation_idx] ||
		    !program.relation_phases[relation_idx].retain_bottom_up) {
			throw InternalException("ResultDB top-down phase has no retained bottom-up producer");
		}

		ResultDBYannakakisPhaseHandle retained_handle(*bottom_up_phases[relation_idx],
		                                             *bottom_up_phases[relation_idx]);
		ResultDBYannakakisPhaseHandle parent_handle(*parent_phase, *parent_state_owner);
		auto retained_scan = make_uniq<PhysicalResultDBYannakakisRetainedScan>(
		    physical_plan, program.relations[relation_idx].types, retained_handle, parent_handle,
		    downward_key_indexes[step_idx]);
		top_down_phases[relation_idx] = make_uniq<PhysicalResultDBYannakakisPhase>(
		    physical_plan, std::move(retained_scan), relation_idx, program.relations[relation_idx].types,
		    memory_type, "RESULTDB_TOP_DOWN");
		top_down_phases[relation_idx]->AddInputKey(parent_handle, downward_key_indexes[step_idx]);
		configure_downward_keys(*top_down_phases[relation_idx], relation_idx);
		configure_outputs(*top_down_phases[relation_idx], relation_idx);
	}

	for (idx_t output_idx = 0; output_idx < output_distinct_indexes.size(); output_idx++) {
		if (output_distinct_indexes[output_idx] == DConstants::INVALID_INDEX) {
			throw InternalException("ResultDB output has no full phase DISTINCT producer");
		}
	}
}

PhysicalResultDBDirectCollector::~PhysicalResultDBDirectCollector() {
}

unique_ptr<GlobalSinkState> PhysicalResultDBDirectCollector::GetGlobalSinkState(ClientContext &context) const {
	return root_bottom_up_phase->GetGlobalSinkState(context);
}

unique_ptr<LocalSinkState> PhysicalResultDBDirectCollector::GetLocalSinkState(ExecutionContext &context) const {
	return root_bottom_up_phase->GetLocalSinkState(context);
}

SinkResultType PhysicalResultDBDirectCollector::Sink(ExecutionContext &context, DataChunk &chunk,
                                                     OperatorSinkInput &input) const {
	return root_bottom_up_phase->Sink(context, chunk, input);
}

SinkCombineResultType PhysicalResultDBDirectCollector::Combine(ExecutionContext &context,
                                                               OperatorSinkCombineInput &input) const {
	return root_bottom_up_phase->Combine(context, input);
}

SinkFinalizeType PhysicalResultDBDirectCollector::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                           OperatorSinkFinalizeInput &input) const {
	return root_bottom_up_phase->Finalize(pipeline, event, context, input);
}

unique_ptr<QueryResult> PhysicalResultDBDirectCollector::GetResult(GlobalSinkState &state) const {
	if (properties.resultdb.tables.empty()) {
		throw InternalException("RESULTDB query has no output tables");
	}

	auto context = root_bottom_up_phase->GetClientContext(state);
	if (properties.resultdb.tables.size() != program.outputs.size()) {
		throw InternalException("ResultDB direct output metadata does not match the prepared program");
	}
	auto statically_empty = program.edges.empty() && program.relations.size() > 1;
	vector<unique_ptr<ColumnDataCollection>> output_collections(properties.resultdb.tables.size());
	for (idx_t output_idx = 0; output_idx < program.outputs.size(); output_idx++) {
		auto &output = program.outputs[output_idx];
		auto types = GetResultDBDirectOutputTypes(output);
		if (statically_empty) {
			output_collections[output.table_metadata_index] =
			    CreateResultDBDirectCollection(*context, memory_type, types);
			continue;
		}
		if (output_distinct_indexes[output_idx] == DConstants::INVALID_INDEX) {
			throw InternalException("ResultDB direct output DISTINCT producer is missing");
		}

		PhysicalResultDBYannakakisPhase *phase;
		GlobalSinkState *phase_state;
		if (output.relation == program.root_relation) {
			phase = root_bottom_up_phase.get();
			phase_state = &state;
		} else {
			if (output.relation >= top_down_phases.size() || !top_down_phases[output.relation] ||
			    !top_down_phases[output.relation]->sink_state) {
				throw InternalException("ResultDB direct top-down output phase is unavailable");
			}
			phase = top_down_phases[output.relation].get();
			phase_state = phase->sink_state.get();
		}
		auto distinct_table =
		    phase->TakeOutputDistinct(*phase_state, output_distinct_indexes[output_idx]);
		output_collections[output.table_metadata_index] =
		    MaterializeResultDBDirectOutput(*context, memory_type, types, *distinct_table);
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
		    std::move(output_collections[table_idx]), context->GetClientProperties());
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
	result.push_back(base_plans[program.root_relation]);
	for (idx_t relation_idx = 0; relation_idx < bottom_up_phases.size(); relation_idx++) {
		if (bottom_up_phases[relation_idx]) {
			result.push_back(*bottom_up_phases[relation_idx]);
		}
	}
	for (idx_t relation_idx = 0; relation_idx < top_down_phases.size(); relation_idx++) {
		if (top_down_phases[relation_idx]) {
			result.push_back(*top_down_phases[relation_idx]);
		}
	}
	return result;
}

void PhysicalResultDBDirectCollector::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	sink_state.reset();
	root_bottom_up_phase->sink_state.reset();
	for (auto &phase : bottom_up_phases) {
		if (phase) {
			phase->sink_state.reset();
		}
	}
	for (auto &phase : top_down_phases) {
		if (phase) {
			phase->sink_state.reset();
		}
	}

	auto &state = meta_pipeline.GetState();
	state.SetPipelineSource(current, *this);

	struct ResultDBPhasePipelines {
		shared_ptr<Pipeline> completion;
		vector<shared_ptr<Pipeline>> consumers;
	};
	vector<ResultDBPhasePipelines> bottom_up_pipelines(program.relations.size());
	for (idx_t relation_idx = 0; relation_idx < base_plans.size(); relation_idx++) {
		PhysicalOperator *phase_sink;
		if (relation_idx == program.root_relation) {
			phase_sink = this;
		} else if (bottom_up_phases[relation_idx]) {
			phase_sink = bottom_up_phases[relation_idx].get();
		} else {
			continue;
		}
		auto &child_meta_pipeline = meta_pipeline.CreateChildMetaPipeline(current, *phase_sink);
		child_meta_pipeline.Build(base_plans[relation_idx].get());
		bottom_up_pipelines[relation_idx].completion = child_meta_pipeline.GetBasePipeline();
		child_meta_pipeline.GetPipelines(bottom_up_pipelines[relation_idx].consumers, false);
	}

	// A parent phase may start only after every child phase has completed
	// Combine and synchronous key-table Finalize. Add the dependency to every
	// pipeline sharing the parent sink, including UNION-created pipelines.
	for (idx_t child_relation = 0; child_relation < program.relations.size(); child_relation++) {
		if (child_relation == program.root_relation || !bottom_up_pipelines[child_relation].completion) {
			continue;
		}
		auto parent_relation = program.parent[child_relation];
		if (parent_relation >= bottom_up_pipelines.size() ||
		    !bottom_up_pipelines[parent_relation].completion) {
			throw InternalException("ResultDB bottom-up pipeline has an invalid parent dependency");
		}
		for (auto &parent_pipeline : bottom_up_pipelines[parent_relation].consumers) {
			parent_pipeline->AddDependency(bottom_up_pipelines[child_relation].completion);
		}
	}

	vector<ResultDBPhasePipelines> top_down_pipelines(program.relations.size());
	for (idx_t relation_idx = 0; relation_idx < top_down_phases.size(); relation_idx++) {
		if (!top_down_phases[relation_idx]) {
			continue;
		}
		auto &phase = *top_down_phases[relation_idx];
		auto &child_meta_pipeline = meta_pipeline.CreateChildMetaPipeline(current, phase);
		child_meta_pipeline.Build(phase.InputPlan());
		top_down_pipelines[relation_idx].completion = child_meta_pipeline.GetBasePipeline();
		child_meta_pipeline.GetPipelines(top_down_pipelines[relation_idx].consumers, false);
	}

	// Each top-down phase waits for both owners of its runtime inputs:
	// its own retained bottom-up rows and its parent's finalized downward key.
	for (idx_t relation_idx = 0; relation_idx < top_down_phases.size(); relation_idx++) {
		if (!top_down_phases[relation_idx]) {
			continue;
		}
		auto parent_relation = program.parent[relation_idx];
		auto &parent_completion =
		    parent_relation == program.root_relation ? bottom_up_pipelines[parent_relation].completion
		                                             : top_down_pipelines[parent_relation].completion;
		if (!bottom_up_pipelines[relation_idx].completion || !parent_completion) {
			throw InternalException("ResultDB top-down pipeline has an invalid phase dependency");
		}
		for (auto &consumer_pipeline : top_down_pipelines[relation_idx].consumers) {
			consumer_pipeline->AddDependency(bottom_up_pipelines[relation_idx].completion);
			consumer_pipeline->AddDependency(parent_completion);
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
