#include "duckdb/execution/operator/helper/physical_resultdb_yannakakis_phase.hpp"

#include "duckdb/common/enum_util.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/aggregate_hashtable.hpp"
#include "duckdb/execution/operator/helper/resultdb_direct_key_set.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parallel/meta_pipeline.hpp"
#include "duckdb/parallel/pipeline.hpp"
#include "duckdb/storage/buffer_manager.hpp"

namespace duckdb {

static unique_ptr<ColumnDataCollection>
CreateResultDBYannakakisPhaseCollection(ClientContext &context, QueryResultMemoryType memory_type,
                                        const vector<LogicalType> &types) {
	switch (memory_type) {
	case QueryResultMemoryType::IN_MEMORY:
		return make_uniq<ColumnDataCollection>(Allocator::DefaultAllocator(), types);
	case QueryResultMemoryType::BUFFER_MANAGED:
		return make_uniq<ColumnDataCollection>(BufferManager::GetBufferManager(*context.db), types,
		                                       ColumnDataCollectionLifetime::THROW_ERROR_AFTER_DATABASE_CLOSES);
	default:
		throw NotImplementedException("CreateResultDBYannakakisPhaseCollection for %s",
		                              EnumUtil::ToString(memory_type));
	}
}

class ResultDBYannakakisPhaseGlobalState : public GlobalSinkState {
public:
	ResultDBYannakakisPhaseGlobalState(ClientContext &context_p, const PhysicalResultDBYannakakisPhase &op,
	                                  QueryResultMemoryType memory_type, bool retain_rows,
	                                  const vector<LogicalType> &relation_types,
	                                  const vector<ResultDBYannakakisPhaseOutputKey> &output_keys,
	                                  const vector<ResultDBYannakakisPhaseOutputDistinct> &output_distincts)
	    : phase(op), context(context_p.shared_from_this()) {
		if (retain_rows) {
			retained_rows = CreateResultDBYannakakisPhaseCollection(context_p, memory_type, relation_types);
		}
		for (auto &output : output_keys) {
			output_key_sets.push_back(make_uniq<ResultDBDirectKeySet>(
			    context_p, op, relation_types, output.build_columns, output.probe_types, output.probe_columns));
		}
		auto &allocator = Allocator::Get(context_p);
		for (auto &output : output_distincts) {
			output_distinct_tables.push_back(
			    make_uniq<GroupedAggregateHashTable>(context_p, allocator, output.types));
		}
	}

	mutex combine_lock;
	reference<const PhysicalResultDBYannakakisPhase> phase;
	shared_ptr<ClientContext> context;
	vector<unique_ptr<ResultDBDirectKeySet>> output_key_sets;
	vector<unique_ptr<GroupedAggregateHashTable>> output_distinct_tables;
	unique_ptr<ColumnDataCollection> retained_rows;
	unique_ptr<ColumnDataCollection> input_rows;
};

struct ResultDBYannakakisPhaseProbe {
	ResultDBYannakakisPhaseProbe(ResultDBDirectKeySet &key_set_p,
	                             unique_ptr<ResultDBDirectKeySetProbeState> state_p)
	    : key_set(key_set_p), state(std::move(state_p)) {
	}

	reference<ResultDBDirectKeySet> key_set;
	unique_ptr<ResultDBDirectKeySetProbeState> state;
};

class ResultDBYannakakisPhaseLocalState : public LocalSinkState {
public:
	vector<ResultDBYannakakisPhaseProbe> probes;
	vector<unique_ptr<ResultDBDirectKeySetLocalBuildState>> output_build_states;
	vector<unique_ptr<GroupedAggregateHashTable>> output_distinct_tables;
	vector<unique_ptr<DataChunk>> output_projected_chunks;
	unique_ptr<ColumnDataCollection> retained_rows;
	ColumnDataAppendState retained_append_state;
	bool input_impossible = false;
};

static ResultDBYannakakisPhaseGlobalState &
CastResultDBYannakakisPhaseState(const PhysicalResultDBYannakakisPhase &phase, GlobalSinkState &state) {
	auto &result = state.Cast<ResultDBYannakakisPhaseGlobalState>();
	if (!RefersToSameObject(result.phase.get(), phase)) {
		throw InternalException("ResultDB Yannakakis sink state does not belong to this phase");
	}
	return result;
}

static ResultDBYannakakisPhaseGlobalState &
GetResultDBYannakakisPhaseState(const ResultDBYannakakisPhaseHandle &handle) {
	if (!handle.state_owner.get().sink_state) {
		throw InternalException("ResultDB Yannakakis phase state owner is not initialized");
	}
	return CastResultDBYannakakisPhaseState(handle.phase.get(), *handle.state_owner.get().sink_state);
}

class ResultDBYannakakisRetainedGlobalState : public GlobalSourceState {
public:
	explicit ResultDBYannakakisRetainedGlobalState(optional_ptr<ColumnDataCollection> collection_p)
	    : collection(collection_p), max_threads(collection ? MaxValue<idx_t>(collection->ChunkCount(), 1) : 1) {
		if (collection) {
			collection->InitializeScan(scan_state);
		}
	}

	idx_t MaxThreads() override {
		return max_threads;
	}

	optional_ptr<ColumnDataCollection> collection;
	ColumnDataParallelScanState scan_state;
	idx_t max_threads;
};

class ResultDBYannakakisRetainedLocalState : public LocalSourceState {
public:
	ColumnDataLocalScanState scan_state;
};

PhysicalResultDBYannakakisRetainedScan::PhysicalResultDBYannakakisRetainedScan(
    PhysicalPlan &physical_plan, vector<LogicalType> types, ResultDBYannakakisPhaseHandle retained_producer_p,
    ResultDBYannakakisPhaseHandle gate_producer_p, idx_t gate_key_index_p)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::COLUMN_DATA_SCAN, std::move(types), 0),
      retained_producer(std::move(retained_producer_p)), gate_producer(std::move(gate_producer_p)),
      gate_key_index(gate_key_index_p) {
}

unique_ptr<GlobalSourceState>
PhysicalResultDBYannakakisRetainedScan::GetGlobalSourceState(ClientContext &context) const {
	if (!consumer) {
		throw InternalException("ResultDB retained scan has no consumer phase");
	}
	auto &gate_state = GetResultDBYannakakisPhaseState(gate_producer);
	if (gate_producer.phase.get().OutputKeyCount(gate_state, gate_key_index) == 0) {
		// The parent eliminated this complete subtree. Drop the retained
		// bottom-up rows now without installing or initializing a scan.
		auto &retained_state = GetResultDBYannakakisPhaseState(retained_producer);
		auto discarded_rows = retained_producer.phase.get().TakeRetainedRows(retained_state);
		discarded_rows.reset();
		return make_uniq<ResultDBYannakakisRetainedGlobalState>(nullptr);
	}

	auto &retained_state = GetResultDBYannakakisPhaseState(retained_producer);
	auto rows = retained_producer.phase.get().TakeRetainedRows(retained_state);
	auto &consumer_state = GetResultDBYannakakisPhaseState(*consumer);
	consumer->phase.get().AdoptInputRows(consumer_state, std::move(rows));
	auto &input_rows = consumer->phase.get().GetInputRows(consumer_state);
	return make_uniq<ResultDBYannakakisRetainedGlobalState>(&input_rows);
}

unique_ptr<LocalSourceState>
PhysicalResultDBYannakakisRetainedScan::GetLocalSourceState(ExecutionContext &context,
                                                            GlobalSourceState &gstate) const {
	return make_uniq<ResultDBYannakakisRetainedLocalState>();
}

SourceResultType PhysicalResultDBYannakakisRetainedScan::GetDataInternal(ExecutionContext &context,
                                                                         DataChunk &chunk,
                                                                         OperatorSourceInput &input) const {
	auto &gstate = input.global_state.Cast<ResultDBYannakakisRetainedGlobalState>();
	if (!gstate.collection) {
		return SourceResultType::FINISHED;
	}
	auto &lstate = input.local_state.Cast<ResultDBYannakakisRetainedLocalState>();
	gstate.collection->Scan(gstate.scan_state, lstate.scan_state, chunk);
	return chunk.size() == 0 ? SourceResultType::FINISHED : SourceResultType::HAVE_MORE_OUTPUT;
}

string PhysicalResultDBYannakakisRetainedScan::GetName() const {
	return "RESULTDB_RETAINED_SCAN";
}

void PhysicalResultDBYannakakisRetainedScan::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	D_ASSERT(children.empty());
	meta_pipeline.GetState().SetPipelineSource(current, *this);
}

void PhysicalResultDBYannakakisRetainedScan::BindConsumer(ResultDBYannakakisPhaseHandle consumer_p) {
	if (consumer) {
		throw InternalException("ResultDB retained scan consumer was bound more than once");
	}
	consumer = make_uniq<ResultDBYannakakisPhaseHandle>(std::move(consumer_p));
}

void PhysicalResultDBYannakakisRetainedScan::ReleaseRetainedRows() const {
	if (!consumer) {
		throw InternalException("ResultDB retained scan has no consumer phase");
	}
	auto &consumer_state = GetResultDBYannakakisPhaseState(*consumer);
	consumer->phase.get().ReleaseInputRows(consumer_state);
}

static PhysicalOperator &
RequireResultDBYannakakisRetainedScan(
    const unique_ptr<PhysicalResultDBYannakakisRetainedScan> &retained_scan) {
	if (!retained_scan) {
		throw InternalException("ResultDB top-down phase requires a retained runtime scan");
	}
	return *retained_scan;
}

PhysicalResultDBYannakakisPhase::PhysicalResultDBYannakakisPhase(
    PhysicalPlan &physical_plan, PhysicalOperator &input_plan_p, idx_t relation_idx_p,
    vector<LogicalType> relation_types, QueryResultMemoryType memory_type_p, bool retain_rows_p,
    string phase_name_p)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::RESULT_COLLECTOR, std::move(relation_types), 0),
      retained_scan(nullptr), input_plan(input_plan_p), relation_idx(relation_idx_p), memory_type(memory_type_p),
      retain_rows(retain_rows_p), phase_name(std::move(phase_name_p)) {
}

PhysicalResultDBYannakakisPhase::PhysicalResultDBYannakakisPhase(
    PhysicalPlan &physical_plan, unique_ptr<PhysicalResultDBYannakakisRetainedScan> retained_scan_p,
    idx_t relation_idx_p, vector<LogicalType> relation_types, QueryResultMemoryType memory_type_p,
    string phase_name_p)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::RESULT_COLLECTOR, std::move(relation_types), 0),
      retained_scan(std::move(retained_scan_p)), input_plan(RequireResultDBYannakakisRetainedScan(retained_scan)),
      relation_idx(relation_idx_p),
      memory_type(memory_type_p), retain_rows(false), phase_name(std::move(phase_name_p)) {
	retained_scan->BindConsumer(ResultDBYannakakisPhaseHandle(*this, *this));
}

PhysicalResultDBYannakakisPhase::~PhysicalResultDBYannakakisPhase() {
}

void PhysicalResultDBYannakakisPhase::AddInputKey(ResultDBYannakakisPhaseHandle producer,
                                                  idx_t producer_key_index) {
	input_keys.emplace_back(std::move(producer), producer_key_index);
}

idx_t PhysicalResultDBYannakakisPhase::AddOutputKey(vector<idx_t> build_columns, vector<LogicalType> probe_types,
                                                    vector<idx_t> probe_columns) {
	ResultDBYannakakisPhaseOutputKey result;
	result.build_columns = std::move(build_columns);
	result.probe_types = std::move(probe_types);
	result.probe_columns = std::move(probe_columns);
	output_keys.push_back(std::move(result));
	return output_keys.size() - 1;
}

idx_t PhysicalResultDBYannakakisPhase::AddOutputDistinct(vector<idx_t> projection_columns,
                                                         vector<LogicalType> output_types) {
	if (projection_columns.empty() || projection_columns.size() != output_types.size()) {
		throw InternalException("ResultDB Yannakakis output DISTINCT projection is invalid");
	}
	for (idx_t column_idx = 0; column_idx < projection_columns.size(); column_idx++) {
		if (projection_columns[column_idx] >= types.size() ||
		    types[projection_columns[column_idx]] != output_types[column_idx]) {
			throw InternalException("ResultDB Yannakakis output DISTINCT column is invalid");
		}
	}
	ResultDBYannakakisPhaseOutputDistinct output;
	output.projection_columns = std::move(projection_columns);
	output.types = std::move(output_types);
	output_distincts.push_back(std::move(output));
	return output_distincts.size() - 1;
}

unique_ptr<GlobalSinkState> PhysicalResultDBYannakakisPhase::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<ResultDBYannakakisPhaseGlobalState>(context, *this, memory_type, retain_rows, types, output_keys,
	                                                     output_distincts);
}

unique_ptr<LocalSinkState> PhysicalResultDBYannakakisPhase::GetLocalSinkState(ExecutionContext &context) const {
	if (!context.pipeline || !context.pipeline->GetSink() || !context.pipeline->GetSink()->sink_state) {
		throw InternalException("ResultDB Yannakakis phase local state requires an initialized pipeline sink");
	}
	auto &gstate =
	    CastResultDBYannakakisPhaseState(*this, *context.pipeline->GetSink()->sink_state);
	auto result = make_uniq<ResultDBYannakakisPhaseLocalState>();
	for (auto &input : input_keys) {
		auto &producer_state = GetResultDBYannakakisPhaseState(input.producer);
		if (input.producer_key_index >= producer_state.output_key_sets.size() ||
		    !producer_state.output_key_sets[input.producer_key_index]) {
			throw InternalException("ResultDB Yannakakis phase references an invalid producer key table");
		}
		auto &key_set = *producer_state.output_key_sets[input.producer_key_index];
		if (key_set.Count() == 0) {
			result->input_impossible = true;
			break;
		}
		result->probes.emplace_back(key_set, key_set.CreateProbeState());
	}
	for (auto &key_set : gstate.output_key_sets) {
		result->output_build_states.push_back(key_set->CreateLocalBuildState());
	}
	auto &allocator = Allocator::Get(context.client);
	for (auto &output : output_distincts) {
		result->output_distinct_tables.push_back(
		    make_uniq<GroupedAggregateHashTable>(context.client, allocator, output.types));
		auto projected_chunk = make_uniq<DataChunk>();
		projected_chunk->Initialize(allocator, output.types);
		result->output_projected_chunks.push_back(std::move(projected_chunk));
	}
	if (gstate.retained_rows) {
		result->retained_rows =
		    CreateResultDBYannakakisPhaseCollection(context.client, memory_type, types);
		result->retained_rows->InitializeAppend(result->retained_append_state);
	}
	return std::move(result);
}

SinkResultType PhysicalResultDBYannakakisPhase::Sink(ExecutionContext &context, DataChunk &chunk,
                                                     OperatorSinkInput &input) const {
	auto &gstate = CastResultDBYannakakisPhaseState(*this, input.global_state);
	auto &lstate = input.local_state.Cast<ResultDBYannakakisPhaseLocalState>();
	if (lstate.input_impossible) {
		return SinkResultType::FINISHED;
	}

	auto survivors = &chunk;
	for (auto &probe : lstate.probes) {
		survivors = &probe.key_set.get().Probe(*survivors, *probe.state);
		if (survivors->size() == 0) {
			return SinkResultType::NEED_MORE_INPUT;
		}
	}

	if (lstate.retained_rows) {
		lstate.retained_rows->Append(lstate.retained_append_state, *survivors);
	}
	for (idx_t key_idx = 0; key_idx < gstate.output_key_sets.size(); key_idx++) {
		gstate.output_key_sets[key_idx]->Build(*survivors, *lstate.output_build_states[key_idx]);
	}
	DataChunk empty_payload;
	unsafe_vector<idx_t> filter;
	for (idx_t output_idx = 0; output_idx < output_distincts.size(); output_idx++) {
		auto &output = output_distincts[output_idx];
		auto &projected_chunk = *lstate.output_projected_chunks[output_idx];
		projected_chunk.Reset();
		for (idx_t column_idx = 0; column_idx < output.projection_columns.size(); column_idx++) {
			projected_chunk.data[column_idx].Reference(
			    survivors->data[output.projection_columns[column_idx]]);
		}
		projected_chunk.SetCardinality(*survivors);
		lstate.output_distinct_tables[output_idx]->AddChunk(projected_chunk, empty_payload, filter);
	}
	return SinkResultType::NEED_MORE_INPUT;
}

SinkCombineResultType PhysicalResultDBYannakakisPhase::Combine(ExecutionContext &context,
                                                               OperatorSinkCombineInput &input) const {
	auto &gstate = CastResultDBYannakakisPhaseState(*this, input.global_state);
	auto &lstate = input.local_state.Cast<ResultDBYannakakisPhaseLocalState>();
	for (idx_t key_idx = 0; key_idx < gstate.output_key_sets.size(); key_idx++) {
		gstate.output_key_sets[key_idx]->Combine(*lstate.output_build_states[key_idx]);
	}
	bool requires_lock = lstate.retained_rows && lstate.retained_rows->Count() > 0;
	for (auto &table : lstate.output_distinct_tables) {
		requires_lock = requires_lock || table->Count() > 0;
	}
	if (requires_lock) {
		lock_guard<mutex> guard(gstate.combine_lock);
		if (lstate.retained_rows && lstate.retained_rows->Count() > 0) {
			gstate.retained_rows->Combine(*lstate.retained_rows);
		}
		for (idx_t output_idx = 0; output_idx < lstate.output_distinct_tables.size(); output_idx++) {
			if (lstate.output_distinct_tables[output_idx]->Count() > 0) {
				gstate.output_distinct_tables[output_idx]->Combine(
				    *lstate.output_distinct_tables[output_idx]);
			}
		}
	}
	return SinkCombineResultType::FINISHED;
}

SinkFinalizeType PhysicalResultDBYannakakisPhase::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                           OperatorSinkFinalizeInput &input) const {
	auto &gstate = CastResultDBYannakakisPhaseState(*this, input.global_state);
	for (auto &key_set : gstate.output_key_sets) {
		key_set->Finalize();
	}
	// Every bottom-up or top-down key table has exactly one consumer in the
	// prepared tree. Once this phase finishes probing, its input key tables can
	// be released without affecting retained rows owned by their producers.
	for (auto &input_key : input_keys) {
		auto &producer_state = GetResultDBYannakakisPhaseState(input_key.producer);
		producer_state.output_key_sets[input_key.producer_key_index].reset();
	}
	if (retained_scan) {
		retained_scan->ReleaseRetainedRows();
	}
	return SinkFinalizeType::READY;
}

shared_ptr<ClientContext> PhysicalResultDBYannakakisPhase::GetClientContext(GlobalSinkState &state) const {
	return CastResultDBYannakakisPhaseState(*this, state).context;
}

idx_t PhysicalResultDBYannakakisPhase::OutputKeyCount(GlobalSinkState &state, idx_t key_index) const {
	auto &gstate = CastResultDBYannakakisPhaseState(*this, state);
	if (key_index >= gstate.output_key_sets.size() || !gstate.output_key_sets[key_index]) {
		throw InternalException("ResultDB Yannakakis phase output key table is unavailable");
	}
	return gstate.output_key_sets[key_index]->Count();
}

unique_ptr<ColumnDataCollection>
PhysicalResultDBYannakakisPhase::TakeRetainedRows(GlobalSinkState &state) const {
	auto &gstate = CastResultDBYannakakisPhaseState(*this, state);
	if (!gstate.retained_rows) {
		throw InternalException("ResultDB Yannakakis retained rows are unavailable");
	}
	return std::move(gstate.retained_rows);
}

void PhysicalResultDBYannakakisPhase::AdoptInputRows(
    GlobalSinkState &state, unique_ptr<ColumnDataCollection> rows) const {
	auto &gstate = CastResultDBYannakakisPhaseState(*this, state);
	if (!rows || gstate.input_rows) {
		throw InternalException("ResultDB Yannakakis input rows cannot be adopted");
	}
	gstate.input_rows = std::move(rows);
}

ColumnDataCollection &PhysicalResultDBYannakakisPhase::GetInputRows(GlobalSinkState &state) const {
	auto &gstate = CastResultDBYannakakisPhaseState(*this, state);
	if (!gstate.input_rows) {
		throw InternalException("ResultDB Yannakakis top-down input rows are unavailable");
	}
	return *gstate.input_rows;
}

void PhysicalResultDBYannakakisPhase::ReleaseInputRows(GlobalSinkState &state) const {
	auto &gstate = CastResultDBYannakakisPhaseState(*this, state);
	// The same single-consumer Finalize path runs when an empty parent gate
	// discarded producer rows before adoption, so no input collection is valid.
	if (!gstate.input_rows) {
		return;
	}
	gstate.input_rows.reset();
}

unique_ptr<GroupedAggregateHashTable>
PhysicalResultDBYannakakisPhase::TakeOutputDistinct(GlobalSinkState &state, idx_t distinct_index) const {
	auto &gstate = CastResultDBYannakakisPhaseState(*this, state);
	if (distinct_index >= gstate.output_distinct_tables.size() ||
	    !gstate.output_distinct_tables[distinct_index]) {
		throw InternalException("ResultDB Yannakakis output DISTINCT state is unavailable");
	}
	return std::move(gstate.output_distinct_tables[distinct_index]);
}

PhysicalOperator &PhysicalResultDBYannakakisPhase::InputPlan() const {
	return input_plan;
}

string PhysicalResultDBYannakakisPhase::GetName() const {
	return phase_name;
}

vector<const_reference<PhysicalOperator>> PhysicalResultDBYannakakisPhase::GetChildren() const {
	return {input_plan};
}

} // namespace duckdb
