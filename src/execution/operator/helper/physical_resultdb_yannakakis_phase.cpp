#include "duckdb/execution/operator/helper/physical_resultdb_yannakakis_phase.hpp"

#include "duckdb/common/enum_util.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/operator/helper/resultdb_direct_key_set.hpp"
#include "duckdb/main/client_context.hpp"
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
	                                  const vector<ResultDBYannakakisPhaseOutputKey> &output_keys)
	    : context(context_p.shared_from_this()) {
		if (retain_rows) {
			retained_rows = CreateResultDBYannakakisPhaseCollection(context_p, memory_type, relation_types);
		}
		for (auto &output : output_keys) {
			output_key_sets.push_back(make_uniq<ResultDBDirectKeySet>(
			    context_p, op, relation_types, output.build_columns, output.probe_types, output.probe_columns));
		}
	}

	mutex combine_lock;
	shared_ptr<ClientContext> context;
	vector<unique_ptr<ResultDBDirectKeySet>> output_key_sets;
	unique_ptr<ColumnDataCollection> retained_rows;
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
	unique_ptr<ColumnDataCollection> retained_rows;
	ColumnDataAppendState retained_append_state;
	bool input_impossible = false;
};

PhysicalResultDBYannakakisPhase::PhysicalResultDBYannakakisPhase(
    PhysicalPlan &physical_plan, PhysicalOperator &input_plan_p, idx_t relation_idx_p,
    vector<LogicalType> relation_types, QueryResultMemoryType memory_type_p, bool retain_rows_p,
    string phase_name_p)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::RESULT_COLLECTOR, std::move(relation_types), 0),
      input_plan(input_plan_p), relation_idx(relation_idx_p), memory_type(memory_type_p),
      retain_rows(retain_rows_p), phase_name(std::move(phase_name_p)) {
}

void PhysicalResultDBYannakakisPhase::AddInputKey(PhysicalOperator &producer, idx_t producer_key_index) {
	input_keys.emplace_back(producer, producer_key_index);
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

unique_ptr<GlobalSinkState> PhysicalResultDBYannakakisPhase::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<ResultDBYannakakisPhaseGlobalState>(context, *this, memory_type, retain_rows, types, output_keys);
}

unique_ptr<LocalSinkState> PhysicalResultDBYannakakisPhase::GetLocalSinkState(ExecutionContext &context) const {
	if (!context.pipeline || !context.pipeline->GetSink() || !context.pipeline->GetSink()->sink_state) {
		throw InternalException("ResultDB Yannakakis phase local state requires an initialized pipeline sink");
	}
	auto &gstate = context.pipeline->GetSink()->sink_state->Cast<ResultDBYannakakisPhaseGlobalState>();
	auto result = make_uniq<ResultDBYannakakisPhaseLocalState>();
	for (auto &input : input_keys) {
		if (!input.producer.get().sink_state) {
			throw InternalException("ResultDB Yannakakis phase producer state is not initialized");
		}
		auto &producer_state = input.producer.get().sink_state->Cast<ResultDBYannakakisPhaseGlobalState>();
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
	if (gstate.retained_rows) {
		result->retained_rows =
		    CreateResultDBYannakakisPhaseCollection(context.client, memory_type, types);
		result->retained_rows->InitializeAppend(result->retained_append_state);
	}
	return std::move(result);
}

SinkResultType PhysicalResultDBYannakakisPhase::Sink(ExecutionContext &context, DataChunk &chunk,
                                                     OperatorSinkInput &input) const {
	auto &gstate = input.global_state.Cast<ResultDBYannakakisPhaseGlobalState>();
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
	return SinkResultType::NEED_MORE_INPUT;
}

SinkCombineResultType PhysicalResultDBYannakakisPhase::Combine(ExecutionContext &context,
                                                               OperatorSinkCombineInput &input) const {
	auto &gstate = input.global_state.Cast<ResultDBYannakakisPhaseGlobalState>();
	auto &lstate = input.local_state.Cast<ResultDBYannakakisPhaseLocalState>();
	for (idx_t key_idx = 0; key_idx < gstate.output_key_sets.size(); key_idx++) {
		gstate.output_key_sets[key_idx]->Combine(*lstate.output_build_states[key_idx]);
	}
	if (lstate.retained_rows && lstate.retained_rows->Count() > 0) {
		lock_guard<mutex> guard(gstate.combine_lock);
		gstate.retained_rows->Combine(*lstate.retained_rows);
	}
	return SinkCombineResultType::FINISHED;
}

SinkFinalizeType PhysicalResultDBYannakakisPhase::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                           OperatorSinkFinalizeInput &input) const {
	auto &gstate = input.global_state.Cast<ResultDBYannakakisPhaseGlobalState>();
	for (auto &key_set : gstate.output_key_sets) {
		key_set->Finalize();
	}
	// Every bottom-up or top-down key table has exactly one consumer in the
	// prepared tree. Once this phase finishes probing, its input key tables can
	// be released without affecting retained rows owned by their producers.
	for (auto &input_key : input_keys) {
		auto &producer_state = input_key.producer.get().sink_state->Cast<ResultDBYannakakisPhaseGlobalState>();
		producer_state.output_key_sets[input_key.producer_key_index].reset();
	}
	return SinkFinalizeType::READY;
}

unique_ptr<ColumnDataCollection>
PhysicalResultDBYannakakisPhase::TakeRetainedRows(GlobalSinkState &state) const {
	auto &gstate = state.Cast<ResultDBYannakakisPhaseGlobalState>();
	return std::move(gstate.retained_rows);
}

shared_ptr<ClientContext> PhysicalResultDBYannakakisPhase::GetClientContext(GlobalSinkState &state) const {
	return state.Cast<ResultDBYannakakisPhaseGlobalState>().context;
}

idx_t PhysicalResultDBYannakakisPhase::OutputKeyCount(GlobalSinkState &state, idx_t key_index) const {
	auto &gstate = state.Cast<ResultDBYannakakisPhaseGlobalState>();
	if (key_index >= gstate.output_key_sets.size() || !gstate.output_key_sets[key_index]) {
		throw InternalException("ResultDB Yannakakis phase output key table is unavailable");
	}
	return gstate.output_key_sets[key_index]->Count();
}

string PhysicalResultDBYannakakisPhase::GetName() const {
	return phase_name;
}

vector<const_reference<PhysicalOperator>> PhysicalResultDBYannakakisPhase::GetChildren() const {
	return {input_plan};
}

} // namespace duckdb
