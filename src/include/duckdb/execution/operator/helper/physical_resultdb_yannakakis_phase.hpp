//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/operator/helper/physical_resultdb_yannakakis_phase.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/main/query_result.hpp"

namespace duckdb {

class ColumnDataCollection;
class GroupedAggregateHashTable;
class ResultDBDirectKeySet;
class ResultDBDirectKeySetLocalBuildState;
class ResultDBDirectKeySetProbeState;
class PhysicalResultDBYannakakisPhase;

//! A phase and the physical operator that owns its GlobalSinkState.
//!
//! Root execution delegates the phase implementation to the result collector,
//! so the phase object and state owner are deliberately not assumed to be the
//! same operator.
struct ResultDBYannakakisPhaseHandle {
	ResultDBYannakakisPhaseHandle(PhysicalResultDBYannakakisPhase &phase_p, PhysicalOperator &state_owner_p)
	    : phase(phase_p), state_owner(state_owner_p) {
	}

	reference<PhysicalResultDBYannakakisPhase> phase;
	reference<PhysicalOperator> state_owner;
};

//! Describes one key-only hash table produced by a ResultDB phase.
struct ResultDBYannakakisPhaseOutputKey {
	vector<idx_t> build_columns;
	vector<LogicalType> probe_types;
	vector<idx_t> probe_columns;
};

//! Describes one finalized key table consumed by a ResultDB phase.
struct ResultDBYannakakisPhaseInputKey {
	ResultDBYannakakisPhaseInputKey(ResultDBYannakakisPhaseHandle producer_p, idx_t producer_key_index_p)
	    : producer(std::move(producer_p)), producer_key_index(producer_key_index_p) {
	}

	ResultDBYannakakisPhaseHandle producer;
	idx_t producer_key_index;
};

//! Describes one projection-level DISTINCT result produced by a phase.
struct ResultDBYannakakisPhaseOutputDistinct {
	vector<idx_t> projection_columns;
	vector<LogicalType> types;
};

//! Runtime source owned by the corresponding top-down consumer.
//! Producer state is resolved only after both pipeline dependencies complete.
//! At source initialization, retained rows move from the bottom-up producer
//! into the consumer phase state; no collection pointer is installed during planning.
class PhysicalResultDBYannakakisRetainedScan : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::INVALID;

	PhysicalResultDBYannakakisRetainedScan(PhysicalPlan &physical_plan, vector<LogicalType> types,
	                                      ResultDBYannakakisPhaseHandle retained_producer,
	                                      ResultDBYannakakisPhaseHandle gate_producer, idx_t gate_key_index);

public:
	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context) const override;
	unique_ptr<LocalSourceState> GetLocalSourceState(ExecutionContext &context,
	                                                 GlobalSourceState &gstate) const override;
	SourceResultType GetDataInternal(ExecutionContext &context, DataChunk &chunk,
	                                 OperatorSourceInput &input) const override;

	bool IsSource() const override {
		return true;
	}
	bool ParallelSource() const override {
		return true;
	}
	string GetName() const override;
	void BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) override;

	void BindConsumer(ResultDBYannakakisPhaseHandle consumer);
	void ReleaseRetainedRows() const;

private:
	ResultDBYannakakisPhaseHandle retained_producer;
	ResultDBYannakakisPhaseHandle gate_producer;
	idx_t gate_key_index;
	unique_ptr<ResultDBYannakakisPhaseHandle> consumer;
};

//! A sink for one dependency-aware Yannakakis phase.
//!
//! A phase probes all input key tables, optionally retains its surviving rows,
//! builds all output key tables, and updates output DISTINCT states in the same pass.
class PhysicalResultDBYannakakisPhase : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::INVALID;

	PhysicalResultDBYannakakisPhase(PhysicalPlan &physical_plan, PhysicalOperator &input_plan, idx_t relation_idx,
	                                vector<LogicalType> relation_types, QueryResultMemoryType memory_type,
	                                bool retain_rows, string phase_name);
	PhysicalResultDBYannakakisPhase(PhysicalPlan &physical_plan,
	                                unique_ptr<PhysicalResultDBYannakakisRetainedScan> retained_scan,
	                                idx_t relation_idx, vector<LogicalType> relation_types,
	                                QueryResultMemoryType memory_type, string phase_name);
	~PhysicalResultDBYannakakisPhase() override;

public:
	void AddInputKey(ResultDBYannakakisPhaseHandle producer, idx_t producer_key_index);
	idx_t AddOutputKey(vector<idx_t> build_columns, vector<LogicalType> probe_types,
	                   vector<idx_t> probe_columns);
	idx_t AddOutputDistinct(vector<idx_t> projection_columns, vector<LogicalType> output_types);

	shared_ptr<ClientContext> GetClientContext(GlobalSinkState &state) const;
	idx_t OutputKeyCount(GlobalSinkState &state, idx_t key_index) const;
	unique_ptr<ColumnDataCollection> TakeRetainedRows(GlobalSinkState &state) const;
	void AdoptInputRows(GlobalSinkState &state, unique_ptr<ColumnDataCollection> rows) const;
	ColumnDataCollection &GetInputRows(GlobalSinkState &state) const;
	void ReleaseInputRows(GlobalSinkState &state) const;
	unique_ptr<GroupedAggregateHashTable> TakeOutputDistinct(GlobalSinkState &state,
	                                                        idx_t distinct_index) const;
	PhysicalOperator &InputPlan() const;

public:
	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;
	unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const override;

	SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override;
	SinkCombineResultType Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const override;
	SinkFinalizeType Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
	                          OperatorSinkFinalizeInput &input) const override;

	bool IsSink() const override {
		return true;
	}
	bool ParallelSink() const override {
		return true;
	}
	bool SinkOrderDependent() const override {
		return false;
	}
	string GetName() const override;
	vector<const_reference<PhysicalOperator>> GetChildren() const override;

private:
	unique_ptr<PhysicalResultDBYannakakisRetainedScan> retained_scan;
	reference<PhysicalOperator> input_plan;
	idx_t relation_idx;
	QueryResultMemoryType memory_type;
	bool retain_rows;
	string phase_name;
	vector<ResultDBYannakakisPhaseInputKey> input_keys;
	vector<ResultDBYannakakisPhaseOutputKey> output_keys;
	vector<ResultDBYannakakisPhaseOutputDistinct> output_distincts;
};

} // namespace duckdb
