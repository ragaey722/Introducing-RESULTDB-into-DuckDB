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
class ResultDBDirectKeySet;
class ResultDBDirectKeySetLocalBuildState;
class ResultDBDirectKeySetProbeState;

//! Describes one key-only hash table produced by a ResultDB phase.
struct ResultDBYannakakisPhaseOutputKey {
	vector<idx_t> build_columns;
	vector<LogicalType> probe_types;
	vector<idx_t> probe_columns;
};

//! Describes one finalized key table consumed by a ResultDB phase.
struct ResultDBYannakakisPhaseInputKey {
	ResultDBYannakakisPhaseInputKey(PhysicalOperator &producer_p, idx_t producer_key_index_p)
	    : producer(producer_p), producer_key_index(producer_key_index_p) {
	}

	reference<PhysicalOperator> producer;
	idx_t producer_key_index;
};

//! A sink for one dependency-aware Yannakakis phase.
//!
//! A phase probes all input key tables, optionally retains its surviving rows,
//! and builds all output key tables in the same pass.
class PhysicalResultDBYannakakisPhase : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::INVALID;

	PhysicalResultDBYannakakisPhase(PhysicalPlan &physical_plan, PhysicalOperator &input_plan, idx_t relation_idx,
	                                vector<LogicalType> relation_types, QueryResultMemoryType memory_type,
	                                bool retain_rows, string phase_name);

public:
	void AddInputKey(PhysicalOperator &producer, idx_t producer_key_index);
	idx_t AddOutputKey(vector<idx_t> build_columns, vector<LogicalType> probe_types,
	                   vector<idx_t> probe_columns);

	unique_ptr<ColumnDataCollection> TakeRetainedRows(GlobalSinkState &state) const;
	shared_ptr<ClientContext> GetClientContext(GlobalSinkState &state) const;
	idx_t OutputKeyCount(GlobalSinkState &state, idx_t key_index) const;

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
	reference<PhysicalOperator> input_plan;
	idx_t relation_idx;
	QueryResultMemoryType memory_type;
	bool retain_rows;
	string phase_name;
	vector<ResultDBYannakakisPhaseInputKey> input_keys;
	vector<ResultDBYannakakisPhaseOutputKey> output_keys;
};

} // namespace duckdb
