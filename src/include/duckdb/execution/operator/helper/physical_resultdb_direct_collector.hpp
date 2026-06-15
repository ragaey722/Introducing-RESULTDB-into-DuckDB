//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/operator/helper/physical_resultdb_direct_collector.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/unordered_map.hpp"
#include "duckdb/execution/operator/helper/physical_result_collector.hpp"

namespace duckdb {

struct PreparedResultDBYannakakisProgram;

//! PhysicalResultDBDirectCollector executes a shared Yannakakis program and returns each reduced relation separately.
class PhysicalResultDBDirectCollector : public PhysicalResultCollector {
public:
	PhysicalResultDBDirectCollector(PhysicalPlan &physical_plan, PreparedStatementData &data);

public:
	unique_ptr<QueryResult> GetResult(GlobalSinkState &state) const override;

public:
	// Sink interface
	SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override;
	SinkCombineResultType Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const override;

	unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const override;
	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;

	vector<const_reference<PhysicalOperator>> GetChildren() const override;
	void BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) override;

	bool ParallelSink() const override;
	bool SinkOrderDependent() const override;

private:
	const PreparedResultDBYannakakisProgram &program;
	vector<reference<PhysicalOperator>> phase_plans;
	// Raw pipeline pointers are valid for the executor lifetime; the meta-pipeline graph owns the pipelines.
	mutable unordered_map<const Pipeline *, idx_t> pipeline_phase_map;
};

} // namespace duckdb
