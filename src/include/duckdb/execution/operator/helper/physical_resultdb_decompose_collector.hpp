//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/operator/helper/physical_resultdb_decompose_collector.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/execution/operator/helper/physical_result_collector.hpp"

namespace duckdb {

//! PhysicalResultDBDecomposeCollector runs the normal flat plan and decomposes chunks into distinct source relations.
class PhysicalResultDBDecomposeCollector : public PhysicalResultCollector {
public:
	PhysicalResultDBDecomposeCollector(PhysicalPlan &physical_plan, PreparedStatementData &data);

public:
	unique_ptr<QueryResult> GetResult(GlobalSinkState &state) const override;

public:
	// Sink interface
	SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override;
	SinkCombineResultType Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const override;

	unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const override;
	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;

	bool ParallelSink() const override;
	bool SinkOrderDependent() const override;

private:
	vector<vector<string>> table_names;
	vector<vector<LogicalType>> table_types;
	vector<vector<idx_t>> table_flat_column_indexes;
};

} // namespace duckdb
