#include "duckdb/execution/operator/helper/physical_resultdb_decompose_collector.hpp"

#include "duckdb/common/enum_util.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/aggregate_hashtable.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/materialized_query_result.hpp"
#include "duckdb/main/prepared_statement_data.hpp"
#include "duckdb/storage/buffer_manager.hpp"

namespace duckdb {

class ResultDBDecomposeCollectorGlobalState : public GlobalSinkState {
public:
	mutex combine_lock;
	vector<unique_ptr<GroupedAggregateHashTable>> tables;
	shared_ptr<ClientContext> context;
};

class ResultDBDecomposeCollectorLocalState : public LocalSinkState {
public:
	vector<unique_ptr<GroupedAggregateHashTable>> tables;
	vector<unique_ptr<DataChunk>> projected_chunks;
};

static vector<LogicalType> GetResultDBDecomposeTableTypes(const ResultDBTableMetadata &table) {
	vector<LogicalType> types;
	for (auto &column : table.columns) {
		types.push_back(column.type);
	}
	return types;
}

static vector<string> GetResultDBDecomposeTableNames(const ResultDBTableMetadata &table) {
	vector<string> names;
	for (auto &column : table.columns) {
		names.push_back(column.name);
	}
	return names;
}

static vector<idx_t> GetResultDBDecomposeFlatColumnIndexes(const ResultDBTableMetadata &table) {
	vector<idx_t> indexes;
	for (auto &column : table.columns) {
		indexes.push_back(column.flat_column_index);
	}
	return indexes;
}

static unique_ptr<ColumnDataCollection> CreateResultDBDecomposeCollection(ClientContext &context,
                                                                          QueryResultMemoryType memory_type,
                                                                          const vector<LogicalType> &types) {
	switch (memory_type) {
	case QueryResultMemoryType::IN_MEMORY:
		return make_uniq<ColumnDataCollection>(Allocator::DefaultAllocator(), types);
	case QueryResultMemoryType::BUFFER_MANAGED:
		return make_uniq<ColumnDataCollection>(BufferManager::GetBufferManager(*context.db), types,
		                                       ColumnDataCollectionLifetime::THROW_ERROR_AFTER_DATABASE_CLOSES);
	default:
		throw NotImplementedException("CreateResultDBDecomposeCollection for %s", EnumUtil::ToString(memory_type));
	}
}

static StatementProperties GetResultDBDecomposeSingleTableProperties(StatementProperties properties, idx_t table_idx) {
	auto table_metadata = properties.resultdb.tables[table_idx];
	for (idx_t column_idx = 0; column_idx < table_metadata.columns.size(); column_idx++) {
		table_metadata.columns[column_idx].flat_column_index = column_idx;
	}
	properties.resultdb.tables.clear();
	properties.resultdb.tables.push_back(std::move(table_metadata));
	return properties;
}

PhysicalResultDBDecomposeCollector::PhysicalResultDBDecomposeCollector(PhysicalPlan &physical_plan,
                                                                       PreparedStatementData &data)
    : PhysicalResultCollector(physical_plan, data) {
	for (auto &table : properties.resultdb.tables) {
		table_names.push_back(GetResultDBDecomposeTableNames(table));
		table_types.push_back(GetResultDBDecomposeTableTypes(table));
		table_flat_column_indexes.push_back(GetResultDBDecomposeFlatColumnIndexes(table));
	}
}

unique_ptr<GlobalSinkState> PhysicalResultDBDecomposeCollector::GetGlobalSinkState(ClientContext &context) const {
	if (properties.resultdb.tables.empty()) {
		throw InternalException("RESULTDB decompose collector expected output tables");
	}

	auto state = make_uniq<ResultDBDecomposeCollectorGlobalState>();
	state->context = context.shared_from_this();
	auto &allocator = Allocator::Get(context);
	for (auto &types : table_types) {
		state->tables.push_back(make_uniq<GroupedAggregateHashTable>(context, allocator, types));
	}
	return std::move(state);
}

unique_ptr<LocalSinkState> PhysicalResultDBDecomposeCollector::GetLocalSinkState(ExecutionContext &context) const {
	auto state = make_uniq<ResultDBDecomposeCollectorLocalState>();
	auto &allocator = Allocator::Get(context.client);
	for (auto &types : table_types) {
		state->tables.push_back(make_uniq<GroupedAggregateHashTable>(context.client, allocator, types));

		auto projected_chunk = make_uniq<DataChunk>();
		projected_chunk->Initialize(allocator, types);
		state->projected_chunks.push_back(std::move(projected_chunk));
	}
	return std::move(state);
}

SinkResultType PhysicalResultDBDecomposeCollector::Sink(ExecutionContext &context, DataChunk &chunk,
                                                        OperatorSinkInput &input) const {
	auto &lstate = input.local_state.Cast<ResultDBDecomposeCollectorLocalState>();
	DataChunk empty_payload;
	unsafe_vector<idx_t> filter;

	for (idx_t table_idx = 0; table_idx < properties.resultdb.tables.size(); table_idx++) {
		auto &flat_column_indexes = table_flat_column_indexes[table_idx];
		auto &projected_chunk = *lstate.projected_chunks[table_idx];
		projected_chunk.Reset();
		for (idx_t column_idx = 0; column_idx < flat_column_indexes.size(); column_idx++) {
			projected_chunk.data[column_idx].Reference(chunk.data[flat_column_indexes[column_idx]]);
		}
		projected_chunk.SetCardinality(chunk.size());
		lstate.tables[table_idx]->AddChunk(projected_chunk, empty_payload, filter);
	}

	return SinkResultType::NEED_MORE_INPUT;
}

SinkCombineResultType PhysicalResultDBDecomposeCollector::Combine(ExecutionContext &context,
                                                                  OperatorSinkCombineInput &input) const {
	auto &gstate = input.global_state.Cast<ResultDBDecomposeCollectorGlobalState>();
	auto &lstate = input.local_state.Cast<ResultDBDecomposeCollectorLocalState>();

	lock_guard<mutex> guard(gstate.combine_lock);
	for (idx_t table_idx = 0; table_idx < lstate.tables.size(); table_idx++) {
		if (lstate.tables[table_idx]->Count() > 0) {
			gstate.tables[table_idx]->Combine(*lstate.tables[table_idx]);
		}
	}
	return SinkCombineResultType::FINISHED;
}

unique_ptr<QueryResult> PhysicalResultDBDecomposeCollector::GetResult(GlobalSinkState &state) const {
	auto &gstate = state.Cast<ResultDBDecomposeCollectorGlobalState>();

	unique_ptr<QueryResult> first_result;
	QueryResult *current_result = nullptr;
	for (idx_t table_idx = 0; table_idx < properties.resultdb.tables.size(); table_idx++) {
		auto &types = table_types[table_idx];
		auto collection = CreateResultDBDecomposeCollection(*gstate.context, memory_type, types);
		ColumnDataAppendState append_state;
		collection->InitializeAppend(append_state);

		AggregateHTScanState scan_state;
		DataChunk distinct_rows;
		DataChunk payload_rows;
		distinct_rows.Initialize(Allocator::DefaultAllocator(), types);
		gstate.tables[table_idx]->InitializeScan(scan_state);
		while (true) {
			distinct_rows.Reset();
			payload_rows.Reset();
			if (!gstate.tables[table_idx]->Scan(scan_state, distinct_rows, payload_rows)) {
				break;
			}
			if (distinct_rows.size() > 0) {
				collection->Append(append_state, distinct_rows);
			}
		}

		auto table_result = make_uniq<MaterializedQueryResult>(
		    statement_type, GetResultDBDecomposeSingleTableProperties(properties, table_idx), table_names[table_idx],
		    std::move(collection), gstate.context->GetClientProperties());
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

bool PhysicalResultDBDecomposeCollector::ParallelSink() const {
	return true;
}

bool PhysicalResultDBDecomposeCollector::SinkOrderDependent() const {
	return false;
}

} // namespace duckdb
