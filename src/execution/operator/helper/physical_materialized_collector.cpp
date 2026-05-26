#include "duckdb/execution/operator/helper/physical_materialized_collector.hpp"

#include "duckdb/main/materialized_query_result.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/result_set_manager.hpp"
#include "duckdb/common/enum_util.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/hash.hpp"
#include "duckdb/common/unordered_set.hpp"
#include "duckdb/storage/buffer_manager.hpp"

namespace duckdb {

PhysicalMaterializedCollector::PhysicalMaterializedCollector(PhysicalPlan &physical_plan, PreparedStatementData &data,
                                                             bool parallel)
    : PhysicalResultCollector(physical_plan, data), parallel(parallel) {
}

class MaterializedCollectorGlobalState : public GlobalSinkState {
public:
	mutex glock;
	unique_ptr<ColumnDataCollection> collection;
	shared_ptr<ClientContext> context;
};

class MaterializedCollectorLocalState : public LocalSinkState {
public:
	unique_ptr<ColumnDataCollection> collection;
	ColumnDataAppendState append_state;
};

struct ResultDBRowKey {
	//! Values for one projected flat source-table row.
	vector<Value> values;
};

struct ResultDBRowKeyHash {
	idx_t operator()(const ResultDBRowKey &key) const {
		hash_t result = 0;
		for (auto &value : key.values) {
			result = CombineHash(result, value.Hash());
		}
		return result;
	}
};

struct ResultDBRowKeyEquality {
	bool operator()(const ResultDBRowKey &left, const ResultDBRowKey &right) const {
		if (left.values.size() != right.values.size()) {
			return false;
		}
		for (idx_t i = 0; i < left.values.size(); i++) {
			if (!Value::NotDistinctFrom(left.values[i], right.values[i])) {
				return false;
			}
		}
		return true;
	}
};

struct ResultDBOutputTable {
	string name;
	vector<string> names;
	vector<LogicalType> types;
	//! Indexes into the flat SELECT result for the columns that belong to this output table.
	vector<idx_t> flat_column_indexes;
	unique_ptr<ColumnDataCollection> collection;
	ColumnDataAppendState append_state;
	DataChunk append_chunk;
	unordered_set<ResultDBRowKey, ResultDBRowKeyHash, ResultDBRowKeyEquality> seen_rows;
};

static unique_ptr<ColumnDataCollection> CreateResultDBCollection(ClientContext &context, QueryResultMemoryType memory_type,
                                                                 const vector<LogicalType> &types) {
	switch (memory_type) {
	case QueryResultMemoryType::IN_MEMORY:
		return make_uniq<ColumnDataCollection>(Allocator::DefaultAllocator(), types);
	case QueryResultMemoryType::BUFFER_MANAGED:
		return make_uniq<ColumnDataCollection>(BufferManager::GetBufferManager(*context.db), types,
		                                       ColumnDataCollectionLifetime::THROW_ERROR_AFTER_DATABASE_CLOSES);
	default:
		throw NotImplementedException("CreateResultDBCollection for %s", EnumUtil::ToString(memory_type));
	}
}

static unique_ptr<ResultDBOutputTable> CreateResultDBOutputTable(ClientContext &context,
                                                                 QueryResultMemoryType memory_type,
                                                                 const ResultDBTableMetadata &metadata) {
	auto result = make_uniq<ResultDBOutputTable>();
	result->name = metadata.name;
	for (auto &column : metadata.columns) {
		result->names.push_back(column.name);
		result->types.push_back(column.type);
		result->flat_column_indexes.push_back(column.flat_column_index);
	}
	result->collection = CreateResultDBCollection(context, memory_type, result->types);
	result->collection->InitializeAppend(result->append_state);
	result->append_chunk.Initialize(Allocator::DefaultAllocator(), result->types, 1);
	return result;
}

static void AppendResultDBRow(ResultDBOutputTable &table, DataChunk &flat_chunk, idx_t row_idx) {
	ResultDBRowKey key;
	key.values.reserve(table.flat_column_indexes.size());
	for (auto flat_column_index : table.flat_column_indexes) {
		key.values.push_back(flat_chunk.GetValue(flat_column_index, row_idx));
	}

	auto insert_result = table.seen_rows.insert(std::move(key));
	if (!insert_result.second) {
		// Joins can repeat the same source row many times; each ResultDB table returns it once.
		return;
	}
	auto &values = insert_result.first->values;
	table.append_chunk.Reset();
	for (idx_t column_idx = 0; column_idx < values.size(); column_idx++) {
		table.append_chunk.SetValue(column_idx, 0, values[column_idx]);
	}
	table.append_chunk.SetCardinality(1);
	table.collection->Append(table.append_state, table.append_chunk);
}

// Build a QueryResult::next chain from one flat materialized SELECT RESULTDB result.
static unique_ptr<QueryResult> BuildResultDBQueryResult(ClientContext &context, StatementType statement_type,
                                                        StatementProperties properties,
                                                        QueryResultMemoryType memory_type,
                                                        ColumnDataCollection &flat_collection) {
	if (properties.resultdb.tables.empty()) {
		throw InternalException("RESULTDB query has no output tables");
	}

	vector<unique_ptr<ResultDBOutputTable>> output_tables;
	for (auto &table : properties.resultdb.tables) {
		output_tables.push_back(CreateResultDBOutputTable(context, memory_type, table));
	}

	// The query has already run as a normal flat SELECT. Scan that materialized result
	// once and project each flat row into the requested per-table output collections.
	ColumnDataScanState scan_state;
	DataChunk flat_chunk;
	flat_collection.InitializeScan(scan_state);
	flat_collection.InitializeScanChunk(flat_chunk);
	while (flat_collection.Scan(scan_state, flat_chunk)) {
		for (idx_t row_idx = 0; row_idx < flat_chunk.size(); row_idx++) {
			for (auto &table : output_tables) {
				AppendResultDBRow(*table, flat_chunk, row_idx);
			}
		}
	}

	unique_ptr<QueryResult> first_result;
	QueryResult *current_result = nullptr;
	for (idx_t table_idx = 0; table_idx < output_tables.size(); table_idx++) {
		auto &table = output_tables[table_idx];
		auto table_properties = properties;
		table_properties.resultdb.tables.clear();
		auto table_metadata = properties.resultdb.tables[table_idx];
		// Each returned QueryResult describes one decomposed table, so its column
		// indexes are local to that table rather than the original flat SELECT output.
		for (idx_t column_idx = 0; column_idx < table_metadata.columns.size(); column_idx++) {
			table_metadata.columns[column_idx].flat_column_index = column_idx;
		}
		table_properties.resultdb.tables.push_back(std::move(table_metadata));
		auto table_result = make_uniq<MaterializedQueryResult>(statement_type, table_properties, table->names,
		                                                       std::move(table->collection),
		                                                       context.GetClientProperties());
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

SinkResultType PhysicalMaterializedCollector::Sink(ExecutionContext &context, DataChunk &chunk,
                                                   OperatorSinkInput &input) const {
	auto &lstate = input.local_state.Cast<MaterializedCollectorLocalState>();
	lstate.collection->Append(lstate.append_state, chunk);
	return SinkResultType::NEED_MORE_INPUT;
}

SinkCombineResultType PhysicalMaterializedCollector::Combine(ExecutionContext &context,
                                                             OperatorSinkCombineInput &input) const {
	auto &gstate = input.global_state.Cast<MaterializedCollectorGlobalState>();
	auto &lstate = input.local_state.Cast<MaterializedCollectorLocalState>();
	if (lstate.collection->Count() == 0) {
		return SinkCombineResultType::FINISHED;
	}

	lock_guard<mutex> l(gstate.glock);
	if (!gstate.collection) {
		gstate.collection = std::move(lstate.collection);
	} else {
		gstate.collection->Combine(*lstate.collection);
	}

	return SinkCombineResultType::FINISHED;
}

unique_ptr<GlobalSinkState> PhysicalMaterializedCollector::GetGlobalSinkState(ClientContext &context) const {
	auto state = make_uniq<MaterializedCollectorGlobalState>();
	state->context = context.shared_from_this();
	return std::move(state);
}

unique_ptr<LocalSinkState> PhysicalMaterializedCollector::GetLocalSinkState(ExecutionContext &context) const {
	auto state = make_uniq<MaterializedCollectorLocalState>();
	state->collection = CreateCollection(context.client);
	state->collection->InitializeAppend(state->append_state);
	return std::move(state);
}

unique_ptr<QueryResult> PhysicalMaterializedCollector::GetResult(GlobalSinkState &state) const {
	auto &gstate = state.Cast<MaterializedCollectorGlobalState>();
	if (!gstate.collection) {
		gstate.collection = CreateCollection(*gstate.context);
	}
	if (properties.resultdb.enabled) {
		return BuildResultDBQueryResult(*gstate.context, statement_type, properties, memory_type, *gstate.collection);
	}
	auto result = make_uniq<MaterializedQueryResult>(statement_type, properties, names, std::move(gstate.collection),
	                                                 gstate.context->GetClientProperties());
	return std::move(result);
}

bool PhysicalMaterializedCollector::ParallelSink() const {
	return parallel;
}

bool PhysicalMaterializedCollector::SinkOrderDependent() const {
	return true;
}

} // namespace duckdb
