#include "duckdb/execution/operator/helper/resultdb_direct_key_set.hpp"

#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/row/partitioned_tuple_data.hpp"
#include "duckdb/common/types/row/tuple_data_collection.hpp"
#include "duckdb/execution/operator/join/physical_hash_join.hpp"
#include "duckdb/execution/join_hashtable.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"

namespace duckdb {

static vector<JoinCondition> BuildResultDBDirectKeySetConditions(const vector<LogicalType> &key_types) {
	vector<JoinCondition> result;
	result.reserve(key_types.size());
	for (idx_t key_idx = 0; key_idx < key_types.size(); key_idx++) {
		auto lhs = make_uniq<BoundReferenceExpression>(key_types[key_idx], key_idx);
		auto rhs = make_uniq<BoundReferenceExpression>(key_types[key_idx], key_idx);
		result.emplace_back(std::move(lhs), std::move(rhs), ExpressionType::COMPARE_EQUAL);
	}
	return result;
}

static vector<idx_t> BuildResultDBDirectKeySetIdentityProjection(idx_t column_count) {
	vector<idx_t> result;
	result.reserve(column_count);
	for (idx_t column_idx = 0; column_idx < column_count; column_idx++) {
		result.push_back(column_idx);
	}
	return result;
}

static vector<LogicalType> ValidateResultDBDirectKeySetColumns(const vector<LogicalType> &input_types,
                                                               const vector<idx_t> &columns,
                                                               const char *input_name) {
	if (columns.empty()) {
		throw InternalException("ResultDB direct key set has no %s key columns", input_name);
	}
	vector<LogicalType> result;
	result.reserve(columns.size());
	for (auto column_idx : columns) {
		if (column_idx >= input_types.size()) {
			throw InternalException("ResultDB direct key set %s key column is out of range", input_name);
		}
		result.push_back(input_types[column_idx]);
	}
	return result;
}

static void ValidateResultDBDirectKeySetInput(const DataChunk &input, const vector<LogicalType> &input_types,
                                              const vector<idx_t> &key_columns, const char *input_name) {
	if (input.ColumnCount() != input_types.size()) {
		throw InternalException("ResultDB direct key set received an invalid %s column count", input_name);
	}
#ifdef DEBUG
	for (idx_t column_idx = 0; column_idx < input_types.size(); column_idx++) {
		if (input.data[column_idx].GetType() != input_types[column_idx]) {
			throw InternalException("ResultDB direct key set received an invalid %s column type", input_name);
		}
	}
	for (auto key_column : key_columns) {
		D_ASSERT(key_column < input.ColumnCount());
	}
#else
	(void)key_columns;
#endif
}

class ResultDBDirectKeySet::Impl {
public:
	Impl(ClientContext &context_p, const PhysicalOperator &op_p, vector<LogicalType> build_input_types_p,
	     vector<idx_t> build_columns_p, vector<LogicalType> probe_input_types_p, vector<idx_t> probe_columns_p)
	    : context(context_p), op(op_p), build_input_types(std::move(build_input_types_p)),
	      build_columns(std::move(build_columns_p)), probe_input_types(std::move(probe_input_types_p)),
	      probe_columns(std::move(probe_columns_p)) {
		key_types = ValidateResultDBDirectKeySetColumns(build_input_types, build_columns, "build");
		auto probe_key_types =
		    ValidateResultDBDirectKeySetColumns(probe_input_types, probe_columns, "probe");
		if (key_types != probe_key_types) {
			throw InternalException("ResultDB direct key set build and probe key types do not match");
		}
		if (build_columns.size() != probe_columns.size()) {
			throw InternalException("ResultDB direct key set build and probe key arities do not match");
		}

		conditions = BuildResultDBDirectKeySetConditions(key_types);
		probe_output_columns = BuildResultDBDirectKeySetIdentityProjection(probe_input_types.size());
		table = CreateTable();
	}

	unique_ptr<JoinHashTable> CreateTable() const {
		vector<LogicalType> payload_types;
		return make_uniq<JoinHashTable>(context, op, conditions, std::move(payload_types), JoinType::SEMI, 0U,
		                                build_output_columns, nullptr, nullptr, probe_output_columns);
	}

	ClientContext &context;
	const PhysicalOperator &op;
	vector<LogicalType> build_input_types;
	vector<idx_t> build_columns;
	vector<LogicalType> probe_input_types;
	vector<idx_t> probe_columns;
	vector<LogicalType> key_types;
	//! JoinHashTable retains references to these three vectors.
	vector<JoinCondition> conditions;
	vector<idx_t> build_output_columns;
	vector<idx_t> probe_output_columns;
	unique_ptr<JoinHashTable> table;
	mutex combine_lock;
	bool finalized = false;
};

class ResultDBDirectKeySetLocalBuildState::Impl {
public:
	Impl(const ResultDBDirectKeySet::Impl &set, unique_ptr<JoinHashTable> table_p)
	    : owner(&set), table(std::move(table_p)) {
		keys.Initialize(Allocator::Get(set.context), set.key_types);
		table->GetSinkCollection().InitializeAppendState(append_state);
	}

	const ResultDBDirectKeySet::Impl *owner;
	unique_ptr<JoinHashTable> table;
	PartitionedTupleDataAppendState append_state;
	DataChunk keys;
	DataChunk empty_payload;
	bool combined = false;
};

class ResultDBDirectKeySetProbeState::Impl {
public:
	explicit Impl(const ResultDBDirectKeySet::Impl &set) : owner(&set) {
		keys.Initialize(Allocator::Get(set.context), set.key_types);
		probe_data.Initialize(Allocator::Get(set.context), set.probe_input_types);
		output.Initialize(Allocator::Get(set.context), set.probe_input_types);
		TupleDataCollection::InitializeChunkState(key_state, set.key_types);
		scan_structure = make_uniq<JoinHashTable::ScanStructure>(*set.table, key_state);
	}

	const ResultDBDirectKeySet::Impl *owner;
	DataChunk keys;
	DataChunk probe_data;
	DataChunk output;
	TupleDataChunkState key_state;
	JoinHashTable::ProbeState probe_state;
	unique_ptr<JoinHashTable::ScanStructure> scan_structure;
};

ResultDBDirectKeySet::ResultDBDirectKeySet(ClientContext &context, const PhysicalOperator &op,
                                           vector<LogicalType> build_input_types, vector<idx_t> build_columns,
                                           vector<LogicalType> probe_input_types, vector<idx_t> probe_columns)
    : impl(make_uniq<Impl>(context, op, std::move(build_input_types), std::move(build_columns),
                           std::move(probe_input_types), std::move(probe_columns))) {
}

ResultDBDirectKeySet::~ResultDBDirectKeySet() {
}

ResultDBDirectKeySetLocalBuildState::ResultDBDirectKeySetLocalBuildState(unique_ptr<Impl> impl_p)
    : impl(std::move(impl_p)) {
}

ResultDBDirectKeySetLocalBuildState::~ResultDBDirectKeySetLocalBuildState() {
}

ResultDBDirectKeySetProbeState::ResultDBDirectKeySetProbeState(unique_ptr<Impl> impl_p) : impl(std::move(impl_p)) {
}

ResultDBDirectKeySetProbeState::~ResultDBDirectKeySetProbeState() {
}

unique_ptr<ResultDBDirectKeySetLocalBuildState> ResultDBDirectKeySet::CreateLocalBuildState() const {
	if (impl->finalized) {
		throw InternalException("Cannot build a finalized ResultDB direct key set");
	}
	return unique_ptr<ResultDBDirectKeySetLocalBuildState>(
	    new ResultDBDirectKeySetLocalBuildState(make_uniq<ResultDBDirectKeySetLocalBuildState::Impl>(
	        *impl, impl->CreateTable())));
}

void ResultDBDirectKeySet::Build(DataChunk &input, ResultDBDirectKeySetLocalBuildState &state) const {
	auto &local = *state.impl;
	if (local.owner != impl.get() || local.combined || impl->finalized) {
		throw InternalException("ResultDB direct key set received an invalid local build state");
	}
	ValidateResultDBDirectKeySetInput(input, impl->build_input_types, impl->build_columns, "build");

	local.keys.Reset();
	for (idx_t key_idx = 0; key_idx < impl->build_columns.size(); key_idx++) {
		local.keys.data[key_idx].Reference(input.data[impl->build_columns[key_idx]]);
	}
	local.keys.SetCardinality(input);
	local.empty_payload.SetCardinality(input);
	local.table->Build(local.append_state, local.keys, local.empty_payload);
}

void ResultDBDirectKeySet::Combine(ResultDBDirectKeySetLocalBuildState &state) {
	auto &local = *state.impl;
	if (local.owner != impl.get() || local.combined || impl->finalized) {
		throw InternalException("ResultDB direct key set received an invalid local build state");
	}
	local.table->GetSinkCollection().FlushAppendState(local.append_state);
	lock_guard<mutex> guard(impl->combine_lock);
	impl->table->Merge(*local.table);
	local.combined = true;
}

void ResultDBDirectKeySet::Finalize() {
	lock_guard<mutex> guard(impl->combine_lock);
	if (impl->finalized) {
		throw InternalException("ResultDB direct key set was finalized more than once");
	}
	impl->table->Unpartition();
	if (impl->table->Count() > 0) {
		impl->table->AllocatePointerTable();
		impl->table->InitializePointerTable(0U, impl->table->capacity);
		impl->table->Finalize(0U, impl->table->GetDataCollection().ChunkCount(), false);
	}
	impl->table->finalized = true;
	impl->finalized = true;
}

unique_ptr<ResultDBDirectKeySetProbeState> ResultDBDirectKeySet::CreateProbeState() const {
	if (!impl->finalized) {
		throw InternalException("Cannot probe an unfinalized ResultDB direct key set");
	}
	return unique_ptr<ResultDBDirectKeySetProbeState>(
	    new ResultDBDirectKeySetProbeState(make_uniq<ResultDBDirectKeySetProbeState::Impl>(*impl)));
}

DataChunk &ResultDBDirectKeySet::Probe(DataChunk &input, ResultDBDirectKeySetProbeState &state) const {
	auto &probe = *state.impl;
	if (probe.owner != impl.get() || !impl->finalized) {
		throw InternalException("ResultDB direct key set received an invalid probe state");
	}
	ValidateResultDBDirectKeySetInput(input, impl->probe_input_types, impl->probe_columns, "probe");

	probe.output.Reset();
	if (input.size() == 0 || impl->table->Count() == 0) {
		return probe.output;
	}

	probe.keys.Reset();
	for (idx_t key_idx = 0; key_idx < impl->probe_columns.size(); key_idx++) {
		probe.keys.data[key_idx].Reference(input.data[impl->probe_columns[key_idx]]);
	}
	probe.keys.SetCardinality(input);
	probe.probe_data.Reference(input);

	impl->table->Probe(*probe.scan_structure, probe.keys, probe.key_state, probe.probe_state);
	probe.scan_structure->Next(probe.keys, probe.probe_data, probe.output);
	D_ASSERT(probe.scan_structure->PointersExhausted());
	return probe.output;
}

idx_t ResultDBDirectKeySet::Count() const {
	if (!impl->finalized) {
		throw InternalException("Cannot inspect an unfinalized ResultDB direct key set");
	}
	return impl->table->Count();
}

idx_t ResultDBDirectKeySet::SizeInBytes() const {
	if (!impl->finalized) {
		throw InternalException("Cannot inspect an unfinalized ResultDB direct key set");
	}
	return impl->table->SizeInBytes();
}

bool ResultDBDirectKeySet::IsFinalized() const {
	return impl->finalized;
}

} // namespace duckdb
