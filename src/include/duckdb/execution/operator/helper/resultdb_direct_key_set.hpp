//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/operator/helper/resultdb_direct_key_set.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/types.hpp"

namespace duckdb {

class ClientContext;
class DataChunk;
class PhysicalOperator;
class ResultDBDirectKeySetLocalBuildState;
class ResultDBDirectKeySetProbeState;

//! An exact, key-only membership set used by ResultDB semijoin phases.
//!
//! The set owns a DuckDB SEMI JoinHashTable. Each producer thread builds into a
//! local state, the states are combined, and Finalize must complete before any
//! probe state is created. Probe returns a chunk containing the complete probe
//! rows whose selected key occurs in the build input.
class ResultDBDirectKeySet {
public:
	ResultDBDirectKeySet(ClientContext &context, const PhysicalOperator &op, vector<LogicalType> build_input_types,
	                     vector<idx_t> build_columns, vector<LogicalType> probe_input_types,
	                     vector<idx_t> probe_columns);
	~ResultDBDirectKeySet();

	ResultDBDirectKeySet(const ResultDBDirectKeySet &) = delete;
	ResultDBDirectKeySet &operator=(const ResultDBDirectKeySet &) = delete;

public:
	unique_ptr<ResultDBDirectKeySetLocalBuildState> CreateLocalBuildState() const;
	void Build(DataChunk &input, ResultDBDirectKeySetLocalBuildState &state) const;
	void Combine(ResultDBDirectKeySetLocalBuildState &state);
	void Finalize();

	unique_ptr<ResultDBDirectKeySetProbeState> CreateProbeState() const;
	DataChunk &Probe(DataChunk &input, ResultDBDirectKeySetProbeState &state) const;

	idx_t Count() const;
	idx_t SizeInBytes() const;
	bool IsFinalized() const;

private:
	friend class ResultDBDirectKeySetLocalBuildState;
	friend class ResultDBDirectKeySetProbeState;
	class Impl;
	unique_ptr<Impl> impl;
};

class ResultDBDirectKeySetLocalBuildState {
public:
	~ResultDBDirectKeySetLocalBuildState();

private:
	friend class ResultDBDirectKeySet;
	class Impl;
	explicit ResultDBDirectKeySetLocalBuildState(unique_ptr<Impl> impl);

	unique_ptr<Impl> impl;
};

class ResultDBDirectKeySetProbeState {
public:
	~ResultDBDirectKeySetProbeState();

private:
	friend class ResultDBDirectKeySet;
	class Impl;
	explicit ResultDBDirectKeySetProbeState(unique_ptr<Impl> impl);

	unique_ptr<Impl> impl;
};

} // namespace duckdb
