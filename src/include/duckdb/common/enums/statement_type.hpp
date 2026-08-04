//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/common/enums/statement_type.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/constants.hpp"
#include "duckdb/common/optional_idx.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/unordered_set.hpp"
#include "duckdb/main/query_parameters.hpp"
#include "duckdb/common/enums/database_modification_type.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// Statement Types
//===--------------------------------------------------------------------===//
enum class StatementType : uint8_t {
	INVALID_STATEMENT,      // invalid statement type
	SELECT_STATEMENT,       // select statement type
	INSERT_STATEMENT,       // insert statement type
	UPDATE_STATEMENT,       // update statement type
	CREATE_STATEMENT,       // create statement type
	DELETE_STATEMENT,       // delete statement type
	PREPARE_STATEMENT,      // prepare statement type
	EXECUTE_STATEMENT,      // execute statement type
	ALTER_STATEMENT,        // alter statement type
	TRANSACTION_STATEMENT,  // transaction statement type,
	COPY_STATEMENT,         // copy type
	ANALYZE_STATEMENT,      // analyze type
	VARIABLE_SET_STATEMENT, // variable set statement type
	CREATE_FUNC_STATEMENT,  // create func statement type
	EXPLAIN_STATEMENT,      // explain statement type
	DROP_STATEMENT,         // DROP statement type
	EXPORT_STATEMENT,       // EXPORT statement type
	PRAGMA_STATEMENT,       // PRAGMA statement type
	VACUUM_STATEMENT,       // VACUUM statement type
	CALL_STATEMENT,         // CALL statement type
	SET_STATEMENT,          // SET statement type
	LOAD_STATEMENT,         // LOAD statement type
	RELATION_STATEMENT,
	EXTENSION_STATEMENT,
	LOGICAL_PLAN_STATEMENT,
	ATTACH_STATEMENT,
	DETACH_STATEMENT,
	MULTI_STATEMENT,
	COPY_DATABASE_STATEMENT,
	UPDATE_EXTENSIONS_STATEMENT,
	MERGE_INTO_STATEMENT
};

DUCKDB_API string StatementTypeToString(StatementType type);

enum class StatementReturnType : uint8_t {
	QUERY_RESULT, // the statement returns a query result (e.g. for display to the user)
	CHANGED_ROWS, // the statement returns a single row containing the number of changed rows (e.g. an insert stmt)
	NOTHING       // the statement returns nothing
};

string StatementReturnTypeToString(StatementReturnType type);

class Catalog;
class ClientContext;

//! User-visible ResultDB planning variants. AUTO is the paper's TDResultDB
//! comparison between an optimized semijoin plan and decompose.
enum class ResultDBStrategy : uint8_t { DECOMPOSE, SEMIJOIN, TDROOT, TDFOLD_NO_TVC, TDFOLD, AUTO };

//! Physical ResultDB representation selected by planning.
enum class ResultDBExecutionStrategy : uint8_t { DECOMPOSE, SEMIJOIN };

DUCKDB_API ResultDBStrategy ResultDBStrategyFromString(const string &strategy);
DUCKDB_API string ResultDBStrategyToString(ResultDBStrategy strategy);

//! Metadata that lets the result collector split a flat SELECT RESULTDB result
//! back into separate per-table results.
struct ResultDBColumnMetadata {
	//! Position of this column in the normal flat SELECT output before decomposition.
	idx_t flat_column_index;
	//! Position of this column in its source relation occurrence.
	idx_t source_column_index = DConstants::INVALID_INDEX;
	string name;
	LogicalType type;
};

struct ResultDBTableMetadata {
	//! Table alias used in the SELECT query.
	string name;
	//! Bound relation occurrence for this table alias.
	idx_t table_index = DConstants::INVALID_INDEX;
	//! Output columns selected for this table, in table-local output order.
	vector<ResultDBColumnMetadata> columns;
};

struct ResultDBJoinColumnMetadata {
	idx_t left_column_index = DConstants::INVALID_INDEX;
	idx_t right_column_index = DConstants::INVALID_INDEX;
};

struct ResultDBJoinEdgeMetadata {
	//! Bound relation occurrences connected by this equality edge.
	idx_t left_table_index = DConstants::INVALID_INDEX;
	idx_t right_table_index = DConstants::INVALID_INDEX;
	vector<ResultDBJoinColumnMetadata> columns;
};

struct ResultDBProperties {
	bool enabled = false;
	//! Strategy requested through SET resultdb_strategy.
	ResultDBStrategy requested_strategy = ResultDBStrategy::DECOMPOSE;
	//! Physical strategy chosen by planning.
	ResultDBExecutionStrategy execution_strategy = ResultDBExecutionStrategy::DECOMPOSE;
	//! TDRoot/TDFold diagnostics retained for tests and thesis tooling.
	idx_t selected_root_table_index = DConstants::INVALID_INDEX;
	vector<vector<idx_t>> selected_folds;
	double estimated_semijoin_cost = 0;
	double estimated_decompose_cost = 0;
	double enumeration_time_ms = 0;
	idx_t fold_candidate_count = 0;
	vector<idx_t> block_sizes;
	bool tvc_enabled = false;
	string planning_reason;
	//! Tables to return after decomposing the flat SELECT result, in output order.
	vector<ResultDBTableMetadata> tables;
	//! Equality join graph used by the semijoin strategy.
	vector<ResultDBJoinEdgeMetadata> join_edges;
};

//! A struct containing various properties of a SQL statement
struct StatementProperties {
	StatementProperties()
	    : requires_valid_transaction(true), output_type(QueryResultOutputType::FORCE_MATERIALIZED),
	      bound_all_parameters(true), return_type(StatementReturnType::QUERY_RESULT), parameter_count(0),
	      always_require_rebind(false) {
	}

	struct CatalogIdentity {
		idx_t catalog_oid;
		optional_idx catalog_version;

		bool operator==(const CatalogIdentity &rhs) const {
			return catalog_oid == rhs.catalog_oid && catalog_version == rhs.catalog_version;
		}

		bool operator!=(const CatalogIdentity &rhs) const {
			return !operator==(rhs);
		}
	};

	struct ModificationInfo {
		CatalogIdentity identity;
		DatabaseModificationType modifications;
	};

	//! The set of databases this statement will read from
	unordered_map<string, CatalogIdentity> read_databases;
	//! The set of databases this statement will modify
	unordered_map<string, ModificationInfo> modified_databases;
	//! Whether or not the statement requires a valid transaction. Almost all statements require this, with the
	//! exception of ROLLBACK
	bool requires_valid_transaction;
	//! Whether or not the result can be streamed to the client
	QueryResultOutputType output_type;
	//! SELECT RESULTDB metadata produced by the binder and consumed by the result collector for decomposition
	ResultDBProperties resultdb;
	//! Whether or not all parameters have successfully had their types determined
	bool bound_all_parameters;
	//! What type of data the statement returns
	StatementReturnType return_type;
	//! The number of prepared statement parameters
	idx_t parameter_count;
	//! Whether or not the statement ALWAYS requires a rebind
	bool always_require_rebind;

	bool IsReadOnly() {
		return modified_databases.empty();
	}

	void RegisterDBRead(Catalog &catalog, ClientContext &context);
	void RegisterDBModify(Catalog &catalog, ClientContext &context, DatabaseModificationType modification);
};

} // namespace duckdb
