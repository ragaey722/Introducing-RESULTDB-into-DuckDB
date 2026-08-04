#include "duckdb/common/enums/statement_type.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"

namespace duckdb {

ResultDBStrategy ResultDBStrategyFromString(const string &strategy) {
	auto normalized = StringUtil::Lower(strategy);
	if (normalized == "decompose") {
		return ResultDBStrategy::DECOMPOSE;
	}
	if (normalized == "semijoin") {
		return ResultDBStrategy::SEMIJOIN;
	}
	if (normalized == "tdroot") {
		return ResultDBStrategy::TDROOT;
	}
	if (normalized == "tdfold_no_tvc" || normalized == "tdfold-no-tvc") {
		return ResultDBStrategy::TDFOLD_NO_TVC;
	}
	if (normalized == "tdfold") {
		return ResultDBStrategy::TDFOLD;
	}
	if (normalized == "auto" || normalized == "tdresultdb") {
		return ResultDBStrategy::AUTO;
	}
	throw InvalidInputException("Unrecognized resultdb_strategy \"%s\". Expected decompose, semijoin, tdroot, "
	                            "tdfold_no_tvc, tdfold, auto, or tdresultdb", strategy);
}

string ResultDBStrategyToString(ResultDBStrategy strategy) {
	switch (strategy) {
	case ResultDBStrategy::DECOMPOSE:
		return "decompose";
	case ResultDBStrategy::SEMIJOIN:
		return "semijoin";
	case ResultDBStrategy::TDROOT:
		return "tdroot";
	case ResultDBStrategy::TDFOLD_NO_TVC:
		return "tdfold_no_tvc";
	case ResultDBStrategy::TDFOLD:
		return "tdfold";
	case ResultDBStrategy::AUTO:
		return "auto";
	}
	throw InternalException("Unknown ResultDB strategy");
}

// LCOV_EXCL_START
string StatementTypeToString(StatementType type) {
	switch (type) {
	case StatementType::SELECT_STATEMENT:
		return "SELECT";
	case StatementType::INSERT_STATEMENT:
		return "INSERT";
	case StatementType::UPDATE_STATEMENT:
		return "UPDATE";
	case StatementType::DELETE_STATEMENT:
		return "DELETE";
	case StatementType::PREPARE_STATEMENT:
		return "PREPARE";
	case StatementType::EXECUTE_STATEMENT:
		return "EXECUTE";
	case StatementType::ALTER_STATEMENT:
		return "ALTER";
	case StatementType::TRANSACTION_STATEMENT:
		return "TRANSACTION";
	case StatementType::COPY_STATEMENT:
		return "COPY";
	case StatementType::COPY_DATABASE_STATEMENT:
		return "COPY_DATABASE";
	case StatementType::ANALYZE_STATEMENT:
		return "ANALYZE";
	case StatementType::VARIABLE_SET_STATEMENT:
		return "VARIABLE_SET";
	case StatementType::CREATE_FUNC_STATEMENT:
		return "CREATE_FUNC";
	case StatementType::EXPLAIN_STATEMENT:
		return "EXPLAIN";
	case StatementType::CREATE_STATEMENT:
		return "CREATE";
	case StatementType::DROP_STATEMENT:
		return "DROP";
	case StatementType::PRAGMA_STATEMENT:
		return "PRAGMA";
	case StatementType::VACUUM_STATEMENT:
		return "VACUUM";
	case StatementType::RELATION_STATEMENT:
		return "RELATION";
	case StatementType::EXPORT_STATEMENT:
		return "EXPORT";
	case StatementType::CALL_STATEMENT:
		return "CALL";
	case StatementType::SET_STATEMENT:
		return "SET";
	case StatementType::LOAD_STATEMENT:
		return "LOAD";
	case StatementType::EXTENSION_STATEMENT:
		return "EXTENSION";
	case StatementType::LOGICAL_PLAN_STATEMENT:
		return "LOGICAL_PLAN";
	case StatementType::ATTACH_STATEMENT:
		return "ATTACH";
	case StatementType::DETACH_STATEMENT:
		return "DETACH";
	case StatementType::MULTI_STATEMENT:
		return "MULTI";
	case StatementType::UPDATE_EXTENSIONS_STATEMENT:
		return "UPDATE_EXTENSIONS";
	case StatementType::MERGE_INTO_STATEMENT:
		return "MERGE_INTO";
	case StatementType::INVALID_STATEMENT:
		break;
	}
	return "INVALID";
}

string StatementReturnTypeToString(StatementReturnType type) {
	switch (type) {
	case StatementReturnType::QUERY_RESULT:
		return "QUERY_RESULT";
	case StatementReturnType::CHANGED_ROWS:
		return "CHANGED_ROWS";
	case StatementReturnType::NOTHING:
		return "NOTHING";
	}
	return "INVALID";
}
// LCOV_EXCL_STOP

void StatementProperties::RegisterDBRead(Catalog &catalog, ClientContext &context) {
	auto catalog_identity = CatalogIdentity {catalog.GetOid(), catalog.GetCatalogVersion(context)};
	D_ASSERT(read_databases.count(catalog.GetName()) == 0 || read_databases[catalog.GetName()] == catalog_identity);
	read_databases[catalog.GetName()] = catalog_identity;
}

void StatementProperties::RegisterDBModify(Catalog &catalog, ClientContext &context,
                                           DatabaseModificationType modification) {
	auto catalog_identity = CatalogIdentity {catalog.GetOid(), catalog.GetCatalogVersion(context)};
	auto entry = modified_databases.insert(make_pair(catalog.GetName(), ModificationInfo()));
	if (entry.second) {
		// new entry - set the identity
		entry.first->second.identity = catalog_identity;
	} else {
		// existing entry - verify this has the same identity
		D_ASSERT(entry.first->second.identity == catalog_identity);
	}
	entry.first->second.modifications |= modification;
}

} // namespace duckdb
