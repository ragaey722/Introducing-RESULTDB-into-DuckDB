#include "catch.hpp"
#include "test_helpers.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/main/connection_manager.hpp"
#include "duckdb/main/materialized_query_result.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/planner/resultdb_reduced_plan.hpp"
#include "duckdb/optimizer/resultdb_plan_enumerator.hpp"
#include "duckdb/common/types/hash.hpp"

#include <algorithm>
#include <chrono>
#include <thread>

using namespace duckdb;

using ResultRow = vector<Value>;

class TestResultDBEnumerationCosts : public ResultDBEnumerationCostProvider {
public:
	explicit TestResultDBEnumerationCosts(vector<double> cardinalities_p)
	    : cardinalities(std::move(cardinalities_p)) {
	}

	double Cardinality(const vector<idx_t> &nodes) override {
		double result = 0;
		for (auto node : nodes) {
			result = std::max(result, cardinalities[node]);
		}
		return result;
	}

	double PayloadWidth(const vector<idx_t> &nodes) override {
		return static_cast<double>(std::max<idx_t>(nodes.size(), 1));
	}

	vector<double> cardinalities;
};

struct ResultRowHash {
	size_t operator()(const ResultRow &row) const {
		hash_t hash = 0;
		for (auto &value : row) {
			hash = CombineHash(hash, value.Hash());
		}
		return NumericCast<size_t>(hash);
	}
};

struct ResultRowEquality {
	bool operator()(const ResultRow &left, const ResultRow &right) const {
		if (left.size() != right.size()) {
			return false;
		}
		for (idx_t column_idx = 0; column_idx < left.size(); column_idx++) {
			if (!Value::DefaultValuesAreEqual(left[column_idx], right[column_idx])) {
				return false;
			}
		}
		return true;
	}
};

using ResultRowCounts = unordered_map<ResultRow, idx_t, ResultRowHash, ResultRowEquality>;

static ResultRowCounts ResultRows(QueryResult &result) {
	auto &materialized = result.Cast<MaterializedQueryResult>();
	ResultRowCounts rows;
	for (idx_t row_idx = 0; row_idx < materialized.RowCount(); row_idx++) {
		ResultRow row;
		row.reserve(result.ColumnCount());
		for (idx_t col_idx = 0; col_idx < result.ColumnCount(); col_idx++) {
			row.push_back(materialized.GetValue(col_idx, row_idx));
		}
		rows[std::move(row)]++;
	}
	return rows;
}

static bool ResultRowsEqual(const ResultRowCounts &left, const ResultRowCounts &right) {
	if (left.size() != right.size()) {
		return false;
	}
	for (auto &entry : left) {
		auto other = right.find(entry.first);
		if (other == right.end() || other->second != entry.second) {
			return false;
		}
	}
	return true;
}

static void RequireResultDBTable(QueryResult &actual, const string &table_name, QueryResult &expected) {
	REQUIRE(actual.properties.resultdb.enabled);
	REQUIRE(actual.properties.resultdb.tables.size() == 1);
	auto &table = actual.properties.resultdb.tables[0];
	REQUIRE(table.name == table_name);
	REQUIRE(table.columns.size() == actual.ColumnCount());
	for (idx_t column_idx = 0; column_idx < table.columns.size(); column_idx++) {
		REQUIRE(table.columns[column_idx].flat_column_index == column_idx);
		REQUIRE(table.columns[column_idx].name == actual.names[column_idx]);
		REQUIRE(table.columns[column_idx].type == actual.types[column_idx]);
	}
	REQUIRE(actual.names == expected.names);
	REQUIRE(actual.types == expected.types);
	REQUIRE(ResultRowsEqual(ResultRows(actual), ResultRows(expected)));
}

static void RequireResultDBStrategy(QueryResult &actual, ResultDBStrategy requested_strategy,
                                    ResultDBStrategy execution_strategy) {
	auto expected_execution = execution_strategy == ResultDBStrategy::SEMIJOIN
	                              ? ResultDBExecutionStrategy::SEMIJOIN
	                              : ResultDBExecutionStrategy::DECOMPOSE;
	auto current = &actual;
	while (current) {
		REQUIRE(current->properties.resultdb.enabled);
		REQUIRE(current->properties.resultdb.requested_strategy == requested_strategy);
		REQUIRE(current->properties.resultdb.execution_strategy == expected_execution);
		current = current->next.get();
	}
}

TEST_CASE("Test comment in CPP API", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);

	con.SendQuery("--ups");
	//! Should not crash
	REQUIRE(1);
}

TEST_CASE("Test StarExpression replace_list parameter", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);
	auto sql = "select * replace(i * $n as i) from range(1, 10) t(i)";
	auto stmts = con.ExtractStatements(sql);

	auto &select_stmt = stmts[0]->Cast<SelectStatement>();
	auto &select_node = select_stmt.node->Cast<SelectNode>();

	REQUIRE(select_node.select_list[0]->HasParameter());
}

TEST_CASE("Test using connection after database is gone", "[api]") {
	auto db = make_uniq<DuckDB>(nullptr);
	auto conn = make_uniq<Connection>(*db);
	// check that the connection works
	auto result = conn->Query("SELECT 42");
	REQUIRE(CHECK_COLUMN(result, 0, {42}));
	// destroy the database
	db.reset();
	// try to use the connection
	// it still works: the database remains until all connections are destroyed
	REQUIRE_NO_FAIL(conn->Query("SELECT 42"));

	// now try it with an open transaction
	db = make_uniq<DuckDB>(nullptr);
	conn = make_uniq<Connection>(*db);

	REQUIRE_NO_FAIL(conn->Query("BEGIN TRANSACTION"));
	result = conn->Query("SELECT 42");
	REQUIRE(CHECK_COLUMN(result, 0, {42}));

	db.reset();

	REQUIRE_NO_FAIL(conn->Query("SELECT 42"));
}

TEST_CASE("Test destroying connections with open transactions", "[api]") {
	auto db = make_uniq<DuckDB>(nullptr);
	{
		Connection con(*db);
		con.Query("BEGIN TRANSACTION");
		con.Query("CREATE TABLE test(i INTEGER);");
	}

	auto conn = make_uniq<Connection>(*db);
	REQUIRE_NO_FAIL(conn->Query("CREATE TABLE test(i INTEGER)"));
}

static void long_running_query(Connection *conn, bool *correct) {
	*correct = true;
	auto result = conn->Query("SELECT i1.i FROM integers i1, integers i2, integers i3, integers i4, integers i5, "
	                          "integers i6, integers i7, integers i8, integers i9, integers i10,"
	                          "integers i11, integers i12, integers i13");
	// the query should fail
	*correct = result->HasError();
}

TEST_CASE("Test closing database during long running query", "[api]") {
	auto db = make_uniq<DuckDB>(nullptr);
	auto conn = make_uniq<Connection>(*db);
	// create the database
	REQUIRE_NO_FAIL(conn->Query("CREATE TABLE integers(i INTEGER)"));
	REQUIRE_NO_FAIL(conn->Query("INSERT INTO integers FROM range(10000)"));
	conn->DisableProfiling();
	// perform a long running query in the background (many cross products)
	bool correct = true;
	auto background_thread = std::thread(long_running_query, conn.get(), &correct);
	// wait a little bit
	std::this_thread::sleep_for(std::chrono::milliseconds(100));
	// destroy the database
	conn->Interrupt();
	db.reset();
	// wait for the thread
	background_thread.join();
	REQUIRE(correct);
	// try to use the connection
	REQUIRE_NO_FAIL(conn->Query("SELECT 42"));
}

TEST_CASE("Test closing result after database is gone", "[api]") {
	auto db = make_uniq<DuckDB>(nullptr);
	auto conn = make_uniq<Connection>(*db);
	// check that the connection works
	auto result = conn->Query("SELECT 42");
	REQUIRE(CHECK_COLUMN(result, 0, {42}));
	// destroy the database
	db.reset();
	conn.reset();
	result.reset();

	// now the streaming result
	db = make_uniq<DuckDB>(nullptr);
	conn = make_uniq<Connection>(*db);
	// check that the connection works
	auto streaming_result = conn->SendQuery("SELECT 42");
	// destroy the database
	db.reset();
	conn.reset();
	REQUIRE(CHECK_COLUMN(streaming_result, 0, {42}));
	streaming_result.reset();
}

TEST_CASE("Test closing database with open prepared statements", "[api]") {
	auto db = make_uniq<DuckDB>(nullptr);
	auto conn = make_uniq<Connection>(*db);

	auto p1 = conn->Prepare("CREATE TABLE a (i INTEGER)");
	REQUIRE_NO_FAIL(p1->Execute());
	auto p2 = conn->Prepare("INSERT INTO a VALUES (42)");
	REQUIRE_NO_FAIL(p2->Execute());

	db.reset();
	conn.reset();

	// the prepared statements are still valid
	// the database is only destroyed when the prepared statements are destroyed
	REQUIRE_NO_FAIL(p2->Execute());
	p1.reset();
	p2.reset();
}

static void parallel_query(Connection *conn, bool *correct, size_t threadnr) {
	correct[threadnr] = true;
	for (size_t i = 0; i < 100; i++) {
		auto result = conn->Query("SELECT * FROM integers ORDER BY i");
		if (!CHECK_COLUMN(result, 0, {1, 2, 3, Value()})) {
			correct[threadnr] = false;
		}
	}
}

TEST_CASE("Test temp_directory defaults", "[api][.]") {
	const char *db_paths[] = {nullptr, "", ":memory:"};
	for (auto &path : db_paths) {
		auto db = make_uniq<DuckDB>(path);
		auto conn = make_uniq<Connection>(*db);

		REQUIRE(db->instance->config.options.temporary_directory == ".tmp");
	}
}

TEST_CASE("Test parallel usage of single client", "[api][.]") {
	auto db = make_uniq<DuckDB>(nullptr);
	auto conn = make_uniq<Connection>(*db);

	REQUIRE_NO_FAIL(conn->Query("CREATE TABLE integers(i INTEGER)"));
	REQUIRE_NO_FAIL(conn->Query("INSERT INTO integers VALUES (1), (2), (3), (NULL)"));

	bool correct[20];
	std::thread threads[20];
	for (size_t i = 0; i < 20; i++) {
		threads[i] = std::thread(parallel_query, conn.get(), correct, i);
	}
	for (size_t i = 0; i < 20; i++) {
		threads[i].join();
		REQUIRE(correct[i]);
	}
}

static void parallel_query_with_new_connection(DuckDB *db, bool *correct, size_t threadnr) {
	correct[threadnr] = true;
	for (size_t i = 0; i < 100; i++) {
		auto conn = make_uniq<Connection>(*db);
		auto result = conn->Query("SELECT * FROM integers ORDER BY i");
		if (!CHECK_COLUMN(result, 0, {1, 2, 3, Value()})) {
			correct[threadnr] = false;
		}
	}
}

TEST_CASE("Test making and dropping connections in parallel to a single database", "[api][.]") {
	auto db = make_uniq<DuckDB>(nullptr);
	auto conn = make_uniq<Connection>(*db);

	REQUIRE_NO_FAIL(conn->Query("CREATE TABLE integers(i INTEGER)"));
	REQUIRE_NO_FAIL(conn->Query("INSERT INTO integers VALUES (1), (2), (3), (NULL)"));

	bool correct[20];
	std::thread threads[20];
	for (size_t i = 0; i < 20; i++) {
		threads[i] = std::thread(parallel_query_with_new_connection, db.get(), correct, i);
	}
	for (size_t i = 0; i < 100; i++) {
		auto result = conn->Query("SELECT * FROM integers ORDER BY i");
		REQUIRE(CHECK_COLUMN(result, 0, {1, 2, 3, Value()}));
	}
	for (size_t i = 0; i < 20; i++) {
		threads[i].join();
		REQUIRE(correct[i]);
	}
	auto result = conn->Query("SELECT * FROM integers ORDER BY i");
	REQUIRE(CHECK_COLUMN(result, 0, {1, 2, 3, Value()}));
}

TEST_CASE("Test multiple result sets", "[api]") {
	duckdb::unique_ptr<QueryResult> result;
	DuckDB db(nullptr);
	Connection con(db);

	con.ForceParallelism();

	result = con.Query("SELECT 42; SELECT 84");
	REQUIRE(CHECK_COLUMN(result, 0, {42}));
	result = std::move(result->next);
	REQUIRE(CHECK_COLUMN(result, 0, {84}));
	REQUIRE(!result->next);

	// also with stream api
	result = con.SendQuery("SELECT 42; SELECT 84");
	REQUIRE(CHECK_COLUMN(result, 0, {42}));
	result = std::move(result->next);
	REQUIRE(CHECK_COLUMN(result, 0, {84}));
	REQUIRE(!result->next);
}

TEST_CASE("Test ResultDB query returns source relations", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);

	REQUIRE_NO_FAIL(con.Query("CREATE TABLE customers(id INTEGER, name VARCHAR)"));
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE orders(id INTEGER, customer_id INTEGER, amount INTEGER)"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO customers VALUES (1, 'Ada'), (2, 'Linus')"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO orders VALUES (10, 1, 100), (11, 1, 200), (12, 2, 50)"));

	duckdb::unique_ptr<QueryResult> result =
	    con.Query("SELECT RESULTDB * FROM customers c JOIN orders o ON c.id = o.customer_id WHERE o.amount > 150");
	REQUIRE(!result->HasError());
	REQUIRE(result->properties.resultdb.enabled);
	REQUIRE(result->properties.resultdb.tables.size() == 1);
	REQUIRE(result->properties.resultdb.tables[0].name == "c");
	REQUIRE(result->names == vector<string> {"id", "name"});
	REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 1);
	REQUIRE(CHECK_COLUMN(result, 0, {1}));
	REQUIRE(CHECK_COLUMN(result, 1, {"Ada"}));

	result = std::move(result->next);
	REQUIRE(result);
	REQUIRE(result->properties.resultdb.enabled);
	REQUIRE(result->properties.resultdb.tables.size() == 1);
	REQUIRE(result->properties.resultdb.tables[0].name == "o");
	REQUIRE(result->names == vector<string> {"id", "customer_id", "amount"});
	REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 1);
	REQUIRE(CHECK_COLUMN(result, 0, {11}));
	REQUIRE(CHECK_COLUMN(result, 1, {1}));
	REQUIRE(CHECK_COLUMN(result, 2, {200}));
	REQUIRE(!result->next);

	result = con.Query("SELECT RESULTDB c.name, o.amount FROM customers c JOIN orders o ON c.id = o.customer_id "
	                   "WHERE o.amount > 150");
	REQUIRE(!result->HasError());
	REQUIRE(result->properties.resultdb.enabled);
	REQUIRE(result->properties.resultdb.tables.size() == 1);
	REQUIRE(result->properties.resultdb.tables[0].name == "c");
	REQUIRE(result->names == vector<string> {"name"});
	REQUIRE(CHECK_COLUMN(result, 0, {"Ada"}));
	result = std::move(result->next);
	REQUIRE(result);
	REQUIRE(result->properties.resultdb.tables.size() == 1);
	REQUIRE(result->properties.resultdb.tables[0].name == "o");
	REQUIRE(result->names == vector<string> {"amount"});
	REQUIRE(CHECK_COLUMN(result, 0, {200}));
	REQUIRE(!result->next);

	result = con.Query("SELECT RESULTDB c.name, c.id FROM customers c WHERE c.id = 1");
	REQUIRE(!result->HasError());
	REQUIRE(result->properties.resultdb.enabled);
	REQUIRE(result->properties.resultdb.tables.size() == 1);
	REQUIRE(result->properties.resultdb.tables[0].name == "c");
	REQUIRE(result->names == vector<string> {"name", "id"});
	REQUIRE(CHECK_COLUMN(result, 0, {"Ada"}));
	REQUIRE(CHECK_COLUMN(result, 1, {1}));
	REQUIRE(!result->next);

	result = con.Query("SELECT RESULTDB o.* FROM customers c JOIN orders o ON c.id = o.customer_id "
	                   "WHERE o.amount > 150");
	REQUIRE(!result->HasError());
	REQUIRE(result->properties.resultdb.tables.size() == 1);
	REQUIRE(result->properties.resultdb.tables[0].name == "o");
	REQUIRE(result->names == vector<string> {"id", "customer_id", "amount"});
	REQUIRE(CHECK_COLUMN(result, 0, {11}));
	REQUIRE(!result->next);

	REQUIRE_NO_FAIL(con.Query("CREATE TABLE shipments(id INTEGER, order_id INTEGER, carrier VARCHAR)"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO customers VALUES (3, 'Grace')"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO orders VALUES (13, 3, 300), (14, 2, 75)"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO shipments VALUES (100, 11, 'DHL'), (101, 13, 'UPS'), (102, 14, 'DHL')"));

	const string three_table_from =
	    " FROM customers c "
	    "JOIN orders o ON c.id = o.customer_id "
	    "JOIN shipments s ON o.id = s.order_id "
	    "WHERE o.amount > 150";

	result = con.Query("SELECT RESULTDB c.name, o.amount, s.carrier" + three_table_from);
	REQUIRE(!result->HasError());
	REQUIRE(result->properties.resultdb.enabled);
	REQUIRE(result->properties.resultdb.tables.size() == 1);
	REQUIRE(result->properties.resultdb.tables[0].name == "c");
	REQUIRE(result->names == vector<string> {"name"});
	REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 2);
	REQUIRE(CHECK_COLUMN(result, 0, {"Ada", "Grace"}));

	result = std::move(result->next);
	REQUIRE(result);
	REQUIRE(result->properties.resultdb.enabled);
	REQUIRE(result->properties.resultdb.tables.size() == 1);
	REQUIRE(result->properties.resultdb.tables[0].name == "o");
	REQUIRE(result->names == vector<string> {"amount"});
	REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 2);
	REQUIRE(CHECK_COLUMN(result, 0, {200, 300}));

	result = std::move(result->next);
	REQUIRE(result);
	REQUIRE(result->properties.resultdb.enabled);
	REQUIRE(result->properties.resultdb.tables.size() == 1);
	REQUIRE(result->properties.resultdb.tables[0].name == "s");
	REQUIRE(result->names == vector<string> {"carrier"});
	REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 2);
	REQUIRE(CHECK_COLUMN(result, 0, {"DHL", "UPS"}));
	REQUIRE(!result->next);

	result = con.Query("SELECT \"resultdb\" FROM (SELECT 42 AS \"resultdb\") t");
	REQUIRE(!result->HasError());
	REQUIRE(CHECK_COLUMN(result, 0, {42}));
	auto reserved_keyword_result = con.Query("SELECT resultdb FROM (SELECT 42 AS \"resultdb\") t");
	REQUIRE(reserved_keyword_result->HasError());

	auto unsupported_outer_join =
	    con.Query("SELECT RESULTDB * FROM customers c LEFT JOIN orders o ON c.id = o.customer_id");
	REQUIRE(unsupported_outer_join->HasError());
}

TEST_CASE("Test ResultDB strategy setting", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);

	REQUIRE_NO_FAIL(con.Query("CREATE TABLE customers(id INTEGER, name VARCHAR)"));
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE orders(id INTEGER, customer_id INTEGER, amount INTEGER)"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO customers VALUES (1, 'Ada'), (2, 'Linus')"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO orders VALUES (10, 1, 100), (11, 1, 200), (12, 2, 50)"));

	auto setting_result = con.Query("SELECT current_setting('resultdb_strategy')");
	REQUIRE(!setting_result->HasError());
	REQUIRE(CHECK_COLUMN(setting_result, 0, {"decompose"}));

	auto invalid_strategy = con.Query("SET resultdb_strategy = 'invalid'");
	REQUIRE(invalid_strategy->HasError());

	struct StrategyCase {
		string input;
		string stored_value;
		ResultDBStrategy requested_strategy;
		ResultDBStrategy execution_strategy;
	};
	vector<StrategyCase> strategy_cases = {
	    {"decompose", "decompose", ResultDBStrategy::DECOMPOSE, ResultDBStrategy::DECOMPOSE},
	    {"SEMIJOIN", "semijoin", ResultDBStrategy::SEMIJOIN, ResultDBStrategy::SEMIJOIN},
	    {"tdroot", "tdroot", ResultDBStrategy::TDROOT, ResultDBStrategy::SEMIJOIN},
	    {"tdfold-no-tvc", "tdfold_no_tvc", ResultDBStrategy::TDFOLD_NO_TVC, ResultDBStrategy::SEMIJOIN},
	    {"tdfold", "tdfold", ResultDBStrategy::TDFOLD, ResultDBStrategy::SEMIJOIN},
	    {"auto", "auto", ResultDBStrategy::AUTO, ResultDBStrategy::SEMIJOIN},
	    {"tdresultdb", "auto", ResultDBStrategy::AUTO, ResultDBStrategy::SEMIJOIN},
	};

	const string from_clause = " FROM customers c JOIN orders o ON c.id = o.customer_id WHERE o.amount >= 100";
	for (auto &strategy_case : strategy_cases) {
		REQUIRE_NO_FAIL(con.Query("SET resultdb_strategy = '" + strategy_case.input + "'"));
		setting_result = con.Query("SELECT current_setting('resultdb_strategy')");
		REQUIRE(!setting_result->HasError());
		REQUIRE(CHECK_COLUMN(setting_result, 0, {strategy_case.stored_value}));

		duckdb::unique_ptr<QueryResult> result = con.Query("SELECT RESULTDB *" + from_clause);
		INFO(result->GetError());
		REQUIRE(!result->HasError());
		RequireResultDBStrategy(*result, strategy_case.requested_strategy, strategy_case.execution_strategy);
		if (strategy_case.execution_strategy == ResultDBStrategy::SEMIJOIN) {
			REQUIRE(result->properties.resultdb.join_edges.size() == 1);
		} else {
			REQUIRE(result->properties.resultdb.join_edges.empty());
		}

		auto expected_customers = con.Query("SELECT DISTINCT c.id, c.name" + from_clause);
		auto expected_orders = con.Query("SELECT DISTINCT o.id, o.customer_id, o.amount" + from_clause);

		RequireResultDBTable(*result, "c", *expected_customers);
		result = std::move(result->next);
		REQUIRE(result);
		RequireResultDBTable(*result, "o", *expected_orders);
		REQUIRE(!result->next);
	}

	REQUIRE_NO_FAIL(con.Query("SET resultdb_strategy = 'semijoin'"));
	duckdb::unique_ptr<QueryResult> reordered_result =
	    con.Query("SELECT RESULTDB o.amount, c.name, o.id" + from_clause);
	REQUIRE(!reordered_result->HasError());
	RequireResultDBStrategy(*reordered_result, ResultDBStrategy::SEMIJOIN, ResultDBStrategy::SEMIJOIN);
	auto expected_reordered_orders = con.Query("SELECT DISTINCT o.amount, o.id" + from_clause);
	auto expected_reordered_customers = con.Query("SELECT DISTINCT c.name" + from_clause);
	RequireResultDBTable(*reordered_result, "o", *expected_reordered_orders);
	reordered_result = std::move(reordered_result->next);
	REQUIRE(reordered_result);
	RequireResultDBTable(*reordered_result, "c", *expected_reordered_customers);
	REQUIRE(!reordered_result->next);
}

TEST_CASE("Test ResultDB semijoin deduplicates duplicate source rows", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);

	REQUIRE_NO_FAIL(con.Query("CREATE TABLE customers(id INTEGER, name VARCHAR)"));
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE orders(id INTEGER, customer_id INTEGER)"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO customers VALUES (1, 'Ada'), (1, 'Ada'), (2, 'Linus')"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO orders VALUES (10, 1), (20, 2)"));

	const string from_clause = " FROM customers c JOIN orders o ON c.id = o.customer_id WHERE o.id = 10";
	auto expected_customers = con.Query("SELECT DISTINCT c.id, c.name" + from_clause);
	auto expected_orders = con.Query("SELECT DISTINCT o.id, o.customer_id" + from_clause);
	REQUIRE(!expected_customers->HasError());
	REQUIRE(!expected_orders->HasError());

	struct StrategyCase {
		string setting;
		ResultDBStrategy requested_strategy;
		ResultDBStrategy execution_strategy;
	};
	vector<StrategyCase> strategy_cases = {
	    {"decompose", ResultDBStrategy::DECOMPOSE, ResultDBStrategy::DECOMPOSE},
	    {"semijoin", ResultDBStrategy::SEMIJOIN, ResultDBStrategy::SEMIJOIN},
	};

	for (auto &strategy_case : strategy_cases) {
		REQUIRE_NO_FAIL(con.Query("SET resultdb_strategy = '" + strategy_case.setting + "'"));
		duckdb::unique_ptr<QueryResult> result = con.Query("SELECT RESULTDB *" + from_clause);
		REQUIRE(!result->HasError());
		RequireResultDBStrategy(*result, strategy_case.requested_strategy, strategy_case.execution_strategy);

		RequireResultDBTable(*result, "c", *expected_customers);
		REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 1);
		REQUIRE(CHECK_COLUMN(result, 0, {1}));
		REQUIRE(CHECK_COLUMN(result, 1, {"Ada"}));

		result = std::move(result->next);
		REQUIRE(result);
		RequireResultDBTable(*result, "o", *expected_orders);
		REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 1);
		REQUIRE(CHECK_COLUMN(result, 0, {10}));
		REQUIRE(CHECK_COLUMN(result, 1, {1}));
		REQUIRE(!result->next);
	}
}

TEST_CASE("Test ResultDB decompose deduplicates duplicate rows with NULL values", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);

	REQUIRE_NO_FAIL(con.Query("CREATE TABLE customers(id INTEGER, name VARCHAR)"));
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE orders(id INTEGER, customer_id INTEGER, note VARCHAR)"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO customers VALUES (1, NULL), (1, NULL), (2, 'Linus')"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO orders VALUES (10, 1, NULL), (10, 1, NULL), (20, 2, 'other')"));

	const string from_clause = " FROM customers c JOIN orders o ON c.id = o.customer_id WHERE o.id = 10";
	auto expected_customers = con.Query("SELECT DISTINCT c.id, c.name" + from_clause);
	auto expected_orders = con.Query("SELECT DISTINCT o.id, o.customer_id, o.note" + from_clause);
	REQUIRE(!expected_customers->HasError());
	REQUIRE(!expected_orders->HasError());

	REQUIRE_NO_FAIL(con.Query("SET resultdb_strategy = 'decompose'"));
	duckdb::unique_ptr<QueryResult> result = con.Query("SELECT RESULTDB *" + from_clause);
	REQUIRE(!result->HasError());
	RequireResultDBStrategy(*result, ResultDBStrategy::DECOMPOSE, ResultDBStrategy::DECOMPOSE);

	RequireResultDBTable(*result, "c", *expected_customers);
	REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 1);
	REQUIRE(CHECK_COLUMN(result, 0, {1}));
	REQUIRE(CHECK_COLUMN(result, 1, {Value()}));

	result = std::move(result->next);
	REQUIRE(result);
	RequireResultDBTable(*result, "o", *expected_orders);
	REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 1);
	REQUIRE(CHECK_COLUMN(result, 0, {10}));
	REQUIRE(CHECK_COLUMN(result, 1, {1}));
	REQUIRE(CHECK_COLUMN(result, 2, {Value()}));
	REQUIRE(!result->next);
}

TEST_CASE("Test ResultDB semijoin physical executor handles NULL keys and parallel base scans", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);

	REQUIRE_NO_FAIL(con.Query("CREATE TABLE a(id INTEGER, label VARCHAR)"));
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE b(id INTEGER, a_id INTEGER)"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO a VALUES (1, 'keep'), (NULL, 'null-key'), (2, 'drop')"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO b VALUES (10, 1), (20, NULL)"));

	const string null_key_from = " FROM a JOIN b ON a.id = b.a_id";
	auto expected_a = con.Query("SELECT DISTINCT a.id, a.label" + null_key_from);
	auto expected_b = con.Query("SELECT DISTINCT b.id, b.a_id" + null_key_from);
	REQUIRE(!expected_a->HasError());
	REQUIRE(!expected_b->HasError());

	REQUIRE_NO_FAIL(con.Query("SET resultdb_strategy = 'semijoin'"));
	duckdb::unique_ptr<QueryResult> result = con.Query("SELECT RESULTDB *" + null_key_from);
	REQUIRE(!result->HasError());
	RequireResultDBStrategy(*result, ResultDBStrategy::SEMIJOIN, ResultDBStrategy::SEMIJOIN);
	RequireResultDBTable(*result, "a", *expected_a);
	REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 1);
	result = std::move(result->next);
	REQUIRE(result);
	RequireResultDBTable(*result, "b", *expected_b);
	REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 1);
	REQUIRE(!result->next);

	REQUIRE_NO_FAIL(con.Query("CREATE TABLE ck_a(k1 INTEGER, k2 INTEGER, label VARCHAR)"));
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE ck_b(k1 INTEGER, k2 INTEGER, label VARCHAR)"));
	REQUIRE_NO_FAIL(
	    con.Query("INSERT INTO ck_a VALUES (1, 1, 'keep'), (1, 2, 'partial'), (2, 1, 'drop'), (NULL, 1, 'null-a')"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO ck_b VALUES (1, 1, 'match'), (1, 3, 'partial'), (NULL, 1, 'null-b')"));

	const string composite_key_from = " FROM ck_a JOIN ck_b ON ck_a.k1 = ck_b.k1 AND ck_a.k2 = ck_b.k2";
	auto expected_ck_a = con.Query("SELECT DISTINCT ck_a.k1, ck_a.k2, ck_a.label" + composite_key_from);
	auto expected_ck_b = con.Query("SELECT DISTINCT ck_b.k1, ck_b.k2, ck_b.label" + composite_key_from);
	REQUIRE(!expected_ck_a->HasError());
	REQUIRE(!expected_ck_b->HasError());

	result = con.Query("SELECT RESULTDB *" + composite_key_from);
	REQUIRE(!result->HasError());
	RequireResultDBStrategy(*result, ResultDBStrategy::SEMIJOIN, ResultDBStrategy::SEMIJOIN);
	RequireResultDBTable(*result, "ck_a", *expected_ck_a);
	REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 1);
	result = std::move(result->next);
	REQUIRE(result);
	RequireResultDBTable(*result, "ck_b", *expected_ck_b);
	REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 1);
	REQUIRE(!result->next);

	REQUIRE_NO_FAIL(con.Query("PRAGMA threads=4"));
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE pa AS SELECT i::INTEGER id FROM range(0, 1000) tbl(i)"));
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE pb AS SELECT i::INTEGER id, i::INTEGER a_id "
	                          "FROM range(0, 1000) tbl(i) WHERE i % 2 = 0"));
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE pc AS SELECT i::INTEGER id, i::INTEGER b_id "
	                          "FROM range(0, 1000) tbl(i) WHERE i % 4 = 0"));

	const string parallel_from =
	    " FROM pa JOIN pb ON pa.id = pb.a_id JOIN pc ON pb.id = pc.b_id";
	auto expected_pa = con.Query("SELECT DISTINCT pa.id" + parallel_from);
	auto expected_pb = con.Query("SELECT DISTINCT pb.id, pb.a_id" + parallel_from);
	auto expected_pc = con.Query("SELECT DISTINCT pc.id, pc.b_id" + parallel_from);
	REQUIRE(!expected_pa->HasError());
	REQUIRE(!expected_pb->HasError());
	REQUIRE(!expected_pc->HasError());

	result = con.Query("SELECT RESULTDB *" + parallel_from);
	REQUIRE(!result->HasError());
	RequireResultDBStrategy(*result, ResultDBStrategy::SEMIJOIN, ResultDBStrategy::SEMIJOIN);
	RequireResultDBTable(*result, "pa", *expected_pa);
	REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 250);
	result = std::move(result->next);
	REQUIRE(result);
	RequireResultDBTable(*result, "pb", *expected_pb);
	REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 250);
	result = std::move(result->next);
	REQUIRE(result);
	RequireResultDBTable(*result, "pc", *expected_pc);
	REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 250);
	REQUIRE(!result->next);
}

TEST_CASE("Test ResultDB semijoin reuses fused composite-key phases at T1 and T6", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);

	REQUIRE_NO_FAIL(con.Query("CREATE TABLE fused_a(k1 INTEGER, k2 INTEGER, payload VARCHAR)"));
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE fused_b(a_k1 INTEGER, a_k2 INTEGER, c_k1 INTEGER, c_k2 INTEGER, "
	                          "payload BIGINT)"));
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE fused_c(k1 INTEGER, k2 INTEGER, payload BOOLEAN)"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO fused_a VALUES "
	                          "(1, 11, 'a-keep'), (1, 11, 'a-keep'), "
	                          "(2, 22, 'a-no-c'), (3, 33, NULL), (5, 55, NULL), "
	                          "(4, NULL, 'a-null-key'), (NULL, 55, 'a-null-key')"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO fused_b VALUES "
	                          "(1, 11, 101, 1001, 1000), (1, 11, 101, 1001, 1000), "
	                          "(2, 22, 202, 2002, 2000), "
	                          "(3, 33, 303, 3003, NULL), (5, 55, 505, 5005, NULL), "
	                          "(4, NULL, 404, 4004, 4000), (NULL, 55, 505, 5005, 5000)"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO fused_c VALUES "
	                          "(101, 1001, TRUE), (101, 1001, TRUE), "
	                          "(202, 9999, FALSE), (303, 3003, NULL), (505, 5005, NULL), "
	                          "(NULL, 5005, TRUE)"));

	const string from_clause =
	    " FROM fused_a a "
	    "JOIN fused_b b ON a.k1 = b.a_k1 AND a.k2 = b.a_k2 "
	    "JOIN fused_c c ON b.c_k1 = c.k1 AND b.c_k2 = c.k2";
	auto expected_a = con.Query("SELECT DISTINCT a.payload" + from_clause);
	auto expected_b = con.Query("SELECT DISTINCT b.payload" + from_clause);
	auto expected_c = con.Query("SELECT DISTINCT c.payload" + from_clause);
	REQUIRE(!expected_a->HasError());
	REQUIRE(!expected_b->HasError());
	REQUIRE(!expected_c->HasError());
	REQUIRE(expected_a->Cast<MaterializedQueryResult>().RowCount() == 2);
	REQUIRE(expected_b->Cast<MaterializedQueryResult>().RowCount() == 2);
	REQUIRE(expected_c->Cast<MaterializedQueryResult>().RowCount() == 2);

	const string resultdb_query = "SELECT RESULTDB a.payload, b.payload, c.payload" + from_clause;
	for (auto &strategy : vector<std::pair<string, ResultDBStrategy>> {
	         {"semijoin", ResultDBStrategy::SEMIJOIN},
	         {"tdroot", ResultDBStrategy::TDROOT},
	         {"tdfold_no_tvc", ResultDBStrategy::TDFOLD_NO_TVC},
	         {"tdfold", ResultDBStrategy::TDFOLD}}) {
		REQUIRE_NO_FAIL(con.Query("SET resultdb_strategy = '" + strategy.first + "'"));
		for (auto &threads : vector<string> {"1", "6"}) {
			REQUIRE_NO_FAIL(con.Query("PRAGMA threads=" + threads));
			auto prepared = con.Prepare(resultdb_query);
			REQUIRE(!prepared->HasError());

			for (idx_t repetition = 0; repetition < 4; repetition++) {
				duckdb::unique_ptr<QueryResult> result = prepared->Execute();
				REQUIRE(!result->HasError());
				RequireResultDBStrategy(*result, strategy.second, ResultDBStrategy::SEMIJOIN);

				RequireResultDBTable(*result, "a", *expected_a);
				REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 2);
				result = std::move(result->next);
				REQUIRE(result);
				RequireResultDBTable(*result, "b", *expected_b);
				REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 2);
				result = std::move(result->next);
				REQUIRE(result);
				RequireResultDBTable(*result, "c", *expected_c);
				REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 2);
				REQUIRE(!result->next);
			}
		}
	}
}

TEST_CASE("Test ResultDB semijoin materializes casted equality join keys", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);

	REQUIRE_NO_FAIL(con.Query("CREATE TABLE cast_left(id INTEGER, label VARCHAR)"));
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE cast_right(id INTEGER, left_id DOUBLE)"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO cast_left VALUES (1, 'keep'), (2, 'drop')"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO cast_right VALUES (10, 1.0), (20, 2.5), (30, NULL)"));

	const string cast_key_from = " FROM cast_left l JOIN cast_right r ON l.id = r.left_id";
	auto expected_left = con.Query("SELECT DISTINCT l.label" + cast_key_from);
	auto expected_right = con.Query("SELECT DISTINCT r.id" + cast_key_from);
	REQUIRE(!expected_left->HasError());
	REQUIRE(!expected_right->HasError());

	REQUIRE_NO_FAIL(con.Query("SET resultdb_strategy = 'semijoin'"));
	duckdb::unique_ptr<QueryResult> result = con.Query("SELECT RESULTDB l.label, r.id" + cast_key_from);
	REQUIRE(!result->HasError());
	RequireResultDBTable(*result, "l", *expected_left);
	REQUIRE(CHECK_COLUMN(result, 0, {"keep"}));
	result = std::move(result->next);
	REQUIRE(result);
	RequireResultDBTable(*result, "r", *expected_right);
	REQUIRE(CHECK_COLUMN(result, 0, {10}));
	REQUIRE(!result->next);
}

TEST_CASE("Test ResultDB semijoin strategy support", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);

	REQUIRE_NO_FAIL(con.Query("CREATE TABLE a(id INTEGER)"));
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE b(id INTEGER, a_id INTEGER)"));
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE c(id INTEGER, b_id INTEGER, a_id INTEGER, note VARCHAR)"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO a VALUES (1), (2)"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO b VALUES (10, 1), (20, 2)"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO c VALUES (100, 10, 1, NULL), (100, 10, 1, NULL), (200, 20, 2, 'x')"));

	const string cyclic_from =
	    " FROM a "
	    "JOIN b ON a.id = b.a_id "
	    "JOIN c ON b.id = c.b_id AND c.a_id = a.id";
	auto expected_cyclic_a = con.Query("SELECT DISTINCT a.id" + cyclic_from);
	auto expected_cyclic_b = con.Query("SELECT DISTINCT b.id, b.a_id" + cyclic_from);
	auto expected_cyclic_c = con.Query("SELECT DISTINCT c.id, c.b_id, c.a_id, c.note" + cyclic_from);
	REQUIRE(!expected_cyclic_a->HasError());
	REQUIRE(!expected_cyclic_b->HasError());
	REQUIRE(!expected_cyclic_c->HasError());

	REQUIRE_NO_FAIL(con.Query("SET resultdb_strategy = 'semijoin'"));
	duckdb::unique_ptr<QueryResult> result = con.Query("SELECT RESULTDB *" + cyclic_from);
	REQUIRE(!result->HasError());
	RequireResultDBStrategy(*result, ResultDBStrategy::SEMIJOIN, ResultDBStrategy::SEMIJOIN);
	REQUIRE(result->properties.resultdb.join_edges.size() == 3);
	RequireResultDBTable(*result, "a", *expected_cyclic_a);
	result = std::move(result->next);
	REQUIRE(result);
	RequireResultDBTable(*result, "b", *expected_cyclic_b);
	result = std::move(result->next);
	REQUIRE(result);
	RequireResultDBTable(*result, "c", *expected_cyclic_c);
	REQUIRE(!result->next);

	struct EnumeratedStrategyCase {
		string setting;
		ResultDBStrategy requested;
	};
	for (auto &strategy : vector<EnumeratedStrategyCase> {
	         {"tdroot", ResultDBStrategy::TDROOT},
	         {"tdfold_no_tvc", ResultDBStrategy::TDFOLD_NO_TVC},
	         {"tdfold", ResultDBStrategy::TDFOLD}}) {
		REQUIRE_NO_FAIL(con.Query("SET resultdb_strategy = '" + strategy.setting + "'"));
		result = con.Query("SELECT RESULTDB *" + cyclic_from);
		INFO(result->GetError());
		REQUIRE(!result->HasError());
		RequireResultDBStrategy(*result, strategy.requested, ResultDBStrategy::SEMIJOIN);
		REQUIRE(result->properties.resultdb.selected_root_table_index != DConstants::INVALID_INDEX);
		REQUIRE(result->properties.resultdb.enumeration_time_ms >= 0);
		REQUIRE(result->properties.resultdb.join_edges.size() == 3);
		RequireResultDBTable(*result, "a", *expected_cyclic_a);
		result = std::move(result->next);
		REQUIRE(result);
		RequireResultDBTable(*result, "b", *expected_cyclic_b);
		result = std::move(result->next);
		REQUIRE(result);
		RequireResultDBTable(*result, "c", *expected_cyclic_c);
		REQUIRE(!result->next);
	}

	REQUIRE_NO_FAIL(con.Query("SET resultdb_strategy = 'auto'"));
	result = con.Query("SELECT RESULTDB *" + cyclic_from);
	REQUIRE(!result->HasError());
	RequireResultDBStrategy(*result, ResultDBStrategy::AUTO, ResultDBStrategy::SEMIJOIN);
	REQUIRE(result->properties.resultdb.join_edges.size() == 3);
	RequireResultDBTable(*result, "a", *expected_cyclic_a);
	result = std::move(result->next);
	REQUIRE(result);
	RequireResultDBTable(*result, "b", *expected_cyclic_b);
	result = std::move(result->next);
	REQUIRE(result);
	RequireResultDBTable(*result, "c", *expected_cyclic_c);
	REQUIRE(!result->next);

	const string non_equality_from = " FROM a JOIN b ON a.id < b.a_id";
	REQUIRE_NO_FAIL(con.Query("SET resultdb_strategy = 'semijoin'"));
	result = con.Query("SELECT RESULTDB *" + non_equality_from);
	REQUIRE(result->HasError());
	for (auto &explicit_strategy : vector<string> {"tdroot", "tdfold_no_tvc", "tdfold"}) {
		REQUIRE_NO_FAIL(con.Query("SET resultdb_strategy = '" + explicit_strategy + "'"));
		result = con.Query("SELECT RESULTDB *" + non_equality_from);
		REQUIRE(result->HasError());
	}

	REQUIRE_NO_FAIL(con.Query("SET resultdb_strategy = 'auto'"));
	result = con.Query("SELECT RESULTDB *" + non_equality_from);
	REQUIRE(!result->HasError());
	RequireResultDBStrategy(*result, ResultDBStrategy::AUTO, ResultDBStrategy::DECOMPOSE);

	const string cross_table_where_from = " FROM a JOIN b ON a.id = b.a_id WHERE a.id = b.a_id";
	REQUIRE_NO_FAIL(con.Query("SET resultdb_strategy = 'semijoin'"));
	result = con.Query("SELECT RESULTDB *" + cross_table_where_from);
	REQUIRE(!result->HasError());
	RequireResultDBStrategy(*result, ResultDBStrategy::SEMIJOIN, ResultDBStrategy::SEMIJOIN);

	REQUIRE_NO_FAIL(con.Query("SET resultdb_strategy = 'auto'"));
	result = con.Query("SELECT RESULTDB *" + cross_table_where_from);
	REQUIRE(!result->HasError());
	RequireResultDBStrategy(*result, ResultDBStrategy::AUTO, ResultDBStrategy::SEMIJOIN);

	const string cross_product_where_from = " FROM a, b WHERE a.id = b.a_id";
	auto expected_cross_product_a = con.Query("SELECT DISTINCT a.id" + cross_product_where_from);
	auto expected_cross_product_b = con.Query("SELECT DISTINCT b.id, b.a_id" + cross_product_where_from);
	REQUIRE(!expected_cross_product_a->HasError());
	REQUIRE(!expected_cross_product_b->HasError());
	REQUIRE_NO_FAIL(con.Query("SET resultdb_strategy = 'semijoin'"));
	result = con.Query("SELECT RESULTDB *" + cross_product_where_from);
	REQUIRE(!result->HasError());
	RequireResultDBStrategy(*result, ResultDBStrategy::SEMIJOIN, ResultDBStrategy::SEMIJOIN);
	RequireResultDBTable(*result, "a", *expected_cross_product_a);
	result = std::move(result->next);
	REQUIRE(result);
	RequireResultDBTable(*result, "b", *expected_cross_product_b);
	REQUIRE(!result->next);

	const string selective_filter_from = " FROM a JOIN b ON a.id = b.a_id WHERE a.id = 1";
	auto expected_selective_a = con.Query("SELECT DISTINCT a.id" + selective_filter_from);
	auto expected_selective_b = con.Query("SELECT DISTINCT b.id, b.a_id" + selective_filter_from);
	REQUIRE(!expected_selective_a->HasError());
	REQUIRE(!expected_selective_b->HasError());
	result = con.Query("SELECT RESULTDB *" + selective_filter_from);
	REQUIRE(!result->HasError());
	RequireResultDBStrategy(*result, ResultDBStrategy::SEMIJOIN, ResultDBStrategy::SEMIJOIN);
	REQUIRE(result->properties.resultdb.join_edges.size() == 1);
	RequireResultDBTable(*result, "a", *expected_selective_a);
	result = std::move(result->next);
	REQUIRE(result);
	RequireResultDBTable(*result, "b", *expected_selective_b);
	REQUIRE(!result->next);

	const string transitive_filter_from =
	    " FROM a JOIN b ON a.id = b.a_id JOIN c ON b.a_id = c.a_id WHERE a.id = 1";
	auto expected_transitive_a = con.Query("SELECT DISTINCT a.id" + transitive_filter_from);
	auto expected_transitive_b = con.Query("SELECT DISTINCT b.id, b.a_id" + transitive_filter_from);
	auto expected_transitive_c = con.Query("SELECT DISTINCT c.id, c.b_id, c.a_id, c.note" + transitive_filter_from);
	REQUIRE(!expected_transitive_a->HasError());
	REQUIRE(!expected_transitive_b->HasError());
	REQUIRE(!expected_transitive_c->HasError());
	result = con.Query("SELECT RESULTDB *" + transitive_filter_from);
	REQUIRE(!result->HasError());
	RequireResultDBStrategy(*result, ResultDBStrategy::SEMIJOIN, ResultDBStrategy::SEMIJOIN);
	REQUIRE(result->properties.resultdb.join_edges.size() == 2);
	RequireResultDBTable(*result, "a", *expected_transitive_a);
	result = std::move(result->next);
	REQUIRE(result);
	RequireResultDBTable(*result, "b", *expected_transitive_b);
	result = std::move(result->next);
	REQUIRE(result);
	RequireResultDBTable(*result, "c", *expected_transitive_c);
	REQUIRE(!result->next);

	const string redundant_equality_from =
	    " FROM a JOIN b ON a.id = b.a_id JOIN c ON b.a_id = c.a_id AND a.id = c.a_id";
	auto expected_redundant_a = con.Query("SELECT DISTINCT a.id" + redundant_equality_from);
	auto expected_redundant_b = con.Query("SELECT DISTINCT b.id, b.a_id" + redundant_equality_from);
	auto expected_redundant_c = con.Query("SELECT DISTINCT c.id, c.b_id, c.a_id, c.note" + redundant_equality_from);
	REQUIRE(!expected_redundant_a->HasError());
	REQUIRE(!expected_redundant_b->HasError());
	REQUIRE(!expected_redundant_c->HasError());
	result = con.Query("SELECT RESULTDB *" + redundant_equality_from);
	REQUIRE(!result->HasError());
	RequireResultDBStrategy(*result, ResultDBStrategy::SEMIJOIN, ResultDBStrategy::SEMIJOIN);
	REQUIRE(result->properties.resultdb.join_edges.size() == 2);
	RequireResultDBTable(*result, "a", *expected_redundant_a);
	result = std::move(result->next);
	REQUIRE(result);
	RequireResultDBTable(*result, "b", *expected_redundant_b);
	result = std::move(result->next);
	REQUIRE(result);
	RequireResultDBTable(*result, "c", *expected_redundant_c);
	REQUIRE(!result->next);

	const string cross_table_non_equality_where_from = " FROM a JOIN b ON a.id = b.a_id WHERE a.id < b.a_id";
	REQUIRE_NO_FAIL(con.Query("SET resultdb_strategy = 'semijoin'"));
	result = con.Query("SELECT RESULTDB *" + cross_table_non_equality_where_from);
	REQUIRE(result->HasError());

	REQUIRE_NO_FAIL(con.Query("SET resultdb_strategy = 'auto'"));
	result = con.Query("SELECT RESULTDB *" + cross_table_non_equality_where_from);
	REQUIRE(!result->HasError());
	RequireResultDBStrategy(*result, ResultDBStrategy::AUTO, ResultDBStrategy::DECOMPOSE);

	REQUIRE_NO_FAIL(con.Query("SET resultdb_strategy = 'semijoin'"));
	result = con.Query("SELECT RESULTDB * FROM a JOIN b ON a.id = b.a_id WHERE 1 = 0");
	REQUIRE(!result->HasError());
	RequireResultDBStrategy(*result, ResultDBStrategy::SEMIJOIN, ResultDBStrategy::SEMIJOIN);
	REQUIRE(result->properties.resultdb.join_edges.empty());
	REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 0);
	duckdb::unique_ptr<QueryResult> next_result = std::move(result->next);
	REQUIRE(next_result);
	REQUIRE(next_result->Cast<MaterializedQueryResult>().RowCount() == 0);
	REQUIRE(!next_result->next);

	auto require_empty_semijoin = [&](const string &from_clause) {
		result = con.Query("SELECT RESULTDB *" + from_clause);
		REQUIRE(!result->HasError());
		RequireResultDBStrategy(*result, ResultDBStrategy::SEMIJOIN, ResultDBStrategy::SEMIJOIN);
		REQUIRE(result->properties.resultdb.join_edges.empty());
		REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 0);
		next_result = std::move(result->next);
		REQUIRE(next_result);
		REQUIRE(next_result->Cast<MaterializedQueryResult>().RowCount() == 0);
		REQUIRE(!next_result->next);
	};
	require_empty_semijoin(" FROM a JOIN b ON a.id = b.a_id WHERE NULL");
	require_empty_semijoin(" FROM a JOIN b ON a.id = b.a_id WHERE a.id = 1 AND a.id = 2");

	REQUIRE_NO_FAIL(con.Query("CREATE TABLE words(id INTEGER, name VARCHAR)"));
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE word_hits(id INTEGER, word_id INTEGER)"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO words VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Alan')"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO word_hits VALUES (10, 1), (20, 2), (30, 3)"));
	const string regex_from =
	    " FROM words w JOIN word_hits h ON w.id = h.word_id WHERE regexp_full_match(w.name, 'A.*')";
	auto expected_regex_words = con.Query("SELECT DISTINCT w.id, w.name" + regex_from);
	auto expected_regex_hits = con.Query("SELECT DISTINCT h.id, h.word_id" + regex_from);
	REQUIRE(!expected_regex_words->HasError());
	REQUIRE(!expected_regex_hits->HasError());
	result = con.Query("SELECT RESULTDB *" + regex_from);
	REQUIRE(!result->HasError());
	RequireResultDBStrategy(*result, ResultDBStrategy::SEMIJOIN, ResultDBStrategy::SEMIJOIN);
	REQUIRE(result->properties.resultdb.join_edges.size() == 1);
	RequireResultDBTable(*result, "w", *expected_regex_words);
	result = std::move(result->next);
	REQUIRE(result);
	RequireResultDBTable(*result, "h", *expected_regex_hits);
	REQUIRE(!result->next);
}

TEST_CASE("Test ResultDB TDResultDB cost comparison selects both executors", "[api][resultdb]") {
	DuckDB db(nullptr);
	Connection con(db);

	REQUIRE_NO_FAIL(con.Query("CREATE TABLE narrow_a(k BOOLEAN)"));
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE narrow_b(k BOOLEAN)"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO narrow_a VALUES (TRUE), (FALSE)"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO narrow_b VALUES (TRUE), (FALSE)"));
	REQUIRE_NO_FAIL(con.Query("SET resultdb_strategy = 'auto'"));
	auto result = con.Query("SELECT RESULTDB * FROM narrow_a a JOIN narrow_b b ON a.k = b.k");
	REQUIRE(!result->HasError());
	RequireResultDBStrategy(*result, ResultDBStrategy::AUTO, ResultDBStrategy::DECOMPOSE);
	REQUIRE(result->properties.resultdb.estimated_decompose_cost <=
	        result->properties.resultdb.estimated_semijoin_cost);
	REQUIRE(result->properties.resultdb.planning_reason.find("selected decompose") != string::npos);

	REQUIRE_NO_FAIL(con.Query("CREATE TABLE wide_a(k INTEGER, payload VARCHAR)"));
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE wide_b(k INTEGER, payload VARCHAR)"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO wide_a VALUES (1, repeat('a', 100)), (2, repeat('b', 100))"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO wide_b VALUES (1, repeat('c', 100)), (2, repeat('d', 100))"));
	result = con.Query("SELECT RESULTDB * FROM wide_a a JOIN wide_b b ON a.k = b.k");
	REQUIRE(!result->HasError());
	RequireResultDBStrategy(*result, ResultDBStrategy::AUTO, ResultDBStrategy::SEMIJOIN);
	REQUIRE(result->properties.resultdb.estimated_semijoin_cost <
	        result->properties.resultdb.estimated_decompose_cost);
	REQUIRE(result->properties.resultdb.planning_reason.find("selected TDFold+TDRoot") != string::npos);
}

TEST_CASE("Test ResultDB query returns larger source relation set", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);

	REQUIRE_NO_FAIL(con.Query("CREATE TABLE regions AS SELECT i::INTEGER id, 'region_' || i::VARCHAR name "
	                          "FROM range(1, 5) tbl(i)"));
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE customers AS SELECT i::INTEGER id, 'customer_' || i::VARCHAR name, "
	                          "(((i - 1) % 4) + 1)::INTEGER region_id FROM range(1, 21) tbl(i)"));
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE orders AS SELECT i::INTEGER id, (((i - 1) % 20) + 1)::INTEGER customer_id, "
	                          "(100 + i)::INTEGER amount FROM range(1, 61) tbl(i)"));
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE products AS SELECT i::INTEGER id, 'product_' || i::VARCHAR name, "
	                          "(10 * i)::INTEGER price FROM range(1, 11) tbl(i)"));
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE lineitems AS SELECT i::INTEGER id, (((i - 1) % 60) + 1)::INTEGER order_id, "
	                          "(((i - 1) % 10) + 1)::INTEGER product_id, (((i - 1) % 3) + 1)::INTEGER quantity "
	                          "FROM range(1, 181) tbl(i)"));

	const string from_clause =
	    " FROM regions r "
	    "JOIN customers c ON c.region_id = r.id "
	    "JOIN orders o ON o.customer_id = c.id "
	    "JOIN lineitems l ON l.order_id = o.id "
	    "JOIN products p ON p.id = l.product_id "
	    "WHERE r.id IN (2, 3) AND p.id <= 4";

	duckdb::unique_ptr<QueryResult> result =
	    con.Query("SELECT RESULTDB r.*, c.id, c.name, o.id, o.amount, l.id, l.quantity, p.id, p.name" + from_clause);
	REQUIRE(!result->HasError());

	auto expected_regions = con.Query("SELECT DISTINCT r.id, r.name" + from_clause);
	auto expected_customers = con.Query("SELECT DISTINCT c.id, c.name" + from_clause);
	auto expected_orders = con.Query("SELECT DISTINCT o.id, o.amount" + from_clause);
	auto expected_lineitems = con.Query("SELECT DISTINCT l.id, l.quantity" + from_clause);
	auto expected_products = con.Query("SELECT DISTINCT p.id, p.name" + from_clause);

	RequireResultDBTable(*result, "r", *expected_regions);
	REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 2);

	result = std::move(result->next);
	REQUIRE(result);
	RequireResultDBTable(*result, "c", *expected_customers);
	REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 4);

	result = std::move(result->next);
	REQUIRE(result);
	RequireResultDBTable(*result, "o", *expected_orders);
	REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 12);

	result = std::move(result->next);
	REQUIRE(result);
	RequireResultDBTable(*result, "l", *expected_lineitems);
	REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 36);

	result = std::move(result->next);
	REQUIRE(result);
	RequireResultDBTable(*result, "p", *expected_products);
	REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 4);
	REQUIRE(!result->next);

	result = con.Query("SELECT RESULTDB c.id, c.name, p.id, p.name" + from_clause);
	REQUIRE(!result->HasError());
	RequireResultDBTable(*result, "c", *expected_customers);
	REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 4);

	result = std::move(result->next);
	REQUIRE(result);
	RequireResultDBTable(*result, "p", *expected_products);
	REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 4);
	REQUIRE(!result->next);

	result = con.Query("SELECT RESULTDB *" + from_clause);
	REQUIRE(!result->HasError());

	auto expected_all_regions = con.Query("SELECT DISTINCT r.id, r.name" + from_clause);
	auto expected_all_customers = con.Query("SELECT DISTINCT c.id, c.name, c.region_id" + from_clause);
	auto expected_all_orders = con.Query("SELECT DISTINCT o.id, o.customer_id, o.amount" + from_clause);
	auto expected_all_lineitems = con.Query("SELECT DISTINCT l.id, l.order_id, l.product_id, l.quantity" + from_clause);
	auto expected_all_products = con.Query("SELECT DISTINCT p.id, p.name, p.price" + from_clause);

	RequireResultDBTable(*result, "r", *expected_all_regions);
	REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 2);

	result = std::move(result->next);
	REQUIRE(result);
	RequireResultDBTable(*result, "c", *expected_all_customers);
	REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 4);

	result = std::move(result->next);
	REQUIRE(result);
	RequireResultDBTable(*result, "o", *expected_all_orders);
	REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 12);

	result = std::move(result->next);
	REQUIRE(result);
	RequireResultDBTable(*result, "l", *expected_all_lineitems);
	REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 36);

	result = std::move(result->next);
	REQUIRE(result);
	RequireResultDBTable(*result, "p", *expected_all_products);
	REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 4);
	REQUIRE(!result->next);

	REQUIRE_NO_FAIL(con.Query("SET resultdb_strategy = 'semijoin'"));
	result = con.Query("SELECT RESULTDB *" + from_clause);
	REQUIRE(!result->HasError());
	RequireResultDBStrategy(*result, ResultDBStrategy::SEMIJOIN, ResultDBStrategy::SEMIJOIN);
	REQUIRE(result->properties.resultdb.join_edges.size() == 4);

	RequireResultDBTable(*result, "r", *expected_all_regions);
	REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 2);

	result = std::move(result->next);
	REQUIRE(result);
	RequireResultDBTable(*result, "c", *expected_all_customers);
	REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 4);

	result = std::move(result->next);
	REQUIRE(result);
	RequireResultDBTable(*result, "o", *expected_all_orders);
	REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 12);

	result = std::move(result->next);
	REQUIRE(result);
	RequireResultDBTable(*result, "l", *expected_all_lineitems);
	REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 36);

	result = std::move(result->next);
	REQUIRE(result);
	RequireResultDBTable(*result, "p", *expected_all_products);
	REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 4);
	REQUIRE(!result->next);
}

TEST_CASE("Test ResultDB JOB-shaped occurrence, subset, DISTINCT, and empty semantics", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);

	REQUIRE_NO_FAIL(con.Query("CREATE TABLE people(id INTEGER, manager_id INTEGER, name VARCHAR)"));
	REQUIRE_NO_FAIL(con.Query(
	    "INSERT INTO people VALUES (1, NULL, 'CEO'), (2, 1, 'Ada'), (3, 1, 'Linus'), (4, 2, 'Grace')"));
	const string self_join_from =
	    " FROM people employee, people manager "
	    "WHERE employee.manager_id = manager.id AND employee.id IN (2, 4)";
	auto expected_employees = con.Query("SELECT DISTINCT employee.name" + self_join_from);
	auto expected_managers = con.Query("SELECT DISTINCT manager.name" + self_join_from);
	REQUIRE(!expected_employees->HasError());
	REQUIRE(!expected_managers->HasError());

	for (auto &strategy : vector<string> {"decompose", "semijoin"}) {
		REQUIRE_NO_FAIL(con.Query("SET resultdb_strategy = '" + strategy + "'"));
		duckdb::unique_ptr<QueryResult> result =
		    con.Query("SELECT RESULTDB employee.name, manager.name" + self_join_from);
		REQUIRE(!result->HasError());
		auto employee_table_index = result->properties.resultdb.tables[0].table_index;
		RequireResultDBTable(*result, "employee", *expected_employees);
		result = std::move(result->next);
		REQUIRE(result);
		auto manager_table_index = result->properties.resultdb.tables[0].table_index;
		REQUIRE(employee_table_index != manager_table_index);
		RequireResultDBTable(*result, "manager", *expected_managers);
		REQUIRE(!result->next);
	}

	REQUIRE_NO_FAIL(con.Query("CREATE TABLE endpoint_a(id INTEGER, label VARCHAR)"));
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE internal_b(id INTEGER, a_id INTEGER)"));
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE endpoint_c(id INTEGER, b_id INTEGER)"));
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE constraint_d(id INTEGER, b_id INTEGER)"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO endpoint_a VALUES (1, 'keep'), (2, 'drop')"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO internal_b VALUES (10, 1), (20, 2)"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO endpoint_c VALUES (100, 10), (200, 20)"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO constraint_d VALUES (1000, 10)"));
	const string selected_endpoints_from =
	    " FROM endpoint_a a "
	    "JOIN internal_b b ON a.id = b.a_id "
	    "JOIN endpoint_c c ON b.id = c.b_id "
	    "JOIN constraint_d d ON b.id = d.b_id";
	auto expected_endpoint_a = con.Query("SELECT DISTINCT a.label" + selected_endpoints_from);
	auto expected_endpoint_c = con.Query("SELECT DISTINCT c.id" + selected_endpoints_from);
	REQUIRE(!expected_endpoint_a->HasError());
	REQUIRE(!expected_endpoint_c->HasError());

	for (auto &strategy : vector<string> {"decompose", "semijoin"}) {
		REQUIRE_NO_FAIL(con.Query("SET resultdb_strategy = '" + strategy + "'"));
		duckdb::unique_ptr<QueryResult> result =
		    con.Query("SELECT RESULTDB a.label, c.id" + selected_endpoints_from);
		REQUIRE(!result->HasError());
		RequireResultDBTable(*result, "a", *expected_endpoint_a);
		REQUIRE(CHECK_COLUMN(result, 0, {"keep"}));
		result = std::move(result->next);
		REQUIRE(result);
		RequireResultDBTable(*result, "c", *expected_endpoint_c);
		REQUIRE(CHECK_COLUMN(result, 0, {100}));
		REQUIRE(!result->next);
	}

	REQUIRE_NO_FAIL(con.Query("CREATE TABLE nullable_projection(id INTEGER, note VARCHAR)"));
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE nullable_hits(a_id INTEGER)"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO nullable_projection VALUES (1, NULL), (2, NULL), (3, 'other')"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO nullable_hits VALUES (1), (2)"));
	const string nullable_projection_from =
	    " FROM nullable_projection a JOIN nullable_hits b ON a.id = b.a_id";
	auto expected_nullable = con.Query("SELECT DISTINCT a.note" + nullable_projection_from);
	REQUIRE(!expected_nullable->HasError());

	for (auto &strategy : vector<string> {"decompose", "semijoin"}) {
		REQUIRE_NO_FAIL(con.Query("SET resultdb_strategy = '" + strategy + "'"));
		duckdb::unique_ptr<QueryResult> result =
		    con.Query("SELECT RESULTDB a.note" + nullable_projection_from);
		REQUIRE(!result->HasError());
		RequireResultDBTable(*result, "a", *expected_nullable);
		REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 1);
		REQUIRE(CHECK_COLUMN(result, 0, {Value()}));
		REQUIRE(!result->next);
	}

	REQUIRE_NO_FAIL(con.Query("CREATE TABLE empty_a(id INTEGER, label VARCHAR)"));
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE empty_b(id INTEGER, a_id INTEGER, payload VARCHAR)"));
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE empty_c(id INTEGER, b_id INTEGER)"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO empty_a VALUES (1, 'a1'), (3, 'a3')"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO empty_b VALUES (10, 2, 'b2'), (20, 4, 'b4')"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO empty_c VALUES (100, 10), (200, 20)"));
	const string runtime_empty_from =
	    " FROM empty_a a JOIN empty_b b ON a.id = b.a_id JOIN empty_c c ON b.id = c.b_id";
	auto expected_empty_a = con.Query("SELECT DISTINCT a.label" + runtime_empty_from);
	auto expected_empty_b = con.Query("SELECT DISTINCT b.payload" + runtime_empty_from);
	REQUIRE(!expected_empty_a->HasError());
	REQUIRE(!expected_empty_b->HasError());

	for (auto &strategy : vector<string> {"decompose", "semijoin"}) {
		REQUIRE_NO_FAIL(con.Query("SET resultdb_strategy = '" + strategy + "'"));
		duckdb::unique_ptr<QueryResult> result =
		    con.Query("SELECT RESULTDB a.label, b.payload" + runtime_empty_from);
		REQUIRE(!result->HasError());
		RequireResultDBTable(*result, "a", *expected_empty_a);
		REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 0);
		result = std::move(result->next);
		REQUIRE(result);
		RequireResultDBTable(*result, "b", *expected_empty_b);
		REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 0);
		REQUIRE(!result->next);
	}
}

TEST_CASE("Test ResultDB top-down child hashes use parent-reduced relay rows", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);

	REQUIRE_NO_FAIL(con.Query("CREATE TABLE wave_d(id INTEGER, payload VARCHAR)"));
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE wave_b(id INTEGER, d_id INTEGER, a_id INTEGER, c_id INTEGER)"));
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE wave_a(id INTEGER, payload VARCHAR)"));
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE wave_c(id INTEGER, payload VARCHAR)"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO wave_d VALUES (1, 'd-keep'), (2, 'd-no-descendant')"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO wave_b VALUES "
	                          "(10, 1, 10, 100), "
	                          "(20, 2, 20, 999), "
	                          "(30, 3, 30, 300)"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO wave_a VALUES (10, 'a-keep'), (20, 'a-no-c'), (30, 'a-no-d')"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO wave_c VALUES "
	                          "(100, 'c-keep'), (300, 'c-must-be-filtered-by-d'), (400, 'c-unrelated')"));

	const string from_clause =
	    " FROM wave_d d "
	    "JOIN wave_b b ON d.id = b.d_id "
	    "JOIN wave_a a ON b.a_id = a.id "
	    "JOIN wave_c c ON b.c_id = c.id";
	auto expected_d = con.Query("SELECT DISTINCT d.payload" + from_clause);
	auto expected_c = con.Query("SELECT DISTINCT c.payload" + from_clause);
	REQUIRE(!expected_d->HasError());
	REQUIRE(!expected_c->HasError());

	REQUIRE_NO_FAIL(con.Query("SET resultdb_strategy = 'semijoin'"));
	duckdb::unique_ptr<QueryResult> result =
	    con.Query("SELECT RESULTDB d.payload, c.payload" + from_clause);
	REQUIRE(!result->HasError());
	RequireResultDBStrategy(*result, ResultDBStrategy::SEMIJOIN, ResultDBStrategy::SEMIJOIN);
	REQUIRE(result->properties.resultdb.join_edges.size() == 3);
	RequireResultDBTable(*result, "d", *expected_d);
	REQUIRE(CHECK_COLUMN(result, 0, {"d-keep"}));
	result = std::move(result->next);
	REQUIRE(result);
	RequireResultDBTable(*result, "c", *expected_c);
	REQUIRE(CHECK_COLUMN(result, 0, {"c-keep"}));
	REQUIRE(!result->next);

	// With only D requested, B, A, and C still constrain D but need no top-down output phase.
	result = con.Query("SELECT RESULTDB d.payload" + from_clause);
	REQUIRE(!result->HasError());
	RequireResultDBStrategy(*result, ResultDBStrategy::SEMIJOIN, ResultDBStrategy::SEMIJOIN);
	RequireResultDBTable(*result, "d", *expected_d);
	REQUIRE(CHECK_COLUMN(result, 0, {"d-keep"}));
	REQUIRE(!result->next);
}

TEST_CASE("Test ResultDB folded relations preserve occurrence outputs with hidden join keys", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);

	REQUIRE_NO_FAIL(con.Query("CREATE TABLE fold_a(id INTEGER, payload VARCHAR)"));
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE fold_b(id INTEGER, a_id INTEGER, payload BIGINT)"));
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE fold_c(id INTEGER, b_id INTEGER, a_id INTEGER, note VARCHAR)"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO fold_a VALUES (1, 'fa-1'), (2, 'fa-2')"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO fold_b VALUES (10, 1, 1000), (20, 2, 2000)"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO fold_c VALUES "
	                          "(100, 10, 1, NULL), (100, 10, 1, NULL), "
	                          "(200, 20, 2, NULL), (300, 10, 2, 'cycle-invalid')"));

	const string from_clause =
	    " FROM fold_a fa "
	    "JOIN fold_b fb ON fa.id = fb.a_id "
	    "JOIN fold_c fc ON fb.id = fc.b_id AND fa.id = fc.a_id";
	auto expected_a = con.Query("SELECT DISTINCT fa.payload" + from_clause);
	auto expected_b = con.Query("SELECT DISTINCT fb.payload" + from_clause);
	auto expected_c = con.Query("SELECT DISTINCT fc.note" + from_clause);
	REQUIRE(!expected_a->HasError());
	REQUIRE(!expected_b->HasError());
	REQUIRE(!expected_c->HasError());

	REQUIRE_NO_FAIL(con.Query("SET resultdb_strategy = 'semijoin'"));
	duckdb::unique_ptr<QueryResult> result =
	    con.Query("SELECT RESULTDB fa.payload, fb.payload, fc.note" + from_clause);
	REQUIRE(!result->HasError());
	RequireResultDBStrategy(*result, ResultDBStrategy::SEMIJOIN, ResultDBStrategy::SEMIJOIN);
	REQUIRE(result->properties.resultdb.join_edges.size() == 3);

	auto fa_table_index = result->properties.resultdb.tables[0].table_index;
	RequireResultDBTable(*result, "fa", *expected_a);
	REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 2);
	result = std::move(result->next);
	REQUIRE(result);
	auto fb_table_index = result->properties.resultdb.tables[0].table_index;
	RequireResultDBTable(*result, "fb", *expected_b);
	REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 2);
	result = std::move(result->next);
	REQUIRE(result);
	auto fc_table_index = result->properties.resultdb.tables[0].table_index;
	RequireResultDBTable(*result, "fc", *expected_c);
	REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 1);
	REQUIRE(CHECK_COLUMN(result, 0, {Value()}));
	REQUIRE(fa_table_index != fb_table_index);
	REQUIRE(fa_table_index != fc_table_index);
	REQUIRE(fb_table_index != fc_table_index);
	REQUIRE(!result->next);
}

TEST_CASE("Test ResultDB prepared schedule prunes unrequested top-down branches", "[api]") {
	PreparedResultDBYannakakisProgram program;
	program.root_relation = 0;
	program.parent = {DConstants::INVALID_INDEX, 0, 0, 0};
	program.parent_edge = {DConstants::INVALID_INDEX, 0, 1, 2};
	program.order = {0, 1, 2, 3};
	program.relations.resize(4);
	for (auto &relation : program.relations) {
		relation.names = {"value"};
		relation.types = {LogicalType::INTEGER};
	}
	for (idx_t child = 1; child < 4; child++) {
		ResultDBYannakakisEdge edge;
		edge.left_relation = 0;
		edge.right_relation = child;
		edge.columns.push_back({0, 0});
		program.edges.push_back(std::move(edge));
	}
	ResultDBYannakakisOutputTable output;
	output.relation = 2;
	output.table_metadata_index = 0;
	output.columns.push_back({0, "value", LogicalType::INTEGER});
	program.outputs.push_back(std::move(output));

	program.BuildReductionSchedule();
	REQUIRE(program.bottom_up_steps.size() == 3);
	REQUIRE(program.bottom_up_steps[0].target_relation == 0);
	REQUIRE(program.bottom_up_steps[0].source_relation == 3);
	REQUIRE(program.bottom_up_steps[1].target_relation == 0);
	REQUIRE(program.bottom_up_steps[1].source_relation == 2);
	REQUIRE(program.bottom_up_steps[2].target_relation == 0);
	REQUIRE(program.bottom_up_steps[2].source_relation == 1);
	REQUIRE(program.top_down_steps.size() == 1);
	REQUIRE(program.top_down_steps[0].target_relation == 2);
	REQUIRE(program.top_down_steps[0].source_relation == 0);
	REQUIRE(program.required_for_output == vector<uint8_t> {1, 0, 1, 0});

	REQUIRE(program.relation_phases.size() == 4);
	REQUIRE(program.relation_phases[0].bottom_up_from_children_steps == vector<idx_t> {0, 1, 2});
	REQUIRE(program.relation_phases[0].top_down_to_children_steps == vector<idx_t> {0});
	REQUIRE(!program.relation_phases[0].retain_bottom_up);
	REQUIRE(program.relation_phases[1].bottom_up_to_parent_step == 2);
	REQUIRE(program.relation_phases[1].top_down_from_parent_step == DConstants::INVALID_INDEX);
	REQUIRE(!program.relation_phases[1].retain_bottom_up);
	REQUIRE(program.relation_phases[2].bottom_up_to_parent_step == 1);
	REQUIRE(program.relation_phases[2].top_down_from_parent_step == 0);
	REQUIRE(program.relation_phases[2].output_indexes == vector<idx_t> {0});
	REQUIRE(program.relation_phases[2].retain_bottom_up);
	REQUIRE(program.relation_phases[3].bottom_up_to_parent_step == 0);
	REQUIRE(program.relation_phases[3].top_down_from_parent_step == DConstants::INVALID_INDEX);
	REQUIRE(!program.relation_phases[3].retain_bottom_up);
}

TEST_CASE("Test ResultDB root-only prepared phases keep all children hash-only", "[api]") {
	PreparedResultDBYannakakisProgram program;
	program.root_relation = 0;
	program.parent = {DConstants::INVALID_INDEX, 0, 0, 0};
	program.parent_edge = {DConstants::INVALID_INDEX, 0, 1, 2};
	program.order = {0, 1, 2, 3};
	program.relations.resize(4);
	for (auto &relation : program.relations) {
		relation.names = {"value"};
		relation.types = {LogicalType::INTEGER};
	}
	for (idx_t child = 1; child < 4; child++) {
		ResultDBYannakakisEdge edge;
		edge.left_relation = 0;
		edge.right_relation = child;
		edge.columns.push_back({0, 0});
		program.edges.push_back(std::move(edge));
	}
	ResultDBYannakakisOutputTable output;
	output.relation = 0;
	output.table_metadata_index = 0;
	output.columns.push_back({0, "value", LogicalType::INTEGER});
	program.outputs.push_back(std::move(output));

	program.BuildReductionSchedule();

	REQUIRE(program.bottom_up_steps.size() == 3);
	REQUIRE(program.top_down_steps.empty());
	REQUIRE(program.required_for_output == vector<uint8_t> {1, 0, 0, 0});
	REQUIRE(program.relation_phases.size() == 4);

	auto &root = program.relation_phases[0];
	REQUIRE(root.bottom_up_to_parent_step == DConstants::INVALID_INDEX);
	REQUIRE(root.bottom_up_from_children_steps == vector<idx_t> {0, 1, 2});
	REQUIRE(root.top_down_from_parent_step == DConstants::INVALID_INDEX);
	REQUIRE(root.top_down_to_children_steps.empty());
	REQUIRE(root.output_indexes == vector<idx_t> {0});
	REQUIRE(!root.retain_bottom_up);

	for (idx_t relation_idx = 1; relation_idx < program.relation_phases.size(); relation_idx++) {
		auto &phase = program.relation_phases[relation_idx];
		REQUIRE(phase.bottom_up_to_parent_step != DConstants::INVALID_INDEX);
		auto &step = program.bottom_up_steps[phase.bottom_up_to_parent_step];
		REQUIRE(step.source_relation == relation_idx);
		REQUIRE(step.target_relation == 0);
		REQUIRE(phase.bottom_up_from_children_steps.empty());
		REQUIRE(phase.top_down_from_parent_step == DConstants::INVALID_INDEX);
		REQUIRE(phase.top_down_to_children_steps.empty());
		REQUIRE(phase.output_indexes.empty());
		REQUIRE(!phase.retain_bottom_up);
	}
}

TEST_CASE("Test ResultDB prepared phases classify root, relay, off-path, and output relations", "[api]") {
	PreparedResultDBYannakakisProgram program;
	program.root_relation = 0; // D
	program.parent = {DConstants::INVALID_INDEX, 0, 1, 1};
	program.parent_edge = {DConstants::INVALID_INDEX, 0, 1, 2};
	program.order = {0, 1, 2, 3};
	program.relations.resize(4);
	for (auto &relation : program.relations) {
		relation.names = {"value"};
		relation.types = {LogicalType::INTEGER};
	}
	// D-B, B-A, B-C
	const vector<std::pair<idx_t, idx_t>> tree_edges {{0, 1}, {1, 2}, {1, 3}};
	for (auto &endpoints : tree_edges) {
		ResultDBYannakakisEdge edge;
		edge.left_relation = endpoints.first;
		edge.right_relation = endpoints.second;
		edge.columns.push_back({0, 0});
		program.edges.push_back(std::move(edge));
	}
	ResultDBYannakakisOutputTable output;
	output.relation = 3; // Only C is requested.
	output.table_metadata_index = 0;
	output.columns.push_back({0, "value", LogicalType::INTEGER});
	program.outputs.push_back(std::move(output));

	program.BuildReductionSchedule();

	REQUIRE(program.bottom_up_steps.size() == 3);
	REQUIRE(program.bottom_up_steps[0].target_relation == 1);
	REQUIRE(program.bottom_up_steps[0].source_relation == 3);
	REQUIRE(program.bottom_up_steps[1].target_relation == 1);
	REQUIRE(program.bottom_up_steps[1].source_relation == 2);
	REQUIRE(program.bottom_up_steps[2].target_relation == 0);
	REQUIRE(program.bottom_up_steps[2].source_relation == 1);
	REQUIRE(program.top_down_steps.size() == 2);
	REQUIRE(program.top_down_steps[0].target_relation == 1);
	REQUIRE(program.top_down_steps[0].source_relation == 0);
	REQUIRE(program.top_down_steps[1].target_relation == 3);
	REQUIRE(program.top_down_steps[1].source_relation == 1);
	REQUIRE(program.required_for_output == vector<uint8_t> {1, 1, 0, 1});

	REQUIRE(program.relation_phases.size() == 4);
	auto &root = program.relation_phases[0];
	REQUIRE(root.bottom_up_to_parent_step == DConstants::INVALID_INDEX);
	REQUIRE(root.bottom_up_from_children_steps == vector<idx_t> {2});
	REQUIRE(root.top_down_from_parent_step == DConstants::INVALID_INDEX);
	REQUIRE(root.top_down_to_children_steps == vector<idx_t> {0});
	REQUIRE(root.output_indexes.empty());
	REQUIRE(!root.retain_bottom_up);

	auto &relay = program.relation_phases[1];
	REQUIRE(relay.bottom_up_to_parent_step == 2);
	REQUIRE(relay.bottom_up_from_children_steps == vector<idx_t> {0, 1});
	REQUIRE(relay.top_down_from_parent_step == 0);
	REQUIRE(relay.top_down_to_children_steps == vector<idx_t> {1});
	REQUIRE(relay.output_indexes.empty());
	REQUIRE(relay.retain_bottom_up);

	auto &off_path_leaf = program.relation_phases[2];
	REQUIRE(off_path_leaf.bottom_up_to_parent_step == 1);
	REQUIRE(off_path_leaf.bottom_up_from_children_steps.empty());
	REQUIRE(off_path_leaf.top_down_from_parent_step == DConstants::INVALID_INDEX);
	REQUIRE(off_path_leaf.top_down_to_children_steps.empty());
	REQUIRE(off_path_leaf.output_indexes.empty());
	REQUIRE(!off_path_leaf.retain_bottom_up);

	auto &output_leaf = program.relation_phases[3];
	REQUIRE(output_leaf.bottom_up_to_parent_step == 0);
	REQUIRE(output_leaf.bottom_up_from_children_steps.empty());
	REQUIRE(output_leaf.top_down_from_parent_step == 1);
	REQUIRE(output_leaf.top_down_to_children_steps.empty());
	REQUIRE(output_leaf.output_indexes == vector<idx_t> {0});
	REQUIRE(output_leaf.retain_bottom_up);
}

TEST_CASE("Test streaming API errors", "[api]") {
	duckdb::unique_ptr<QueryResult> result, result2;
	DuckDB db(nullptr);
	Connection con(db);

	// multiple streaming result
	result = con.SendQuery("SELECT 42;");
	result2 = con.SendQuery("SELECT 42;");
	// "result" is invalidated
	REQUIRE_THROWS(CHECK_COLUMN(result, 0, {42}));
	// "result2" we can read
	REQUIRE(CHECK_COLUMN(result2, 0, {42}));

	// streaming result followed by non-streaming result
	result = con.SendQuery("SELECT 42;");
	result2 = con.Query("SELECT 42;");
	// "result" is invalidated
	REQUIRE_THROWS(CHECK_COLUMN(result, 0, {42}));
	// "result2" we can read
	REQUIRE(CHECK_COLUMN(result2, 0, {42}));

	// error in binding
	result = con.SendQuery("SELECT * FROM nonexistanttable");
	REQUIRE(!result->ToString().empty());
	REQUIRE(result->type == QueryResultType::MATERIALIZED_RESULT);
	REQUIRE_FAIL(result);

	// error in stream that only happens after fetching
	result = con.SendQuery(
	    "SELECT x::INT FROM (SELECT x::VARCHAR x FROM range(10) tbl(x) UNION ALL SELECT 'hello' x) tbl(x);");
	while (!result->HasError()) {
		auto chunk = result->Fetch();
		if (!chunk || chunk->size() == 0) {
			break;
		}
	}
	REQUIRE(!result->ToString().empty());
	REQUIRE_FAIL(result);

	// same query but call Materialize
	result = con.SendQuery(
	    "SELECT x::INT FROM (SELECT x::VARCHAR x FROM range(10) tbl(x) UNION ALL SELECT 'hello' x) tbl(x);");
	REQUIRE(!result->ToString().empty());
	REQUIRE(result->type == QueryResultType::STREAM_RESULT);
	result = ((StreamQueryResult &)*result).Materialize();
	REQUIRE_FAIL(result);

	// same query but call materialize after fetching
	result = con.SendQuery(
	    "SELECT x::INT FROM (SELECT x::VARCHAR x FROM range(10) tbl(x) UNION ALL SELECT 'hello' x) tbl(x);");
	while (!result->HasError()) {
		auto chunk = result->Fetch();
		if (!chunk || chunk->size() == 0) {
			break;
		}
	}
	REQUIRE(!result->ToString().empty());
	REQUIRE(result->type == QueryResultType::STREAM_RESULT);
	result = ((StreamQueryResult &)*result).Materialize();
	REQUIRE_FAIL(result);
}

TEST_CASE("Test fetch API", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);

	duckdb::unique_ptr<QueryResult> result;

	// fetch from an error
	result = con.Query("SELECT 'hello'::INT");
	REQUIRE_THROWS(result->Fetch());

	result = con.SendQuery("CREATE TABLE test (a INTEGER);");

	result = con.Query("select a from test where 1 <> 1");
	REQUIRE(CHECK_COLUMN(result, 0, {}));

	result = con.SendQuery("INSERT INTO test VALUES (42)");
	result = con.SendQuery("SELECT a from test");
	REQUIRE(CHECK_COLUMN(result, 0, {42}));

	auto materialized_result = con.Query("select a from test");
	REQUIRE(CHECK_COLUMN(materialized_result, 0, {42}));

	// override fetch result
	result = con.SendQuery("SELECT a from test");
	result = con.SendQuery("SELECT a from test");
	result = con.SendQuery("SELECT a from test");
	result = con.SendQuery("SELECT a from test");
	REQUIRE(CHECK_COLUMN(result, 0, {42}));
}

TEST_CASE("Test fetch API not to completion", "[api]") {
	auto db = make_uniq<DuckDB>(nullptr);
	auto conn = make_uniq<Connection>(*db);
	// remove connection with active stream result
	auto result = conn->SendQuery("SELECT 42");
	// close the connection
	conn.reset();
	// now try to fetch a chunk, this should not return a nullptr
	auto chunk = result->Fetch();
	REQUIRE(chunk);
	// Only if we would call Fetch again would we Close the QueryResult
	// this is testing that it can get cleaned up without this.

	db.reset();
}

TEST_CASE("Test fetch API robustness", "[api]") {
	auto db = make_uniq<DuckDB>(nullptr);
	auto conn = make_uniq<Connection>(*db);

	// remove connection with active stream result
	auto result = conn->SendQuery("SELECT 42");
	// close the connection
	conn.reset();
	// now try to fetch a chunk, this should not return a nullptr
	auto chunk = result->Fetch();
	REQUIRE(chunk);

	// now close the entire database
	conn = make_uniq<Connection>(*db);
	result = conn->SendQuery("SELECT 42");

	db.reset();
	// fetch should not fail
	chunk = result->Fetch();
	REQUIRE(chunk);
	// new queries on the connection should not fail either
	REQUIRE_NO_FAIL(conn->SendQuery("SELECT 42"));

	// override fetch result
	db = make_uniq<DuckDB>(nullptr);
	conn = make_uniq<Connection>(*db);
	auto result1 = conn->SendQuery("SELECT 42");
	auto result2 = conn->SendQuery("SELECT 84");
	REQUIRE_NO_FAIL(*result1);
	REQUIRE_NO_FAIL(*result2);

	// result1 should be closed now
	REQUIRE_THROWS(result1->Fetch());
	// result2 should work
	REQUIRE(result2->Fetch());

	// test materialize
	result1 = conn->SendQuery("SELECT 42");
	REQUIRE(result1->type == QueryResultType::STREAM_RESULT);
	auto materialized = ((StreamQueryResult &)*result1).Materialize();
	result2 = conn->SendQuery("SELECT 84");

	// we can read materialized still, even after opening a new result
	REQUIRE(CHECK_COLUMN(materialized, 0, {42}));
	REQUIRE(CHECK_COLUMN(result2, 0, {84}));
}

static void VerifyStreamResult(duckdb::unique_ptr<QueryResult> result) {
	REQUIRE(result->types[0] == LogicalType::INTEGER);
	size_t current_row = 0;
	int current_expected_value = 0;
	size_t expected_rows = 500 * 5;
	while (true) {
		auto chunk = result->Fetch();
		if (!chunk || chunk->size() == 0) {
			break;
		}
		auto col1_data = FlatVector::GetData<int>(chunk->data[0]);
		for (size_t k = 0; k < chunk->size(); k++) {
			if (current_row % 500 == 0) {
				current_expected_value++;
			}
			REQUIRE(col1_data[k] == current_expected_value);
			current_row++;
		}
	}
	REQUIRE(current_row == expected_rows);
}

TEST_CASE("Test fetch API with big results", "[api][.]") {
	DuckDB db(nullptr);
	Connection con(db);

	// create table that consists of multiple chunks
	REQUIRE_NO_FAIL(con.Query("BEGIN TRANSACTION"));
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE test(a INTEGER)"));
	for (size_t i = 0; i < 500; i++) {
		REQUIRE_NO_FAIL(con.Query("INSERT INTO test VALUES (1); INSERT INTO test VALUES (2); INSERT INTO test VALUES "
		                          "(3); INSERT INTO test VALUES (4); INSERT INTO test VALUES (5);"));
	}
	REQUIRE_NO_FAIL(con.Query("COMMIT"));

	// stream the results using the Fetch() API
	auto result = con.SendQuery("SELECT CAST(a AS INTEGER) FROM test ORDER BY a");
	VerifyStreamResult(std::move(result));
	// we can also stream a materialized result
	auto materialized = con.Query("SELECT CAST(a AS INTEGER) FROM test ORDER BY a");
	VerifyStreamResult(std::move(materialized));
	// return multiple results using the stream API
	result = con.SendQuery("SELECT CAST(a AS INTEGER) FROM test ORDER BY a; SELECT CAST(a AS INTEGER) FROM test ORDER "
	                       "BY a; SELECT CAST(a AS INTEGER) FROM test ORDER BY a;");
	auto next = std::move(result->next);
	while (next) {
		auto nextnext = std::move(next->next);
		VerifyStreamResult(std::move(nextnext));
		next = std::move(nextnext);
	}
	VerifyStreamResult(std::move(result));
}

TEST_CASE("Test TryFlushCachingOperators interrupted ExecutePushInternal", "[api][.]") {
	DuckDB db;
	Connection con(db);

	con.Query("create table tbl as select 100000 a from range(2) t(a);");
	con.Query("pragma threads=1");

	// Use PhysicalCrossProduct with a very low amount of produced tuples, this caches the result in the
	// CachingOperatorState This gets flushed with FinalExecute in PipelineExecutor::TryFlushCachingOperator
	auto pending_query = con.PendingQuery("select unnest(range(a.a)) from tbl a, tbl b;");

	// Through `unnest(range(a.a.))` this FinalExecute multiple chunks, more than the ExecutionBudget can handle with
	// PROCESS_PARTIAL
	pending_query->ExecuteTask();

	// query the connection as normal after
	auto res = pending_query->Execute();
	REQUIRE(!res->HasError());
	auto &materialized_res = res->Cast<MaterializedQueryResult>();
	idx_t initial_tuples = 2 * 2;
	REQUIRE(materialized_res.RowCount() == initial_tuples * 100000);
	for (idx_t i = 0; i < initial_tuples; i++) {
		for (idx_t j = 0; j < 100000; j++) {
			auto value = static_cast<idx_t>(materialized_res.GetValue<int64_t>(0, (i * 100000) + j));
			REQUIRE(value == j);
		}
	}
}

TEST_CASE("Test streaming query during stack unwinding", "[api]") {
	DuckDB db;
	Connection con(db);

	try {
		auto result = con.SendQuery("SELECT * FROM range(1000000)");

		throw std::runtime_error("hello");
	} catch (...) {
	}
}

TEST_CASE("Test prepare dependencies with multiple connections", "[catalog]") {
	duckdb::unique_ptr<QueryResult> result;
	DuckDB db(nullptr);
	auto con = make_uniq<Connection>(db);
	auto con2 = make_uniq<Connection>(db);
	auto con3 = make_uniq<Connection>(db);

	// simple prepare: begin transaction before the second client calls PREPARE
	REQUIRE_NO_FAIL(con->Query("CREATE TABLE integers(i INTEGER)"));
	// open a transaction in con2, this forces the prepared statement to be kept around until this transaction is closed
	REQUIRE_NO_FAIL(con2->Query("BEGIN TRANSACTION"));
	// we prepare a statement in con
	REQUIRE_NO_FAIL(con->Query("PREPARE s1 AS SELECT * FROM integers"));
	// now we drop con while the second client still has an active transaction
	con.reset();
	// now commit the transaction in the second client
	REQUIRE_NO_FAIL(con2->Query("COMMIT"));

	con = make_uniq<Connection>(db);
	// three transactions
	// open a transaction in con2, this forces the prepared statement to be kept around until this transaction is closed
	REQUIRE_NO_FAIL(con2->Query("BEGIN TRANSACTION"));
	// create a prepare, this creates a dependency from s1 -> integers
	REQUIRE_NO_FAIL(con->Query("PREPARE s1 AS SELECT * FROM integers"));
	// drop the client
	con.reset();
	// now begin a transaction in con3
	REQUIRE_NO_FAIL(con3->Query("BEGIN TRANSACTION"));
	// drop the table integers with cascade, this should drop s1 as well
	REQUIRE_NO_FAIL(con3->Query("DROP TABLE integers CASCADE"));
	REQUIRE_NO_FAIL(con2->Query("COMMIT"));
	REQUIRE_NO_FAIL(con3->Query("COMMIT"));
}

TEST_CASE("Test connection API", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);

	// extract a plan node
	REQUIRE_NOTHROW(con.ExtractPlan("SELECT 42"));
	// can only extract one statement at a time
	REQUIRE_THROWS(con.ExtractPlan("SELECT 42; SELECT 84"));

	// append to a table
	con.Query("CREATE TABLE integers(i integer);");
	auto table_info = con.TableInfo("integers");

	// no transaction active
	REQUIRE_THROWS(con.Commit());
	REQUIRE_THROWS(con.Rollback());

	// cannot start a transaction within a transaction
	REQUIRE_NOTHROW(con.BeginTransaction());
	REQUIRE_THROWS(con.BeginTransaction());

	con.SetAutoCommit(false);
	REQUIRE(!con.IsAutoCommit());

	con.SetAutoCommit(true);
	REQUIRE(con.IsAutoCommit());
}

TEST_CASE("Test parser tokenize", "[api]") {
	Parser parser;
	REQUIRE_NOTHROW(parser.Tokenize("SELECT * FROM table WHERE i+1=3 AND j='hello'; --tokenize example query"));
}

TEST_CASE("Test opening an invalid database file", "[api]") {
	duckdb::unique_ptr<DuckDB> db;
	bool success = false;
	try {
		db = make_uniq<DuckDB>("duckdb:data/parquet-testing/blob.parquet");
		success = true;
	} catch (std::exception &ex) {
		REQUIRE(StringUtil::Contains(ex.what(), "DuckDB"));
	}
	REQUIRE(!success);
	try {
		db = make_uniq<DuckDB>("duckdb:data/parquet-testing/h2oai/h2oai_group_small.parquet");
		success = true;
	} catch (std::exception &ex) {
		REQUIRE(StringUtil::Contains(ex.what(), "DuckDB"));
	}
	REQUIRE(!success);
}

TEST_CASE("Test large number of connections to a single database", "[api]") {
	auto db = make_uniq<DuckDB>(nullptr);
	auto context = make_uniq<ClientContext>((*db).instance);
	auto &connection_manager = ConnectionManager::Get(*context);

	duckdb::vector<duckdb::unique_ptr<Connection>> connections;
	size_t createdConnections = 5000;
	size_t remainingConnections = 500;
	size_t toRemove = createdConnections - remainingConnections;

	for (size_t i = 0; i < createdConnections; i++) {
		auto conn = make_uniq<Connection>(*db);
		connections.push_back(std::move(conn));
	}

	REQUIRE(connection_manager.GetConnectionCount() == createdConnections);

	for (size_t i = 0; i < toRemove; i++) {
		connections.erase(connections.begin());
	}

	REQUIRE(connection_manager.GetConnectionCount() == remainingConnections);
}

TEST_CASE("Issue #4583: Catch Insert/Update/Delete errors", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);
	duckdb::unique_ptr<QueryResult> result;

	REQUIRE_NO_FAIL(con.Query("CREATE TABLE t0 (c0 int);"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO t0 VALUES (1);"));

	result = con.SendQuery(
	    "INSERT INTO t0(VALUES('\\x15\\x00\\x00\\x00\\x00@\\x01\\x0A\\x27:!\\x0A\\x00\\x00x12e\"\\x00'::BLOB));");
	//! Should not terminate the process
	REQUIRE_FAIL(result);

	result = con.SendQuery("SELECT MIN(c0) FROM t0;");
	REQUIRE(CHECK_COLUMN(result, 0, {1}));
}

TEST_CASE("Issue #14130: InsertStatement::ToString causes InternalException later on", "[api][.]") {
	auto db = DuckDB(nullptr);
	auto conn = Connection(db);

	conn.Query("CREATE TABLE foo(a int, b varchar, c int)");

	auto query = "INSERT INTO Foo values (1, 'qwerty', 42)";

	auto stmts = conn.ExtractStatements(query);
	auto &stmt = stmts[0];

	// Issue was here: calling ToString destroyed the 'alias' of the ValuesList
	stmt->ToString();
	// Which caused an 'InternalException: expected non-empty binding_name' here
	auto prepared_stmt = conn.Prepare(std::move(stmt));
	REQUIRE(!prepared_stmt->HasError());
	REQUIRE_NO_FAIL(prepared_stmt->Execute());
}

TEST_CASE("Issue #6284: CachingPhysicalOperator in pull causes issues", "[api][.]") {
	DBConfig config;
	config.options.maximum_threads = 8;
	DuckDB db(nullptr, &config);
	Connection con(db);

	REQUIRE_NO_FAIL(con.Query("select setseed(0.1); CREATE TABLE T0 AS SELECT DISTINCT (RANDOM()*9999999)::BIGINT "
	                          "record_nb, 0.0 x_0, 1.0 y_0 FROM range(1000000) tbl"));
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE T1 AS SELECT record_nb, 0.0 x_1, 1.0 y_1 FROM T0"));
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE T2 AS SELECT record_nb, 0.0 x_2, 1.0 y_2 FROM T0"));
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE T3 AS SELECT record_nb, 0.0 x_3, 1.0 y_3 FROM T0"));
	auto result = con.SendQuery(R"(
        SELECT T0.record_nb,
            T1.x_1 x_1,
            T1.y_1 y_1,
            T2.x_2 x_2,
            T2.y_2 y_2,
            T3.x_3 x_3,
            T3.y_3 y_3
         FROM T0
           INNER JOIN T1 on T0.record_nb = T1.record_nb
           INNER JOIN T2 on T0.record_nb = T2.record_nb
           INNER JOIN T3 on T0.record_nb = T3.record_nb
    )");

	idx_t count = 0;
	while (true) {
		auto chunk = result->Fetch();
		if (!chunk) {
			break;
		}
		if (chunk->size() == 0) {
			break;
		}
		count += chunk->size();
	}

	REQUIRE(951382 == count);
}

TEST_CASE("Fuzzer 50 - Alter table heap-use-after-free", "[api]") {
	// FIXME: not fixed yet
	return;
	DuckDB db(nullptr);
	Connection con(db);

	con.SendQuery("CREATE TABLE t0(c0 INT);");
	con.SendQuery("ALTER TABLE t0 ADD c1 TIMESTAMP_SEC;");
}

TEST_CASE("Test loading database with enable_external_access set to false", "[api]") {
	DBConfig config;
	config.SetOptionByName("enable_external_access", false);
	auto path = TestCreatePath("external_access_test.db");
	DuckDB db(path, &config);
	Connection con(db);

	REQUIRE_FAIL(con.Query("ATTACH 'mydb.db'"));
}

TEST_CASE("Test checkpointing initial database with enable_external_access set to false", "[api]") {
	DBConfig config;
	config.SetOptionByName("enable_external_access", false);
	auto path = TestCreatePath("external_access_test.db");
	DuckDB db(path, &config);
	Connection con(db);

	REQUIRE_NO_FAIL(con.Query("CREATE TABLE tbl(i INTEGER)"));
	REQUIRE_NO_FAIL(con.Query("CHECKPOINT"));
}

TEST_CASE("Test insert returning in CPP API", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);
	con.Query("CREATE TABLE test(val VARCHAR);");

	con.Query("INSERT INTO test(val) VALUES ('query_1')");
	auto res = con.Query("INSERT INTO test(val) VALUES ('query_2') returning *");
	REQUIRE(CHECK_COLUMN(res, 0, {"query_2"}));

	con.Query("INSERT INTO test(val) VALUES (?);", "query_arg_1");
	auto returning_args = con.Query("INSERT INTO test(val) VALUES (?) RETURNING *;", "query_arg_2");
	REQUIRE(CHECK_COLUMN(returning_args, 0, {"query_arg_2"}));

	con.Prepare("INSERT INTO test(val) VALUES (?);")->Execute("prepared_arg_1");
	auto prepared_returning_args =
	    con.Prepare("INSERT INTO test(val) VALUES (?) returning *;")->Execute("prepared_arg_2");
	REQUIRE(CHECK_COLUMN(prepared_returning_args, 0, {"prepared_arg_2"}));

	// make sure all inserts actually inserted
	auto result = con.Query("SELECT * from test;");
	REQUIRE(CHECK_COLUMN(result, 0,
	                     {"query_1", "query_2", "query_arg_1", "query_arg_2", "prepared_arg_1", "prepared_arg_2"}));
}

TEST_CASE("Test a logical execute still has types after an optimization pass", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);
	con.Query("PREPARE test AS SELECT 42::INTEGER;");
	const auto query_plan = con.ExtractPlan("EXECUTE test");
	REQUIRE((query_plan->type == LogicalOperatorType::LOGICAL_EXECUTE));
	REQUIRE((query_plan->types.size() == 1));
	REQUIRE((query_plan->types[0].id() == LogicalTypeId::INTEGER));
}

TEST_CASE("Test SqlStatement::ToString for UPDATE, INSERT, DELETE statements with alias of RETURNING clause", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);
	std::string sql;
	con.Query("CREATE TABLE test(id INT);");

	sql = "INSERT INTO test (id) VALUES (1) RETURNING id AS inserted";
	auto stmts = con.ExtractStatements(sql);
	REQUIRE(stmts[0]->ToString() == "INSERT INTO test (id) (VALUES (1)) RETURNING id AS inserted");

	sql = "UPDATE test SET id = 1 RETURNING id AS updated";
	stmts = con.ExtractStatements(sql);
	REQUIRE(stmts[0]->ToString() == sql);

	sql = "DELETE FROM test WHERE (id = 1) RETURNING id AS deleted";
	stmts = con.ExtractStatements(sql);
	REQUIRE(stmts[0]->ToString() == sql);
}

TEST_CASE("Test buffer managed query result", "[api]") {
	auto db = make_uniq<DuckDB>(nullptr);
	auto con = make_uniq<Connection>(*db);

	// Send query with in-memory result
	QueryParameters parameters;
	parameters.output_type = QueryResultOutputType::FORCE_MATERIALIZED;
	parameters.memory_type = QueryResultMemoryType::IN_MEMORY;
	auto result = con->SendQuery("SELECT 42;", parameters);

	// Query result is accessible
	REQUIRE_NOTHROW(result->ToString());

	// Reset connection AND db
	con.reset();
	db.reset();

	// Query result is still accessible after resetting
	REQUIRE_NOTHROW(result->ToString());

	// Do it again with a buffer-managed query result
	db = make_uniq<DuckDB>(nullptr);
	con = make_uniq<Connection>(*db);
	parameters.memory_type = QueryResultMemoryType::BUFFER_MANAGED;
	result = con->SendQuery("SELECT 42;", parameters);

	// Query result is accessible
	REQUIRE_NOTHROW(result->ToString());

	// Reset connection AND db
	con.reset();
	db.reset();

	// Query result is no longer accessible
	REQUIRE_THROWS(result->ToString());

	// And again with order preservation disabled
	db = make_uniq<DuckDB>(nullptr);
	con = make_uniq<Connection>(*db);
	result = con->SendQuery("SET preserve_insertion_order=false;");
	result = con->SendQuery("SELECT 42;", parameters);

	// Query result is accessible
	REQUIRE_NOTHROW(result->ToString());

	// Reset connection AND db
	con.reset();
	db.reset();

	// Query result is no longer accessible
	REQUIRE_THROWS(result->ToString());
}

TEST_CASE("Test ClientInterruptState suppresses interrupts after irreversible operations", "[api]") {
	// Verify the three-state interrupt mechanism that prevents a completed COMMIT
	// from being incorrectly reported as failed due to a late Interrupt() call.

	DuckDB db;
	Connection con(db);
	auto &context = *con.context;

	SECTION("Normal interrupt works") {
		context.Interrupt();
		REQUIRE(context.IsInterrupted());
		REQUIRE_THROWS(context.InterruptCheck());
		context.ClearInterrupt();
		REQUIRE(!context.IsInterrupted());
		REQUIRE_NOTHROW(context.InterruptCheck());
	}

	SECTION("SuppressInterrupts blocks subsequent Interrupt calls") {
		context.SuppressInterrupts();
		// Interrupt() uses CAS: NOT_INTERRUPTED -> INTERRUPTED
		// Since state is SUPPRESSED, CAS fails and interrupt is discarded
		context.Interrupt();
		REQUIRE(!context.IsInterrupted());
		REQUIRE_NOTHROW(context.InterruptCheck());
		context.ClearInterrupt();
	}

	SECTION("SuppressInterrupts overrides a pending interrupt") {
		context.Interrupt();
		REQUIRE(context.IsInterrupted());
		// SuppressInterrupts unconditionally stores SUPPRESSED,
		// overriding the INTERRUPTED state
		context.SuppressInterrupts();
		REQUIRE(!context.IsInterrupted());
		REQUIRE_NOTHROW(context.InterruptCheck());
		context.ClearInterrupt();
	}

	SECTION("ClearInterrupt resets from SUPPRESSED to allow future interrupts") {
		context.SuppressInterrupts();
		context.ClearInterrupt();
		// Now back to NOT_INTERRUPTED, Interrupt() should work again
		context.Interrupt();
		REQUIRE(context.IsInterrupted());
		REQUIRE_THROWS(context.InterruptCheck());
		context.ClearInterrupt();
	}

	SECTION("End-to-end: COMMIT suppresses interrupts") {
		REQUIRE_NO_FAIL(con.Query("BEGIN TRANSACTION"));
		REQUIRE_NO_FAIL(con.Query("CREATE TABLE suppress_test (x INTEGER)"));
		REQUIRE_NO_FAIL(con.Query("COMMIT"));
		// After COMMIT, state should be SUPPRESSED — Interrupt() should be discarded
		context.Interrupt();
		REQUIRE(!context.IsInterrupted());
		REQUIRE_NOTHROW(context.InterruptCheck());
		// Cleanup
		context.ClearInterrupt();
		REQUIRE_NO_FAIL(con.Query("DROP TABLE suppress_test"));
	}
}

TEST_CASE("Test ResultDB TDRoot enumerates roots and explicit sibling orders", "[api][resultdb]") {
	ResultDBEnumerationInput input;
	input.nodes = {{10, 100, false}, {20, 10, false}, {30, 1, true}};
	input.edges = {{0, 1}, {1, 2}};
	TestResultDBEnumerationCosts costs({100, 10, 1});

	auto result = ResultDBPlanEnumerator::EnumerateRoot(input, costs);
	REQUIRE(result.valid);
	// The high-cardinality unrequested endpoint is cheapest because it hashes
	// the already reduced middle relation instead of being used as a hash source.
	REQUIRE(result.root == 0);
	REQUIRE(result.order == vector<idx_t> {0, 1, 2});
	REQUIRE(result.bottom_up_children[1] == vector<idx_t> {2});
	REQUIRE(result.top_down_children[0] == vector<idx_t> {1});
	REQUIRE(result.top_down_children[1] == vector<idx_t> {2});

	ResultDBEnumerationInput star;
	star.nodes = {{1, 100, true}, {2, 50, false}, {3, 1, false}, {4, 20, false}};
	star.edges = {{0, 1}, {0, 2}, {0, 3}};
	TestResultDBEnumerationCosts star_costs({100, 50, 1, 20});
	auto star_result = ResultDBPlanEnumerator::EnumerateRoot(star, star_costs);
	REQUIRE(star_result.valid);
	if (star_result.root == 0) {
		REQUIRE(star_result.bottom_up_children[0] == vector<idx_t> {2, 3, 1});
	}
}

TEST_CASE("Test ResultDB TDFold enumerates blocks, SC candidates, and TVCs", "[api][resultdb]") {
	ResultDBEnumerationInput triangle;
	triangle.nodes = {{1, 10, true}, {2, 20, true}, {3, 30, true}};
	triangle.edges = {{0, 1}, {1, 2}, {0, 2}};
	TestResultDBEnumerationCosts triangle_costs({10, 20, 30});

	auto no_tvc = ResultDBPlanEnumerator::EnumerateFolds(triangle, triangle_costs, false);
	REQUIRE(no_tvc.valid);
	REQUIRE(no_tvc.block_sizes == vector<idx_t> {3});
	REQUIRE(no_tvc.candidate_count > 1);
	REQUIRE(!no_tvc.folds.empty());

	// Two triangles sharing an edge have the adjacent two-vertex cut {0,1}.
	ResultDBEnumerationInput shared_edge;
	shared_edge.nodes = {{1, 10, true}, {2, 10, true}, {3, 5, false}, {4, 5, false}};
	shared_edge.edges = {{0, 1}, {0, 2}, {1, 2}, {0, 3}, {1, 3}};
	TestResultDBEnumerationCosts shared_costs({10, 10, 5, 5});
	auto without_tvc = ResultDBPlanEnumerator::EnumerateFolds(shared_edge, shared_costs, false);
	auto with_tvc = ResultDBPlanEnumerator::EnumerateFolds(shared_edge, shared_costs, true);
	REQUIRE(without_tvc.valid);
	REQUIRE(with_tvc.valid);
	REQUIRE(with_tvc.used_tvc);
	REQUIRE(with_tvc.candidate_count >= without_tvc.candidate_count);
}
