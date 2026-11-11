#include "native_row_oracle_test_helpers.hpp"
#include "catch.hpp"
#include "test_helpers.hpp"
#include "duckdb/main/prepared_statement_data.hpp"
#include "duckdb/optimizer/native_row_oracle.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/planner/resultdb_reduced_plan.hpp"
#include "duckdb/main/client_config.hpp"
#include "duckdb/main/settings.hpp"
#include "duckdb/main/profiling_node.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"
#include "duckdb/function/table/table_scan.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"

#include <set>

using namespace duckdb;

TEST_CASE("Native oracle settings preserve existing configuration aliases", "[native_row_oracle]") {
	REQUIRE(!DBConfig::GetOptionByName("resultdb_cardinality_file"));
	REQUIRE(!DBConfig::GetOptionByName("resultdb_cardinality_dataset"));
	for (auto &names : vector<std::pair<string, string>> {
	         {"worker_threads", "threads"}, {"user", "username"}, {"profiling_output", "profile_output"}}) {
		auto alias = DBConfig::GetOptionByName(names.first);
		auto canonical = DBConfig::GetOptionByName(names.second);
		REQUIRE(alias);
		REQUIRE(canonical);
		REQUIRE(alias == canonical);
	}
}



TEST_CASE("Native row oracle validates immutable profile fingerprints and values", "[native_row_oracle]") {
	auto query = NativeRowOracleFile::Hash("query");
	auto schema = NativeRowOracleFile::Hash("schema");
	auto sql = "SELECT COUNT(*) FROM t";
	auto key = NativeRowOracleFile::Key(query, schema, sql);
	auto prefix = "{\"version\":1,\"database_path\":\"fixture.duckdb\",\"database_sha256\":\"" + string(64, '0') +
	              "\",\"database_size\":0,\"generator\":\"fixture\",\"entries\":[";
	auto entry = "{\"key\":\"" + key + "\",\"query_sha256\":\"" + query + "\",\"schema\":\"" + schema +
	             "\",\"sql\":\"" + sql + "\",\"cardinality\":0}";
	REQUIRE(NativeRowOracleFile::Parse(prefix + entry + "]}", "fixture")->entries.at(key).cardinality == 0);
	REQUIRE_THROWS(NativeRowOracleFile::Parse(prefix + entry + "," + entry + "]}", "fixture"));
	REQUIRE_THROWS(NativeRowOracleFile::Parse("{\"version\":1,\"version\":1}", "fixture"));
	for (auto exact : {uint64_t(9007199254740993ULL), uint64_t(54854411587723550ULL),
	                   uint64_t(9223372036854775807ULL)}) {
		auto large = entry;
		large.replace(large.find("\"cardinality\":0"), 15, "\"cardinality\":" + std::to_string(exact));
		REQUIRE(NativeRowOracleFile::Parse(prefix + large + "]}", "fixture")->entries.at(key).cardinality == exact);
	}
	for (auto invalid : {string("-1"), string("1.5"), string("9223372036854775808"), string("null")}) {
		auto bad = entry;
		bad.replace(bad.find("\"cardinality\":0"), 15, "\"cardinality\":" + invalid);
		REQUIRE_THROWS(NativeRowOracleFile::Parse(prefix + bad + "]}", "fixture"));
	}
	entry.replace(entry.find(key), key.size(), string(64, 'f'));
	REQUIRE_THROWS(NativeRowOracleFile::Parse(prefix + entry + "]}", "fixture"));
}

static bool HasWorkingAggregation(PhysicalOperator &op) {
	if (op.type == PhysicalOperatorType::HASH_GROUP_BY || op.type == PhysicalOperatorType::PERFECT_HASH_GROUP_BY ||
	    op.type == PhysicalOperatorType::UNGROUPED_AGGREGATE) {
		return true;
	}
	for (auto &child : op.children) {
		if (HasWorkingAggregation(child.get())) {
			return true;
		}
	}
	return false;
}

// Ignore unary working projections and build/probe swaps, but preserve every
// binary join split. Unique base names in this fixture identify occurrences.
static string WorkingJoinTree(PhysicalOperator &op) {
	if (op.type == PhysicalOperatorType::TABLE_SCAN) {
		auto &scan = op.Cast<PhysicalTableScan>();
		auto name = scan.bind_data->Cast<TableScanBindData>().table.name;
		return name == "a" ? "x" : name == "b" ? "y" : name == "c" ? "z" : "w";
	}
	if (op.children.size() == 1) {
		return WorkingJoinTree(op.children[0]);
	}
	REQUIRE((op.type == PhysicalOperatorType::HASH_JOIN || op.type == PhysicalOperatorType::CROSS_PRODUCT));
	REQUIRE(op.children.size() == 2);
	auto left = WorkingJoinTree(op.children[0]);
	auto right = WorkingJoinTree(op.children[1]);
	return "(" + MinValue(left, right) + "," + MaxValue(left, right) + ")";
}

static string UnorientedJoinTree(const string &tree, idx_t &pos) {
	if (tree[pos] != '(') {
		return string(1, tree[pos++]);
	}
	pos++;
	auto left = UnorientedJoinTree(tree, pos);
	REQUIRE(tree[pos++] == ',');
	auto right = UnorientedJoinTree(tree, pos);
	REQUIRE(tree[pos++] == ')');
	return "(" + MinValue(left, right) + "," + MaxValue(left, right) + ")";
}

static void CheckPreparedSchedule(const PreparedResultDBYannakakisProgram &program) {
	std::set<idx_t> completed;
	vector<idx_t> bu_position(program.relations.size(), DConstants::INVALID_INDEX);
	for (idx_t i = 0; i < program.bottom_up_steps.size(); i++) {
		auto &step = program.bottom_up_steps[i];
		REQUIRE(program.parent[step.source_relation] == step.target_relation);
		for (idx_t child = 0; child < program.relations.size(); child++) {
			if (program.parent[child] == step.source_relation) {
				REQUIRE(completed.count(child) == 1);
			}
		}
		REQUIRE(completed.insert(step.source_relation).second);
		bu_position[step.source_relation] = i;
	}
	REQUIRE(completed.size() + 1 == program.relations.size());
	if (!program.bottom_up_children.empty()) {
		for (auto &siblings : program.bottom_up_children) {
			for (idx_t i = 1; i < siblings.size(); i++) {
				REQUIRE(bu_position[siblings[i - 1]] < bu_position[siblings[i]]);
			}
		}
	}
	std::set<idx_t> required, reduced {program.root_relation};
	for (auto &output : program.outputs) {
		for (auto node = output.relation; node != program.root_relation; node = program.parent[node]) {
			required.insert(node);
		}
	}
	for (auto &step : program.top_down_steps) {
		REQUIRE(reduced.count(step.source_relation) == 1);
		REQUIRE(required.erase(step.target_relation) == 1);
		REQUIRE(program.parent[step.target_relation] == step.source_relation);
		REQUIRE(reduced.insert(step.target_relation).second);
	}
	REQUIRE(required.empty());
}

static idx_t ProfiledFiveRowFold(ProfilingNode &node) {
	auto &info = node.GetProfilingInfo();
	idx_t found = 0;
	if (info.metrics.count(MetricType::OPERATOR_NAME) &&
	    info.GetMetricValue<string>(MetricType::OPERATOR_NAME) == "HASH_JOIN") {
		// With runtime join filters disabled, this is the native A/B working
		// bag: two duplicate A rows times two B rows, plus the key-2 pair.
		REQUIRE(info.GetMetricValue<idx_t>(MetricType::OPERATOR_CARDINALITY) == 5);
		found++;
	}
	for (auto &child : node.children) {
		found += ProfiledFiveRowFold(*child);
	}
	return found;
}

TEST_CASE("Internal ResultDB plans preserve fold bags and consume exact rows", "[native_row_oracle][internal_oracle]") {
	bool optimize_internal = true;
	SECTION("Additional internal optimization enabled") {
	}
	SECTION("Additional internal optimization disabled") {
		optimize_internal = false;
	}
	auto database = TestCreatePath("internal_resultdb_oracle.duckdb");
	auto profile = TestCreatePath("internal_resultdb_oracle.json");
	DeleteDatabase(database);
	{
		DuckDB writable(database);
		Connection create(writable);
		REQUIRE_NO_FAIL(create.Query(
		    "CREATE TABLE a(s INTEGER,d INTEGER,p VARCHAR); CREATE TABLE b(s BIGINT,d INTEGER); "
		    "CREATE TABLE c(s INTEGER,d INTEGER); "
		    "INSERT INTO a VALUES (1,1,'x'),(1,1,'x'),(2,1,'y'),(NULL,1,'n'); "
		    "INSERT INTO b VALUES (1,1),(1,1),(2,1),(NULL,1); "
		    "INSERT INTO c VALUES (1,1),(1,1),(NULL,1); "
		    "CREATE TABLE b_empty(s BIGINT,d INTEGER); INSERT INTO b_empty VALUES (9,1); CHECKPOINT"));
	}
	DBConfig config;
	config.options.access_mode = AccessMode::READ_ONLY;
	DuckDB db(database, &config);
	Connection con(db), counter(db);
	REQUIRE(Settings::Get<ResultdbOptimizeInternalPlansSetting>(*con.context));
	REQUIRE_NO_FAIL(con.Query(string("SET resultdb_optimize_internal_plans=") +
	                         (optimize_internal ? "true" : "false")));
	REQUIRE_NO_FAIL(con.Query("SET disabled_optimizers='join_filter_pushdown'"));
	vector<string> queries {
	    "SELECT RESULTDB x.s,x.p,y.s,z.s FROM a x,b y,c z WHERE CAST(x.s AS BIGINT)=y.s AND x.d=z.s AND y.d=z.d",
	    "SELECT RESULTDB x.s,y.s FROM a x,b y WHERE CAST(x.s AS BIGINT)=y.s AND x.s=1",
	    "SELECT RESULTDB x.s,y.s FROM a x,a y WHERE x.s=y.s AND x.s=99",
	    "SELECT RESULTDB x.s FROM a x WHERE x.s IS NULL",
	    "SELECT RESULTDB x.s,y.s,z.s FROM a x,b_empty y,c z WHERE CAST(x.s AS BIGINT)=y.s AND x.d=z.s AND y.d=z.d",
	    "SELECT RESULTDB x.s FROM a x,b y,c z WHERE CAST(x.s AS BIGINT)=y.s AND y.s=z.s"};
	vector<string> strategies {"semijoin", "tdfold", "tdfold_no_tvc", "auto", "decompose"};
	REQUIRE_NO_FAIL(con.Query("SET resultdb_enumeration_limit=3000000; "
	                          "SET optimizer_statistics_discovery=true"));
	vector<shared_ptr<NativeRowOracle>> audits;
	bool converged = false;
	for (idx_t pass = 0; pass < 12; pass++) {
		idx_t missing = 0;
		for (auto &strategy : strategies) {
			REQUIRE_NO_FAIL(con.Query("SET resultdb_strategy='" + strategy + "'"));
			for (idx_t q = 0; q < queries.size(); q++) {
				// Direct execution requires a connected multi-relation graph.
				if (q == 3 && strategy != "decompose") { continue; }
				auto prepared = con.Prepare(queries[q]);
				INFO(strategy + ": " + (prepared->HasError() ? prepared->GetError() : queries[q]));
				REQUIRE(!prepared->HasError());
				auto audit = prepared->data->properties.native_row_oracle;
				REQUIRE(audit);
				missing += audit->missing;
				audits.push_back(audit);
				REQUIRE(prepared->Execute()->HasError());
			}
		}
		REQUIRE_NO_FAIL(con.Query("SET resultdb_strategy='tdroot'"));
		for (auto q : {idx_t(1), idx_t(5)}) {
			auto tree = con.Prepare(queries[q]);
			INFO((tree->HasError() ? tree->GetError() : "tdroot"));
			REQUIRE(!tree->HasError());
			audits.push_back(tree->data->properties.native_row_oracle);
			missing += audits.back()->missing;
		}
		NativeWrite(profile, NativeProfile(database, audits, counter));
		REQUIRE_NO_FAIL(con.Query("SET optimizer_statistics_file='" + profile + "'"));
		if (!missing) { converged = true; break; }
	}
	REQUIRE(converged);
	REQUIRE_NO_FAIL(con.Query("SET optimizer_statistics_discovery=false; SET optimizer_statistics_mode='oracle'"));
	idx_t working_distinct = 0, build_probe = 0, final_physical = 0;
	string required_fold_bag_key;
	for (auto &strategy : strategies) {
		REQUIRE_NO_FAIL(con.Query("SET resultdb_strategy='" + strategy + "'"));
		for (idx_t q = 0; q < queries.size(); q++) {
			if (q == 3 && strategy != "decompose") { continue; }
			if (q == 0 && strategy == "semijoin") {
				con.EnableProfiling();
				con.context->config.emit_profiler_output = false;
			}
			auto prepared = con.Prepare(queries[q]);
			INFO(strategy + ": " + (prepared->HasError() ? prepared->GetError() : queries[q]));
			REQUIRE(!prepared->HasError());
			auto audit = prepared->data->properties.native_row_oracle;
			REQUIRE(audit->missing == 0);
			auto &chosen = prepared->data->properties.resultdb;
			REQUIRE((!chosen.tvc_used || chosen.tvc_enabled));
			if (chosen.execution_strategy == ResultDBExecutionStrategy::DECOMPOSE || strategy == "tdfold_no_tvc") {
				REQUIRE(!chosen.tvc_used);
			}
			REQUIRE(!ClientConfig::GetConfig(*con.context).native_oracle_active);
			if ((q == 0 || q == 5) && prepared->data->resultdb_yannakakis_program) {
				auto &program = *prepared->data->resultdb_yannakakis_program;
				CheckPreparedSchedule(program);
				for (auto &base : prepared->data->resultdb_yannakakis_program->base_plans) {
					REQUIRE(!HasWorkingAggregation(base->Root()));
				}
				if (strategy == "tdfold" || strategy == "tdfold_no_tvc") {
					auto descriptions = StringUtil::Split(prepared->data->properties.resultdb.selected_join_tree, ';');
					REQUIRE(descriptions.size() == program.base_plans.size());
					for (idx_t i = 0; i < descriptions.size(); i++) {
						idx_t position = 0;
						REQUIRE(WorkingJoinTree(program.base_plans[i]->Root()) ==
						        UnorientedJoinTree(descriptions[i], position));
						REQUIRE(position == descriptions[i].size());
					}
				}
			}
			if (q == 0 && strategy == "semijoin") {
				auto &program = *prepared->data->resultdb_yannakakis_program;
				bool checked_fold = false;
				for (idx_t i = 0; i < program.relations.size(); i++) {
					if (program.relations[i].table_indices.size() != 2) { continue; }
					std::set<string> names;
					for (auto table : program.relations[i].table_indices) {
						names.insert(prepared->data->properties.resultdb.occurrence_aliases.at(table));
					}
					REQUIRE(names == std::set<string> {"x", "y"});
					// Independently: the A/B join retains 2*2+1=5 bag rows,
					// even though its working projection has only two distinct tuples.
					REQUIRE(program.base_plans[i]->Root().estimated_cardinality == 5);
					auto phase = "internal[" + std::to_string(i) + "]";
					for (auto &item : audit->requests) {
						auto &request = item.second;
						if (request.stage.find(phase) != string::npos &&
						    request.stage.find("physical_input") != string::npos && request.cardinality == 5) {
							required_fold_bag_key = item.first;
						}
					}
					checked_fold = true;
				}
				REQUIRE(checked_fold);
			}
			for (auto &item : audit->requests) {
				auto &request = item.second;
				REQUIRE(request.found);
				auto counted = counter.Query(request.sql);
				REQUIRE_NO_FAIL(*counted);
				REQUIRE(counted->GetValue(0, 0).GetValue<uint64_t>() == request.cardinality);
				if (request.stage.find("internal[") != string::npos) {
					working_distinct += request.sql.find("SELECT DISTINCT") != string::npos;
					build_probe += request.stage.find("before_build_probe") != string::npos;
					final_physical += request.stage.find("physical_input") != string::npos;
				}
			}
			REQUIRE_NO_FAIL(counter.Query("SET resultdb_strategy='decompose'"));
			auto expected = counter.Query(queries[q]);
			auto actual = prepared->Execute();
			REQUIRE_NO_FAIL(*actual); REQUIRE_NO_FAIL(*expected);
			NativeCompare(*actual, *expected);
			if (q == 0 && strategy == "semijoin") {
				REQUIRE(con.GetProfilingTree());
				REQUIRE(ProfiledFiveRowFold(*con.GetProfilingTree()) == 1);
				con.DisableProfiling();
			}
		}
	}
	REQUIRE_NO_FAIL(con.Query("SET resultdb_strategy='TD_root'"));
	auto tree = con.Prepare(queries[5]);
	REQUIRE(!tree->HasError());
	CheckPreparedSchedule(*tree->data->resultdb_yannakakis_program);
	auto tree_result = tree->Execute();
	auto tree_expected = counter.Query(queries[5]);
	REQUIRE_NO_FAIL(*tree_result);
	REQUIRE_NO_FAIL(*tree_expected);
	NativeCompare(*tree_result, *tree_expected);
	REQUIRE(working_distinct == 0);
	REQUIRE(!required_fold_bag_key.empty());
	REQUIRE((optimize_internal ? build_probe > 0 : build_probe == 0));
	REQUIRE(final_physical > 0);
	REQUIRE_NO_FAIL(con.Query("SET resultdb_strategy='semijoin'"));
	auto saved = con.Prepare(queries[0]);
	REQUIRE(!saved->HasError());
	auto old_file = saved->data->properties.native_row_oracle->file;
	auto old_root = saved->data->properties.resultdb.selected_root_table_index;
	auto old_folds = saved->data->properties.resultdb.selected_folds;
	NativeWrite(profile, NativeProfile(database, audits, counter, true));
	REQUIRE_NO_FAIL(con.Query("SET optimizer_statistics_file='" + profile + "'"));
	auto zero = con.Prepare(queries[0]);
	INFO((zero->HasError() ? zero->GetError() : "zero heuristic"));
	REQUIRE(!zero->HasError());
	REQUIRE(zero->data->properties.resultdb.selected_root_table_index == old_root);
	REQUIRE(zero->data->properties.resultdb.selected_folds == old_folds);
	auto zero_result = zero->Execute(); auto zero_expected = counter.Query(queries[0]);
	REQUIRE_NO_FAIL(*zero_result); REQUIRE_NO_FAIL(*zero_expected);
	NativeCompare(*zero_result, *zero_expected);
	REQUIRE_NO_FAIL(con.Query("RESET optimizer_statistics_file; SET optimizer_statistics_mode='estimated'"));
	saved->data->properties.always_require_rebind = true;
	auto rebound = saved->Execute(); auto rebound_expected = counter.Query(queries[0]);
	REQUIRE_NO_FAIL(*rebound); REQUIRE_NO_FAIL(*rebound_expected);
	NativeCompare(*rebound, *rebound_expected);
	REQUIRE(saved->data->properties.native_row_oracle->file == old_file);
	REQUIRE(Settings::Get<OptimizerStatisticsModeSetting>(*con.context) == "estimated");
	// A missing internal count must fail strictly and must not leak its adapter.
	NativeWrite(profile, NativeProfile(database, audits, counter, false, required_fold_bag_key));
	REQUIRE_NO_FAIL(con.Query("SET optimizer_statistics_file='" + profile + "'; SET optimizer_statistics_mode='oracle'; SET resultdb_strategy='semijoin'"));
	auto missing = con.Prepare(queries[0]);
	REQUIRE(missing->HasError());
	REQUIRE(missing->GetError().find("Missing native row cardinality") != string::npos);
	REQUIRE(missing->GetError().find(required_fold_bag_key) != string::npos);
	REQUIRE(missing->GetError().find("internal[") != string::npos);
	REQUIRE(!ClientConfig::GetConfig(*con.context).native_oracle_active);
}

TEST_CASE("Paper four-relation folds execute the costed join splits", "[native_row_oracle][internal_oracle]") {
	auto database = TestCreatePath("native_four_relation_fold.duckdb");
	auto profile = TestCreatePath("native_four_relation_fold.json");
	DeleteDatabase(database);
	{
		DuckDB writable(database);
		Connection create(writable);
		REQUIRE_NO_FAIL(create.Query("CREATE TABLE a(k INTEGER); CREATE TABLE b(k INTEGER); "
		                             "CREATE TABLE c(k INTEGER); CREATE TABLE d(k INTEGER); "
		                             "INSERT INTO a VALUES (1); INSERT INTO b VALUES (1); "
		                             "INSERT INTO c VALUES (1); INSERT INTO d VALUES (1); CHECKPOINT"));
	}
	DBConfig config;
	config.options.access_mode = AccessMode::READ_ONLY;
	DuckDB db(database, &config);
	Connection con(db), counter(db);
	string query = "SELECT RESULTDB x.k,y.k,z.k FROM a x,b y,c z,d w "
	               "WHERE x.k=y.k AND x.k=z.k AND x.k=w.k AND y.k=z.k AND y.k=w.k AND z.k=w.k";
	SealResultDBOracle(con, counter, database, profile, {query}, false, {"TD_fold", "TD_fold_no_tvc"});
	for (auto strategy : {"TD_fold", "TD_fold_no_tvc"}) {
		REQUIRE_NO_FAIL(con.Query(string("SET resultdb_strategy='") + strategy + "'"));
		auto prepared = con.Prepare(query);
		INFO((prepared->HasError() ? prepared->GetError() : query));
		REQUIRE(!prepared->HasError());
		auto &properties = prepared->data->properties.resultdb;
		// Independently: full folding costs 3 join rows + 3 output occurrences.
		// Every split adds Algorithm 3 neighbor work, so full folding wins.
		REQUIRE(properties.selected_folds.size() == 1);
		REQUIRE(properties.selected_folds[0].size() == 4);
		REQUIRE(properties.estimated_semijoin_cost == 6);
		auto &program = *prepared->data->resultdb_yannakakis_program;
		REQUIRE(program.base_plans.size() == 1);
		idx_t position = 0;
		REQUIRE(WorkingJoinTree(program.base_plans[0]->Root()) ==
		        UnorientedJoinTree(properties.selected_join_tree, position));
		REQUIRE(position == properties.selected_join_tree.size());
		auto actual = prepared->Execute();
		auto expected = counter.Query(query);
		REQUIRE_NO_FAIL(*actual);
		REQUIRE_NO_FAIL(*expected);
		NativeCompare(*actual, *expected);
	}
}

TEST_CASE("Native oracle handles repeated subplans or rejects their mappings explicitly", "[native_row_oracle]") {
	auto database = TestCreatePath("native_repeated_subplans.duckdb");
	auto profile = TestCreatePath("native_repeated_subplans.json");
	DeleteDatabase(database);
	{
		DuckDB writable(database);
		Connection create(writable);
		REQUIRE_NO_FAIL(create.Query("CREATE TABLE a(s INTEGER); CREATE TABLE b(s INTEGER); "
		                             "INSERT INTO a VALUES (1),(1),(2),(NULL); "
		                             "INSERT INTO b VALUES (1),(1),(2),(NULL); CHECKPOINT"));
	}
	DBConfig config;
	config.options.access_mode = AccessMode::READ_ONLY;
	DuckDB db(database, &config);
	Connection con(db), counter(db);
	vector<string> queries {"SELECT x.s,y.s FROM (SELECT a.s FROM a,b WHERE a.s=b.s) x, "
	                        "(SELECT a.s FROM a,b WHERE a.s=b.s) y WHERE x.s=y.s",
	                        "WITH ab AS MATERIALIZED (SELECT a.s FROM a,b WHERE a.s=b.s) "
	                        "SELECT x.s,y.s FROM ab x,ab y WHERE x.s=y.s"};
	for (auto &query : queries) {
		auto expected = counter.Query(query);
		REQUIRE_NO_FAIL(*expected);
		// The two subplans each produce four key-1 rows and one key-2 row.
		REQUIRE(expected->RowCount() == 17);
		REQUIRE_NO_FAIL(con.Query("RESET optimizer_statistics_file; SET optimizer_statistics_mode='estimated'; "
		                          "SET optimizer_statistics_discovery=true"));
		auto discovered = con.Prepare(query);
		if (discovered->HasError()) {
			// Unsupported derived bindings/CTEs must fail in discovery too;
			// they cannot silently enter a strict experiment with estimates.
			REQUIRE(discovered->GetError().find("Native row oracle:") != string::npos);
			REQUIRE(!ClientConfig::GetConfig(*con.context).native_oracle_active);
			REQUIRE_NO_FAIL(
			    con.Query("SET optimizer_statistics_discovery=false; SET optimizer_statistics_mode='oracle'"));
			REQUIRE(con.Prepare(query)->HasError());
		} else {
			REQUIRE(discovered->Execute()->HasError());
			SealResultDBOracle(con, counter, database, profile, {query}, false, {"Plain"});
			auto strict = con.Prepare(query);
			INFO((strict->HasError() ? strict->GetError() : query));
			REQUIRE(!strict->HasError());
			REQUIRE(strict->data->properties.native_row_oracle->missing == 0);
			auto actual = strict->Execute();
			REQUIRE_NO_FAIL(*actual);
			NativeCompare(*actual, *expected);
		}
	}
}

TEST_CASE("ResultDB heuristic root uses Mutable index ties without cardinality inputs", "[native_row_oracle]") {
	DuckDB db(nullptr);
	Connection con(db), counter(db);
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE a(k INTEGER); INSERT INTO a VALUES (1),(1),(2),(NULL); "
	                          "CREATE TABLE b(k BIGINT); INSERT INTO b VALUES (1),(2),(NULL); "
	                          "CREATE TABLE c(k INTEGER); INSERT INTO c VALUES (1)"));
	REQUIRE_NO_FAIL(con.Query("SET resultdb_strategy='semijoin'"));
	vector<string> queries {
	    "SELECT RESULTDB x.k,z.k FROM a x,b y,c z WHERE CAST(x.k AS BIGINT)=y.k AND y.k=z.k",
	    "SELECT RESULTDB x.k,z.k FROM c z,b y,a x WHERE CAST(x.k AS BIGINT)=y.k AND y.k=z.k",
	    "SELECT RESULTDB x.k,z.k FROM a x,a y,c z WHERE x.k=y.k AND y.k=z.k AND x.k=99"};
	for (idx_t i = 0; i < queries.size(); i++) {
		auto prepared = con.Prepare(queries[i]);
		REQUIRE(!prepared->HasError());
		REQUIRE(!prepared->data->properties.native_row_oracle);
		auto &properties = prepared->data->properties.resultdb;
		REQUIRE(properties.selected_folds.empty());
		if (i < 2) {
			// Requested endpoints have equal degree. Keep the first occurrence,
			// including when its four input rows exceed the other endpoint's one.
			REQUIRE(properties.occurrence_aliases.at(properties.selected_root_table_index) == (i == 0 ? "x" : "z"));
		}
		auto actual = prepared->Execute();
		auto expected = counter.Query(queries[i]);
		REQUIRE_NO_FAIL(*actual);
		REQUIRE_NO_FAIL(*expected);
		NativeCompare(*actual, *expected);
	}
	REQUIRE_NO_FAIL(con.Query("SET optimizer_statistics_mode='oracle'"));
	auto rejected = con.Prepare(queries[0]);
	REQUIRE(rejected->HasError());
	REQUIRE(rejected->GetError().find("optimizer_statistics_file") != string::npos);
}

TEST_CASE("Heuristic folds use Mutable index ties instead of smaller oracle inputs", "[native_row_oracle]") {
	auto database = TestCreatePath("native_heuristic_fold_ties.duckdb");
	DeleteDatabase(database);
	{
		DuckDB writable(database);
		Connection create(writable);
		REQUIRE_NO_FAIL(create.Query(
		    "CREATE TABLE a(s INTEGER, d INTEGER); CREATE TABLE b(s INTEGER, d INTEGER); "
		    "CREATE TABLE c(s INTEGER, d INTEGER); CREATE TABLE d(s INTEGER, d INTEGER); "
		    "CREATE TABLE e(s INTEGER, d INTEGER); CREATE TABLE f(s INTEGER, d INTEGER); "
		    "INSERT INTO a VALUES (1,1),(2,2),(3,3),(4,4),(5,5),(6,6); INSERT INTO b VALUES (1,1); "
		    "INSERT INTO c VALUES (1,1),(2,1); INSERT INTO d VALUES (1,1),(2,1); "
		    "INSERT INTO e VALUES (1,1),(2,2),(3,3); INSERT INTO f VALUES (1,1),(2,2),(3,3); CHECKPOINT"));
	}
	DBConfig config;
	config.options.access_mode = AccessMode::READ_ONLY;
	DuckDB db(database, &config);
	Connection con(db), counter(db);
	vector<string> queries;
	for (auto &from : vector<string> {"a a,b b,c c,d d,e e,f f", "b b,a a,c c,d d,e e,f f",
	                                 "d d,a a,b b,c c,e e,f f"}) {
		queries.push_back("SELECT RESULTDB a.s,a.d,b.s,b.d,c.s,c.d,d.s,d.d,e.s,e.d,f.s,f.d FROM " + from +
		                  " WHERE a.s=b.s AND a.d=c.s AND b.d=c.d AND c.d=d.s AND d.s=e.s "
		                  "AND d.d=f.s AND e.d=f.d");
	}
	REQUIRE_NO_FAIL(con.Query("SET resultdb_strategy='semijoin'"));
	// Reference choices for the two triangles linked by c-d. The first query
	// must take a despite its six rows versus b's one; reordering FROM changes
	// the first-index choice, while merging a leading d must retain its rank.
	vector<std::set<string>> expected {{"a", "c", "d", "e"}, {"b", "c", "d", "e"}, {"a", "c", "d", "e"}};
	for (idx_t i = 0; i < queries.size(); i++) {
		auto prepared = con.Prepare(queries[i]);
		INFO((prepared->HasError() ? prepared->GetError() : queries[i]));
		REQUIRE(!prepared->HasError());
		auto &properties = prepared->data->properties.resultdb;
		REQUIRE(properties.selected_folds.size() == 1);
		std::set<string> selected;
		for (auto index : properties.selected_folds[0]) {
			selected.insert(properties.occurrence_aliases.at(index));
		}
		REQUIRE(selected == expected[i]);
		REQUIRE(!prepared->data->properties.native_row_oracle);
		auto actual = prepared->Execute();
		auto expected_result = counter.Query(queries[i]);
		REQUIRE_NO_FAIL(*actual);
		REQUIRE_NO_FAIL(*expected_result);
		NativeCompare(*actual, *expected_result);
	}
}

TEST_CASE("Native row oracle consumes exact rows with native planning and preserves results", "[native_row_oracle]") {
	auto database = TestCreatePath("native_rows.duckdb");
	auto profile = TestCreatePath("native_rows.json");
	DeleteDatabase(database);
	{
		DuckDB writable(database);
		Connection create(writable);
		REQUIRE_NO_FAIL(create.Query("CREATE TABLE a(id INTEGER, v VARCHAR); CREATE TABLE b(id BIGINT, flag INTEGER);"
		                             "CREATE TABLE c(id INTEGER); INSERT INTO a VALUES (1,'x'),(1,'x'),(2,NULL),(3,'z');"
		                             "INSERT INTO b VALUES (1,1),(1,2),(2,1),(9,1); INSERT INTO c VALUES (1),(2),(8); CHECKPOINT"));
	}
	DBConfig config;
	config.options.access_mode = AccessMode::READ_ONLY;
	DuckDB db(database, &config);
	Connection con(db), counter(db);
	vector<string> queries {
	    "SELECT x.id,x.v FROM a x,b y,c z WHERE CAST(x.id AS BIGINT)=y.id AND x.id=z.id AND (y.flag=1 OR x.v IS NULL)",
	    "SELECT RESULTDB x.id,x.v,y.flag FROM a x,b y,c z WHERE CAST(x.id AS BIGINT)=y.id AND x.id=z.id AND (y.flag=1 OR x.v IS NULL)",
	    "SELECT x.id,y.id FROM a x,a y WHERE x.id=y.id AND x.v='x'",
	    "SELECT id FROM a WHERE v IS NULL",
	    "SELECT x.id FROM a x,b y WHERE x.id=y.id AND y.flag=42",
	    "SELECT x.id FROM a x,b y WHERE x.id=y.id AND x.v IN ('x','z','a','b','c',NULL)",
	    "SELECT x.id FROM a x,b y WHERE x.id=y.id AND x.v NOT IN ('z','a','b','c','d',NULL)",
	    "SELECT RESULTDB x.id,x.v,y.flag FROM a x,b y WHERE x.id=y.id AND x.v IN ('x','z','a','b','c',NULL)"};
	vector<unique_ptr<MaterializedQueryResult>> expected;
	for (auto &query : queries) {
		auto result = counter.Query(query);
		REQUIRE_NO_FAIL(*result);
		expected.push_back(std::move(result));
	}
	REQUIRE_NO_FAIL(con.Query("SET optimizer_statistics_discovery=true"));
	vector<shared_ptr<NativeRowOracle>> audits;
	for (auto &query : queries) {
		auto prepared = con.Prepare(query);
		INFO((prepared->HasError() ? prepared->GetError() : string()));
		REQUIRE(!prepared->HasError());
		auto audit = prepared->data->properties.native_row_oracle;
		REQUIRE(audit);
		REQUIRE(audit->requests.size() > 0);
		REQUIRE(prepared->Execute()->HasError());
		audits.push_back(std::move(audit));
	}
	NativeWrite(profile, NativeProfile(database, audits, counter));
	REQUIRE_NO_FAIL(con.Query("SET optimizer_statistics_discovery=false; SET optimizer_statistics_file='" + profile +
	                         "'; SET optimizer_statistics_mode='oracle'"));
	// Exact estimates can reveal different native candidate subsets: extend offline coverage until stable.
	for (idx_t round = 0; round < 4; round++) {
		REQUIRE_NO_FAIL(con.Query("SET optimizer_statistics_discovery=true"));
		for (auto &query : queries) {
			auto prepared = con.Prepare(query);
			INFO((prepared->HasError() ? prepared->GetError() : string()));
		REQUIRE(!prepared->HasError());
			audits.push_back(prepared->data->properties.native_row_oracle);
		}
		NativeWrite(profile, NativeProfile(database, audits, counter));
		REQUIRE_NO_FAIL(con.Query("SET optimizer_statistics_file='" + profile + "'; SET optimizer_statistics_discovery=false"));
	}
	for (idx_t i = 0; i < queries.size(); i++) {
		auto prepared = con.Prepare(queries[i]);
		INFO((prepared->HasError() ? prepared->GetError() : string()));
		REQUIRE(!prepared->HasError());
		auto audit = prepared->data->properties.native_row_oracle;
		REQUIRE(audit->missing == 0);
		for (auto &item : audit->requests) {
			auto count = counter.Query(item.second.sql);
			REQUIRE_NO_FAIL(*count);
			REQUIRE(item.second.cardinality == count->GetValue(0, 0).GetValue<idx_t>());
		}
		if (i < 3) { REQUIRE(audit->native_join_searches > 0); }
		if (i == 0) {
			// Independent tuple arithmetic: two copies of (1,'x') and one (2,NULL)
			// survive both joins and the OR predicate. The full bag has three rows.
			bool checked_root = false;
			for (auto &item : audit->requests) {
				if (item.second.stage.find("PROJECTION") != string::npos) {
					REQUIRE(item.second.cardinality == 3);
					checked_root = true;
				}
			}
			REQUIRE(checked_root);
		}
		auto actual = prepared->Execute();
		REQUIRE_NO_FAIL(*actual);
		REQUIRE(actual->names == expected[i]->names);
		REQUIRE(actual->types == expected[i]->types);
		NativeCompare(*actual, *expected[i]);
	}
	// Independently specified DISTINCT outputs, rather than another ResultDB run.
	auto decomposed = con.Query(queries[1]);
	REQUIRE_NO_FAIL(*decomposed);
	auto expected_x = counter.Query("SELECT * FROM (VALUES (1,'x'),(2,NULL)) v(id,v)");
	auto expected_y = counter.Query("SELECT 1::INTEGER AS flag");
	REQUIRE(decomposed->properties.resultdb.tables[0].name == "x");
	REQUIRE(decomposed->next);
	REQUIRE(decomposed->next->properties.resultdb.tables[0].name == "y");
	REQUIRE(decomposed->names == expected_x->names);
	REQUIRE(decomposed->types == expected_x->types);
	REQUIRE(NativeRows(*decomposed) == NativeRows(*expected_x));
	REQUIRE(decomposed->next->names == expected_y->names);
	REQUIRE(decomposed->next->types == expected_y->types);
	REQUIRE(NativeRows(*decomposed->next) == NativeRows(*expected_y));
	REQUIRE(!decomposed->next->next);
	// A prepared plan retains its file and mode after subsequent configuration changes.
	auto saved = con.Prepare(queries[0]);
	REQUIRE(!saved->HasError());
	auto saved_file = saved->data->properties.native_row_oracle->file;
	REQUIRE_NO_FAIL(con.Query("RESET optimizer_statistics_file; SET optimizer_statistics_mode='estimated'"));
	REQUIRE(saved->data->properties.native_row_oracle->file == saved_file);
	// Exercise the snapshot guard even when execution has to rebind the SQL.
	saved->data->properties.always_require_rebind = true;
	REQUIRE_NO_FAIL(*saved->Execute());
	REQUIRE(saved->data->properties.native_row_oracle->file == saved_file);
	REQUIRE(saved->data->properties.native_statistics_mode == "oracle");
	REQUIRE(!con.Prepare(queries[0])->data->properties.native_row_oracle);
	REQUIRE_NO_FAIL(con.Query("SET optimizer_statistics_mode='oracle'"));
	REQUIRE(con.Prepare(queries[0])->HasError());
	REQUIRE_NO_FAIL(con.Query("SET optimizer_statistics_mode='estimated'"));
	// All-zero input remains advisory: it must never change actual rows.
	NativeWrite(profile, NativeProfile(database, audits, counter, true));
	REQUIRE_NO_FAIL(con.Query("SET optimizer_statistics_file='" + profile + "'; SET optimizer_statistics_mode='oracle'; SET optimizer_statistics_discovery=true"));
	for (idx_t round = 0; round < 4; round++) {
		auto prepared = con.Prepare(queries[0]);
		INFO((prepared->HasError() ? prepared->GetError() : string()));
		REQUIRE(!prepared->HasError());
		audits.push_back(prepared->data->properties.native_row_oracle);
		NativeWrite(profile, NativeProfile(database, audits, counter, true));
		REQUIRE_NO_FAIL(con.Query("SET optimizer_statistics_file='" + profile + "'"));
	}
	REQUIRE_NO_FAIL(con.Query("SET optimizer_statistics_discovery=false"));
	auto zero = con.Prepare(queries[0]);
	REQUIRE(!zero->HasError());
	auto actual = zero->Execute();
	REQUIRE_NO_FAIL(*actual);
	auto reference = counter.Query(queries[0]);
	NativeCompare(*actual, *reference);
	// Disabling the optimizer cannot turn strict oracle mode into estimated execution.
	REQUIRE_NO_FAIL(con.Query("SET debug_disable_optimizer=true"));
	REQUIRE(con.Prepare(queries[0])->HasError());
	REQUIRE_NO_FAIL(con.Query("RESET debug_disable_optimizer"));
	// Changing query predicates cannot silently consume a different query's profile.
	REQUIRE(con.Prepare("SELECT id FROM a WHERE v='different'")->HasError());
}

TEST_CASE("ResultDB AUTO compares native-oracle joins with TDFold on optimized inputs",
          "[native_row_oracle][resultdb_paper][resultdb_auto_native]") {
	auto database = TestCreatePath("resultdb_auto_native.duckdb");
	auto profile = TestCreatePath("resultdb_auto_native.json");
	DeleteDatabase(database);
	{
		DuckDB writable(database);
		Connection create(writable);
		REQUIRE_NO_FAIL(create.Query(
		    "CREATE TABLE narrow_a(k BOOLEAN); CREATE TABLE narrow_b(k BOOLEAN);"
		    "INSERT INTO narrow_a VALUES (TRUE),(FALSE); INSERT INTO narrow_b VALUES (TRUE),(FALSE);"
		    "CREATE TABLE wide_a(k INTEGER, payload VARCHAR); CREATE TABLE wide_b(k INTEGER, payload VARCHAR);"
		    "INSERT INTO wide_a SELECT 1,'a'||i::VARCHAR FROM range(10) r(i);"
		    "INSERT INTO wide_b SELECT 1,'b'||i::VARCHAR FROM range(10) r(i);"
		    "CREATE TABLE t(k INTEGER, note VARCHAR); INSERT INTO t VALUES (7,'x'),(7,'x'),(8,NULL),(NULL,'n');"
		    "INSERT INTO t SELECT i,'other' FROM range(100,200) r(i); CHECKPOINT"));
	}
	DBConfig config;
	config.options.access_mode = AccessMode::READ_ONLY;
	DuckDB db(database, &config);
	Connection con(db), counter(db);
	vector<string> queries {
	    "SELECT RESULTDB * FROM narrow_a a JOIN narrow_b b ON a.k=b.k",
	    "SELECT RESULTDB * FROM wide_a a JOIN wide_b b ON a.k=b.k",
	    "SELECT RESULTDB a.k,a.note,c.k FROM t a,t b,t c WHERE a.k=b.k AND b.k=c.k AND a.k=7",
	    "SELECT RESULTDB a.k,a.note,b.k FROM t a,t b,t c WHERE CAST(a.k AS BIGINT)=CAST(b.k AS BIGINT) "
	    "AND b.k=c.k AND a.k=c.k AND (a.note='x' OR a.note IS NULL)",
	    "SELECT RESULTDB a.k,b.k FROM t a,t b WHERE a.k=b.k AND a.k=1234567"};
	SealResultDBOracle(con, counter, database, profile, queries);
	for (idx_t i = 0; i < queries.size(); i++) {
		REQUIRE_NO_FAIL(con.Query("SET resultdb_strategy='auto'"));
		auto prepared = con.Prepare(queries[i]);
		INFO((prepared->HasError() ? prepared->GetError() : queries[i]));
		REQUIRE(!prepared->HasError());
		auto &properties = prepared->data->properties.resultdb;
		auto audit = prepared->data->properties.native_row_oracle;
		REQUIRE(audit->missing == 0);
		REQUIRE(properties.cardinality_dataset == "optimized-inputs");
		for (auto &item : audit->requests) {
			auto count = counter.Query(item.second.sql);
			REQUIRE_NO_FAIL(*count);
			REQUIRE(item.second.cardinality == count->GetValue(0, 0).GetValue<idx_t>());
		}
		if (i < 4) {
			REQUIRE(audit->native_join_searches > 0);
			REQUIRE(properties.planning_reason.find("after native join ordering") != string::npos);
		}
		if (i == 0) {
			// Two singleton joins: bag=2, join cost=2, two outputs cost=4.
			// Two semijoins each cost 2 + 3*2 = 8.
			REQUIRE(properties.estimated_decompose_cost == 6);
			REQUIRE(properties.estimated_semijoin_cost == 16);
			REQUIRE(properties.execution_strategy == ResultDBExecutionStrategy::DECOMPOSE);
			REQUIRE_NO_FAIL(con.Query("SET resultdb_strategy='decompose'"));
			auto native = con.Prepare(queries[i]);
			REQUIRE(!native->HasError());
			REQUIRE(prepared->data->physical_plan->Root().ToString() == native->data->physical_plan->Root().ToString());
		}
		if (i == 1) {
			// Ten rows per side, all matching: bag=100, join+decompose=300.
			// Survivor bags are ten each: two semijoins cost 40+40.
			REQUIRE(properties.estimated_decompose_cost == 300);
			REQUIRE(properties.estimated_semijoin_cost == 80);
			REQUIRE(properties.execution_strategy == ResultDBExecutionStrategy::SEMIJOIN);
			REQUIRE(prepared->data->resultdb_yannakakis_program);
		}
		if (i == 2) {
			REQUIRE(properties.join_edges.size() == 2);
			REQUIRE(properties.estimated_decompose_cost == 28); // Join bags 4+8, two outputs from bag 8.
			REQUIRE(properties.estimated_semijoin_cost == 32); // Four reductions, each 2+3*2.
			idx_t optimized_singletons = 0;
			for (auto &item : audit->requests) {
				auto &request = item.second;
				if (request.stage.find("resultdb_join_bag") != string::npos &&
				    request.sql.find(", ") == string::npos && request.sql.find("EXISTS") == string::npos) {
					REQUIRE(request.cardinality == 2); // Each occurrence receives k=7, rather than raw 104.
					optimized_singletons++;
				}
			}
			REQUIRE(optimized_singletons == 3);
		}
		if (i == 3) {
			REQUIRE(properties.join_edges.size() == 3); // Original cycle remains authoritative.
		}
		auto actual = prepared->Execute();
		REQUIRE_NO_FAIL(*actual);
		auto expected = counter.Query(queries[i]);
		REQUIRE_NO_FAIL(*expected);
		NativeCompare(*actual, *expected);
		for (auto strategy : {"tdroot", "tdfold_no_tvc", "tdfold"}) {
			if (i == 3 && string(strategy) == "tdroot") {
				continue; // TDRoot is defined only for the original acyclic graph.
			}
			REQUIRE_NO_FAIL(con.Query(string("SET resultdb_strategy='") + strategy + "'"));
			auto direct = con.Query(queries[i]);
			INFO((direct->HasError() ? direct->GetError() : queries[i]));
			REQUIRE_NO_FAIL(*direct);
			auto reference = counter.Query(queries[i]);
			NativeCompare(*direct, *reference);
		}
	}
	// A genuine tie chooses the native decompose candidate deterministically.
	REQUIRE_NO_FAIL(con.Query("SET resultdb_strategy='auto'; SET resultdb_cost_build=0.5"));
	auto tie = con.Prepare(queries[0]);
	REQUIRE(!tie->HasError());
	REQUIRE(tie->data->properties.resultdb.estimated_semijoin_cost == 6);
	REQUIRE(tie->data->properties.resultdb.execution_strategy == ResultDBExecutionStrategy::DECOMPOSE);
	REQUIRE_NO_FAIL(con.Query("RESET resultdb_cost_build"));
	// Existing snapshots survive a subsequent RESET, including a forced rebind.
	auto saved = con.Prepare(queries[1]);
	REQUIRE(!saved->HasError());
	auto saved_file = saved->data->properties.native_row_oracle->file;
	REQUIRE_NO_FAIL(con.Query("RESET optimizer_statistics_file; SET optimizer_statistics_mode='estimated'"));
	saved->data->properties.always_require_rebind = true;
	auto saved_result = saved->Execute();
	REQUIRE_NO_FAIL(*saved_result);
	REQUIRE(saved->data->properties.native_row_oracle->file == saved_file);
	REQUIRE(con.Prepare(queries[0])->HasError());
	// Counts from other predicates cannot be substituted, and missing counts fail.
	REQUIRE_NO_FAIL(con.Query("SET optimizer_statistics_file='" + profile + "'; SET optimizer_statistics_mode='oracle'"));
	REQUIRE(con.Prepare(queries[2] + " AND a.note='different'")->HasError());
	REQUIRE_NO_FAIL(con.Query("SET disabled_optimizers='join_order'"));
	REQUIRE(con.Prepare(queries[0])->HasError());
	REQUIRE_NO_FAIL(con.Query("RESET disabled_optimizers"));
	// Remove a reduction from the same query/profile: strict mode must identify
	// that precise request, rather than reuse the old local-filter cardinalities.
	auto complete = con.Prepare(queries[1]);
	REQUIRE(!complete->HasError());
	auto reduction_audit = complete->data->properties.native_row_oracle;
	string excluded;
	for (auto &item : reduction_audit->requests) {
		if (item.second.stage.find("resultdb_reduction") != string::npos) {
			excluded = item.first;
			break;
		}
	}
	REQUIRE(!excluded.empty());
	NativeWrite(profile, NativeProfile(database, {reduction_audit}, counter, false, excluded));
	REQUIRE_NO_FAIL(con.Query("SET optimizer_statistics_file='" + profile + "'"));
	auto missing = con.Prepare(queries[1]);
	REQUIRE(missing->HasError());
	REQUIRE(missing->GetError().find(excluded) != string::npos);
	REQUIRE(missing->GetError().find("resultdb_reduction") != string::npos);
	// Advisory zeros must preserve actual outputs for both native AUTO and direct TDFold.
	SealResultDBOracle(con, counter, database, profile, {queries[1]}, true);
	for (auto strategy : {"auto", "tdfold"}) {
		REQUIRE_NO_FAIL(con.Query(string("SET resultdb_strategy='") + strategy + "'"));
		auto actual = con.Query(queries[1]);
		REQUIRE_NO_FAIL(*actual);
		auto expected = counter.Query(queries[1]);
		NativeCompare(*actual, *expected);
	}
}
