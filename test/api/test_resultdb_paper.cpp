#include "catch.hpp"
#include "test_helpers.hpp"
#include "duckdb/main/client_config.hpp"
#include "duckdb/main/prepared_statement_data.hpp"
#include "duckdb/optimizer/resultdb_cardinalities.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"

#include <algorithm>
#include <fstream>
#include <functional>
#include <cmath>
#include <set>

using namespace duckdb;

namespace {
using Mask = uint64_t;
using CardFunction = std::function<double(Mask, Mask)>;

static string PaperCards(const vector<string> &aliases, const CardFunction &card) {
	auto names = [&](Mask subset) {
		string result = "[";
		for (idx_t i = 0; i < aliases.size(); i++) {
			if (subset & (Mask(1) << i)) {
				if (result.size() > 1) {
					result += ",";
				}
				result += "\"" + aliases[i] + "\"";
			}
		}
		return result + "]";
	};
	string json = "{\"fixture\":[";
	auto all = (Mask(1) << aliases.size()) - 1;
	for (Mask left = 1; left <= all; left++) {
		if (left > 1) {
			json += ",";
		}
		json += "{\"relations\":" + names(left) + ",\"size\":" + std::to_string(card(left, 0)) + ",\"reductions\":[";
		bool first = true;
		for (auto right = all ^ left; right; right = (right - 1) & (all ^ left)) {
			if (!first) {
				json += ",";
			}
			first = false;
			json += "{\"right_relations\":" + names(right) + ",\"size\":" + std::to_string(card(left, right)) + "}";
		}
		json += "]}";
	}
	return json + "]}";
}

static ResultDBEnumerationInput PaperGraph(idx_t n, const vector<ResultDBEnumerationEdge> &edges, Mask requested) {
	ResultDBEnumerationInput graph;
	for (idx_t i = 0; i < n; i++) {
		graph.nodes.push_back({i, 0, (requested & (Mask(1) << i)) != 0, Mask(1) << i});
	}
	graph.edges = edges;
	return graph;
}

static std::multiset<string> PaperRows(QueryResult &result) {
	std::multiset<string> rows;
	while (auto chunk = result.Fetch()) {
		for (idx_t row = 0; row < chunk->size(); row++) {
			string key;
			for (idx_t col = 0; col < chunk->ColumnCount(); col++) {
				auto value = chunk->GetValue(col, row);
				auto text = value.IsNull() ? string("<NULL>") : value.ToString();
				key += std::to_string(text.size()) + ":" + text;
			}
			rows.insert(key);
		}
	}
	return rows;
}

static void InstallPaperCards(Connection &con, const vector<string> &aliases, double value = 10) {
	ClientConfig::GetConfig(*con.context).resultdb_cardinalities =
	    ResultDBCardinalityFile::Parse(PaperCards(aliases, [=](Mask, Mask) { return value; }));
	REQUIRE_NO_FAIL(con.Query("SET resultdb_cardinality_dataset='fixture'"));
	REQUIRE_NO_FAIL(con.Query("SET resultdb_optimizer='paper'"));
}

static idx_t ScansWithFiveFilter(PhysicalOperator &op) {
	idx_t count = 0;
	if (op.type == PhysicalOperatorType::TABLE_SCAN) {
		auto &scan = op.Cast<PhysicalTableScan>();
		if (scan.table_filters) {
			for (auto &entry : *scan.table_filters) {
				if (entry.Filter().ToString("key").find("5") != string::npos) {
					count++;
					break;
				}
			}
		}
	}
	for (auto &child : op.children) {
		count += ScansWithFiveFilter(child.get());
	}
	return count;
}
} // namespace

TEST_CASE("ResultDB strategies share graph capture and whole-query filter propagation",
          "[api][resultdb][resultdb_paper]") {
	DuckDB db(nullptr);
	Connection con(db);
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE a(k INTEGER); INSERT INTO a VALUES (5),(5),(7),(NULL);"
	                          "CREATE TABLE b(k INTEGER); INSERT INTO b VALUES (5),(5),(7),(8),(NULL);"
	                          "CREATE TABLE c(k INTEGER); INSERT INTO c VALUES (5),(5),(7),(9),(NULL)"));
	const string from = " FROM a,b,c WHERE a.k=b.k AND b.k=c.k AND a.k=5";
	auto expected = con.Query("SELECT DISTINCT a.k" + from);
	REQUIRE_NO_FAIL(*expected);
	auto rows = PaperRows(*expected);
	// Independently count bags/survivors of the original filtered graph. The
	// propagated B/C scan inputs are smaller; the paper model retains its scope.
	auto json = PaperCards({"a", "b", "c"}, [&](Mask left, Mask right) {
		auto tables = [](Mask mask) {
			string out;
			for (idx_t i = 0; i < 3; i++) {
				if (mask & (Mask(1) << i)) {
					if (!out.empty()) { out += ","; }
					out += string(1, char('a' + i));
				}
			}
			return out;
		};
		auto predicates = [](Mask mask, Mask outside) {
			string out = "TRUE";
			if (mask & 1) { out += " AND a.k=5"; }
			for (auto edge : {Mask(3), Mask(6)}) {
				if (((mask | outside) & edge) == edge && (mask & edge)) {
					out += edge == 3 ? " AND a.k=b.k" : " AND b.k=c.k";
				}
			}
			return out;
		};
		string sql = "SELECT COUNT(*) FROM " + tables(left) + " WHERE " + predicates(left, 0);
		if (right) {
			sql += " AND EXISTS (SELECT 1 FROM " + tables(right) + " WHERE " + predicates(right, left) + ")";
		}
		auto count = con.Query(sql);
		REQUIRE_NO_FAIL(*count);
		return count->GetValue(0, 0).GetValue<double>();
	});
	ClientConfig::GetConfig(*con.context).resultdb_cardinalities = ResultDBCardinalityFile::Parse(json);
	REQUIRE_NO_FAIL(con.Query("SET resultdb_optimizer='paper'; SET resultdb_cardinality_dataset='fixture'"));
	for (auto strategy : {"semijoin", "tdroot", "tdfold_no_tvc", "tdfold"}) {
		REQUIRE_NO_FAIL(con.Query(string("SET resultdb_strategy='") + strategy + "'"));
		auto prepared = con.Prepare("SELECT RESULTDB a.k,b.k,c.k" + from);
		REQUIRE(!prepared->HasError());
		REQUIRE(prepared->data->properties.resultdb.join_edges.size() == 2);
		idx_t filtered_scans = 0;
		auto &program = prepared->data->resultdb_yannakakis_program;
		if (program) {
			REQUIRE(program->relations.size() == 3); // Inference must not create a cyclic fold.
			for (auto &base : program->base_plans) {
				filtered_scans += ScansWithFiveFilter(base->Root());
			}
		} else {
			filtered_scans = ScansWithFiveFilter(prepared->data->physical_plan->Root());
		}
		REQUIRE(filtered_scans == 3);
		auto result = prepared->Execute();
		REQUIRE_NO_FAIL(*result);
		for (auto current = result.get(); current; current = current->next.get()) {
			REQUIRE(current->names == vector<string> {"k"});
			REQUIRE(current->types == vector<LogicalType> {LogicalType::INTEGER});
			REQUIRE(PaperRows(*current) == rows);
		}
		REQUIRE(result->next);
		REQUIRE(result->next->next);
		REQUIRE(!result->next->next->next);
	}
}

TEST_CASE("ResultDB shared capture retains original cycles and handles proven empty inputs",
          "[api][resultdb][resultdb_paper]") {
	DuckDB db(nullptr);
	Connection con(db);
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE t(k INTEGER, note VARCHAR); "
	                          "INSERT INTO t VALUES (5,'x'),(5,'x'),(7,NULL),(NULL,'n')"));
	InstallPaperCards(con, {"a", "b", "c"}, 0); // Estimates cannot prove emptiness.
	const string from = " FROM t a,t b,t c WHERE a.k=b.k AND b.k=c.k AND a.k=c.k AND a.k=5";
	auto expected = con.Query("SELECT DISTINCT a.note" + from);
	REQUIRE_NO_FAIL(*expected);
	auto rows = PaperRows(*expected);
	for (auto strategy : {"semijoin", "tdfold_no_tvc", "tdfold"}) {
		REQUIRE_NO_FAIL(con.Query(string("SET resultdb_strategy='") + strategy + "'"));
		auto result = con.Query("SELECT RESULTDB a.note,c.note" + from);
		REQUIRE_NO_FAIL(*result);
		REQUIRE(result->properties.resultdb.join_edges.size() == 3);
		REQUIRE(PaperRows(*result) == rows);
		REQUIRE(result->next);
		REQUIRE(PaperRows(*result->next) == rows);
		REQUIRE(!result->next->next);
		result = con.Query("SELECT RESULTDB a.note,c.note" + from + " AND a.k=7");
		REQUIRE_NO_FAIL(*result);
		REQUIRE(PaperRows(*result).empty());
		REQUIRE(result->next);
		REQUIRE(result->next->names == vector<string> {"note"});
		REQUIRE(PaperRows(*result->next).empty());
		REQUIRE(!result->next->next);
	}
	REQUIRE_NO_FAIL(con.Query("SET resultdb_strategy='tdroot'"));
	REQUIRE_FAIL(con.Query("SELECT RESULTDB a.note" + from));
	REQUIRE_FAIL(con.Query("SELECT RESULTDB a.note" + from + " AND FALSE"));
}

TEST_CASE("Paper ResultDB strict cardinality parsing and occurrence identity", "[api][resultdb][resultdb_paper]") {
	for (auto json :
	     {"{}garbage", "{\"d\":[{\"relations\":[\"a\"],\"size\":}]}", "{\"d\":[{\"relations\":[\"a\"],\"size\":-1}]}",
	      "{\"d\":[{\"relations\":[\"a\",\"a\"],\"size\":1}]}",
	      "{\"d\":[{\"relations\":[\"a\"],\"size\":1,\"size\":2}]}",
	      "{\"d\":[{\"relations\":[\"a\"],\"size\":1},{\"relations\":[\"a\"],\"size\":2}]}",
	      "{\"d\":[{\"relations\":[\"a\"],\"size\":1,\"reductions\":[{\"right_relations\":[\"a\"],\"size\":1}]}]}"}) {
		REQUIRE_THROWS(ResultDBCardinalityFile::Parse(json));
	}
	auto file = ResultDBCardinalityFile::Parse("{\"d\":[{\"relations\":[\"a\"],\"size\":7}]}");
	REQUIRE(file->sha256.size() == 64);
	ResultDBInjectedCardinalities cards(file, "d", {"b", "a"});
	REQUIRE(cards.ReducedCardinality(2, 0) == 7);
	REQUIRE_THROWS_WITH(cards.ReducedCardinality(1, 2), Catch::Contains("relations=[b], right_relations=[a]"));
	REQUIRE_THROWS(ResultDBInjectedCardinalities(file, "missing", {"a"}));
	REQUIRE_THROWS(ResultDBInjectedCardinalities(file, "d", {"x"}));
	REQUIRE_THROWS(ResultDBInjectedCardinalities(file, "d", {"a", "a"}));
}

TEST_CASE("Paper TDRoot charges the root top-down pass and considers unrequested roots",
          "[api][resultdb][resultdb_paper]") {
	auto json = PaperCards(
	    {"a", "b"}, [](Mask left, Mask right) { return left == 1 ? (right ? 5.0 : 100.0) : (right ? 7.0 : 10.0); });
	ResultDBInjectedCardinalities cards(ResultDBCardinalityFile::Parse(json), "fixture", {"a", "b"});
	auto input = PaperGraph(2, {{0, 1}}, 2);
	auto plan = ResultDBPlanEnumerator::EnumeratePaperRoot(input, cards, {});
	REQUIRE(plan.root == 0);
	// 100 + 3*10 bottom up; 10 + 3*5 top down. Root a is not an output.
	REQUIRE(plan.cost == 155);
	REQUIRE(plan.top_down_children[0] == vector<idx_t> {1});
	REQUIRE(plan.fully_reduced_cardinality == vector<double> {5, 7});
	input.nodes[0].requested = true;
	input.nodes[1].requested = false;
	plan = ResultDBPlanEnumerator::EnumeratePaperRoot(input, cards, {});
	REQUIRE(plan.root == 0);
	REQUIRE(plan.cost == 130);
	REQUIRE(plan.top_down_children[0].empty());
}

TEST_CASE("Paper TDRoot orders reductions by parent selectivity", "[api][resultdb][resultdb_paper]") {
	auto json = PaperCards({"a", "b", "c"}, [](Mask left, Mask right) {
		if (left == 1) {
			return right == 6 ? 0.0 : right == 4 ? 10.0 : right == 2 ? 500.0 : 1000.0;
		}
		return left == 2 ? 20.0 : 200.0;
	});
	ResultDBInjectedCardinalities cards(ResultDBCardinalityFile::Parse(json), "fixture", {"a", "b", "c"});
	auto plan = ResultDBPlanEnumerator::EnumeratePaperRoot(PaperGraph(3, {{0, 1}, {0, 2}}, 7), cards, {});
	REQUIRE(plan.root == 0);
	REQUIRE(plan.bottom_up_children[0] == vector<idx_t> {2, 1});
	REQUIRE(plan.cost == 1890);
}

TEST_CASE("Paper TDRoot agrees with an independent all-roots schedule oracle", "[api][resultdb][resultdb_paper]") {
	const idx_t n = 4;
	vector<ResultDBEnumerationEdge> possible;
	for (idx_t a = 0; a < n; a++) {
		for (idx_t b = a + 1; b < n; b++) {
			possible.push_back({a, b});
		}
	}
	CardFunction card = [](Mask left, Mask right) {
		double count = 100 + 13 * left;
		for (Mask bit = 1; bit <= 8; bit <<= 1) {
			if (right & bit) {
				count /= 1 + bit;
			}
		}
		return count;
	};
	auto file = ResultDBCardinalityFile::Parse(PaperCards({"a", "b", "c", "d"}, card));
	idx_t checked = 0;
	for (Mask selected = 1; selected < (Mask(1) << possible.size()); selected++) {
		vector<ResultDBEnumerationEdge> edges;
		vector<vector<idx_t>> adjacent(n);
		for (idx_t e = 0; e < possible.size(); e++) {
			if (selected & (Mask(1) << e)) {
				edges.push_back(possible[e]);
				adjacent[possible[e].left].push_back(possible[e].right);
				adjacent[possible[e].right].push_back(possible[e].left);
			}
		}
		if (edges.size() != n - 1) {
			continue;
		}
		vector<idx_t> reachable {0};
		for (idx_t p = 0; p < reachable.size(); p++) {
			for (auto next : adjacent[reachable[p]]) {
				if (std::find(reachable.begin(), reachable.end(), next) == reachable.end()) {
					reachable.push_back(next);
				}
			}
		}
		if (reachable.size() != n) {
			continue;
		}
		for (Mask requested = 1; requested < 16; requested++) {
			double best = std::numeric_limits<double>::infinity();
			idx_t best_root = n;
			for (idx_t root = 0; root < n; root++) {
				vector<idx_t> parent(n, n), order {root};
				for (idx_t p = 0; p < order.size(); p++) {
					for (auto child : adjacent[order[p]]) {
						if (child != parent[order[p]]) {
							parent[child] = order[p];
							order.push_back(child);
						}
					}
				}
				vector<Mask> subtree(n);
				vector<double> bottom(n);
				double cost = 0;
				for (auto it = order.rbegin(); it != order.rend(); ++it) {
					auto node = *it;
					subtree[node] = Mask(1) << node;
					vector<std::pair<double, idx_t>> children;
					for (auto child : adjacent[node]) {
						if (child != parent[node]) {
							children.emplace_back(card(Mask(1) << node, subtree[child]), child);
						}
					}
					std::sort(children.begin(), children.end());
					bottom[node] = card(Mask(1) << node, 0);
					for (auto entry : children) {
						auto child = entry.second;
						cost += bottom[node] + 3 * bottom[child];
						subtree[node] |= subtree[child];
						bottom[node] = std::min(bottom[node], card(Mask(1) << node, subtree[node] ^ (Mask(1) << node)));
					}
				}
				for (auto child : order) {
					if (child != root && (subtree[child] & requested)) {
						cost += bottom[child] + 3 * card(Mask(1) << parent[child], 15 ^ (Mask(1) << parent[child]));
					}
				}
				if (cost < best) {
					best = cost;
					best_root = root;
				}
			}
			ResultDBInjectedCardinalities cards(file, "fixture", {"a", "b", "c", "d"});
			auto actual = ResultDBPlanEnumerator::EnumeratePaperRoot(PaperGraph(n, edges, requested), cards, {});
			REQUIRE(actual.cost == Approx(best).epsilon(0.00001));
			REQUIRE(actual.root == best_root);
			checked++;
		}
	}
	REQUIRE(checked == 240);
}

TEST_CASE("Paper TDFold preserves original occurrences through blocks and recursive TVCs",
          "[api][resultdb][resultdb_paper]") {
	vector<string> aliases {"a", "b", "c", "d", "e", "f"};
	auto file = ResultDBCardinalityFile::Parse(
	    PaperCards(aliases, [](Mask left, Mask) { return (left & (left - 1)) == 0 ? 1.0 : 1000.0; }));
	vector<vector<ResultDBEnumerationEdge>> graphs {{{0, 1}, {1, 2}, {2, 0}, {2, 3}, {3, 4}, {4, 2}, {4, 5}},
	                                                {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 0}, {1, 4}},
	                                                {{0, 1}, {1, 2}, {2, 0}, {2, 3}, {3, 4}, {4, 5}, {5, 3}}};
	for (auto &edges : graphs) {
		for (bool tvc : {false, true}) {
			ResultDBInjectedCardinalities cards(file, "fixture", aliases);
			auto input = PaperGraph(6, edges, 63);
			auto result = ResultDBPlanEnumerator::EnumeratePaper(input, cards, {}, tvc);
			REQUIRE(result.valid);
			Mask assigned = 0;
			for (auto &fold : result.folds) {
				Mask mask = 0;
				for (auto member : fold) {
					REQUIRE((assigned & (Mask(1) << member)) == 0);
					assigned |= Mask(1) << member;
					mask |= Mask(1) << member;
				}
				REQUIRE(result.join_splits.count(mask) == 1);
			}
			REQUIRE(result.cost == Approx(result.join_cost + result.decomposition_cost + result.reduction_cost));
			REQUIRE_THROWS_WITH(ResultDBPlanEnumerator::EnumeratePaper(input, cards, {}, tvc, 1),
			                    Catch::Contains("work limit"));
		}
	}
}

TEST_CASE("Paper SC1F and SC2F agree with exhaustive connected-partition costing", "[api][resultdb][resultdb_paper]") {
	vector<vector<ResultDBEnumerationEdge>> graphs {{{0, 1}, {1, 2}, {2, 3}, {3, 0}},
	                                                {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {0, 2}},
	                                                {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {0, 2}, {1, 3}}};
	auto count = [](Mask mask) {
		idx_t size = 0;
		while (mask) {
			size++;
			mask &= mask - 1;
		}
		return size;
	};
	for (auto &edges : graphs) {
		vector<Mask> adjacency(4, 0);
		for (auto &edge : edges) {
			adjacency[edge.left] |= Mask(1) << edge.right;
			adjacency[edge.right] |= Mask(1) << edge.left;
		}
		auto connected = [&](Mask mask) {
			auto reached = mask & (~mask + 1);
			for (idx_t i = 0; i < 4; i++) {
				for (idx_t node = 0; node < 4; node++) {
					if (reached & (Mask(1) << node))
						reached |= adjacency[node] & mask;
				}
			}
			return reached == mask;
		};
		for (bool exponential : {false, true}) {
			CardFunction card = [&](Mask left, Mask) {
				return exponential ? std::pow(10.0, count(left)) : 100.0;
			};
			auto file = ResultDBCardinalityFile::Parse(PaperCards({"a", "b", "c", "d"}, card));
			// Independent bottom-up DP over every bit subset, not the production connected-set traversal.
			vector<double> joins(16, std::numeric_limits<double>::infinity());
			for (Mask subset = 1; subset < 16; subset++) {
				if (count(subset) == 1) {
					joins[subset] = 0;
					continue;
				}
				if (!connected(subset))
					continue;
				for (Mask left = 1; left < subset; left++) {
					if ((left & subset) != left)
						continue;
					joins[subset] = std::min(joins[subset], joins[left] + joins[subset ^ left] + card(subset, 0));
				}
			}
			for (Mask requested = 1; requested < 16; requested++) {
				auto score = [&](Mask fold) {
					double value = joins[fold];
					if (count(fold) > 1)
						value += card(fold, 0) * count(fold & requested);
					if (fold != 15)
						value += card(fold, 0) * (1 + (((15 ^ fold) & requested) != 0));
					return value;
				};
				double best = score(15);
				vector<Mask> expected {15};
				for (Mask left = 1; left < 15; left++) {
					auto right = 15 ^ left;
					if (left > right || !connected(left) || !connected(right))
						continue;
					double value = score(left) + score(right);
					vector<Mask> folds {left, right};
					if (value < best || (value == best && folds < expected)) {
						best = value;
						expected = folds;
					}
				}
				ResultDBInjectedCardinalities cards(file, "fixture", {"a", "b", "c", "d"});
				auto actual =
				    ResultDBPlanEnumerator::EnumeratePaper(PaperGraph(4, edges, requested), cards, {}, false);
				vector<Mask> folds;
				Mask used = 0;
				for (auto &group : actual.folds) {
					Mask fold = 0;
					for (auto node : group)
						fold |= Mask(1) << node;
					folds.push_back(fold);
					used |= fold;
				}
				for (idx_t node = 0; node < 4; node++)
					if (!(used & (Mask(1) << node)))
						folds.push_back(Mask(1) << node);
				std::sort(folds.begin(), folds.end());
				REQUIRE(folds == expected);
				REQUIRE(actual.join_costs.at(15) == joins[15]);
			}
		}
	}
}

TEST_CASE("Paper recursive TVC candidate can beat one-fold and two-fold candidates",
          "[api][resultdb][resultdb_paper]") {
	vector<string> aliases {"a", "b", "c", "d", "e", "f"};
	auto file = ResultDBCardinalityFile::Parse(PaperCards(aliases, [](Mask left, Mask) {
		double result = 1;
		for (; left; left &= left - 1)
			result *= 10;
		return result;
	}));
	ResultDBInjectedCardinalities cards(file, "fixture", aliases);
	auto graph = PaperGraph(6, {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 0}, {1, 4}}, 63);
	auto no_tvc = ResultDBPlanEnumerator::EnumeratePaper(graph, cards, {}, false);
	auto tvc = ResultDBPlanEnumerator::EnumeratePaper(graph, cards, {}, true);
	REQUIRE(tvc.used_tvc);
	REQUIRE(tvc.folds.size() == 3);
	REQUIRE(tvc.cost < no_tvc.cost);
	REQUIRE(tvc.folds == vector<vector<idx_t>> {{0, 5}, {1, 4}, {2, 3}});
}

TEST_CASE("Paper ResultDB native execution preserves cycles, predicates and output semantics",
          "[api][resultdb][resultdb_paper]") {
	DuckDB db(nullptr);
	Connection con(db);
	REQUIRE_NO_FAIL(
	    con.Query("CREATE TABLE a(id INTEGER, note VARCHAR); INSERT INTO a VALUES (1,'x'),(1,'x'),(2,NULL),(NULL,'n');"
	              "CREATE TABLE b(id BIGINT); INSERT INTO b VALUES (1),(1),(2),(3),(NULL);"
	              "CREATE TABLE c(id INTEGER); INSERT INTO c VALUES (1),(2),(NULL)"));
	auto from =
	    " FROM a,b,c WHERE CAST(a.id AS BIGINT)=b.id AND b.id=c.id AND a.id=c.id AND (a.note='x' OR a.note IS NULL)";
	auto expected_a = con.Query(string("SELECT DISTINCT a.id,a.note") + from);
	auto expected_b = con.Query(string("SELECT DISTINCT b.id") + from);
	auto expected_plain = con.Query(string("SELECT a.id,a.note,b.id") + from);
	REQUIRE_NO_FAIL(*expected_a);
	REQUIRE_NO_FAIL(*expected_b);
	auto rows_a = PaperRows(*expected_a);
	auto rows_b = PaperRows(*expected_b);
	REQUIRE_NO_FAIL(*expected_plain);
	auto rows_plain = PaperRows(*expected_plain);
	InstallPaperCards(con, {"a", "b", "c"}, 0); // Wrong zeros must affect planning only.
	for (auto strategy : {"tdfold_no_tvc", "tdfold"}) {
		REQUIRE_NO_FAIL(con.Query(string("SET resultdb_strategy='") + strategy + "'"));
		auto result = con.Query(string("SELECT RESULTDB a.id,a.note,b.id") + from);
		REQUIRE_NO_FAIL(*result);
		REQUIRE(result->properties.resultdb.join_edges.size() == 3);
		REQUIRE(result->properties.resultdb.optimizer_mode == "paper");
		REQUIRE(result->properties.resultdb.cardinality_sha256.size() == 64);
		REQUIRE(result->names == vector<string> {"id", "note"});
		REQUIRE(result->types == vector<LogicalType> {LogicalType::INTEGER, LogicalType::VARCHAR});
		REQUIRE(PaperRows(*result) == rows_a);
		REQUIRE(result->next);
		REQUIRE(PaperRows(*result->next) == rows_b);
		REQUIRE(!result->next->next);
	}
	REQUIRE_NO_FAIL(con.Query("SET resultdb_strategy='tdroot'"));
	REQUIRE_FAIL(con.Query(string("SELECT RESULTDB a.id") + from));
	auto tree = con.Query("SELECT RESULTDB a.id FROM a,b,c WHERE CAST(a.id AS BIGINT)=b.id AND b.id=c.id");
	REQUIRE_NO_FAIL(*tree);
	REQUIRE(tree->properties.resultdb.join_edges.size() == 2);
	REQUIRE_FAIL(con.Query("SET resultdb_join_planner='cout'")); // Retired planner option.
	REQUIRE_NO_FAIL(con.Query("SET resultdb_strategy='decompose'"));
	auto plain = con.Query(string("SELECT a.id,a.note,b.id") + from);
	REQUIRE_NO_FAIL(*plain);
	REQUIRE(!plain->properties.resultdb.paper_planned);
	REQUIRE(PaperRows(*plain) == rows_plain);
	auto ordinary = con.Query(string("SELECT RESULTDB a.id,a.note,b.id") + from);
	REQUIRE_NO_FAIL(*ordinary);
	REQUIRE(!ordinary->properties.resultdb.paper_planned);
	REQUIRE(PaperRows(*ordinary) == rows_a);
	REQUIRE(ordinary->next);
	REQUIRE(PaperRows(*ordinary->next) == rows_b);
	REQUIRE(!ordinary->next->next);
	REQUIRE_NO_FAIL(con.Query("SET resultdb_strategy='tdfold'; DELETE FROM c"));
	auto empty = con.Query(string("SELECT RESULTDB a.id,a.note,b.id") + from);
	REQUIRE_NO_FAIL(*empty);
	REQUIRE(PaperRows(*empty).empty());
	REQUIRE(empty->next);
	REQUIRE(empty->next->names == vector<string> {"id"});
	REQUIRE(empty->next->types == vector<LogicalType> {LogicalType::BIGINT});
	REQUIRE(PaperRows(*empty->next).empty());
	REQUIRE(!empty->next->next);
}

TEST_CASE("Paper ResultDB settings load atomically and prepared queries retain the snapshot",
          "[api][resultdb][resultdb_paper]") {
	DuckDB db(nullptr);
	Connection con(db);
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE t(id INTEGER); INSERT INTO t VALUES (1),(1),(2),(NULL)"));
	auto path = TestCreatePath("resultdb_paper_cardinalities.json");
	{
		std::ofstream out(path);
		out << PaperCards({"employee", "manager"}, [](Mask, Mask) { return 4.0; });
	}
	REQUIRE_NO_FAIL(con.Query("SET resultdb_cardinality_file='" + path + "'"));
	REQUIRE_NO_FAIL(con.Query(
	    "SET resultdb_cardinality_dataset='fixture'; SET resultdb_optimizer='paper'; SET resultdb_strategy='tdroot'"));
	auto prepared =
	    con.Prepare("SELECT RESULTDB employee.id,manager.id FROM t employee,t manager WHERE employee.id=manager.id");
	REQUIRE(!prepared->HasError());
	auto snapshot = prepared->data->properties.resultdb.cardinality_snapshot;
	auto parameterized = con.Prepare("SELECT RESULTDB employee.id FROM t employee,t manager "
	                                 "WHERE employee.id=manager.id AND employee.id=$1");
	REQUIRE(!parameterized->HasError());
	{
		std::ofstream out(path);
		out << "invalid JSON";
	}
	REQUIRE_FAIL(con.Query("SET resultdb_cardinality_file='" + path + "'"));
	REQUIRE(ClientConfig::GetConfig(*con.context).resultdb_cardinalities == snapshot);
	REQUIRE_NO_FAIL(con.Query("RESET resultdb_cardinality_file"));
	auto result = prepared->Execute();
	REQUIRE_NO_FAIL(*result);
	REQUIRE(result->properties.resultdb.cardinality_snapshot == snapshot);
	REQUIRE(result->next);
	REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 2);
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE force_catalog_rebind(i INTEGER)"));
	auto rebound = prepared->Execute();
	REQUIRE_NO_FAIL(*rebound);
	REQUIRE(rebound->properties.resultdb.cardinality_snapshot == snapshot);
	auto parameter_result = parameterized->Execute(1);
	REQUIRE_NO_FAIL(*parameter_result);
	REQUIRE(parameter_result->properties.resultdb.cardinality_snapshot == snapshot);
	REQUIRE(parameter_result->Cast<MaterializedQueryResult>().RowCount() == 1);
	REQUIRE(!ClientConfig::GetConfig(*con.context).resultdb_cardinalities);
	REQUIRE_FAIL(con.Query("SELECT RESULTDB employee.id FROM t employee,t manager WHERE employee.id=manager.id"));
	REQUIRE_FAIL(con.Query("SET resultdb_cost_build=0"));
	REQUIRE_FAIL(con.Query("SET resultdb_optimizer='unknown'"));
}

TEST_CASE("Paper ResultDB executes against a reopened persistent database", "[api][resultdb][resultdb_paper]") {
	auto path = TestCreatePath("resultdb_paper_persistent.duckdb");
	{
		DuckDB db(path);
		Connection con(db);
		REQUIRE_NO_FAIL(con.Query("CREATE TABLE a(i INTEGER); INSERT INTO a VALUES (1),(1),(2); "
		                          "CREATE TABLE b(i INTEGER); INSERT INTO b VALUES (1),(NULL)"));
	}
	{
		DBConfig config;
		config.options.access_mode = AccessMode::READ_ONLY;
		DuckDB db(path, &config);
		Connection con(db);
		InstallPaperCards(con, {"a", "b"});
		REQUIRE_NO_FAIL(con.Query("SET resultdb_strategy='tdroot'"));
		auto result = con.Query("SELECT RESULTDB a.i,b.i FROM a,b WHERE a.i=b.i");
		REQUIRE_NO_FAIL(*result);
		REQUIRE(result->Cast<MaterializedQueryResult>().RowCount() == 1);
		REQUIRE(result->next);
		REQUIRE(result->next->Cast<MaterializedQueryResult>().RowCount() == 1);
		REQUIRE(!result->next->next);
	}
}


TEST_CASE("Paper AUTO rejects historical C_out cardinalities without a native oracle",
          "[api][resultdb][resultdb_paper]") {
	DuckDB db(nullptr);
	Connection con(db);
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE narrow_a(k BOOLEAN); CREATE TABLE narrow_b(k BOOLEAN)"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO narrow_a VALUES (TRUE), (FALSE); "
	                          "INSERT INTO narrow_b VALUES (TRUE), (FALSE)"));
	InstallPaperCards(con, {"a", "b"}, 2);
	REQUIRE_NO_FAIL(con.Query("SET resultdb_strategy='auto'"));
	auto result = con.Query("SELECT RESULTDB * FROM narrow_a a JOIN narrow_b b ON a.k=b.k");
	REQUIRE_FAIL(result);
	REQUIRE(result->GetError().find("native row oracle") != string::npos);
}
