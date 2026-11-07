#include "duckdb/optimizer/resultdb_plan_enumerator.hpp"
#include "duckdb/common/exception.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>

namespace duckdb {
namespace resultdb_paper {
using Mask = uint64_t;

static idx_t Count(Mask mask) {
	idx_t count = 0;
	for (; mask; mask &= mask - 1) {
		count++;
	}
	return count;
}
static idx_t First(Mask mask) {
	if (!mask) {
		throw InternalException("Empty ResultDB occurrence set");
	}
	idx_t result = 0;
	while (!(mask & 1)) {
		mask >>= 1;
		result++;
	}
	return result;
}
static Mask Bit(idx_t i) {
	return Mask(1) << i;
}

//! Graph slots are local. members always refers to original relation occurrences.
struct Graph {
	vector<Mask> members;
	vector<Mask> edges;
	Mask All() const {
		return Bit(members.size()) - 1;
	}
	Mask Original(Mask nodes) const {
		Mask result = 0;
		for (; nodes; nodes &= nodes - 1) {
			result |= members[First(nodes)];
		}
		return result;
	}
	Mask Neighbors(Mask nodes) const {
		Mask result = 0;
		for (auto pending = nodes; pending; pending &= pending - 1) {
			result |= edges[First(pending)];
		}
		return result & ~nodes;
	}
	Mask Reach(Mask seed, Mask allowed) const {
		Mask reached = seed & allowed;
		while (true) {
			auto next = reached | (Neighbors(reached) & allowed);
			if (next == reached) {
				return reached;
			}
			reached = next;
		}
	}
	bool Connected(Mask nodes) const {
		return nodes && Reach(nodes & (~nodes + 1), nodes) == nodes;
	}
	vector<Mask> Components(Mask nodes) const {
		vector<Mask> result;
		while (nodes) {
			auto component = Reach(nodes & (~nodes + 1), nodes);
			result.push_back(component);
			nodes &= ~component;
		}
		return result;
	}
	vector<Mask> Blocks(Mask scope) const {
		vector<idx_t> discovery(members.size(), 0), low(members.size(), 0);
		vector<std::pair<idx_t, idx_t>> stack;
		vector<Mask> blocks;
		idx_t clock = 0;
		std::function<void(idx_t, idx_t)> visit = [&](idx_t node, idx_t parent) {
			discovery[node] = low[node] = ++clock;
			for (auto neighbors = edges[node] & scope; neighbors; neighbors &= neighbors - 1) {
				auto child = First(neighbors);
				if (child == parent) {
					continue;
				}
				if (!discovery[child]) {
					stack.emplace_back(node, child);
					visit(child, node);
					low[node] = std::min(low[node], low[child]);
					if (low[child] >= discovery[node]) {
						Mask block = 0;
						while (!stack.empty()) {
							auto edge = stack.back();
							stack.pop_back();
							block |= Bit(edge.first) | Bit(edge.second);
							if (edge.first == node && edge.second == child) {
								break;
							}
						}
						if (Count(block) >= 3) {
							blocks.push_back(block);
						}
					}
				} else if (discovery[child] < discovery[node]) {
					stack.emplace_back(node, child);
					low[node] = std::min(low[node], discovery[child]);
				}
			}
		};
		for (auto nodes = scope; nodes; nodes &= nodes - 1) {
			auto node = First(nodes);
			if (!discovery[node]) {
				visit(node, DConstants::INVALID_INDEX);
			}
		}
		std::sort(blocks.begin(), blocks.end());
		return blocks;
	}
	Graph Contract(const vector<Mask> &groups) const {
		Graph result;
		result.edges.resize(groups.size(), 0);
		for (idx_t i = 0; i < groups.size(); i++) {
			result.members.push_back(Original(groups[i]));
			for (idx_t j = 0; j < i; j++) {
				if (Neighbors(groups[i]) & groups[j]) {
					result.edges[i] |= Bit(j);
					result.edges[j] |= Bit(i);
				}
			}
		}
		return result;
	}
};

static Graph MakeGraph(const ResultDBEnumerationInput &input) {
	if (input.nodes.empty() || input.nodes.size() > 62) {
		throw InvalidInputException("Paper ResultDB supports 1..62 relation occurrences");
	}
	Graph graph;
	graph.edges.resize(input.nodes.size(), 0);
	Mask used = 0;
	for (idx_t i = 0; i < input.nodes.size(); i++) {
		auto members = input.nodes[i].relations ? input.nodes[i].relations : Bit(i);
		if (members & used) {
			throw InternalException("Overlapping ResultDB graph occurrences");
		}
		used |= members;
		graph.members.push_back(members);
	}
	for (auto &edge : input.edges) {
		if (edge.left >= graph.members.size() || edge.right >= graph.members.size() || edge.left == edge.right) {
			throw InternalException("Invalid ResultDB paper graph edge");
		}
		graph.edges[edge.left] |= Bit(edge.right);
		graph.edges[edge.right] |= Bit(edge.left);
	}
	if (!graph.Connected(graph.All())) {
		throw InvalidInputException("Paper ResultDB requires a connected equality-join graph");
	}
	return graph;
}

struct Direction {
	bool computed = false;
	bool required = false;
	Mask subtree = 0;
	double bottom = 0;
	double full = 0;
	double cost = 0;
	vector<idx_t> children;
};

struct RootSearch {
	const ResultDBEnumerationInput &input;
	const Graph &graph;
	ResultDBEnumerationCostProvider &cards;
	const ResultDBCostWeights &weights;
	vector<vector<Direction>> memo;

	RootSearch(const ResultDBEnumerationInput &input, const Graph &graph, ResultDBEnumerationCostProvider &cards,
	           const ResultDBCostWeights &weights)
	    : input(input), graph(graph), cards(cards), weights(weights),
	      memo(graph.members.size(), vector<Direction>(graph.members.size() + 1)) {
	}

	Direction &Evaluate(idx_t node, idx_t parent) {
		auto &state = memo[node][parent];
		if (state.computed) {
			return state;
		}
		state.subtree = graph.members[node];
		state.required = input.nodes[node].requested;
		vector<std::pair<double, idx_t>> children;
		for (auto remaining = graph.edges[node]; remaining; remaining &= remaining - 1) {
			auto child = First(remaining);
			if (child == parent) {
				continue;
			}
			auto &child_state = Evaluate(child, node);
			children.emplace_back(cards.ReducedCardinality(graph.members[node], child_state.subtree), child);
		}
		std::sort(children.begin(), children.end(), [&](const auto &a, const auto &b) {
			return a.first != b.first ? a.first < b.first
			                          : input.nodes[a.second].stable_id < input.nodes[b.second].stable_id;
		});
		state.bottom = cards.ReducedCardinality(graph.members[node], 0);
		Mask by = 0;
		for (auto &entry : children) {
			auto child = entry.second;
			auto &other = Evaluate(child, node);
			state.children.push_back(child);
			state.cost += other.cost + weights.probe * state.bottom + weights.build * other.bottom;
			by |= other.subtree;
			state.bottom = std::min(state.bottom, cards.ReducedCardinality(graph.members[node], by));
			state.subtree |= other.subtree;
			state.required = state.required || other.required;
		}
		state.full =
		    parent == graph.members.size()
		        ? state.bottom
		        : std::min(state.bottom, cards.ReducedCardinality(graph.members[node],
		                                                          graph.Original(graph.All()) & ~graph.members[node]));
		// Algorithm 2 charges each outgoing top-down reduction here, including at the global root.
		for (auto child : state.children) {
			auto &other = Evaluate(child, node);
			if (other.required) {
				state.cost += weights.probe * other.bottom + weights.build * state.full;
			}
		}
		state.computed = true;
		return state;
	}

	ResultDBRootEnumerationResult Run() {
		idx_t edges = 0;
		for (auto neighbors : graph.edges) {
			edges += Count(neighbors);
		}
		if (edges / 2 + 1 != graph.members.size()) {
			throw InvalidInputException("Paper TDRoot requires an acyclic graph; use tdfold for cyclic queries");
		}
		ResultDBRootEnumerationResult result;
		for (idx_t node = 0; node < graph.members.size(); node++) {
			auto &state = Evaluate(node, graph.members.size());
			if (!result.valid || state.cost < result.cost ||
			    (state.cost == result.cost && input.nodes[node].stable_id < input.nodes[result.root].stable_id)) {
				result.valid = true;
				result.root = node;
				result.cost = state.cost;
			}
		}
		auto n = graph.members.size();
		result.parent.resize(n, DConstants::INVALID_INDEX);
		result.parent_edge.resize(n, DConstants::INVALID_INDEX);
		result.bottom_up_children.resize(n);
		result.top_down_children.resize(n);
		result.bottom_up_cardinality.resize(n);
		result.fully_reduced_cardinality.resize(n);
		result.order.push_back(result.root);
		for (idx_t i = 0; i < result.order.size(); i++) {
			auto node = result.order[i];
			auto parent = node == result.root ? n : result.parent[node];
			auto &state = Evaluate(node, parent);
			result.bottom_up_cardinality[node] = state.bottom;
			result.fully_reduced_cardinality[node] = state.full;
			result.bottom_up_children[node] = state.children;
			for (auto child : state.children) {
				result.parent[child] = node;
				for (idx_t e = 0; e < input.edges.size(); e++) {
					auto &edge = input.edges[e];
					if ((edge.left == node && edge.right == child) || (edge.right == node && edge.left == child)) {
						result.parent_edge[child] = e;
						break;
					}
				}
				result.order.push_back(child);
				if (Evaluate(child, node).required) {
					result.top_down_children[node].push_back(child);
				}
			}
		}
		return result;
	}
};

struct Solution {
	double cost = std::numeric_limits<double>::infinity();
	vector<Mask> folds;
	bool tvc = false;
};

struct FoldSearch {
	const ResultDBEnumerationInput &input;
	Graph original;
	ResultDBEnumerationCostProvider &cards;
	const ResultDBCostWeights &weights;
	bool use_tvc;
	idx_t limit;
	idx_t work = 0;
	idx_t candidates = 0;
	Mask requested = 0;
	ResultDBPaperPlan result;
	unordered_map<string, Solution> memo;
	unordered_map<string, double> fold_costs;

	FoldSearch(const ResultDBEnumerationInput &input, ResultDBEnumerationCostProvider &cards,
	           const ResultDBCostWeights &weights, bool use_tvc, idx_t limit)
	    : input(input), original(MakeGraph(input)), cards(cards), weights(weights), use_tvc(use_tvc), limit(limit) {
		for (idx_t i = 0; i < input.nodes.size(); i++) {
			if (input.nodes[i].requested) {
				requested |= original.members[i];
			}
		}
	}
	void Tick() {
		if (++work > limit) {
			throw InvalidInputException("Paper ResultDB enumeration work limit (%llu) exceeded", limit);
		}
	}
	//! Enumerate connected complements once, anchoring the left set at its smallest node.
	void Pairs(const Graph &g, Mask scope, const std::function<void(Mask, Mask)> &emit) {
		std::function<void(Mask, Mask)> extend = [&](Mask left, Mask excluded) {
			Tick();
			auto right = scope & ~left;
			if (right && g.Connected(right)) {
				emit(left, right);
			}
			for (auto frontier = g.Neighbors(left) & scope & ~excluded; frontier; frontier &= frontier - 1) {
				auto next = frontier & (~frontier + 1);
				extend(left | next, excluded);
				excluded |= next;
			}
		};
		extend(scope & (~scope + 1), 0);
	}
	double Join(Mask subset) {
		auto cached = result.join_costs.find(subset);
		if (cached != result.join_costs.end()) {
			return cached->second;
		}
		Tick();
		if (Count(subset) == 1) {
			cards.ReducedCardinality(subset, 0);
			return result.join_costs[subset] = 0;
		}
		if (Count(subset) > 20) {
			throw InvalidInputException("Paper ResultDB supports at most 20 relations per enumerated join");
		}
		double output = weights.join * cards.ReducedCardinality(subset, 0);
		double best = std::numeric_limits<double>::infinity();
		Mask split = 0;
		Pairs(original, subset, [&](Mask left, Mask right) {
			auto candidate = Join(left) + Join(right) + output;
			if (candidate < best || (candidate == best && left < split)) {
				best = candidate;
				split = left;
			}
		});
		if (!split) {
			throw InternalException("Paper ResultDB join subset is disconnected");
		}
		result.join_splits[subset] = split;
		return result.join_costs[subset] = best;
	}
	double Decompose(Mask fold) {
		if (Count(fold) == 1 || !(fold & requested)) {
			return 0;
		}
		return weights.decompose * Count(fold & requested) * cards.ReducedCardinality(fold, original.All() & ~fold);
	}
	string Key(const Graph &g, Mask block, Mask problem) const {
		string key;
		for (auto member : g.members) {
			key += std::to_string(member) + ",";
		}
		return key + ":" + std::to_string(block) + ":" + std::to_string(problem);
	}
	//! Algorithm 3: work at this fold only; neighboring subtrees are summarized by their occurrence sets.
	double FoldCost(const Graph &g, Mask problem, Mask fold) {
		auto key = Key(g, problem, fold);
		auto cached = fold_costs.find(key);
		if (cached != fold_costs.end()) {
			return cached->second;
		}
		Tick();
		auto left = g.Original(fold);
		auto whole_problem = g.Original(problem);
		struct Neighbor {
			Mask region;
			double rank;
		};
		vector<Neighbor> neighbors;
		auto outside = g.Components(g.All() & ~problem);
		for (auto component : outside) {
			if (g.Neighbors(fold) & component) {
				auto region = g.Original(component);
				neighbors.push_back({region, cards.ReducedCardinality(whole_problem, region)});
			}
		}
		auto other = problem & ~fold;
		if (other) {
			Mask region = g.Original(other);
			for (auto component : outside) {
				if (g.Neighbors(other) & component) {
					region |= g.Original(component);
				}
			}
			neighbors.push_back({region, cards.ReducedCardinality(left, region)});
		}
		std::sort(neighbors.begin(), neighbors.end(), [](const Neighbor &a, const Neighbor &b) {
			return a.rank != b.rank ? a.rank < b.rank : a.region < b.region;
		});
		double current = cards.ReducedCardinality(left, 0);
		double heuristic = 0;
		Mask by = 0;
		for (auto &neighbor : neighbors) {
			heuristic += weights.probe * current;
			by |= neighbor.region;
			current = std::min(current, cards.ReducedCardinality(left, by));
		}
		for (auto &neighbor : neighbors) {
			if (neighbor.region & requested) {
				// This heuristic counts this fold's tuples once per required outgoing reduction (paper Alg. 3).
				heuristic += weights.probe * current;
			}
		}
		return fold_costs[key] = Join(left) + Decompose(left) + heuristic;
	}
	static void Consider(Solution &best, Solution candidate) {
		std::sort(candidate.folds.begin(), candidate.folds.end());
		if (candidate.cost < best.cost || (candidate.cost == best.cost && candidate.folds < best.folds)) {
			best = std::move(candidate);
		}
	}
	Solution BEP(const Graph &g, Mask block, Mask problem) {
		auto key = Key(g, block, problem);
		auto found = memo.find(key);
		if (found != memo.end()) {
			return found->second;
		}
		Tick();
		candidates++;
		Solution best {FoldCost(g, problem, problem), {g.Original(problem)}, false};
		if (problem == block) {
			Pairs(g, problem, [&](Mask left, Mask right) {
				candidates++;
				Consider(best, {FoldCost(g, problem, left) + FoldCost(g, problem, right),
				                {g.Original(left), g.Original(right)},
				                false});
			});
		}
		if (use_tvc) {
			vector<Mask> cuts;
			for (auto nodes = problem; nodes; nodes &= nodes - 1) {
				auto a = First(nodes);
				for (auto others = g.edges[a] & problem; others; others &= others - 1) {
					auto b = First(others);
					if (a < b) {
						Tick();
						auto pair = Bit(a) | Bit(b);
						if (g.Components(block & ~pair).size() > 1) {
							cuts.push_back(pair);
						}
					}
				}
			}
			unordered_map<Mask, idx_t> overlaps;
			for (auto a : cuts) {
				for (auto b : cuts) {
					overlaps[a] += a != b && (a & b) ? 1 : 0;
				}
			}
			std::sort(cuts.begin(), cuts.end(), [&](Mask a, Mask b) {
				return overlaps[a] != overlaps[b] ? overlaps[a] < overlaps[b] : g.Original(a) < g.Original(b);
			});
			Mask used = 0;
			vector<Mask> groups;
			for (auto pair : cuts) {
				if (!(pair & used)) {
					groups.push_back(pair);
					used |= pair;
				}
			}
			if (used) {
				for (auto rest = g.All() & ~used; rest; rest &= rest - 1) {
					groups.push_back(rest & (~rest + 1));
				}
				std::sort(groups.begin(), groups.end(),
				          [&](Mask a, Mask b) { return First(g.Original(a)) < First(g.Original(b)); });
				auto contracted = g.Contract(groups);
				Mask scope = 0, excluded = 0;
				for (idx_t i = 0; i < groups.size(); i++) {
					if (groups[i] & block) {
						scope |= Bit(i);
					}
					if (groups[i] & (block & ~problem)) {
						excluded |= Bit(i);
					}
				}
				auto candidate = Problems(contracted, scope, excluded);
				candidate.tvc = true;
				Consider(best, std::move(candidate));
			}
		}
		memo[key] = best;
		return best;
	}
	Solution Problems(const Graph &g, Mask scope, Mask excluded) {
		Tick();
		auto blocks = g.Blocks(scope);
		vector<uint8_t> done(blocks.size(), 0);
		Solution result {0, {}, false};
		Mask covered = 0;
		for (auto block : blocks) {
			covered |= block;
		}
		for (idx_t start = 0; start < blocks.size(); start++) {
			if (done[start]) {
				continue;
			}
			vector<idx_t> component {start};
			done[start] = 1;
			for (idx_t i = 0; i < component.size(); i++) {
				for (idx_t j = 0; j < blocks.size(); j++) {
					if (!done[j] && (blocks[component[i]] & blocks[j])) {
						done[j] = 1;
						component.push_back(j);
					}
				}
			}
			Solution best;
			for (auto root : component) {
				// Traverse the bipartite block/articulation forest. All siblings lose the same parent cut vertex.
				vector<idx_t> pending {root};
				vector<uint8_t> visited(blocks.size(), 0);
				vector<Mask> unavailable(blocks.size(), excluded);
				visited[root] = 1;
				Solution candidate {0, {}, false};
				for (idx_t i = 0; i < pending.size(); i++) {
					auto block_id = pending[i];
					auto block = blocks[block_id];
					if (Count(block & unavailable[block_id]) > 1) {
						candidate.cost = std::numeric_limits<double>::infinity();
						break;
					}
					auto solution = BEP(g, block, block & ~unavailable[block_id]);
					candidate.cost += solution.cost;
					candidate.folds.insert(candidate.folds.end(), solution.folds.begin(), solution.folds.end());
					candidate.tvc = candidate.tvc || solution.tvc;
					for (auto other : component) {
						auto cut = block & blocks[other];
						if (!visited[other] && cut) {
							visited[other] = 1;
							unavailable[other] |= cut;
							pending.push_back(other);
						}
					}
				}
				Consider(best, std::move(candidate));
			}
			if (!std::isfinite(best.cost)) {
				throw InternalException("Paper ResultDB found no valid BEP assignment");
			}
			result.cost += best.cost;
			result.folds.insert(result.folds.end(), best.folds.begin(), best.folds.end());
			result.tvc = result.tvc || best.tvc;
		}
		for (auto remaining = scope & ~covered & ~excluded; remaining; remaining &= remaining - 1) {
			auto node = remaining & (~remaining + 1);
			result.cost += FoldCost(g, node, node);
			result.folds.push_back(g.Original(node));
		}
		return result;
	}
};
} // namespace resultdb_paper

ResultDBRootEnumerationResult ResultDBPlanEnumerator::EnumeratePaperRoot(const ResultDBEnumerationInput &input,
                                                                         ResultDBEnumerationCostProvider &cards,
                                                                         const ResultDBCostWeights &weights) {
	auto graph = resultdb_paper::MakeGraph(input);
	return resultdb_paper::RootSearch(input, graph, cards, weights).Run();
}

ResultDBPaperPlan ResultDBPlanEnumerator::EnumeratePaper(const ResultDBEnumerationInput &input,
                                                         ResultDBEnumerationCostProvider &cards,
	                                                         const ResultDBCostWeights &weights, bool use_tvc,
	                                                         idx_t work_limit) {
	using namespace resultdb_paper;
	resultdb_paper::FoldSearch search(input, cards, weights, use_tvc, work_limit);
	auto all = search.original.All();
	auto solution = search.Problems(search.original, all, 0);
	std::sort(solution.folds.begin(), solution.folds.end(), [](Mask a, Mask b) { return First(a) < First(b); });
	auto contracted = search.original.Contract(solution.folds);
	ResultDBEnumerationInput tree;
	for (idx_t i = 0; i < solution.folds.size(); i++) {
		auto fold = solution.folds[i];
		ResultDBEnumerationNode node;
		node.stable_id = input.nodes[First(fold)].stable_id;
		node.requested = (fold & search.requested) != 0;
		node.relations = fold;
		tree.nodes.push_back(node);
		search.result.join_cost += search.Join(fold);
		search.result.decomposition_cost += search.Decompose(fold);
		if (Count(fold) > 1) {
			vector<idx_t> group;
			for (auto remaining = fold; remaining; remaining &= remaining - 1) {
				group.push_back(First(remaining));
			}
			search.result.folds.push_back(std::move(group));
		}
		for (idx_t j = 0; j < i; j++) {
			if (contracted.edges[i] & Bit(j)) {
				tree.edges.push_back({j, i});
			}
		}
	}
	auto root = EnumeratePaperRoot(tree, cards, weights);
	search.result.reduction_cost = root.cost;
	static_cast<ResultDBRootEnumerationResult &>(search.result) = std::move(root);
	search.result.cost += search.result.join_cost + search.result.decomposition_cost;
	search.result.fold_cost = search.result.join_cost + search.result.decomposition_cost;
	search.result.used_tvc = solution.tvc;
	search.result.candidate_count = search.candidates;
	search.result.work = search.work;
	for (auto block : search.original.Blocks(all)) {
		search.result.block_sizes.push_back(Count(block));
	}
	return std::move(search.result);
}
} // namespace duckdb
