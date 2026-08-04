//===----------------------------------------------------------------------===//
//                         DuckDB
//
// resultdb_plan_enumerator.cpp
//
//===----------------------------------------------------------------------===//

#include "duckdb/optimizer/resultdb_plan_enumerator.hpp"

#include "duckdb/common/exception.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace duckdb {

namespace {

using EnumerationAdjacency = vector<vector<std::pair<idx_t, idx_t>>>;

static EnumerationAdjacency BuildAdjacency(const ResultDBEnumerationInput &input) {
	EnumerationAdjacency result(input.nodes.size());
	for (idx_t edge_idx = 0; edge_idx < input.edges.size(); edge_idx++) {
		auto &edge = input.edges[edge_idx];
		if (edge.left >= input.nodes.size() || edge.right >= input.nodes.size() || edge.left == edge.right) {
			throw InternalException("ResultDB enumeration edge has invalid endpoints");
		}
		result[edge.left].emplace_back(edge.right, edge_idx);
		result[edge.right].emplace_back(edge.left, edge_idx);
	}
	for (auto &neighbors : result) {
		std::sort(neighbors.begin(), neighbors.end(), [&](const auto &left, const auto &right) {
			auto left_id = input.nodes[left.first].stable_id;
			auto right_id = input.nodes[right.first].stable_id;
			return left_id != right_id ? left_id < right_id : left.second < right.second;
		});
	}
	return result;
}

static bool IsConnected(const ResultDBEnumerationInput &input, const EnumerationAdjacency &adjacency) {
	if (input.nodes.empty()) {
		return false;
	}
	vector<uint8_t> seen(input.nodes.size(), 0);
	vector<idx_t> pending {0};
	seen[0] = 1;
	for (idx_t pos = 0; pos < pending.size(); pos++) {
		for (auto &neighbor : adjacency[pending[pos]]) {
			if (!seen[neighbor.first]) {
				seen[neighbor.first] = 1;
				pending.push_back(neighbor.first);
			}
		}
	}
	return pending.size() == input.nodes.size();
}

static double ClampCardinality(double value) {
	if (!std::isfinite(value) || value < 0) {
		return 0;
	}
	return value;
}

struct RootState {
	double cost = 0;
	double cardinality = 0;
};

struct RootEvaluator {
	RootEvaluator(const ResultDBEnumerationInput &input_p, ResultDBEnumerationCostProvider &costs_p,
	              const EnumerationAdjacency &adjacency_p)
	    : input(input_p), costs(costs_p), adjacency(adjacency_p), bottom_up_children(input.nodes.size()),
	      top_down_children(input.nodes.size()), bottom_up_cardinality(input.nodes.size(), 0),
	      fully_reduced_cardinality(input.nodes.size(), 0), required_for_output(input.nodes.size(), 0) {
	}

	RootState BottomUp(idx_t node, idx_t parent) {
		struct ChildState {
			idx_t node;
			RootState state;
		};
		vector<ChildState> children;
		for (auto &neighbor : adjacency[node]) {
			if (neighbor.first == parent) {
				continue;
			}
			children.push_back({neighbor.first, BottomUp(neighbor.first, node)});
		}
		std::sort(children.begin(), children.end(), [&](const ChildState &left, const ChildState &right) {
			if (left.state.cardinality != right.state.cardinality) {
				return left.state.cardinality < right.state.cardinality;
			}
			return input.nodes[left.node].stable_id < input.nodes[right.node].stable_id;
		});

		RootState result;
		result.cardinality = ClampCardinality(input.nodes[node].cardinality);
		for (auto &child : children) {
			bottom_up_children[node].push_back(child.node);
			result.cost += child.state.cost;
			result.cost += 3.0 * child.state.cardinality + result.cardinality;
			result.cardinality = std::min(result.cardinality, child.state.cardinality);
		}
		bottom_up_cardinality[node] = result.cardinality;
		return result;
	}

	bool MarkRequired(idx_t node, idx_t parent) {
		bool required = input.nodes[node].requested;
		for (auto &neighbor : adjacency[node]) {
			if (neighbor.first != parent) {
				required = MarkRequired(neighbor.first, node) || required;
			}
		}
		required_for_output[node] = required;
		return required;
	}

	double TopDown(idx_t node, idx_t parent, double parent_cardinality) {
		double cost = 0;
		auto current = bottom_up_cardinality[node];
		if (parent != DConstants::INVALID_INDEX) {
			cost += 3.0 * parent_cardinality + current;
			current = std::min(current, parent_cardinality);
		}
		fully_reduced_cardinality[node] = current;
		vector<idx_t> children;
		for (auto &neighbor : adjacency[node]) {
			if (neighbor.first != parent && required_for_output[neighbor.first]) {
				children.push_back(neighbor.first);
			}
		}
		std::sort(children.begin(), children.end(), [&](idx_t left, idx_t right) {
			if (bottom_up_cardinality[left] != bottom_up_cardinality[right]) {
				return bottom_up_cardinality[left] < bottom_up_cardinality[right];
			}
			return input.nodes[left].stable_id < input.nodes[right].stable_id;
		});
		top_down_children[node] = children;
		for (auto child : children) {
			cost += TopDown(child, node, current);
		}
		return cost;
	}

	const ResultDBEnumerationInput &input;
	ResultDBEnumerationCostProvider &costs;
	const EnumerationAdjacency &adjacency;
	vector<vector<idx_t>> bottom_up_children;
	vector<vector<idx_t>> top_down_children;
	vector<double> bottom_up_cardinality;
	vector<double> fully_reduced_cardinality;
	vector<uint8_t> required_for_output;
};

static void BuildRootedOrder(const ResultDBEnumerationInput &input, const EnumerationAdjacency &adjacency,
	                         const vector<vector<idx_t>> &children, idx_t root, ResultDBRootEnumerationResult &result) {
	result.parent.assign(input.nodes.size(), DConstants::INVALID_INDEX);
	result.parent_edge.assign(input.nodes.size(), DConstants::INVALID_INDEX);
	result.order.clear();
	result.order.push_back(root);
	for (idx_t pos = 0; pos < result.order.size(); pos++) {
		auto node = result.order[pos];
		for (auto child : children[node]) {
			idx_t edge_idx = DConstants::INVALID_INDEX;
			for (auto &neighbor : adjacency[node]) {
				if (neighbor.first == child) {
					edge_idx = neighbor.second;
					break;
				}
			}
			if (edge_idx == DConstants::INVALID_INDEX) {
				throw InternalException("ResultDB root enumeration lost a tree edge");
			}
			result.parent[child] = node;
			result.parent_edge[child] = edge_idx;
			result.order.push_back(child);
		}
	}
}

static bool BetterRoot(const ResultDBRootEnumerationResult &candidate, const ResultDBRootEnumerationResult &best,
	                   const ResultDBEnumerationInput &input) {
	if (!best.valid || candidate.cost < best.cost) {
		return true;
	}
	if (candidate.cost > best.cost) {
		return false;
	}
	return input.nodes[candidate.root].stable_id < input.nodes[best.root].stable_id;
}

struct TarjanState {
	const ResultDBEnumerationInput &input;
	const EnumerationAdjacency &adjacency;
	vector<idx_t> discovery;
	vector<idx_t> low;
	vector<std::pair<idx_t, idx_t>> edge_stack;
	vector<vector<idx_t>> blocks;
	idx_t clock = 0;

	void Visit(idx_t node, idx_t parent_edge) {
		discovery[node] = low[node] = ++clock;
		for (auto &neighbor : adjacency[node]) {
			auto next = neighbor.first;
			auto edge_idx = neighbor.second;
			if (edge_idx == parent_edge) {
				continue;
			}
			if (!discovery[next]) {
				edge_stack.emplace_back(node, next);
				Visit(next, edge_idx);
				low[node] = std::min(low[node], low[next]);
				if (low[next] >= discovery[node]) {
					vector<idx_t> block;
					while (!edge_stack.empty()) {
						auto edge = edge_stack.back();
						edge_stack.pop_back();
						block.push_back(edge.first);
						block.push_back(edge.second);
						if (edge.first == node && edge.second == next) {
							break;
						}
					}
					std::sort(block.begin(), block.end());
					block.erase(std::unique(block.begin(), block.end()), block.end());
					if (block.size() >= 3) {
						blocks.push_back(std::move(block));
					}
				}
			} else if (discovery[next] < discovery[node]) {
				edge_stack.emplace_back(node, next);
				low[node] = std::min(low[node], discovery[next]);
			}
		}
	}
};

static vector<vector<idx_t>> FindBlocks(const ResultDBEnumerationInput &input) {
	auto adjacency = BuildAdjacency(input);
	TarjanState state {input, adjacency, vector<idx_t>(input.nodes.size(), 0), vector<idx_t>(input.nodes.size(), 0)};
	for (idx_t node = 0; node < input.nodes.size(); node++) {
		if (!state.discovery[node]) {
			state.Visit(node, DConstants::INVALID_INDEX);
		}
	}
	return state.blocks;
}

static string PartitionKey(const vector<vector<idx_t>> &partition) {
	string result;
	for (auto &group : partition) {
		result += "[";
		for (auto node : group) {
			result += std::to_string(node) + ",";
		}
		result += "]";
	}
	return result;
}

static void NormalizePartition(vector<vector<idx_t>> &partition) {
	for (auto &group : partition) {
		std::sort(group.begin(), group.end());
		group.erase(std::unique(group.begin(), group.end()), group.end());
	}
	std::sort(partition.begin(), partition.end(), [](const auto &left, const auto &right) {
		return left.front() != right.front() ? left.front() < right.front() : left < right;
	});
}

static vector<vector<idx_t>> MergeGroups(const vector<vector<idx_t>> &partition, const vector<idx_t> &group_indexes) {
	vector<uint8_t> merge(partition.size(), 0);
	vector<idx_t> combined;
	for (auto group_idx : group_indexes) {
		merge[group_idx] = 1;
		combined.insert(combined.end(), partition[group_idx].begin(), partition[group_idx].end());
	}
	vector<vector<idx_t>> result;
	for (idx_t group_idx = 0; group_idx < partition.size(); group_idx++) {
		if (!merge[group_idx]) {
			result.push_back(partition[group_idx]);
		}
	}
	result.push_back(std::move(combined));
	NormalizePartition(result);
	return result;
}

static vector<vector<idx_t>> MergeTwoSides(const vector<vector<idx_t>> &partition, const vector<idx_t> &left,
	                                       const vector<idx_t> &right) {
	auto result = partition;
	if (left.size() > 1) {
		result = MergeGroups(result, left);
		// Rebuild the right-side indexes after normalization.
		vector<idx_t> right_nodes;
		for (auto group_idx : right) {
			right_nodes.insert(right_nodes.end(), partition[group_idx].begin(), partition[group_idx].end());
		}
		vector<idx_t> mapped;
		for (idx_t group_idx = 0; group_idx < result.size(); group_idx++) {
			for (auto node : result[group_idx]) {
				if (std::find(right_nodes.begin(), right_nodes.end(), node) != right_nodes.end()) {
					mapped.push_back(group_idx);
					break;
				}
			}
		}
		if (mapped.size() > 1) {
			result = MergeGroups(result, mapped);
		}
	} else if (right.size() > 1) {
		result = MergeGroups(result, right);
	}
	return result;
}

static ResultDBEnumerationInput Contract(const ResultDBEnumerationInput &input,
	                                     ResultDBEnumerationCostProvider &costs,
	                                     const vector<vector<idx_t>> &partition) {
	ResultDBEnumerationInput result;
	vector<idx_t> node_to_group(input.nodes.size(), DConstants::INVALID_INDEX);
	for (idx_t group_idx = 0; group_idx < partition.size(); group_idx++) {
		ResultDBEnumerationNode node;
		node.stable_id = DConstants::INVALID_INDEX;
		for (auto original : partition[group_idx]) {
			node_to_group[original] = group_idx;
			node.requested = node.requested || input.nodes[original].requested;
			node.stable_id = std::min(node.stable_id, input.nodes[original].stable_id);
		}
		node.cardinality = ClampCardinality(costs.Cardinality(partition[group_idx]));
		result.nodes.push_back(node);
	}
	for (auto &edge : input.edges) {
		auto left = node_to_group[edge.left];
		auto right = node_to_group[edge.right];
		if (left == right) {
			continue;
		}
		if (left > right) {
			std::swap(left, right);
		}
		bool exists = false;
		for (auto &current : result.edges) {
			if (current.left == left && current.right == right) {
				exists = true;
				break;
			}
		}
		if (!exists) {
			result.edges.push_back({left, right});
		}
	}
	return result;
}

static bool SubgraphConnected(const vector<idx_t> &nodes, const EnumerationAdjacency &adjacency,
	                          const vector<uint8_t> &in_block) {
	if (nodes.empty()) {
		return false;
	}
	vector<uint8_t> seen(in_block.size(), 0);
	vector<idx_t> pending {nodes[0]};
	seen[nodes[0]] = 1;
	for (idx_t pos = 0; pos < pending.size(); pos++) {
		for (auto &neighbor : adjacency[pending[pos]]) {
			if (in_block[neighbor.first] && !seen[neighbor.first]) {
				seen[neighbor.first] = 1;
				pending.push_back(neighbor.first);
			}
		}
	}
	return pending.size() == nodes.size();
}

static bool IsTwoVertexCut(const vector<idx_t> &block, idx_t left, idx_t right,
	                      const EnumerationAdjacency &adjacency) {
	vector<uint8_t> allowed(adjacency.size(), 0);
	vector<idx_t> remaining;
	for (auto node : block) {
		if (node != left && node != right) {
			allowed[node] = 1;
			remaining.push_back(node);
		}
	}
	if (remaining.size() < 2) {
		return false;
	}
	return !SubgraphConnected(remaining, adjacency, allowed);
}

static double FoldJoinCost(const ResultDBEnumerationInput &input, ResultDBEnumerationCostProvider &costs,
	                       const vector<idx_t> &nodes) {
	if (nodes.size() < 2) {
		return 0;
	}
	if (nodes.size() > 62) {
		throw InvalidInputException("ResultDB fold contains too many relations for exact enumeration");
	}
	auto adjacency = BuildAdjacency(input);
	uint64_t full = 0;
	for (auto node : nodes) {
		full |= uint64_t(1) << node;
	}
	unordered_map<uint64_t, double> dp;
	for (auto node : nodes) {
		dp[uint64_t(1) << node] = 0;
	}
	for (idx_t size = 2; size <= nodes.size(); size++) {
		for (uint64_t subset = full; subset; subset = (subset - 1) & full) {
			uint64_t bits = subset;
			idx_t bit_count = 0;
			while (bits) {
				bits &= bits - 1;
				bit_count++;
			}
			if (bit_count != size) {
				continue;
			}
			double best = std::numeric_limits<double>::infinity();
			for (uint64_t left = (subset - 1) & subset; left; left = (left - 1) & subset) {
				auto right = subset ^ left;
				if (!right || left > right || dp.find(left) == dp.end() || dp.find(right) == dp.end()) {
					continue;
				}
				bool joined = false;
				for (auto &edge : input.edges) {
					if (((left >> edge.left) & 1) != ((left >> edge.right) & 1) &&
					    ((subset >> edge.left) & 1) && ((subset >> edge.right) & 1)) {
						joined = true;
						break;
					}
				}
				if (!joined) {
					continue;
				}
				vector<idx_t> subset_nodes;
				for (auto node : nodes) {
					if ((subset >> node) & 1) {
						subset_nodes.push_back(node);
					}
				}
				best = std::min(best, dp[left] + dp[right] + costs.Cardinality(subset_nodes));
			}
			if (std::isfinite(best)) {
				dp[subset] = best;
			}
		}
	}
	auto entry = dp.find(full);
	return entry == dp.end() ? std::numeric_limits<double>::infinity() : entry->second;
}

struct FoldSearch {
	const ResultDBEnumerationInput &input;
	ResultDBEnumerationCostProvider &costs;
	bool use_tvc;
	idx_t work_limit;
	unordered_set<string> visited;
	ResultDBFoldEnumerationResult best;

	void Evaluate(const vector<vector<idx_t>> &partition) {
		if (visited.size() >= work_limit) {
			throw InvalidInputException("ResultDB fold enumeration exceeded its work limit of %llu candidates",
			                            work_limit);
		}
		auto key = PartitionKey(partition);
		if (!visited.insert(key).second) {
			return;
		}
		auto contracted = Contract(input, costs, partition);
		auto adjacency = BuildAdjacency(contracted);
		if (!IsConnected(contracted, adjacency)) {
			return;
		}
		if (contracted.edges.size() + 1 == contracted.nodes.size()) {
			auto root = ResultDBPlanEnumerator::EnumerateRoot(contracted, costs);
			if (!root.valid) {
				return;
			}
			double fold_cost = 0;
			vector<vector<idx_t>> folds;
			for (auto &group : partition) {
				if (group.size() <= 1) {
					continue;
				}
				folds.push_back(group);
				auto join_cost = FoldJoinCost(input, costs, group);
				fold_cost += join_cost + costs.Cardinality(group) * costs.PayloadWidth(group);
			}
			auto total = root.cost + fold_cost;
			if (!best.valid || total < best.cost ||
			    (total == best.cost && PartitionKey(partition) < PartitionKey(best.folds))) {
				static_cast<ResultDBRootEnumerationResult &>(best) = std::move(root);
				best.folds = std::move(folds);
				best.fold_cost = fold_cost;
				best.cost = total;
				best.used_tvc = use_tvc;
			}
			return;
		}

		auto blocks = FindBlocks(contracted);
		if (blocks.empty()) {
			return;
		}
		for (auto &block : blocks) {
			// SC1F: fold the complete block-enumeration problem.
			Evaluate(MergeGroups(partition, block));

			// SC2F: enumerate connected top-level complement pairs.
			if (block.size() <= 20) {
				auto block_adjacency = BuildAdjacency(contracted);
				auto combinations = uint64_t(1) << block.size();
				for (uint64_t mask = 1; mask + 1 < combinations; mask++) {
					if (!(mask & 1)) {
						continue;
					}
					vector<idx_t> left;
					vector<idx_t> right;
					vector<uint8_t> in_left(contracted.nodes.size(), 0);
					vector<uint8_t> in_right(contracted.nodes.size(), 0);
					for (idx_t pos = 0; pos < block.size(); pos++) {
						if ((mask >> pos) & 1) {
							left.push_back(block[pos]);
							in_left[block[pos]] = 1;
						} else {
							right.push_back(block[pos]);
							in_right[block[pos]] = 1;
						}
					}
					if (SubgraphConnected(left, block_adjacency, in_left) &&
					    SubgraphConnected(right, block_adjacency, in_right)) {
						Evaluate(MergeTwoSides(partition, left, right));
					}
				}
			}

			if (use_tvc) {
				auto block_adjacency = BuildAdjacency(contracted);
				vector<std::pair<idx_t, idx_t>> cuts;
				for (auto left : block) {
					for (auto &neighbor : block_adjacency[left]) {
						auto right = neighbor.first;
						if (left < right && std::binary_search(block.begin(), block.end(), right) &&
						    IsTwoVertexCut(block, left, right, block_adjacency)) {
							cuts.emplace_back(left, right);
						}
					}
				}
				unordered_map<idx_t, idx_t> overlaps;
				for (auto &cut : cuts) {
					overlaps[cut.first]++;
					overlaps[cut.second]++;
				}
				std::sort(cuts.begin(), cuts.end(), [&](const auto &left, const auto &right) {
					auto left_overlap = overlaps[left.first] + overlaps[left.second];
					auto right_overlap = overlaps[right.first] + overlaps[right.second];
					if (left_overlap != right_overlap) {
						return left_overlap < right_overlap;
					}
					return left < right;
				});
				vector<uint8_t> used(contracted.nodes.size(), 0);
					auto tvc_partition = partition;
				for (auto &cut : cuts) {
					if (used[cut.first] || used[cut.second]) {
						continue;
					}
					used[cut.first] = used[cut.second] = 1;
					vector<idx_t> original_nodes = partition[cut.first];
					original_nodes.insert(original_nodes.end(), partition[cut.second].begin(),
					                              partition[cut.second].end());
					vector<idx_t> mapped;
					for (idx_t group_idx = 0; group_idx < tvc_partition.size(); group_idx++) {
						bool contains = false;
						for (auto node : tvc_partition[group_idx]) {
							if (std::find(original_nodes.begin(), original_nodes.end(), node) != original_nodes.end()) {
								contains = true;
								break;
							}
						}
						if (contains) {
							mapped.push_back(group_idx);
						}
					}
					if (mapped.size() > 1) {
						tvc_partition = MergeGroups(tvc_partition, mapped);
					}
				}
				if (tvc_partition != partition) {
					Evaluate(tvc_partition);
				}
			}
		}
	}
};

} // namespace

ResultDBRootEnumerationResult ResultDBPlanEnumerator::EnumerateRoot(const ResultDBEnumerationInput &input,
	                                                               ResultDBEnumerationCostProvider &costs) {
	ResultDBRootEnumerationResult best;
	if (input.nodes.empty() || input.edges.size() + 1 != input.nodes.size()) {
		best.error = "TDRoot requires a nonempty connected acyclic graph";
		return best;
	}
	auto adjacency = BuildAdjacency(input);
	if (!IsConnected(input, adjacency)) {
		best.error = "TDRoot requires a connected graph";
		return best;
	}

	for (idx_t root = 0; root < input.nodes.size(); root++) {
		RootEvaluator evaluator(input, costs, adjacency);
		auto bottom_up = evaluator.BottomUp(root, DConstants::INVALID_INDEX);
		evaluator.MarkRequired(root, DConstants::INVALID_INDEX);
		auto top_down_cost = evaluator.TopDown(root, DConstants::INVALID_INDEX, bottom_up.cardinality);

		ResultDBRootEnumerationResult candidate;
		candidate.valid = true;
		candidate.root = root;
		candidate.cost = bottom_up.cost + top_down_cost;
		candidate.bottom_up_children = evaluator.bottom_up_children;
		candidate.top_down_children = evaluator.top_down_children;
		candidate.bottom_up_cardinality = evaluator.bottom_up_cardinality;
		candidate.fully_reduced_cardinality = evaluator.fully_reduced_cardinality;
		BuildRootedOrder(input, adjacency, candidate.bottom_up_children, root, candidate);
		if (candidate.order.size() != input.nodes.size()) {
			throw InternalException("TDRoot produced an incomplete relation order");
		}
		if (BetterRoot(candidate, best, input)) {
			best = std::move(candidate);
		}
	}
	return best;
}

ResultDBFoldEnumerationResult ResultDBPlanEnumerator::EnumerateFolds(const ResultDBEnumerationInput &input,
	                                                                ResultDBEnumerationCostProvider &costs,
	                                                                bool use_tvc, idx_t work_limit) {
	ResultDBFoldEnumerationResult result;
	if (input.nodes.size() < 2) {
		result.error = "TDFold requires at least two relations";
		return result;
	}
	if (input.nodes.size() > 62) {
		result.error = "TDFold supports at most 62 relation occurrences";
		return result;
	}
	auto adjacency = BuildAdjacency(input);
	if (!IsConnected(input, adjacency)) {
		result.error = "TDFold requires a connected graph";
		return result;
	}
	auto blocks = FindBlocks(input);
	for (auto &block : blocks) {
		result.block_sizes.push_back(block.size());
	}
	vector<vector<idx_t>> partition;
	for (idx_t node = 0; node < input.nodes.size(); node++) {
		partition.push_back({node});
	}
	FoldSearch search {input, costs, use_tvc, work_limit};
	search.best.block_sizes = result.block_sizes;
	search.Evaluate(partition);
	search.best.candidate_count = search.visited.size();
	search.best.block_sizes = std::move(result.block_sizes);
	if (!search.best.valid) {
		search.best.error = "TDFold did not produce an acyclic contracted graph";
	}
	return std::move(search.best);
}

double ResultDBPlanEnumerator::EstimateJoinCost(const ResultDBEnumerationInput &input,
	                                            ResultDBEnumerationCostProvider &costs,
	                                            const vector<idx_t> &nodes) {
	return FoldJoinCost(input, costs, nodes);
}

} // namespace duckdb
