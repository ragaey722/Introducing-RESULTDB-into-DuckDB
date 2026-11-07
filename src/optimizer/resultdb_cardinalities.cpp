#include "duckdb/optimizer/resultdb_cardinalities.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/settings.hpp"
#include "mbedtls_wrapper.hpp"
#include "yyjson.hpp"

#include <algorithm>
#include <cmath>

namespace duckdb {
namespace resultdb_json {
using namespace duckdb_yyjson;

static vector<string> Names(yyjson_val *value, const string &path) {
	if (!yyjson_is_arr(value) || yyjson_arr_size(value) == 0) {
		throw InvalidInputException("ResultDB cardinality file %s: expected a nonempty relation-name array", path);
	}
	vector<string> result;
	for (idx_t i = 0; i < yyjson_arr_size(value); i++) {
		auto item = yyjson_arr_get(value, i);
		if (!yyjson_is_str(item) || yyjson_get_len(item) == 0) {
			throw InvalidInputException("ResultDB cardinality file %s: relation names must be nonempty strings", path);
		}
		result.emplace_back(yyjson_get_str(item), yyjson_get_len(item));
	}
	std::sort(result.begin(), result.end());
	if (std::adjacent_find(result.begin(), result.end()) != result.end()) {
		throw InvalidInputException("ResultDB cardinality file %s: duplicate relation name", path);
	}
	return result;
}

static double Size(yyjson_val *value, const string &path) {
	if (!yyjson_is_num(value)) {
		throw InvalidInputException("ResultDB cardinality file %s: size must be numeric", path);
	}
	double size = yyjson_get_num(value);
	if (!std::isfinite(size) || size < 0) {
		throw InvalidInputException("ResultDB cardinality file %s: size must be finite and nonnegative", path);
	}
	return size;
}

static void UniqueFields(yyjson_val *object, const string &path) {
	if (!yyjson_is_obj(object)) {
		throw InvalidInputException("ResultDB cardinality file %s: expected an object", path);
	}
	unordered_set<string> fields;
	auto iter = yyjson_obj_iter_with(object);
	while (auto key = yyjson_obj_iter_next(&iter)) {
		if (!fields.insert(string(yyjson_get_str(key), yyjson_get_len(key))).second) {
			throw InvalidInputException("ResultDB cardinality file %s: duplicate JSON field", path);
		}
	}
}
} // namespace resultdb_json

shared_ptr<const ResultDBCardinalityFile> ResultDBCardinalityFile::Parse(const string &json, const string &path) {
	using namespace duckdb_yyjson;
	yyjson_read_err error;
	unique_ptr<yyjson_doc, void (*)(yyjson_doc *)> doc(
	    yyjson_read_opts(const_cast<char *>(json.data()), json.size(), YYJSON_READ_NOFLAG, nullptr, &error),
	    yyjson_doc_free);
	if (!doc) {
		throw InvalidInputException("ResultDB cardinality file %s: invalid JSON at byte %llu: %s", path,
		                            idx_t(error.pos), error.msg);
	}
	auto result = make_shared_ptr<ResultDBCardinalityFile>();
	result->path = path;
	auto hash = duckdb_mbedtls::MbedTlsWrapper::ComputeSha256Hash(json);
	static const char hex[] = "0123456789abcdef";
	for (auto byte : hash) {
		result->sha256 += hex[(static_cast<unsigned char>(byte) >> 4) & 15];
		result->sha256 += hex[static_cast<unsigned char>(byte) & 15];
	}
	auto root = yyjson_doc_get_root(doc.get());
	resultdb_json::UniqueFields(root, path);
	auto datasets = yyjson_obj_iter_with(root);
	while (auto key = yyjson_obj_iter_next(&datasets)) {
		auto rows = yyjson_obj_iter_get_val(key);
		if (!yyjson_is_arr(rows)) {
			throw InvalidInputException("ResultDB cardinality file %s: dataset must contain an array", path);
		}
		auto &entries = result->datasets[string(yyjson_get_str(key), yyjson_get_len(key))];
		// Length-prefix names so aliases containing separators cannot collide.
		unordered_map<string, double> seen;
		auto add = [&](Entry entry) {
			string identity;
			for (auto &name : entry.relations) {
				identity += std::to_string(name.size()) + ":" + name;
			}
			identity += "|";
			for (auto &name : entry.reduced_by) {
				if (std::binary_search(entry.relations.begin(), entry.relations.end(), name)) {
					throw InvalidInputException("ResultDB cardinality file %s: reduction sets overlap", path);
				}
				identity += std::to_string(name.size()) + ":" + name;
			}
			auto inserted = seen.emplace(identity, entry.cardinality);
			if (!inserted.second && inserted.first->second != entry.cardinality) {
				throw InvalidInputException("ResultDB cardinality file %s: conflicting cardinalities", path);
			}
			if (inserted.second) {
				entries.push_back(std::move(entry));
			}
		};
		for (idx_t i = 0; i < yyjson_arr_size(rows); i++) {
			auto row = yyjson_arr_get(rows, i);
			resultdb_json::UniqueFields(row, path);
			auto names = resultdb_json::Names(yyjson_obj_get(row, "relations"), path);
			auto size = resultdb_json::Size(yyjson_obj_get(row, "size"), path);
			add({names, {}, size});
			auto reductions = yyjson_obj_get(row, "reductions");
			if (!reductions) {
				continue;
			}
			if (!yyjson_is_arr(reductions)) {
				throw InvalidInputException("ResultDB cardinality file %s: reductions must be an array", path);
			}
			for (idx_t j = 0; j < yyjson_arr_size(reductions); j++) {
				auto reduction = yyjson_arr_get(reductions, j);
				resultdb_json::UniqueFields(reduction, path);
				auto reduced_size = resultdb_json::Size(yyjson_obj_get(reduction, "size"), path);
				if (reduced_size > size) {
					throw InvalidInputException("ResultDB cardinality file %s: reduction exceeds unreduced size", path);
				}
				add({names, resultdb_json::Names(yyjson_obj_get(reduction, "right_relations"), path), reduced_size});
			}
		}
	}
	return result;
}

shared_ptr<const ResultDBCardinalityFile> ResultDBCardinalityFile::Load(ClientContext &context, const string &path) {
	if (!Settings::Get<EnableExternalAccessSetting>(context)) {
		throw PermissionException("Loading ResultDB cardinalities requires external file access");
	}
	auto &fs = FileSystem::GetFileSystem(context);
	auto handle = fs.OpenFile(path, FileFlags::FILE_FLAGS_READ);
	string json(handle->GetFileSize(), '\0');
	if (!json.empty()) {
		handle->Read(&json[0], json.size(), 0);
	}
	return Parse(json, path);
}

ResultDBInjectedCardinalities::ResultDBInjectedCardinalities(shared_ptr<const ResultDBCardinalityFile> file_p,
                                                             const string &dataset, vector<string> aliases_p)
    : aliases(std::move(aliases_p)), file(std::move(file_p)) {
	if (!file || aliases.empty() || aliases.size() > 62) {
		throw InvalidInputException("Paper ResultDB requires a cardinality file and 1..62 relation occurrences");
	}
	auto entry = file->datasets.find(dataset);
	if (entry == file->datasets.end()) {
		throw InvalidInputException("ResultDB cardinality dataset '%s' not found in %s", dataset, file->path);
	}
	unordered_map<string, idx_t> positions;
	for (idx_t i = 0; i < aliases.size(); i++) {
		if (!positions.emplace(aliases[i], i).second) {
			throw InvalidInputException("Paper ResultDB requires unambiguous occurrence aliases: %s", aliases[i]);
		}
	}
	auto mask = [&](const vector<string> &names) {
		uint64_t result = 0;
		for (auto &name : names) {
			auto pos = positions.find(name);
			if (pos == positions.end()) {
				throw InvalidInputException("ResultDB cardinality alias '%s' does not occur in this query", name);
			}
			result |= uint64_t(1) << pos->second;
		}
		return result;
	};
	for (auto &value : entry->second) {
		values[mask(value.relations)][mask(value.reduced_by)] = value.cardinality;
	}
}

string ResultDBInjectedCardinalities::Describe(uint64_t mask) const {
	vector<string> names;
	for (idx_t i = 0; i < aliases.size(); i++) {
		if (mask & (uint64_t(1) << i)) {
			names.push_back(aliases[i]);
		}
	}
	return "[" + StringUtil::Join(names, ",") + "]";
}

double ResultDBInjectedCardinalities::ReducedCardinality(uint64_t relations, uint64_t reduced_by) {
	lookups++;
	if (!relations || ((relations | reduced_by) >> aliases.size()) || (relations & reduced_by)) {
		throw InternalException("Invalid ResultDB cardinality occurrence masks");
	}
	auto left = values.find(relations);
	if (left != values.end()) {
		auto right = left->second.find(reduced_by);
		if (right != left->second.end()) {
			return right->second;
		}
	}
	throw InvalidInputException("Missing ResultDB cardinality: relations=%s, right_relations=%s in %s",
	                            Describe(relations), Describe(reduced_by), file->path);
}

double ResultDBInjectedCardinalities::Cardinality(const vector<idx_t> &nodes) {
	uint64_t mask = 0;
	for (auto node : nodes) {
		if (node >= aliases.size()) {
			throw InternalException("Invalid ResultDB cardinality node");
		}
		mask |= uint64_t(1) << node;
	}
	return ReducedCardinality(mask, 0);
}

double ResultDBInjectedCardinalities::PayloadWidth(const vector<idx_t> &) {
	throw InternalException("Paper ResultDB costs count output relations, not payload bytes");
}
} // namespace duckdb
