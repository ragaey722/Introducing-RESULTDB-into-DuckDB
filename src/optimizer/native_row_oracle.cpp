#include "duckdb/optimizer/native_row_oracle.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/schema_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/settings.hpp"
#include "duckdb/optimizer/join_order/query_graph_manager.hpp"
#include "duckdb/planner/expression/list.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/list.hpp"
#include "duckdb/planner/operator/logical_column_data_get.hpp"
#include "mbedtls_wrapper.hpp"
#include "yyjson.hpp"

#include <algorithm>
#include <cmath>

namespace duckdb {
namespace native_oracle_json {
using namespace duckdb_yyjson;
static void Object(yyjson_val *value) {
	if (!yyjson_is_obj(value)) {
		throw InvalidInputException("Native row oracle: expected JSON object");
	}
	unordered_set<string> fields;
	auto iter = yyjson_obj_iter_with(value);
	while (auto key = yyjson_obj_iter_next(&iter)) {
		if (!fields.insert(yyjson_get_str(key)).second) {
			throw InvalidInputException("Native row oracle: duplicate JSON field %s", yyjson_get_str(key));
		}
	}
}
static string Text(yyjson_val *value, const char *name) {
	auto item = yyjson_obj_get(value, name);
	if (!yyjson_is_str(item) || !yyjson_get_len(item)) {
		throw InvalidInputException("Native row oracle: missing/non-string field %s", name);
	}
	return string(yyjson_get_str(item), yyjson_get_len(item));
}
static idx_t Integer(yyjson_val *value, const char *name, uint64_t maximum = 9007199254740991ULL) {
	auto item = yyjson_obj_get(value, name);
	if (!yyjson_is_uint(item) || yyjson_get_uint(item) > maximum) {
		throw InvalidInputException("Native row oracle: %s must be an integer in [0,%llu]", name, maximum);
	}
	return yyjson_get_uint(item);
}
static void SHA(const string &value) {
	if (value.size() != 64 || value.find_first_not_of("0123456789abcdef") != string::npos) {
		throw InvalidInputException("Native row oracle: invalid SHA256");
	}
}
static string Quote(const string &value) {
	string result = "\"";
	for (auto c : value) {
		switch (c) {
		case '\\': result += "\\\\"; break;
		case '"': result += "\\\""; break;
		case '\n': result += "\\n"; break;
		case '\r': result += "\\r"; break;
		case '\t': result += "\\t"; break;
		default:
			if (static_cast<unsigned char>(c) < 32) {
				static const char hex[] = "0123456789abcdef";
				result += "\\u00";
				result += hex[(static_cast<unsigned char>(c) >> 4) & 15];
				result += hex[static_cast<unsigned char>(c) & 15];
			} else {
				result += c;
			}
		}
	}
	return result + "\"";
}
static string Identifier(const string &name) {
	return "\"" + StringUtil::Replace(name, "\"", "\"\"") + "\"";
}
} // namespace native_oracle_json

string NativeRowOracleFile::Hash(const string &text) {
	duckdb_mbedtls::MbedTlsWrapper::SHA256State hash;
	hash.AddString(text);
	char result[65];
	hash.FinishHex(result);
	return string(result, 64);
}
string NativeRowOracleFile::Key(const string &query, const string &schema, const string &sql) {
	return Hash("native-row-v1|" + query + "|" + schema + "|" + sql);
}

shared_ptr<const NativeRowOracleFile> NativeRowOracleFile::Parse(const string &json, const string &path) {
	using namespace duckdb_yyjson;
	using namespace native_oracle_json;
	yyjson_read_err error;
	unique_ptr<yyjson_doc, void (*)(yyjson_doc *)> doc(
	    yyjson_read_opts(const_cast<char *>(json.data()), json.size(), YYJSON_READ_NOFLAG, nullptr, &error), yyjson_doc_free);
	if (!doc) {
		throw InvalidInputException("Native row oracle %s: invalid JSON at byte %llu", path, idx_t(error.pos));
	}
	auto root = yyjson_doc_get_root(doc.get());
	Object(root);
	if (Integer(root, "version") != 1) {
		throw InvalidInputException("Native row oracle: unsupported profile version");
	}
	auto result = make_shared_ptr<NativeRowOracleFile>();
	result->path = path;
	result->sha256 = Hash(json);
	result->database_path = Text(root, "database_path");
	result->database_sha256 = Text(root, "database_sha256");
	SHA(result->database_sha256);
	result->database_size = Integer(root, "database_size");
	result->generator = Text(root, "generator");
	auto rows = yyjson_obj_get(root, "entries");
	if (!yyjson_is_arr(rows)) {
		throw InvalidInputException("Native row oracle: entries must be an array");
	}
	for (idx_t i = 0; i < yyjson_arr_size(rows); i++) {
		auto row = yyjson_arr_get(rows, i);
		Object(row);
		Entry entry;
		entry.query_sha256 = Text(row, "query_sha256");
		SHA(entry.query_sha256);
		entry.schema = Text(row, "schema");
		entry.sql = Text(row, "sql");
		// Preserve the exact COUNT(*) integer in the profile and audit. Native
		// costing still converts to double in its existing cardinality paths.
		entry.cardinality = Integer(row, "cardinality", 9223372036854775807ULL);
		auto key = Text(row, "key");
		if (key != Key(entry.query_sha256, entry.schema, entry.sql)) {
			throw InvalidInputException("Native row oracle: request fingerprint mismatch %s", key);
		}
		if (!result->entries.emplace(key, std::move(entry)).second) {
			throw InvalidInputException("Native row oracle: duplicate/conflicting request %s", key);
		}
	}
	return result;
}

shared_ptr<const NativeRowOracleFile> NativeRowOracleFile::Load(ClientContext &context, const string &path) {
	if (!Settings::Get<EnableExternalAccessSetting>(context)) {
		throw PermissionException("Loading native oracle statistics requires external file access");
	}
	auto &fs = FileSystem::GetFileSystem(context);
	auto handle = fs.OpenFile(path, FileFlags::FILE_FLAGS_READ);
	string json(handle->GetFileSize(), '\0');
	if (!json.empty()) {
		handle->Read(&json[0], json.size(), 0);
	}
	// Verify the complete frozen database once at SET time, outside preparation timing.
	auto parsed = Parse(json, path);
	auto result = make_shared_ptr<NativeRowOracleFile>(*parsed);
	result->database_path = fs.ConvertSeparators(result->database_path);
	auto data = fs.OpenFile(result->database_path, FileFlags::FILE_FLAGS_READ);
	if (data->GetFileSize() != result->database_size || fs.FileExists(result->database_path + ".wal")) {
		throw InvalidInputException("Native row oracle: database size/WAL mismatch");
	}
	result->database_mtime = fs.GetLastModifiedTime(*data).value;
	duckdb_mbedtls::MbedTlsWrapper::SHA256State hash;
	vector<data_t> buffer(1024 * 1024);
	for (idx_t offset = 0; offset < result->database_size;) {
		auto size = MinValue<idx_t>(buffer.size(), result->database_size - offset);
		data->Read(buffer.data(), size, offset);
		hash.AddBytes(buffer.data(), size);
		offset += size;
	}
	char hex[65];
	hash.FinishHex(hex);
	if (string(hex, 64) != result->database_sha256) {
		throw InvalidInputException("Native row oracle: database SHA256 mismatch");
	}
	return result;
}

void NativeRowOracle::CollectScans(ClientContext &context, LogicalOperator &op) {
	using namespace native_oracle_json;
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_PROJECTION:
	case LogicalOperatorType::LOGICAL_FILTER:
	case LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
	case LogicalOperatorType::LOGICAL_EMPTY_RESULT:
		break;
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
		if (op.Cast<LogicalComparisonJoin>().join_type != JoinType::INNER) {
			throw InvalidInputException("Native row oracle supports inner SPJ only");
		}
		break;
	case LogicalOperatorType::LOGICAL_GET: {
		auto &get = op.Cast<LogicalGet>();
		auto table = get.GetTable();
		if (!table || !get.children.empty() || get.extra_info.sample_options || table->HasGeneratedColumns()) {
			throw InvalidInputException("Native row oracle requires ordinary unsampled base tables");
		}
		auto path = FileSystem::GetFileSystem(context).ConvertSeparators(table->catalog.GetDBPath());
		if (path.empty() || path == ":memory:" || !table->catalog.GetAttached().IsReadOnly()) {
			throw InvalidInputException("Native row oracle requires a frozen read-only database");
		}
		if (database_path.empty()) {
			database_path = path;
		}
		if (database_path != path || (file && file->database_path != path)) {
			throw InvalidInputException("Native row oracle: database path mismatch/multiple databases (scan=%s, profile=%s)",
			                            path, file ? file->database_path : database_path);
		}
		string from = Identifier(table->catalog.GetName()) + "." + Identifier(table->schema.name) + "." + Identifier(table->name);
		string schema = from;
		for (idx_t i = 0; i < get.names.size(); i++) {
			schema += "|" + std::to_string(get.names[i].size()) + ":" + get.names[i] + ":" + get.returned_types[i].ToString();
		}
		auto id = get.table_index.index;
		if (!scans.emplace(id, Scan {&get, from + " AS " + Identifier("r" + std::to_string(id)), schema, 0}).second) {
			throw InvalidInputException("Native row oracle: duplicate occurrence identity");
		}
		scan_order.push_back(id);
		break;
	}
	default:
		throw InvalidInputException("Native row oracle: unsupported operator %s", op.GetName());
	}
	for (auto &child : op.children) {
		CollectScans(context, *child);
	}
}

NativeRowOracle::NativeRowOracle(ClientContext &context, LogicalOperator &plan, const string &query,
                               shared_ptr<const NativeRowOracleFile> file_p, bool discovery_p)
    : file(std::move(file_p)), query_sha256(NativeRowOracleFile::Hash(query)), discovery(discovery_p) {
	if (!file && !discovery) {
		throw InvalidInputException("Native row oracle requires optimizer_statistics_file");
	}
	CollectScans(context, plan);
	if (scan_order.empty() || scan_order.size() > 62) {
		throw InvalidInputException("Native row oracle requires 1..62 base relation occurrences");
	}
	std::sort(scan_order.begin(), scan_order.end());
	for (idx_t i = 0; i < scan_order.size(); i++) {
		scans.at(scan_order[i]).ordinal = i;
	}
	if (file) {
		auto &fs = FileSystem::GetFileSystem(context);
		auto handle = fs.OpenFile(database_path, FileFlags::FILE_FLAGS_READ);
		if (handle->GetFileSize() != file->database_size || fs.GetLastModifiedTime(*handle).value != file->database_mtime ||
		    fs.FileExists(database_path + ".wal")) {
			throw InvalidInputException("Native row oracle: frozen database changed since configuration");
		}
	}
}

string NativeRowOracle::Render(const Expression &expression) const {
	auto copy = expression.Copy();
	if (copy->IsVolatile() || copy->HasParameter() || copy->HasSubquery()) {
		throw InvalidInputException("Native row oracle requires deterministic parameter-free predicates");
	}
	ExpressionIterator::VisitExpressionMutable<BoundColumnRefExpression>(copy, [&](BoundColumnRefExpression &col, unique_ptr<Expression> &slot) {
		auto mark = constant_marks.find(col.binding.table_index.index);
		if (!col.depth && col.binding.column_index == ProjectionIndex(0) && mark != constant_marks.end()) {
			// The replacement has base-column aliases resolved already. Preserve SQL
			// three-valued IN/NOT IN semantics, including NULL list values.
			slot = mark->second->Copy();
			return;
		}
		auto entry = scans.find(col.binding.table_index.index);
		if (col.depth || entry == scans.end()) {
			throw InvalidInputException("Native row oracle: unknown/derived column binding %s", col.binding.ToString());
		}
		auto &get = *entry->second.get;
		col.SetAlias(native_oracle_json::Identifier("r" + std::to_string(get.table_index.index)) + "." +
		             native_oracle_json::Identifier(get.GetColumnName(get.GetColumnIndex(col.binding))));
	});
	return copy->ToString();
}

NativeRowOracle::Description NativeRowOracle::Describe(LogicalOperator &op) {
	if (op.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		auto &join = op.Cast<LogicalComparisonJoin>();
		if (join.join_type == JoinType::MARK && join.children.size() == 2 &&
		    join.children[1]->type == LogicalOperatorType::LOGICAL_CHUNK_GET && join.conditions.size() == 1) {
			auto &condition = join.conditions[0];
			auto &values = join.children[1]->Cast<LogicalColumnDataGet>();
			if (!condition.IsComparison() || condition.GetComparisonType() != ExpressionType::COMPARE_EQUAL ||
			    values.chunk_types.size() != 1 || condition.GetRHS().GetExpressionType() != ExpressionType::BOUND_COLUMN_REF) {
				throw InvalidInputException("Native row oracle: unsupported constant MARK join");
			}
			auto &rhs = condition.GetRHS().Cast<BoundColumnRefExpression>();
			if (rhs.binding.table_index != values.table_index || rhs.binding.column_index != ProjectionIndex(0) || rhs.depth) {
				throw InvalidInputException("Native row oracle: unsupported constant MARK binding");
			}
			auto left = Describe(*join.children[0]);
			auto predicate = make_uniq<BoundOperatorExpression>(ExpressionType::COMPARE_IN, LogicalType::BOOLEAN);
			auto input = condition.GetLHS().Copy();
			// Resolve base aliases once before storing the replacement expression.
			ExpressionIterator::VisitExpressionMutable<BoundColumnRefExpression>(input, [&](BoundColumnRefExpression &col, unique_ptr<Expression> &) {
				auto entry = scans.find(col.binding.table_index.index);
				if (col.depth || entry == scans.end()) {
					throw InvalidInputException("Native row oracle: unsupported constant MARK input");
				}
				auto &get = *entry->second.get;
				col.SetAlias(native_oracle_json::Identifier("r" + std::to_string(get.table_index.index)) + "." +
				             native_oracle_json::Identifier(get.GetColumnName(get.GetColumnIndex(col.binding))));
			});
			predicate->children.push_back(std::move(input));
			ColumnDataScanState state;
			DataChunk chunk;
			values.collection->InitializeScanChunk(chunk);
			values.collection->InitializeScan(state);
			while (values.collection->Scan(state, chunk)) {
				for (idx_t row = 0; row < chunk.size(); row++) {
					predicate->children.push_back(make_uniq<BoundConstantExpression>(chunk.GetValue(0, row)));
				}
			}
			if (predicate->children.size() == 1) {
				throw InvalidInputException("Native row oracle: empty constant MARK collection");
			}
			constant_marks[join.mark_index.index] = std::move(predicate);
			// MARK appends a boolean without filtering or multiplying the left bag.
			return left;
		}
	}
	Description result;
	for (auto &child : op.children) {
		auto part = Describe(*child);
		result.relations |= part.relations;
		result.predicates.insert(result.predicates.end(), part.predicates.begin(), part.predicates.end());
	}
	if (op.type == LogicalOperatorType::LOGICAL_GET) {
		auto &get = op.Cast<LogicalGet>();
		auto entry = scans.find(get.table_index.index);
		if (entry == scans.end()) {
			throw InvalidInputException("Native row oracle: unknown scan occurrence");
		}
		result.relations |= uint64_t(1) << entry->second.ordinal;
		for (auto &filter : get.table_filters) {
			if (filter.Filter().filter_type == TableFilterType::OPTIONAL_FILTER || filter.Filter().IsOnlyForZoneMapFiltering()) {
				continue;
			}
			auto &column = get.GetColumnIndex(filter.GetIndex());
			BoundColumnRefExpression ref(get.GetColumnType(column), ColumnBinding(get.table_index, filter.GetIndex()));
			result.predicates.push_back(Render(*filter.Filter().ToExpression(ref)));
		}
	} else if (op.type == LogicalOperatorType::LOGICAL_FILTER) {
		for (auto &expr : op.expressions) {
			result.predicates.push_back(Render(*expr));
		}
	} else if (op.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		auto &join = op.Cast<LogicalComparisonJoin>();
		if (join.join_type != JoinType::INNER) {
			throw InvalidInputException("Native row oracle only costs inner joins");
		}
		for (auto &cond : join.conditions) {
			if (cond.IsComparison()) {
				result.predicates.push_back(Render(*BoundComparisonExpression::Create(cond.GetComparisonType(), cond.GetLHS().Copy(), cond.GetRHS().Copy())));
			} else {
				result.predicates.push_back(Render(cond.GetJoinExpression()));
			}
		}
	} else if (op.type != LogicalOperatorType::LOGICAL_PROJECTION && op.type != LogicalOperatorType::LOGICAL_CROSS_PRODUCT) {
		throw InvalidInputException("Native row oracle: unsupported cardinality stage %s", op.GetName());
	}
	return result;
}

idx_t NativeRowOracle::Lookup(Description description, idx_t fallback, const string &stage) {
	return LookupReduction(std::move(description), Description {}, fallback, stage);
}

idx_t NativeRowOracle::LookupReduction(Description description, Description right, idx_t fallback, const string &stage) {
	if (description.relations & right.relations) {
		throw InternalException("Native row oracle: overlapping reduction occurrences");
	}
	vector<string> from, reducing_from, schemas;
	for (auto id : scan_order) {
		auto &scan = scans.at(id);
		if (description.relations & (uint64_t(1) << scan.ordinal)) {
			from.push_back(scan.from);
		}
		if (right.relations & (uint64_t(1) << scan.ordinal)) {
			reducing_from.push_back(scan.from);
		}
		if ((description.relations | right.relations) & (uint64_t(1) << scan.ordinal)) {
			schemas.push_back(scan.schema);
		}
	}
	if (from.empty()) {
		throw InvalidInputException("Native row oracle: empty relation request");
	}
	std::sort(description.predicates.begin(), description.predicates.end());
	description.predicates.erase(std::unique(description.predicates.begin(), description.predicates.end()), description.predicates.end());
	string sql = "SELECT COUNT(*) FROM " + StringUtil::Join(from, ", ");
	if (!description.predicates.empty()) {
		sql += " WHERE (" + StringUtil::Join(description.predicates, ") AND (") + ")";
	}
	if (!reducing_from.empty()) {
		std::sort(right.predicates.begin(), right.predicates.end());
		right.predicates.erase(std::unique(right.predicates.begin(), right.predicates.end()), right.predicates.end());
		sql += description.predicates.empty() ? " WHERE " : " AND ";
		sql += "EXISTS (SELECT 1 FROM " + StringUtil::Join(reducing_from, ", ");
		if (!right.predicates.empty()) {
			sql += " WHERE (" + StringUtil::Join(right.predicates, ") AND (") + ")";
		}
		sql += ")";
	}
	auto schema = NativeRowOracleFile::Hash(StringUtil::Join(schemas, "\n"));
	auto key = NativeRowOracleFile::Key(query_sha256, schema, sql);
	auto existing = requests.find(key);
	if (existing != requests.end()) {
		existing->second.calls++;
		if (("|" + existing->second.stage + "|").find("|" + stage + "|") == string::npos) {
			existing->second.stage += "|" + stage;
		}
		return existing->second.found ? existing->second.cardinality : fallback;
	}
	Request request;
	request.key = key;
	request.schema = schema;
	request.sql = sql;
	request.stage = stage;
	request.calls = 1;
	if (file) {
		auto entry = file->entries.find(key);
		if (entry != file->entries.end()) {
			request.found = true;
			request.cardinality = entry->second.cardinality;
		}
	}
	if (!request.found) {
		missing++;
	}
	requests.emplace(key, request);
	if (!request.found && !discovery) {
		throw InvalidInputException("Missing native row cardinality: key=%s, stage=%s, SQL=%s", key, stage, sql);
	}
	return request.found ? request.cardinality : fallback;
}

idx_t NativeRowOracle::Raw(LogicalGet &get, idx_t fallback) {
	Description description;
	description.relations = uint64_t(1) << scans.at(get.table_index.index).ordinal;
	return Lookup(std::move(description), fallback, "raw_scan");
}

idx_t NativeRowOracle::Pin(LogicalOperator &op) {
	if (op.type == LogicalOperatorType::LOGICAL_CHUNK_GET) {
		// Native IN rewriting creates a known constant list. It is not a database
		// occurrence and its native count is already exact.
		return op.estimated_cardinality;
	}
	auto value = Lookup(Describe(op), op.estimated_cardinality, op.GetName());
	op.SetEstimatedCardinality(value);
	pinned[&op] = value;
	return value;
}

shared_ptr<NativeRowOracleGraph> NativeRowOracle::BindGraph(QueryGraphManager &graph) {
	auto result = make_shared_ptr<NativeRowOracleGraph>(*this);
	for (idx_t i = 0; i < graph.relation_manager.NumRelations(); i++) {
		result->relations.push_back(Describe(graph.relation_manager.GetRelation(i).op));
	}
	for (auto &filter : graph.GetFilterBindings()) {
		if (filter->join_type != JoinType::INNER) {
			throw InvalidInputException("Native row oracle only costs inner SPJ graphs");
		}
		uint64_t mask = 0;
		for (idx_t i = 0; i < filter->set.get().count; i++) {
			mask |= uint64_t(1) << filter->set.get().relations[i].index;
		}
		result->predicates.emplace_back(mask, Render(*filter->filter));
	}
	native_join_searches++;
	return result;
}

idx_t NativeRowOracleGraph::Cardinality(JoinRelationSet &set, idx_t fallback) {
	auto found = cache.find(set.ToString());
	if (found != cache.end()) {
		return found->second;
	}
	NativeRowOracle::Description description;
	uint64_t mask = 0;
	for (idx_t i = 0; i < set.count; i++) {
		auto id = set.relations[i].index;
		mask |= uint64_t(1) << id;
		auto &part = relations.at(id);
		description.relations |= part.relations;
		description.predicates.insert(description.predicates.end(), part.predicates.begin(), part.predicates.end());
	}
	for (auto &predicate : predicates) {
		if ((predicate.first & mask) == predicate.first) {
			description.predicates.push_back(predicate.second);
		}
	}
	auto value = oracle.Lookup(std::move(description), fallback, set.count == 1 ? "filtered_relation" : "candidate_join");
	cache.emplace(set.ToString(), value);
	return value;
}

string NativeRowOracle::ToJSON() const {
	using native_oracle_json::Quote;
	string result = "{\"query_sha256\":" + Quote(query_sha256) + ",\"database_path\":" + Quote(database_path) +
	                ",\"discovery\":" + (discovery ? "true" : "false") + ",\"missing\":" + std::to_string(missing) +
	                ",\"native_join_searches\":" + std::to_string(native_join_searches) + ",\"requests\":[";
	vector<string> keys;
	for (auto &entry : requests) {
		keys.push_back(entry.first);
	}
	std::sort(keys.begin(), keys.end());
	for (auto &key : keys) {
		if (result.back() != '[') {
			result += ",";
		}
		auto &entry = requests.at(key);
		result += "{\"key\":" + Quote(key) + ",\"query_sha256\":" + Quote(query_sha256) + ",\"schema\":" + Quote(entry.schema) +
		          ",\"sql\":" + Quote(entry.sql) + ",\"stage\":" + Quote(entry.stage) + ",\"found\":" + (entry.found ? "true" : "false") +
		          ",\"cardinality\":" + std::to_string(entry.cardinality) + ",\"calls\":" + std::to_string(entry.calls) + "}";
	}
	return result + "]}";
}
} // namespace duckdb
