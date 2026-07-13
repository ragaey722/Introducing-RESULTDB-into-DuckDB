//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/resultdb_yannakakis_optimizer.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/unique_ptr.hpp"

namespace duckdb {

class Binder;
class ClientContext;
class LogicalOperator;
struct ResultDBYannakakisProgram;

class ResultDBYannakakisOptimizer {
public:
	ResultDBYannakakisOptimizer(Binder &binder, ClientContext &context);

	unique_ptr<LogicalOperator> Optimize(unique_ptr<LogicalOperator> plan,
	                                     unique_ptr<ResultDBYannakakisProgram> &resultdb_yannakakis_program);

private:
	Binder &binder;
	ClientContext &context;
};

} // namespace duckdb
