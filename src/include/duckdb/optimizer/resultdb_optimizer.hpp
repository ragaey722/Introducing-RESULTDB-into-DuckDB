//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/resultdb_optimizer.hpp
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
struct ResultDBSemijoinAnalysis;

class ResultDBOptimizer {
public:
	ResultDBOptimizer(Binder &binder, ClientContext &context);
	~ResultDBOptimizer();

	//! Phase 1: retain the original SPJ graph before expression/filter rewrites.
	void CaptureGraph(LogicalOperator &plan);
	//! Phase 2: attach optimized inputs, then dispatch root/fold/schedule selection.
	unique_ptr<LogicalOperator> Optimize(unique_ptr<LogicalOperator> plan,
	                                     unique_ptr<ResultDBYannakakisProgram> &program);
	//! Phase 3, AUTO only: compare against the actual native-oracle join tree.
	unique_ptr<LogicalOperator> FinalizeAuto(unique_ptr<LogicalOperator> plan,
	                                        unique_ptr<ResultDBYannakakisProgram> &program);

private:
	void PlanEnumerated(unique_ptr<ResultDBYannakakisProgram> &program);
	Binder &binder;
	ClientContext &context;
	unique_ptr<ResultDBSemijoinAnalysis> captured;
	unique_ptr<ResultDBYannakakisProgram> auto_program;
	bool enumerated = false;
};

} // namespace duckdb
