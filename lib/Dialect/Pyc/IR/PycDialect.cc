#include "mlir/Dialect/Pyc/IR/Pyc.hh"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::pyc;

#include "mlir/Dialect/Pyc/IR/PycOpsDialect.cpp.inc"
#include "mlir/Dialect/Pyc/IR/PycOpsInterfaces.cpp.inc"

void PycDialect::initialize() {
    // clang-format off
    addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Pyc/IR/PycOps.cpp.inc"
    >();
    // clang-format on
}
