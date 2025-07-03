#include "mlir/Dialect/Pyc/IR/Pyc.hh"
#include "mlir/IR/Builders.h"
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

::mlir::Operation *PycDialect::parseOperation(uint8_t opCode, uint8_t opArg,
                                              ::mlir::Location loc,
                                              ::mlir::OpBuilder &builder) {
    // clang-format off
    #define GENERATE_PARSER
    #include "mlir/Dialect/Pyc/IR/GeneratedOps.td"
    // clang-format on
    auto nameAttr = builder.getStringAttr(opName);
    NamedAttrList attrs;
    attrs.set("opArg", builder.getI32IntegerAttr(opArg));
    return builder.create(loc, nameAttr, SmallVector<Value>{},
                          SmallVector<Type>{}, attrs.getAttrs());
}
