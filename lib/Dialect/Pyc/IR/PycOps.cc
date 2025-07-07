#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"

#include "mlir/Dialect/Pyc/IR/Pyc.hh"

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Pyc/IR/PycOps.cpp.inc"

// enum
#include "mlir/Dialect/Pyc/IR/PycOpsEnums.cpp.inc"

namespace mlir::pyc {

llvm::LogicalResult
RefOp::verifySymbolUses(::mlir::SymbolTableCollection &symbolTable) {
    auto moduleOp = getOperation()->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp)
        return emitOpError("unable to find module operation");
    auto &table = symbolTable.getSymbolTable(moduleOp);
    auto *op = table.lookup(getRef());
    if (!op) {
        return emitOpError("cannot find symbol '") << getRef() << "'";
    }
    return success();
}

::llvm::LogicalResult KeyValuePairOp::verify() {
    // only size two
    auto *body = getBody();
    if (body->getOperations().size() != 2) {
        return emitOpError("can only have two operations");
    }
    return success();
}

} // namespace mlir::pyc