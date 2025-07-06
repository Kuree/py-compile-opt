#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
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
