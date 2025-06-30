#ifndef PY_COMPILE_OPT_PYC_HH
#define PY_COMPILE_OPT_PYC_HH

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

// Dialect
#include "mlir/Dialect/Pyc/IR/PycDialect.h.inc"

// Interfaces
#include "mlir/Dialect/Pyc/IR/PycOpsInterfaces.h.inc"

// Operations
#define GET_OP_CLASSES
#include "mlir/Dialect/Pyc/IR/PycOps.h.inc"

#endif // PY_COMPILE_OPT_PYC_HH
