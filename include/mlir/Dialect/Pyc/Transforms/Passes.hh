#ifndef MLIR_DIALECT_PYC_TRANSFORMS_PASSES_HH
#define MLIR_DIALECT_PYC_TRANSFORMS_PASSES_HH

#include "mlir/Pass/Pass.h"

namespace mlir::pyc {
#define GEN_PASS_DECL
#include "mlir/Dialect/Pyc/Transforms/Passes.hh.inc"

#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/Pyc/Transforms/Passes.hh.inc"
}

#endif // MLIR_DIALECT_PYC_TRANSFORMS_PASSES_HH
