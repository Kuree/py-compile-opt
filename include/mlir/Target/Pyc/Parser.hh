#ifndef PY_COMPILE_OPT_PARSER_HH
#define PY_COMPILE_OPT_PARSER_HH

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/MemoryBuffer.h"

namespace mlir {

mlir::LogicalResult parseModule(llvm::MemoryBuffer &buffer,
                                mlir::ModuleOp moduleOp,
                                mlir::OpBuilder &builder);

}

#endif // PY_COMPILE_OPT_PARSER_HH
