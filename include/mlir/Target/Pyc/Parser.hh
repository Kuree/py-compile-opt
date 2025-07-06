#ifndef PY_COMPILE_OPT_PARSER_HH
#define PY_COMPILE_OPT_PARSER_HH

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

namespace mlir {

mlir::LogicalResult parseModule(const llvm::MemoryBuffer &buffer,
                                mlir::ModuleOp moduleOp,
                                mlir::OpBuilder &builder);

OwningOpRef<Operation *> parseModule(llvm::SourceMgr &srcMgr,
                                     mlir::MLIRContext *ctx);

} // namespace mlir

#endif // PY_COMPILE_OPT_PARSER_HH
