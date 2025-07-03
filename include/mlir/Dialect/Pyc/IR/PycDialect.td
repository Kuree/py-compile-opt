#ifndef PYC_DIALECT_TD
#define PYC_DIALECT_TD

include "mlir/IR/OpBase.td"

def Pyc_Dialect : Dialect {
  let name = "pyc";
  let summary = "Pyc dialect";
  let description = [{
    The pyc dialect represents python bytecode
  }];

  let cppNamespace = "::mlir::pyc";

  let extraClassDeclaration = [{
  static ::mlir::Operation *parseOperation(uint8_t opCode, uint8_t opArg, ::mlir::Location loc, ::mlir::OpBuilder &builder);
  }];
}

#endif  // PYC_DIALECT_TD
