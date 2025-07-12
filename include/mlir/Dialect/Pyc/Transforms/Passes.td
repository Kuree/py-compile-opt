#ifndef MLIR_DIALECT_PYC_TRANSFORMS_PASSES_TD
#define MLIR_DIALECT_PYC_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def DecodeByteCode: Pass<"decode-byte-code"> {
  let summary = "Decode python bytecode instructions";
}

def DecodeSourceLocation : Pass<"decode-source-location"> {
  let summary = "Decode python source location";
}

def ElaborateReference : Pass<"elaborate-reference"> {
  let summary = "Elaborate references";
}

#endif
