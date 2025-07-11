#ifndef MLIR_DIALECT_PYC_TRANSFORMS_PASSES_TD
#define MLIR_DIALECT_PYC_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def DecodeByteCode: Pass<"decode-byte-code"> {
	let summary = "Decode python bytecode instructions";
}

#endif
