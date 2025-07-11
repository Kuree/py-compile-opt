set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.hh.inc -gen-pass-decls)
add_public_tablegen_target(MLIRPycTransformsIncGen)
