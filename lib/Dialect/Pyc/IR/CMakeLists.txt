add_mlir_dialect_library(MLIRPycDialect
        PycOps.cc
        PycDialect.cc

        ADDITIONAL_HEADER_DIRS

        DEPENDS
        MLIRPycOpsIncGen
        MLIRPycOpsInterfacesIncGen
)
