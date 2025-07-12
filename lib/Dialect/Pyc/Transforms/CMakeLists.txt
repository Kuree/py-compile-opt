add_mlir_library(MLIRPycTransforms
        DecodeByteCode.cc
        DecodeSourceLocation.cc
        ElaborateReferences.cc

        ADDITIONAL_HEADER_DIRS

        DEPENDS
        MLIRPycTransformsIncGen

        LINK_LIBS PUBLIC
        MLIRPycDialect
        MLIRIR
        MLIRPass
        MLIRTransforms
)
