add_mlir_tool(pyc-translate
        pyc-translate.cc
)

target_link_libraries(pyc-translate
    PRIVATE
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRPycDialect
    MLIRTranslateLib
    MLIRSupport
    MLIRPycTarget
)
