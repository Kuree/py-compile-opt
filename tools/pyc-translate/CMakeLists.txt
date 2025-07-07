add_llvm_executable(pyc-translate
        pyc-translate.cc
)
llvm_update_compile_flags(pyc-translate)

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
