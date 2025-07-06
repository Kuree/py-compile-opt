set(PY_GENERATE_PARSER ${CMAKE_SOURCE_DIR}/utils/pyc-parser-gen/pyc-parser-gen.py)
set(GENERATED_PARSER ${CMAKE_CURRENT_BINARY_DIR}/GeneratedParser.cc.inc)

add_custom_command(
        OUTPUT ${GENERATED_PARSER}
        COMMAND python3 ${PY_GENERATE_PARSER} -o ${GENERATED_PARSER}
        DEPENDS ${PY_GENERATE_PARSER})

add_custom_target(PycGeneratedParser DEPENDS ${GENERATED_PARSER})

add_mlir_library(MLIRPycTarget
    Parser.cc
    TranslateRegistration.cc

    LINK_LIBS
    LLVMSupport
    MLIRPycDialect

    DEPENDS
    PycGeneratedParser
)

target_include_directories(MLIRPycTarget
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
)