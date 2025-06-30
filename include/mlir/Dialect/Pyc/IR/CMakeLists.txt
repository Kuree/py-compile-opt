set(PY_GENERATE_TD ${CMAKE_SOURCE_DIR}/utils/pyc-opt-gen/generate-td.py)
set(GENERATED_TD ${CMAKE_CURRENT_BINARY_DIR}/GeneratedOps.td)

add_custom_command(
        OUTPUT ${GENERATED_TD}
        COMMAND python3 ${PY_GENERATE_TD} -o ${GENERATED_TD}
        DEPENDS ${PY_GENERATE_TD})

add_custom_target(PycGeneratedTd DEPENDS ${GENERATED_TD})

set(LLVM_TARGET_DEFINITIONS PycOps.td)

add_mlir_dialect(PycOps pyc)
add_mlir_interface(PycOpsInterfaces)

add_dependencies(MLIRPycOpsIncGen PycGeneratedTd)
