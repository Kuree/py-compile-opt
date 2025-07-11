add_llvm_executable(pyc-opt pyc-opt.cc)
llvm_update_compile_flags(pyc-opt)

target_link_libraries(
        pyc-opt
        PRIVATE MLIROptLib
        MLIRPycTransforms)
