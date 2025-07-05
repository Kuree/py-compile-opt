add_llvm_tool(pyc-parser pyc-parser.cc)

target_link_libraries(pyc-parser
    PRIVATE
    LLVMSupport
    MLIRPycDialect
)
