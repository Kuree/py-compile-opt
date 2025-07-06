#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"

#include "mlir/Dialect/Pyc/IR/Pyc.hh"
#include "mlir/Target/Pyc/Parser.hh"

namespace {
llvm::cl::opt<std::string> outputFilename{"o", llvm::cl::init("-"),
                                          llvm::cl::desc("Output filename")};
llvm::cl::opt<std::string>
    inputFileName(llvm::cl::Positional, llvm::cl::desc("<Specify input file>"));
llvm::cl::opt<bool> asmOutput("S", llvm::cl::init(false),
                              llvm::cl::desc("Output in assembly"));

mlir::LogicalResult printModule(mlir::ModuleOp moduleOp) {
    if (outputFilename != "-") {
        std::error_code ec;
        auto os = llvm::raw_fd_ostream(outputFilename, ec);
        if (!ec) {
            llvm::errs() << "error writing to file " << outputFilename
                         << ". reason: " << ec.message() << "\n";
        }
        if (asmOutput) {
            moduleOp.print(os);
        } else {
            // write to byte code
            auto res = mlir::writeBytecodeToFile(moduleOp, os);
            if (mlir::failed(res)) {
                llvm::errs() << "error writing byte code to file "
                             << outputFilename << "\n";
            }
            return res;
        }
    } else {
        if (asmOutput) {
            moduleOp.print(llvm::outs());
        } else {
            llvm::errs() << "Refuse to output binary to console\n";
            return mlir::failure();
        }
    }
    return mlir::success();
}

int runMain() {
    auto buffer = llvm::MemoryBuffer::getFile(inputFileName);
    if (!buffer) {
        llvm::errs() << "error opening file " << inputFileName
                     << ". reason: " << buffer.getError().message() << "\n";
        return EXIT_FAILURE;
    }

    mlir::MLIRContext context;
    mlir::DialectRegistry registry;
    registry.insert<mlir::pyc::PycDialect>();
    context.appendDialectRegistry(registry);
    mlir::OpBuilder builder(&context);
    auto loc = mlir::NameLoc::get(builder.getStringAttr(inputFileName));
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(loc);

    // parse the pyc
    if (mlir::failed(mlir::parseModule(*buffer.get(), module.get(), builder))) {
        return EXIT_FAILURE;
    }

    // print out the IR
    auto res = printModule(module.get());
    return mlir::succeeded(res) ? EXIT_SUCCESS : EXIT_FAILURE;
}

} // namespace

int main(int argc, char **argv) {
    llvm::cl::ParseCommandLineOptions(argc, argv, "Pyc parser");
    return runMain();
}
