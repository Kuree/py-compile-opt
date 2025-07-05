#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"

#include "mlir/Dialect/Pyc/IR/Pyc.hh"

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

class Parser {
  public:
    Parser(llvm::MemoryBuffer &buffer) : head(buffer.getBuffer()), pos(0) {}

    uint8_t getUInt8() { return parseBytes<uint8_t>(); }
    int8_t getSInt8() { return parseBytes<int8_t>(); }
    uint16_t getUInt16() { return parseBytes<uint16_t>(); }
    int16_t getSInt16() { return parseBytes<int16_t>(); }
    uint32_t getUInt32() { return parseBytes<uint32_t>(); }
    int32_t getSInt32() { return parseBytes<int32_t>(); }

  private:
    llvm::StringRef head;
    uint64_t pos;

    template <typename T> T parseBytes() {
        auto constexpr size = sizeof(T);
        auto res = head.slice(pos, pos + size);
        pos += size;
        T v;
        std::memcpy(&v, res.data(), size);
        return v;
    }
};

// copied from pycdc
// cross-reference: https://github.com/python/cpython/blob/main/Python/marshal.c
enum class ObjType {
    // From the Python Marshallers
    TYPE_NULL = '0',                 // Python 1.0 ->
    TYPE_NONE = 'N',                 // Python 1.0 ->
    TYPE_FALSE = 'F',                // Python 2.3 ->
    TYPE_TRUE = 'T',                 // Python 2.3 ->
    TYPE_STOPITER = 'S',             // Python 2.2 ->
    TYPE_ELLIPSIS = '.',             // Python 1.4 ->
    TYPE_INT = 'i',                  // Python 1.0 ->
    TYPE_INT64 = 'I',                // Python 1.5 - 3.3
    TYPE_FLOAT = 'f',                // Python 1.0 ->
    TYPE_BINARY_FLOAT = 'g',         // Python 2.5 ->
    TYPE_COMPLEX = 'x',              // Python 1.4 ->
    TYPE_BINARY_COMPLEX = 'y',       // Python 2.5 ->
    TYPE_LONG = 'l',                 // Python 1.0 ->
    TYPE_STRING = 's',               // Python 1.0 ->
    TYPE_INTERNED = 't',             // Python 2.4 - 2.7, 3.4 ->
    TYPE_STRINGREF = 'R',            // Python 2.4 - 2.7
    TYPE_OBREF = 'r',                // Python 3.4 ->
    TYPE_TUPLE = '(',                // Python 1.0 ->
    TYPE_LIST = '[',                 // Python 1.0 ->
    TYPE_DICT = '{',                 // Python 1.0 ->
    TYPE_CODE = 'c',                 // Python 1.3 ->
    TYPE_CODE2 = 'C',                // Python 1.0 - 1.2
    TYPE_UNICODE = 'u',              // Python 1.6 ->
    TYPE_UNKNOWN = '?',              // Python 1.0 ->
    TYPE_SET = '<',                  // Python 2.5 ->
    TYPE_FROZENSET = '>',            // Python 2.5 ->
    TYPE_ASCII = 'a',                // Python 3.4 ->
    TYPE_ASCII_INTERNED = 'A',       // Python 3.4 ->
    TYPE_SMALL_TUPLE = ')',          // Python 3.4 ->
    TYPE_SHORT_ASCII = 'z',          // Python 3.4 ->
    TYPE_SHORT_ASCII_INTERNED = 'Z', // Python 3.4 ->
};

// https://github.com/python/cpython/blob/5de7e3f9739b01ad180fffb242ac57cea930e74d/Python/marshal.c#L642
mlir::LogicalResult parseObj(Parser &parser, mlir::OpBuilder &builder) {
    auto type = parser.getUInt8();
}

mlir::LogicalResult parseModule(llvm::MemoryBuffer &buffer,
                                mlir::ModuleOp moduleOp,
                                mlir::OpBuilder &builder) {
    Parser parser(buffer);
    auto version = parser.getUInt32();
    moduleOp->setAttr("pyc.version", builder.getUI32IntegerAttr(version));
    auto flags = parser.getUInt32();
    moduleOp->setAttr("pyc.flags", builder.getUI32IntegerAttr(flags));

    if (flags & 0x1) {
        // checksum
        parser.getUInt32();
        parser.getUInt32();
    } else {
        // timestamp
        parser.getUInt32();

        // size parameter
        parser.getUInt32();
    }
    return parseObj(parser, builder);
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
    builder.setInsertionPointToEnd(module->getBody());
    if (mlir::failed(parseModule(*buffer.get(), module.get(), builder))) {
        llvm::errs() << "Failed to parse code object\n";
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
