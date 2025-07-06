#include "mlir/Target/Pyc/Parser.hh"
#include "mlir/Dialect/Pyc/IR/Pyc.hh"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"

namespace {

#include "GeneratedParser.cc.inc"

using llvm::dyn_cast;
using llvm::dyn_cast_or_null;
using llvm::failed;
using llvm::failure;
using llvm::success;

std::string getRefSymbolName(uint32_t idx) {
    return llvm::formatv("ref_%s", idx);
}

struct ParserContext {
    ParserContext(mlir::ModuleOp moduleOp, mlir::OpBuilder &builder) {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(moduleOp.getBody());
        refs =
            builder.create<mlir::pyc::RefCollectionOp>(builder.getUnknownLoc());
    }

    mlir::OpBuilder::InsertPoint getRefInsertionPoint() {
        auto *block = &refs->getRegion(0).front();
        return {block, block->end()};
    }

    uint32_t addRefOp(mlir::Operation *op, mlir::OpBuilder &builder) {
        auto idx = refCount;
        auto name = getRefSymbolName(idx);
        refCount += 1;
        auto nameAttr = builder.getStringAttr(name);
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.restoreInsertionPoint(getRefInsertionPoint());
        auto attr = mlir::FlatSymbolRefAttr::get(nameAttr);
        auto makeRef =
            builder.create<mlir::pyc::MakeRefOp>(builder.getUnknownLoc(), attr);
        auto *block = builder.createBlock(&makeRef.getRegion());
        op->moveBefore(block, block->end());
        return idx;
    }

  private:
    mlir::pyc::RefCollectionOp refs;

    uint32_t refCount = 0;
};

class Parser {
  public:
    explicit Parser(llvm::MemoryBuffer &buffer)
        : buffer(buffer.getBuffer()), pos(0) {}

    auto getUInt8() { return parseBytes<uint8_t>(); }
    auto getSInt8() { return parseBytes<int8_t>(); }
    auto getUInt16() { return parseBytes<uint16_t>(); }
    auto getSInt16() { return parseBytes<int16_t>(); }
    auto getUInt32() { return parseBytes<uint32_t>(); }
    auto getSInt32() { return parseBytes<int32_t>(); }

  private:
    llvm::StringRef buffer;
    uint64_t pos;

    template <typename T> llvm::FailureOr<T> parseBytes() {
        auto constexpr size = sizeof(T);
        auto nextPos = pos + size;
        if (buffer.size() < nextPos) {
            llvm::errs() << "expect " << size << "bytes at position " << pos
                         << "\n";
            return failure();
        }
        auto res = buffer.slice(pos, pos + size);
        pos += size;
        T v;
        std::memcpy(&v, res.data(), size);
        return v;
    }
};

mlir::Operation *parseObj(ParserContext &ctx, Parser &parser,
                          mlir::OpBuilder &builder);

mlir::Operation *parseCodeObj(ParserContext &ctx, Parser &parser,
                              mlir::OpBuilder &builder) {
    auto codeOp = builder.create<mlir::pyc::CodeOp>(builder.getUnknownLoc());
    auto *block = builder.createBlock(&codeOp.getBodyRegion());
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(block);

    // use macro to control the parsing

#ifdef PYC_ARGCOUNT
    auto argCount = parser.getSInt32();
    if (failed(argCount))
        return nullptr;
    codeOp->setAttr("pyc.arg_count", builder.getI32IntegerAttr(*argCount));
#endif

#ifdef PYC_POSONLYARGCOUNT
    auto posOnlyArgCount = parser.getSInt32();
    if (failed(posOnlyArgCount))
        return nullptr;
    codeOp->setAttr("pyc.pos_only_arg_count",
                    builder.getI32IntegerAttr(*posOnlyArgCount));
#endif

#ifdef PYC_KWONLYARGCOUNT
    auto kwOnlyArgCount = parser.getSInt32();
    if (failed(kwOnlyArgCount))
        return nullptr;
    codeOp->setAttr("pyc.kw_only_arg_count",
                    builder.getI32IntegerAttr(*kwOnlyArgCount));
#endif

#ifdef PYC_STACKSIZE
    auto stackSize = parser.getSInt32();
    if (failed(stackSize))
        return nullptr;
    codeOp->setAttr("pyc.stackSize", builder.getI32IntegerAttr(*stackSize));
#endif

#ifdef PYC_FLAGS
    auto flags = parser.getSInt32();
    if (failed(flags))
        return nullptr;
    codeOp->setAttr("pyc.flags", builder.getI32IntegerAttr(*flags));
#endif

#ifdef PYC_CODE
    // need to create ops

#endif

#ifdef PYC_CONSTS
    if (auto constants = dyn_cast_or_null<mlir::pyc::CodeObjectMember>(
            parseObj(ctx, parser, builder)))
        constants.setMemberType(mlir::pyc::CodeObjectMemberType::Constants);
    else
        return nullptr;
#endif

#ifdef PYC_NAMES
    if (auto names = dyn_cast_or_null<mlir::pyc::CodeObjectMember>(
            parseObj(ctx, parser, builder)))
        names.setMemberType(mlir::pyc::CodeObjectMemberType::Names);
    else
        return nullptr;
#endif

#ifdef PYC_LOCALSPLUSNAMES
    if (auto names = dyn_cast_or_null<mlir::pyc::CodeObjectMember>(
            parseObj(ctx, parser, builder)))
        names.setMemberType(mlir::pyc::CodeObjectMemberType::LocalPlusNames);
    else
        return nullptr;
#endif

#ifdef PYC_LOCALSPLUSKINDS
    if (auto localPlusKinds = dyn_cast_or_null<mlir::pyc::CodeObjectMember>(
            parseObj(ctx, parser, builder)))
        localPlusKinds.setMemberType(
            mlir::pyc::CodeObjectMemberType::LocalPlusKinds);
    else
        return nullptr;
#endif

#ifdef PYC_FILENAME
    if (auto filename = dyn_cast_or_null<mlir::pyc::CodeObjectMember>(
            parseObj(ctx, parser, builder)))
        filename.setMemberType(mlir::pyc::CodeObjectMemberType::Filename);
    else
        return nullptr;
#endif

#ifdef PYC_NAME
    if (auto name = dyn_cast_or_null<mlir::pyc::CodeObjectMember>(
            parseObj(ctx, parser, builder)))
        name.setMemberType(mlir::pyc::CodeObjectMemberType::Name);
    else
        return nullptr;
#endif

#ifdef PYC_QUALNAME
    if (auto name = dyn_cast_or_null<mlir::pyc::CodeObjectMember>(
            parseObj(ctx, parser, builder)))
        name.setMemberType(mlir::pyc::CodeObjectMemberType::QualName);
    else
        return nullptr;
#endif

#ifdef PYC_FIRSTLINENO
    auto firstLineNo = parser.getSInt32();
    if (failed(firstLineNo))
        return nullptr;
    codeOp->setAttr("pyc.first_line_no",
                    builder.getI32IntegerAttr(*firstLineNo));
#endif

#ifdef PYC_LINETABLE
    if (auto table = dyn_cast_or_null<mlir::pyc::CodeObjectMember>(
            parseObj(ctx, parser, builder)))
        table.setMemberType(mlir::pyc::CodeObjectMemberType::LineTable);
    else
        return nullptr;
#endif

#ifdef PYC_EXCEPTIONTABLE
    if (auto table = dyn_cast_or_null<mlir::pyc::CodeObjectMember>(
            parseObj(ctx, parser, builder)))
        table.setMemberType(mlir::pyc::CodeObjectMemberType::LineTable);
    else
        return nullptr;
#endif

    return codeOp;
}

mlir::pyc::ConstantOp parseString(ObjectType type, Parser &parser,
                                  mlir::OpBuilder &builder) {}

mlir::pyc::CollectionOp parseCollectionOp(ObjectType type, Parser &parser,
                                          mlir::OpBuilder &builder) {}

mlir::pyc::ConstantOp parsePrimitiveType(ObjectType type, Parser &parser,
                                         mlir::OpBuilder &builder) {}

mlir::Operation *makeRefObject(uint32_t idx, mlir::OpBuilder &builder) {
    std::string refName = getRefSymbolName(idx);
    auto ref = builder.getStringAttr(refName);
    return builder.create<mlir::pyc::RefOp>(builder.getUnknownLoc(), ref);
}

mlir::Operation *parseObj(ParserContext &ctx, Parser &parser,
                          mlir::OpBuilder &builder) {
    auto type = parser.getUInt8();
    if (failed(type))
        return nullptr;
    uint8_t constexpr kTypeObjRef = 0x80;
    if (type == static_cast<int8_t>(ObjectType::TYPE_REF)) {
        auto idx = parser.getUInt32();
        if (failed(idx))
            return nullptr;
        // the index name is global to the entire module
        return makeRefObject(*idx, builder);
    }

    // based on object type
    mlir::Operation *res = nullptr;
    auto objType = static_cast<ObjectType>(*type);
    switch (objType) {
    case ObjectType::TYPE_ASCII:
    case ObjectType::TYPE_ASCII_INTERNED:
    case ObjectType::TYPE_SHORT_ASCII:
    case ObjectType::TYPE_SHORT_ASCII_INTERNED:
    case ObjectType::TYPE_STRING:
    case ObjectType::TYPE_INTERNED:
    case ObjectType::TYPE_UNICODE:
        res = parseString(objType, parser, builder);
        break;
    case ObjectType::TYPE_TUPLE:
    case ObjectType::TYPE_LIST:
    case ObjectType::TYPE_SMALL_TUPLE:
    case ObjectType::TYPE_SET:
    case ObjectType::TYPE_FROZENSET:
    case ObjectType::TYPE_DICT:
        res = parseCollectionOp(objType, parser, builder);
        break;
    case ObjectType::TYPE_CODE:
        res = parseCodeObj(ctx, parser, builder);
        break;
    case ObjectType::TYPE_BINARY_FLOAT:
    case ObjectType::TYPE_FLOAT:
    case ObjectType::TYPE_INT:
    case ObjectType::TYPE_NONE:
    case ObjectType::TYPE_TRUE:
    case ObjectType::TYPE_FALSE:
    case ObjectType::TYPE_LONG:
        res = parsePrimitiveType(objType, parser, builder);
        break;
    default:
        llvm::errs() << "Unsupported obj type '" << *type << "'\n";
        return nullptr;
    }

    if (*type & kTypeObjRef && res) {
        // flag set, move it to the reference collection
        // and replace it with a ref
        auto idx = ctx.addRefOp(res, builder);
        res = makeRefObject(idx, builder);
    }
    return res;
}

} // namespace

namespace mlir {

LogicalResult parseModule(llvm::MemoryBuffer &buffer, ModuleOp moduleOp,
                          OpBuilder &builder) {
    Parser parser(buffer);
    auto version = parser.getUInt32();
    if (failed(version))
        return failure();
    moduleOp->setAttr("pyc.version", builder.getUI32IntegerAttr(*version));
    auto flags = parser.getUInt32();
    if (failed(flags))
        return failure();
    moduleOp->setAttr("pyc.flags", builder.getUI32IntegerAttr(*flags));

    if (*flags & 0x1) {
        // checksum
        // not supported
        return moduleOp->emitError("checksum not supported");
    } else {
        // timestamp
        auto timestamp = parser.getUInt32();
        if (failed(timestamp))
            return failure();
        moduleOp->setAttr("pyc.timestamp",
                          builder.getUI32IntegerAttr(*timestamp));

        // size parameter
        auto size = parser.getUInt32();
        if (failed(size))
            return failure();
        moduleOp->setAttr("pyc.size", builder.getUI32IntegerAttr(*size));
    }

    builder.setInsertionPointToEnd(moduleOp.getBody());
    ParserContext parserContext(moduleOp, builder);
    return success(parseObj(parserContext, parser, builder));
}
} // namespace mlir