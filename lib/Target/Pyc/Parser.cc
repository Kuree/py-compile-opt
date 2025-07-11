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
    return llvm::formatv("ref_{0}", idx);
}

uint8_t constexpr kTypeObjRef = 0x80;

struct ParserContext {
    explicit ParserContext(mlir::ModuleOp moduleOp) : moduleOp(moduleOp) {}

    mlir::OpBuilder::InsertPoint getRefInsertionPoint() {
        auto *block = moduleOp.getBody();
        return {block, block->end()};
    }

    void addRefOp(mlir::StringRef refName, mlir::OpBuilder &builder) {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.restoreInsertionPoint(getRefInsertionPoint());
        builder.create<mlir::pyc::MakeRefOp>(builder.getUnknownLoc(), refName);
    }

    uint32_t getNextRefCount() { return refCount++; }

  private:
    mlir::ModuleOp moduleOp;

    uint32_t refCount = 0;
};

class Parser {
  public:
    explicit Parser(const llvm::MemoryBuffer &buffer)
        : buffer(buffer.getBuffer()), pos(0) {}
    explicit Parser(llvm::StringRef buffer) : buffer(buffer), pos(0) {}

    auto getUInt8() { return parseBytes<uint8_t>(); }
    auto getSInt8() { return parseBytes<int8_t>(); }
    auto getUInt16() { return parseBytes<uint16_t>(); }
    [[maybe_unused]] auto getSInt16() { return parseBytes<int16_t>(); }
    auto getUInt32() { return parseBytes<uint32_t>(); }
    auto getSInt32() { return parseBytes<int32_t>(); }
    auto getDouble() { return parseBytes<double>(); }

    llvm::FailureOr<llvm::StringRef> getBytes(uint32_t size) {
        auto nextPos = pos + size;
        if (buffer.size() < nextPos) {
            llvm::errs() << "expect " << size << "bytes at position " << pos
                         << "\n";
            return failure();
        }
        auto res = buffer.slice(pos, pos + size);
        pos += size;
        return res;
    }

  private:
    llvm::StringRef buffer;
    uint64_t pos;

    template <typename T> llvm::FailureOr<T> parseBytes() {
        auto constexpr size = sizeof(T);
        auto res = getBytes(size);
        if (failed(res))
            return failure();
        T v;
        std::memcpy(&v, res->data(), size);
        return v;
    }
};

mlir::Operation *parseObj(ParserContext &ctx, Parser &parser,
                          mlir::OpBuilder &builder);

// NOLINTNEXTLINE
mlir::Operation *parseCodeObj(ParserContext &ctx, Parser &parser,
                              mlir::OpBuilder &builder) {
    auto codeObjOp =
        builder.create<mlir::pyc::CodeObjectOp>(builder.getUnknownLoc());
    auto *block = builder.createBlock(&codeObjOp.getBodyRegion());
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(block);

    // use macro to control the parsing

#ifdef PYC_ARGCOUNT
    auto argCount = parser.getSInt32();
    if (failed(argCount))
        return nullptr;
    codeObjOp->setAttr("pyc.arg_count", builder.getI32IntegerAttr(*argCount));
#endif

#ifdef PYC_POSONLYARGCOUNT
    auto posOnlyArgCount = parser.getSInt32();
    if (failed(posOnlyArgCount))
        return nullptr;
    codeObjOp->setAttr("pyc.pos_only_arg_count",
                       builder.getI32IntegerAttr(*posOnlyArgCount));
#endif

#ifdef PYC_KWONLYARGCOUNT
    auto kwOnlyArgCount = parser.getSInt32();
    if (failed(kwOnlyArgCount))
        return nullptr;
    codeObjOp->setAttr("pyc.kw_only_arg_count",
                       builder.getI32IntegerAttr(*kwOnlyArgCount));
#endif

#ifdef PYC_STACKSIZE
    auto stackSize = parser.getSInt32();
    if (failed(stackSize))
        return nullptr;
    codeObjOp->setAttr("pyc.stackSize", builder.getI32IntegerAttr(*stackSize));
#endif

#ifdef PYC_FLAGS
    auto flags = parser.getSInt32();
    if (failed(flags))
        return nullptr;
    codeObjOp->setAttr("pyc.flags", builder.getI32IntegerAttr(*flags));
#endif

#ifdef PYC_CODE

    // for now, we just load the code as raw string
    // then replace it later
    if (auto code = dyn_cast_or_null<mlir::pyc::CodeObjectMember>(
            parseObj(ctx, parser, builder)))
        code.setMemberType(mlir::pyc::CodeObjectMemberType::Code);
    else
        return nullptr;

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
    codeObjOp->setAttr("pyc.first_line_no",
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
        table.setMemberType(mlir::pyc::CodeObjectMemberType::ExceptionTable);
    else
        return nullptr;
#endif

    return codeObjOp;
}

mlir::pyc::ConstantOp parseString(ObjectType type, Parser &parser,
                                  mlir::OpBuilder &builder) {
    int32_t size;
    if (type == ObjectType::TYPE_SHORT_ASCII ||
        type == ObjectType::TYPE_SHORT_ASCII_INTERNED) {
        auto s = parser.getSInt8();
        if (failed(s))
            return {};
        // NOLINTNEXTLINE
        size = *s;
    } else {
        auto s = parser.getSInt32();
        if (failed(s))
            return {};
        size = *s;
    }
    if (size < 0)
        return {};
    auto str = parser.getBytes(size);
    if (failed(str))
        return {};
    auto attr = builder.getStringAttr(*str);

    return builder.create<mlir::pyc::ConstantOp>(builder.getUnknownLoc(), attr);
}

// NOLINTNEXTLINE
mlir::pyc::CollectionOp parseCollectionOp(ObjectType type, ParserContext &ctx,
                                          Parser &parser,
                                          mlir::OpBuilder &builder) {
    using mlir::pyc::CollectionType;
    if (type == ObjectType::TYPE_DICT) {
        // this is null terminated
        auto res = builder.create<mlir::pyc::CollectionOp>(
            builder.getUnknownLoc(), CollectionType::dict);
        mlir::OpBuilder::InsertionGuard guard(builder);
        auto *block = builder.createBlock(&res.getBodyRegion());
        builder.setInsertionPointToEnd(block);
        // dict is null terminated
        while (auto key = parseObj(ctx, parser, builder)) {
            auto value = parseObj(ctx, parser, builder);
            auto kvp = builder.create<mlir::pyc::KeyValuePairOp>(
                builder.getUnknownLoc());
            auto *kvpBlock = builder.createBlock(&kvp.getBodyRegion());
            key->moveBefore(kvpBlock, kvpBlock->end());
            value->moveBefore(kvpBlock, kvpBlock->end());
        }
        return res;
    } else {
        CollectionType collectionType;
        switch (type) {
        case ObjectType::TYPE_TUPLE:
        case ObjectType::TYPE_SMALL_TUPLE:
            collectionType = CollectionType::tuple;
            break;
        case ObjectType::TYPE_LIST:
            collectionType = CollectionType::list;
            break;

        case ObjectType::TYPE_FROZENSET:
            collectionType = CollectionType::frozenset;
            break;
        case ObjectType::TYPE_SET:
            collectionType = CollectionType::set;
            break;
        default:
            llvm::errs() << "Unsupported collection type\n";
            return {};
        }

        uint32_t size;
        if (type == ObjectType::TYPE_SMALL_TUPLE) {
            auto s = parser.getUInt8();
            if (failed(s))
                return {};
            size = *s;
        } else {
            auto s = parser.getUInt32();
            if (failed(s))
                return {};
            size = *s;
        }
        auto res = builder.create<mlir::pyc::CollectionOp>(
            builder.getUnknownLoc(), collectionType);
        mlir::OpBuilder::InsertionGuard guard(builder);
        auto *block = builder.createBlock(&res.getBodyRegion());
        builder.setInsertionPointToEnd(block);
        for (auto i = 0u; i < size; i++) {
            parseObj(ctx, parser, builder);
        }
        return res;
    }
}

mlir::pyc::ConstantOp parsePrimitiveType(ObjectType type, Parser &parser,
                                         mlir::OpBuilder &builder) {
    mlir::TypedAttr value;
    switch (type) {
    case ObjectType::TYPE_INT: {
        // 32b
        auto i = parser.getSInt32();
        if (failed(i))
            return {};
        value = builder.getI32IntegerAttr(*i);
        break;
    }
    case ObjectType::TYPE_LONG: {
        // packed as 16bits
        auto size = parser.getUInt32();
        if (failed(size))
            return {};
        llvm::SmallVector<uint64_t> values((int)(std::ceil(*size / 4.0)));
        std::fill(values.begin(), values.end(), 0);
        for (auto i = 0; i < *size; i++) {
            uint64_t sh = i % 4;
            auto idx = i / 4;
            auto &v = values[idx];
            auto s = parser.getUInt16();
            if (failed(s))
                return {};
            v |= static_cast<uint64_t>(*s) << sh;
        }
        auto totalBits = *size * 16;
        llvm::APInt apInt{totalBits, values};
        value =
            builder.getIntegerAttr(builder.getIntegerType(totalBits), apInt);
        break;
    }
    case ObjectType::TYPE_TRUE: {
        value = builder.getBoolAttr(true);
        break;
    }
    case ObjectType::TYPE_FALSE: {
        value = builder.getBoolAttr(false);
        break;
    }
    case ObjectType::TYPE_BINARY_FLOAT: {
        auto v = parser.getDouble();
        if (failed(v))
            return {};
        value = builder.getF64FloatAttr(*v);
        break;
    }
    case ObjectType::TYPE_NONE:
        value = builder.getZeroAttr(builder.getIntegerType(0));
        break;
    default:
        llvm::errs() << "invalid/unsupported primitive type\n";
        return {};
    }
    return builder.create<mlir::pyc::ConstantOp>(
        builder.getUnknownLoc(), value, mlir::StringAttr{},
        mlir::pyc::CodeObjectMemberTypeAttr{});
}

mlir::Operation *makeRefObject(uint32_t idx, mlir::OpBuilder &builder) {
    std::string refName = getRefSymbolName(idx);
    auto ref = builder.getStringAttr(refName);
    return builder.create<mlir::pyc::RefOp>(
        builder.getUnknownLoc(), ref, mlir::pyc::CodeObjectMemberTypeAttr{});
}

/// NOLINTNEXTLINE
mlir::Operation *parseObj(ParserContext &ctx, Parser &parser,
                          mlir::OpBuilder &builder) {
    auto type = parser.getUInt8();
    if (failed(type))
        return nullptr;

    if (type == static_cast<int8_t>(ObjectType::TYPE_REF)) {
        auto idx = parser.getUInt32();
        if (failed(idx))
            return nullptr;
        // the index name is global to the entire module
        return makeRefObject(*idx, builder);
    }

    // based on object type
    mlir::Operation *res = nullptr;
    auto objType = static_cast<ObjectType>(*type & 0x7F);

    if (objType == ObjectType::TYPE_NULL) {
        // don't care
        return nullptr;
    }

    // make reference to this object before we start parsing
    std::optional<uint32_t> ref;
    if (*type & kTypeObjRef) {
        ref = ctx.getNextRefCount();
    }

    switch (objType) {
    case ObjectType::TYPE_NULL:
        return nullptr;
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
        res = parseCollectionOp(objType, ctx, parser, builder);
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
        llvm::errs() << "Unsupported obj type '" << *type << "' ("
                     << static_cast<uint32_t>(*type) << ")\n";
        return nullptr;
    }

    if (ref) {
        assert(res);
        // flag set, move it to the reference collection
        // and replace it with a ref
        auto name = getRefSymbolName(*ref);
        auto referenceOp =
            dyn_cast<mlir::pyc::ReferencableObjectInterface>(res);
        if (!referenceOp) {
            llvm::errs() << "Unable to makre a reference object\n";
            return nullptr;
        }
        referenceOp.setReference(builder.getStringAttr(name));
        ctx.addRefOp(name, builder);
    }
    return res;
}

} // namespace

namespace mlir {

LogicalResult parseModule(const llvm::MemoryBuffer &buffer, ModuleOp moduleOp,
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
    ParserContext parserContext(moduleOp);
    return success(parseObj(parserContext, parser, builder));
}

OwningOpRef<Operation *> parseModule(llvm::SourceMgr &srcMgr,
                                     mlir::MLIRContext *ctx) {
    if (srcMgr.getNumBuffers() != 1) {
        llvm::errs() << "Only one buffer supported\n";
        return {};
    }
    ctx->loadAllAvailableDialects();

    auto buffer = srcMgr.getMemoryBuffer(srcMgr.getMainFileID());
    OpBuilder builder(ctx);
    auto loc = mlir::FileLineColLoc::get(
        builder.getStringAttr(buffer->getBufferIdentifier()), 0, 0);
    OwningOpRef<ModuleOp> mod = ModuleOp::create(loc);

    if (failed(parseModule(*buffer, *mod, builder)))
        return {};
    return mod;
}

} // namespace mlir