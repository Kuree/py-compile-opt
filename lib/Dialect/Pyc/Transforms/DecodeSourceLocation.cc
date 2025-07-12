#include "mlir/Dialect/Pyc/IR/Pyc.hh"
#include "mlir/Dialect/Pyc/Transforms/Passes.hh"

#include "mlir/IR/Attributes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DEF_DECODESOURCELOCATION
#include "mlir/Dialect/Pyc/Transforms/Passes.hh.inc"

namespace {

using mlir::failure;
using mlir::success;

struct AddressRange {
    uint32_t startLine = 0;
    uint32_t endLine = 0;
    uint32_t startColumn = 0;
    uint32_t endColumn = 0;

    uint32_t numCode = 0;
};

class Parser {
  public:
    explicit Parser(llvm::StringRef buffer) : buffer(buffer) {}

    [[nodiscard]] bool eof() const { return pos >= buffer.size(); }
    uint8_t readByte() { return buffer[pos++]; }

    uint64_t readVariant() {
        uint64_t result = 0;
        auto b = readByte();
        result = b & 63;
        uint64_t shift = 0;
        while (b & 64) {
            b = readByte();
            shift += 6;
            result |= (b & 63) << shift;
        }
        return result;
    }

    int64_t readSVariant() {
        auto u = readVariant();
        if (u & 1) {
            return -static_cast<int64_t>((u >> 1));
        }
        return static_cast<int64_t>(u) >> 1;
    }

  private:
    llvm::StringRef buffer;
    uint32_t pos = 0;
};

// for python 3.11+
// https://github.com/python/cpython/blob/8d1b3dfa09135affbbf27fb8babcf3c11415df49/Objects/lnotab_notes.txt
llvm::SmallVector<AddressRange> decodeAddressRange(llvm::StringRef buffer,
                                                   uint32_t firstLine) {
    uint32_t index = 0;
    uint32_t opIndex = 0; // we do not need to divide it by 2
    llvm::SmallVector<AddressRange> result;
    Parser parser(buffer);

    uint32_t currentLine = firstLine;

    while (!parser.eof()) {
        auto byte = parser.readByte();
        auto numCode = static_cast<uint32_t>(byte & 7) + 1;
        auto code = (byte >> 3) & 15;

        AddressRange addr{.numCode = numCode};

        if (code <= 9) {
            // short form
            addr.startLine = currentLine;
            addr.endLine = currentLine;

            auto secondByte = parser.readByte();
            addr.startColumn = (code * 8) + ((secondByte >> 4) & 7);
            addr.endColumn = addr.startColumn + (secondByte & 15);
        } else if (code <= 12) {
            // one line form
            currentLine += code - 10;
            addr.startLine = currentLine;
            addr.endLine = currentLine;
            addr.startColumn = parser.readByte();
            addr.endColumn = parser.readByte();
        } else if (code == 13) {
            // no column info
            auto delta = parser.readSVariant();
            currentLine += delta;
            addr.startLine = currentLine;
            addr.endLine = currentLine;
        } else if (code == 14) {
            // long form
            currentLine += parser.readSVariant();
            addr.startLine = currentLine;
            addr.endLine = addr.startLine + parser.readVariant();
            addr.startColumn = parser.readVariant();
            addr.endColumn = parser.readVariant();
        } else {
            assert(code == 15);
            // none
            continue;
        }
        result.emplace_back(addr);
    }
    return result;
}

struct ConvertLineTable : mlir::OpRewritePattern<mlir::pyc::ConstantOp> {
    using mlir::OpRewritePattern<mlir::pyc::ConstantOp>::OpRewritePattern;

    mlir::LogicalResult
    matchAndRewrite(mlir::pyc::ConstantOp op,
                    mlir::PatternRewriter &rewriter) const override {
        // has to be code object type
        auto type = op.getObjType();
        if (type != mlir::pyc::CodeObjectMemberType::LineTable)
            return failure();

        auto byteCode = mlir::dyn_cast<mlir::StringAttr>(op.getValue());
        if (!byteCode) {
            return op->emitOpError("invalid linetable object");
        }

        auto buffer = byteCode.getValue();

        // parent op must have first line num attr
        auto parentOp = op->getParentOp();
        if (!parentOp)
            return failure();
        auto firstLineAttr =
            parentOp->getAttrOfType<mlir::IntegerAttr>("pyc.first_line_no");
        if (!firstLineAttr)
            return failure();
        auto firstLine = firstLineAttr.getInt();

        // need to get decoded code object
        auto *parentBlock = op->getBlock();
        mlir::pyc::CodeOp codeOp;
        llvm::StringRef filename = "<unknown>";
        for (auto &operation : *parentBlock) {
            if (auto c = mlir::dyn_cast<mlir::pyc::CodeOp>(operation)) {
                codeOp = c;
            } else if (auto obj =
                           mlir::dyn_cast<mlir::pyc::ConstantOp>(operation)) {
                if (obj.getMemberType() ==
                    mlir::pyc::CodeObjectMemberType::Filename) {
                    filename = mlir::cast<mlir::StringAttr>(obj.getValue());
                }
            }
        }
        if (!codeOp)
            return failure();

        // index it to a vec
        llvm::SmallVector<mlir::Operation *> ops;
        for (auto &inst : *codeOp.getBody()) {
            ops.emplace_back(&inst);
        }

        auto addrs = decodeAddressRange(buffer, firstLine);

        // decode the line table
        uint32_t opIdx = 0;
        for (auto const &addr : addrs) {
            auto nextIdx = opIdx + addr.numCode;
            // TODO: use custom location to enclose the range
            auto loc =
                mlir::FileLineColLoc::get(rewriter.getStringAttr(filename),
                                          addr.startLine, addr.startColumn);
            for (; opIdx < nextIdx; opIdx++) {
                ops[opIdx]->setLoc(loc);
            }
        }

        // erase this op since it's no longer useful
        rewriter.eraseOp(op);

        return success();
    }
};

struct DecodeSourceLocation
    : impl::DecodeSourceLocationBase<DecodeSourceLocation> {
    void runOnOperation() override {
        mlir::RewritePatternSet patterns(&getContext());
        patterns.insert<ConvertLineTable>(&getContext());
        if (failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                      std::move(patterns))))
            return signalPassFailure();
    }
};

} // namespace

namespace mlir::pyc {
std::unique_ptr<::mlir::Pass> createDecodeSourceLocation() {
    return std::make_unique<DecodeSourceLocation>();
}
} // namespace mlir::pyc
