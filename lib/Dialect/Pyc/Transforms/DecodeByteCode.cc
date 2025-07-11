#include "mlir/Dialect/Pyc/IR/Pyc.hh"
#include "mlir/Dialect/Pyc/Transforms/Passes.hh"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DEF_DECODEBYTECODE
#include "mlir/Dialect/Pyc/Transforms/Passes.hh.inc"

namespace {

using mlir::failure;
using mlir::success;

struct ConvertCodeObject : mlir::OpRewritePattern<mlir::pyc::ConstantOp> {
    using mlir::OpRewritePattern<mlir::pyc::ConstantOp>::OpRewritePattern;

    mlir::LogicalResult
    matchAndRewrite(mlir::pyc::ConstantOp op,
                    mlir::PatternRewriter &rewriter) const override {
        // has to be code object type
        auto type = op.getObjType();
        if (type != mlir::pyc::CodeObjectMemberType::Code)
            return failure();

        auto byteCode = mlir::dyn_cast<mlir::StringAttr>(op.getValue());
        if (!byteCode) {
            return op->emitOpError("invalid bytecode object");
        }

        auto buffer = byteCode.getValue();

        rewriter.setInsertionPoint(op);
        auto code =
            rewriter.create<mlir::pyc::CodeOp>(rewriter.getUnknownLoc());
        auto *codeBlock = rewriter.createBlock(&code.getRegion());
        rewriter.setInsertionPointToEnd(codeBlock);
        auto size = byteCode.size();
        assert(size % 2 == 0);
        for (auto i = 0u; i < size / 2; i++) {
            uint8_t opCode = buffer[i * 2];
            uint8_t opArg = buffer[i * 2 + 1];
            mlir::pyc::PycDialect::parseOperation(
                opCode, opArg, rewriter.getUnknownLoc(), rewriter);
        }
        rewriter.eraseOp(op);
        return success();
    }
};

struct DecodeByteCode : impl::DecodeByteCodeBase<DecodeByteCode> {
    void runOnOperation() override {
        mlir::RewritePatternSet patterns(&getContext());
        patterns.insert<ConvertCodeObject>(&getContext());
        if (failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                      std::move(patterns))))
            return signalPassFailure();
    }
};

} // namespace

namespace mlir::pyc {
std::unique_ptr<::mlir::Pass> createDecodeByteCode() {
    return std::make_unique<DecodeByteCode>();
}
} // namespace mlir::pyc
