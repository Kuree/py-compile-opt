#include "mlir/Dialect/Pyc/IR/Pyc.hh"
#include "mlir/Dialect/Pyc/Transforms/Passes.hh"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DEF_ELABORATEREFERENCE
#include "mlir/Dialect/Pyc/Transforms/Passes.hh.inc"

namespace {

using mlir::failure;
using mlir::success;

struct ElaborateRefOp : mlir::OpRewritePattern<mlir::pyc::MakeRefOp> {
    using mlir::OpRewritePattern<mlir::pyc::MakeRefOp>::OpRewritePattern;

    mlir::LogicalResult
    matchAndRewrite(mlir::pyc::MakeRefOp op,
                    mlir::PatternRewriter &rewriter) const override {
        auto symbol = op.getSymNameAttr();
        // find users
        mlir::pyc::ReferencableObjectInterface reference;
        llvm::SmallVector<mlir::Operation *> operations;
        auto symbolTableOp = mlir::SymbolTable::getNearestSymbolTable(op);
        if (auto uses =
                mlir::SymbolTable::getSymbolUses(symbol, symbolTableOp)) {
            for (auto use : *uses) {
                auto *user = use.getUser();
                if (auto ref =
                        mlir::dyn_cast<mlir::pyc::ReferencableObjectInterface>(
                            user)) {
                    reference = ref;
                } else {
                    operations.emplace_back(user);
                }
            }
        }

        if (!reference) {
            rewriter.eraseOp(op);
            return success();
        }

        for (auto *user : operations) {
            rewriter.setInsertionPoint(user);
            auto *newOp = rewriter.clone(*reference);
            // TODO: use better names
            newOp->removeAttr("reference");
            // copy all attributes over
            for (auto attr : user->getAttrs()) {
                if (attr.getName() != "ref") {
                    newOp->setAttr(attr.getName(), attr.getValue());
                }
            }
            rewriter.eraseOp(user);
        }

        rewriter.eraseOp(op);
        return success();
    }
};

struct ElaborateReference : impl::ElaborateReferenceBase<ElaborateReference> {
    void runOnOperation() override {
        mlir::RewritePatternSet patterns(&getContext());
        patterns.insert<ElaborateRefOp>(&getContext());
        if (failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                      std::move(patterns))))
            return signalPassFailure();
    }
};

} // namespace

namespace mlir::pyc {
std::unique_ptr<::mlir::Pass> createElaborateReference() {
    return std::make_unique<ElaborateReference>();
}
} // namespace mlir::pyc
