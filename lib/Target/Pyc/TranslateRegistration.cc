#include "mlir/Dialect/Pyc/IR/Pyc.hh"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Target/Pyc/Parser.hh"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir;

namespace mlir {
void registerFromPycTranslation() {
    static TranslateToMLIRRegistration registration{
        "import-pyc", "Translate PYC to MLIR",
        [](llvm::SourceMgr &sourceMgr,
           MLIRContext *context) -> OwningOpRef<Operation *> {
            return mlir::parseModule(sourceMgr, context);
        },
        [](DialectRegistry &registry) { registry.insert<pyc::PycDialect>(); }};
    (void)registration;
}
} // namespace mlir
