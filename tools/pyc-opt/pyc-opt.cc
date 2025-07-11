#include "mlir/Dialect/Pyc/IR/Pyc.hh"
#include "mlir/Dialect/Pyc/Transforms/Passes.hh"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char *argv[]) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::pyc::PycDialect>();
    mlir::MlirOptMainConfig config;
    mlir::pyc::registerPasses();
    return mlir::failed(
        mlir::MlirOptMain(argc, argv, "Pyc Optimization Tool", registry));
}
