#include "mlir/Dialect/Pyc/InitAllTranslations.hh"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

int main(int argc, char **argv) {
    mlir::registerFromPycTranslation();

    return failed(
        mlir::mlirTranslateMain(argc, argv, "Pyc Translation Testing Tool"));
}
