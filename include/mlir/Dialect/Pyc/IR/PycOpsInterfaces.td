include "mlir/IR/OpBase.td"

class PyCOpInterface<string name> : OpInterface<name> {
  let cppNamespace = "::mlir::pyc";
}

def StackInterface : PyCOpInterface<"StackInterface"> {
  let methods = [
    InterfaceMethod<"", "int32_t", "getNumOfStackPopped">,
    InterfaceMethod<"", "int32_t", "getNumOfStackPushed">,
  ];
}
