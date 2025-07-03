include "mlir/IR/OpBase.td"

class PycOpInterface<string name> : OpInterface<name> {
  let cppNamespace = "::mlir::pyc";
}

def StackInterface : PycOpInterface<"StackInterface"> {
  let methods = [
    InterfaceMethod<"", "int32_t", "getNumOfStackPopped">,
    InterfaceMethod<"", "int32_t", "getNumOfStackPushed">,
  ];
}

def SerializableInterface : PycOpInterface<"Serializable"> {
  let methods = [
    InterfaceMethod<"", "std::pair<uint8_t, uint8_t>", "serialize">
  ];
}
