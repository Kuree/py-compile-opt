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


def CodeObjectMemberInterface : PycOpInterface<"CodeObjectMember"> {
  let methods = [
    InterfaceMethod<"", "std::optional<::mlir::pyc::CodeObjectMemberType>", "getMemberType",
    (ins), /*methodBody=*/[{}],
    /*defaultImplementation=*/[{
    if (auto attr = $_op->template getAttrOfType<::mlir::pyc::CodeObjectMemberTypeAttr>("objType")) return attr.getValue();
      return std::nullopt;
    }]>,
    InterfaceMethod<"", "void", "setMemberType",
    (ins "::mlir::pyc::CodeObjectMemberType":$value), /*methodBody=*/[{}],
    /*defaultImplementation=*/[{
      $_op.setObjType(value);
    }]>
  ];
}
