include "mlir/IR/EnumAttr.td"

def Pyc_CollectionType : I32EnumAttr<
    "CollectionType", "",
    [
      I32EnumAttrCase<"tuple", 0>,
      I32EnumAttrCase<"list", 1>,
      I32EnumAttrCase<"set", 2>,
      I32EnumAttrCase<"dist", 3>,
    ]> {
  let cppNamespace = "::mlir::pyc";
}

def Pyc_CodeObjectMemberType : I32EnumAttr<
    "CodeObjectMemberType", "",
    [
      I32EnumAttrCase<"ArgCount", 0>,
      I32EnumAttrCase<"PosOnlyArgCount", 1>,
      I32EnumAttrCase<"KWOnlyArgCount", 2>,
      I32EnumAttrCase<"StackSize", 3>,
      I32EnumAttrCase<"Flags", 4>,
      I32EnumAttrCase<"Code", 5>,
      I32EnumAttrCase<"Constants", 6>,
      I32EnumAttrCase<"Names", 7>,
      I32EnumAttrCase<"LocalPlusNames", 8>,
      I32EnumAttrCase<"LocalPlusKinds", 9>,
      I32EnumAttrCase<"Filename", 10>,
      I32EnumAttrCase<"Name", 11>,
      I32EnumAttrCase<"QualName", 12>,
      I32EnumAttrCase<"FirstLineNo", 13>,
      I32EnumAttrCase<"LineTable", 14>,
      I32EnumAttrCase<"ExceptionTable", 15>,
    ]> {
  let cppNamespace = "::mlir::pyc";
}
