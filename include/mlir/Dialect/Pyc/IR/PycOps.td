include "mlir/Dialect/Pyc/IR/PycDialect.td"
include "mlir/Dialect/Pyc/IR/PycOpsInterfaces.td"
include "mlir/Dialect/Pyc/IR/PycOpsEnums.td"

include "mlir/IR/BuiltinAttributes.td"

// op traits
def IsConstantPop : NativeOpTrait<"IsConstantPop"> {
  let cppNamespace = "::mlir::pyc";
}
def IsConstantPush : NativeOpTrait<"IsConstantPush"> {
  let cppNamespace = "::mlir::pyc";
}
def PopsStack : NativeOpTrait<"PopsStack"> {
  let cppNamespace = "::mlir::pyc";
}
def PushesStack : NativeOpTrait<"PushesStack"> {
  let cppNamespace = "::mlir::pyc";
}
def DoesNotChangeStackSize : NativeOpTrait<"DoesNotChangeStackSize"> {
  let cppNamespace = "::mlir::pyc";
}

class Pyc_Op<string mnemonic, list<Trait> traits = []> : Op<Pyc_Dialect, mnemonic, traits> {

}

class Pyc_ObjOp<string mnemonic, list<Trait> traits = []> : Pyc_Op<mnemonic, traits> {
    let arguments = (ins Builtin_IntegerAttr:$opArg);

    let assemblyFormat = "attr-dict $opArg";
}

def ConstantPop : NativeOpTrait<"ConstantPop"> {
  let cppNamespace = "::mlir::pyc";
}

def ConstantPush : NativeOpTrait<"ConstantPush"> {
  let cppNamespace = "::mlir::pyc";
}

def PopStack : NativeOpTrait<"PopStack"> {
  let cppNamespace = "::mlir::pyc";
}

def PushStack : NativeOpTrait<"PushStack"> {
  let cppNamespace = "::mlir::pyc";
}

def StackUnchanged : NativeOpTrait<"StackUnchanged"> {
  let cppNamespace = "::mlir::pyc";
}

def Pyc_CodeOp : Pyc_Op<"code", [IsolatedFromAbove, NoTerminator, SingleBlock]> {
  let regions = (region SizedRegion<1>:$region);

  let assemblyFormat = "$region attr-dict";
}

def Pyc_ConstantOp : Pyc_Op<"constant", [CodeObjectMemberInterface]> {
   let arguments = (ins TypedAttrInterface:$value, OptionalAttr<Pyc_CodeObjectMemberType>:$objType);

   let assemblyFormat = "$value ($objType^)? attr-dict";
}

def Pyc_CollectionOp : Pyc_Op<"collection", [IsolatedFromAbove, NoTerminator, SingleBlock,
                                             CodeObjectMemberInterface]> {
  let arguments = (ins Pyc_CollectionType:$collectionType, OptionalAttr<Pyc_CodeObjectMemberType>:$objType);
  let regions = (region SizedRegion<1>:$region);

  let assemblyFormat = "$collectionType ($objType^)? $region attr-dict";
}

#define INCLUDE_TABLEGEN
include "mlir/Dialect/Pyc/IR/GeneratedOps.td"
