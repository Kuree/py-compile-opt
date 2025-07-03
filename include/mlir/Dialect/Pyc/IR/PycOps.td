include "mlir/Dialect/Pyc/IR/PycDialect.td"
include "mlir/Dialect/Pyc/IR/PycOpsInterfaces.td"

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

class Pyc_CodeOp<string mnemonic, list<Trait> traits = []> : Pyc_Op<mnemonic, traits> {
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

#define INCLUDE_TABLEGEN
include "mlir/Dialect/Pyc/IR/GeneratedOps.td"
