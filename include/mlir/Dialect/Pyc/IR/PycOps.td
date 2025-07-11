include "mlir/Dialect/Pyc/IR/PycDialect.td"
include "mlir/Dialect/Pyc/IR/PycOpsInterfaces.td"
include "mlir/Dialect/Pyc/IR/PycOpsEnums.td"

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/SymbolInterfaces.td"

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

def Pyc_CodeObjectOp : Pyc_Op<"code_obj", [IsolatedFromAbove, NoTerminator, SingleBlock, NoRegionArguments,
                                           ReferencableObjectInterface]> {
  let arguments = (ins DefaultValuedOptionalAttr<SymbolNameAttr, "{}">:$reference);
  let regions = (region SizedRegion<1>:$region);

  let assemblyFormat = "($reference^)? $region attr-dict";
}

def Pyc_CodeOp : Pyc_Op<"code", [IsolatedFromAbove, NoTerminator, SingleBlock, NoRegionArguments,
                                 HasParent<"CodeObjectOp">]> {
  let regions = (region SizedRegion<1>:$region);

  let assemblyFormat = "$region attr-dict";
}

def Pyc_ConstantOp : Pyc_Op<"constant", [CodeObjectMemberInterface, ReferencableObjectInterface]> {
   let arguments = (ins TypedAttrInterface:$value, DefaultValuedOptionalAttr<SymbolNameAttr, "{}">:$reference,
                    OptionalAttr<Pyc_CodeObjectMemberType>:$objType);

    let builders = [
      OpBuilder<(ins "TypedAttr":$attr), [{
        $_state.addAttribute("value", attr);
      }]>,
    ];

   let assemblyFormat = "($reference^)? $value ($objType^)? attr-dict";
}

def Pyc_CollectionOp : Pyc_Op<"collection", [IsolatedFromAbove, NoTerminator, SingleBlock, NoRegionArguments,
                                             CodeObjectMemberInterface, ReferencableObjectInterface]> {
  let arguments = (ins Pyc_CollectionType:$collectionType, DefaultValuedOptionalAttr<SymbolNameAttr, "{}">:$reference,
                   OptionalAttr<Pyc_CodeObjectMemberType>:$objType);
  let regions = (region SizedRegion<1>:$region);

  let builders = [
    OpBuilder<(ins "CollectionType":$type), [{
      $_state.addAttribute("collectionType", $_builder.getAttr<CollectionTypeAttr>(type));
      $_state.addRegion();
    }]>,
  ];

  let assemblyFormat = "($reference^)? $collectionType ($objType^)? $region attr-dict";
}

def Pyc_MakeRefOp : Pyc_Op<"make_ref", [Symbol]> {
    let arguments = (ins SymbolNameAttr:$sym_name);

    let assemblyFormat = "$sym_name attr-dict";
}

def Pyc_RefOp : Pyc_Op<"ref", [SymbolUserOpInterface, CodeObjectMemberInterface]> {
  let arguments = (ins FlatSymbolRefAttr:$ref, OptionalAttr<Pyc_CodeObjectMemberType>:$objType);

  let extraClassDeclaration = [{
    llvm::LogicalResult verifySymbolUses(::mlir::SymbolTableCollection &symbolTable);
  }];

  let assemblyFormat = "$ref  ($objType^)? attr-dict";
}

def Pyc_KeyValuePairOp : Pyc_Op<"pair", [IsolatedFromAbove, NoTerminator, SingleBlock, NoRegionArguments,
                                         HasParent<"CollectionOp">]> {
    let regions = (region SizedRegion<1>:$region);

    let assemblyFormat = "$region attr-dict";

    let hasVerifier = 1;
}

#define INCLUDE_TABLEGEN
include "mlir/Dialect/Pyc/IR/GeneratedOps.td"
