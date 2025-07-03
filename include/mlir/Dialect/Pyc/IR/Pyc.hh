#ifndef PY_COMPILE_OPT_PYC_HH
#define PY_COMPILE_OPT_PYC_HH

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

// Dialect
#include "mlir/Dialect/Pyc/IR/PycOpsDialect.h.inc"

// Interfaces
#include "mlir/Dialect/Pyc/IR/PycOpsInterfaces.h.inc"

// Traits
namespace mlir::pyc {
template <typename ConcreteType>
class IsConstantPop
    : public mlir::OpTrait::TraitBase<ConcreteType, IsConstantPop> {};
template <typename ConcreteType>
class IsConstantPush
    : public mlir::OpTrait::TraitBase<ConcreteType, IsConstantPush> {};
template <typename ConcreteType>
class PopsStack : public mlir::OpTrait::TraitBase<ConcreteType, PopsStack> {};
template <typename ConcreteType>
class PushesStack : public mlir::OpTrait::TraitBase<ConcreteType, PushesStack> {
};
template <typename ConcreteType>
class DoesNotChangeStackSize
    : public mlir::OpTrait::TraitBase<ConcreteType, DoesNotChangeStackSize> {};
} // namespace mlir::pyc

// Operations
#define GET_OP_CLASSES
#include "mlir/Dialect/Pyc/IR/PycOps.h.inc"

#endif // PY_COMPILE_OPT_PYC_HH
