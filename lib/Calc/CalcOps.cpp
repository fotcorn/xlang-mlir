//===- CalcOps.cpp - Calc dialect ops ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Calc/CalcOps.h"
#include "Calc/CalcDialect.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "Calc/CalcOps.cpp.inc"

namespace calc {
void ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       int64_t value) {
  auto dataType = builder.getIntegerType(64, true);
  auto dataAttribute = mlir::IntegerAttr::get(dataType, value);
  ConstantOp::build(builder, state, dataType, dataAttribute);
}

void GetVariableOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                          llvm::StringRef variableName) {
  GetVariableOp::build(builder, state, builder.getIntegerType(64, true),
                       variableName);
}

// arithmetic Ops

void AddOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(builder.getIntegerType(64, true));
  state.addOperands({lhs, rhs});
}

void SubOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(builder.getIntegerType(64, true));
  state.addOperands({lhs, rhs});
}

void MulOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(builder.getIntegerType(64, true));
  state.addOperands({lhs, rhs});
}

void DivOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(builder.getIntegerType(64, true));
  state.addOperands({lhs, rhs});
}

} // namespace calc
