//===- CalcOps.td - Calc dialect ops -----------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CALC_OPS
#define CALC_OPS

include "Calc/CalcDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def ConstantOp : Calc_Op<"constant",
    [ConstantLike, Pure]> {
  let summary = "constant";
  let arguments = (ins SI64Attr:$value);
  let results = (outs SI64);
  let builders = [
    OpBuilder<(ins "int64_t":$value)>
  ];
}

def AddOp : Calc_Op<"add"> {
  let summary = "add";
  let arguments = (ins SI64:$lhs, SI64:$rhs);
  let results = (outs SI64);
  let builders = [
    OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs)>
  ];
}

def SubOp : Calc_Op<"sub"> {
  let summary = "sub";
  let arguments = (ins SI64:$lhs, SI64:$rhs);
  let results = (outs SI64);
  let builders = [
    OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs)>
  ];
}

def MulOp : Calc_Op<"mul"> {
  let summary = "mul";
  let arguments = (ins SI64:$lhs, SI64:$rhs);
  let results = (outs SI64);
  let builders = [
    OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs)>
  ];
}

def DivOp : Calc_Op<"div"> {
  let summary = "div";
  let arguments = (ins SI64:$lhs, SI64:$rhs);
  let results = (outs SI64);
  let builders = [
    OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs)>
  ];
}

def GetVariableOp : Calc_Op<"getVar"> {
  let summary = "getVar";
  let arguments = (ins StrAttr:$varName);
  let results = (outs SI64);
  let builders = [
    OpBuilder<(ins "llvm::StringRef":$varName)>
  ];
}

def SetVariableOp : Calc_Op<"setVar"> {
  let summary = "setVar";
  let arguments = (ins StrAttr:$varName, SI64:$value);
}

def PrintOp : Calc_Op<"print"> {
  let summary = "print";
  let arguments = (ins SI64:$value);
}

#endif // CALC_OPS
