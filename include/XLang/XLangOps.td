//===- XLangOps.td - XLang dialect ops -----------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef XLANG_OPS
#define XLANG_OPS

include "XLang/XLangDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def ConstantOp : XLang_Op<"constant",
    [ConstantLike, Pure]> {
  let summary = "constant";
  let arguments = (ins SI64Attr:$value);
  let results = (outs SI64);
  let builders = [
    OpBuilder<(ins "int64_t":$value)>
  ];
}

def AddOp : XLang_Op<"add"> {
  let summary = "add";
  let arguments = (ins SI64:$lhs, SI64:$rhs);
  let results = (outs SI64);
  let builders = [
    OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs)>
  ];
}

def SubOp : XLang_Op<"sub"> {
  let summary = "sub";
  let arguments = (ins SI64:$lhs, SI64:$rhs);
  let results = (outs SI64);
  let builders = [
    OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs)>
  ];
}

def MulOp : XLang_Op<"mul"> {
  let summary = "mul";
  let arguments = (ins SI64:$lhs, SI64:$rhs);
  let results = (outs SI64);
  let builders = [
    OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs)>
  ];
}

def DivOp : XLang_Op<"div"> {
  let summary = "div";
  let arguments = (ins SI64:$lhs, SI64:$rhs);
  let results = (outs SI64);
  let builders = [
    OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs)>
  ];
}

def GetVariableOp : XLang_Op<"getVar"> {
  let summary = "getVar";
  let arguments = (ins StrAttr:$varName);
  let results = (outs SI64);
  let builders = [
    OpBuilder<(ins "llvm::StringRef":$varName)>
  ];
}

def SetVariableOp : XLang_Op<"setVar"> {
  let summary = "setVar";
  let arguments = (ins StrAttr:$varName, SI64:$value);
}

def PrintOp : XLang_Op<"print"> {
  let summary = "print";
  let arguments = (ins SI64:$value);
}

#endif // XLANG_OPS
