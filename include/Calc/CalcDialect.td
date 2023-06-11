//===- CalcDialect.td - Calc dialect -----------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CALC_DIALECT
#define CALC_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Calc dialect definition.
//===----------------------------------------------------------------------===//

def Calc_Dialect : Dialect {
    let name = "calc";
    let summary = "A calc out-of-tree MLIR dialect.";
    let cppNamespace = "::calc";
}

//===----------------------------------------------------------------------===//
// Base calc operation definition.
//===----------------------------------------------------------------------===//

class Calc_Op<string mnemonic, list<Trait> traits = []> :
        Op<Calc_Dialect, mnemonic, traits>;

#endif // CALC_DIALECT
