//===- XLangDialect.td - XLang dialect -----------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef XLANG_DIALECT
#define XLANG_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// XLang dialect definition.
//===----------------------------------------------------------------------===//

def XLang_Dialect : Dialect {
    let name = "xlang";
    let summary = "A xlang out-of-tree MLIR dialect.";
    let cppNamespace = "::xlang";
}

//===----------------------------------------------------------------------===//
// Base xlang operation definition.
//===----------------------------------------------------------------------===//

class XLang_Op<string mnemonic, list<Trait> traits = []> :
        Op<XLang_Dialect, mnemonic, traits>;

#endif // XLANG_DIALECT
