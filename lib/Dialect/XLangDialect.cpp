//===- XLangDialect.cpp - XLang dialect ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "XLang/XLangDialect.h"
#include "XLang/XLangOps.h"

using namespace mlir;
using namespace xlang;

#include "XLang/XLangOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// XLang dialect.
//===----------------------------------------------------------------------===//

void XLangDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "XLang/XLangOps.cpp.inc"
      >();
}
