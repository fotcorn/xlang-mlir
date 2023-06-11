//===- CalcDialect.cpp - Calc dialect ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Calc/CalcDialect.h"
#include "Calc/CalcOps.h"

using namespace mlir;
using namespace calc;

#include "Calc/CalcOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Calc dialect.
//===----------------------------------------------------------------------===//

void CalcDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Calc/CalcOps.cpp.inc"
      >();
}
