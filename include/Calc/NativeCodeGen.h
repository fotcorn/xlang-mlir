#ifndef CALC_NATIVECODEGEN_H
#define CALC_NATIVECODEGEN_H

#include <memory>
#include <string>

#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

namespace mlir {
class ModuleOp;
class MLIRContext;
} // namespace mlir

namespace calc {
llvm::Error generateNativeBinary(mlir::ModuleOp &Module,
                                 llvm::StringRef FilePath);
} // namespace calc
#endif // CALC_NATIVECODEGEN_H
