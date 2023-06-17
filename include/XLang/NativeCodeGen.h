#ifndef XLANG_NATIVECODEGEN_H
#define XLANG_NATIVECODEGEN_H

#include <memory>
#include <string>

#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

namespace mlir {
class ModuleOp;
class MLIRContext;
} // namespace mlir

namespace xlang {
llvm::Error generateNativeBinary(mlir::ModuleOp &Module,
                                 llvm::StringRef FilePath);
} // namespace xlang
#endif // XLANG_NATIVECODEGEN_H
