#include <iostream>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/Error.h"

#include "Calc/NativeCodeGen.h"

int main() {
  mlir::MLIRContext context;

  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::arith::ArithDialect>();
  context.loadDialect<mlir::LLVM::LLVMDialect>();

  mlir::ModuleOp mod = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  mlir::OpBuilder builder(&context);

  // setup printf
  auto llvmI32Ty = mlir::IntegerType::get(&context, 32);
  auto llvmI8PtrTy =
      mlir::LLVM::LLVMPointerType::get(mlir::IntegerType::get(&context, 8));
  auto printfType = mlir::LLVM::LLVMFunctionType::get(llvmI32Ty, llvmI8PtrTy,
                                                      /*isVarArg=*/true);
  builder.setInsertionPointToStart(mod.getBody());
  builder.create<mlir::LLVM::LLVMFuncOp>(builder.getUnknownLoc(), "printf",
                                         printfType);

  // create function with no arguments, i32 return type
  mlir::FunctionType funcType = mlir::FunctionType::get(
      &context, {}, mlir::IntegerType::get(&context, 32));

  mlir::func::FuncOp func = builder.create<mlir::func::FuncOp>(
      builder.getUnknownLoc(), "main", funcType);

  mlir::Block *entryBlock = func.addEntryBlock();

  builder.setInsertionPointToStart(entryBlock);

  mlir::Value op1 = builder.create<mlir::arith::ConstantIntOp>(
      builder.getUnknownLoc(), 13, 32);
  mlir::Value op2 = builder.create<mlir::arith::ConstantIntOp>(
      builder.getUnknownLoc(), 29, 32);

  mlir::Value result =
      builder.create<mlir::arith::AddIOp>(builder.getUnknownLoc(), op1, op2);

  mlir::Value formatStr = mlir::LLVM::createGlobalString(
      builder.getUnknownLoc(), builder, "formatStr", "%i\n",
      mlir::LLVM::Linkage::Internal, false);
  builder.create<mlir::LLVM::CallOp>(
      builder.getUnknownLoc(), llvmI32Ty, "printf",
      llvm::ArrayRef<mlir::Value>({formatStr, result}));

  mlir::Value retVal = builder.create<mlir::arith::ConstantIntOp>(
      builder.getUnknownLoc(), 0, 32);
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), retVal);

  // Print the module.
  mod.dump();

  // Verify the function and module.
  if (mlir::failed(verify(func)) || mlir::failed(mlir::verify(mod))) {
    llvm::errs() << "Error: module verification failed\n";
    return 1;
  }

  llvm::Error err = calc::generateNativeBinary(mod, "main");
  if (err) {
    llvm::errs() << "Error generating native code: \n";
    llvm::errs() << llvm::toString(std::move(err)) << '\n';
    return 1;
  }
  return 0;
}
