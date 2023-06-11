set(LLVM_LINK_COMPONENTS
  Support
  Core
  Option
  native
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_llvm_executable(calc-codegen
  main.cpp
  NativeCodeGen.cpp
  )
llvm_update_compile_flags(calc-codegen)
target_link_libraries(calc-codegen
  PRIVATE
  ${dialect_libs}
  ${translation_libs}
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTranslateLib
  MLIRSupport
  MLIRFuncToLLVM
  MLIRArithToLLVM
  )

mlir_check_link_libraries(calc-codegen)

clang_target_link_libraries(calc-codegen
  PRIVATE
  clangBasic
  clangDriver
  clangFrontend
)
