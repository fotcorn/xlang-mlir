add_mlir_dialect_library(MLIRCalc
        CalcDialect.cpp
        CalcOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Calc

        DEPENDS
        MLIRCalcOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
        MLIRInferTypeOpInterface
        MLIRLLVMCommonConversion
        MLIRLLVMToLLVMIRTranslation
	)
