add_mlir_dialect_library(MLIRXLang
        XLangDialect.cpp
        XLangOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/XLang

        DEPENDS
        MLIRXLangOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
        MLIRInferTypeOpInterface
        MLIRLLVMCommonConversion
        MLIRLLVMToLLVMIRTranslation
	)
