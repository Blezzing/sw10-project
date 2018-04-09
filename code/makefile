cc = g++
execname = first
ldflags = -L/usr/local/lib#output of `llvm-config --ldflags`
cudalibs = -lcuda 
syslibs = -lrt -ldl -lpthread -lm#output of `llvm-config --system-libs`
llvmlibs = -lLLVMNVPTXCodeGen -lLLVMNVPTXDesc -lLLVMNVPTXInfo -lLLVMNVPTXAsmPrinter -lLLVMFuzzMutate -lLLVMSymbolize -lLLVMDebugInfoPDB -lLLVMDebugInfoDWARF -lLLVMDlltoolDriver -lLLVMCoroutines -lLLVMLibDriver -lLLVMTableGen -lLLVMGlobalISel  -lLLVMSelectionDAG -lLLVMAsmPrinter -lLLVMDebugInfoCodeView -lLLVMDebugInfoMSF -lLLVMLTO -lLLVMPasses -lLLVMObjCARCOpts -lLLVMipo -lLLVMInstrumentation -lLLVMVectorize -lLLVMLinker -lLLVMIRReader -lLLVMAsmParser -lLLVMCodeGen -lLLVMTarget -lLLVMScalarOpts -lLLVMInstCombine -lLLVMBitWriter -lLLVMAggressiveInstCombine -lLLVMTransformUtils -lLLVMAnalysis -lLLVMProfileData -lLLVMObject -lLLVMMCParser -lLLVMMC -lLLVMCore -lLLVMBinaryFormat -lLLVMSupport -lLLVMDemangle#subset of output of `llvm-config --libs`
includes = $(cudalibs) $(llvmlibs) $(syslibs)
cxxflags = -fno-rtti

all: exec strip
	
exec:
	$(cc) main.c $(cxxflags) $(ldflags) $(includes) -o $(execname)

strip:
	strip $(execname)

run:
	./$(execname)