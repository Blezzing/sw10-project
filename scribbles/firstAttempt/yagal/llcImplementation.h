//===-- llc.cpp - Implement the LLVM Native Code Generator ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the llc code generator driver. It provides a convenient
// command-line interface for generating native assembly-language code
// or C code, given LLVM bitcode.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/CodeGen/CommandFlags.def"
#include "llvm/CodeGen/LinkAllAsmWriterComponents.h"
#include "llvm/CodeGen/LinkAllCodegenComponents.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/AutoUpgrade.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Pass.h"
//#include "llvm/Support/CommandLine.h"
//#include "llvm/Support/Debug.h"
//#include "llvm/Support/FileSystem.h"
//#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PluginLoader.h"
//#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <memory> 
using namespace llvm;

// General options for llc.  Other pass-specific options are specified
// within the corresponding llc passes, and target-specific options
// and back-end code generation options are specified with the target machine.
//
std::string InputLanguage("ir");
std::string OutputFilename("-");
bool NoIntegratedAssembler = false;
bool PreserveComments = true;
char OptLevel = '2';
bool SplitDwarfFile = false;
bool NoVerify = false;
bool DisableSimplifyLibCalls = false;
bool ShowMCEncoding = false;
bool EnableDwarfDirectory = false;
bool AsmVerbose = false;
bool CompileTwice = false;
std::vector<std::string> IncludeDirs;

namespace {
static ManagedStatic<std::vector<std::string>> RunPassNames;

struct RunPassOption {
  void operator=(const std::string &Val) const {
    if (Val.empty())
      return;
    SmallVector<StringRef, 8> PassNames;
    StringRef(Val).split(PassNames, ',', -1, false);
    for (auto PassName : PassNames)
      RunPassNames->push_back(PassName);
  }
};
}

static RunPassOption RunPassOpt;

static int compileModule(char **, LLVMContext &);

struct LLCDiagnosticHandler : public DiagnosticHandler {
  bool *HasError;
  LLCDiagnosticHandler(bool *HasErrorPtr) : HasError(HasErrorPtr) {}
  bool handleDiagnostics(const DiagnosticInfo &DI) override {
    if (DI.getSeverity() == DS_Error)
      *HasError = true;

    if (auto *Remark = dyn_cast<DiagnosticInfoOptimizationBase>(&DI))
      if (!Remark->isEnabled())
        return true;

    DiagnosticPrinterRawOStream DP(errs());
    errs() << LLVMContext::getDiagnosticMessagePrefix(DI.getSeverity()) << ": ";
    DI.print(DP);
    errs() << "\n";
    return true;
  }
};

static void InlineAsmDiagHandler(const SMDiagnostic &SMD, void *Context,
                                 unsigned LocCookie) {
  bool *HasError = static_cast<bool *>(Context);
  if (SMD.getKind() == SourceMgr::DK_Error)
    *HasError = true;

  SMD.print(nullptr, errs());

  // For testing purposes, we print the LocCookie here.
  if (LocCookie)
    errs() << "note: !srcloc = " << LocCookie << "\n";
}

// main - Entry point for the llc compiler.
//
int llcMain(int argc, char **argv) {
  //sys::PrintStackTraceOnErrorSignal(argv[0]);
  //PrettyStackTraceProgram X(argc, argv);

  LLVMContext Context;
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  // Initialize targets first, so that --version shows registered targets.
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();

  // Initialize codegen and IR passes used by llc so that the -print-after,
  // -print-before, and -stop-after options work.
  PassRegistry *Registry = PassRegistry::getPassRegistry();
  initializeCore(*Registry);
  initializeCodeGen(*Registry);
  initializeLoopStrengthReducePass(*Registry);
  initializeLowerIntrinsicsPass(*Registry);
  initializeEntryExitInstrumenterPass(*Registry);
  initializePostInlineEntryExitInstrumenterPass(*Registry);
  initializeUnreachableBlockElimLegacyPassPass(*Registry);
  initializeConstantHoistingLegacyPassPass(*Registry);
  initializeScalarOpts(*Registry);
  initializeVectorization(*Registry);
  initializeScalarizeMaskedMemIntrinPass(*Registry);
  initializeExpandReductionsPass(*Registry);

  // Initialize debugging passes.
  initializeScavengerTestPass(*Registry);

  Context.setDiscardValueNames(false);

  // Set a diagnostic handler that doesn't exit on the first error
  bool HasError = false;
  Context.setDiagnosticHandler(
      llvm::make_unique<LLCDiagnosticHandler>(&HasError));
  Context.setInlineAsmDiagnosticHandler(InlineAsmDiagHandler, &HasError);
  
  int RetVal = compileModule(argv, Context);
}

static bool addPass(PassManagerBase &PM, const char *argv0,
                    StringRef PassName, TargetPassConfig &TPC) {
  if (PassName == "none")
    return false;

  const PassRegistry *PR = PassRegistry::getPassRegistry();
  const PassInfo *PI = PR->getPassInfo(PassName);
  if (!PI) {
    errs() << argv0 << ": run-pass " << PassName << " is not registered.\n";
    return true;
  }

  Pass *P;
  if (PI->getNormalCtor())
    P = PI->getNormalCtor()();
  else {
    errs() << argv0 << ": cannot create pass: " << PI->getPassName() << "\n";
    return true;
  }
  std::string Banner = std::string("After ") + std::string(P->getPassName());
  PM.add(P);
  TPC.printAndVerify(Banner);

  return false;
}

static int compileModule(char **argv, LLVMContext &Context) {
  // Load the module to be compiled...
  SMDiagnostic Err;
  std::unique_ptr<Module> M;
  std::unique_ptr<MIRParser> MIR;
  Triple TheTriple;

  TheTriple.setTriple(llvm::Twine("nvptx64-nvidia-cuda"));


  // Get the target specific parser.
  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget(MArch, TheTriple,
                                                         Error);
  if (!TheTarget) {
    errs() << argv[0] << ": " << Error;
    return 1;
  }

  std::string CPUStr = getCPUStr(), FeaturesStr = getFeaturesStr();

  CodeGenOpt::Level OLvl = CodeGenOpt::Default;
  switch (OptLevel) {
  default:
    errs() << argv[0] << ": invalid optimization level.\n";
    return 1;
  case ' ': break;
  case '0': OLvl = CodeGenOpt::None; break;
  case '1': OLvl = CodeGenOpt::Less; break;
  case '2': OLvl = CodeGenOpt::Default; break;
  case '3': OLvl = CodeGenOpt::Aggressive; break;
  }

  TargetOptions Options = InitTargetOptionsFromCodeGenFlags();
  Options.DisableIntegratedAS = NoIntegratedAssembler;
  Options.MCOptions.ShowMCEncoding = ShowMCEncoding;
  Options.MCOptions.MCUseDwarfDirectory = EnableDwarfDirectory;
  Options.MCOptions.AsmVerbose = AsmVerbose;
  Options.MCOptions.PreserveAsmComments = PreserveComments;
  Options.MCOptions.IASSearchPaths = IncludeDirs;
  Options.MCOptions.SplitDwarfFile = SplitDwarfFile;

  std::unique_ptr<TargetMachine> Target(TheTarget->createTargetMachine(
      TheTriple.getTriple(), CPUStr, FeaturesStr, Options, getRelocModel(),
      getCodeModel(), OLvl));

  assert(Target && "Could not allocate target machine!");
  assert(M && "Should have exited if we didn't have a module!");
  if (FloatABIForCalls != FloatABI::Default)
    Options.FloatABIType = FloatABIForCalls;

  // Figure out where we are going to send the output.

  // Build up all of the passes that we want to do to the module.
  legacy::PassManager PM;

  // Add an appropriate TargetLibraryInfo pass for the module's triple.
  TargetLibraryInfoImpl TLII(Triple(M->getTargetTriple()));

  // The -disable-simplify-libcalls flag actually disables all builtin optzns.
  if (DisableSimplifyLibCalls)
    TLII.disableAllFunctions();
  PM.add(new TargetLibraryInfoWrapperPass(TLII));

  // Add the target data from the target machine, if it exists, or the module.
  M->setDataLayout(Target->createDataLayout());

  // This needs to be done after setting datalayout since it calls verifier
  // to check debug info whereas verifier relies on correct datalayout.
  UpgradeDebugInfo(*M);

  // Verify module immediately to catch problems before doInitialization() is
  // called on any passes.
  if (!NoVerify && verifyModule(*M, &errs())) {
    errs() << argv[0] << ": "
           << ": error: input module is broken!\n";
    return 1;
  }

  // Override function attributes based on CPUStr, FeaturesStr, and command line
  // flags.
  setFunctionAttributes(CPUStr, FeaturesStr, *M);

  if (RelaxAll.getNumOccurrences() > 0 &&
      FileType != TargetMachine::CGFT_ObjectFile)
    errs() << argv[0]
             << ": warning: ignoring -mc-relax-all because filetype != obj";

  {
    //raw_pwrite_stream *OS() = &Out->os();

    // Manually do the buffering rather than using buffer_ostream,
    // so we can memcmp the contents in CompileTwice mode
    SmallVector<char, 0> Buffer;
    std::unique_ptr<raw_svector_ostream> BOS;

    const char *argv0 = argv[0];
    LLVMTargetMachine &LLVMTM = static_cast<LLVMTargetMachine&>(*Target);
    MachineModuleInfo *MMI = new MachineModuleInfo(&LLVMTM);

    // Construct a custom pass pipeline that starts after instruction
    // selection.
    if (!RunPassNames->empty()) {
      if (!MIR) {
        errs() << argv0 << ": run-pass is for .mir file only.\n";
        return 1;
      }
      TargetPassConfig &TPC = *LLVMTM.createPassConfig(PM);
      if (TPC.hasLimitedCodeGenPipeline()) {
        errs() << argv0 << ": run-pass cannot be used with "
               << TPC.getLimitedCodeGenPipelineReason(" and ") << ".\n";
        return 1;
      }

      TPC.setDisableVerify(NoVerify);
      PM.add(&TPC);
      PM.add(MMI);
      TPC.printAndVerify("");
      for (const std::string &RunPassName : *RunPassNames) {
        if (addPass(PM, argv0, RunPassName, TPC))
          return 1;
      }
      TPC.setInitialized();/*
      PM.add(createPrintMIRPass(*OS));
      PM.add(createFreeMachineFunctionPass());
    } else if (Target->addPassesToEmitFile(PM, *OS, FileType, NoVerify, MMI)) {
      errs() << argv0 << ": target does not support generation of this"
             << " file type!\n";
      return 1;*/
    }

    if (MIR) {
      assert(MMI && "Forgot to create MMI?");
      if (MIR->parseMachineFunctions(*M, *MMI))
        return 1;
    }

    PM.run(*M);

    auto HasError = ((const LLCDiagnosticHandler *)(Context.getDiagHandlerPtr()))->HasError;
    if (*HasError)
      return 1;

    if (BOS) {
      //Out->os() << Buffer;
    }
  }

  return 0;
}