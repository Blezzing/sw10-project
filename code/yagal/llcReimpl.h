#pragma once

#include <string>
#include <iostream>

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Pass.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
//#include "llvm/IR/DebugInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
//#include "llvm/CodeGen/CommandFlags.def"


namespace yagal::generator::llc{
    namespace {
        static printer::Printer _p("llc", printer::Printer::Mode::Silent);
    }

    std::string compileModule(llvm::LLVMContext& context, llvm::Module& module){
        std::string arch("nvptx64");
        llvm::Triple triple(llvm::Twine("nvptx64-nvidia-cuda"));
        std::string error;
        const llvm::Target *target(llvm::TargetRegistry::lookupTarget(arch, triple, error));
        std::string cpuStr("sm_20");
        std::string featureStr("");
        llvm::CodeGenOpt::Level optLevel(llvm::CodeGenOpt::Default);

        llvm::TargetOptions options;
        options.DisableIntegratedAS = false;
        options.MCOptions.ShowMCEncoding = false;
        options.MCOptions.MCUseDwarfDirectory = false;
        options.MCOptions.AsmVerbose = false;
        options.MCOptions.PreserveAsmComments = true;
        options.MCOptions.IASSearchPaths = std::vector<std::string>();
        options.MCOptions.SplitDwarfFile = false;

        std::unique_ptr<llvm::TargetMachine> targetMachine(target->createTargetMachine(
            triple.getTriple(), 
            cpuStr, 
            featureStr, 
            options, 
            llvm::None, 
            llvm::CodeModel::Small, 
            optLevel));

        llvm::legacy::PassManager passManager;

        //std::unique_ptr<llvm::Module> module;

        assert(targetMachine && "targetmachine is null");
        //assert(module && "module is null");

        llvm::TargetLibraryInfoImpl tlii(llvm::Triple(module.getTargetTriple()));
        //already done iirc
        //module->setDataLayout(targetMachine->createDataLayout());

        passManager.add(new llvm::TargetLibraryInfoWrapperPass(tlii));

        //huh?
        //llvm::UpgradeDebugInfo(module);

        //setFunctionAttributes(cpuStr, featureStr, module);

        llvm::SmallVector<char, 0> buffer;
        auto bufferStream = std::make_unique<llvm::raw_svector_ostream>(buffer);
        auto outputStream = bufferStream.get();
        //llvm::raw_svector_ostream outputStream(buffer);

        llvm::LLVMTargetMachine &llvmtm = static_cast<llvm::LLVMTargetMachine&>(*targetMachine);
        llvm::MachineModuleInfo *mmi = new llvm::MachineModuleInfo(&llvmtm);


        //denne linje virker bagvendt
        module.setDataLayout(targetMachine->createDataLayout());
        targetMachine->addPassesToEmitFile(passManager, *outputStream, llvm::TargetMachine::CGFT_AssemblyFile, false, mmi);

        //do it
        passManager.run(module);

        return std::string(buffer.begin(), buffer.end());
    }

    std::string translate(llvm::LLVMContext& context, llvm::Module& module){
        //dangerous?
        //llvm::llvm_shutdown_obj scopedShutdown;


        LLVMInitializeNVPTXTargetInfo();
        LLVMInitializeNVPTXTarget();
        LLVMInitializeNVPTXTargetMC();
        LLVMInitializeNVPTXAsmPrinter();

        llvm::PassRegistry *registry = llvm::PassRegistry::getPassRegistry();
        llvm::initializeCore(*registry);
        llvm::initializeCodeGen(*registry);
        llvm::initializeLoopStrengthReducePass(*registry);
        llvm::initializeLowerIntrinsicsPass(*registry);
        llvm::initializeEntryExitInstrumenterPass(*registry);
        llvm::initializePostInlineEntryExitInstrumenterPass(*registry);
        llvm::initializeUnreachableBlockElimLegacyPassPass(*registry);
        llvm::initializeConstantHoistingLegacyPassPass(*registry);
        llvm::initializeScalarOpts(*registry);
        llvm::initializeVectorization(*registry);
        llvm::initializeScalarizeMaskedMemIntrinPass(*registry);
        llvm::initializeExpandReductionsPass(*registry);

        context.setDiscardValueNames(false);

        return compileModule(context, module);
    }
}