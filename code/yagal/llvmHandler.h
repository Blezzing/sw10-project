#pragma once

#include <iostream>
#include <string>
#include <sstream>
#include <iostream>
#include <memory>
#include <tuple>
#include <iostream>

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/BasicBlock.h"
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
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineModuleInfo.h"

#include "printer.hpp"

namespace yagal::generator{
    namespace {
        printer::Printer _p("llvmHandler", printer::Printer::Mode::Verbose);
    }

    //Representation of a module in llvm ir.
    //Contains the logic to configure a llvm module and provide functions for the rest of the library to add functionality to a module
    //REQUIRES the use of createKernel() -> add functionality to the kernel -> finalizeKernel(kernel) work flow.
    class IRModule{
    public:
        llvm::LLVMContext context;
        llvm::Module module;

        std::vector<llvm::Function*> kernels;
        llvm::Function* getThreadIdxIntrinsic;

        IRModule(): 
            module("yagalModule", context),
            getThreadIdxIntrinsic(llvm::Intrinsic::getDeclaration(&module, llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x))
        {
            //Set platform specific variables for the module.
            module.setDataLayout("e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64");
            module.setTargetTriple("nvptx64-nvidia-cuda");

            _p.debug() << "ir module constructed" << std::endl;
        }

        //Create a function ready for insertion with a IRBuilder.
        //TODO: Make this able to construct kernels with different numbers of parameters.
        //TODO: Make this use the function vector to allow easier management of kernel.
        llvm::Function* createKernel(){
            std::vector<llvm::Type *> kernel_arg_types{
                llvm::Type::getFloatPtrTy(context, 1)
            };
            auto kernel = llvm::Function::Create(
                llvm::FunctionType::get(llvm::Type::getVoidTy(context), kernel_arg_types, false),
                llvm::Function::ExternalLinkage,
                llvm::Twine("kernel"),
                &module
            );
            kernel->setCallingConv(llvm::CallingConv::PTX_Kernel);

            return kernel;
        }

        //Creates the return point of a kernel, and links blocks together, to effectively make them labels.
        void finalizeKernel(llvm::Function* kernel){
            auto exit_block = llvm::BasicBlock::Create(context, "exit", kernel);
            llvm::IRBuilder<> builder(exit_block);
            builder.CreateRetVoid();

            auto thisBB = kernel->begin();
            auto nextBB = kernel->begin(); nextBB++;
            
            while(nextBB != kernel->end()){
                builder.SetInsertPoint(&(*thisBB));
                builder.CreateBr(&(*nextBB));
                thisBB++; nextBB++;
            }
        }

        //Update metadata of module to correctly tag the kernel functions.
        //TODO: Make this consider any kernel, not just the hardcoded one.
        void updateMetadata(){
            auto metadata = module.getOrInsertNamedMetadata("nvvm.annotations");
            auto oneconstant = llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), 1);
            std::vector<llvm::Metadata *> ops{
                llvm::ValueAsMetadata::getConstant(module.getNamedValue("kernel")),
                llvm::MDString::get(context, "kernel"),
                llvm::ValueAsMetadata::getConstant(oneconstant)
            };
            auto metadata_node = llvm::MDTuple::get(context, ops); 
            metadata->addOperand(metadata_node);
        }

        //Returns a new free name for llvm basicblocks to avoid name clashes
        std::string getNextBasicBlockName(){
            static int basicBlockNumber = 0;
            std::string str = "bb" + std::to_string(basicBlockNumber++);
            return str;
        }

        //Return a string with the llvm ir as human readable text. For debugging.
        std::string toString(){
            std::string ret;
            llvm::raw_string_ostream stream(ret);
            stream << module;
            stream.flush();
            return ret;
        }
    };

    //Representation of a module in ptx.
    //Contains the logic to translate ir to ptx, and is a nifty workaround not using llc.
    class PTXModule{
    private:
        std::string _string;

        //Use C macros to enable target info for llvm
        static void initializeLlvmTargetIfNeeded(){
            //Only allow this once
            static bool isLlvmTargetInitialized = false;
            if(isLlvmTargetInitialized){
                return;
            }

            //Make nvptx64 a valid target
            LLVMInitializeNVPTXTargetInfo();
            LLVMInitializeNVPTXTarget();
            LLVMInitializeNVPTXTargetMC();
            LLVMInitializeNVPTXAsmPrinter();

            isLlvmTargetInitialized = true;
            _p.debug() << "initialized llvm target" << std::endl;
        }

        //Configure global pass registry object for use in the llvm pass manager
        static void initializeLlvmPassRegistryIfNeeded(){
            //Only allow this once
            static bool isLlvmPassRegistryInitialized = false;
            if(isLlvmPassRegistryInitialized){
                return;
            }

            //Get pass registry
            auto registry = llvm::PassRegistry::getPassRegistry();

            //Core passes
            llvm::initializeCore(*registry);
            llvm::initializeCodeGen(*registry);

            //Default passes
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
            
            //Experimental passes
            llvm::initializeInstructionCombiningPassPass(*registry);
            llvm::initializeDeadMachineInstructionElimPass(*registry);

            isLlvmPassRegistryInitialized = true;
            _p.debug() << "initialized llvm pass registry" << std::endl;
        }

    public:
        //Constructor from ir, go through the compilation process
        PTXModule(IRModule& ir){
            //Load target information to be able to generate a ptx module
            initializeLlvmTargetIfNeeded();

            //Register passes to perform on ir
            initializeLlvmPassRegistryIfNeeded();

            //Set variables needed to create a target machine
            std::string arch("nvptx64");
            llvm::Triple triple(llvm::Twine("nvptx64-nvidia-cuda"));
            std::string error;
            const llvm::Target *target(llvm::TargetRegistry::lookupTarget(arch, triple, error));
            std::string cpuStr("sm_20");
            std::string featureStr("");
            llvm::CodeGenOpt::Level optLevel(llvm::CodeGenOpt::Aggressive);
            llvm::TargetOptions options;

            //Create target machine
            std::unique_ptr<llvm::TargetMachine> targetMachine(target->createTargetMachine(
                triple.getTriple(), 
                cpuStr, 
                featureStr, 
                options, 
                llvm::None, 
                llvm::CodeModel::Small, 
                optLevel));

            //Create PassManager
            llvm::legacy::PassManager passManager;

            //Create output buffer, and a stream to write to
            llvm::SmallVector<char, 512> buffer;
            auto bufferStream = std::make_unique<llvm::raw_svector_ostream>(buffer);
            auto outputStream = bufferStream.get();

            //Get MachineModuleInfo
            llvm::LLVMTargetMachine &llvmtm = static_cast<llvm::LLVMTargetMachine&>(*targetMachine);
            llvm::MachineModuleInfo *mmi = new llvm::MachineModuleInfo(&llvmtm);

            //Reset data layout of module
            ir.module.setDataLayout(targetMachine->createDataLayout());
            targetMachine->addPassesToEmitFile(passManager, *outputStream, llvm::TargetMachine::CGFT_AssemblyFile, false, mmi);

            //Run passes on module
            passManager.run(ir.module);

            //Write to string
            _string = std::string(buffer.begin(), buffer.end());

            _p.debug() << "ptx module constructed" << std::endl;
        }

        //Construct from file content
        PTXModule(const std::string& filename){
            try{
                std::ifstream file(filename);
                _string = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
            }
            catch(...){
                _p.error() << "error while constructing ptx module from file" << std::endl;
            }
        }

        //Return copy of string to allo cuda driver to load it
        std::string toString(){
            //Return copy of string
            return _string;
        }
    };
}
