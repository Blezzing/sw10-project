#pragma once

#include <iostream>
#include <string>
#include <sstream>
#include <memory>
#include <tuple>

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/Support/raw_ostream.h"

#include "llcReimpl.h"
#include "printer.hpp"

namespace yagal::generator{
    namespace {
        printer::Printer _p("llvmHandler", printer::Printer::Mode::Verbose);
    }

    class IRModule{
        int basicBlockNumber = 0;
    public:
        llvm::LLVMContext context;
        llvm::Module module;

        std::vector<llvm::Function*> kernels;
        llvm::Function* getThreadIdxIntrinsic;


        IRModule(): 
            module("kernelModule", context),
            getThreadIdxIntrinsic(llvm::Intrinsic::getDeclaration(&module, llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x))
        {
            module.setDataLayout("e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64");
            module.setTargetTriple("nvptx64-nvidia-cuda");
        }

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

        std::string getNextBasicBlockName(){
            std::string str = "bb" + std::to_string(basicBlockNumber++);
            return str;
        }

        std::string toString(){
            std::string ret;
            llvm::raw_string_ostream stream(ret);
            stream << module;
            stream.flush();
            return ret;
        }
    };
}
