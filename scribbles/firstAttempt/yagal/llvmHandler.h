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
#include "types.h"

namespace yagal::generator{
    std::string moduleToString(llvm::Module& module){
        std::string ret;
        llvm::raw_string_ostream stream(ret);
        stream << module;
        stream.flush();
        return ret;
    }

    std::string generatePTX(llvm::LLVMContext& context, llvm::Module& module){
        return yagal::generator::llc::translate(context, module);
    }

    void generateIR(llvm::LLVMContext& context, llvm::Module& module){
        //Set information about architecture
        module.setDataLayout("e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64");
        module.setTargetTriple("nvptx64-nvidia-cuda");

        //intrinsics
        auto *get_thread_idx = llvm::Intrinsic::getDeclaration(&module, llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x);

        //kernel
        std::vector<llvm::Type *> kernel_arg_types{
            llvm::Type::getFloatPtrTy(context, 1)
        };
        auto *kernel = llvm::Function::Create(
            llvm::FunctionType::get(llvm::Type::getVoidTy(context), kernel_arg_types, false),
            llvm::Function::ExternalLinkage,
            llvm::Twine("kernel"),
            &module
        );
        kernel->setCallingConv(llvm::CallingConv::PTX_Kernel);

        auto entry_block = llvm::BasicBlock::Create(context, "entry", kernel);
        llvm::IRBuilder<> builder(entry_block);
        auto arg_iter = kernel->arg_begin();
        auto vec_val = arg_iter++;
        vec_val->setName("vec");

        auto idx_val = builder.CreateCall(get_thread_idx, llvm::None, llvm::Twine("idx"));
        auto ptr_val = builder.CreateGEP(vec_val, idx_val, "ptr");
        auto tmp_val = builder.CreateAlignedLoad(ptr_val, 4, "tmp");
        auto ret_val = builder.CreateFAdd(tmp_val, tmp_val, "ret");
        builder.CreateAlignedStore(ret_val, ptr_val, 4);
        builder.CreateRetVoid();

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

    std::string buildPTX(){
        llvm::LLVMContext context;
        llvm::Module module("testModule", context);

        generateIR(context, module);
        auto ptx = generatePTX(context, module);
        return ptx;
    }
}