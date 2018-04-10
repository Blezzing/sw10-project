#pragma once

#include "llvm/IR/IRBuilder.h"

#include "printer.hpp"

namespace yagal::internal{
namespace{
    printer::Printer _p("action", printer::Printer::Mode::Debug);
}

class Action{
public:
    virtual void generateIR(yagal::generator::IRModule& ir, llvm::Function* kernel){
        _p.debug() << "generateIR for base action called, makes no sense." << std::endl;
    }
};

template <typename T>
class AddAction : public Action{
public:
    T value;

    AddAction(T v): value(v) {}

    void generateIR(yagal::generator::IRModule& ir, llvm::Function* kernel){
        _p.debug() << "generateIR for add action called." << std::endl;

        auto entry_block = llvm::BasicBlock::Create(ir.context, llvm::Twine(ir.getNextBasicBlockName()), kernel);
        llvm::IRBuilder<> builder(entry_block);
        auto arg_iter = kernel->arg_begin();
        auto vec_val = arg_iter++;
        vec_val->setName("vec");

        auto idx_val = builder.CreateCall(ir.getThreadIdxIntrinsic, llvm::None, llvm::Twine("idx"));
        auto ptr_val = builder.CreateGEP(vec_val, idx_val, "ptr");
        auto tmp_val = builder.CreateAlignedLoad(ptr_val, 4, "tmp");
        auto input_const = llvm::ConstantFP::get(llvm::Type::getFloatTy(ir.context), (double)value);
        auto ret_val = builder.CreateFAdd(tmp_val, input_const, "ret");
        builder.CreateAlignedStore(ret_val, ptr_val, 4);
    }
};

template <typename T>
class AddToSelfAction : public Action{
public:
    T value;

    AddToSelfAction(T v): value(v) {}

    void generateIR(yagal::generator::IRModule& ir, llvm::Function* kernel){
        _p.debug() << "generateIR for add action called." << std::endl;

        auto entry_block = llvm::BasicBlock::Create(ir.context, llvm::Twine(ir.getNextBasicBlockName()), kernel);
        llvm::IRBuilder<> builder(entry_block);
        auto arg_iter = kernel->arg_begin();
        auto vec_val = arg_iter++;
        vec_val->setName("vec");

        auto idx_val = builder.CreateCall(ir.getThreadIdxIntrinsic, llvm::None, llvm::Twine("idx"));
        auto ptr_val = builder.CreateGEP(vec_val, idx_val, "ptr");
        auto tmp_val = builder.CreateAlignedLoad(ptr_val, 4, "tmp");
        auto ret_val = builder.CreateFAdd(tmp_val, tmp_val, "ret");
        builder.CreateAlignedStore(ret_val, ptr_val, 4);
    }
};
}