#pragma once

#include "llvm/IR/IRBuilder.h"

#include "printer.hpp"

namespace yagal{
    template <typename T>
    class Vector;
}

namespace yagal::internal{
namespace{
    printer::Printer _p("action", printer::Printer::Mode::Debug);
}

class Action{
public:
    virtual void generateIR(yagal::generator::IRModule& ir, llvm::Function* kernel, int& inputVectorCounter){
        _p.debug() << "generateIR for base action called, makes no sense." << std::endl;
    }
    virtual bool requiresCudaParameter(){
        return false;
    }
    virtual ~Action() = default;
};


template <typename T>
class ParameterAction : public Action{
    Vector<T>& value;

public:
    ParameterAction(Vector<T>& v) : value(v) {}

    virtual bool requiresCudaParameter(){
        return true;
    }

    virtual CUdeviceptr* getDevicePtrPtr(){
        value.getDevicePtrPtr();
    }
};

template <typename T>
class AddAction : public Action{
public:
    T value;

    AddAction(T v): value(v) {}

    void generateIR(yagal::generator::IRModule& ir, llvm::Function* kernel, int& inputVectorCounter){
        _p.debug() << "generateIR for add action called." << std::endl;

        auto entry_block = llvm::BasicBlock::Create(ir.context, llvm::Twine(ir.getNextBasicBlockName()), kernel);
        llvm::IRBuilder<> builder(entry_block);
        auto arg_iter = kernel->arg_begin();
        auto vec_val = arg_iter++;
        vec_val->setName("vec");

        auto indexVar = builder.CreateAlignedLoad(ir.currentIndexValue, 4, "i");
        auto ptr_val = builder.CreateGEP(vec_val, indexVar, "ptr");
        auto tmp_val = builder.CreateAlignedLoad(ptr_val, 4, "tmp");
        auto input_const = llvm::ConstantFP::get(llvm::Type::getFloatTy(ir.context), (double)value);
        auto ret_val = builder.CreateFAdd(tmp_val, input_const, "ret");
        builder.CreateAlignedStore(ret_val, ptr_val, 4);

        ir.userBlocks.push_back(entry_block);
    }
};
template<>
class AddAction<int> : public Action {
public:
    int value;

    AddAction(int v): value(v) {}

    void generateIR(yagal::generator::IRModule& ir, llvm::Function* kernel, int& inputVectorCounter){
        _p.debug() << "generateIR for add action<int> called." << std::endl;

        auto entry_block = llvm::BasicBlock::Create(ir.context, llvm::Twine(ir.getNextBasicBlockName()), kernel);
        llvm::IRBuilder<> builder(entry_block);
        auto arg_iter = kernel->arg_begin();
        auto vec_val = arg_iter++; 
        vec_val->setName("vec");

        auto idx_val = builder.CreateCall(ir.getThreadIdxIntrinsic, llvm::None, llvm::Twine("idx"));
        auto ptr_val = builder.CreateGEP(vec_val, idx_val, "ptr");
        auto tmp_val = builder.CreateLoad(ptr_val, "tmp");
        auto input_const = llvm::ConstantInt::get(tmp_val->getType(), value);
        
        _p.debug() << "HER " << input_const->getType() << ", " << tmp_val->getType() << std::endl;
        
        auto ret_val = builder.CreateAdd(tmp_val, input_const, "ret");
        builder.CreateAlignedStore(ret_val, ptr_val, 4);
    }
};

template <typename T>
class MultAction : public Action{
public:
    T value;

    MultAction(T v): value(v) {}

    void generateIR(yagal::generator::IRModule& ir, llvm::Function* kernel, int& inputVectorCounter){
        _p.debug() << "generateIR for add action called." << std::endl;

        auto entry_block = llvm::BasicBlock::Create(ir.context, llvm::Twine(ir.getNextBasicBlockName()), kernel);
        llvm::IRBuilder<> builder(entry_block);
        auto arg_iter = kernel->arg_begin();
        auto vec_val = arg_iter++;
        vec_val->setName("vec");

        auto indexVar = builder.CreateAlignedLoad(ir.currentIndexValue, 4, "i");
        auto ptr_val = builder.CreateGEP(vec_val, indexVar, "ptr");
        auto tmp_val = builder.CreateAlignedLoad(ptr_val, 4, "tmp");
        auto input_const = llvm::ConstantFP::get(llvm::Type::getFloatTy(ir.context), (double)value);
        auto ret_val = builder.CreateFMul(tmp_val, input_const, "ret");
        builder.CreateAlignedStore(ret_val, ptr_val, 4);

        ir.userBlocks.push_back(entry_block);
    }
};

template <typename T>
class AddVectorAction : public ParameterAction<T>{
public:
    AddVectorAction(Vector<T>& v): ParameterAction<T>(v) {}

    void generateIR(yagal::generator::IRModule& ir, llvm::Function* kernel, int& inputVectorCounter){
        _p.debug() << "generateIR for add vector action called." << std::endl;

        inputVectorCounter++;

        auto entry_block = llvm::BasicBlock::Create(ir.context, llvm::Twine(ir.getNextBasicBlockName()), kernel);
        llvm::IRBuilder<> builder(entry_block);

        auto argIter = kernel->arg_begin();
        auto vec1Val = argIter;
        for(int i = 0; i < inputVectorCounter; i++){
            argIter++;
        }
        auto vec2Val = argIter;

        vec1Val->setName("vec1");
        vec2Val->setName("vec2");

        auto indexVar = builder.CreateAlignedLoad(ir.currentIndexValue, 4, "i");
        auto ptr1Val = builder.CreateGEP(vec1Val, indexVar, "ptr1");
        auto ptr2Val = builder.CreateGEP(vec2Val, indexVar, "ptr2");
        auto tmp1Val = builder.CreateAlignedLoad(ptr1Val, 4, "tmp1");
        auto tmp2Val = builder.CreateAlignedLoad(ptr2Val, 4, "tmp2");
        auto retVal = builder.CreateFAdd(tmp1Val, tmp2Val, "ret");
        builder.CreateAlignedStore(retVal, ptr1Val, 4);

        ir.userBlocks.push_back(entry_block);
    }
};

}
