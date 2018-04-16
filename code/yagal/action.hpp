#pragma once

#include "llvm/IR/IRBuilder.h"

#include "printer.hpp"

//Forward declaration of vector to avoid circular dependency
namespace yagal{
    template <typename T>
    class Vector;
}

namespace yagal::internal{
    namespace{
        printer::Printer _p("action", printer::Printer::Mode::Debug);
    }

    //Base class, provide base functionality
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

    //Intermediate class, to store value of simple action
    template <typename T>
    class SimpleAction : public Action{
    protected:
        T value;
    public:
        SimpleAction(T v) : value(v) {}
    };

    //Intermediate class, to allow caller to get a device ptr ptr
    template <typename T>
    class ParameterAction : public Action{
    protected:
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

    //Add a value to every value in the vector
    template <typename T>
    class AddAction : public SimpleAction<T>{
    public:
        AddAction(T v): SimpleAction<T>(v) {}

        void generateIR(yagal::generator::IRModule& ir, llvm::Function* kernel, int& inputVectorCounter){
            _p.debug() << "generateIR for add action called." << std::endl;

            //Prepare the block to fill in
            auto actionBlock = llvm::BasicBlock::Create(ir.context, llvm::Twine(ir.getNextBasicBlockName()), kernel);
            llvm::IRBuilder<> builder(actionBlock);
            ir.userBlocks.push_back(actionBlock);

            //Prepare argument
            auto vecVal = kernel->arg_begin();
            vecVal->setName("vec");

            //Build llvm ir
            int alignment = 4;
            auto indexVar = builder.CreateAlignedLoad(ir.currentIndexValue, alignment, "i");
            auto ptrVal = builder.CreateGEP(vecVal, indexVar, "ptr");
            auto tmpVal = builder.CreateAlignedLoad(ptrVal, alignment, "tmp");
            auto inputConst = llvm::ConstantFP::get(llvm::Type::getFloatTy(ir.context), (float)this->value);
            auto retVal = builder.CreateFAdd(tmpVal, inputConst, "ret");
            builder.CreateAlignedStore(retVal, ptrVal, alignment);
        }
    };
    
    //Subtact a value to every value in the vector
        template <typename T>
        class SubAction : public SimpleAction<T>{
        public:
            SubAction(T v): SimpleAction<T>(v) {}
    
            void generateIR(yagal::generator::IRModule& ir, llvm::Function* kernel, int& inputVectorCounter){
                _p.debug() << "generateIR for add action called." << std::endl;
    
                //Prepare the block to fill in
                auto actionBlock = llvm::BasicBlock::Create(ir.context, llvm::Twine(ir.getNextBasicBlockName()), kernel);
                llvm::IRBuilder<> builder(actionBlock);
                ir.userBlocks.push_back(actionBlock);
    
                //Prepare argument
                auto vecVal = kernel->arg_begin();
                vecVal->setName("vec");
    
                //Build llvm ir
                int alignment = 4;
                auto indexVar = builder.CreateAlignedLoad(ir.currentIndexValue, alignment, "i");
                auto ptrVal = builder.CreateGEP(vecVal, indexVar, "ptr");
                auto tmpVal = builder.CreateAlignedLoad(ptrVal, alignment, "tmp");
                auto inputConst = llvm::ConstantFP::get(llvm::Type::getFloatTy(ir.context), (float)this->value);
                auto retVal = builder.CreateFSub(tmpVal, inputConst, "ret");
                builder.CreateAlignedStore(retVal, ptrVal, alignment);
            }
        };

    //Multipy every value of the vector with a value
    template <typename T>
    class MultAction : public SimpleAction<T>{
    public:
        MultAction(T v): SimpleAction<T>(v) {}

        void generateIR(yagal::generator::IRModule& ir, llvm::Function* kernel, int& inputVectorCounter){
            _p.debug() << "generateIR for add action called." << std::endl;

            //Prepare the block to fill in
            auto actionBlock = llvm::BasicBlock::Create(ir.context, llvm::Twine(ir.getNextBasicBlockName()), kernel);
            llvm::IRBuilder<> builder(actionBlock);
            ir.userBlocks.push_back(actionBlock);

            //Prepare argument
            auto vecVal = kernel->arg_begin();
            vecVal->setName("vec");

            //Build llvm ir
            int alignment = 4;
            auto indexVar = builder.CreateAlignedLoad(ir.currentIndexValue, 4, "i");
            auto ptrVal = builder.CreateGEP(vecVal, indexVar, "ptr");
            auto tmpVal = builder.CreateAlignedLoad(ptrVal, alignment, "tmp");
            auto inputConst = llvm::ConstantFP::get(llvm::Type::getFloatTy(ir.context), (float)this->value);
            auto retVal = builder.CreateFMul(tmpVal, inputConst, "ret");
            builder.CreateAlignedStore(retVal, ptrVal, alignment);
        }
    };

    //Divide by every value of the vector with a value
    template <typename T>
    class DivAction : public SimpleAction<T>{
    public:
        DivAction(T v): SimpleAction<T>(v) {}

        void generateIR(yagal::generator::IRModule& ir, llvm::Function* kernel, int& inputVectorCounter){
            _p.debug() << "generateIR for add action called." << std::endl;

            //Prepare the block to fill in
            auto actionBlock = llvm::BasicBlock::Create(ir.context, llvm::Twine(ir.getNextBasicBlockName()), kernel);
            llvm::IRBuilder<> builder(actionBlock);
            ir.userBlocks.push_back(actionBlock);

            //Prepare argument
            auto vecVal = kernel->arg_begin();
            vecVal->setName("vec");

            //Build llvm ir
            int alignment = 4;
            auto indexVar = builder.CreateAlignedLoad(ir.currentIndexValue, 4, "i");
            auto ptrVal = builder.CreateGEP(vecVal, indexVar, "ptr");
            auto tmpVal = builder.CreateAlignedLoad(ptrVal, alignment, "tmp");
            auto inputConst = llvm::ConstantFP::get(llvm::Type::getFloatTy(ir.context), (float)this->value);
            auto retVal = builder.CreateFDiv(tmpVal, inputConst, "ret");
            builder.CreateAlignedStore(retVal, ptrVal, alignment);
        }
    };

    //For each value of the vector, add the corresponsing value of another vector to it
    template <typename T>
    class AddVectorAction : public ParameterAction<T>{
    public:
        AddVectorAction(Vector<T>& v): ParameterAction<T>(v) {}

        void generateIR(yagal::generator::IRModule& ir, llvm::Function* kernel, int& inputVectorCounter){
            _p.debug() << "generateIR for add vector action called." << std::endl;

            //Increment counter, to let succeeding calls get correct vectors;
            inputVectorCounter++;

            //Prepare the block to fill in
            auto actionBlock = llvm::BasicBlock::Create(ir.context, llvm::Twine(ir.getNextBasicBlockName()), kernel);
            llvm::IRBuilder<> builder(actionBlock);
            ir.userBlocks.push_back(actionBlock);

            //Prepare arguments
            auto argIter = kernel->arg_begin();
            auto vec1Val = argIter;
            for(int i = 0; i < inputVectorCounter; i++){
                argIter++;
            }
            auto vec2Val = argIter;
            vec1Val->setName("vec1");
            vec2Val->setName("vec2");

            //Build llvm ir
            int alignment = 4;
            auto indexVar = builder.CreateAlignedLoad(ir.currentIndexValue, alignment, "i");
            auto ptr1Val = builder.CreateGEP(vec1Val, indexVar, "ptr1");
            auto ptr2Val = builder.CreateGEP(vec2Val, indexVar, "ptr2");
            auto tmp1Val = builder.CreateAlignedLoad(ptr1Val, alignment, "tmp1");
            auto tmp2Val = builder.CreateAlignedLoad(ptr2Val, alignment, "tmp2");
            auto retVal = builder.CreateFAdd(tmp1Val, tmp2Val, "ret");
            builder.CreateAlignedStore(retVal, ptr1Val, alignment);
        }
    };

    //For each value of the vector, subtract with the corresponsing value of another vector
    template <typename T>
    class SubVectorAction : public ParameterAction<T>{
    public:
        SubVectorAction(Vector<T>& v): ParameterAction<T>(v) {}

        void generateIR(yagal::generator::IRModule& ir, llvm::Function* kernel, int& inputVectorCounter){
            _p.debug() << "generateIR for add vector action called." << std::endl;

            //Increment counter, to let succeeding calls get correct vectors;
            inputVectorCounter++;

            //Prepare the block to fill in
            auto actionBlock = llvm::BasicBlock::Create(ir.context, llvm::Twine(ir.getNextBasicBlockName()), kernel);
            llvm::IRBuilder<> builder(actionBlock);
            ir.userBlocks.push_back(actionBlock);

            //Prepare arguments
            auto argIter = kernel->arg_begin();
            auto vec1Val = argIter;
            for(int i = 0; i < inputVectorCounter; i++){
                argIter++;
            }
            auto vec2Val = argIter;
            vec1Val->setName("vec1");
            vec2Val->setName("vec2");

            //Build llvm ir
            int alignment = 4;
            auto indexVar = builder.CreateAlignedLoad(ir.currentIndexValue, alignment, "i");
            auto ptr1Val = builder.CreateGEP(vec1Val, indexVar, "ptr1");
            auto ptr2Val = builder.CreateGEP(vec2Val, indexVar, "ptr2");
            auto tmp1Val = builder.CreateAlignedLoad(ptr1Val, alignment, "tmp1");
            auto tmp2Val = builder.CreateAlignedLoad(ptr2Val, alignment, "tmp2");
            auto retVal = builder.CreateFSub(tmp1Val, tmp2Val, "ret");
            builder.CreateAlignedStore(retVal, ptr1Val, alignment);
        }
    };

    //For each value of the vector, multiply by the corresponsing value of another vector
    template <typename T>
    class MultVectorAction : public ParameterAction<T>{
    public:
        MultVectorAction(Vector<T>& v): ParameterAction<T>(v) {}

        void generateIR(yagal::generator::IRModule& ir, llvm::Function* kernel, int& inputVectorCounter){
            _p.debug() << "generateIR for add vector action called." << std::endl;

            //Increment counter, to let succeeding calls get correct vectors;
            inputVectorCounter++;

            //Prepare the block to fill in
            auto actionBlock = llvm::BasicBlock::Create(ir.context, llvm::Twine(ir.getNextBasicBlockName()), kernel);
            llvm::IRBuilder<> builder(actionBlock);
            ir.userBlocks.push_back(actionBlock);

            //Prepare arguments
            auto argIter = kernel->arg_begin();
            auto vec1Val = argIter;
            for(int i = 0; i < inputVectorCounter; i++){
                argIter++;
            }
            auto vec2Val = argIter;
            vec1Val->setName("vec1");
            vec2Val->setName("vec2");

            //Build llvm ir
            int alignment = 4;
            auto indexVar = builder.CreateAlignedLoad(ir.currentIndexValue, alignment, "i");
            auto ptr1Val = builder.CreateGEP(vec1Val, indexVar, "ptr1");
            auto ptr2Val = builder.CreateGEP(vec2Val, indexVar, "ptr2");
            auto tmp1Val = builder.CreateAlignedLoad(ptr1Val, alignment, "tmp1");
            auto tmp2Val = builder.CreateAlignedLoad(ptr2Val, alignment, "tmp2");
            auto retVal = builder.CreateFMul(tmp1Val, tmp2Val, "ret");
            builder.CreateAlignedStore(retVal, ptr1Val, alignment);
        }
    };

    //For each value of the vector, divide by the corresponsing value of another vector
    template <typename T>
    class DivVectorAction : public ParameterAction<T>{
    public:
        DivVectorAction(Vector<T>& v): ParameterAction<T>(v) {}

        void generateIR(yagal::generator::IRModule& ir, llvm::Function* kernel, int& inputVectorCounter){
            _p.debug() << "generateIR for add vector action called." << std::endl;

            //Increment counter, to let succeeding calls get correct vectors;
            inputVectorCounter++;

            //Prepare the block to fill in
            auto actionBlock = llvm::BasicBlock::Create(ir.context, llvm::Twine(ir.getNextBasicBlockName()), kernel);
            llvm::IRBuilder<> builder(actionBlock);
            ir.userBlocks.push_back(actionBlock);

            //Prepare arguments
            auto argIter = kernel->arg_begin();
            auto vec1Val = argIter;
            for(int i = 0; i < inputVectorCounter; i++){
                argIter++;
            }
            auto vec2Val = argIter;
            vec1Val->setName("vec1");
            vec2Val->setName("vec2");

            //Build llvm ir
            int alignment = 4;
            auto indexVar = builder.CreateAlignedLoad(ir.currentIndexValue, alignment, "i");
            auto ptr1Val = builder.CreateGEP(vec1Val, indexVar, "ptr1");
            auto ptr2Val = builder.CreateGEP(vec2Val, indexVar, "ptr2");
            auto tmp1Val = builder.CreateAlignedLoad(ptr1Val, alignment, "tmp1");
            auto tmp2Val = builder.CreateAlignedLoad(ptr2Val, alignment, "tmp2");
            auto retVal = builder.CreateFDiv(tmp1Val, tmp2Val, "ret");
            builder.CreateAlignedStore(retVal, ptr1Val, alignment);
        }
    };

}
