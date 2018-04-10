#pragma once

#include <vector>
#include <functional>

#include "cudaHandler.h"
#include "llvmHandler.h"
#include "printer.hpp"
#include "action.hpp"

namespace yagal{
    namespace {
        printer::Printer _p("vector", printer::Printer::Mode::Debug);
    }
    // Forward declaration
    template <typename T> class Vector;

    template <typename T>
    class Vector{
    private:
        CUdeviceptr _devicePtr;
        size_t _count;
        std::vector<std::shared_ptr<internal::Action>> _actions;

    public:
        // Constructors
        Vector(int elementCount)
            : _count(elementCount)
        {
            _devicePtr = yagal::cuda::malloc(_count * sizeof(T));
        }

        Vector(const std::vector<T>& source)
            : _count(source.size())
        {
            _devicePtr = yagal::cuda::malloc(_count * sizeof(T));
            yagal::cuda::copyToDevice(_devicePtr, source.data(), _count * sizeof(T));
        }

        Vector()
            : Vector(0)
        {}

        void dump(){
            auto& o = _p.info();
            std::vector<T> v(*this);
            for (const T& e : v){
                o << e << " ";
            }
            o << std::endl;
        }

        Vector<T>& add(T value) {
            _actions.emplace_back(new internal::AddAction<T>(value));
            return *this;
        }

        //the do/execute function, genererer kernel og eksekverer
        void exec(){
            //We can concatenate actions and do other optimizations here, eg add(5) + add(5) = add(10);

            yagal::generator::IRModule ir;

            auto kernel = ir.createKernel();
            for (const auto& a : _actions){
                a->generateIR(ir, kernel);
            }
            ir.finalizeKernel(kernel);
            ir.updateMetadata();


            _p.debug() << ir.toString() << std::endl;
            auto ptx = yagal::generator::llc::translate(ir.context, ir.module);
            yagal::cuda::executePtxOnData(ptx, _devicePtr, _count);
        }

        //functions
        Vector<T>& map(std::function<T(T)> lambda){
            _p.info()<< "Map queued" << std::endl;
            return *this;
        }

        std::vector<T> copyToHostVector(){
            std::vector<T> result(_count);
            yagal::cuda::copyToHost(result.data(), _devicePtr, _count * sizeof(T));
            return result;
        }

        T getElement(int index){
            T result;
            yagal::cuda::copyToHost(&result, _devicePtr+(index*sizeof(T)), sizeof(T));
            return result;
        }

        void setElement(int index, T value){
            yagal::cuda::copyToDevice(_devicePtr+(index*sizeof(T)), &value, sizeof(T));
        }

        //auto conversion to std vector to allow use of std vector function
        operator std::vector<T>(){
            return copyToHostVector();
        }

        // Destructor
        ~Vector(){
            yagal::cuda::free(_devicePtr);
        }
    };
}
