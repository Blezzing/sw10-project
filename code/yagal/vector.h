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

        //Mutaters that transform all values with a single value
        Vector<T>& add(T value) {
            _actions.emplace_back(new internal::AddAction<T>(value));
            return *this;
        }

        Vector<T>& multiply(T value) {
            _actions.emplace_back(new internal::MultAction<T>(value));
            return *this;
        }

        //Mutaters that transform a vector with another vector
        Vector<T>& add(Vector<T>& other){
            _actions.emplace_back(new internal::AddVectorAction<T>(other));
            return *this;
        }


        //the do/execute function, genererer kernel og eksekverer
        Vector<T>& exec(){
            //We can concatenate actions and do other optimizations here, eg add(5) + add(5) = add(10);

            yagal::generator::IRModule ir(_count);

            //Count number of cuda parameters needed, starting at 1 to include the vector itself.
            int totalVectorCount = 1;
            std::vector<CUdeviceptr*> devicePointers({&_devicePtr});
            for (auto& a : _actions){
                if(a->requiresCudaParameter()){
                    totalVectorCount++;
                    auto pa = static_cast<internal::ParameterAction<T>*>(a.get());
                    auto ptr = pa->getDevicePtrPtr();
                    devicePointers.push_back(ptr);
                }
            }

            //Generate llvm ir blocks.
            int inputVectorCounter = 0;
            auto kernel = ir.createKernel(totalVectorCount);
            for (const auto& a : _actions){
                a->generateIR(ir, kernel, inputVectorCounter);
            }

            //Link blocks and update metadata.
            ir.finalizeKernel(kernel);
            ir.updateMetadata();


            _p.debug() << ir.toString() << std::endl;
            yagal::generator::PTXModule ptx(ir);
            auto ptxSource = ptx.toString();
            _p.debug() << ptx.toString() << std::endl;
            yagal::cuda::executePtxWithParams(ptxSource, devicePointers);
            //yagal::cuda::executePtxOnData(ptxSource, _devicePtr, _devicePtr, _count);
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

        CUdeviceptr* getDevicePtrPtr(){
            return &_devicePtr;
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
