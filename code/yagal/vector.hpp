#pragma once

#include <vector>
#include <functional>

#include "cudaHandler.hpp"
#include "llvmHandler.hpp"
#include "printer.hpp"
#include "action.hpp"

namespace yagal{
    namespace {
        printer::Printer _p("vector", printer::Printer::Mode::Standard);
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

        Vector<T>& subtract(T value) {
            _actions.emplace_back(new internal::SubAction<T>(value));
            return *this;
        }

        Vector<T>& multiply(T value) {
            _actions.emplace_back(new internal::MultAction<T>(value));
            return *this;
        }

        Vector<T>& divide(T value) {
            _actions.emplace_back(new internal::DivAction<T>(value));
            return *this;
        }

        //Mutaters that transform a vector with another vector
        Vector<T>& add(Vector<T>& other){
            _actions.emplace_back(new internal::AddVectorAction<T>(other));
            return *this;
        }

        Vector<T>& subtract(Vector<T>& other){
            _actions.emplace_back(new internal::SubVectorAction<T>(other));
            return *this;
        }

        Vector<T>& multiply(Vector<T>& other){
            _actions.emplace_back(new internal::MultVectorAction<T>(other));
            return *this;
        }

        Vector<T>& divide(Vector<T>& other){
            _actions.emplace_back(new internal::DivVectorAction<T>(other));
            return *this;
        }


        //the do/execute function, genererer kernel og eksekverer
        Vector<T>& exec(std::tuple<int, int, int> blockDimensions = {128, 1, 1}, std::tuple<int, int, int> gridDimensions = {128, 1, 1}){
            //We can concatenate actions and do other optimizations here, eg add(5) + add(5) = add(10);

            yagal::generator::IRModule ir(_count);

            //Count number of cuda parameters needed, starting at 1 to include the vector itself.
            std::vector<CUdeviceptr*> devicePointers({&_devicePtr});
            for (auto& a : _actions){
                if(a->requiresCudaParameter()){
                    auto pa = static_cast<internal::ParameterAction<T>*>(a.get());
                    auto ptr = pa->getDevicePtrPtr();
                    devicePointers.push_back(ptr);
                }
            }

            //Generate llvm ir blocks.
            int inputVectorCounter = 0;
            auto kernel = ir.createKernel(devicePointers.size());
            for (const auto& a : _actions){
                a->generateIR(ir, kernel, inputVectorCounter);
            }

            //Link blocks and update metadata.
            ir.finalizeKernel(kernel);
            ir.updateMetadata();

            //Generate code
            _p.debug() << ir.toString() << std::endl;
            yagal::generator::PTXModule ptx(ir);
            auto ptxSource = ptx.toString();
            _p.debug() << ptx.toString() << std::endl;
            
            //Execute kernel
            yagal::cuda::executePtxWithParams(ptxSource, devicePointers, blockDimensions, gridDimensions);

            //Cleanup
            _actions.clear();

            return *this;
        }

        Vector<T>& exec(const std::string& ptxSource, const std::vector<CUdeviceptr*>& otherVectors, std::tuple<int, int, int> blockDimensions = {128, 1, 1}, std::tuple<int, int, int> gridDimensions = {128, 1, 1}){
            std::vector<CUdeviceptr*> devicePointers({&_devicePtr});
            for(const auto& e: otherVectors){
                devicePointers.push_back(e);
            }

            //Execute kernel
            yagal::cuda::executePtxWithParams(ptxSource, devicePointers, blockDimensions, gridDimensions);

            //Cleanup
            _actions.clear();
        }

        std::string exportPtx(bool clearActions = true){
            yagal::generator::IRModule ir(_count);

            //Count number of cuda parameters needed, starting at 1 to include the vector itself.
            std::vector<CUdeviceptr*> devicePointers({&_devicePtr});
            for (auto& a : _actions){
                if(a->requiresCudaParameter()){
                    auto pa = static_cast<internal::ParameterAction<T>*>(a.get());
                    auto ptr = pa->getDevicePtrPtr();
                    devicePointers.push_back(ptr);
                }
            }

            //Generate llvm ir blocks.
            int inputVectorCounter = 0;
            auto kernel = ir.createKernel(devicePointers.size());
            for (const auto& a : _actions){
                a->generateIR(ir, kernel, inputVectorCounter);
            }

            //Link blocks and update metadata.
            ir.finalizeKernel(kernel);
            ir.updateMetadata();

            //Generate code
            _p.debug() << ir.toString() << std::endl;
            yagal::generator::PTXModule ptx(ir);
            auto ptxSource = ptx.toString();
            _p.debug() << ptx.toString() << std::endl;

            //Cleanup
            if(clearActions){
                _actions.clear();
            }

            return ptxSource;
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

        size_t getSize() const{
            return _count;
        }

        //auto conversion to std vector to allow use of std vector function
        operator std::vector<T>(){
            std::vector<T> result(_count);
            yagal::cuda::copyToHost(result.data(), _devicePtr, _count * sizeof(T));
            return result;
        }

        // Destructor
        ~Vector(){
            yagal::cuda::free(_devicePtr);
        }
    };
}
