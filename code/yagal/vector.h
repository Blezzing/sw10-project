#pragma once

#include <vector>
#include <functional>

#include "cudaHandler.h"
#include "llvmHandler.h"

namespace yagal{
    // Forward declaration
    template <typename T> class Vector;

    template <typename T>
    class Vector{
    private:
        CUdeviceptr _devicePtr;
        size_t _count;

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
            std::vector<T> v(*this);
            for (const T& e : v){
                std::cout << e << " ";
            }
            std::cout << std::endl;
        }

        //the do/execute function, genererer kernel og eksekverer
        void exec(){
            auto ptx = yagal::generator::buildPTX();
            yagal::cuda::executePtxOnData(ptx, _devicePtr, _count);
        }

        //functions
        Vector<T>& map(std::function<T(T)> lambda){
            //std::cout << "Map queued" << std::endl;
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
