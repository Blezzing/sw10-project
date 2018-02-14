#pragma once

//for memcpy, remove once we go gpu
#include <string.h>

#include <array>

namespace sgl{
    template<typename T, long unsigned int N>
    class Array{
        T* _device_ptr;
        size_t _alloc_size;
    public:
        Array()
            : _alloc_size(N * sizeof(T))
        {
            _device_ptr = (T*)malloc(_alloc_size);
        }

        Array(const std::array<T,N>& source)
            : Array()
        {
            memcpy(_device_ptr, source.data(), _alloc_size);
        }

        Array(const sgl::Array<T,N>& other)
            : Array()
        {
            memcpy(_device_ptr, other._device_ptr, _alloc_size);
        }



        T getAt(int index) const{
            return _device_ptr[index];            
        }

        std::array<T,N> get() const{
            std::array<T,N> ret;
            memcpy(&ret.data(), _device_ptr, _alloc_size);
            return ret;
        }

        T operator[](long unsigned int index) const{
            return getAt(index);
        }

        ~Array(){
            free(_device_ptr);
        }
    };
}