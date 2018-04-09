#pragma once

#include <iostream>
#include <fstream>
#include <cassert>

#include "cuda.h"

namespace yagal::cuda{
    CUdevice   cudaDevice;
    int        cudaDeviceCount(0);
    CUcontext  cudaContext;

    void checkCudaErrors(CUresult err, const std::string& sufix = "SOMEWHERE") {
        if(!err) return;

        std::cerr << "CUDA ERROR: ";

        const char* str = NULL;
        cuGetErrorString(err, &str);
        if (!str){
            std::cerr << "cuda error with unrecognized error value " << sufix << std::endl;
            return;
        }

        std::cout << str << " " << sufix << std::endl;

        assert(err == CUDA_SUCCESS);
    }

    void initIfNeeded(){
        if(cudaContext) return;
        
        if(!cudaDevice){
            checkCudaErrors(cuInit(0));
            checkCudaErrors(cuDeviceGetCount(&cudaDeviceCount));
            checkCudaErrors(cuDeviceGet(&cudaDevice, 0));
            std::cout << "cuda device initialized" << std::endl;
        }

        checkCudaErrors(cuCtxCreate(&cudaContext, 0, cudaDevice));
        std::cout << "cuda context created" << std::endl;
    }

    void checkDevice(){
        initIfNeeded();

        char name[128];
        checkCudaErrors(cuDeviceGetName(name, 128, cudaDevice));
        std::cout << "Using CUDA Device [0]: " << name << std::endl;

        int devMajor, devMinor;
        checkCudaErrors(cuDeviceComputeCapability(&devMajor, &devMinor, cudaDevice));
        std::cout << "Device Compute Capability: " << devMajor << "." << devMinor << std::endl;
        if (devMajor < 2) {
            std::cerr << "CUDA ERROR: Device 0 is not SM 2.0 or greater" << std::endl;
        }
    }

    void copyToHost(void* dest, CUdeviceptr src, size_t size){
        cuMemcpyDtoH(dest, src, size);
    }

    void copyToDevice(CUdeviceptr dest, const void* src, size_t size){
        cuMemcpyHtoD(dest, src, size);
    }

    CUdeviceptr malloc(size_t size){
        initIfNeeded();

        CUdeviceptr device_ptr;
        auto status = cuMemAlloc(&device_ptr, size);
        checkCudaErrors(status,  "at malloc");
        return device_ptr;
    }

    void free(CUdeviceptr device_ptr){
        auto status = cuMemFree(device_ptr);
        checkCudaErrors(status,  "at free");
    }

    std::string loadPTXToString(const std::string& filename){
        std::ifstream file(filename);
        std::string str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        return str;
    }

    int executePtxOnData(const std::string& ptx, CUdeviceptr data_ptr, size_t n){
        CUmodule    cudaModule;
        CUfunction  function;
        CUlinkState linker;
        int         devCount;

        initIfNeeded();

        // Create module for object
        checkCudaErrors(cuModuleLoadDataEx(&cudaModule, ptx.c_str(), 0, 0, 0));

        // Get kernel function
        checkCudaErrors(cuModuleGetFunction(&function, cudaModule, "kernel"));

        // Define where data is, to set kernel params

        // Set kernel configuration
        unsigned blockSizeX = 16;
        unsigned blockSizeY = 1;
        unsigned blockSizeZ = 1;
        unsigned gridSizeX  = 1;
        unsigned gridSizeY  = 1;
        unsigned gridSizeZ  = 1;

        // Hand over input to kernel
        void *KernelParams[] = {&data_ptr};

        // Kernel launch
        checkCudaErrors(cuLaunchKernel(function, gridSizeX, gridSizeY, gridSizeZ,
                                        blockSizeX, blockSizeY, blockSizeZ,
                                        0, NULL, KernelParams, NULL));

        // Cleanup
        checkCudaErrors(cuModuleUnload(cudaModule));

        return 0;
    }
}