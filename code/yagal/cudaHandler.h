#pragma once

#include <iostream>
#include <fstream>
#include <cassert>
#include "printer.hpp"
#include "cuda.h"

namespace yagal::cuda{
    namespace {
        printer::Printer _p("cudaHandler", printer::Printer::Mode::Debug);
    }

    CUdevice   cudaDevice;
    int        cudaDeviceCount(0);
    CUcontext  cudaContext;

    void checkCudaErrors(CUresult err, const std::string& sufix = "SOMEWHERE") {
        if(!err) return;

        auto& o = _p.error();

        const char* str;
        cuGetErrorString(err, &str);
        if (!str){
            o << "cuda error with unrecognized error value " << sufix << std::endl;
            return;
        }

        o << str << " " << sufix << std::endl;

        exit(1);
    }

    void initIfNeeded(){
        if(cudaContext) return;
        
        if(!cudaDevice){
            checkCudaErrors(cuInit(0));
            checkCudaErrors(cuDeviceGetCount(&cudaDeviceCount));
            checkCudaErrors(cuDeviceGet(&cudaDevice, 0));
            _p.debug() << "cuda device initialized" << std::endl;
        }

        checkCudaErrors(cuCtxCreate(&cudaContext, 0, cudaDevice));
        _p.debug() << "cuda context created" << std::endl;
    }

    void checkDevice(){
        initIfNeeded();

        char name[128];
        checkCudaErrors(cuDeviceGetName(name, 128, cudaDevice));
        _p.debug() << "Using CUDA Device [0]: " << name << std::endl;

        int devMajor, devMinor;
        checkCudaErrors(cuDeviceComputeCapability(&devMajor, &devMinor, cudaDevice));
        _p.debug() << "Device Compute Capability: " << devMajor << "." << devMinor << std::endl;
        if (devMajor < 2) {
            _p.error() << "CUDA Device 0 is not SM 2.0 or greater" << std::endl;
        }
    }

    void copyToHost(void* dest, CUdeviceptr src, size_t size){
        checkCudaErrors(cuMemcpyDtoH(dest, src, size), "at copyToHost");
    }

    void copyToDevice(CUdeviceptr dest, const void* src, size_t size){
        checkCudaErrors(cuMemcpyHtoD(dest, src, size), "at copyToDevice");
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
        _p.info() << "cuda kernel launching" << std::endl;
        checkCudaErrors(cuLaunchKernel(function, gridSizeX, gridSizeY, gridSizeZ,
                                        blockSizeX, blockSizeY, blockSizeZ,
                                        0, NULL, KernelParams, NULL));

        // Cleanup
        checkCudaErrors(cuModuleUnload(cudaModule));

        return 0;
    }
}
