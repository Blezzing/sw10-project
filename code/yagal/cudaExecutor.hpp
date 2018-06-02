#include "vector.hpp"
#include "cudaHandler.hpp"
#include "llvmHandler.hpp"

namespace yagal::cuda{

    //User access to execute ptx directly
    void executePtxString(const std::string& ptx, const std::vector<CUdeviceptr*>& devicePtrs){
        executePtxWithParams(ptx, devicePtrs);
    }

    void executePtxFile(const std::string& fileName, const std::vector<CUdeviceptr*>& devicePtrs){
        yagal::generator::PTXModule mod(fileName);
        executePtxWithParams(mod.toString(), devicePtrs);
    }
}