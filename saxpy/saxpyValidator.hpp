#include <vector>

namespace validator{
    bool isValid(const std::vector<float>& cpuResult, const std::vector<float>& gpuResult){
        for(size_t i = 0; i < cpuResult.size(); i++){
            if (cpuResult[i] != gpuResult[i]){
                std::cout << cpuResult[i] << " and " << gpuResult[i] << " at index " << i << " are not identical" << std::endl;
                return false;
            }
        }
        return true;
    }

    int totalDeviance(const std::vector<float>& cpuResult, const std::vector<float>& gpuResult){
        float dif = 0;
        for(size_t i = 0; i < cpuResult.size(); i++){
            dif += cpuResult[i] - gpuResult[i];
        }
        return dif;
    }

    void computeSaxpy(float a, std::vector<float>& x, std::vector<float>& y){
        std::transform(x.begin(), x.end(), y.begin(), x.begin(), [=](float ex, float ey)->float{return a * ex + ey;});
    }
}