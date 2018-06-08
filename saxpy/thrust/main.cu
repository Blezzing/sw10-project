#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/copy.h>
#include <algorithm>
#include <chrono>
#include "../saxpyValidator.hpp"

#include <cuPrintf.cuh>

typedef std::chrono::high_resolution_clock Clock;

int main(void)
{
    size_t N = 1 << 29;
    float a = 11;

    thrust::host_vector<float> h_x(N);
    thrust::host_vector<float> h_y(N);
    
    std::generate(h_x.begin(), h_x.end(), rand);
    std::generate(h_y.begin(), h_y.end(), rand);
    
    thrust::device_vector<float> d_x = h_x;
    thrust::device_vector<float> d_y = h_y;

    auto t0 = Clock::now();
    thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), d_x.begin(), [=]__device__(float x, float y)->float{cuPrintf("hej"); return a * x + y;});
    auto t1 = Clock::now();

    h_x = d_x;
    h_y = d_y;
    
    std::cout
            << "elapsed: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() << "ns" << std::endl;
            
    return 0;
}