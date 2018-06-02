#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>

int main(void)
{
    size_t N = 1 << 24;
    float a = 11;

    thrust::host_vector<float> h_x(N);
    thrust::host_vector<float> h_y(N);

    std::generate(h_x.begin(), h_x.end(), rand);
    std::generate(h_y.begin(), h_y.end(), rand);

    thrust::device_vector<float> d_x = h_x;
    thrust::device_vector<float> d_y = h_y;

    thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), d_x.begin(), [=]__device__(float x, float y)->float{return a * x + y;});

    thrust::copy(d_x.begin(), d_x.end(), h_x.begin());

    return 0;
}