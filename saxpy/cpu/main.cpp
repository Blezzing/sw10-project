#include <vector>
#include <iostream>
#include <algorithm>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;

int main(void)
{
    size_t N = 1 << 29;
    float a = 11;

    std::vector<float> x(N);
    std::vector<float> y(N);

    std::generate(x.begin(), x.end(), rand);
    std::generate(y.begin(), y.end(), rand);

    auto t0 = Clock::now();
    std::transform(x.begin(), x.end(), y.begin(), x.begin(), [=](float x, float y)->float{return a * x + y;});
    auto t1 = Clock::now();

    std::cout
            << "elapsed: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() << "ns" << std::endl;

    return 0;     
}