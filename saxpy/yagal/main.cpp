#include "yagal/vector.hpp"
#include <vector>
#include <iostream>
#include <chrono>
#include "../saxpyValidator.hpp"

typedef std::chrono::high_resolution_clock Clock;

int main(){
    size_t N = 1 << 29;
    float a = 11;

    std::vector<float> h_x(N);
    std::vector<float> h_y(N);

    std::generate(h_x.begin(), h_x.end(), rand);
    std::generate(h_y.begin(), h_y.end(), rand);

    yagal::Vector<float> d_x(h_x);
    yagal::Vector<float> d_y(h_x);

    //warmup
    d_x.multiply(a).add(d_y).exec();
    
    auto t0 = Clock::now();
    d_x.multiply(a).add(d_y).exec();
    auto t1 = Clock::now();

    auto ptx = d_x.multiply(a).add(d_y).exportPtx();
    auto t2 = Clock::now();
    d_x.exec(ptx, {d_y.getDevicePtrPtr()});
    auto t3 = Clock::now();

    std::cout
        << "elapsed including build: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() << "ns" << std::endl
        << "elapsed excluding build: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count() << "ns" << std::endl;
    return 0;
}