#include "yagal/vector.hpp"
#include <vector>
#include <iostream>

int main(){
    size_t N = 1 << 24;
    float a = 11;

    std::vector<float> h_x(N);
    std::vector<float> h_y(N);

    std::generate(h_x.begin(), h_x.end(), rand);
    std::generate(h_y.begin(), h_y.end(), rand);

    yagal::Vector<float> d_x(h_x);
    yagal::Vector<float> d_y(h_x);
    

    std::cout << d_x.getElement(0);
    d_x.multiply(a).add(d_y).exec();
    std::cout << "-" << d_x.getElement(0) << std::endl;
    return 0;
}