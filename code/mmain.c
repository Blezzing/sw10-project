#include "yagal/vector.h"

#include <algorithm>

void clusterfuck(){
    yagal::Vector<float> v({1.0, 2.0, 3.0, 4.0, 5.0});
    yagal::Vector<float> v2({5.0, 4.0, 3.0, 2.0, 1.0});
    v.dump();
    v.add(v2).exec();
    v.dump();
    v2.dump();
}

void saxpy(){
    float a = 5;
    yagal::Vector<float> x({0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
    yagal::Vector<float> y({5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0});

    x.multiply(a).add(y).exec();

    x.dump();
}

int main(){
    saxpy();
}