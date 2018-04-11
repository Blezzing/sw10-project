#include "yagal/vector.h"

#include <algorithm>

int main(){
    std::vector<float> std_v(512);
    std::generate(std_v.begin(), std_v.end(), [](){return 0.0;});
    yagal::Vector<float> v(std_v);
    v.dump();
    v.add(1).add(1).exec();
    v.dump();
}