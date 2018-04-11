#include "yagal/vector.h"
#include <vector>
/*
void construction(){
    yagal::cuda::checkDevice();

    int a = 5;
    yagal::Vector<int> s(1);
    yagal::Vector<int> x(100);
    yagal::Vector<int> y(100);

    std::vector<int> sx({1,2,3,4,5,6});
    std::vector<int> sy({2,4,6,8,10,12});

    yagal::Vector<int> dx(sx);
    yagal::Vector<int> dy(sy);


    dx.setElement(3,10);
    auto test2 = dx.getElement(2);
    auto test = dx.copyToHostVector();

    yagal::Vector<int> ermeged({9,8,7,6,5,4,3,2,1});

    std::cout << "test[2] = " << test2 << std::endl;
}

void plz(){
    yagal::Vector<float> v({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 ,11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 ,11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 ,11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 ,11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    v.dump();    
    v.exec();
    v.dump();
}

void map(){
    yagal::Vector<int> v({1,2,3,4,5,6,7,8,9,0});

    v.map([](int x){return x + 5;}).exec();
}
*/
void actual(){
    yagal::Vector<float> v({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 ,11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    v.dump();
    v.add(5).add(4).exec();
    v.dump();
}

void actualInt(){
    yagal::Vector<int> v({1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,11, 12, 13, 14, 15, 16});
    v.dump();
    v.add(5).add(4).exec();
    v.dump();
}

int main(){
    actualInt();
}