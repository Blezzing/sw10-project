#include "yagal/vector.hpp"
#include "yagal/cudaHandler.hpp"
#include "yagal/cudaExecutor.hpp"
#include <vector>
#include <iostream>

#include <chrono>
typedef std::chrono::high_resolution_clock Clock;
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
    yagal::Vector<int> v({1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,11, 12, 13, 14, 15, 16, 17, 18, 19});
    v.dump();
    v.add(5).add(4).exec();
    v.dump();
}

void timedTasks(){
    //std::vector<float> src(1000000000); //max
    std::vector<float> src(100000000);
    std::srand(0);
    std::generate(src.begin(), src.end(), std::rand);

    yagal::Vector<float> v1(src);
    yagal::Vector<float> v2(src);
    auto ptx = v1.add(v2).exportPtx();
    auto t1 = Clock::now();
    yagal::cuda::executePtxString(ptx, {v1.getDevicePtrPtr(), v2.getDevicePtrPtr()});
    auto t2 = Clock::now();

    auto t3 = Clock::now();
    v1.add(v2).exec();
    auto t4 = Clock::now();

    std::vector<float> vx1(src);
    std::vector<float> vx2(src);
    auto t5 = Clock::now();
    for(int i = 0; i < vx1.size(); i++){
        vx1[i] = vx1[i] + vx2[i];
    }
    auto t6 = Clock::now();

    auto t7 = Clock::now();
    v1.add(v2).exec({1,1,1},{1,1,1});
    auto t8 = Clock::now();

    auto addConst = 5;
    auto singleAddKernel = v1.add(addConst).exportPtx();
    auto doubleAddKernel = v1.add(addConst).add(addConst).exportPtx();

    auto t9 = Clock::now();
    yagal::cuda::executePtxString(doubleAddKernel, {v1.getDevicePtrPtr()});
    auto t10 = Clock::now();

    auto t11 = Clock::now();
    yagal::cuda::executePtxString(singleAddKernel, {v1.getDevicePtrPtr()});
    yagal::cuda::executePtxString(singleAddKernel, {v1.getDevicePtrPtr()});
    auto t12 = Clock::now();

    std::cout << "Delta raw:                                  " 
              << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
              << " nanoseconds" << std::endl;    
    std::cout << "Delta with build:                           " 
              << std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count()
              << " nanoseconds" << std::endl;
    std::cout << "Delta on cpu:                               " 
              << std::chrono::duration_cast<std::chrono::nanoseconds>(t6 - t5).count()
              << " nanoseconds" << std::endl;
    std::cout << "Delta with build and custom parameters:     " 
              << std::chrono::duration_cast<std::chrono::nanoseconds>(t8 - t7).count()
              << " nanoseconds" << std::endl;
    std::cout << "Delta on single kernel adding two constants:         " 
              << std::chrono::duration_cast<std::chrono::nanoseconds>(t10 - t9).count()
              << " nanoseconds" << std::endl;
    std::cout << "Delta on two kernels adding a single constant:       " 
              << std::chrono::duration_cast<std::chrono::nanoseconds>(t12 - t11).count()
              << " nanoseconds" << std::endl;
}

bool assertEqual(const std::vector<float>& v1, const std::vector<float>& v2){
    if (v1.size() != v2.size()) {
        return false;
    }

    auto it1 = v1.begin();
    auto it2 = v2.begin();
    while(it1 != v1.end() && it2 != v2.end()){
        if (*it1 != *it2) {
            return false;
        }
        it1++;
        it2++;
    }
    return true;
}

void cpuTest(){
    std::vector<float> src(1 << 29);
    std::srand(0);
    std::generate(src.begin(), src.end(), std::rand);

    auto t0 = Clock::now();
    std::transform(src.begin(), src.end(), src.begin(), [](float x){return x+1;});
    auto t1 = Clock::now();

    std::cout 
        << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " milliseconds" 
        << " on cpu:"
        << std::endl;
}

void defaultParamterTest(){
    std::vector<float> src(1 << 29);
    std::srand(0);
    std::generate(src.begin(), src.end(), std::rand);
    yagal::Vector<float> v(src);

    auto ptx = v.add(1).exportPtx();

    for(int x = 1; x <= 1024; x *= 2){
        for(int y = 1; y <= 1024; y *= 2){
            auto t0 = Clock::now();
            v.exec(ptx, {}, {x,1,1}, {y,1,1});
            auto t1 = Clock::now();

            //std::transform(src.begin(), src.end(), src.begin(), [](float x){return x+1;});
            //auto valid = assertEqual(src, v);

            std::cout 
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " milliseconds" 
              << " with "
              << "blockDim: {" << x <<",1,1}, gridDim: {" << y << ",1,1}: "
              //<< ((valid)?"(valid)":"(FAIL)") 
              << std::endl;
        }
    }
}

void defaultParamterTestOnSmallData(){
    std::vector<float> src(1024);
    std::srand(0);
    std::generate(src.begin(), src.end(), std::rand);
    yagal::Vector<float> v(src);

    auto ptx = v.add(1).exportPtx();

    for(int x = 1; x <= 1024; x *= 2){
        for(int y = 1; y <= 1024; y *= 2){
            auto t0 = Clock::now();
            v.exec(ptx, {}, {x,1,1}, {y,1,1});
            auto t1 = Clock::now();

            //std::transform(src.begin(), src.end(), src.begin(), [](float x){return x+1;});
            //auto valid = assertEqual(src, v);

            std::cout 
              << std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() << " nanoseconds" 
              << " with "
              << "blockDim: {" << x <<",1,1}, gridDim: {" << y << ",1,1}: "
              //<< ((valid)?"(valid)":"(FAIL)") 
              << std::endl;
        }
    }
}

void floatTest(){
    std::cout << "\nFLOATS:" << std::endl;
    yagal::Vector<float> v({1,2,3,4,5,6,7,8,9,0});
    v.dump();
    v.add(5).exec({2,1,1},{2,1,1});
    v.dump();
}

void intTest(){
    std::cout << "\nINTS:" << std::endl;
    yagal::Vector<int> v({1,2,3,4,5,6,7,8,9,0});
    v.dump();
    v.add(5).exec({2,1,1},{1,1,1});
    v.dump();
}

void sanityTest(){
    std::vector<float> src(1024);
    std::srand(0);
    std::generate(src.begin(), src.end(), std::rand);
    yagal::Vector<float> v1(src);
    yagal::Vector<float> v2(src);

    yagal::Vector<float> v3(src);
    yagal::Vector<float> v4(src);
    yagal::Vector<float> v5(src);
    yagal::Vector<float> v6(src);

    v1.add(1).exec();
    v2.exec(v2.add(1).exportPtx(),{});

    v3.multiply(2).exec();
    v4.add(v5).exec();
    v5.exec(v5.add(v6).exportPtx(),{v6.getDevicePtrPtr()});


    bool success = assertEqual(v1,v2)
                && assertEqual(v3,v4)
                && assertEqual(v4,v5);

    std::cout << (success?"We are sane!":"We are NOT sane!") << std::endl;
}

void timeBuild(){
    std::vector<float> src(1<<16);
    std::srand(0);
    std::generate(src.begin(), src.end(), std::rand);
    yagal::Vector<float> v(src);


    auto t0 = Clock::now();
    v.add(1).exec();
    auto t1 = Clock::now();
    auto ptx = v.add(1).exportPtx();
    auto t2 = Clock::now();
    v.exec(ptx,{});
    auto t3 = Clock::now();

    std::cout 
        << std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() << " nanoseconds" 
        << ": execution including build" << std::endl
        << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << " nanoseconds" 
        << ": pure build build" << std::endl
        << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count() << " nanoseconds" 
        << ": pure execution" << std::endl;
}

void ptxOptimExample(){
    yagal::Vector<float> v(100);
    v.add(5).add(5).exec();
}

int main(){
    //floatTest();
    //intTest();
    //cpuTest();
    //defaultParamterTest();
    //sanityTest();
    //timeBuild();
    ptxOptimExample();
}