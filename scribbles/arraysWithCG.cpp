gpu::array gpuAr({1,2,3,4,5,6,7,8,9,10});
gpuAr.filter([](int x){return x % 2 == 0;})
     .map([](int x){return x * 2;})
     .filter([](int x){return x % 3 == 0;})
     .reduce([](int x, int y){return x + y;}, 0)
     .execute(); //måske nødvendig/rart?

std::array cpuAr(gpuAr);
