#include <gpu> //vores gudebibliotek

gpu::array gpuAr({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20});

//med resultat i andet array
gpu::array result = gpuAr.sub(0,10)                                   //1,2,3,4,5,6,7,8,9,10
                         .filter([](int x){return x % 2 == 0;})       //2,4,6,8,10
                         .map([](int x){return x * 2;})               //4,8,12,16,20
                         .filter([](int x){return x % 3 == 0;})       //12
                         .reduce([](int x, int y){return x + y;}, 0)  //12
                         .execute(); //måske nødvendig/rart?

//inplace?
gpuAr.filter([](int x){return x % 2 == 0;})
     .executeInplace();


std::array cpuAr(gpuAr);

assert(gpuAr[0], ???);
assert(gpuAr[0], 1);  //<-- hvis funktionerne fungerer som man forventer "funktionelt".
assert(gpuAr[0], 12); //hvis den udfører alt inklusivt reduce in place.