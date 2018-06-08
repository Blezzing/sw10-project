#include <iostream>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;

__global__ void kernel(int n, float a, float* x, float* y){
    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){
        x[i] = a * x[i] + y[i];
    }
}

int main(void){
    int N = 1 << 29;
    float a = 11.0;

    float *h_x;
    float *h_y; 
    float *d_x;
    float *d_y;
    h_x = (float*)malloc(N*sizeof(float));
    h_y = (float*)malloc(N*sizeof(float));

    cudaMalloc(&d_x, N*sizeof(float)); 
    cudaMalloc(&d_y, N*sizeof(float));

    for(int i = 0; i < N; i++){
        h_x[i] = rand();
        h_y[i] = rand();
    }

    cudaMemcpy(d_x, h_x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N*sizeof(float), cudaMemcpyHostToDevice);


    auto t0 = Clock::now();
    kernel<<<128, 128>>>(N, a, d_x, d_y);
    auto t1 = Clock::now();

    cudaMemcpy(h_x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

    std::cout
            << "elapsed: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() << "ns" << std::endl;
       
}
