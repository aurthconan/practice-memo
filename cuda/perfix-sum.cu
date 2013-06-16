#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

__global__ void fill_one(int* d_array, size_t length) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if ( index >= length ) {
        return;
    }
    d_array[index] = 1;
}

#define BLOCK_SIZE 1024

__global__ void perfix_sum(int* d_array, size_t length) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if ( index >= length ) {
        return;
    }
    __shared__ int cache[BLOCK_SIZE];

    cache[threadIdx.x] = d_array[index];

    for ( size_t stride = 1; stride <= threadIdx.x; stride *= 2 ) {
        __syncthreads();
        cache[threadIdx.x] += cache[threadIdx.x - stride];
    }
    // write back
    d_array[index] = cache[threadIdx.x];
}

#define BLOCK_NUM 256

int main(int argc, char** argv) {
    int* d_array = NULL; 
    checkCudaErrors(cudaMalloc(&d_array, sizeof(int) * BLOCK_SIZE * BLOCK_NUM));

    fill_one<<<BLOCK_NUM, BLOCK_SIZE>>>(d_array, BLOCK_SIZE * BLOCK_NUM);

    perfix_sum<<<BLOCK_NUM, BLOCK_SIZE>>>(d_array, BLOCK_SIZE * BLOCK_NUM);

    int h_array[BLOCK_NUM*BLOCK_SIZE] = {0};

    checkCudaErrors(cudaMemcpy(h_array, d_array, sizeof(int) * BLOCK_SIZE * BLOCK_NUM, cudaMemcpyDeviceToHost));

    for ( size_t i = 0; i < BLOCK_NUM * BLOCK_SIZE; ++i ) {
        std::cout << h_array[i] << " ";
        if ( (i % BLOCK_SIZE) == (BLOCK_SIZE-1) ) {
            std::cout << std::endl;
        }
    }

    checkCudaErrors(cudaFree(d_array));
}
