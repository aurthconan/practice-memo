#include <stdio.h>
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

#define BLOCK_SIZE 2 

__global__ void perfix_sum_simple(int* d_array, size_t length) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    printf("index %d length %d", index, length);
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

__global__ void perfix_sum( int* d_array, size_t block_size, size_t length) {
    const int index = threadIdx.x + blockIdx.x * blockDim.x;
    const int start = index * block_size;

    if ( start >= length ) {
        return;
    }
    __shared__ int cache[BLOCK_SIZE];
    int local_copy[BLOCK_SIZE];

    for ( size_t i = 0; i < block_size; ++i ) {
        local_copy[i] = d_array[ start + i ];
    }

    for ( size_t stride = 1; stride < BLOCK_SIZE; stride *= 2 ) {
        cache[threadIdx.x] = local_copy[block_size-1];
        __syncthreads();
        int operend = cache[threadIdx.x-stride];
        for ( size_t i = 0; i < block_size; ++i ) {
            local_copy[i] += operend;
        }
    }

    // write back
    for ( size_t i = 0; i < block_size; ++i ) {
        d_array[ start + i ] = local_copy[i];
    }
}

#define BLOCK_NUM 1

int main(int argc, char** argv) {
    int* d_array = NULL; 
    checkCudaErrors(cudaMalloc(&d_array, sizeof(int) * BLOCK_SIZE * BLOCK_NUM));

    fill_one<<<BLOCK_NUM, BLOCK_SIZE>>>(d_array, BLOCK_SIZE * BLOCK_NUM);

    perfix_sum<<<BLOCK_NUM, BLOCK_SIZE>>>(d_array, 1, BLOCK_SIZE * BLOCK_NUM);
    cudaDeviceSynchronize();

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
