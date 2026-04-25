#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

#define SIZE (1024LL * 1024 * 1024)
#define CHUNK_SIZE (1024 * 1024 * 128)
#define BLOCK_SIZE 256

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = call;                                        \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(err));      \
        exit(EXIT_FAILURE);                                        \
    }                                                              \
} while (0)

#define HOST_CHECK(ptr) do {                                       \
    if ((ptr) == NULL) {                                           \
        fprintf(stderr, "Host malloc failed at %s:%d\n",           \
                __FILE__, __LINE__);                               \
        exit(EXIT_FAILURE);                                        \
    }                                                              \
} while (0)

__global__ void vectorAdd(int* A, int* B, int* C, int n){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n){
        C[index] = A[index] + B[index];
    }
}

void randomNum(int* x, int size){
    for (int i = 0; i < size; i++){
        x[i] = rand() % 100;
    }
}

int main(){
    int* chunk_a; int* chunk_b; int* chunk_c;
    int* d_A; int* d_B; int* d_C;

    size_t maxChunkBytes = CHUNK_SIZE * sizeof(int);

    chunk_a = (int*)malloc(maxChunkBytes);
    chunk_b = (int*)malloc(maxChunkBytes);
    chunk_c = (int*)malloc(maxChunkBytes);

    HOST_CHECK(chunk_a);
    HOST_CHECK(chunk_b);
    HOST_CHECK(chunk_c);

    CUDA_CHECK(cudaMalloc((void**)&d_A, maxChunkBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, maxChunkBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, maxChunkBytes));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (long long offset = 0; offset < SIZE; offset += CHUNK_SIZE){
        int currentChunkSize = (int)((SIZE - offset < CHUNK_SIZE)
            ? (SIZE - offset)
            : CHUNK_SIZE);

        size_t currentBytes = currentChunkSize * sizeof(int);

        randomNum(chunk_a, currentChunkSize);
        randomNum(chunk_b, currentChunkSize);

        CUDA_CHECK(cudaMemcpy(d_A, chunk_a, currentBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, chunk_b, currentBytes, cudaMemcpyHostToDevice));

        int numBlocks = (currentChunkSize + BLOCK_SIZE - 1) / BLOCK_SIZE;

        vectorAdd<<<numBlocks, BLOCK_SIZE>>>(d_A, d_B, d_C, currentChunkSize);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(chunk_c, d_C, currentBytes, cudaMemcpyDeviceToHost));

        printf("Processed chunk offset: %lld / %lld\n", offset, SIZE);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    printf("Total execution time: %f ms\n", milliseconds);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    free(chunk_a);
    free(chunk_b);
    free(chunk_c);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}