#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

#define SIZE (1024LL * 1024 * 1024)        
#define CHUNK_SIZE (1024 * 1024 * 128)     
#define BLOCK_SIZE 1024

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

int main()
{
    int* chunk_a; int* chunk_b; int* chunk_c;
    int* d_A; int* d_B; int* d_C;

    size_t maxChunkBytes = CHUNK_SIZE * sizeof(int);

    chunk_a = (int*)malloc(maxChunkBytes);
    chunk_b = (int*)malloc(maxChunkBytes);
    chunk_c = (int*)malloc(maxChunkBytes);

    cudaMalloc((void**)&d_A, maxChunkBytes);
    cudaMalloc((void**)&d_B, maxChunkBytes);
    cudaMalloc((void**)&d_C, maxChunkBytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (long long offset = 0; offset < SIZE; offset += CHUNK_SIZE)
    {
        int currentChunkSize = (int)((SIZE - offset < CHUNK_SIZE)
            ? (SIZE - offset)
            : CHUNK_SIZE);

        size_t currentBytes = currentChunkSize * sizeof(int);

        randomNum(chunk_a, currentChunkSize);
        randomNum(chunk_b, currentChunkSize);

        cudaMemcpy(d_A, chunk_a, currentBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, chunk_b, currentBytes, cudaMemcpyHostToDevice);

        int numBlocks = (currentChunkSize + BLOCK_SIZE - 1) / BLOCK_SIZE;

        vectorAdd << <numBlocks, BLOCK_SIZE >> > (d_A, d_B, d_C, currentChunkSize);

        cudaMemcpy(chunk_c, d_C, currentBytes, cudaMemcpyDeviceToHost);

        printf("Processed chunk offset: %lld / %lld\n", offset, SIZE);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Total execution time: %f ms\n", milliseconds);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(chunk_a);
    free(chunk_b);
    free(chunk_c);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}