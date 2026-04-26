#include "cuda_runtime.h"
#include <iostream>
#include <cstdlib>
#include <cmath>

constexpr int TILE_SIZE = 16;
constexpr int N = 1024;

__global__ void mm(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;

        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

__global__ void mm_tiled(float* A, float* B, float* C, int N) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int tiledCol = t * TILE_SIZE + threadIdx.x;
        int tiledRow = t * TILE_SIZE + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] = A[row * N + tiledCol];
        sB[threadIdx.y][threadIdx.x] = B[tiledRow * N + col];

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads();
    }

    C[row * N + col] = sum;
}

void cpu_mm(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;

            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }

            C[i * N + j] = sum;
        }
    }
}

int main() {
    size_t bytes = N * N * sizeof(float);

    float* h_A = new float[N * N];
    float* h_B = new float[N * N];
    float* h_C = new float[N * N];
    float* h_C_GPU = new float[N * N];

    float* d_A;
    float* d_B;
    float* d_C;

    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = static_cast<float>(i % 10);
        h_B[i] = static_cast<float>(i % 10);
        h_C[i] = 0.0f;
        h_C_GPU[i] = 0.0f;
    }

    cpu_mm(h_A, h_B, h_C, N);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 numThreads(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (N + TILE_SIZE - 1) / TILE_SIZE
    );

    mm_tiled << <numBlocks, numThreads >> > (d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C_GPU, d_C, bytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    const float eps = 1e-3f;

    for (int i = 0; i < N * N; i++) {
        if (std::fabs(h_C_GPU[i] - h_C[i]) > eps) errors++;
    }

    if (errors == 0) std::cout << "everything is ok\n";
    else             std::cout << "wrong results, errors: " << errors << "\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_GPU;

    return 0;
}
