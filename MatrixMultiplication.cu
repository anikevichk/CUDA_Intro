#include "cuda_runtime.h"
#include <iostream>
#include <cstdlib>
#include <cmath>

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
    const int N = 512;
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

    int blockSize = 16;

    dim3 numThreads(blockSize, blockSize);
    dim3 numBlocks(
        (N + blockSize - 1) / blockSize,
        (N + blockSize - 1) / blockSize
    );

    mm << <numBlocks, numThreads >> > (d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C_GPU, d_C, bytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    const float eps = 1e-3f;

    for (int i = 0; i < N * N; i++) {
        if (std::fabs(h_C_GPU[i] - h_C[i]) > eps) {
            errors++;
        }
    }

    if (errors == 0)
        std::cout << "everything is ok\n";
    else
        std::cout << "wrong results, errors: " << errors << "\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_GPU;

    return 0;
}
