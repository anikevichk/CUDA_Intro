#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>

__global__ void reduce_safe(float* input, float* output, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int index = 2 * blockIdx.x * blockDim.x + tid;

    float sum = 0.0f;

    if (index < n) {
        sum += input[index];
    }

    if (index + blockDim.x < n) {
        sum += input[index + blockDim.x];
    }

    sdata[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 32; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid < 32) {
        sdata[tid] += sdata[tid + 32];
        sdata[tid] += sdata[tid + 16];
        sdata[tid] += sdata[tid + 8];
        sdata[tid] += sdata[tid + 4];
        sdata[tid] += sdata[tid + 2];
        sdata[tid] += sdata[tid + 1];
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

float cpu_reduce(float* input, int size) {
    float sum = 0.0f;

    for (int i = 0; i < size; ++i) {
        sum += input[i];
    }

    return sum;
}

int main() {
    int n = 1024 * 1024;
    size_t bytes = n * sizeof(float);

    float* h_input = new float[n];

    for (int i = 0; i < n; i++) {
        h_input[i] = static_cast<float>(i + 1);
    }

    float* d_input;
    float* d_output;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    int blockSize = 128;

    float total_sum = cpu_reduce(h_input, n);
    std::cout << "Total sum (CPU): " << total_sum << std::endl;

    int currentSize = n;

    while (currentSize > 1) {
        int gridSize = (currentSize + blockSize * 2 - 1) / (blockSize * 2);

        reduce_safe << <gridSize, blockSize, blockSize * sizeof(float) >> >
            (d_input, d_output, currentSize);

        cudaDeviceSynchronize();

        std::swap(d_input, d_output);

        currentSize = gridSize;
    }

    cudaMemcpy(h_input, d_input, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Final sum (GPU): " << h_input[0] << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;

    return 0;
}
