#include <iostream>
#include <cuda_runtime.h>

__global__ void reduce_in_place(float* input, int n) {
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();

        if (tid % (2 * stride) == 0 && index + stride < n) {
            input[index] += input[index + stride];
        }
    }

    if (tid == 0) {
        input[blockIdx.x] = input[blockIdx.x * blockDim.x];
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
    float* d_input;

    for (int i = 0; i < n; i++) {
        h_input[i] = static_cast<float>(i + 1);
    }

    cudaMalloc(&d_input, bytes);
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    int blockSize = 256;

    float total_sum = cpu_reduce(h_input, n);
    std::cout << "Total sum (CPU): " << total_sum << std::endl;

    int currentSize = n;

    while (currentSize > 1) {
        int gridSize = (currentSize + blockSize - 1) / blockSize;

        reduce_in_place << <gridSize, blockSize >> > (d_input, currentSize);
        cudaDeviceSynchronize();

        currentSize = gridSize;
    }

    cudaMemcpy(h_input, d_input, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Final sum (GPU): " << h_input[0] << std::endl;

    cudaFree(d_input);
    delete[] h_input;

    return 0;
}