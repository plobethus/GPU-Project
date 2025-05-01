// matrix_generator_cuda.cu

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <curand.h>
#include <chrono>
#include <vector>
#include <iostream>
using namespace std;

// Error-checking macros
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t e = (call);                                               \
        if (e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(e));               \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

#define CURAND_CHECK(call)                                                    \
    do {                                                                      \
        curandStatus_t s = (call);                                            \
        if (s != CURAND_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "cuRAND error %s:%d: %d\n",                    \
                    __FILE__, __LINE__, s);                                   \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

// Kernel to scale uniform [0,1) floats -> [0,10)
__global__ void scale_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= 10.0f;
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <n>\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);
    size_t row_elems = (size_t)n;

    // Allocate device buffer for one row
    float* d_row;
    CUDA_CHECK(cudaMalloc(&d_row, row_elems * sizeof(float)));

    // Allocate host buffer
    vector<float> h_row(row_elems);

    // Setup cuRAND
    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    unsigned long seed = (unsigned long)
        chrono::high_resolution_clock::now()
        .time_since_epoch().count();
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));

    // Open output file
    FILE* f = fopen("Matrix.txt", "w");
    if (!f) { perror("fopen"); return 1; }
    // Large buffer to reduce flushes (16 MB)
    static char buf[1<<24];
    setvbuf(f, buf, _IOFBF, sizeof(buf));

    const int TPB = 256;
    int blocks = (n + TPB - 1) / TPB;

    // Generate and write each row
    for (int i = 0; i < n; ++i) {
        // 1) Generate uniform [0,1)
        CURAND_CHECK(curandGenerateUniform(gen, d_row, row_elems));
        // 2) Scale to [0,10)
        scale_kernel<<<blocks, TPB>>>(d_row, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 3) Copy back
        CUDA_CHECK(cudaMemcpy(h_row.data(), d_row,
                              row_elems * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // 4) Write ASCII row
        for (int j = 0; j < n; ++j) {
            fprintf(f, "%.6f", h_row[j]);
            if (j + 1 < n) fputc(' ', f);
        }
        // Delimit rows
        if (i + 1 < n)
            fputs(",\n", f);
        else
            fputs(".\n", f);
    }

    fclose(f);

    // Cleanup
    CURAND_CHECK(curandDestroyGenerator(gen));
    CUDA_CHECK(cudaFree(d_row));

    printf("Generated %d×%d matrix in Matrix.txt\n", n, n);
    return 0;
}