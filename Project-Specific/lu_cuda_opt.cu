
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>

using namespace std;
using Clock = chrono::high_resolution_clock;

// Error‐checking macro
auto gpuAssert = [](cudaError_t code, const char *file, int line, bool abort=true){
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
};
#define CUDA_CHECK(ans) gpuAssert((ans), __FILE__, __LINE__)

// Tile size
#define B 32

// Panel factorization kernel: in-place LU of the B×B block A[k..k+B-1][k..k+B-1]
__global__ void panel_factor(float *A, int n, int k) {
    __shared__ float sA[B][B];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = k + ty;
    int col = k + tx;

    // Load block into shared memory
    if (row < n && col < n)
        sA[ty][tx] = A[row * n + col];
    else
        sA[ty][tx] = 0.0f;
    __syncthreads();

    // Doolittle in shared memory (L diag = 1 implicit)
    for (int j = 0; j < B; ++j) {
        if (ty > j && tx == j)
            sA[ty][j] /= sA[j][j];
        __syncthreads();
        if (ty > j && tx > j)
            sA[ty][tx] -= sA[ty][j] * sA[j][tx];
        __syncthreads();
    }

    // Write back
    if (row < n && col < n)
        A[row * n + col] = sA[ty][tx];
}

int main() {
    // --- parse timing ---
    auto t0 = Clock::now();

    // Read Matrix.txt (rows separated by commas)
    ifstream infile("Matrix.txt");
    if (!infile.is_open()) return 1;
    vector<string> rows;
    string line;
    while (getline(infile, line, ',')) rows.push_back(line);
    infile.close();
    int n = rows.size();

    // Read b.txt
    vector<float> h_b(n);
    ifstream bfile("b.txt");
    if (!bfile.is_open()) return 1;
    for (int i = 0; i < n; ++i) bfile >> h_b[i];
    bfile.close();

    // Parse into flat h_A
    vector<float> h_A(n*n);
    for (int i = 0; i < n; ++i) {
        stringstream ss(rows[i]);
        for (int j = 0; j < n; ++j)
            ss >> h_A[i * n + j];
    }
    auto t_parse = Clock::now();

    // --- H2D upload timing ---
    auto t_h2d_start = Clock::now();
    float *d_A;
    size_t bytes = n * n * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    auto t_h2d = Clock::now();

    // --- GPU kernel timing ---
    cudaEvent_t gpu_start, gpu_end;
    CUDA_CHECK(cudaEventCreate(&gpu_start));
    CUDA_CHECK(cudaEventCreate(&gpu_end));
    CUDA_CHECK(cudaEventRecord(gpu_start));

    // cuBLAS setup
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float one = 1.0f, minus1 = -1.0f;

    // Blocked LU
    for (int k = 0; k < n; k += B) {
        int nb = min(B, n - k);

        // Panel factorization
        dim3 threads(nb, nb);
        panel_factor<<<1, threads>>>(d_A, n, k);
        CUDA_CHECK(cudaGetLastError());

        int m = n - (k + nb);
        if (m <= 0) continue;

        // Compute U tiles: inv(L_kk) * A[k..k+nb][k+nb..]
        cublasSetStream(handle, 0);
        cublasStrsm(handle,
                    CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_N,      CUBLAS_DIAG_NON_UNIT,
                    nb, m,
                    &one,
                    d_A + k*n + k, n,
                    d_A + k*n + (k+nb), n);

        // Compute L tiles: A[k+nb..][k..k+nb] * inv(U_kk)
        cublasStrsm(handle,
                    CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                    CUBLAS_OP_N,       CUBLAS_DIAG_UNIT,
                    m, nb,
                    &one,
                    d_A + k*n + k, n,
                    d_A + (k+nb)*n + k, n);

        // Trailing update: A -= L * U
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    m, m, nb,
                    &minus1,
                    d_A + (k+nb)*n + k, n,
                    d_A + k*n + (k+nb), n,
                    &one,
                    d_A + (k+nb)*n + (k+nb), n);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(gpu_end));
    CUDA_CHECK(cudaEventSynchronize(gpu_end));
    float ms_kernel = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_kernel, gpu_start, gpu_end));
    CUDA_CHECK(cudaEventDestroy(gpu_start));
    CUDA_CHECK(cudaEventDestroy(gpu_end));
    auto t_kernel = Clock::now();

    // --- D2H download timing ---
    auto t_d2h_start = Clock::now();
    CUDA_CHECK(cudaMemcpy(h_A.data(), d_A, bytes, cudaMemcpyDeviceToHost));
    auto t_d2h = Clock::now();

    // --- Host forward/backward solve timing ---
    auto t_host_start = Clock::now();
    vector<float> y(n), x(n);
    // forward (L diag = 1)
    for (int i = 0; i < n; ++i) {
        float sum = 0;
        for (int j = 0; j < i; ++j)
            sum += h_A[i*n + j] * y[j];
        y[i] = h_b[i] - sum;
    }
    // backward
    for (int i = n-1; i >= 0; --i) {
        float sum = 0;
        for (int j = i+1; j < n; ++j)
            sum += h_A[i*n + j] * x[j];
        x[i] = (y[i] - sum) / h_A[i*n + i];
    }
    auto t_host = Clock::now();

    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_A);

    // --- Print timings ---
    auto ms_parse = chrono::duration_cast<chrono::milliseconds>(t_parse - t0).count();
    auto ms_h2d   = chrono::duration_cast<chrono::milliseconds>(t_h2d   - t_h2d_start).count();
    auto ms_d2h   = chrono::duration_cast<chrono::milliseconds>(t_d2h   - t_d2h_start).count();
    auto ms_host  = chrono::duration_cast<chrono::milliseconds>(t_host  - t_host_start).count();
    auto ms_total = chrono::duration_cast<chrono::milliseconds>(t_host  - t0).count();

    printf("Parse          : %lld\n", (long long)ms_parse);
    printf("H2D upload     : %lld\n", (long long)ms_h2d);
    printf("GPU LU kernels : %.4f\n", ms_kernel);
    printf("D2H download   : %lld\n", (long long)ms_d2h);
    printf("Host solve     : %lld\n", (long long)ms_host);
    printf("Total          : %lld\n", (long long)ms_total);

    return 0;
}
