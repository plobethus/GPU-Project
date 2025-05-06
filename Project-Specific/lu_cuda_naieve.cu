#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
using namespace std;
using Clock = std::chrono::high_resolution_clock;

// Error‐checking macro
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Flat indexing
__device__ inline int IDX(int row, int col, int n) {
    return row * n + col;
}

// Compute L[:,k]
__global__ void compute_L(const float* __restrict__ a,
                          float* __restrict__ l,
                          const float* __restrict__ u,
                          int k, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i   = tid + k;
    if (i < n) {
        float sum = 0.0f;
        for (int p = 0; p < k; ++p)
            sum += l[IDX(i,p,n)] * u[IDX(p,k,n)];
        l[IDX(i,k,n)] = a[IDX(i,k,n)] - sum;
    }
}

// Compute U[k,:]
__global__ void compute_U(const float* __restrict__ a,
                          const float* __restrict__ l,
                          float* __restrict__ u,
                          int k, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int j   = tid + k;
    if (j < n) {
        if (j == k) {
            u[IDX(k,k,n)] = 1.0f;
        } else {
            float sum = 0.0f;
            for (int p = 0; p < k; ++p)
                sum += l[IDX(k,p,n)] * u[IDX(p,j,n)];
            u[IDX(k,j,n)] = (a[IDX(k,j,n)] - sum) / l[IDX(k,k,n)];
        }
    }
}

int main() {
    // --- Start parse timing ---
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

    // Parse into flat A
    vector<float> h_a(n*n), h_l(n*n,0.0f), h_u(n*n,0.0f);
    for (int i = 0; i < n; ++i) {
        stringstream ss(rows[i]);
        for (int j = 0; j < n; ++j)
            ss >> h_a[i*n + j];
    }
    auto t_parse = Clock::now();

    // --- H2D upload timing ---
    auto t_h2d_start = Clock::now();
    float *d_a, *d_l, *d_u;
    size_t bytes = n * n * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_l, bytes));
    CUDA_CHECK(cudaMalloc(&d_u, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_l, 0, bytes));
    CUDA_CHECK(cudaMemset(d_u, 0, bytes));
    auto t_h2d = Clock::now();

    // --- GPU kernel timing ---
    cudaEvent_t gpu_start, gpu_end;
    CUDA_CHECK(cudaEventCreate(&gpu_start));
    CUDA_CHECK(cudaEventCreate(&gpu_end));
    CUDA_CHECK(cudaEventRecord(gpu_start));

    const int TPB = 256;
    for (int k = 0; k < n; ++k) {
        int len = n - k;
        int blocks = (len + TPB - 1) / TPB;
        compute_L<<<blocks,TPB>>>(d_a,d_l,d_u,k,n);
        compute_U<<<blocks,TPB>>>(d_a,d_l,d_u,k,n);
    }

    CUDA_CHECK(cudaEventRecord(gpu_end));
    CUDA_CHECK(cudaEventSynchronize(gpu_end));
    float t_gpu_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&t_gpu_ms, gpu_start, gpu_end));
    CUDA_CHECK(cudaEventDestroy(gpu_start));
    CUDA_CHECK(cudaEventDestroy(gpu_end));
    auto t_kernel = Clock::now();  // for host‐side total

    // --- D2H download timing ---
    auto t_d2h_start = Clock::now();
    CUDA_CHECK(cudaMemcpy(h_l.data(), d_l, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_u.data(), d_u, bytes, cudaMemcpyDeviceToHost));
    auto t_d2h = Clock::now();

    // --- Host forward/backward solve timing ---
    auto t_host_start = Clock::now();
    vector<float> y(n), x(n);
    // forward
    for (int i = 0; i < n; ++i) {
        float sum = 0;
        for (int j = 0; j < i; ++j)
            sum += h_l[i*n + j] * y[j];
        y[i] = (h_b[i] - sum) / h_l[i*n + i];
    }
    // backward
    for (int i = n-1; i >= 0; --i) {
        float sum = 0;
        for (int j = i+1; j < n; ++j)
            sum += h_u[i*n + j] * x[j];
        x[i] = y[i] - sum;
    }
    auto t_host = Clock::now();

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_l);
    cudaFree(d_u);

    // --- Compute all durations ---
    auto ms_parse    = chrono::duration_cast<chrono::milliseconds>(t_parse   - t0         ).count();
    auto ms_h2d      = chrono::duration_cast<chrono::milliseconds>(t_h2d     - t_h2d_start ).count();
    auto ms_kernel   = t_gpu_ms;  // already in ms
    auto ms_d2h      = chrono::duration_cast<chrono::milliseconds>(t_d2h     - t_d2h_start ).count();
    auto ms_host     = chrono::duration_cast<chrono::milliseconds>(t_host    - t_host_start).count();
    auto ms_total    = chrono::duration_cast<chrono::milliseconds>(t_host    - t0         ).count();

    // --- Print only timings ---
    printf("Parse          : %lld\n", (long long)ms_parse);
    printf("H2D upload     : %lld\n", (long long)ms_h2d);
    printf("GPU LU kernels : %.4f\n", ms_kernel);
    printf("D2H download   : %lld\n", (long long)ms_d2h);
    printf("Host solve     : %lld\n", (long long)ms_host);
    printf("Total          : %lld\n", (long long)ms_total);

    return 0;
}