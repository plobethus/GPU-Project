// lu_cuda_extended.CU

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cuda_runtime.h>
using namespace std;

// ——— error‐checking macro ———
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// flat indexing: row-major
__device__ inline int IDX(int row, int col, int n) {
    return row * n + col;
}

// Kernel to compute L[i][k] for i = k..n-1
__global__ void compute_L(const float* __restrict__ a,
                          float* __restrict__ l,
                          const float* __restrict__ u,
                          int k, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i   = tid + k;
    if (i < n) {
        float sum = 0.0f;
        for (int p = 0; p < k; ++p) {
            sum += l[IDX(i,p,n)] * u[IDX(p,k,n)];
        }
        l[IDX(i,k,n)] = a[IDX(i,k,n)] - sum;
    }
}

// Kernel to compute U[k][j] for j = k..n-1
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
            for (int p = 0; p < k; ++p) {
                sum += l[IDX(k,p,n)] * u[IDX(p,j,n)];
            }
            u[IDX(k,j,n)] = (a[IDX(k,j,n)] - sum) / l[IDX(k,k,n)];
        }
    }
}

int main() {
    const char* filename = "Matrix.txt";
    ifstream infile(filename);
    if (!infile.is_open()) {
        cerr << "Error opening " << filename << "\n";
        return 1;
    }

    // --- Read A into vector<string> for rows ---
    vector<string> rows;
    string line;
    while (getline(infile, line, ',')) {
        rows.push_back(line);
    }
    infile.close();
    int n = (int)rows.size();

    // --- Read b vector from b.txt ---
    vector<float> h_b(n);
    ifstream bfile("b.txt");
    if (!bfile.is_open()) {
        cerr << "Error opening b.txt\n";
        return 1;
    }
    for (int i = 0; i < n; ++i) {
        bfile >> h_b[i];
    }
    bfile.close();

    // --- Dynamic host storage for A, L, U ---
    vector<float> h_a(n*n), h_l(n*n,0.0f), h_u(n*n,0.0f);
    for (int i = 0; i < n; ++i) {
        stringstream ss(rows[i]);
        for (int j = 0; j < n; ++j) {
            ss >> h_a[i*n + j];
        }
    }

    // Device buffers
    float *d_a, *d_l, *d_u;
    size_t bytes = n * n * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_l, bytes));
    CUDA_CHECK(cudaMalloc(&d_u, bytes));

    // Copy A, zero L and U
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_l, 0, bytes));
    CUDA_CHECK(cudaMemset(d_u, 0, bytes));

    // --- LU decomposition ---
    const int TPB = 256;
    for (int k = 0; k < n; ++k) {
        int len    = n - k;
        int blocks = (len + TPB - 1) / TPB;
        compute_L<<<blocks,TPB>>>(d_a,d_l,d_u,k,n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        compute_U<<<blocks,TPB>>>(d_a,d_l,d_u,k,n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Copy back L, U to host
    CUDA_CHECK(cudaMemcpy(h_l.data(), d_l, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_u.data(), d_u, bytes, cudaMemcpyDeviceToHost));

    // Forward substitution: solve L*y = b
    vector<float> y(n);
    for (int i = 0; i < n; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < i; ++j)
            sum += h_l[i*n + j] * y[j];
        y[i] = (h_b[i] - sum) / h_l[i*n + i];
    }
    // Backward substitution: solve U*x = y
    vector<float> x(n);
    for (int i = n-1; i >= 0; --i) {
        float sum = 0.0f;
        for (int j = i+1; j < n; ++j)
            sum += h_u[i*n + j] * x[j];
        // U diagonal is 1
        x[i] = y[i] - sum;
    }

    // Print solution
    cout << "\nSolution x:\n";
    for (int i = 0; i < n; ++i)
        printf("%10.6f\n", x[i]);

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_l);
    cudaFree(d_u);
    return 0;
}
