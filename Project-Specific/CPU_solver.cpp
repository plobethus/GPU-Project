#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <string>

using Clock = std::chrono::high_resolution_clock;
using namespace std;
typedef vector<vector<double>> Matrix;

void lu_decompose(const Matrix &A, Matrix &L, Matrix &U) {
    int n = A.size();
    L.assign(n, vector<double>(n, 0));
    U.assign(n, vector<double>(n, 0));
    for (int i = 0; i < n; i++) {
        // Upper
        for (int k = i; k < n; k++) {
            double sum = 0;
            for (int j = 0; j < i; j++)
                sum += L[i][j]*U[j][k];
            U[i][k] = A[i][k] - sum;
        }
        // Lower
        for (int k = i; k < n; k++) {
            if (i == k) {
                L[i][i] = 1.0;
            } else {
                double sum = 0;
                for (int j = 0; j < i; j++)
                    sum += L[k][j]*U[j][i];
                L[k][i] = (A[k][i] - sum) / U[i][i];
            }
        }
    }
}

vector<double> forward_sub(const Matrix &L, const vector<double> &b) {
    int n = b.size();
    vector<double> y(n);
    for (int i = 0; i < n; i++) {
        double sum = 0;
        for (int j = 0; j < i; j++)
            sum += L[i][j]*y[j];
        y[i] = b[i] - sum;
    }
    return y;
}

vector<double> backward_sub(const Matrix &U, const vector<double> &y) {
    int n = y.size();
    vector<double> x(n);
    for (int i = n-1; i >= 0; i--) {
        double sum = 0;
        for (int j = i+1; j < n; j++)
            sum += U[i][j]*x[j];
        x[i] = (y[i] - sum) / U[i][i];
    }
    return x;
}

int main() {
    auto t_start = Clock::now();

    // --- Read A from Matrix.txt ---
    Matrix A;
    ifstream inA("Matrix.txt");
    if (!inA) return 1;
    string line;
    while (getline(inA, line)) {
        stringstream ss(line);
        vector<double> row;
        double v;
        while (ss >> v) row.push_back(v);
        if (!row.empty()) A.push_back(row);
    }
    inA.close();

    int n = A.size();
    // --- Read b from b.txt ---
    vector<double> b(n);
    ifstream inB("b.txt");
    if (!inB) return 1;
    for (int i = 0; i < n; i++) inB >> b[i];
    inB.close();

    auto t_io_done = Clock::now();

    // --- LU solve ---
    Matrix L, U;
    lu_decompose(A, L, U);
    auto y = forward_sub(L, b);
    auto x = backward_sub(U, y);

    auto t_solve_done = Clock::now();

    // Compute durations
    auto ms_io    = std::chrono::duration_cast<std::chrono::milliseconds>(t_io_done    - t_start).count();
    auto ms_solve = std::chrono::duration_cast<std::chrono::milliseconds>(t_solve_done - t_io_done ).count();
    auto ms_total = std::chrono::duration_cast<std::chrono::milliseconds>(t_solve_done - t_start   ).count();

    // Print only timings
    cout << "IO    (ms): " << ms_io    << "\n";
    cout << "Solve (ms): " << ms_solve << "\n";
    cout << "Total (ms): " << ms_total << "\n";

    return 0;
}