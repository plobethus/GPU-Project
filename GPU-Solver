#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

using namespace std;

typedef vector<vector<double>> Matrix;

void print_matrix(const Matrix &A) {
    for (const auto &row : A) {
        for (double val : row)
            cout << val << " ";
        cout << endl;
    }
}

void lu_decompose(const Matrix &A, Matrix &L, Matrix &U) {
    int n = A.size();
    L = Matrix(n, vector<double>(n, 0));
    U = Matrix(n, vector<double>(n, 0));

    for (int i = 0; i < n; i++) {
        // Upper Triangular
        for (int k = i; k < n; k++) {
            double sum = 0;
            for (int j = 0; j < i; j++)
                sum += (L[i][j] * U[j][k]);
            U[i][k] = A[i][k] - sum;
        }

        // Lower Triangular
        for (int k = i; k < n; k++) {
            if (i == k)
                L[i][i] = 1; // Diagonal as 1
            else {
                double sum = 0;
                for (int j = 0; j < i; j++)
                    sum += (L[k][j] * U[j][i]);
                L[k][i] = (A[k][i] - sum) / U[i][i];
            }
        }
    }
}

vector<double> forward_substitution(const Matrix &L, const vector<double> &b) {
    int n = b.size();
    vector<double> y(n);
    for (int i = 0; i < n; i++) {
        double sum = 0;
        for (int j = 0; j < i; j++)
            sum += L[i][j] * y[j];
        y[i] = b[i] - sum;
    }
    return y;
}

vector<double> backward_substitution(const Matrix &U, const vector<double> &y) {
    int n = y.size();
    vector<double> x(n);
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0;
        for (int j = i + 1; j < n; j++)
            sum += U[i][j] * x[j];
        x[i] = (y[i] - sum) / U[i][i];
    }
    return x;
}

vector<double> solve_lu(const Matrix &A, const vector<double> &b) {
    Matrix L, U;
    lu_decompose(A, L, U);
    vector<double> y = forward_substitution(L, b);
    return backward_substitution(U, y);
}

int main() {
    Matrix A = {
        {4, 3},
        {6, 3}
    };
    vector<double> b = {10, 12};

    vector<double> x = solve_lu(A, b);

    cout << "Solution x: ";
    for (double val : x)
        cout << val << " ";
    cout << endl;

    return 0;
}