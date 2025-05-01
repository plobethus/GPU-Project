// matrix_generator.cpp

#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <iomanip>
#include <string>
using namespace std;

int main(int argc, char** argv) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <size>\n";
        return 1;
    }
    int n = stoi(argv[1]);

    // Always write to Matrix.txt
    ofstream ofs("Matrix.txt");
    if (!ofs) {
        cerr << "Error: could not open Matrix.txt for writing\n";
        return 1;
    }

    // Random number generator seeded by current time
    mt19937 gen((unsigned)chrono::system_clock::now().time_since_epoch().count());
    uniform_real_distribution<float> dist(0.0f, 10.0f);

    // Generate and write an n x n matrix
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            ofs << fixed << setprecision(6) << dist(gen);
            if (j + 1 < n)
                ofs << ' ';
        }
        if (i + 1 < n)
            ofs << ",\n";    
        else
            ofs << ".\n";  
    }

    cout << "Generated " << n << "x" << n << " matrix to Matrix.txt\n";
    return 0;
}
