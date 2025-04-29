#include<iostream>
#include<cstdio>
 
using namespace std;
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
using namespace std;

void lu(float[][10], float[][10], float[][10], int n);
void output(float[][10], int);

int main(int argc, char **argv)
{
    float a[10][10], l[10][10], u[10][10];
    int n = 0, j = 0;
    ifstream infile("testMatrixes.txt");

    if (!infile.is_open()) {
        cout << "Error opening file!" << endl;
        return 1;
    }

    string line;
    while (getline(infile, line, ','))  // Read until comma
    {
        stringstream ss(line);
        j = 0;
        while (ss >> a[n][j]) { // Read numbers in the row
            j++;
        }
        n++;  // Move to next row after comma
    }
    infile.close();

    lu(a, l, u, n);

    cout << "\nL Decomposition\n\n";
    output(l, n);

    cout << "\nU Decomposition\n\n";
    output(u, n);

    return 0;
}

void lu(float a[][10], float l[][10], float u[][10], int n)
{
    int i = 0, j = 0, k = 0;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            if (j < i)
                l[j][i] = 0;
            else
            {
                l[j][i] = a[j][i];
                for (k = 0; k < i; k++)
                {
                    l[j][i] = l[j][i] - l[j][k] * u[k][i];
                }
            }
        }
        for (j = 0; j < n; j++)
        {
            if (j < i)
                u[i][j] = 0;
            else if (j == i)
                u[i][j] = 1;
            else
            {
                u[i][j] = a[i][j] / l[i][i];
                for (k = 0; k < i; k++)
                {
                    u[i][j] = u[i][j] - ((l[i][k] * u[k][j]) / l[i][i]);
                }
            }
        }
    }
}
void output(float x[][10], int n)
{
    int i = 0, j = 0;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            printf("%f ", x[i][j]);
        }
        cout << "\n";
    }
}