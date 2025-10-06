#include <stdio.h>
#include <iostream>
#include <sstream>

#include "demo.h"

#define M 256
#define N 256
#define P 256

void matrixMulCPU(int *a, int *b, int *c, int m, int n, int p)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < p; j++)
        {
            c[i * p + j] = 0;
            for (int k = 0; k < n; k++)
            {
                int a_ik = a[i * n + k];
                int b_kj = b[k * p + j];
                c[i * p + j] += a_ik * b_kj;
            }
        }
    }
}

void checkMatMul()
{
    int *a, *b, *c_cpu, *c_gpu; // Allocate a solution matrix for both the CPU and the GPU operations

    int size_a = M * N * sizeof(int); // Number of bytes of an M x N matrix
    int size_b = N * P * sizeof(int); // Number of bytes of an N x P matrix
    int size_c = M * P * sizeof(int); // Number of bytes of an M x P matrix

    // Allocate memory in host
    a = (int *)malloc(size_a);
    b = (int *)malloc(size_b);
    c_cpu = (int *)malloc(size_c);
    c_gpu = (int *)malloc(size_c);

    // Initialize memory; create 2D matrices
    for (int row = 0; row < M; ++row)
        for (int col = 0; col < N; ++col)
        {
            a[row * N + col] = row;
        }
    for (int row = 0; row < N; ++row)
        for (int col = 0; col < P; ++col)
        {
            b[row * P + col] = col + 2;
        }
    for (int row = 0; row < M; ++row)
        for (int col = 0; col < P; ++col)
        {
            c_cpu[row * P + col] = 0;
            c_gpu[row * P + col] = 0;
        }

    matMul(a, b, c_gpu, M, N, P);
    // setScalarItems(1, a, c_cpu, M, N);

    // Call the CPU version to check our work
    matrixMulCPU(a, b, c_cpu, M, N, P);

    // // Compare the two answers to make sure they are equal
    bool error = false;
    for (int row = 0; row < M && !error; ++row)
        for (int col = 0; col < P && !error; ++col)
            if (c_cpu[row * P + col] != c_gpu[row * P + col])
            {
                printf("FOUND ERROR at c[%d][%d]\n", row, col);
                error = true;
                break;
            }
    if (!error)
        printf("Success!\n");

    // Free all our allocated memory
    free(a);
    free(b);
    free(c_cpu);
    free(c_gpu);
}

void checkSetScalarItems()
{
    int *a;

    int size_a = M * N * sizeof(int); // Number of bytes of an M x N matrix

    // Allocate memory in host
    a = (int *)malloc(size_a);

    int scalar = 42;
    setScalarItems(scalar, a, M, N);

    // // Compare the two answers to make sure they are equal
    bool error = false;
    for (int row = 0; row < M && !error; ++row)
        for (int col = 0; col < N && !error; ++col)
            if (scalar != a[row * N + col])
            {
                printf("FOUND ERROR at c[%d][%d]\n", row, col);
                error = true;
                break;
            }
    if (!error)
        printf("Success!\n");

    // Free all our allocated memory
    free(a);
}

int main(int argc, char **argv)
{
    // call checkMatMul() or checkSetScalarItems() according to argument
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " <matmul|setscalar>" << std::endl;
        return 1;
    }
    std::string arg = argv[1];
    if (arg == "matmul")
    {
        checkMatMul();
    }
    else if (arg == "setscalar")
    {
        checkSetScalarItems();
    }
    else
    {
        std::cout << "Unknown argument: " << arg << std::endl;
        return 1;
    }
}
