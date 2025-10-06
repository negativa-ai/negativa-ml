#include <stdio.h>
#include <iostream>
#include <sstream>

#include <cuda_runtime.h>

#include "demo.h"

__global__ void matrixMulGPU(int *a, int *b, int *c, int m, int n, int p)
{
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling,
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   *
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  /// BEGIN YOUR SOLUTION
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;
  size_t size = m * p;
  for (; gid < size; gid += stride)
  {
    c[gid] = 0;
    size_t i = gid / p;
    size_t j = gid % p;
    for (int k = 0; k < n; k++)
    {
      int a_ik = a[i * n + k];
      int b_kj = b[k * p + j];
      c[gid] += a_ik * b_kj;
    }
  }
  /// END YOUR SOLUTION
}

void matMul(int *a, int *b, int *c, int m, int n, int p)
{

  int size_a = m * n * sizeof(int); // Number of bytes of an M x N matrix
  int size_b = n * p * sizeof(int); // Number of bytes of an N x P matrix
  int size_c = m * p * sizeof(int); // Number of bytes of an M x P matrix

  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, size_a);
  cudaMalloc(&d_b, size_b);
  cudaMalloc(&d_c, size_c);

  cudaMemcpy(d_a, a, size_a, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size_b, cudaMemcpyHostToDevice);

  /*
   * Assign `threads_per_block` and `number_of_blocks` 2D values
   * that can be used in matrixMulGPU above.
   */

  dim3 threads_per_block = 1024;
  dim3 number_of_blocks = 32;

  matrixMulGPU<<<number_of_blocks, threads_per_block, 0>>>(d_a, d_b, d_c, m, n, p);

  cudaMemcpy(c, d_c, size_c, cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();
  // Free all our allocated memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

__global__ void setScalarItemGPU(int scalar, int *output, int height, int width)
{

  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;
  size_t size = height * width;
  for (; gid < size; gid += stride)
  {
    output[gid] = scalar;
  }
}

void setScalarItems(int scalar, int *output, int height, int width)
{
  int size_output = height * width * sizeof(int);

  int *d_output;
  cudaMalloc(&d_output, size_output);

  dim3 threads_per_block = 1024;
  dim3 number_of_blocks = 32;

  setScalarItemGPU<<<number_of_blocks, threads_per_block, 0>>>(scalar, d_output, height, width);

  cudaMemcpy(output, d_output, size_output, cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();
  // Free all our allocated memory
  cudaFree(d_output);
}