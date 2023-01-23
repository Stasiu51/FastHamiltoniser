/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

 /**
  * Vector addition: C = A + B.
  *
  * This sample is a very basic sample that implements element by element
  * vector addition. It is the same as the sample illustrating Chapter 2
  * of the programming guide with some additions like error checking.
  */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

  // For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "error_utils.h"
/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void vectorAdd(const double* A, const double* B, double* C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

/**
 * Host main routine
 */

double* vector_add_gpu(double* h_A, double* h_B, int numElements) {
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    size_t size = numElements * sizeof(double);

    printf("[Vector addition of %d elements]\n", numElements);

    // Allocate the host output vector C
    double* h_C = (double*)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector A
    double* d_A = NULL;
    err = cudaMalloc((void**)&d_A, size);
    checkErr(err, "Allocating device vector A");

    // Allocate the device input vector B
    double* d_B = NULL;
    err = cudaMalloc((void**)&d_B, size);
    checkErr(err, "Allocate device vector B");

    // Allocate the device output vector C
    double* d_C = NULL;
    err = cudaMalloc((void**)&d_C, size);
    checkErr(err, "Allocate device vector C");

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    checkErr(err, "Copy A host to device");

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    checkErr(err, "Copy B host to device");

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    vectorAdd << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, numElements);
    err = cudaGetLastError();
    checkErr(err, "Launch vectorAdd kernel");

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    checkErr(err, "Copy C device to host");

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    // Free device global memory
    err = cudaFree(d_A);
    checkErr(err, "Free device vector A");

    err = cudaFree(d_B);
    checkErr(err, "Free device vector B");

    err = cudaFree(d_C);
    checkErr(err, "Free device vector C");

    printf("Done\n");
    return h_C;
}
