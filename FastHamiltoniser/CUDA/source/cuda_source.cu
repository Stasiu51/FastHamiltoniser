//#include "cuda_source.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

  // For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "error_utils.h"
#include <cuComplex.h>
#include <complex.h>

inline cuDoubleComplex operator + (cuDoubleComplex c1, cuDoubleComplex c2) { return cuCadd(c1, c2); }
inline cuDoubleComplex operator - (cuDoubleComplex c1, cuDoubleComplex c2) { return cuCsub(c1, c2); }
   
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

__global__ void vectorAddComplex(const cuDoubleComplex* A, const cuDoubleComplex* B, cuDoubleComplex* C, int numElements)
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
extern "C" {
    _Dcomplex* vector_add_gpu_complex(_Dcomplex* h_A_input, _Dcomplex* h_B_input, int numElements) {

        cuDoubleComplex* h_A = (cuDoubleComplex*)h_A_input;
        cuDoubleComplex* h_B = (cuDoubleComplex*)h_B_input;
        // Error code to check return values for CUDA calls
        cudaError_t err = cudaSuccess;

        size_t size = numElements * sizeof(cuDoubleComplex);

        printf("[Vector addition of %d (complex) elements]\n", numElements);

        // Allocate the host output vector C
        cuDoubleComplex* h_C = (cuDoubleComplex*)malloc(size);

        // Verify that allocations succeeded
        if (h_A == NULL || h_B == NULL || h_C == NULL)
        {
            fprintf(stderr, "Failed to allocate complex host vectors!\n");
            exit(EXIT_FAILURE);
        }

        // Allocate the device input vector A
        cuDoubleComplex* d_A = NULL;
        err = cudaMalloc((void**)&d_A, size);
        checkErr(err, "Allocating device vector A");

        // Allocate the device input vector B
        cuDoubleComplex* d_B = NULL;
        err = cudaMalloc((void**)&d_B, size);
        checkErr(err, "Allocate device vector B");

        // Allocate the device output vector C
        cuDoubleComplex* d_C = NULL;
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
        vectorAddComplex << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, numElements);
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
            if (cuCabs(cuCsub(cuCadd(h_A[i],h_B[i]), h_C[i])) > 1e-5)
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
        return (_Dcomplex*) h_C;
    }

    double* vector_add_gpu(double* h_A, double* h_B, int numElements) {

        _Dcomplex *a = new _Dcomplex { 1.0,2.0 };
        cuDoubleComplex *c = (cuDoubleComplex* )a;
        // Error code to check return values for CUDA calls
        cudaError_t err = cudaSuccess;
        double cr = cuCreal(*c);
        double ci = cuCimag(*c);
        printf("r,i: %f,%f\n", cr, ci);

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
}