#include <stdio.h>
#include <stdlib.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

void checkErr (cudaError_t err, char const * msg){
	if (err != cudaSuccess) {
		fprintf(stderr, "Error during step: %s\nError string: %s\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}