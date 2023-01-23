#pragma once
#include <cuda_runtime.h>
extern "C" {
	void checkErr(cudaError_t err, char const* msg);
}