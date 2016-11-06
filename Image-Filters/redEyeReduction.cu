#include <stdio.h>
#include <cuda.h>
#include "redEyeReduction.h"
#include "cuda_runtime.h"

namespace redEyeReduction {
	int cpuMap(size_t rows, size_t cols, float *imgPtr) {
		return 1;
	}

	int gpuMap(size_t rows, size_t cols, float *imgPtr) {
		return 1;
	}
}
