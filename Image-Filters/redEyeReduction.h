#include <stdio.h>
#include <cuda.h>
#include <vector>
#include "cuda_runtime.h"

using namespace std;
typedef vector<vector<vector<int>>> Img;

namespace redEyeReduction {
	int cpuMap(size_t rows, size_t cols, float *imgPtr);
	int gpuMap(size_t rows, size_t cols, float *imgPtr);
}