#include <stdio.h>
#include <cuda.h>
#include <vector>
#include "cuda_runtime.h"
#include <iostream>
#include <algorithm>

using namespace std;

namespace toneMapping {
	int cpuMap(size_t rows, size_t cols, float *imgPtr);
	int gpuMap(size_t rows, size_t cols, float *imgPtr);;
}