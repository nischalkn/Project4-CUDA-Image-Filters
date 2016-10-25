#include <stdio.h>
#include <cuda.h>
#include <vector>
#include "cuda_runtime.h"

using namespace std;
typedef vector<vector<vector<int>>> Img;

namespace redEyeReduction {
	Img cpuRER(Img in);
	Img gpuRER(Img in);
}