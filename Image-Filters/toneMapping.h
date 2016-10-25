#include <stdio.h>
#include <cuda.h>
#include <vector>
#include "cuda_runtime.h"

using namespace std;
typedef vector<vector<vector<int>>> Img;

namespace toneMapping {
	Img cpuMap(Img in);
	Img gpuMap(Img in);
}