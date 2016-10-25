#include <stdio.h>
#include <cuda.h>
#include "redEyeReduction.h"
#include "cuda_runtime.h"

namespace redEyeReduction {
	Img cpuRER(Img in) {
		return in;
	}

	Img gpuRER(Img in) {
		return in;
	}
}
