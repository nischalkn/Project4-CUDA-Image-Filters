#include <stdio.h>
#include <cuda.h>
#include "toneMapping.h"
#include "cuda_runtime.h"

namespace toneMapping {
	Img cpuMap(Img in) {
		return in;
	}

	Img gpuMap(Img in) {
		return in;
	}
}
