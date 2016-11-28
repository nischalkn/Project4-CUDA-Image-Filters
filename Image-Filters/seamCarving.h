#include <stdio.h>
#include <cuda.h>
#include <vector>
#include "cuda_runtime.h"
#include <iostream>
#include <algorithm>
#include "opencv2/highgui/highgui.hpp"
#include "timer.h"

using namespace std;
using namespace cv;

typedef unsigned int uint;

inline int ilog2(int x) {
	int lg = 0;
	while (x >>= 1) {
		++lg;
	}
	return lg;
}

inline int ilog2ceil(int x) {
	return ilog2(x - 1) + 1;
}

namespace seamCarving {
	Mat cpuCarve(cv::Mat, int, int);
	Mat gpuCarve(cv::Mat, int, int);
}