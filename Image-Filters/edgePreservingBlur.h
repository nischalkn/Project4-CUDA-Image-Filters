#include <stdio.h>
#include <cuda.h>
#include <vector>
#include "cuda_runtime.h"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

inline float3 multiply(const float a, const float3 b)
{
	return{ a * b.x, a * b.y, a * b.z };
}

inline float3 add(const float3 a, const float3 b)
{
	return{ a.x + b.x, a.y + b.y, a.z + b.z };
}

namespace edgePreservingBlur {
	Mat cpuBlur(Mat im, const float euclidean_delta, const int cols, const int rows, const int filter_radius);
	Mat gpuBlur(Mat im, const float euclidean_delta, const int cols, const int rows, const int filter_radius);
}