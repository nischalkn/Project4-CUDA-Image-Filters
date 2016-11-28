#include <stdio.h>
#include <cuda.h>
#include <vector>
#include "cuda_runtime.h"
#include <iostream>
#include <algorithm>
#include "opencv2/highgui/highgui.hpp"
#include "toneMapping.h"

using namespace std;
using namespace cv;

namespace redEyeReduction {
	Mat cpuRER(Mat im, Mat eyeTemplate);
	Mat gpuRER(Mat im, Mat eyeTemplate);
}