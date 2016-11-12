#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include "toneMapping.h"
#include "redEyeReduction.h"
#include "seamCarving.h"
#include "edgePreservingBlur.h"
#include "timer.h"

#define PROFILE 1
#define TONEMAPPING 1
#define SEAMCARVING 0
#define REDEYEREDUCTION 0
#define BLUR 0
#define CPU 0
#define GPU 1

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	#if TONEMAPPING
		Mat im = imread("../image/input1.jpg", CV_LOAD_IMAGE_COLOR | CV_LOAD_IMAGE_ANYDEPTH);
	#endif
	#if SEAMCARVING
		Mat im = imread("../image/input2.png");
	#endif
	#if BLUR
		Mat im = imread("../image/input4.jpg");
	#endif
	#if REDEYEREDUCTION
		Mat im = imread("../image/input3.jpg");
		Mat im2 = imread("../image/red_eye_template.jpg");
	#endif
	if (im.empty())	{
		cout << "Cannot open image!" << endl;
		return -1;
	}
	imshow("Input", im);

	#if PROFILE
		CpuTimer timerCPU;
		GpuTimer timerGPU;
	#endif
	
	//Tone Mapping
	#if TONEMAPPING
	#if CPU
	#if PROFILE
		timerCPU.Start();
	#endif
	im = toneMapping::cpuMap(im);
	#if PROFILE
		timerCPU.Stop();
		printf("Total CPU time: %f s.\n", timerCPU.Elapsed());
	#endif
	#endif
	#if GPU
	#if PROFILE
		timerGPU.Start();
	#endif
	im = toneMapping::gpuMap(im);
	#if PROFILE
		timerGPU.Stop();
		printf("Total GPU time: %f msecs.\n", timerGPU.Elapsed());
	#endif
	#endif
	#endif
	
	//Seam carving
	#if SEAMCARVING
	#if CPU
	#if PROFILE
		timerCPU.Start();
	#endif
		// cpuCarve(image, direction, number of seams),    0 -> horizontal, 1 -> vertical
		//im = seamCarving::cpuCarve(im, 0, 100);
		im = seamCarving::cpuCarve(im, 1, 100);
	#if PROFILE
		timerCPU.Stop();
		printf("Total CPU time: %f s.\n", timerCPU.Elapsed());
	#endif
	#endif
	#if GPU
	#if PROFILE
		timerGPU.Start();
	#endif
		// gpuCarve(image, direction, number of seams),    0 -> horizontal, 1 -> vertical
		//im = seamCarving::gpuCarve(im, 0, 10);
		im = seamCarving::gpuCarve(im, 1, 100);
	#if PROFILE
		timerGPU.Stop();
		printf("Total GPU time: %f msecs.\n", timerGPU.Elapsed());
	#endif
	#endif
	#endif

	//Edge Preserving Blur
	#if BLUR
	const float euclidean_delta = 1.0f;
	const int filter_radius = 5;
	#if CPU
	#if PROFILE
		timerCPU.Start();
	#endif
		im = edgePreservingBlur::cpuBlur(im, euclidean_delta, filter_radius);
	#if PROFILE
		timerCPU.Stop();
		printf("Total CPU time: %f s.\n", timerCPU.Elapsed());
	#endif
	#endif
	#if GPU
	#if PROFILE
		timerGPU.Start();
	#endif
		im = edgePreservingBlur::gpuBlur(im, euclidean_delta, filter_radius);
	#if PROFILE
		timerGPU.Stop();
		printf("Total GPU time: %f msecs.\n", timerGPU.Elapsed());
	#endif
	#endif
	#endif

	//Red Eye Reduction
	#if REDEYEREDUCTION
	#if CPU
	#if PROFILE
		timerCPU.Start();
	#endif
		im = redEyeReduction::cpuRER(im, im2);
	#if PROFILE
		timerCPU.Stop();
		printf("Total CPU time: %f s.\n", timerCPU.Elapsed());
	#endif
	#endif
	#if GPU
	#if PROFILE
		timerGPU.Start();
	#endif
		im = redEyeReduction::gpuRER(im, im2);
	#if PROFILE
		timerGPU.Stop();
		printf("Total GPU time: %f msecs.\n", timerGPU.Elapsed());
	#endif
	#endif
	#endif

	imshow("Output",im);
	
	waitKey(0);

	return 0;
}
