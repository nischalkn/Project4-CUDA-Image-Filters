#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include "toneMapping.h"
#include "redEyeReduction.h"
#include "seamCarving.h"

using namespace cv;
using namespace std;

typedef vector<vector<float>> vec2d;

void readImg(Mat in, size_t *numRows, size_t *numCols, float **imgPtr) {
	
	if (in.type() != CV_32FC3){
		in.convertTo(in, CV_32FC3);
		cout << "Converted" << endl;
	}
	if (!in.isContinuous()) {
		std::cerr << "Image isn't continuous!" << std::endl;
		exit(1);
	}

	*imgPtr = new float[in.rows * in.cols * in.channels()];

	float *cvPtr = in.ptr<float>(0);
	for (size_t i = 0; i < in.rows * in.cols * in.channels(); ++i) {
		(*imgPtr)[i] = cvPtr[i];
	}

	*numRows = in.rows;
	*numCols = in.cols;
}

void displayImg(String file_name, size_t *numRows, size_t *numCols, float *imgPtr) {

	int sizes[2];
	sizes[0] = *numRows;
	sizes[1] = *numCols;
	cv::Mat imageHDR(2, sizes, CV_32FC3, (void *)imgPtr);
	imshow(file_name, imageHDR);
	imageHDR = imageHDR * 255;
	cv::imwrite(file_name+".png", imageHDR);

}

int main(int argc, char** argv)
{
//	Mat im = imread("lena.jpg", CV_LOAD_IMAGE_COLOR | CV_LOAD_IMAGE_ANYDEPTH);
	Mat im = imread("surfing.png");
	if (im.empty())	{
		cout << "Cannot open image!" << endl;
		return -1;
	}
	imshow("original", im);
	size_t numRows;
	size_t numCols;
	float *imgPtr;
	
	readImg(im, &numRows, &numCols, &imgPtr);
	size_t size = im.rows * im.cols * im.channels();
	float *outImg = new float[size];
	for (size_t i = 0; i < size; i++)
		outImg[i] = imgPtr[i];

	/*toneMapping::cpuMap(numRows, numCols, outImg);
	displayImg("cpu", &numRows, &numCols, outImg);
	toneMapping::gpuMap(numRows, numCols, imgPtr);
	displayImg("gpu", &numRows, &numCols, imgPtr);*/
	
	//im = seamCarving::cpuCarve(im, 0, 100);
	//im = seamCarving::cpuCarve(im, 1, 100);
	//im = seamCarving::gpuCarve(im, 0, 100);
	//im = seamCarving::gpuCarve(im, 1, 100);
	imshow("cropped",im);
	
	waitKey(0);

	return 0;
}
