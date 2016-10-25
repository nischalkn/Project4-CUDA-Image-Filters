#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include "toneMapping.h"

using namespace cv;
using namespace std;

Img readImg(Mat in) {
	int height = in.rows;
	int width = in.cols;
	Img data;
	vector<int> pixel;
	vector<vector<int>> row;

	cout << "Loading Image" << endl;
	for (int i = 0; i < width; i++) {
		row.clear();
		for (int j = 0; j < height; j++) {
			pixel.clear();
			pixel.push_back(in.at<cv::Vec3b>(j, i)[0]);
			pixel.push_back(in.at<cv::Vec3b>(j, i)[1]);
			pixel.push_back(in.at<cv::Vec3b>(j, i)[2]);
			row.push_back(pixel);
			//cout << "column: " << j << endl;
		}
		data.push_back(row);
		//cout << "row: " << i << endl;
	}
	return data;
}

void displayImg(Mat in, Img data) {
	int height = data[0].size();
	int width = data.size();
	//Mat out = in.clone();
	Mat out(height, width, CV_8UC3);
	cout << out.rows << ", " << out.cols << ", " << out.depth() << endl;
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			out.at<cv::Vec3b>(j, i)[0] = data[i][j][0];
			out.at<cv::Vec3b>(j, i)[1] = data[i][j][1];
			out.at<cv::Vec3b>(j, i)[2] = data[i][j][2];
		}
	}
	imshow("image", out);
}

int main(int argc, char** argv)
{
	Mat im = imread(argc == 2 ? argv[1] : "lena.jpg", 1);
	if (im.empty())	{
		cout << "Cannot open image!" << endl;
		return -1;
	}
	Img in,out;
	in = readImg(im);
	cout << "Computing Mapped Image" << endl;
	out = toneMapping::cpuMap(in);
	cout << "Displaying Image" << endl;
	displayImg(im,out);

	waitKey(0);

	return 0;
}
