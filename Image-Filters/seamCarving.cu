#include <stdio.h>
#include <cuda.h>
#include "seamCarving.h"
#include "cuda_runtime.h"

#define BLOCK_SIZE 128
const dim3 blockSize(32, 16, 1);

namespace seamCarving {

	void computeFullEnergyGPU(size_t rows, size_t cols, unsigned char *imgPtr, unsigned int *energy) {

		for (size_t i = 1; i < rows-1; i++) {
			for (size_t j = 1; j < (cols-1); j++)
			{
				int val = 0;

				val += pow(imgPtr[((i - 1)*cols * 3) + (3 * j)] - imgPtr[((i + 1)*cols * 3) + (3 * j)], 2);
				val += pow(imgPtr[((i - 1)*cols * 3) + ((3 * j) + 1)] - imgPtr[((i + 1)*cols * 3) + ((3 * j) + 1)], 2);
				val += pow(imgPtr[((i - 1)*cols * 3) + ((3 * j) + 2)] - imgPtr[((i + 1)*cols * 3) + ((3 * j) + 2)], 2);

				val += pow(imgPtr[(i * cols * 3) + ((3 * j) + 3)] - imgPtr[(i * cols * 3) + ((3 * j) - 3)], 2);
				val += pow(imgPtr[(i * cols * 3) + ((3 * j) + 4)] - imgPtr[(i * cols * 3) + ((3 * j) - 2)], 2);
				val += pow(imgPtr[(i * cols * 3) + ((3 * j) + 5)] - imgPtr[(i * cols * 3) + ((3 * j) - 1)], 2);

				energy[i*cols + j] = val;

			}
		}
	}

	void computeFullEnergy(Mat im, unsigned int *ene) {
		//Ensure that the size of the energy matrix matches that of the image
		//energy = Mat(image.rows, image.cols, CV_32S, Scalar(195075));
		Mat energy(im.rows, im.cols, CV_32S, Scalar(195075));

		//Scan through the image and update the energy values. Ignore boundary pixels.
		for (int i = 1; i < im.rows - 1; ++i) {
			uchar* prev = im.ptr<uchar>(i - 1);	//Pointer to previous row
			uchar* curr = im.ptr<uchar>(i);		//Pointer to current row
			uchar* next = im.ptr<uchar>(i + 1);	//Pointer to next row

			for (int j = 1; j < im.cols - 1; ++j) {
				int val = 0;
				//Energy along the x-axis
				val += (prev[3 * j] - next[3 * j]) * (prev[3 * j] - next[3 * j]);
				val += (prev[3 * j + 1] - next[3 * j + 1]) * (prev[3 * j + 1] - next[3 * j + 1]);
				val += (prev[3 * j + 2] - next[3 * j + 2]) * (prev[3 * j + 2] - next[3 * j + 2]);

				//Energy along the y-axis
				val += (curr[3 * j + 3] - curr[3 * j - 3]) * (curr[3 * j + 3] - curr[3 * j - 3]);
				val += (curr[3 * j + 4] - curr[3 * j - 2]) * (curr[3 * j + 4] - curr[3 * j - 2]);
				val += (curr[3 * j + 5] - curr[3 * j - 1]) * (curr[3 * j + 5] - curr[3 * j - 1]);

				energy.at<unsigned int>(i, j) = val;
			}
		}
		/*for (size_t i = 0; i < im.rows; i++)
		{
			for (size_t j = 0; j < im.cols; j++)
			{
				cout << i*im.rows + j << ", " << energy.at<unsigned int>(i, j) << endl;
			}
		}*/
		for (size_t i = 0; i < im.rows; i++) {
			for (size_t j = 0; j < im.cols; j++) {
				ene[(i*im.cols) + j] = (int)energy.at<unsigned int>(i, j);
				cout << i*im.cols + j << ", " << ene[i*im.cols + j] << endl;
			}
		}				
	}

	unsigned int getEnergy(unsigned int *energy, unsigned int row, unsigned int col, size_t cols) {
		cout << row << ", " << col << ", " << row*cols + col << endl;
		return energy[row*cols+col];
	}

	vector<uint> findVerticalSeam(unsigned int *energy, size_t rows, size_t cols) {
		vector<uint> seam(rows);
		unsigned int** distTo = new unsigned int*[rows];
		short** edgeTo = new short*[rows];
		for (int i = 0; i < rows; ++i) {
			distTo[i] = new unsigned int[cols];
			edgeTo[i] = new short[cols];
		}

		//Initialize the distance and edge matrices
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				if (i == 0)		distTo[i][j] = 0;
				else			distTo[i][j] = numeric_limits<unsigned int>::max();
				edgeTo[i][j] = 0;
			}
		}

		// Relax the edges in topological order
		for (int row = 0; row < rows - 1; ++row) {
			for (int col = 0; col < cols; ++col) {
				//Check the pixel to the bottom-left
				if (col != 0)
					if (distTo[row + 1][col - 1] > distTo[row][col] + energy[(row + 1) * cols + (col - 1)]) {// getEnergy(energy, row + 1, col - 1, cols)) {
						distTo[row + 1][col - 1] = distTo[row][col] + energy[(row + 1) * cols + (col - 1)];
						edgeTo[row + 1][col - 1] = 1;
					}
				//Check the pixel right below
				if (distTo[row + 1][col] > distTo[row][col] + energy[(row + 1)*cols + col]) {
					distTo[row + 1][col] = distTo[row][col] + energy[(row + 1)*cols + col];//getEnergy(energy, row + 1, col, cols);
					edgeTo[row + 1][col] = 0;
				}
				//Check the pixel to the bottom-right
				if (col != cols-1)
					if (distTo[row + 1][col + 1] > distTo[row][col] + energy[(row+1)*cols + (col +1 )]) {//getEnergy(energy, row + 1, col + 1, cols)) {
						distTo[row + 1][col + 1] = distTo[row][col] + energy[(row + 1)*cols + (col + 1)]; //getEnergy(energy, row + 1, col + 1, cols);
						edgeTo[row + 1][col + 1] = -1;
					}
			}
		}

		/*for (size_t i = 0; i < rows; i++) {
			for (size_t j = 0; j < cols; j++) {
				cout << i << ", " << j << ", " << distTo[i][j] << endl;
			}
		}
*/
		//Find the bottom of the min-path
		unsigned int min_index = 0, min = distTo[rows - 1][0];
		for (int i = 1; i < cols; ++i)
		if (distTo[rows - 1][i] < min) {
			min_index = i;
			min = distTo[rows - 1][i];
		}

		//Retrace the min-path and update the 'seam' vector
		seam[rows - 1] = min_index;
		for (int i = rows - 1; i > 0; --i)
			seam[i - 1] = seam[i] + edgeTo[i][seam[i]];

		/*for (size_t j = 0; j < seam.size(); j++)
		{
		cout << j << ", " << seam[j] << endl;
		}*/
		return seam;
	}

	void removeVerticalSeam(vector<uint> seam, Mat *im) {
		//Move all the pixels to the right of the seam, one pixel to the left
		size_t rows = (*im).rows;
		size_t cols = (*im).cols;
		for (int row = 0; row < rows; ++row) {
			for (int col = seam[row]; col < cols - 1; ++col){
				//cout << row << ", " << col << endl;
				(*im).at<Vec3b>(row, col) = (*im).at<Vec3b>(row, col + 1);
			}
		}

		//Resize the image to remove the last column
		*im = (*im)(Rect(0, 0, cols - 1, rows));
	}

	Mat cpuCarve(cv::Mat im, int direction, int seams) {
		unsigned char *imgPtr = new unsigned char[im.rows * im.cols * im.channels()];

		unsigned char *cvPtr = im.ptr<unsigned char>(0);
		for (size_t i = 0; i < im.rows * im.cols * im.channels(); ++i) {
			imgPtr[i] = cvPtr[i];
		}

		//unsigned char *imgPtr = im.data;

		//size_t numPixels = rows * cols;
		//vector<unsigned char> red;
		//vector<unsigned char> green;
		//vector<unsigned char> blue;
		////vector<int> energy;
		
		//for (size_t i = 0; i < numPixels; ++i) {
		//	blue.push_back(imgPtr[3 * i + 0]);
		//	green.push_back(imgPtr[3 * i + 1]);
		//	red.push_back(imgPtr[3 * i + 2]);
		//}

		//computeFullEnergy(rows, cols, &red, &green, &blue, &energy);
		/*for (size_t i = 0; i < im.rows*im.cols; i++) {
			cout << i << ", " << (int)energy[i] << endl;
		}*/
		size_t rows = im.rows;
		size_t cols = im.cols;
		unsigned int *energy = new unsigned int[rows*cols];
		// Horizontal
		if (direction == 0) {
			for (int i = 0; i < seams; ++i) {
				size_t rows = im.rows;
				size_t cols = im.cols;
				transpose(im, im);
				computeFullEnergy(im, energy);
				vector<uint> seam = findVerticalSeam(energy, cols, rows);
				removeVerticalSeam(seam, &im);
				transpose(im, im);
			}
		}
		// Vertical
		else {
			for (int i = 0; i < seams; ++i) {
				size_t rows = im.rows;
				size_t cols = im.cols;
				/*unsigned int *energy = new unsigned int[rows*cols];*/
				computeFullEnergy(im, energy);
				vector<uint> seam = findVerticalSeam(energy, rows, cols);
				removeVerticalSeam(seam,&im);
			}
		}
		return im;
	}

	__global__ void computeEnergyGPU(size_t rows, size_t cols, unsigned char *img, unsigned int *energy) {
		int  ny = rows;
		int  nx = cols;
		int2 image_index_2d = make_int2((blockIdx.x * blockDim.x) + threadIdx.x, (blockIdx.y * blockDim.y) + threadIdx.y);
		int  image_index_1d = (nx * image_index_2d.y) + image_index_2d.x;

		if (image_index_2d.x < nx && image_index_2d.y < ny)	{
			if (image_index_2d.x == 0 || image_index_2d.x == cols - 1 || image_index_2d.y == 0 || image_index_2d.y == rows - 1) {
				energy[image_index_1d] = 195075;
			}
			else {
				int val = 0;
				int i = image_index_2d.y;
				int j = image_index_2d.x;
				val += powf(img[((i - 1)*cols * 3) + (3 * j)] - img[((i + 1)*cols * 3) + (3 * j)], 2);
				val += powf(img[((i - 1)*cols * 3) + ((3 * j) + 1)] - img[((i + 1)*cols * 3) + ((3 * j) + 1)], 2);
				val += powf(img[((i - 1)*cols * 3) + ((3 * j) + 2)] - img[((i + 1)*cols * 3) + ((3 * j) + 2)], 2);

				val += powf(img[(i * cols * 3) + ((3 * j) + 3)] - img[(i * cols * 3) + ((3 * j) - 3)], 2);
				val += powf(img[(i * cols * 3) + ((3 * j) + 4)] - img[(i * cols * 3) + ((3 * j) - 2)], 2);
				val += powf(img[(i * cols * 3) + ((3 * j) + 5)] - img[(i * cols * 3) + ((3 * j) - 1)], 2);

				energy[image_index_1d] = val;
			}
		}
	}

	/*__global__ void removeVerticalSeamGPU(size_t rows, size_t cols, unsigned char *dev_img, unsigned int *dev_seams) {
		int global_index_1d = (blockIdx.x * blockDim.x) + threadIdx.x;

		if (global_index_1d < rows*cols*3) {
			if (global_index_1d % 3 != 0)
				return;
			unsigned char temp1, temp2, temp3;
			for (size_t row = 0; row < rows; row++) {
				temp1 = dev_img[row*(cols - row) + global_index_1d + 3];
				temp2 = dev_img[row*(cols - row) + global_index_1d + 4];
				temp3 = dev_img[row*(cols - row) + global_index_1d + 5];
				__syncthreads();
				if (global_index_1d >= (row*(cols-row) + (dev_seams[row]*3))) {
					dev_img[row*cols + global_index_1d] = temp1;
					dev_img[row*cols + global_index_1d + 1] = temp2;
					dev_img[row*cols + global_index_1d + 2] = temp3;
				}
				__syncthreads();
			}
		}
	}*/

	__global__ void kernMapToBoolean(size_t rows, size_t cols, int *bools, unsigned int *idata) {
		unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (index >= rows*cols*3)
			return;
		bools[index] = 1;
		__syncthreads();
		for (size_t row = 0; row < rows; row++) {
			if (index == (row*cols*3 + idata[row] * 3)) {
				bools[index] = 0;
				bools[index+1] = 0;
				bools[index+2] = 0;
			}
		}
	}

	__global__ void copyElements(int n, int *src, int *dest) {
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (index >= n)
			return;
		dest[index] = src[index];
	}

	__global__ void upSweep(int n, int *idata, int d) {
		int k = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (k >= n)
			return;
		if (k % (d * 2) == (d * 2) - 1) {
			idata[k] += idata[k - d];
		}

	}

	__global__ void downSweep(int n, int *idata, int d) {
		int k = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (k >= n)
			return;
		int temp;
		if (k % (d * 2) == (d * 2) - 1) {
			//printf("kernel: %d", k);
			temp = idata[k - d];
			idata[k - d] = idata[k];  // Set left child to this node’s value
			idata[k] += temp;
		}

	}

	__global__ void makeElementZero(int *data, int index) {
		int k = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (index == k) {
			data[k] = 0;
		}
	}

	__global__ void kernScatter(int n, unsigned char *odata, const unsigned char *idata, const int *bools, const int *indices) {
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (index >= n)
			return;
		if (bools[index] == 1)
			odata[indices[index]] = idata[index];
	}

	size_t removeVerticalSeamGPU(size_t rows, size_t cols, unsigned char *dev_idata, unsigned char *dev_odata, unsigned int *dev_seams, int *dev_boolean, int *dev_indices) {

		int count=1;
		int pixelSize = rows*cols * 3;
		int paddedArraySize = 1 << ilog2ceil(pixelSize);

		dim3 fullBlocksPerGrid((pixelSize + BLOCK_SIZE - 1) / BLOCK_SIZE);
		dim3 fullBlocksPerGridPadded((paddedArraySize + BLOCK_SIZE - 1) / BLOCK_SIZE);
		kernMapToBoolean << <fullBlocksPerGrid, BLOCK_SIZE >> >(rows, cols, dev_boolean, dev_seams);
		//unsigned int *boolean = new unsigned int[rows*cols * 3];
		//cudaMemcpy(boolean, dev_boolean, sizeof(unsigned int)*rows*cols*3, cudaMemcpyDeviceToHost);
		//for (size_t i = 0; i < rows*cols*3; i++)
		//{
		//	cout << i << ", " << boolean[i] << endl;
		//}

		copyElements << <fullBlocksPerGrid, BLOCK_SIZE >> >(pixelSize, dev_boolean, dev_indices);


		for (int d = 0; d < ilog2ceil(paddedArraySize); d++) {
			upSweep << <fullBlocksPerGridPadded, BLOCK_SIZE >> >(paddedArraySize, dev_indices, 1 << d);
		}

		makeElementZero << <fullBlocksPerGridPadded, BLOCK_SIZE >> >(dev_indices, paddedArraySize - 1);

		for (int d = ilog2ceil(paddedArraySize) - 1; d >= 0; d--) {
			downSweep << <fullBlocksPerGridPadded, BLOCK_SIZE >> >(paddedArraySize, dev_indices, 1 << d);
		}

		kernScatter << <fullBlocksPerGrid, BLOCK_SIZE >> >(pixelSize, dev_odata, dev_idata, dev_boolean, dev_indices);

		cudaMemcpy(dev_idata, dev_odata, pixelSize*sizeof(unsigned char), cudaMemcpyDeviceToDevice);
		cudaMemcpy(&count, dev_indices + paddedArraySize - 1, sizeof(int), cudaMemcpyDeviceToHost);
		return count;
	}
	
	__global__ void transposeNaive(unsigned char *odata, const unsigned char *idata)
	{
				
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		int width = gridDim.x * blockDim.x;

		for (int j = 0; j < blockDim.x; j += blockDim.y)
			odata[x*width + (y + j)] = idata[(y + j)*width + x];
	}

	Mat gpuCarve(cv::Mat im, int direction, int seams) {
		unsigned char *dev_imgPtr, *dev_imgPtrBuffer;
		unsigned int *dev_energy;
		unsigned int *dev_seams;
		int *dev_boolean, *dev_indices;
		unsigned char *imgPtr = new unsigned char[im.rows * im.cols * im.channels()];

		if (direction == 0) {
			transpose(im, im);
		}
		unsigned char *cvPtr = im.ptr<unsigned char>(0);
		for (size_t i = 0; i < im.rows * im.cols * im.channels(); ++i) {
			imgPtr[i] = cvPtr[i];
		}

		int paddedArraySize = 1 << ilog2ceil(im.rows*im.cols*3);
		cudaMalloc((void**)&dev_imgPtr, sizeof(unsigned char)*im.rows * im.cols * im.channels());
		cudaMalloc((void**)&dev_imgPtrBuffer, sizeof(unsigned char)*im.rows * im.cols * im.channels());
		cudaMalloc((void**)&dev_energy, sizeof(unsigned int)*im.rows * im.cols);
		cudaMalloc((void**)&dev_seams, sizeof(unsigned int)*im.cols);
		cudaMalloc((void**)&dev_boolean, paddedArraySize * sizeof(int));
		cudaMalloc((void**)&dev_indices, paddedArraySize * sizeof(int));
		cudaMemcpy(dev_imgPtr, imgPtr, sizeof(unsigned char)*im.rows*im.cols*im.channels(), cudaMemcpyHostToDevice);

		/*size_t rows = im.rows;
		size_t cols = im.cols;
		unsigned int *energy2 = new unsigned int[rows*cols];
		for (size_t i = 0; i < rows*cols; i++) {
		energy2[i] = 195075;
		}
		computeFullEnergyGPU(rows, cols, imgPtr, energy2);*/
		size_t rows = im.rows;
		size_t cols = im.cols;
		if (direction == 0) {
			for (int i = 0; i < seams; ++i) {
				const dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y, 1);
				computeEnergyGPU << <gridSize, blockSize >> >(rows, cols, dev_imgPtr, dev_energy);
				unsigned int *energy = new unsigned int[rows*cols * 3];
				cudaMemcpy(energy, dev_energy, sizeof(unsigned int)*rows*cols, cudaMemcpyDeviceToHost);
				vector<uint> seam = findVerticalSeam(energy, rows, cols);
				cudaMemcpy(dev_seams, &seam[0], sizeof(unsigned int)*rows, cudaMemcpyHostToDevice);
				removeVerticalSeamGPU(rows, cols, dev_imgPtr, dev_imgPtrBuffer, dev_seams, dev_boolean, dev_indices);
				cols--;
			}
		}
		// Vertical
		else {
			for (int i = 0; i < seams; ++i) {
				const dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y, 1);
				computeEnergyGPU << <gridSize, blockSize >> >(rows, cols, dev_imgPtr, dev_energy);
				unsigned int *energy = new unsigned int[rows*cols * 3];
				cudaMemcpy(energy, dev_energy, sizeof(unsigned int)*rows*cols, cudaMemcpyDeviceToHost);
				vector<uint> seam = findVerticalSeam(energy, rows, cols);
				cudaMemcpy(dev_seams, &seam[0], sizeof(unsigned int)*rows, cudaMemcpyHostToDevice);
				removeVerticalSeamGPU(rows, cols, dev_imgPtr, dev_imgPtrBuffer, dev_seams, dev_boolean, dev_indices);
				cols--;
			}
		}
		cudaMemcpy(imgPtr, dev_imgPtr, sizeof(unsigned char)*rows*(cols)* 3, cudaMemcpyDeviceToHost);
		int sizes[2];
		sizes[0] = rows;
		sizes[1] = cols;
		cv::Mat carved(2, sizes, CV_8UC3, (void *)imgPtr);
		if (direction == 0) {
			transpose(carved, carved);
		}

		cudaFree(&dev_imgPtr);
		cudaFree(&dev_imgPtrBuffer);
		cudaFree(&dev_energy);
		cudaFree(&dev_seams);
		cudaFree(&dev_boolean);
		cudaFree(&dev_indices);

		return carved;

	}
}