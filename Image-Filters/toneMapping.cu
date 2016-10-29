#include <stdio.h>
#include <cuda.h>
#include "toneMapping.h"
#include "cuda_runtime.h"

size_t numBins = 1024;
#define BLOCK_SIZE 32
const dim3 blockSize(32, 16, 1);
const int numThreads = 192;

namespace toneMapping {
	void rgb_to_xyY(size_t rows, size_t cols, float *red, float *green, float *blue, float *x, float *y, float *logY) {
		float X, Y, Z, L;
		for (size_t i = 0; i < rows*cols; i++) {
				X = (red[i] * 0.4124f) + (green[i] * 0.3576f) + (blue[i] * 0.1805f);
				Y = (red[i] * 0.2126f) + (green[i] * 0.7152f) + (blue[i] * 0.0722f);
				Z = (red[i] * 0.0193f) + (green[i] * 0.1192f) + (blue[i] * 0.9505f);
				L = X + Y + Z;
				x[i] = X / L;
				y[i] = Y / L;
				logY[i] = log10f(0.0001 + Y);
		}
	}

	void calculateCDF(size_t rows, size_t cols, size_t numBins, float *luminance, unsigned int* cdf, float &min_logLum, float &max_logLum) {
		min_logLum = luminance[0];
		max_logLum = luminance[0];

		for (size_t i = 1; i < rows*cols; ++i) {
				min_logLum = std::min(luminance[i], min_logLum);
				max_logLum = std::max(luminance[i], max_logLum);
		}

		float logLumRange = max_logLum - min_logLum;

		unsigned int *histo = new unsigned int[numBins];

		for (size_t i = 0; i < numBins; ++i) histo[i] = 0;

		for (size_t i = 0; i < rows*cols; ++i) {
				unsigned int bin = std::min(static_cast<unsigned int>(numBins - 1),
					static_cast<unsigned int>((luminance[i] - min_logLum) / logLumRange * numBins));
				histo[bin]++;
		}

		cdf[0] = 0;
		for (size_t i = 1; i < numBins; ++i) {
			cdf[i] = cdf[i - 1] + histo[i - 1];
		}

		delete[] histo;
	}

	void mapImage(size_t rows, size_t cols, float min_logLum, float max_logLum, unsigned int* cdf, 
		float *x, float *y, float *logY, float *red, float *green, float *blue) {
		float *norm_cdf = (float *)malloc(sizeof(float)*numBins);

		const float normalization_constant = 1.f / cdf[numBins - 1];
		for (size_t i = 0; i < numBins; i++)
		{
			unsigned int input_value = cdf[i];
			float        output_value = (float)input_value * normalization_constant;
			norm_cdf[i] = output_value;
		}

		float log_Y_range = max_logLum - min_logLum;

		for (size_t i = 0; i < rows*cols; i++) {
				float temp_x = x[i];
				float temp_y = y[i];
				float temp_log_Y = logY[i];
				int   bin_index = std::min((int)numBins - 1, int((numBins * (temp_log_Y - min_logLum)) / log_Y_range));
				float Y_new = norm_cdf[bin_index];
				float X_new = temp_x * (Y_new / temp_y);
				float Z_new = (1 - temp_x - temp_y) * (Y_new / temp_y);

				float r_new = (X_new *  3.2406f) + (Y_new * -1.5372f) + (Z_new * -0.4986f);
				float g_new = (X_new * -0.9689f) + (Y_new *  1.8758f) + (Z_new *  0.0415f);
				float b_new = (X_new *  0.0557f) + (Y_new * -0.2040f) + (Z_new *  1.0570f);

				red[i] = r_new;
				green[i] = g_new;
				blue[i] = b_new;
			}

		delete[] norm_cdf;
	}

	int cpuMap(size_t rows, size_t cols, float *imgPtr) {
		size_t numPixels = rows * cols;
		float *red = new float[numPixels];
		float *green = new float[numPixels];
		float *blue = new float[numPixels];
		float *x = new float[numPixels];
		float *y = new float[numPixels];
		float *logY = new float[numPixels];

		for (size_t i = 0; i < numPixels; ++i) {
			blue[i] = imgPtr[3 * i + 0];
			green[i] = imgPtr[3 * i + 1];
			red[i] = imgPtr[3 * i + 2];
		}

		unsigned int *cdf = (unsigned int *)malloc(sizeof(unsigned int)*numBins);
		float min_logLum, max_logLum;
		rgb_to_xyY(rows, cols, red, green, blue, x, y, logY);
		/*std::cout << "CDF" << std::endl;
		for (size_t i = 0; i < numPixels; i++)
		{
			std::cout << i << ", " << logY[i] << std::endl;
		}*/
		calculateCDF(rows, cols, numBins, logY, cdf, min_logLum, max_logLum);
		mapImage(rows, cols, min_logLum, max_logLum, cdf, x, y, logY, red, green, blue);

		for (int i = 0; i < numPixels; ++i) {
			imgPtr[3 * i + 0] = blue[i];
			imgPtr[3 * i + 1] = green[i];
			imgPtr[3 * i + 2] = red[i];
		}
		delete[] cdf;
		delete[] red;
		delete[] green;
		delete[] blue;
		delete[] x;
		delete[] y;
		delete[] logY;
		return 1;
	}

	__global__ void rgb2xyY(float* red, float* green, float* blue, float* d_x, float* d_y,
		float* d_logY, int rows, int cols) {
		int  ny = rows;
		int  nx = cols;
		int2 image_index_2d = make_int2((blockIdx.x * blockDim.x) + threadIdx.x, (blockIdx.y * blockDim.y) + threadIdx.y);
		int  image_index_1d = (nx * image_index_2d.y) + image_index_2d.x;

		if (image_index_2d.x < nx && image_index_2d.y < ny)
		{
			float r = red[image_index_1d];
			float g = green[image_index_1d];
			float b = blue[image_index_1d];

			float X = (r * 0.4124f) + (g * 0.3576f) + (b * 0.1805f);
			float Y = (r * 0.2126f) + (g * 0.7152f) + (b * 0.0722f);
			float Z = (r * 0.0193f) + (g * 0.1192f) + (b * 0.9505f);

			float L = X + Y + Z;
			float x = X / L;
			float y = Y / L;

			float log_Y = log10f(0.0001f + Y);

			d_x[image_index_1d] = x;
			d_y[image_index_1d] = y;
			d_logY[image_index_1d] = log_Y;
		}
	}

	// calculate reduce max or min and stick the value in d_answer.
	__global__
		void reduce_minmax_kernel(const float* const d_in, float* d_out, const size_t size, int minmax) {
			extern __shared__ float shared[];

			int mid = threadIdx.x + blockDim.x * blockIdx.x;
			int tid = threadIdx.x;

			// we have 1 thread per block, so copying the entire block should work fine
			if (mid < size) {
				shared[tid] = d_in[mid];
			}
			else {
				if (minmax == 0)
					shared[tid] = FLT_MAX;
				else
					shared[tid] = -FLT_MAX;
			}

			// wait for all threads to copy the memory
			__syncthreads();

			// don't do any thing with memory if we happen to be far off ( I don't know how this works with
			// sync threads so I moved it after that point )
			if (mid >= size) {
				if (tid == 0) {
					if (minmax == 0)
						d_out[blockIdx.x] = FLT_MAX;
					else
						d_out[blockIdx.x] = -FLT_MAX;

				}
				return;
			}

			for (unsigned int s = blockDim.x / 2; s > 0; s /= 2) {
				if (tid < s) {
					if (minmax == 0) {
						shared[tid] = min(shared[tid], shared[tid + s]);
					}
					else {
						shared[tid] = max(shared[tid], shared[tid + s]);
					}
				}

				__syncthreads();
			}

			if (tid == 0) {
				d_out[blockIdx.x] = shared[0];
			}
		}

	__global__ void histogram_kernel(unsigned int* d_bins, const float* d_in, const int bin_count, const float lum_min, const float lum_max, const int size) {
		int mid = threadIdx.x + blockDim.x * blockIdx.x;
		if (mid >= size)
			return;
		float lum_range = lum_max - lum_min;
		int bin = ((d_in[mid] - lum_min) / lum_range) * bin_count;

		atomicAdd(&d_bins[bin], 1);
	}

	__global__ void scan_kernel(unsigned int* d_bins, int size) {
			int mid = threadIdx.x + blockDim.x * blockIdx.x;
			if (mid >= size)
				return;

			for (int s = 1; s <= size; s *= 2) {
				int spot = mid - s;

				unsigned int val = 0;
				if (spot >= 0)
					val = d_bins[spot];
				__syncthreads();
				if (spot >= 0)
					d_bins[mid] += val;
				__syncthreads();

			}
		}

	__global__ void normalize_cdf(unsigned int* d_input_cdf, float* d_output_cdf, int n)
	{
		const float normalization_constant = 1.f / d_input_cdf[n - 1];

		int global_index_1d = (blockIdx.x * blockDim.x) + threadIdx.x;

		if (global_index_1d < n)
		{
			unsigned int input_value = d_input_cdf[global_index_1d];
			float        output_value = input_value * normalization_constant;

			d_output_cdf[global_index_1d] = output_value;
		}
	}

	__global__ void tonemap(
		float* d_x,
		float* d_y,
		float* d_log_Y,
		float* d_cdf_norm,
		float* d_r_new,
		float* d_g_new,
		float* d_b_new,
		float  min_log_Y,
		float  max_log_Y,
		float  log_Y_range,
		int    num_bins,
		int    num_pixels_y,
		int    num_pixels_x)
	{
		int  ny = num_pixels_y;
		int  nx = num_pixels_x;
		int2 image_index_2d = make_int2((blockIdx.x * blockDim.x) + threadIdx.x, (blockIdx.y * blockDim.y) + threadIdx.y);
		int  image_index_1d = (nx * image_index_2d.y) + image_index_2d.x;

		if (image_index_2d.x < nx && image_index_2d.y < ny)
		{
			float x = d_x[image_index_1d];
			float y = d_y[image_index_1d];
			float log_Y = d_log_Y[image_index_1d];
			int   bin_index = min(num_bins - 1, int((num_bins * (log_Y - min_log_Y)) / log_Y_range));
			float Y_new = d_cdf_norm[bin_index];

			float X_new = x * (Y_new / y);
			float Z_new = (1 - x - y) * (Y_new / y);

			float r_new = (X_new *  3.2406f) + (Y_new * -1.5372f) + (Z_new * -0.4986f);
			float g_new = (X_new * -0.9689f) + (Y_new *  1.8758f) + (Z_new *  0.0415f);
			float b_new = (X_new *  0.0557f) + (Y_new * -0.2040f) + (Z_new *  1.0570f);

			d_r_new[image_index_1d] = r_new;
			d_g_new[image_index_1d] = g_new;
			d_b_new[image_index_1d] = b_new;
		}
	}

	float reduce_minmax(const float* const d_in, const size_t size, int minmax) {
		// we need to keep reducing until we get to the amount that we consider 
		// having the entire thing fit into one block size
		size_t curr_size = size;
		float* dev_data;

		cudaMalloc(&dev_data, sizeof(float)* size);
		cudaMemcpy(dev_data, d_in, sizeof(float)* size, cudaMemcpyDeviceToDevice);


		float* dev_temp;

		dim3 thread_dim(BLOCK_SIZE);
		const int shared_mem_size = sizeof(float)*BLOCK_SIZE;
		int maxSize;
		while (1) {
			maxSize = (int)ceil((float)curr_size / (float)BLOCK_SIZE) + 1;
			cudaMalloc(&dev_temp, sizeof(float)* maxSize);

			dim3 block_dim((int)ceil((float)size / (float)BLOCK_SIZE) + 1);
			reduce_minmax_kernel << <block_dim, thread_dim, shared_mem_size >> >(dev_data, dev_temp, curr_size, minmax);
			cudaDeviceSynchronize();


			// move the current input to the output, and clear the last input if necessary
			cudaFree(dev_data);
			dev_data = dev_temp;

			if (curr_size <  BLOCK_SIZE)
				break;

			curr_size = maxSize;
		}

		// theoretically we should be 
		float result;
		cudaMemcpy(&result, dev_temp, sizeof(float), cudaMemcpyDeviceToHost);
		cudaFree(dev_temp);
		return result;
	}

	int gpuMap(size_t rows, size_t cols, float *imgPtr) {
		float *dev_red, *dev_green, *dev_blue, *dev_x, *dev_y, *dev_logY;
		size_t numPixels = rows * cols;
		cudaMalloc((void**)&dev_red, sizeof(float)*numPixels);
		cudaMalloc((void**)&dev_green, sizeof(float)*numPixels);
		cudaMalloc((void**)&dev_blue, sizeof(float)*numPixels);
		cudaMalloc((void**)&dev_x, sizeof(float)*numPixels);
		cudaMalloc((void**)&dev_y, sizeof(float)*numPixels);
		cudaMalloc((void**)&dev_logY, sizeof(float)*numPixels);

		float *red = new float[numPixels];
		float *green = new float[numPixels];
		float *blue = new float[numPixels];

		for (size_t i = 0; i < numPixels; ++i) {
			blue[i] = imgPtr[3 * i + 0];
			green[i] = imgPtr[3 * i + 1];
			red[i] = imgPtr[3 * i + 2];
		}

		cudaMemcpy(dev_red, red, sizeof(float)*numPixels, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_green, green, sizeof(float)*numPixels, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_blue, blue, sizeof(float)*numPixels, cudaMemcpyHostToDevice);

		const dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y, 1);
		rgb2xyY << <gridSize, blockSize >> >(dev_red, dev_green, dev_blue,	dev_x, dev_y, dev_logY, rows, cols);

		float *logY = (float *)malloc(sizeof(float)*numPixels);
		cudaMemcpy(logY, dev_logY, numPixels*sizeof(float), cudaMemcpyDeviceToHost);

		// Calculate CDF
		float min_logLum, max_logLum;
		unsigned int *dev_cdf;
		cudaMalloc((void**)&dev_cdf, sizeof(unsigned int)*numBins);
		min_logLum = reduce_minmax(dev_logY, numPixels, 0);
		max_logLum = reduce_minmax(dev_logY, numPixels, 1);

		printf("got min of %f\n", min_logLum);
		printf("got max of %f\n", max_logLum);
		printf("numBins %d\n", numBins);

		unsigned int* dev_bins;

		cudaMalloc(&dev_bins, sizeof(unsigned int)*numBins);
		cudaMemset(dev_bins, 0, sizeof(unsigned int)*numBins);
		dim3 thread_dim(1024);
		dim3 hist_block_dim((int)ceil((float)numPixels / (float)thread_dim.x) + 1);
		histogram_kernel << <hist_block_dim, thread_dim >> >(dev_bins, dev_logY, numBins, min_logLum, max_logLum, numPixels);

		
		dim3 scan_block_dim((int)ceil((float)numBins / (float)thread_dim.x) + 1);

		scan_kernel << <scan_block_dim, thread_dim >> >(dev_bins, numBins);
		cudaDeviceSynchronize();

		cudaMemcpy(dev_cdf, dev_bins, sizeof(unsigned int)*numBins, cudaMemcpyDeviceToDevice);
		cudaFree(dev_bins);


		// Map
		float *dev_cdfNorm;
		cudaMalloc(&dev_cdfNorm, sizeof(float)* numBins);

		normalize_cdf << < (numBins + numThreads - 1) / numThreads, numThreads >> >(dev_cdf, dev_cdfNorm, numBins);

		cudaDeviceSynchronize();
		float log_Y_range = max_logLum - min_logLum;
		tonemap << <gridSize, blockSize >> >(dev_x, dev_y, dev_logY,
			dev_cdfNorm,
			dev_red, dev_green, dev_blue,
			min_logLum, max_logLum,
			log_Y_range, numBins,
			rows, cols);
		cudaDeviceSynchronize();

		// Copy back data
		cudaMemcpy(red, dev_red, sizeof(float)* numPixels, cudaMemcpyDeviceToHost);
		cudaMemcpy(green, dev_green, sizeof(float)* numPixels, cudaMemcpyDeviceToHost);
		cudaMemcpy(blue, dev_blue, sizeof(float)* numPixels, cudaMemcpyDeviceToHost);

		for (int i = 0; i < numPixels; ++i) {
			imgPtr[3 * i + 0] = blue[i];
			imgPtr[3 * i + 1] = green[i];
			imgPtr[3 * i + 2] = red[i];
		}

		cudaFree(dev_red);
		cudaFree(dev_green);
		cudaFree(dev_blue);
		cudaFree(dev_x);
		cudaFree(dev_y);
		cudaFree(dev_logY);
		cudaFree(dev_cdf);
		cudaFree(dev_cdfNorm);

		delete[] red;
		delete[] green;
		delete[] blue;


		return 1;
	}
}
