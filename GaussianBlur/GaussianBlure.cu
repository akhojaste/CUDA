///The following functions transforms an RGB image into a Gray scale image using CUDA
#include "iostream"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes CUDA
#include <cuda_runtime.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples

using namespace cv;

void __global__ rgb2gray(const unsigned char *pfimagIn, unsigned char *pfimgOut, const int iWidth, const int iHeight, const int iChannels);

extern "C"
void ImageProcessingGPU(Mat image);

#define BLOCK_SIZE 8

void ImageProcessingGPU(Mat image)
{
	//Image dimensions and channels
	int iWidth = image.cols;
	int iHeight = image.rows;
	int iCn = image.channels();

	//Total number of Bytes of the image
	size_t count = iWidth * iHeight * iCn * sizeof(unsigned char);

	//Input image is the one we read from file and want to change
	unsigned char *h_ImagIn = image.data;
  
	//The buffers in the GPU 
	unsigned char *d_ImagIn;
	unsigned char *d_ImagOut;
  
	//Allocate memory in GPU
	checkCudaErrors(cudaMalloc((void **)&d_ImagIn, count));
	checkCudaErrors(cudaMalloc((void **)&d_ImagOut, count));

	//Test, pass this to the output to test if the cudamemcpy works
	//error = cudaMemset((void *)d_ImagOut, 255, count);

	//Transfer data to GPU
	checkCudaErrors(cudaMemcpy((void *)d_ImagIn, (void *)h_ImagIn, count, cudaMemcpyHostToDevice));
  
	//Compue the results in GPU
	dim3 dNumThreadsPerBlock(BLOCK_SIZE, BLOCK_SIZE); //  Each thread block contains this much threads
	// This amount of thread blocks
	//Total number of threads that will be launched are dimGrid.x * dimGird.y * dimBlocks.x * dimBlocks.y
	//NOTE: the toal numer of thread per block, i.e. dimBlock.x * dimBlock.y should not excede 1024 and
	//in some system 512
	dim3 dNumBlocks(iWidth / dNumThreadsPerBlock.x,  iHeight / dNumThreadsPerBlock.y); 
  
	//GPU Kernel
	rgb2gray <<< dNumBlocks, dNumThreadsPerBlock >>>  (d_ImagIn, d_ImagOut, iWidth, iHeight, iCn);
  
	//Transfer data back from GPU to CPU
	checkCudaErrors(cudaMemcpy((void *)h_ImagIn, (void *)d_ImagOut, count, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_ImagIn));
	checkCudaErrors(cudaFree(d_ImagOut));
}

//GPU Kernel
void __global__ rgb2gray(const unsigned char *pfimagIn, unsigned char *pfimgOut, const int iWidth, const int iHeight, const int iChannels)
{
	int iRow = blockIdx.y * blockDim.y + threadIdx.y;
	int iCol = blockIdx.x * blockDim.x + threadIdx.x;

	double dbFilterCoeff[9] = { -0.0833333333333333, -0.0833333333333333, -0.0833333333333333,
								-0.0833333333333333, 1.66666666666667, -0.0833333333333333,
								-0.0833333333333333, -0.0833333333333333, -0.0833333333333333 };

	// The B value of the pixel, images in the OpenCV are in BGR format not RGB
	//long long lPixelIdx = (iRow * iWidth + iCol ) * iChannels; 
	
	short sFilterRow = 0;
	short sFilterCol = 0;

	double dbFilterOutputB = 0.0;
	double dbFilterOutputG = 0.0;
	double dbFilterOutputR = 0.0;

	for (int iFilterCnt = 0; iFilterCnt < 9; ++iFilterCnt)
	{
		sFilterRow = iFilterCnt / 3;
		sFilterCol = iFilterCnt / 3;

		int lImageRow = iRow + sFilterRow - 1;
		int lImageCol = iCol + sFilterCol - 1;

		if (lImageRow < 0 || lImageRow >= iHeight || lImageCol < 0 || lImageCol >= iWidth)
			continue;

		dbFilterOutputB += dbFilterCoeff[sFilterRow * 3 + sFilterCol] * (float)pfimagIn[lImageRow * iWidth * iChannels + lImageCol * iChannels];
		dbFilterOutputG += dbFilterCoeff[sFilterRow * 3 + sFilterCol] * (float)pfimagIn[lImageRow * iWidth * iChannels + lImageCol * iChannels + 1];
		dbFilterOutputR += dbFilterCoeff[sFilterRow * 3 + sFilterCol] * (float)pfimagIn[lImageRow * iWidth * iChannels + lImageCol * iChannels + 2];
	}

	if (dbFilterOutputB < 0.0)
		dbFilterOutputB = 0.0;
	else if (dbFilterOutputB >= 255.0)
		dbFilterOutputB = 255.0;

	if (dbFilterOutputG < 0.0)
		dbFilterOutputG = 0.0;
	else if (dbFilterOutputG >= 255.0)
		dbFilterOutputG = 255.0;

	if (dbFilterOutputR < 0.0)
		dbFilterOutputR = 0.0;
	else if (dbFilterOutputR >= 255.0)
		dbFilterOutputR = 255.0;

	pfimgOut[iRow * iWidth * iChannels + iCol * iChannels + 0] = dbFilterOutputB; //B
	pfimgOut[iRow * iWidth * iChannels + iCol * iChannels + 1] = dbFilterOutputG; //G
	pfimgOut[iRow * iWidth * iChannels + iCol * iChannels + 2] = dbFilterOutputR; //R
}
