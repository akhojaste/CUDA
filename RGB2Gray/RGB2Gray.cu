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

	// The B value of the pixel, images in the OpenCV are in BGR format not RGB
	long long lPixelIdx = (iRow * iWidth + iCol ) * iChannels; 

	unsigned char ucB = pfimagIn[lPixelIdx];
	unsigned char ucG = pfimagIn[lPixelIdx + 1];
	unsigned char ucR = pfimagIn[lPixelIdx + 2];

	//Weighted method or luminosity method. The controbution of each color
	//to the final gray scale image is different. So if we take simply the
	//average of RGB, i.e. (R + G + B) / 3, then the image has low contrast
	//and it is mostly black. Also our eyes interpret different colors differently
	float temp = 0.11 * (float)ucB + 0.59 * (float)ucG + 0.3 * (float)ucR;

	//Convert the RGB colorful image to the grayscale
	pfimgOut[lPixelIdx + 0] = (unsigned char)temp;
	pfimgOut[lPixelIdx + 1] = (unsigned char)temp;
	pfimgOut[lPixelIdx + 2] = (unsigned char)temp;
}
