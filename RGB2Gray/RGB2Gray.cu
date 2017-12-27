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

#define BLOCK_SIZE 16

void ImageProcessingGPU(Mat image)
{
  //Image dimensions and channels
  int iWidth = image.cols;
  int iHeight = image.rows;
  int iCn = image.channels();

  size_t count = iWidth * iHeight * iCn * sizeof(unsigned char);

  //Input image is the one we read from file and want to change
  unsigned char *h_ImagIn = image.data;
  
  unsigned char *d_ImagIn;
  unsigned char *d_ImagOut;
  
  //Allocate memory in GPU
  cudaError error = cudaSuccess;
  error = cudaMalloc((void **)&d_ImagIn, count);
  error = cudaMalloc((void **)&d_ImagOut, count);
  
  //Test
  //error = cudaMemset((void *)d_ImagOut, 255, count);

  //Transfer data to GPU
  error = cudaMemcpy((void *)d_ImagIn, (void *)h_ImagIn, count, cudaMemcpyHostToDevice);
  
  //Compue the results in GPU
  dim3 dNumThreadsPerBlock(8,8); //  Each thread block contains this much threads
  // This amount of thread blocks
  //Total number of threads that will be launched are dimGrid.x * dimGird.y * dimBlocks.x * dimBlocks.y
  //NOTE: the toal numer of thread per block, i.e. dimBlock.x * dimBlock.y should not excede 1024 and
  //in some system 512
  dim3 dNumBlocks(iWidth / dNumThreadsPerBlock.x,  iHeight / dNumThreadsPerBlock.y); 
  
  rgb2gray <<< dNumBlocks, dNumThreadsPerBlock >>>  (d_ImagIn, d_ImagOut, iWidth, iHeight, iCn);
  
  //Transfer data back from GPU to CPU
  error = cudaMemcpy((void *)h_ImagIn, (void *)d_ImagOut, count, cudaMemcpyDeviceToHost);
  
  cudaFree(d_ImagIn);
  cudaFree(d_ImagOut);
}

//GPU Kernel
void __global__ rgb2gray(const unsigned char *pfimagIn, unsigned char *pfimgOut, const int iWidth, const int iHeight, const int iChannels)
{

  int iCol = blockIdx.x * blockDim.x + threadIdx.x;
  int iRow = blockIdx.y * blockDim.y + threadIdx.y;

  long long lPixelIdx = (iRow * iWidth + iCol ) * iChannels; // The B value of the pixel

  unsigned char ucB = pfimagIn[lPixelIdx];
  unsigned char ucG = pfimagIn[lPixelIdx + 1];
  unsigned char ucR = pfimagIn[lPixelIdx + 2];

 
  pfimgOut[lPixelIdx + 0] = (ucB + ucG + ucR) / 3;
  pfimgOut[lPixelIdx + 1] = (ucB + ucG + ucR) / 3;
  pfimgOut[lPixelIdx + 2] = (ucB + ucG + ucR) / 3;
}
