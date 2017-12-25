///The following functions transforms an RGB image into a Gray scale image using CUDA
#include "iostream"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes CUDA
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples

extern "C"
void RGB2GrayCPU(float *reference, float *idata, const unsigned int len);

void __global__ rgb2gray(const float *pfimagIn, float *pfimgOut, const int iWidth, const int iHeight);

void ReadAndProcessImage();

#define BLOCK_SIZE 16

int main()
{
  //Here we call the rgb2gray function
  ReadAndProcessImage();

  std::cin.get();

}

void ReadAndProcessImage()
{
  int iWidth = 256;
  int iHeight = 256;
  
  float *h_ImagIn = new float[iWidth * iHeight];
  float *h_ImagOut = new float[iWidth * iHeight];
  
  //I need to get the values of the input filled with something
  
  float *d_ImagIn;
  float *d_ImagOut;
  
  //Allocate memory in GPU
  cudaMalloc((void **) &d_ImagIn, iWidth * iHeight * sizeof(float));
  cudaMalloc((void **) &d_ImagOut, iWidth * iHeight * sizeof(float));
  
  //Transfer data to GPU
  cudaMemcpy((void *) &d_ImagIn, (void *) &h_ImagIn, iWidth * iHeight * sizeof(float), cudaMemcpyHostToDevice);
  
  //Compue the results in GPU
  dim3 dimBlocks(BLOCK_SIZE, BLOCK_SIZE); //  Each thread block contains this much threads
  dim3 dimGrid(iWidth / BLOCK_SIZE, iHeight / BLOCK_SIZE); // This amount of thread blocks
  //Total number of threads that will be launched are dimGrid.x * dimGird.y * dimBlocks.x * dimBlocks.y
  //NOTE: the toal numer of thread per block, i.e. dimBlock.x * dimBlock.y should not excede 1024 and
  //in some system 512
  
  rgb2gray <<< dimGrid, dimBlocks>>>  (d_ImagIn, h_ImagOut, iWidth, iHeight);
  
  //Transfer data back from GPU to CPU
  cudaMemcpy((void *) &h_ImagOut, (void *) &d_ImagOut, iWidth * iHeight * sizeof(float), cudaMemcpyDeviceToHost);
  
  cudaFree(d_ImagIn);
  cudaFree(d_ImagOut);
}

//GPU Kernel
void __global__ rgb2gray(const float *pfimagIn, float *pfimgOut, const int iWidth, const int iHeight)
{
  
  int iRow = blockIdx.y * blockDim.y + threadIdx.y;
  int iCol = blockIdx.x + blockDim.x + threadIdx.x;

  long lPixel = pfimagIn[iRow * iWidth + iCol];
  lPixel = lPixel & (0x000000FF); // Just get the R value; image format is ARGB

  long lPixelOut = lPixel & (lPixel << 8) & (lPixel << 16); //Gray scale so R = G = B
  lPixelOut = lPixelOut && 0x00000000; //No Transparency in Alpha channel;
  
  pfimgOut[iRow * iWidth + iCol] = lPixelOut;
}
