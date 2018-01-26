// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes CUDA
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples


__global__ void Scan(float * g_DataIn, float *g_DataOut)
{
	//first get all the data for each thread and put it in the shared memory
	extern __shared__ float sData[]; //data size is determined by host.
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int tId = threadIdx.x;

	sData[idx] =  tId > 0 ? g_DataIn[idx - 1] : 0; //shift data one to right
	__syncthreads();

	for (int k = 1; k < blockDim.x; k *= 2)
	{
		if (tId >= k)
		{
			sData[tId] += sData[tId - k];
		}
		
		__syncthreads();
	}

	g_DataOut[tId] = sData[tId];
}

extern "C"
void Scan(float *h_DataIn, float *h_DataOut, unsigned int ArraySize)
{
	float *d_DataIn, *d_DataOut;

	//Allocate memory on GPU
	checkCudaErrors(cudaMalloc((void **)&d_DataIn, ArraySize * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_DataOut, ArraySize * sizeof(float)));

	//Transfer inpput data to GPU
	checkCudaErrors(cudaMemcpy(d_DataIn, h_DataIn, ArraySize * sizeof(float), cudaMemcpyHostToDevice));

	int blocks = 1;
	int threadsPerblock = ArraySize;
	unsigned int iSharedMemSize = threadsPerblock * sizeof(float);

	Scan << <blocks, threadsPerblock, iSharedMemSize >> > (d_DataIn, d_DataOut);

	//Transfer data back to host
	checkCudaErrors(cudaMemcpy(h_DataOut, d_DataOut, ArraySize * sizeof(float), cudaMemcpyDeviceToHost));
}