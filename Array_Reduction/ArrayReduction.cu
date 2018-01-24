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


__global__ void Array_Reduction_Kernel(float * g_iData, float *g_Intermediat)
{
	extern __shared__ float sData[]; //Size is determined by the host

	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int tId = threadIdx.x;

	//First every thread in the block puts its value intor the shared memory
	sData[tId] = g_iData[index];
	__syncthreads();

	for (int t = 1; t < blockDim.x; t *= 2)
	{
		if (tId % (2 * t) == 0) //Even threads or even elements of the array. i.e. thread0, thread2, ...
		{
			sData[tId] += sData[tId + t];
		}

		//At each iteration of the for loop. Thread wait at this barrier so other threads will read here.
		__syncthreads();
	}

	//Move the summation to the global memory now
	if (tId == 0)
	{
		g_Intermediat[blockIdx.x] = sData[0];
	}
	
	__syncthreads();
}

extern "C"
void Array_Reduction(float *h_ArrayReduction, unsigned int ArraySize, long &lSum)
{
	int blocks = 1024;
	int threadPerBlock = ArraySize / 1024; // 1024 blocks of 1024 threads per each block
	int iMemSize = threadPerBlock * sizeof(float); // to pass to the cuda kernel so it allocates shared memory there

	//Device array
	float *d_ArrayReduction = nullptr;
	checkCudaErrors(cudaMalloc((void **)&d_ArrayReduction, ArraySize * sizeof(float)));

	float *d_ArrayReductionIntermediate = nullptr;
	checkCudaErrors(cudaMalloc((void **)&d_ArrayReductionIntermediate, blocks * sizeof(float)));

	float *d_ArrayReductionOut = nullptr;
	checkCudaErrors(cudaMalloc((void **)&d_ArrayReductionOut, 1 * sizeof(float)));

	//Move the data from host to device
	checkCudaErrors(cudaMemcpy(d_ArrayReduction, h_ArrayReduction, ArraySize * sizeof(float), cudaMemcpyHostToDevice));

	//First call, each block will compute its summation and will put it in one element of output array
	//So we get one array of 1024 elements.
	Array_Reduction_Kernel << <blocks, threadPerBlock, iMemSize >> >(d_ArrayReduction, d_ArrayReductionIntermediate);

	//Second call, we have 1024 elemets left, with 1 block of 1024 elemnts each, we compute the summation.
	blocks = 1;
	threadPerBlock = 1024;
	Array_Reduction_Kernel << <blocks, threadPerBlock, iMemSize >> >(d_ArrayReductionIntermediate, d_ArrayReductionOut);

	checkCudaErrors(cudaMemcpy(h_ArrayReduction, d_ArrayReductionOut, 1 * sizeof(float), cudaMemcpyDeviceToHost));

	lSum = h_ArrayReduction[0];

	checkCudaErrors(cudaFree(d_ArrayReduction));
}
