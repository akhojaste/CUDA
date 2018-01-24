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


__global__ void Array_Reduction_Kernel(float * g_iData)
{
	extern __shared__ float sData[]; //Size is determined by the host

	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int tId = threadIdx.x;

	//First every thread in the block puts its value intor the shared memory
	sData[index] = g_iData[index];
	__syncthreads();

	for (int t = 1; t < blockDim.x; t *= 2)
	{
		if (index % (2 * t) == 0) //Even threads or even elements of the array. i.e. thread0, thread2, ...
		{
			sData[index] += sData[index + t];
		}

		//At each iteration of the for loop. Thread wait at this barrier so other threads will read here.
		__syncthreads();
	}

	//Move the summation to the global memory now
	g_iData[0] = sData[0];

}

extern "C"
void Array_Reduction(float *h_ArrayReduction, unsigned int ArraySize, long &lSum)
{
	//Device array
	float *d_ArrayReduction = nullptr;
	checkCudaErrors(cudaMalloc((void **)&d_ArrayReduction, ArraySize * sizeof(float)));

	//Move the data from host to device
	checkCudaErrors(cudaMemcpy(d_ArrayReduction, h_ArrayReduction, ArraySize * sizeof(float), cudaMemcpyHostToDevice));

	dim3 gridArrayReduction(1, 1, 1);
	dim3 blockArrayReduction(ArraySize, 1, 1);
	int iMemSize = ArraySize * sizeof(float); // to pass to the cuda kernel so it allocates shared memory there

	Array_Reduction_Kernel << <gridArrayReduction, blockArrayReduction, iMemSize >> >(d_ArrayReduction);

	checkCudaErrors(cudaMemcpy(h_ArrayReduction, d_ArrayReduction, ArraySize * sizeof(float), cudaMemcpyDeviceToHost));

	lSum = h_ArrayReduction[0];

	cudaFree(d_ArrayReduction);
}
