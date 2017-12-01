////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

/* Template project which demonstrates the basics on how to setup a project
* example application.
* Host code.
*/

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

const long VecSize = 256*256*256;

void runMyTest();
__global__ void
vecAddGPU(double *pdbA, double *pdbB, double *pdbC)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	pdbC[i] = pdbA[i] + pdbB[i];
}

void vecAddCPU(double *pdbA, double *pdbB, double *pdbC)
{
	for (int i = 0; i < VecSize; ++i)
	{
		pdbC[i] = pdbA[i] + pdbB[i];
	}
	
}
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	runMyTest();

	system("pause");
}

void runMyTest()
{
	std::cout << "Running my Test ..." << std::endl;

	double *pdbA, *pdbB, *pdbC;

	//Pin memory instead of Pagable memory. Pin memory is faster, apparently, to copy from host to device
	//Host(CPU) first copies the data from pagable memory to the pin memory and then copies it from pin memory
	//To device memory. Thats why if the memory in the host side is allocated in the pin memory it is faster.
	//User cannot get too much pin memory though.
	checkCudaErrors(cudaMallocHost((void **)&pdbA, VecSize *  sizeof(double)));
	checkCudaErrors(cudaMallocHost((void **)&pdbB, VecSize *  sizeof(double)));
	checkCudaErrors(cudaMallocHost((void **)&pdbC, VecSize *  sizeof(double)));


	for (int i = 0; i < VecSize; i++)
	{
		pdbA[i] = i;
		pdbB[i] = i;
	}

	double *d_dataA, *d_dataB, *d_dataC;

	StopWatchInterface *timer1 = 0;
	sdkCreateTimer(&timer1);
	sdkStartTimer(&timer1);

	checkCudaErrors(cudaMalloc((void **)&d_dataA, VecSize * sizeof(double)));
	// copy host memory to device
	cudaError e = cudaMemcpy(d_dataA, pdbA, VecSize * sizeof(double),
		cudaMemcpyHostToDevice);

	checkCudaErrors(cudaMalloc((void **)&d_dataB, VecSize * sizeof(double)));
	// copy host memory to device
	checkCudaErrors(cudaMemcpy(d_dataB, pdbB, VecSize * sizeof(double),
		cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **)&d_dataC, VecSize * sizeof(double)));

	int threadsPerBlock = 512;
	int blocksPerGrid = (VecSize + threadsPerBlock - 1) / threadsPerBlock;

	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

	// execute the kernel
	vecAddGPU << < blocksPerGrid, threadsPerBlock >> > (d_dataA, d_dataB, d_dataC);

	cudaError err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// copy result from device to host
	checkCudaErrors(cudaMemcpy(pdbC, d_dataC, VecSize * sizeof(double),
		cudaMemcpyDeviceToHost));

	sdkStopTimer(&timer1);
	printf("Processing time GPU: %f (ms)\n", sdkGetTimerValue(&timer1));

	//Test with CPU
	//QueryPerformanceCounter((LARGE_INTEGER*)&lStart);	
	StopWatchInterface *timer6 = 0;
	sdkCreateTimer(&timer6);
	sdkStartTimer(&timer6);

	vecAddCPU(pdbA, pdbB, pdbC);

	sdkStopTimer(&timer6);
	printf("Processing time CPU: %f (ms)\n", sdkGetTimerValue(&timer6));

	cudaFreeHost(pdbA);
	cudaFreeHost(pdbB);
	cudaFreeHost(pdbC);
}
