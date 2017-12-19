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

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
	int width;
	int height;
	float* elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 16
// Forward declaration of the matrix multiplication kernel

__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);
// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE

void MatMul(const Matrix A, const Matrix B, Matrix C);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	//Use the functions here
	
	Matrix A, B, C = {0};
	
	A.width = 256;
	A.height = 256;
	
	B.width = 256;
	B.height = 256;
	
	C.width = A.width;
	C.height = B.hieght;
	
	A.elements = new float[A.width * A.height];
	B.elements = new float[B.width * B.height];
	C.elements = new float[C.width * C.height];
	
	for(int i =0; i < A.width * A.height; ++i)
	{
		A.elements[i] = i;
	}
	
	
	for(int i =0; i < B.width * B.height; ++i)
	{
		B.elements[i] = i;
	}
	
	MatMul(A, B, C);
	
	//Print the result matrix
	for(int i =0; i < C.width * C.height; ++i)
	{
		int row = i / C.width;
		int col = i % C.width;
		
		std::cout << row << ", " << col << " :" C.elements[i];
		
		if(col == 0 && i != 0)
		{
			std::cout << endl;
		}
	}
	
	delete A.elements;
	delete B.elements;
	delete C.elements;

	system("pause");
}

void MatMul(const Matrix A, const Matrix B, Matrix C)
{
	// Load A and B to device memory
	Matrix d_A;
	d_A.width = A.width; d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(float);

	cudaMalloc(&d_A.elements, size);
	cudaMemcpy(d_A.elements, A.elements, size,
		cudaMemcpyHostToDevice);

	Matrix d_B;
	d_B.width = B.width; d_B.height = B.height;
	size = B.width * B.height * sizeof(float);
	cudaMalloc(&d_B.elements, size);

	cudaMemcpy(d_B.elements, B.elements, size,
		cudaMemcpyHostToDevice);

	// Allocate C in device memory
	Matrix d_C;
	d_C.width = C.width; d_C.height = C.height;
	size = C.width * C.height * sizeof(float);

	cudaMalloc(&d_C.elements, size);

	// Invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);

	MatMulKernel <<<dimGrid, dimBlock >>> (d_A, d_B, d_C);
	// Read C from device memory
	cudaMemcpy(C.elements, d_C.elements, size,
		cudaMemcpyDeviceToHost);

	// Free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
	// Each thread computes one element of C
	// by accumulating results into Cvalue
	float Cvalue = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	for (int e = 0; e < A.width; ++e)
		Cvalue += A.elements[row * A.width + e]
		* B.elements[e * B.width + col];
	C.elements[row * C.width + col] = Cvalue;
}
