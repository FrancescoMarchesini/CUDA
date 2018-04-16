/**
 * somma di vettore di c = a + b
 *
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include "cudaUtility.h"


__global__ void vecAdd(float* A, float* B, float* C, int N)
{
	int idx    = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	
	for(int i = idx; i < N; i += stride)
	{
		C[i] = A[i] + B[i];	
	}
}

int main(void)
{
	int numElement = 500000;
	size_t size = numElement * sizeof(float);
	
	float *A, *B, *C;

	CUDA(cudaMallocManaged(&A, size));
	CUDA(cudaMallocManaged(&B, size));
	CUDA(cudaMallocManaged(&C, size));	

	if(A == NULL || B == NULL || C == NULL)
	{
		printf("fallito a fare il malloc dei puntatori cpu\n");
		exit(1);
	}

	//inizializzo i puntatori cpu
	for(int i = 0; i < numElement; i++)
	{
		A[i] = rand()/(float)RAND_MAX;
		B[i] = rand()/(float)RAND_MAX;
	        C[i] = 0;
	}

	//configuriamo il kernel
	size_t threads_per_blocks = 256;
	size_t blocks_per_grid = (numElement + threads_per_blocks -1) / threads_per_blocks;
	printf("thread = %lu, blocks = %lu, Tot = %lu\n", threads_per_blocks, blocks_per_grid, threads_per_blocks * blocks_per_grid);
	vecAdd<<<threads_per_blocks, blocks_per_grid>>>(A, B, C, numElement);
	CUDA(cudaGetLastError());
	CUDA(cudaDeviceSynchronize());

	for(int i = 0; i < numElement; i++)
	{
		if(C[i] != 10 )
		{
			printf("il numero %0.f è diverso da 10\n", C[i]);
			//exit(1);
		}
	}

	CUDA(cudaFree(A));
	CUDA(cudaFree(B));
	CUDA(cudaFree(C));
	return 0;
}


