/**
 * somma di vettore di c = a + b
 *
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include "cudaUtility.h"

__global__ void vecAdd(float* A, float* B, float* C, int N)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if( idx < N)
	{
		C[idx] = A[idx] + B[idx];	
	}
}


int main(void)
{
	int numElement = 50000;
	size_t size = numElement * sizeof(float);
	
	float *A, *B, *C;

	CUDA(cudaMallocManaged(&A, size));
	CUDA(cudaMallocManaged(&B, size));
	CUDA(cudaMallocManaged(&C, size));
	//alloco i tre puntatori a b e c della grandezza numElements
	float *h_A = (float*)malloc(numElement * sizeof(float));
	float *h_B = (float*)malloc(numElement * sizeof(float));
	float *h_C = (float*)malloc(numElement * sizeof(float));

	if(h_A == NULL || h_B == NULL || h_C == NULL)
	{
		printf("fallito a fare il malloc dei puntatori cpu\n");
		exit(1);
	}
	
	//inizializzo i puntatori cpu
	for(int i = 0; i < numElement; i++)
	{
		h_A[i] = rand()/(float)RAND_MAX;
		h_B[i] = rand()/(float)RAND_MAX;
	}

	//alloco i tre puntari della memoria GPU
	float *d_A = NULL;
	float *d_B = NULL;
	float *d_C = NULL;
	CUDA(cudaMalloc((void**)&d_A, size));
	CUDA(cudaMalloc((void**)&d_B, size));
	CUDA(cudaMalloc((void**)&d_C, size));
	
	//copio gli host input nei rispettivi device input
	CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
	CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
	CUDA(cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice));
	
	//configuriamo il kernel
	size_t threads_per_blocks = 256;
	size_t blocks_per_grid = (numElement + threads_per_blocks -1) / threads_per_blocks;
	printf("thread = %lu, blocks = %lu, Tot = %lu\n", threads_per_blocks, blocks_per_grid, threads_per_blocks * blocks_per_grid);
	vecAdd<<<threads_per_blocks, blocks_per_grid>>>(d_A, d_B, d_C, numElement);
	CUDA(cudaGetLastError());

	//copio i risulati nel array C
	CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

	//LIbero la memoria cuda
	CUDA(cudaFree(d_A));
	CUDA(cudaFree(d_B));
	CUDA(cudaFree(d_C));

	//libero la CPU
	free(h_A);
	free(h_B);
	free(h_C);
	
	printf("Done\n");
	return 0;
	
}


