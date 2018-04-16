/**
 * Grid_Stride_Loop
 * Serve per elaborare array che sono piu grandi della griglia di kernel
 * la soluzione è che un ogni thread vengono eseguiti piu volte sospoandosi trami gli indice 
 * di un gridDim * gridSize.
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include "cudaUtility.h"

__global__ void grid_stride(float* A, int N)
{
	int idx    = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x ;
	
	for(int i = idx; i < N; i += stride)
	{	
		A[idx] *= 2;
	}
}

bool check(float *A, int N)
{
	for(int i = 0; i < N; i++)
	{
		if(A[i] != i*2) return false;
	}
	return false;	
}

int main(void)
{
	//elemento
	int N = 1000;
	float *a;

	//allocazione di entrambe le memoria CPU e GPU da parte di CUDA
	size_t size = N * sizeof(float);
	CUDA(cudaMallocManaged(&a, size));
	
	//init di a:
	for(int i = 0; i < N; i++)
	{
		a[i] = rand()/(float)RAND_MAX;	
	}	
	
	//configuro il kernel
	size_t threads_per_blocks = 256;
	size_t blocks_per_grid 	  = 32;
	grid_stride<<<threads_per_blocks, blocks_per_grid>>>(a, N);
	CUDA(cudaGetLastError());
	CUDA(cudaDeviceSynchronize());

	bool test = check(a, N);
	printf("Tutti i numeri sono doppi? %s\n", test ? "VERO" : "FALSO");

	CUDA(cudaFree(a));
	return 0;
	
}


