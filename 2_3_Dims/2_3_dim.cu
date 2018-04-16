/**
 * Grid_Stride_Loop
 * Serve per elaborare array che sono piu grandi della griglia di kernel
 * la soluzione è che un ogni thread vengono eseguiti piu volte sospoandosi trami gli indice 
 * di un gridDim * gridSize.
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include "cudaUtility.h"

#define N 6

__global__ void matrixMult(int *a, int *b, int *c)
{
	int val = 0;
	int col = blockDim.y * blockIdx.y + threadIdx.y;
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	if(row < N && col < N)
	{
		for(int k = 0; k < N; k++)
		{
			val += a[row * N + k] * b[k * N + col];		
		}
		c[row * N + col ] = val;	
	}
}

int main(void)
{
	//dichiarazione delle varibili
	int *a, *b, *c;
	size_t size = N * N * sizeof(int);

	//allocazione della memoria per le var define
	CUDA(cudaMallocManaged(&a, size));
	CUDA(cudaMallocManaged(&b, size));
	CUDA(cudaMallocManaged(&c, size));
	
	//inizializzazone della memoria
	for(int row = 0; row < N; row++)
	{
		for(int col = 0; col < N ; col++)
		{
			printf("a[row * N + col ] = %d\n", row);
			a[row * N + col ] = row;
			printf("b[row * N + col ] = %d\n", col);
			b[row * N + col ] = col*2;
			c[row * N + col ] = 0;		
		}	
	}

	//parametri per il kernel
	dim3 threads_per_block(2, 2, 1);
	dim3 number_blocks((N / threads_per_block.x ) + 1, ( N / threads_per_block.y ) + 1);
 
	//lancio il kernal
	matrixMult<<<number_blocks, threads_per_block>>>(a, b, c);
	CUDA(cudaGetLastError());

        // chiamo la versione di cpu per il check errori
	CUDA(cudaDeviceSynchronize());
	
        //comporao le risposte per vedere se sono coerrette
  	for(int row = 0; row < N; row++)
	{
		for(int col = 0; col < N ; col++)
		{
			printf("C[%d][%d] = %d\n", row, col, c[row * N + col ]);
		}	
	}

       //rilasco la memoria
       CUDA(cudaFree(a));
       CUDA(cudaFree(b));
       CUDA(cudaFree(c));
       return 0;


	
}


