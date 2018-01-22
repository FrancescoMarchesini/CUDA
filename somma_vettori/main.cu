
/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//libreria cuda
#include <cuda.h>
// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Helper functions and utilities to work with CUDA
//#include <helper_functions.h>
//#include <helper_cuda.h>
/*
 * somma di vettori in cpu
 */
#define LOG_CUDA "[CUDA] "

void vecAdd(float* h_A, float* h_B, float* h_C, int n)
{
	int i;
	for(i=0; i<n; i++) h_C[i] = h_A[i] + h_B[i];
}

void cudaAllocaMemoria(void** source, int size)
{
	printf("%salloco memoria sulla GPU grandezza = %d \n",LOG_CUDA, size);
	cudaError_t err = cudaMalloc(source, size);
	if(err != cudaSuccess){
		printf("%s%s in %s alla linea %d\n", LOG_CUDA, cudaGetErrorString(err), __FILE__, __LINE__);
				exit(EXIT_FAILURE);
	}
}

void cudaCopiaMemoria(float* dst, float* src, int byteSize, enum cudaMemcpyKind dir)
{
	char* a;
	switch(dir){
		case 1 : a = "cudaMemcpyHostToDevice";break;
		case 2 : a = "cudaMemcpyDeviceToHost"; break;
	}
	printf("%sCopio i dati modo = %s  grandezza = %d \n", LOG_CUDA,  a, byteSize);
	cudaError_t err = cudaMemcpy(dst, src, byteSize, dir);
	if(err != cudaSuccess){
		printf("%s%s in %s alla linea %d\n",LOG_CUDA, cudaGetErrorString(err), __FILE__, __LINE__);
				exit(EXIT_FAILURE);
	}
}

//__global__: idica che il metodo seguente è un metodo kernel. E che quindi può essere chiamata da un
//			  metodo dell'host per generare una griglia di thread sul device. Puo essere chiamata solo
//			  dal codice dell host
__global__ void vecAddKernel(float* A, float* B, float* C, int n)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	printf("%s thread numero: %d", LOG_CUDA, i);
	if(i < n ) C[i] = A[i] + B[i];
}

void cudaVecAdd(float* h_A, float* h_B, float* h_C, int n)
{
	int size = n * sizeof(float);
	printf("%sgrandezza vettore = %d grandezza byte float=%4.2f\n",LOG_CUDA, n, sizeof(float));
	printf("%sdetermino quando memoria mi occore size = n * sizeof(flaot) = %d \n" ,LOG_CUDA, size);

	//gli indizirizzi dei vettori nella "devices memory"
	float *d_A, *d_B, *d_C;

	//args 1: un puntatore generico all'indirizzo del vettore che deve essere allocato
	//args 2: la grandezza in byte del vettore da allocare
	//->A_d punterea quindi alla "device memory" per il vettore a cui A_d punta
	cudaAllocaMemoria((void **)&d_A, size);
	cudaAllocaMemoria((void **)&d_B, size);
	cudaAllocaMemoria((void **)&d_C, size);

	//1.1: copio della momeria
	//arg 1: Destinanzione: puntatore alla destinanzione di dove verra copia l'oggetto
	//arg 2: Sorgente: puntatore all'oggetto da copiare
	//arg 3: Byte: quantitia di dati da copiare, lunghezza vettore in byte
	//arg 4: Tipo di Copia: da .. a .., host / devices
	cudaCopiaMemoria(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaCopiaMemoria(d_B, h_B, size, cudaMemcpyHostToDevice);

	printf("%sLancio dell'esecuzione in parallelo della funzione vecAdd\n", LOG_CUDA);
	// <<< : parametri configurazione del kernel
	// size: numero di blocchi di thread nella griglia
	// 256.0: il numero di threah all'interno di ogni blocco
	/*printf("%snumero blocchi= %d\n"
		   "%snumero thread per blocco 256\n"
		   "%stread totali = %4.2f\n",
		   LOG_CUDA, ceil(size/256.0)*256, LOG_CUDA, LOG_CUDA, ceil(size/256.0)*256 );*/
	int threadPerBlock = 256;
	int blockPerGrid = (n + threadPerBlock -1 ) / threadPerBlock;
	//int blockPerGrid = threadPerBlock;
	int tot = threadPerBlock * blockPerGrid;
	printf("%sblocchi = %d thread = %d tot = %d\n", LOG_CUDA, blockPerGrid, threadPerBlock, tot);

	vecAddKernel<<<blockPerGrid, threadPerBlock>>>(d_A, d_B, d_C, n);

	cudaThreadSynchronize();


	printf("%sCopia C dalla memoria del device e rilascia la memoria del device\n", LOG_CUDA);
	//ritorno il dato
	cudaCopiaMemoria(h_C, d_C, size, cudaMemcpyDeviceToHost);

	//cancello la memoria occupata dal vettore
	//arg : il puntatore da liberare
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

}

#define LOG_MAIN "[MAIN] "
int main(int argc, char **argv)
{
	printf("%s--------Start---------------\n",LOG_MAIN);
	int n = 5;
	float a[5]={1.0, 50.6, 89.4, 2.3, 5.7};
	float b[5]={3.0, 30.6, 8.4, 25.3, 25.7};
	float c[5];

	//vecAdd(a, b, c, n);
	//for(int i=0; i<n; i++) printf("index: %d, float: %4.2f \n", i, c[i]);

	cudaVecAdd(a, b, c, 5);
	for(int i=0; i<n; i++) printf("%sindex: %d, float: %.1f \n",LOG_MAIN,i, c[i]);
	printf("%s--------End-----------------\n",LOG_MAIN);
	return 0;
};
