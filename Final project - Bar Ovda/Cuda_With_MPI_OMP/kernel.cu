
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


double f(double* Xi, int K, double* W);
int cheackClassifier(double fXiSign, double set);
void whightUpdate(int K, double alpha, double* Xi, int sign, double** W);
double sign(double val);



cudaError_t addWithCuda(double *allPointsCoor, double* allPointsVel, int * NK, double* curruentTimeAnddt);
void checkError(cudaError_t cudaStatus,
	double *dev_allPointsCoor, int *dev_NK, double *dev_allPointsVel,
	const char* errorMessage);

void freeCudaMemory(double *dev_allPointsCoor, double *dev_allPointsVel, int *dev_NK);


__global__ void addKernel(double *allPointsCoor, double* allPointsVel, int * NK, double* curruentTimeAnddt)
{
	int K = NK[1], N = NK[0];
	int i = threadIdx.x + (blockIdx.x * blockDim.x);
	int j;
	double currentTime = curruentTimeAnddt[0], dt = curruentTimeAnddt[1];

	for (j = 0; j < K; j++) {
		allPointsCoor[i*K + j] += allPointsVel[i*K + j] * currentTime;
	}
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(double *allPointsCoor, double* allPointsVel, int * NK, double* curruentTimeAnddt)
{
	char errorBuffer[100];
	double* dev_allPointsCoor = 0;
	double* dev_allPointsVel = 0;
	double* dev_currentTimeAnddt = 0;
	int *dev_NK = 0;
	int K = NK[1], N = NK[0];


	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	checkError(cudaStatus, dev_allPointsCoor, dev_NK, dev_allPointsVel,
		"cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_allPointsCoor, K*N * sizeof(double));
	checkError(cudaStatus, dev_allPointsCoor, dev_NK, dev_allPointsVel, "cudaMalloc failed!");

	cudaStatus = cudaMalloc((void**)&dev_NK, sizeof(int) * 2);
	checkError(cudaStatus, dev_allPointsCoor, dev_NK, dev_allPointsVel, "cudaMalloc failed!");


	cudaStatus = cudaMalloc((void**)&dev_allPointsVel, (K*N) * sizeof(double));
	checkError(cudaStatus, dev_allPointsCoor, dev_NK, dev_allPointsVel, "cudaMalloc failed!");

	cudaStatus = cudaMalloc((void**)&dev_currentTimeAnddt, 2 * sizeof(double));
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_allPointsCoor, allPointsCoor, K*N * sizeof(double), cudaMemcpyHostToDevice);
	checkError(cudaStatus, dev_allPointsCoor, dev_NK, dev_allPointsVel, "cudaMemcopy failed!");

	cudaStatus = cudaMemcpy(dev_allPointsVel, allPointsVel, K*N * sizeof(double), cudaMemcpyHostToDevice);

	cudaStatus = cudaMemcpy(dev_currentTimeAnddt, curruentTimeAnddt, 2 * sizeof(double), cudaMemcpyHostToDevice);
	checkError(cudaStatus, dev_allPointsCoor, dev_NK, dev_allPointsVel, "cudaMemcopy failed!");

	cudaStatus = cudaMemcpy(dev_NK, NK, 2 * sizeof(int), cudaMemcpyHostToDevice);
	// Launch a kernel on the GPU with one thread for each element.
	addKernel << <1000, N / 1000 >> > (dev_allPointsCoor, dev_allPointsVel, dev_NK, dev_currentTimeAnddt);//N max is 500000, max

																										  // Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	sprintf(errorBuffer, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	checkError(cudaStatus, dev_allPointsCoor, dev_NK, dev_allPointsVel, errorBuffer);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	sprintf(errorBuffer, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	checkError(cudaStatus, dev_allPointsCoor, dev_NK, dev_allPointsVel, errorBuffer);

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(allPointsCoor, dev_allPointsCoor, sizeof(double)*N*K, cudaMemcpyDeviceToHost);
	checkError(cudaStatus, dev_allPointsCoor, dev_NK, dev_allPointsVel, "cudaMemcopy failed!");

	freeCudaMemory(dev_allPointsCoor, dev_allPointsVel, dev_NK);

	return cudaStatus;
}

void freeCudaMemory(double *dev_allPointsCoor, double *dev_allPointsVel, int *dev_NK)
{
	cudaFree(dev_allPointsCoor);
	cudaFree(dev_allPointsVel);
	cudaFree(dev_NK);
}
void checkError(cudaError_t cudaStatus,
	double *dev_allPointsCoor, int *dev_NK, double *dev_allPointsVel,
	const char* errorMessage)
{
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, errorMessage);
		fprintf(stderr, "\n");
		freeCudaMemory(dev_allPointsCoor, dev_allPointsVel, dev_NK);
	}
}



double sign(double val) {
	return (val >= 0) ? 1 : -1;
}
void whightUpdate(int K, double alpha, double* Xi, int sign, double** W) {
	int i;
	(*W)[0] = (*W)[0] + alpha*(double)sign*(-1.0);
	for (i = 1; i < K + 1; i++) {
		double diff = (alpha)* (double)sign * Xi[i - 1] * (-1.0);
		(*W)[i] = (*W)[i] + diff;
	}
}

int cheackClassifier(double fXiSign, double set) {
	return fXiSign*set > 0 ? 1 : 0;
}
double f(double* Xi, int K, double* W) {
	int i;
	double result = W[0];
	for (i = 1; i < K + 1; i++) {
		result += W[i] * Xi[i - 1];
	}
	return result;
}

int cycleThrowPoints(double* myPoints, double** W, double* allPointsSets, int N, int K, double dt, double tmax, double alpha, int limit,
	double QC) {
	int i, j, Q = 1;
					
	double funcXi, funcXisign;
	for (i = 0; i < limit; i++) {
		int minIndex = N;
#pragma omp parallel for reduction(min : minIndex)
		for (j = 0; j < N; j++) {
			funcXi = f(myPoints + j*K, K, *W);
			funcXisign = sign(funcXi);

			if (cheackClassifier(funcXisign, allPointsSets[j]) == 0)
				if (j < minIndex) {
					minIndex = j;
					break;	//added: the first missclassified point that thread found stops thread itretion
				}
		}
		if (minIndex == N) {
			return 1;
		}
		else {
			funcXi = f(myPoints + minIndex, K, *W);
			funcXisign = sign(funcXi);
			whightUpdate(K, alpha, myPoints + minIndex*K, funcXisign, W);
		}
	}
	return 0;

}
