#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define FILE_NAME "C:\\Users\\cudauser\\Desktop\\data1.txt"
#define OUTPUT_FILE_NAME "C:\\Users\\cudauser\\Desktop\\output.txt"
#define MASTER 0
#define MAX_PROCCESS 10
#define MIN_PROCCESS 2
#define MAX_DIMENSIONES 20
#define DATA_TAG 0
#define TERMINATION_TAG 1
#define QUALITY_NOT_SATISFAIES 0
#define QUALITY_REACHED 1
#define MAX_POINTS 500000 

/*advence points to current work time 
O(K)*/
extern cudaError_t addWithCuda(double *allPointsCoor, double* allPointsVel, int * NK, double* curruentTimeAnddt);
/*check classification limit times on N points in current time work.
return 1 :all point classified else return 0
O(limit*N*K)*/
extern int cycleThrowPoints(double* myPoints, double** W, double* allPointsSets, int N, int K, double dt, double tmax, double alpha, int limit,
	double QC);
void printPoint(double* coors, double*vels, double set, int K);
void printAllPoints(double* allPointsCoor, double* allPointsVel, double* allPointsSet, int N, int K, double time);
int readValuesFromFile(const char* fileName, int* N, int* K, double* dt,
	double* tmax, double* alpha, int* limit, double* QC, double** allPointsCoor, double** allPointsVel, double **allPointsSet);
/*pack all detailes for distribution to processes */
char* packItems(double* allPointsCoor, double* allPointsVel, double* allPointsSet, int N, int K, double dt, double tmax, double alpha, int limit,
	double QC, int*position);
/*the processes open the packed items*/
void openPackedItems(double** allPointsCoor, double** allPointsVel, double** allPointsSet, int *N, int *K, double *dt, double *tmax, double *alpha, int *limit,
	double *QC, char* buffer);
/*pack items for sending result to master process*/
char* packItemsForOutPut(double t, double* W, int K, double q, int* position);
double* unpackItemsForOutput(char* buffer, double* q, double* t, int K);
/*recive all remain tasks after first solution find or all jobs sent*/
double* reciveLastTasks(int numOfProcesses, int processReached, int jobsToBeDone, double* minT, double* q, int K, double* W);
/*the main program of Maste
if solution found return W else return NULL*/
double* masterProcess(int numOfProcesses, double dt, int K, double tmax, double* qmin, double* tmin);
/*slaves program*/
void slaveProcess(double* allPointsCoor, double* allPointsVel, double* allPointsSet, int N, int K, double dt, double tmax, double alpha, int limit,
	double QC);
/*Master managment of send and recive Tasks
end when all jobs sent or first soultion recived*/
double* dynamicManagment(double processTimeWork, int* processReached, int numOfProccesses, double dt, int jobsToBeDone, double* q, double * minT, int K);
/*send jobs to all proccess*/
double firstJobs(int numOfProccesses, int jobsToBeDone, double dt);
void signalTarminitionToSlaves(int numOfProccesses, int jobsToBeDone);
double f(double* Xi, int K, double* W);
/*cycle throw all N points and return the number of misclassified points
O(N*K)*/
int countNmis(double* W, int K, int N, double* allPointsCoor, double* allPointsSet);
/*initial W values to zero*/
void initW(double* W, int K);
void freeAll(double* allPointsSet, double* allPointsVel, double* allPointsCoor);
void printSolution(double* W, int K, double q, double time, int endTime, int startTime);
int writeSolutionToFile(char* fileName, double* W, int K, double q, double time, int endTime, int startTime);