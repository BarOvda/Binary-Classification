#include "main.h"

int main(int argc, char *argv[])
{
	int myrank, size;
	clock_t start = clock();
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int N, K, limit;
	double dt, tmax, alpha, QC;
	double * allPointsCoor, *allPointsVel, *allPointsSet;

	if (size > MAX_PROCCESS) {

		size = MAX_PROCCESS;
	}

	if (myrank == MASTER)
	{
		if (!readValuesFromFile(FILE_NAME, &N, &K, &dt, &tmax, &alpha
			, &limit, &QC, &allPointsCoor, &allPointsVel, &allPointsSet)) {
			printf("file Error!\n"); fflush(NULL);
			return 1;
		}
	
		int position = 0;
		char* buffer = packItems(allPointsCoor, allPointsVel, allPointsSet, N, K, dt, tmax, alpha, limit, QC, &position);
		int p;

		for (p = 1; p<size; p++)
			MPI_Send(buffer, position, MPI_PACKED, p, 0, MPI_COMM_WORLD);
		free(buffer);
	}
	else if(myrank<size){
		int bufferSize = sizeof(int) * 3 + sizeof(double) * 4 +
			MAX_POINTS * (MAX_DIMENSIONES * (sizeof(double)) * 2 + sizeof(double));
		int position = 0;
		char* buffer = (char*)malloc(bufferSize);
		MPI_Status  s;

		MPI_Recv(buffer, bufferSize, MPI_PACKED, MASTER, 0, MPI_COMM_WORLD, &s);

		openPackedItems(&allPointsCoor, &allPointsVel, &allPointsSet, &N, &K, &dt, &tmax, &alpha, &limit, &QC, buffer);

		free(buffer);
	}
	int jobsToBeDone = (int)(tmax / dt) + 1;
	if (size > jobsToBeDone)
		size = jobsToBeDone+1;
	double* W;
	double qmin, tmin;
	
	if (myrank == MASTER) {
		W = masterProcess(size, dt, K, tmax, &qmin, &tmin);
	}
	else if(myrank<size) {
		slaveProcess(allPointsCoor, allPointsVel, allPointsSet, N, K, dt, tmax, alpha, limit, QC);
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	clock_t end = clock();
	
	if (myrank == MASTER) {
		if (W != NULL) {
			printSolution(W, K, qmin, tmin, (int)end, (int)start);
			writeSolutionToFile(OUTPUT_FILE_NAME, W, K, qmin, tmin, (int)end, (int)start);
		}
		else {
			printf("solution not found!");
		}
	}if(myrank<size)
		freeAll(allPointsSet, allPointsVel, allPointsCoor);
	
	MPI_Finalize();
	return 0;
}
int writeSolutionToFile(char* fileName, double* W, int K, double q, double time, int endTime, int startTime) {
	int i;
	FILE* fp;
	fp = fopen(fileName, "w");
	if (!fp)
	{
		printf("file open Err\n");
		return NULL;
	}
	fprintf(fp, "execute time : %d mili secs\nminimum time = %.2f q = %lf\n", endTime - startTime, time, q);
	for (i = 0; i < K + 1; i++)
		fprintf(fp,"W[%d] = %lf\n", i, W[i]); 
	fclose(fp);
	return 1;
}
void printSolution(double* W, int K,double q,double time,int endTime,int startTime) {
	int i;
	printf("execute time : %d mili secs\n", endTime - startTime);
	printf("minimum time = %.2f q = %lf\n", time, q); fflush(NULL);
	for(i = 0; i < K + 1; i++) 
		printf("W[%d] = %lf\n",i,W[i]); fflush(NULL);
	
}
void printAllPoints(double* allPointsCoor, double* allPointsVel, double* allPointsSet, int N, int K, double time) {
	int i;
	printf("Time dimension:%lf K=%dN=%d\n", time, K, N); fflush(NULL);
	for (i = 0; i<N; i++) {
		printf("\nPoint NO. : %d\n", i); fflush(NULL);
		printPoint(allPointsCoor + i*K, allPointsVel + i*K, allPointsSet[i], K);
		printf("\n");
	}
}
void printPoint(double* coors, double*vels, double set, int K) {
	int i;
	printf("Coor:(");
	for (i = 0; i < K; i++) {
		printf("%lf,", coors[i]); fflush(NULL);
	}
	printf(")\nVel:("); fflush(NULL);
	for (i = 0; i < K; i++) {
		printf("%lf,", vels[i]); fflush(NULL);
	}
	printf(" set=%lf", set); fflush(NULL);
	printf(")"); fflush(NULL);
}


int readValuesFromFile(const char* fileName, int* N, int* K, double* dt,
	double* tmax, double* alpha, int* limit, double* QC, double** allPointsCoor, double** allPointsVel, double **allPointsSet) {
	
	FILE* fp;
	int i, j;

	fp = fopen(fileName, "r");
	if (!fp)
	{
		printf("file open Err\n");
		return NULL;
	}
	fscanf(fp, "%d  %d  %lf  %lf  %lf  %d  %lf", N, K, dt, tmax, alpha, limit, QC);
	if (*N == 0)
	{
		fclose(fp);
		return NULL;
	}

	*(allPointsCoor) = (double*)malloc((*N)*(*K) * sizeof(double));
	*(allPointsVel) = (double*)malloc((*N)*(*K) * sizeof(double));
	*(allPointsSet) = (double*)malloc((*N) * sizeof(double));
	if (*allPointsCoor == NULL)
		printf("MallocPoints failed\n\n"); fflush(NULL);
	char buff[60];
	fgets(buff, 60, fp);
	for (i = 0; i < *N; i++)

	{		

		fgets(buff, 60, fp);
		char *ptr = strtok(buff, " ");
		int z = 0;
		while (ptr != NULL) {

			if (z < *K) {
				(*allPointsCoor)[i*(*K) + z] = atof(ptr);
			}
			else if (z < *K * 2) {
				(*allPointsVel)[i*(*K) + z - (*K)] = atof(ptr);
			}
			else if (z == *K * 2) {
				(*allPointsSet)[i] = atof(ptr);
			}
		
			ptr = strtok(NULL, " ");
			z++;
		}
	}
	
	fclose(fp);

	return 1;
}

char* packItems(double* allPointsCoor, double* allPointsVel, double* allPointsSet, int N, int K, double dt, double tmax, double alpha, int limit,
	double QC, int*position) {
	int bufferSize = sizeof(int) * 3 + sizeof(double) * 4 +
		N*(K*(sizeof(double)) * 2 + sizeof(double));
	char* buffer = (char*)malloc(bufferSize);

	int i;
	MPI_Pack(&K, 1, MPI_INT, buffer, bufferSize, position, MPI_COMM_WORLD);
	MPI_Pack(&N, 1, MPI_INT, buffer, bufferSize, position, MPI_COMM_WORLD);
	MPI_Pack(&limit, 1, MPI_INT, buffer, bufferSize, position, MPI_COMM_WORLD);
	MPI_Pack(&dt, 1, MPI_DOUBLE, buffer, bufferSize, position, MPI_COMM_WORLD);
	MPI_Pack(&tmax, 1, MPI_DOUBLE, buffer, bufferSize, position, MPI_COMM_WORLD);
	MPI_Pack(&alpha, 1, MPI_DOUBLE, buffer, bufferSize, position, MPI_COMM_WORLD);
	MPI_Pack(&QC, 1, MPI_DOUBLE, buffer, bufferSize, position, MPI_COMM_WORLD);
	MPI_Pack(allPointsCoor, K*N, MPI_DOUBLE, buffer, bufferSize, position, MPI_COMM_WORLD);
	MPI_Pack(allPointsVel, K*N, MPI_DOUBLE, buffer, bufferSize, position, MPI_COMM_WORLD);
	MPI_Pack(allPointsSet, N, MPI_DOUBLE, buffer, bufferSize, position, MPI_COMM_WORLD);

	return buffer;
}

void openPackedItems(double** allPointsCoor, double** allPointsVel, double** allPointsSet, int *N, int *K, double *dt, double *tmax, double *alpha, int *limit,
	double *QC, char* buffer) {
	int position = 0;
	int bufferSize = sizeof(int) * 3 + sizeof(double) * 4 +
		MAX_POINTS* (MAX_DIMENSIONES* (sizeof(double)) * 2 + sizeof(double));//sizeOf(K+N+dt+tmax+QC+limit+alpha
																			//+allPointsCoor+allPointsVel+allPointsSet)
	int i;

	MPI_Unpack(buffer, bufferSize, &position, K, 1, MPI_INT, MPI_COMM_WORLD);
	MPI_Unpack(buffer, bufferSize, &position, N, 1, MPI_INT, MPI_COMM_WORLD);
	MPI_Unpack(buffer, bufferSize, &position, limit, 1, MPI_INT, MPI_COMM_WORLD);
	MPI_Unpack(buffer, bufferSize, &position, dt, 1, MPI_DOUBLE, MPI_COMM_WORLD);
	MPI_Unpack(buffer, bufferSize, &position, tmax, 1, MPI_DOUBLE, MPI_COMM_WORLD);
	MPI_Unpack(buffer, bufferSize, &position, alpha, 1, MPI_DOUBLE, MPI_COMM_WORLD);
	MPI_Unpack(buffer, bufferSize, &position, QC, 1, MPI_DOUBLE, MPI_COMM_WORLD);

	*(allPointsCoor) = (double*)malloc((*N)*(*K) * sizeof(double));
	*(allPointsVel) = (double*)malloc((*N)*(*K) * sizeof(double));
	*(allPointsSet) = (double*)malloc((*N) * sizeof(double));
	MPI_Unpack(buffer, bufferSize, &position, *allPointsCoor, (*N)*(*K), MPI_DOUBLE, MPI_COMM_WORLD);
	MPI_Unpack(buffer, bufferSize, &position, *allPointsVel, (*N)*(*K), MPI_DOUBLE, MPI_COMM_WORLD);
	MPI_Unpack(buffer, bufferSize, &position, *allPointsSet, (*N), MPI_DOUBLE, MPI_COMM_WORLD);

}
char* packItemsForOutPut(double t, double* W, int K, double q, int* position) {
	int bufferSize = sizeof(double) * 2 + sizeof(double)*(K + 1);
	char* buffer = (char*)malloc(bufferSize);

	int i;
	MPI_Pack(&t, 1, MPI_DOUBLE, buffer, bufferSize, position, MPI_COMM_WORLD);
	MPI_Pack(&q, 1, MPI_DOUBLE, buffer, bufferSize, position, MPI_COMM_WORLD);
	MPI_Pack(W, K + 1, MPI_DOUBLE, buffer, bufferSize, position, MPI_COMM_WORLD);
	return buffer;
}
double* unpackItemsForOutput(char* buffer, double* q, double* t, int K) {
	
	int position = 0;
	int bufferSize = sizeof(double) * 2 + sizeof(double)*(K + 1);

	int i;

	MPI_Unpack(buffer, bufferSize, &position, t, 1, MPI_DOUBLE, MPI_COMM_WORLD);

	MPI_Unpack(buffer, bufferSize, &position, q, 1, MPI_DOUBLE, MPI_COMM_WORLD);
	double* W = (double*)malloc((K + 1) * sizeof(double));

	MPI_Unpack(buffer, bufferSize, &position, W, K + 1, MPI_DOUBLE, MPI_COMM_WORLD);

	return W;
}

double firstJobs(int numOfProccesses, int jobsToBeDone, double dt) {
	int id;
	double processTimeWork = 0;
	for (id = 1; id < numOfProccesses; id++)
	{
		MPI_Send(&processTimeWork, 1, MPI_DOUBLE, id, DATA_TAG, MPI_COMM_WORLD);
		processTimeWork += dt;
	}
	return processTimeWork;
}

double* dynamicManagment(double processTimeWork,int* processReached, int numOfProccesses, double dt, int jobsToBeDone, double* q, double * minT, int K) {
	int source, recvtag, jobSent;
	double* W;
	int bufferSize = sizeof(double) * 2 + sizeof(double)*(K + 1);
	char* buffer = (char*)malloc(bufferSize);
	MPI_Status  s;
	for (jobSent = numOfProccesses - 1; jobSent < jobsToBeDone; jobSent++) {
		
		int position = 0;
		
		MPI_Recv(buffer, bufferSize, MPI_PACKED, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &s);
		
		source = s.MPI_SOURCE;		
		recvtag = s.MPI_TAG;

		
		if (recvtag == QUALITY_REACHED) {
			W = unpackItemsForOutput(buffer, q, minT, K);
			*processReached = source;
			return W;
		}
		MPI_Send(&processTimeWork, 1, MPI_DOUBLE, source, DATA_TAG, MPI_COMM_WORLD);//one last sending not needed											
		processTimeWork += dt;
	}
	return NULL;
}
double* reciveLastTasks(int numOfProcesses,int processReached, int jobsToBeDone, double* minT, double* q, int K, double* W) {
	int id, source, recvtag, position;
	double recvTime, T, qnew;

	int bufferSize = sizeof(double) * 2 + sizeof(double)*(K + 1);
	char* buffer = (char*)malloc(bufferSize);
	MPI_Status  s;
	for (id = 1; id < numOfProcesses; id++) {
		if (id != processReached) {
			position = 0;
			MPI_Recv(buffer, bufferSize, MPI_PACKED, id, MPI_ANY_TAG, MPI_COMM_WORLD, &s);
			source = s.MPI_SOURCE;
			recvtag = s.MPI_TAG;
			double* Wnew = unpackItemsForOutput(buffer, &qnew, &T, K);

			if (recvtag == QUALITY_REACHED&&T < *minT) {//Update result if q<QC and the work time is smaller
				W = Wnew;
				*q = qnew;
				*minT = T;
			}
		}
	}
	return W;
}
void signalTarminitionToSlaves(int numOfProccesses, int jobsToBeDone) {
	int id;
	for (id = 1; id < numOfProccesses; id++) {
		MPI_Send(NULL, 0, MPI_DOUBLE, id, TERMINATION_TAG, MPI_COMM_WORLD);
	}
}

double* masterProcess(int numOfProcesses, double dt, int K, double tmax, double* qmin, double* tmin) {

	int jobsToBeDone = (int)(tmax / dt) + 1;
	*tmin = tmax + 1.0;
	
	int firstProcessReached = numOfProcesses + 1;
	
	double processTimeWork = firstJobs(numOfProcesses, jobsToBeDone, dt);
	double* W = dynamicManagment(processTimeWork,&firstProcessReached, numOfProcesses, dt, jobsToBeDone, qmin, tmin, K);
	W = reciveLastTasks(numOfProcesses, firstProcessReached,jobsToBeDone, tmin, qmin, K, W);
	signalTarminitionToSlaves(numOfProcesses, jobsToBeDone);
	return W;
}
void slaveProcess(double* allPointsCoor, double* allPointsVel, double* allPointsSet, int N, int K, double dt, double tmax, double alpha, int limit,
	double QC) {
	double myTime,lastWorkTime=0.0,deltat, q;
	double* W = (double*)malloc(sizeof(double)*(K + 1));
	int tag, Nmis;
	MPI_Status status;
	
	while (1) {
		MPI_Recv(&myTime, 1, MPI_DOUBLE, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		deltat = myTime - lastWorkTime;
		tag = status.MPI_TAG;
		if (tag != TERMINATION_TAG) {

			int NK[2] = { N,K };
			double currenttimenDt[2] = { deltat,dt };
			addWithCuda(allPointsCoor, allPointsVel, NK, currenttimenDt);
			initW(W, K);
			
			int allPointsClassified = cycleThrowPoints(allPointsCoor, &W, allPointsSet, N, K, dt, tmax, alpha, limit, QC);//all W =100?
			
			if (allPointsClassified != 1) {//not all points claasifed
				Nmis = countNmis(W, K, N, allPointsCoor, allPointsSet);
				q = (double)Nmis / (double)N;
				int position = 0;
				char* buffer = packItemsForOutPut(myTime, W, K, q, &position);

				if (q < QC) {//The quality is satisfaies
					MPI_Send(buffer, position, MPI_PACKED, MASTER, QUALITY_REACHED, MPI_COMM_WORLD);
				}
				else {//Quality is not satisfaies
					MPI_Send(buffer, position, MPI_PACKED, MASTER, QUALITY_NOT_SATISFAIES, MPI_COMM_WORLD);
				}

			}
			else {//all points classifed
				q = 0;
				int position = 0;
				char* buffer = packItemsForOutPut(myTime, W, K, q, &position);

				MPI_Send(buffer, position, MPI_PACKED, MASTER, QUALITY_REACHED, MPI_COMM_WORLD);
			}
			lastWorkTime = myTime;
		}
		else {
			return;
		}
	}
}
void initW(double* W, int K) {
	int i;
	for (i = 0; i < K + 1; i++) {
		W[i] = 0;
	}
}

int countNmis(double* W, int K, int N, double* allPointsCoor, double* allPointsSet) {
	int Nmis = 0, i;
#pragma omp parallel for reduction (+:Nmis)
	for (i = 0; i < N; i++) {
		double funcXi = f(allPointsCoor + i*K, K, W);
		int Psign = funcXi >= 0 ? 1 : -1;
		if (Psign*allPointsSet[i]< 0) {
			Nmis++;
		}
	}
	return Nmis;
}
void freeAll(double* allPointsSet, double* allPointsVel, double* allPointsCoor) {
	free(allPointsSet);
	free(allPointsVel);
	free(allPointsCoor);
}