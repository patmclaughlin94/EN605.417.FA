#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <math.h>
#include "kmeans.h"
using namespace std;
// Put global variables here
int REGS_PER_BLOCK, WARP_SIZE, *MAX_THREADS_DIM;
size_t TOTAL_GLOBAL_MEM, TOTAL_CONST_MEM;
int MAX_THREADS_PER_BLOCK = 512;

// Put utility functions here
/*
 *checkCuda: will check to see if there is an error returned by CUDA runtime
 */
inline
void checkCuda(cudaError_t errMsg, const char* errContext)
{
	if(errMsg != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error From %s: %s\n", errContext, cudaGetErrorString(errMsg));
		exit(EXIT_FAILURE);
	}
}

/*
 * getCUDAInfo() - originally named "getHardwareContraints in module 3
 * 							 - this function will get CUDA information pertaining to the hardware 
 * 							   on which we are operating... the code can then reason on these reports to determine
 * 								 the best way to structure memory transfers between the host and device
 */
void getCUDAInfo() {
	//=============================Gets number of cuda devices===========================================
    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "Failed deviceCount load");

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
    {
        printf("There are no available device(s) that support CUDA\n");
    }
    else
    {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }
//=============================Gets number of cuda devices===========================================
    
    // for each device found, store this device in some type of object
    int device;

    for (device = 0; device < deviceCount; device++) {
	
			// Sets the context of the device so that we know which device we are working with if there
			// are multiple	
			cudaSetDevice(device);
  		cudaDeviceProp deviceProp;

			// gets the "properties" struct that stores the properties of a device
			// from this property struct, we can query the limitations of this device
  		cudaGetDeviceProperties(&deviceProp, device);

			printf("\nDevice: %d \"%s\"\n===========================================\n", device, deviceProp.name);
	
			TOTAL_GLOBAL_MEM = deviceProp.totalGlobalMem;
			REGS_PER_BLOCK = deviceProp.regsPerBlock;
			WARP_SIZE = deviceProp.warpSize;
			//MAX_THREADS_PER_BLOCK = deviceProp.maxThreadsPerBlock;
			MAX_THREADS_DIM = deviceProp.maxThreadsDim;
			TOTAL_CONST_MEM = deviceProp.totalConstMem; 
	printf("The %s has:\n\t-%zu total bytes of global memory\n\t-%zu bytes of constant memory\n\t-%d registers per block\n\t-%d threads per warp\n\t-A maximum of %d threads per block\n\t-A maximum thread dimension of %d x %d x %d\n", deviceProp.name, TOTAL_GLOBAL_MEM,TOTAL_CONST_MEM, REGS_PER_BLOCK, WARP_SIZE, MAX_THREADS_PER_BLOCK, MAX_THREADS_DIM[0], MAX_THREADS_DIM[1], MAX_THREADS_DIM[2]); 
	// What I think we care about:
	// 1. totalGlobalMem
	// 2. regsPerBlock
	// 3. warpSize (i.e. numThreadsPerBlock (is this equal to regsPerBlock??)
	// 4. maxThreadsperBlock
	// 5. maxThreadsDim[3]
    }
}

// Kmeans stuff=====================================================
__device__ void km_divide(Flower* f, long denom) {
	f->sepal_length = f->sepal_length / ((float) denom);
	f->sepal_width = f->sepal_width / ((float) denom);
	f->petal_length = f->petal_length / ((float) denom);
	f->petal_width = f->petal_width / ((float) denom);
	return; 
}

__device__ float euclid_dist(Flower* f, Centroid* c) {
	float d_s_l = f->sepal_length - c->sepal_length;
  float d_s_w = f->sepal_width - c->sepal_width;
  float d_p_l = f->petal_length - c->petal_length;
  float d_p_w = f->petal_width- c->petal_width;
  return sqrtf((d_s_l*d_s_l) + (d_s_w*d_s_w) + (d_p_l*d_p_l) + (d_p_w*d_p_w));
}

// For each data point, calculate assign a cluster depending on which centroid is the closest
__global__ void group_by_cluster(Flower * flowers, Centroid * centroids,
																 int num_centroids, int num_flowers) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	int i = 0;
	
	float dist = 100000000.0f;
	
	if(idx < num_flowers) {
		for(i = 0; i < num_centroids; i++) {
			float curr_dist = euclid_dist(&flowers[idx], &centroids[i]);
			if(curr_dist < dist) {
				dist = curr_dist;
				flowers[idx].clust = i;
		}
		}
		
	}																 

}

__global__ void sum_points_cluster(Flower *flowers, Centroid * centroids, Sum * sum,
																	 int num_centroids, int num_flowers)
{
	// 
	extern __shared__ Sum s_sum[];
	
	int threadId = threadIdx.x;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	// initialize global memory
	if (idx < num_centroids) {
		
		sum[idx].s_l_sum = 0.0;
		sum[idx].s_w_sum = 0.0;
		sum[idx].p_l_sum = 0.0;
		sum[idx].p_w_sum = 0.0;
		sum[threadId].num_points = 0.0;
	}
	
	// initialize shared memory
	if(threadId < num_centroids) {
		s_sum[threadId].s_l_sum = 0.0;
		s_sum[threadId].s_w_sum = 0.0;
		s_sum[threadId].p_l_sum = 0.0;
		s_sum[threadId].p_w_sum = 0.0;
		s_sum[threadId].num_points = 0.0;
	}
	__syncthreads();
	if(idx < num_flowers) {
		int i = flowers[idx].clust;
		atomicAdd(&s_sum[i].s_l_sum, flowers[idx].sepal_length);
		atomicAdd(&s_sum[i].s_w_sum, flowers[idx].sepal_width);
		atomicAdd(&s_sum[i].p_l_sum, flowers[idx].petal_length);
		atomicAdd(&s_sum[i].p_w_sum, flowers[idx].petal_width);
		atomicAdd(&s_sum[i].num_points, 1);
	}
	__syncthreads();
	
	if(threadId < num_centroids) {
		atomicAdd(&sum[threadId].s_l_sum, s_sum[threadId].s_l_sum);
		atomicAdd(&sum[threadId].s_w_sum, s_sum[threadId].s_w_sum);
		atomicAdd(&sum[threadId].p_l_sum, s_sum[threadId].p_l_sum);
		atomicAdd(&sum[threadId].p_w_sum, s_sum[threadId].p_w_sum);
		atomicAdd(&sum[threadId].num_points, s_sum[threadId].num_points);
	}
}

__global__ void update_centroids(Centroid * centroids, Sum * sums, int num_centroids) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < num_centroids) {
		if(sums[idx].num_points > 0) {
			centroids[idx].sepal_length = sums[idx].s_l_sum / sums[idx].num_points;
			centroids[idx].sepal_width = sums[idx].s_w_sum/ sums[idx].num_points;
			centroids[idx].petal_length = sums[idx].p_l_sum / sums[idx].num_points;
			centroids[idx].petal_width = sums[idx].p_w_sum / sums[idx].num_points;
		}
	}
}

void getClusterResults(Flower * flowers, int num_flowers, int num_clusters) {
	int currentSpecies = 0;
	int i;
	int j;
	int * assignedClusters = new int[num_clusters];
	for(i = 0; i < num_clusters; i++) {
		assignedClusters[i] = 0;
	}
	//float * accuracy = new float[num_clusters];
	float accuracy;
	int count = 0;
	for(i = 0; i < num_flowers; i++) {
		if(currentSpecies != flowers[i].species || i == (num_flowers-1)) {
			int max = 0;
			for(j = 0; j < num_clusters; j++) {
				if(assignedClusters[j] > max) {
					max = assignedClusters[j];
				}
			}
			accuracy =(float) max / (float)count;
			printf("Accuracy for species %d: %f\n", currentSpecies, accuracy);
			for(j = 0; j < num_clusters; j++) {
				assignedClusters[j] = 0;
			}
			count = 0;
			currentSpecies = flowers[i].species;
		}
		assignedClusters[flowers[i].clust]++;
		count++;
	}
	delete [] assignedClusters;
}
//==================================================================

int main(int argc, char** argv) {
	
    KMeans kmeans;
    // 1. Parse iris.csv into vector of "Flower" structs
	// Initial factors will likely be specified via commandline
	// For now they are hardcoded
  string data_file = "Iris.csv";
	int numClusters = 3;
	int clusterInitType = 0; // Initiate cluster at random sample of dataset
	int numIterations = 10;
	
	// specify via cmd args
	if(argc == 4) {
		string tmp(argv[1]);
		data_file = tmp;
		numClusters = atoi(argv[2]);
		numIterations =atoi(argv[3]);
	}
  kmeans.parseDataset(data_file.c_str());
    
  // Print datafile
 // kmeans.printDataset();
	
	// Initialize centroids
	kmeans.initClusters(clusterInitType, numClusters);
	kmeans.printCentroids();
	
	// Initialize CUDA variables
	//getCUDAInfo();
	
	// Allocate CUDA memory
	vector<Flower> flow_vec = kmeans.getFlowers();
	vector<Centroid> cent_vec = kmeans.getCentroids();
	Flower flowers_host[flow_vec.size()];
	copy(flow_vec.begin(), flow_vec.end(), flowers_host);
	Flower* flowers_dev;
	Flower* flowers_old;
	Sum * sums_dev;
	Centroid centroids_host[cent_vec.size()];
	copy(cent_vec.begin(), cent_vec.end(), centroids_host);
	Centroid* centroids_dev;
	
	
	int num_points = flow_vec.size();
	int num_centroids = cent_vec.size();
	Flower* flowers_res = (Flower*)malloc(num_points*sizeof(Flower));
	Centroid* centroids_res = (Centroid*)malloc(num_centroids*sizeof(Centroid));
	//int h_res = 1;
	int *dres;
	
	cudaMalloc((void**) &dres, sizeof(int));
	cudaMalloc((void**) &flowers_old, sizeof(Flower) * num_points);
	cudaMalloc((void**) &flowers_dev, sizeof(Flower) * num_points);
	cudaMalloc((void**) &centroids_dev, sizeof(Centroid) * num_centroids);
	cudaMalloc((void**) &sums_dev, sizeof(Sum) * num_centroids);
	
	cudaMemcpy(flowers_dev, flowers_host, sizeof(Flower) * num_points, cudaMemcpyHostToDevice);
	cudaMemcpy(centroids_dev, centroids_host, sizeof(Centroid) * num_centroids, cudaMemcpyHostToDevice);
	
	// Set up timing event variables
	cudaEvent_t startEvent, stopEvent;
	
	// Create events for timing
	checkCuda(cudaEventCreate(&startEvent), "startEvent event creation");
	checkCuda(cudaEventCreate(&stopEvent), "stopEvent event creation");
	float duration;
	
	// Initialize timer 
	checkCuda(cudaEventRecord(startEvent, 0), "startEvent event record");
	for(int i = 0; i < numIterations; i++) {
		// Set clusters
		group_by_cluster<<<(int)ceil(num_points/100),100>>>(flowers_dev, centroids_dev, num_centroids, num_points);
		cudaDeviceSynchronize();
	
		// Sum values over each dimension (will be used to compute new mean)
		sum_points_cluster<<<(int)ceil(num_points/100), 100, num_centroids*sizeof(Centroid)>>>(flowers_dev, centroids_dev, sums_dev, num_centroids, num_points);
		cudaDeviceSynchronize();
	
		update_centroids<<<1, 3	>>>(centroids_dev, sums_dev, num_centroids);
		cudaDeviceSynchronize();
	}
	
	// Stop timer
	checkCuda(cudaEventRecord(stopEvent, 0), "record stopEvent");
	checkCuda(cudaEventSynchronize(stopEvent), "synchronize stopEvent");
	
	// Calculate duration
	checkCuda(cudaEventElapsedTime(&duration, startEvent, stopEvent), "calculate pageable duration");
	
	cudaMemcpy(flowers_res, flowers_dev, sizeof(Flower) * num_points, cudaMemcpyDeviceToHost);
	cudaMemcpy(centroids_res, centroids_dev, sizeof(Centroid) * num_centroids, cudaMemcpyDeviceToHost);
	
	printf("kmeans CUDA results:\n");
	getClusterResults(flowers_res, num_points, num_centroids);
	
	for (int i = 0; i < num_centroids; i++) {
		printf("Cluster %d: %f %f %f %f  %d\n", i,
               centroids_res[i].sepal_length, centroids_res[i].sepal_width,
               centroids_res[i].petal_length, centroids_res[i].petal_width,
			   i);
	}
	
	printf("\n\nkmeans CUDA run took %f milliseconds\n\n", duration);
	
	
	checkCuda(cudaFree((void*)flowers_dev), "Free flowers_dev");
	checkCuda(cudaFree((void*)centroids_dev), "Free centroids_dev");
	checkCuda(cudaFree((void*)flowers_old), "Free flowers_old");
	checkCuda(cudaFree((void*)dres), "Free dres");
	checkCuda(cudaFree((void*)sums_dev), "Free dres");
	
	free(flowers_res);
	free(centroids_res);
  return(EXIT_SUCCESS);
  
  // 214 Oak Street Dumore PA 18512
}
