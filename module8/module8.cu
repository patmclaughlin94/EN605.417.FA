/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

#include <stdio.h> 
#include <time.h> 
#include <cuda.h> 
/* we need these includes for CUDA's random number stuff */
#include <curand.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#define index(i,j,ld) (((j)*(ld))+(i))

// Put global variables here
int REGS_PER_BLOCK, WARP_SIZE, MAX_THREADS_PER_BLOCK, *MAX_THREADS_DIM;
size_t TOTAL_GLOBAL_MEM, TOTAL_CONST_MEM;
// Kernels go here
/* this GPU kernel function is used to initialize the random states */
__global__ void init(unsigned int seed, curandState_t* states) {
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  /* we have to initialize the state */
  curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
              tid, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &states[tid]);
}
 
/* this GPU kernel takes an array of states, and an array of ints, and puts a random int into each */
__global__ void randoms(curandState_t* states, float* numbers) {
  /* curand works like rand - except that it takes a state as a parameter */
  const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  numbers[tid] = float(curand(&states[tid]) % 100);
}
 
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

inline 
void checkCUBLAS(cublasStatus_t errMsg, const char* errContext)
{
	if(errMsg != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "CUBLAS Error From %s\n", errContext);
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
			MAX_THREADS_PER_BLOCK = deviceProp.maxThreadsPerBlock;
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
// Easily print matrix of arbitrary dimension
void printMat(float*P,int uWP,int uHP){
  int i,j;
  for(i=0;i<uHP;i++){

      printf("\n");

      for(j=0;j<uWP;j++)
          printf("%f ",P[index(i,j,uHP)]);
  }
  printf("\n\n");
}

/* Host implementation of a simple version of sgemm */
static void simple_sgemm(int n, float alpha, const float *A, const float *B,
                         float beta, float *C)
{
    int i;
    int j;
    int k;

    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < n; ++j)
        {
            float prod = 0;

            for (k = 0; k < n; ++k)
            {
                prod += A[k * n + i] * B[j * n + k];
            }

            C[j * n + i] = alpha * prod + beta * C[j * n + i];
        }
    }
}

int main ( int argc, char **argv ) 
{ 

	// Take block size and data size as input
	// Check block size to make sure it does not exceed the maximum number of threads per block
  int blockSize = 512;
  int N = 5;
	if(argc == 3) {
		blockSize = atoi(argv[1]);
		N = atoi(argv[2]);
	}
	
  float host_a[N*N], host_b[N*N], host_result[N*N], *host_result_ref; 
  float *device_a, *device_b, *device_result; 
  float alpha = 1;
  float beta = 0;
  // Timing
	float elapsedTime = 0;
  cudaEvent_t start, stop;
  cudaEventCreate( &start ); 
  cudaEventCreate( &stop );
  
  
  cublasHandle_t handle;
  
  getCUDAInfo();
  int gridSize =  (((N*N) % blockSize) == 0) ? ((N*N) / blockSize) : (((N*N) / blockSize) + 1);
  
  // create random data
  /* CUDA's random number library uses curandState_t to keep track of the seed value
     we will store a random state for every thread  */
  curandState_t* states1;
  curandState_t* states2;

  
  // Allocate device memory for matrices
  checkCuda(cudaMalloc((void **)&device_a, N * N * sizeof(device_a[0])), "allocating device_a");
  checkCuda(cudaMalloc((void **)&device_b, N * N * sizeof(device_b[0])), "allocating device_b");
  checkCuda(cudaMalloc((void **)&device_result, N * N * sizeof(device_result[0])), "allocating device_result");
  
  /* allocate space on the GPU for the random states */
  cudaMalloc((void**) &states1, N * N * sizeof(curandState_t));
  cudaMalloc((void**) &states2, N * N * sizeof(curandState_t));

  /* invoke the GPU to initialize all of the random states */
  time_t t, t2;
  init<<<gridSize, blockSize, blockSize * sizeof(int)>>>(time(&t), states1);
  init<<<gridSize, blockSize, blockSize * sizeof(int)>>>(time(&t2), states2);

  /* invoke the kernel to get some random numbers */
  randoms<<<gridSize, blockSize, blockSize * sizeof(int)>>>(states1, device_a);
  randoms<<<gridSize, blockSize, blockSize * sizeof(int)>>>(states2, device_b);

  /* copy the random numbers back */
  cudaMemcpy(host_a, device_a, N * N * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(host_b, device_b, N * N * sizeof(int), cudaMemcpyDeviceToHost);

  /* print them out */
  printMat(host_a, N, N);
  printMat(host_b, N, N);
  /*for (int i = 0; i < N*N; i++) {
    printf("%f\t%f\n", host_a[i], host_b[i]);
  }*/

  /* free the memory we allocated for the states and numbers */
  cudaFree(states1);
  cudaFree(states2);
  
  // Create CUBLAS Handle
  checkCUBLAS(cublasCreate(&handle), "creating CUBLAS handle");
  checkCUBLAS(cublasSetVector(N*N, sizeof(host_a[0]), host_a, 1, device_a, 1), "seting d_a vector");
  checkCUBLAS(cublasSetVector(N*N, sizeof(host_b[0]), host_b, 1, device_b, 1), "seting d_b vector");
  
  // Time naive sgemm
	cudaEventRecord(start,0);
  simple_sgemm(N, alpha, host_a, host_b, beta, host_result);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop); 
  
  printf("simple_sgemm elapsedTime: %f\n", elapsedTime);
  host_result_ref = host_result;
  
  /* Performs operation using cublas */
  // time cuBLAS sgemm
  cudaEventRecord(start, 0);
  checkCUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, device_a, N, device_b, N, &beta, device_result, N), "calculating Sgemm");
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("elapsed time for cublas_sgemm: %f\n", elapsedTime);
  
  /* Read the result back*/
  checkCUBLAS(cublasGetVector(N*N, sizeof(host_result[0]), device_result, 1, host_result, 1), "get d_c vector");
  
  /* check result against reference */
  float error_norm = 0;
  float ref_norm = 0;
  
  for(int i = 0; i < N*N; ++i) {
  	int diff = host_result_ref[i] - host_result[i];
  	error_norm += (float) diff * diff;
  	ref_norm += (float) host_result_ref[i] * host_result_ref[i];
  }
  
  // Calculate error between serial and GPU calculation
  error_norm = (float)(sqrt((double)error_norm));
  ref_norm = (float)(sqrt((double)ref_norm));
  
  // Free Memory
  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_result);
  
  // Destroy cublas handle
  checkCUBLAS(cublasDestroy(handle), "destroying handle");
  
  // If error is small enough, GPU and Serial sgemm calculation yielded same result
  if(error_norm / ref_norm < 1e-6f) {
  	printf("CUBLAS test passed.\n");
  	exit(EXIT_SUCCESS);
  }
  
  // Otherwise 
  else {
  	printf("CUBLAS test failed.\n");
  	exit(EXIT_FAILURE);
  }
  
}
