/*https://cdac.in/index.aspx?id=ev_hpc_gpu-comp-nvidia-cuda-streams#hetr-cuda-prog-cuda-streams*/

#include <stdio.h> 
#include <time.h> 
#include <cuda.h> 

//#define sizeOfArray 1024*1024

// 1. Execute Everything synchronously
// 2. Execute Everything asynchronously
// 3. Execute Memcpy Synchronously and kernel launch Asynchronously
// 4. Execute Memcpy asynchronously and kernel launch synchronously
// ** Measure variability across the above by chaning data size **
// ** Utilize Multiple Streams like in the example: https://devblogs.nvidia.com/gpu-pro-tip-cuda-7-streams-simplify-concurrency/ **
 
// Put global variables here
int REGS_PER_BLOCK, WARP_SIZE, MAX_THREADS_PER_BLOCK, *MAX_THREADS_DIM;
size_t TOTAL_GLOBAL_MEM, TOTAL_CONST_MEM;

__global__ void arrayAddition(int *device_a, int *device_b, int *device_result, int sizeOfArray)
{

	int threadId = threadIdx.x + blockIdx.x * blockDim.x ;

	if (threadId < sizeOfArray) 
        device_result[threadId]= device_a[threadId]+device_b[threadId]; 
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

bool verifyResult(int *host_a, int *host_b, int **host_result,
					 				int num_streams, int sizeOfArray) {
	bool testPassed = true;
	for(int i = 0; i < num_streams; i++) {
		for(int j = 0; j < sizeOfArray; j++) {
			if((host_a[j] + host_b[j]) != host_result[i][j]) {
  			testPassed = false;
  			printf("Case %d Failed on Stream %d, iteration %d: %d + %d = Serial Result: %d ParallelResult: %d\n", 0,i, j, host_a[j], host_b[j], host_a[j] + host_b[j], host_result[i][j]);
  		}
		}
	}
	
	return testPassed;				 				
}

float testStream(int **device_a, int **device_b, int **device_result,
					 int *host_a, int *host_b, int **host_result,
					 cudaStream_t *streams, cudaEvent_t start, cudaEvent_t stop,
					 int num_streams, int sizeOfArray,int sync_test, 
					 int gridSize, int blockSize) {
	float elapsedTime = 0;
	for(int i = 0; i < num_streams; i++) {
		// Create new stream on each iteration
		checkCuda(cudaStreamCreate(&streams[i]), "cudaStreamCreate");
		
		// Allocate device memory for each iteration
  	cudaMalloc( ( void**)& device_a[i], sizeOfArray * sizeof ( **device_a ) ); 
  	cudaMalloc( ( void**)& device_b[i],sizeOfArray * sizeof ( **device_b ) ); 
  	cudaMalloc( ( void**)& device_result[i], sizeOfArray * sizeof ( **device_result ) ); 
		cudaEventRecord(start,0);
  	switch(sync_test) {
  		// Synchronous memcpy and kernel launch
  		case 0: 
  			checkCuda(cudaMemcpy(device_a[i], host_a,sizeOfArray * sizeof ( int ), cudaMemcpyHostToDevice), "host to device async memcpy 1"); 
  			checkCuda(cudaMemcpy(device_b[i], host_b, sizeOfArray * sizeof ( int ), cudaMemcpyHostToDevice), "host to device async memcpy 2"); 
  			arrayAddition <<<gridSize, blockSize, blockSize * sizeof(int)>>>(device_a[i], device_b[i], device_result[i], sizeOfArray); 
  			checkCuda(cudaMemcpy(host_result[i], device_result[i], sizeOfArray * sizeof ( int ), cudaMemcpyDeviceToHost), "device to host async memcpy");
  			break;
  		// Asynchronous memcpy and synchronous kernel launch
  		case 1:
  			checkCuda(cudaMemcpyAsync(device_a[i], host_a,sizeOfArray * sizeof ( int ), cudaMemcpyHostToDevice, streams[i]), "host to device async memcpy 1"); 
  			checkCuda(cudaMemcpyAsync(device_b[i], host_b, sizeOfArray * sizeof ( int ), cudaMemcpyHostToDevice, streams[i]), "host to device async memcpy 2"); 
  			arrayAddition <<<gridSize, blockSize, blockSize * sizeof(int)>>>(device_a[i], device_b[i], device_result[i], sizeOfArray); 
  			checkCuda(cudaMemcpyAsync(host_result[i], device_result[i], sizeOfArray * sizeof ( int ), cudaMemcpyDeviceToHost, streams[i]), "device to host async memcpy");
  			break;
  		// Synchronous memcpy and asynchronous kernel launch
  		case 2:
  			checkCuda(cudaMemcpy(device_a[i], host_a,sizeOfArray * sizeof ( int ), cudaMemcpyHostToDevice), "host to device async memcpy 1"); 
  			checkCuda(cudaMemcpy(device_b[i], host_b, sizeOfArray * sizeof ( int ), cudaMemcpyHostToDevice), "host to device async memcpy 2"); 
  			arrayAddition <<<gridSize, blockSize, blockSize * sizeof(int), streams[i]>>>(device_a[i], device_b[i], device_result[i], sizeOfArray); 
  			checkCuda(cudaMemcpy(host_result[i], device_result[i], sizeOfArray * sizeof ( int ), cudaMemcpyDeviceToHost), "device to host async memcpy");
  			break;
  		case 3: 
  			checkCuda(cudaMemcpyAsync(device_a[i], host_a,sizeOfArray * sizeof ( int ), cudaMemcpyHostToDevice, streams[i]), "host to device async memcpy 1"); 
  			checkCuda(cudaMemcpyAsync(device_b[i], host_b, sizeOfArray * sizeof ( int ), cudaMemcpyHostToDevice, streams[i]), "host to device async memcpy 2");
  			arrayAddition <<<gridSize, blockSize, blockSize * sizeof(int), streams[i]>>>(device_a[i], device_b[i], device_result[i], sizeOfArray); 
  			checkCuda(cudaMemcpyAsync(host_result[i], device_result[i], sizeOfArray * sizeof ( int ), cudaMemcpyDeviceToHost, streams[i]), "device to host async memcpy");
  			break;	
  		default:
  			break;
  	}

		
	}
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop); 
	return elapsedTime;
}


/* Check for safe return of all calls to the device */ 

int main ( int argc, char **argv ) 
{ 

	// Take block size and data size as input
	// Check block size to make sure it does not exceed the maximum number of threads per block
  int blockSize = 512;
  int sizeOfArray = 1024 * 1024;
	if(argc == 3) {
		//numRuns = atoi(argv[1]);
		blockSize = atoi(argv[1]);
		sizeOfArray = atoi(argv[2]);
	}
  const int num_streams = 10;
  int *host_a, *host_b, *host_result[num_streams]; 
  int *device_a[num_streams], *device_b[num_streams], *device_result[num_streams]; 
  getCUDAInfo();
  int gridSize =  ((sizeOfArray % blockSize) == 0) ? (sizeOfArray / blockSize) : ((sizeOfArray / blockSize) + 1);
  // Create array to store each stream:
  cudaStream_t streams[num_streams]; 

  cudaEvent_t start, stop; 
  float elapsedTime0, elapsedTime1 = 0, elapsedTime2 = 0, elapsedTime3 = 0 ; 

  cudaEventCreate( &start ); 
  cudaEventCreate( &stop ); 
	
	// Allocate host memory
	cudaHostAlloc((void **)&host_a, sizeOfArray*sizeof(int), cudaHostAllocDefault);
  cudaHostAlloc((void **)&host_b, sizeOfArray*sizeof(int), cudaHostAllocDefault);
  for(int i = 0; i < num_streams; i++) {
  	cudaHostAlloc((void **)&host_result[i], num_streams*sizeOfArray*sizeof(int), cudaHostAllocDefault);
  }
  
	
	// Initiailize host memory
  for(int index = 0; index < sizeOfArray; index++) 
  { 
   host_a[index] = rand()%10; 
   host_b[index] = rand()%10; 
  } 
	
	//cudaEventRecord(start,0);
	elapsedTime0 = testStream(device_a, device_b, device_result, host_a, host_b, host_result, streams, start, stop, num_streams,  sizeOfArray, 0, gridSize, blockSize);
  //cudaEventRecord(stop, 0);
  //cudaEventSynchronize(stop);
  //cudaEventElapsedTime(&elapsedTime0, start, stop); 
  verifyResult(host_a, host_b, host_result, num_streams, sizeOfArray);
  
  //cudaEventRecord(start, 0);
	elapsedTime1 = testStream(device_a, device_b, device_result, host_a, host_b, host_result, streams, start, stop, num_streams,  sizeOfArray, 1, gridSize, blockSize);
  //cudaEventRecord(stop, 0);
  //cudaEventSynchronize(stop);
  //cudaEventElapsedTime(&elapsedTime1, start, stop);
  verifyResult(host_a, host_b, host_result, num_streams, sizeOfArray);
  
  
  //cudaEventRecord(start, 0);
	elapsedTime2 = testStream(device_a, device_b, device_result, host_a, host_b, host_result, streams, start, stop, num_streams,  sizeOfArray, 2, gridSize, blockSize);
  //cudaEventRecord(stop, 0);
  //cudaEventSynchronize(stop);
  //cudaEventElapsedTime(&elapsedTime2, start, stop);
  verifyResult(host_a, host_b, host_result, num_streams, sizeOfArray);
  
  
  //cudaEventRecord(start, 0);
	elapsedTime3 = testStream(device_a, device_b, device_result, host_a, host_b, host_result, streams, start, stop, num_streams, sizeOfArray, 3, gridSize, blockSize);
  //cudaEventRecord(stop, 0);
  //cudaEventSynchronize(stop);
  //cudaEventElapsedTime(&elapsedTime3, start, stop);
  verifyResult(host_a, host_b, host_result, num_streams, sizeOfArray);
  /*cudaStream_t stream; 
  cudaStreamCreate(&stream); */
 

  //printf("*********** CDAC - Tech Workshop : hyPACK-2013 \n"); 
  printf("\n Size of array : %d \n", sizeOfArray); 
  printf("\n Time taken for synchronous memcpy and synchronous kernel launch: %f ms \n", elapsedTime0); 
  printf("\n Time taken for asynchronous memcpy and synchronous kernel launch: %f ms \n", elapsedTime1); 
  printf("\n Time taken for synchronous memcpy and asynchronous kernel launch: %f ms \n", elapsedTime2); 
  printf("\n Time taken for asynchronous memcpy and asynchronous kernel launch: %f ms \n", elapsedTime3); 
	
	for(int i = 0; i < num_streams; i++) {
		cudaStreamDestroy(streams[i]);	
	}
	cudaEventDestroy(stop);
	cudaEventDestroy(start);
  cudaFreeHost(host_a); 
  cudaFreeHost(host_b); 
  cudaFreeHost(host_result); 
  cudaFree(device_a); 
  cudaFree(device_b); 
  cudaFree(device_result); 

  return 0; 
}
