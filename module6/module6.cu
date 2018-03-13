#include <stdio.h>
#include <string.h>
#include <time.h>

#define NUM_WEIGHTS 512
// References:

// http://www.cs.usfca.edu/~peter/cs625/code/cuda-dot/dot3.cu
// https://devblogs.nvidia.com/using-shared-memory-cuda-cc/
// https://github.com/JHU-EP-Intro2GPU/EN605.417.FA/blob/master/module4/host_memory.cu

// Allocate constant memory for device weights
__constant__ float const_weights_dev[NUM_WEIGHTS];
static float const_weights_host[NUM_WEIGHTS];

// Put global variables here
int REGS_PER_BLOCK, WARP_SIZE, *MAX_THREADS_DIM;
size_t TOTAL_GLOBAL_MEM, TOTAL_CONST_MEM;
int MAX_THREADS_PER_BLOCK = 512;
// Put kernel Code here
__global__
void dotProduct(float *weights, float *input, float *activityOutput) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int localId = threadIdx.x;
	
	// Initialize shared memory to store element-wise vector product
	extern __shared__ float prod[];
	prod[localId] = input[id] * weights[localId];
	
	// Synchronize threads before adding results
	__syncthreads();
	
	// Loop over array to combine input
	for(int i = blockDim.x/2; i > 0; i = i/2) {
		if(localId < i) {
			prod[localId] += prod[localId + i];
		}
		__syncthreads();
	}
	if(threadIdx.x == 0) {
		activityOutput[blockIdx.x] = prod[0];
	}
}

/*
 * dotProduct2 computes dot product using constant weights vector
 */
__global__
void dotProduct2(float * input, float * activityOutput) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int localId = threadIdx.x;
	
	extern __shared__ float prod[];
	prod[localId] = input[id] * const_weights_dev[localId];
	
	__syncthreads();
	
	// loop over array to combine input
	for(int i = blockDim.x/2; i > 0; i = i/2) {
		if(localId < i) {
			prod[localId] += prod[localId + i];
		}
		__syncthreads();
	}
	if(threadIdx.x == 0) {
		activityOutput[blockIdx.x] = prod[0];
	}
}

/*
 * dotProduct3 computes dot product using weights from register mem
 */
__global__
void dotProduct3(float *weights, float *input, float *activityOutput) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int localId = threadIdx.x;
	
	// Initialize shared memory to store element-wise vector product
	extern __shared__ float prod[];
	float weightTmp = weights[localId];
	prod[localId] = input[id] * weightTmp;
	
	// Synchronize threads before adding results
	__syncthreads();
	
	// Loop over array to combine input
	for(int i = blockDim.x/2; i > 0; i = i/2) {
		if(localId < i) {
			prod[localId] += prod[localId + i];
		}
		__syncthreads();
	}
	if(threadIdx.x == 0) {
		activityOutput[blockIdx.x] = prod[0];
	}
}

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

/*
 * allocMem
 */
void allocMem(float** inputs, float** weights, float** outputs, float** outputsConst, float** outputsRegister,
									void** inputsPinned, void** weightsPinned, void** outputsPinned,
									float** d_inputs, float** d_weights, float** d_outputs,
									int dataSize, int gridSize) {
	const unsigned int dataBytes = dataSize * sizeof(float);
 	const unsigned int gridBytes = gridSize * sizeof(float);
 	
 	// Host Pageable Mem
 	*inputs = (float*)malloc(dataBytes);
 	*weights = (float*)malloc(dataBytes);
 	*outputs = (float*)malloc(gridBytes);
 	*outputsConst = (float*)malloc(gridBytes);
 	*outputsRegister = (float*)malloc(gridBytes);
 
  // Host Pinned Mem
	checkCuda(cudaMallocHost(inputsPinned,dataBytes), "inputsPinned host allocation");
	checkCuda(cudaMallocHost(weightsPinned,dataBytes), "weightsPinned host allocation");
	checkCuda(cudaMallocHost(outputsPinned,gridBytes), "outputsPined host allocation");
	
	// Device Global Mem
	checkCuda(cudaMalloc(d_inputs, dataBytes), "d_inputs device allocation");
  checkCuda(cudaMalloc(d_weights, dataBytes), "d_weights device allocation"); 
  checkCuda(cudaMalloc(d_outputs, gridBytes), "d_outputs device allocation");
  
}
 
/*
 *
 */
void populate(float** inputs, float** weights, float* constWeights,
							float** inputsPinned, float** weightsPinned,
							float** d_inputs, float** d_weights,
							int dataSize) {
	time_t t;
	srand((unsigned) time(&t));	
	// instantiate variables with values (will be random for now)
	for(int i = 0; i < dataSize; i++) {
		(*inputs)[i] = rand() % 255;
		(*inputsPinned)[i] = (*inputs)[i];
		if(i < NUM_WEIGHTS) {
			(*weights)[i] = rand() % 30;
			(*weightsPinned)[i] = (*weights)[i];
			constWeights[i] = (*weights)[i];
		}
	}				
}


/*
 * runDot
 */
void runDot(float* inputs, float* weights, float** outputs,
						float* d_inputs, float* d_weights, float* d_outputs,
						int dataSize, int gridSize, int nExecutions, int constSelector) {
						
	const unsigned int dataBytes = dataSize * sizeof(float);
	const unsigned int sharedMem = MAX_THREADS_PER_BLOCK * sizeof(float);
	//const unsigned int sharedMem = 512 * sizeof(float);
	const unsigned int gridBytes = gridSize * sizeof(float);
	//printf("Grid size: %d\n", gridSize);
	//bool hasRunOnce = false;
	float finalOut = 0;
	for(int i = 0; i < nExecutions; i++) {
		checkCuda(cudaMemcpy(d_inputs, inputs, dataBytes, cudaMemcpyHostToDevice), "copy inputs to device");
		checkCuda(cudaMemcpy(d_weights, weights, dataBytes, cudaMemcpyHostToDevice), "copy weights to device");
		//dotProduct<<<gridSize, MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK * sizeof(float)>>>(d_inputs, d_weights, d_outputs);
		switch(constSelector) {
			case 0: dotProduct<<<gridSize, MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK * sizeof(float)>>>(d_weights, d_inputs, d_outputs);
			case 1: dotProduct2<<<gridSize, MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK * sizeof(float)>>>(d_inputs, d_outputs);
			case 2: dotProduct3<<<gridSize, MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK * sizeof(float)>>>(d_weights, d_inputs, d_outputs);
			//case 1: dotProduct2<<<gridSize, 512, 512 * sizeof(float)>>>(d_inputs, d_outputs);
		}
		cudaThreadSynchronize();
		// Copy result of dotProduct back to host
		checkCuda(cudaMemcpy(*outputs, d_outputs, gridSize * sizeof(float), cudaMemcpyDeviceToHost), "copy outputs from device to host");
	}
	
}


// Put main function here
int main(int argc, char* argv[]) {

	// MODIFICATION: We will allocate layer weights to be constant
	// For this module, just allocate weights for first layer
	// Then we will add more layers and also implement convolution
	// setup data transfer variables
	float *inputs, *weights, *outputs, *outputsConst, *outputsRegister,
			  *inputsPinned, *weightsPinned, *outputsPinned,
			  *d_inputs, *d_weights, *d_outputs;
	
	int numRuns = 1;
	int blockSizeMult = 1;
	if(argc == 2) {
		//numRuns = atoi(argv[1]);
		blockSizeMult = atoi(argv[1]);
	}
	printf("args %d\n", argc);
	// Set up timing event variables
	cudaEvent_t startEvent, stopEvent;

	// Set up variables for storing duration
	float durGlobal, durConst, durRegister;
	
	
	// Retrieve CUDA Device information
	getCUDAInfo();
	const unsigned int inputSize = (MAX_THREADS_PER_BLOCK * blockSizeMult);
	const unsigned int blockSize = MAX_THREADS_PER_BLOCK; // it should be noted that this is an integer
																												// multiple of warp size
	//const unsigned int inputSize = (512 * blockSizeMult);
	//const unsigned int blockSize = 512; // it should be noted that this is an integer
																												// multiple of warp size
	const unsigned int gridSize = ((inputSize/blockSize) > 0) ? (inputSize / blockSize) : 1;
	
	// Allocate pageable memory on host for dotproduct
	// These get initialized by passing &inputs, &weights, &outputs, dataSize, gridSize to initHostPage
	allocMem(&inputs, &weights, &outputs, &outputsConst, &outputsRegister,
					 (void**)&inputsPinned, (void**)&weightsPinned, (void**)&outputsPinned, 
					 &d_inputs, &d_weights, &d_outputs,
					 inputSize, gridSize);
	
	// Create events for timing
	checkCuda(cudaEventCreate(&startEvent), "startEvent event creation");
	checkCuda(cudaEventCreate(&stopEvent), "stopEvent event creation");
	
	// seed random number generator
	time_t t;
	srand((unsigned) time(&t));	
	populate(&inputs, &weights, const_weights_host,
						&inputsPinned, &weightsPinned,
						&d_inputs, &d_weights,
						inputSize);
	
	// Copy constant weights on host to constant weights on device:
	checkCuda(cudaMemcpyToSymbol(const_weights_dev, const_weights_host, NUM_WEIGHTS * sizeof(float)), "copy to constant dev weights");
	
	// Initialize timer for shared memory weights 
	checkCuda(cudaEventRecord(startEvent, 0), "startEvent event record");
	
	runDot(inputs, weights, &outputs,
				 d_inputs, d_weights, d_outputs,
				 inputSize, gridSize, numRuns, 0);
	// Stop timer
	checkCuda(cudaEventRecord(stopEvent, 0), "record stopEvent");
	checkCuda(cudaEventSynchronize(stopEvent), "synchronize stopEvent");
	
	// Calculate duration
	checkCuda(cudaEventElapsedTime(&durGlobal, startEvent, stopEvent), "calculate pageable duration");
	
	printf("\n\nThe dotProduct kernel executed using global memory weights %f milliseconds\n\n", durGlobal);
	
	// Initialize timer for constant memory weights 
	checkCuda(cudaEventRecord(startEvent, 0), "startEvent event record");			
	 
	runDot(inputsPinned, weights, &outputsConst,
				 d_inputs, d_weights, d_outputs,
				 inputSize, gridSize, numRuns, 1);
	
	// Stop timer
	checkCuda(cudaEventRecord(stopEvent, 0), "record stopEvent");
	checkCuda(cudaEventSynchronize(stopEvent), "synchronize stopEvent");
	
	// Calculate duration
	checkCuda(cudaEventElapsedTime(&durConst, startEvent, stopEvent), "calculate pageable duration");
	
	printf("\n\nThe dotProduct kernel executed using constant memory weights took %f milliseconds\n\n", durConst);
	
	// Initialize timer for register memory weights
	checkCuda(cudaEventRecord(startEvent, 0), "startEvent event record");
	
	runDot(inputs, weights, &outputsRegister,
				 d_inputs, d_weights, d_outputs,
				 inputSize, gridSize, numRuns, 2);
	
	// Stop timer
	checkCuda(cudaEventRecord(stopEvent, 0), "record stopEvent");
	checkCuda(cudaEventSynchronize(stopEvent), "synchronize stopEvent");
	
	// Calculate duration
	checkCuda(cudaEventElapsedTime(&durRegister, startEvent, stopEvent), "calculate pageable duration");
	
	printf("\n\nThe dotProduct kernel executed using register memory weights took %f milliseconds\n\n", durRegister);
	
	// Compute serial result
	float serialOutput = 0.00;
	int weightsInd = 0;
	for(int i = 0; i < inputSize; i++) {
		serialOutput += inputs[i]*weights[weightsInd];
		weightsInd++;
		if (i > 0 && ((i+1) % blockSize) == 0) {
			printf("Output %d -- Serial: %f, Parallel Global: %f, Parallel Const: %f, Parallel Register: %f\n", ((i+1)/blockSize), serialOutput, outputs[((i+1)/blockSize) - 1], outputsConst[((i+1)/blockSize) - 1],  outputsRegister[((i+1)/blockSize) - 1]);
			serialOutput = 0.00;
			weightsInd = 0;
		}
	}
	
	// Destroy timing events
	checkCuda(cudaEventDestroy(startEvent), "destroy startEvent");
	checkCuda(cudaEventDestroy(stopEvent), "destroy stopEvent");
	
	// Free device memory
	checkCuda(cudaFree(d_inputs), "freeing device memory");
	checkCuda(cudaFree(d_weights), "freeing device memory");
	checkCuda(cudaFree(d_outputs), "freeing device memory");
	
	// Free host pageable memory
	free(inputs);
	free(weights);
	free(outputs);
	free(outputsConst);
	free(outputsRegister);
	
	// Free host pinned memory
	checkCuda(cudaFreeHost(inputsPinned), "freeing host pinned memory");
	checkCuda(cudaFreeHost(weightsPinned), "freeing host pinned memory");
	checkCuda(cudaFreeHost(outputsPinned), "freeing host pinned memory");
	
	return(EXIT_SUCCESS);
}
