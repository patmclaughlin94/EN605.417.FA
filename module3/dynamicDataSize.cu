#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
// Step 1: Query Device for maximum block sizes and thread sizes
// (not really sure what we care about)
// Step 2: Take in user input to specify data dimensions
// Step 3: Check to make sure user inputs match with specified program device queries

int TOTAL_GLOBAL_MEM, REGS_PER_BLOCK, WARP_SIZE, MAX_THREADS_PER_BLOCK, *MAX_THREADS_DIM;

__global__
void vec_mult_add(const unsigned int * A, const unsigned int * B, const unsigned int c, unsigned int * D, int num_elements)
{
    /*int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }*/

    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < num_elements) {
	D[id] = c*A[id] + B[id];
    } 
} 
// Device Query Information:
void getHardwareConstraints() {

//=============================Gets number of cuda devices===========================================
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

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
	printf("The %s has:\n\t-%zu total bytes of global memory\n\t-%d registers per block\n\t-%d threads per warp\n\t-A maximum of %d threads per block\n\t-And a maximum thread dimension of %d x %d x %d\n", deviceProp.name, TOTAL_GLOBAL_MEM, REGS_PER_BLOCK, WARP_SIZE, MAX_THREADS_PER_BLOCK, MAX_THREADS_DIM[0], MAX_THREADS_DIM[1], MAX_THREADS_DIM[2]); 
	// What I think we care about:
	// 1. totalGlobalMem
	// 2. regsPerBlock
	// 3. warpSize (i.e. numThreadsPerBlock (is this equal to regsPerBlock??)
	// 4. maxThreadsperBlock
	// 5. maxThreadsDim[3]
    }

}

void printVector(unsigned int * A, int numElements) {
    printf("\n[");
    for(int i = 0; i < numElements; i++){
        printf("\n   %d",A[i]);
    }
    printf("\n]\n");
}

void checkOutput(unsigned int * A, int numElements) {
    bool incorrect = false;
    for(int i = 0; i < numElements; i++) {
	if(A[i] != (2*i + i)) {
	    incorrect = true;
	    printf("\nIncorrect value of %d at index %d... should be %d\n", A[i], i, (2*i + i));   
	}
    } 
    if(!incorrect) {
    	printf("\nOutput is Correct!\n");
    }
}
void elementWiseMult_Add() {

}

int main(int argc, char* argv[]) {
    
    int numThreads;
    int blockSize;
    numThreads = atoi(argv[1]);
    blockSize = atoi(argv[2]);
    
    getHardwareConstraints();    
    /*std::cout << "This program will take two arrays of integers from 0-1023, multiply them the first array by a constant of 2, and add the first array to the second array\n";
    std::cout << "You must first specify a number of threads you would like to use for this operation: ";
    std::cin >> numThreads;
    std::cout << "\nAnd a block size: ";
    std::cin >> blockSize;*/
    printf("numThreads: %d and blockSize: %d\n", numThreads, blockSize);
    // keep track of errors from cuda operations
    cudaError_t err = cudaSuccess;

// this will be taken from cmd
    int numElements = 1024;

    size_t size = numElements * sizeof(int);
    // Allocate the host input vector A
    unsigned int *h_A = (unsigned int *)malloc(size);

    // Allocate the host input vector B
    unsigned int *h_B = (unsigned int *)malloc(size);

    // Allocate the host output vector C
    unsigned int *h_D = (unsigned int *)malloc(size);
    
    //srand(time(NULL));
    int c = 2;
    
    // initialize host input vectors
    for(int i = 0; i < numElements; i++){
	h_A[i] = i;
	h_B[i] = i;
    }
    // DEBUG
    //printf("%d * \n", c);
    //printVector(h_A, numElements);
    //printf(" + \n");
    //printVector(h_B, numElements);

    // Allocate device input vector
    // Allocate the device input vector A
    unsigned int *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Allocate device input vector B
    unsigned int *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate device output vector D
    // Allocate device input vector B
    unsigned int *d_D = NULL;
    err = cudaMalloc((void **)&d_D, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector D (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    //printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // calculate appropriate thread size 
    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = blockSize;
    int blocksPerGrid = numThreads / threadsPerBlock;    
//int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    //printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    vec_mult_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, c, d_D, numElements);
    err = cudaGetLastError();

    if(err != cudaSuccess) {
	fprintf(stderr, "Failed to execute kernel (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(h_D, d_D, size, cudaMemcpyDeviceToHost);
    
    if(err != cudaSuccess) {
	fprintf(stderr, "Failed to copy kernel back to host (error code %s)!\n", cudaGetErrorString(err));
    	exit(EXIT_FAILURE);
    }
    checkOutput(h_D, numElements);
    //printf(" = \n");
    //printVector(h_D, numElements);
    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_D);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_D);

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Done\n");
   
    //printf("Here are random numbers %d %d\n", (rand() % 11), (rand() % 11));
    
    return(EXIT_SUCCESS);
}
