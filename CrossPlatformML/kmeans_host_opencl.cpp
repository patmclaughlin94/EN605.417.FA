#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include "kmeans.h"
using namespace std;
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

// Declare global
int MAX_WORK_GROUP_SIZE;
// Function to check and handle OpenCL errors
inline void
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void CL_CALLBACK contextCallback(
                                 const char * errInfo,
                                 const void * private_info,
                                 size_t cb,
                                 void * user_data)
{
	std::cout << "Error occured during context use: " << errInfo << std::endl;
	// should really perform any clearup and so on at this point
	// but for simplicitly just exit.
	exit(1);
}

// Get first platform you find
// TODO: Return all platforms
void getPlatform(cl_platform_id * platform) {
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id * platformIDs;
    
    // Get Number of platforms
    errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkErr((errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS),
             "clGetPlatformIDs");
    
    // Get Platform IDs
    platformIDs = (cl_platform_id *) malloc(sizeof(cl_platform_id) * numPlatforms);
    errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
    checkErr((errNum != CL_SUCCESS) ? errNum: (numPlatforms <= 0 ? -1 : CL_SUCCESS), "clGetPlatformIDs");
    
    // Use the first platform
    *platform = platformIDs[0];
    
    // Clean up your mess...
    delete [] platformIDs;
}
void getDeviceInfo(cl_device_id device, cl_device_info info, int type) {
	size_t paramValueSize;
	// first query for size of parameter
	checkErr(clGetDeviceInfo(device, info, 0, NULL, &paramValueSize), "getDeviceInfo");
	switch(type) {
		// size_t
		case 0:
			size_t value;
			checkErr(clGetDeviceInfo(device, info, paramValueSize, &value, NULL), "getDeviceInfo");
			
			MAX_WORK_GROUP_SIZE = (int) value;
			break;
		// char[]
		case 1:
			char strValue[1024];
			checkErr(clGetDeviceInfo(device, info, paramValueSize, &strValue, NULL), "getDeviceInfo");
			break;
		default: break;
	}
	
}
// Get first device you find
// TODO: Return all devices and assess which device makes the most sense for each context
void getDevice(cl_platform_id platform, cl_device_id * device) {
    cl_int errNum;
    cl_uint numDevices;
    cl_device_id * deviceIDs;
    
    errNum = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
    if(errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND) {
        checkErr(errNum, "clGetDeviceIDs");
    } else if(numDevices > 0) {
        deviceIDs = (cl_device_id *) malloc(sizeof(cl_device_id) * numDevices);
        errNum = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, deviceIDs, NULL);
        checkErr(errNum, "clGetDeviceIDs");
    }
    
    // Use first device
    *device = deviceIDs[0];
	
	getDeviceInfo(*device, CL_DEVICE_NAME, 1);
    
    // Clean up your mess...
    delete [] deviceIDs;
}


// Create context given 1 platform and a set of devices
cl_context createContext(cl_platform_id platform, cl_device_id * deviceIDs, cl_uint numDevices){
	cl_int errNum;
    cl_context context = NULL;
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platform,
        0
    };
    context = clCreateContext(
                              contextProperties,
                              numDevices,
                              deviceIDs,
                              &contextCallback,
                              NULL,
                              &errNum);
	checkErr(errNum, "clCreateContext");
    return context;
    
}

// Create a program for the given context and given kernel file
cl_program createProgram(const char * fileName, cl_context context,
                         cl_device_id * deviceIDs, cl_uint numDevices) {
    cl_int errNum;
    cl_program program;
    
    std::ifstream kernelFile(fileName, std::ios::in);
    if (!kernelFile.is_open())
    {
        std::cerr << "Failed to open file for reading: " << fileName << std::endl;
        return NULL;
    }
    
    std::ostringstream oss;
    oss << kernelFile.rdbuf();
    
    std::string srcStdStr = oss.str();
    const char *srcStr = srcStdStr.c_str();
    program = clCreateProgramWithSource(context, 1,
                                        (const char**)&srcStr,
                                        NULL, NULL);
    if (program == NULL)
    {
        std::cerr << "Failed to create CL program from source." << std::endl;
        return NULL;
    }
    
    errNum = clBuildProgram(program, numDevices, deviceIDs, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, deviceIDs[0], CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);
        
        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);
        return NULL;
    }
    
    return program;
    
}
// Create Mem Objects
bool createMemReadWrite(cl_context context, cl_mem * memObjects, int numObjects, Flower * flowers_h, Centroid * centroids_h, int * array_sizes) {
	for(int i = 0; i < numObjects; i++) {
		switch(i) {
			case 0: 
				memObjects[i] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(Flower) * array_sizes[i], flowers_h, NULL);
				break;
			case 1: 
				memObjects[i] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(Centroid) * array_sizes[i], centroids_h, NULL);
				break;
			case 2: 
				memObjects[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(Sum) * array_sizes[i], NULL, NULL);
				break;
			default: break;
		}
		if(memObjects[i] == NULL) {
			cerr << "Error creating memory object " << i << endl;
			return false;
		}
	}
	return true;
}
// Set arguments
void setArgsBlocking(cl_kernel * kernels, cl_mem * memReadWrite, int * intArgs, int workItems) {
	// kmeans assignment kernel
	checkErr(clSetKernelArg(kernels[0], 0, sizeof(cl_mem), &memReadWrite[0]), "set first flower arg"); // flowers
	checkErr(clSetKernelArg(kernels[0], 1, sizeof(cl_mem), &memReadWrite[1]), "set first centroid arg"); // centroids
	checkErr(clSetKernelArg(kernels[0], 2, sizeof(int), &intArgs[0]), "set first num_flowers arg"); // num_flowers
	checkErr(clSetKernelArg(kernels[0], 3, sizeof(int), &intArgs[1]), "set first num_centroids arg"); // num_centroids
	
	// sum kernel
	checkErr(clSetKernelArg(kernels[1], 0, sizeof(cl_mem), &memReadWrite[0]), "set second flower arg"); // flowers
	checkErr(clSetKernelArg(kernels[1], 1, sizeof(cl_mem), &memReadWrite[2]), "set second centroid arg"); // sums
	checkErr(clSetKernelArg(kernels[1], 2, 3*workItems, NULL), "set second local_sum arg"); // local sum argument
	checkErr(clSetKernelArg(kernels[1], 3, sizeof(int), &intArgs[0]), "set second num_flowers arg"); // num_flowers
	checkErr(clSetKernelArg(kernels[1], 4, sizeof(int), &intArgs[1]), "set second num_centroids arg"); // num_centroids
	
	// kmeans update kernel
	checkErr(clSetKernelArg(kernels[2], 0, sizeof(cl_mem), &memReadWrite[2]), "set third sums arg"); // sums
	checkErr(clSetKernelArg(kernels[2], 1, sizeof(cl_mem), &memReadWrite[1]), "set third centroids arg"); // centroids
	checkErr(clSetKernelArg(kernels[2], 2, sizeof(int), &intArgs[2]), "set third work_groups arg"); // work_groups
	checkErr(clSetKernelArg(kernels[2], 3, sizeof(int), &intArgs[1]), "set third num_centroids arg"); // num_centroids
	
}
int getNearestPowOf2(int base) {
	int nearestPowOf2 = 1;
	int current = (int)pow(2, nearestPowOf2);
	while(base > current) {
		nearestPowOf2++;
		current = (int)pow(2,nearestPowOf2);
	}
	return current;
}
void createKernels(cl_kernel * kernels, cl_program program) {
    cl_int errNum;
    kernels[0] = clCreateKernel(program, "kmeans_assignment", &errNum);
    kernels[1] = clCreateKernel(program, "sum", &errNum);
    kernels[2] = clCreateKernel(program, "kmeans_update", &errNum);
    checkErr(errNum, "clCreateKernel");
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
			int clustMax = 0;
			for(j = 0; j < num_clusters; j++) {
				if(assignedClusters[j] > max) {
					max = assignedClusters[j];
					clustMax = j;
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
void runKernelBlocking(cl_command_queue queue, cl_kernel * kernels, cl_mem * memReadWrite, int array_size, int workItems, int numIterations, int num_centroids) {
	size_t globalWorkSize[1];
	globalWorkSize[0] = (size_t)array_size;
	
	size_t localWorkSize[1];
	localWorkSize[0] = (size_t)workItems;
	
	
	cl_event events[3];
	cl_ulong ev_start_time;
	cl_ulong ev_end_time;
	size_t return_bytes;
	for(int i = 0; i < numIterations; i++) {
	// execute kmeans_assignment
	checkErr(clEnqueueNDRangeKernel(queue, kernels[0], 1, NULL, globalWorkSize,
																	localWorkSize, 0, NULL, &events[0]), "enqueue kernel 1");
																
	// execute sum
	checkErr(clEnqueueNDRangeKernel(queue, kernels[1], 1, NULL, globalWorkSize,
																	localWorkSize, 1, &events[0], &events[1]), "enqueue kernel 2");
	
	// execute kmeans_update
	checkErr(clEnqueueNDRangeKernel(queue, kernels[2], 1, NULL, globalWorkSize,
																	localWorkSize, 1, &events[1], &events[2]), "enqueue kernel 3");
	}
	checkErr(clWaitForEvents(1, &events[0]), "wait for evt 1");
	Flower * flowers_res = new Flower[array_size];
	Centroid * centroid_res = new Centroid[num_centroids];
	checkErr(clEnqueueReadBuffer(queue, memReadWrite[0], CL_TRUE,
															 0, array_size * sizeof(Flower), flowers_res,
															 0, NULL, NULL), "reading flower buffer");
	checkErr(clEnqueueReadBuffer(queue, memReadWrite[1], CL_TRUE,
															 0, num_centroids * sizeof(Centroid), centroid_res,
															 0, NULL, NULL), "reading flower buffer");
		
	checkErr(clWaitForEvents(1, &events[1]), "wait for evt 2");			  
	checkErr(clWaitForEvents(1, &events[2]), "wait for evt 3");
	/*for(int i = 0; i < array_size; i++) {
		printf("%f %f %f %f %d %d %d\n",
               flowers_res[i].sepal_length, flowers_res[i].sepal_width,
               flowers_res[i].petal_length, flowers_res[i].petal_width,
               flowers_res[i].species, flowers_res[i].clust, flowers_res[i].id);
	}	*/
	printf("Results: \n");
	getClusterResults(flowers_res, array_size, num_centroids);
	for(int i = 0; i < num_centroids; i++) {
		printf("Centroid %d: %f %f %f %f\n", i, centroid_res[i].sepal_length, centroid_res[i].sepal_width,
               centroid_res[i].petal_length, centroid_res[i].petal_width);
	}
	checkErr(clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &ev_start_time, &return_bytes), "evt profiling1");
	checkErr(clGetEventProfilingInfo(events[2], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &ev_end_time, &return_bytes), "evt profiling2");
	
	double run_time = (double)(ev_end_time - ev_start_time);
	
	printf("kmeans OpenCL run took %f ms\n", run_time*1e-6);
	
}

void getResults(cl_command_queue queue, cl_mem * memReadWrite, Flower * flowers_res, Centroid * centroids_res, int * array_sizes) {
	checkErr(clEnqueueReadBuffer(queue, memReadWrite[0], CL_TRUE, 0, array_sizes[0] * sizeof(Flower), flowers_res, 0, NULL, NULL), "reading buffer");
	checkErr(clEnqueueReadBuffer(queue, memReadWrite[1], CL_TRUE, 0, array_sizes[1] * sizeof(Centroid), centroids_res, 0, NULL, NULL), "reading buffer");

}

// Best to isolate below code in a "k_means" main because this setup is required
// regardless of if we are using OpenCL, CUDA, or plain old C++
int main(int argc, char** argv) {
	cl_int errNum;
    // OpenCL boilerplate objects
	cl_platform_id platform = 0;
	cl_device_id device = 0;
	cl_context context = NULL;
	cl_program program = NULL;
	cl_command_queue queue;
	cl_kernel * kernels = new cl_kernel[3];
	cl_mem * memReadWrite = new cl_mem[3];
	
  KMeans kmeans;
  // 1. Parse iris.csv into vector of "Flower" structs
  string data_file = "Iris.csv";
	int numClusters = 3;
	int numIterations = 10;
	int clusterInitType = 0; // Initiate cluster at random sample of dataset
	
	// specify via cmd args
	if(argc == 4) {
		string tmp(argv[1]);
		data_file = tmp;
		numClusters = atoi(argv[2]);
		numIterations =atoi(argv[3]);
	}
	
  kmeans.parseDataset(data_file.c_str());
    
  // Print datafile
  //kmeans.printDataset();
	
	// Initialize centroids
	kmeans.initClusters(clusterInitType, numClusters);
	kmeans.printCentroids();
	
	vector<Flower> flowers = kmeans.getFlowers();
	int num_flowers = flowers.size();
	Flower * flowers_h = new Flower[num_flowers];
	copy(flowers.begin(), flowers.end(), flowers_h);
	
	vector<Centroid> centroids = kmeans.getCentroids();
	int num_centroids = centroids.size();
	Centroid * centroids_h = new Centroid[num_centroids];
	copy(centroids.begin(), centroids.end(), centroids_h);
	
	
	
    // 2. Set up platform, device, context, kernels, program
    //      - Must stream kernels and make sure they all wait for preceeding to complete
	// Choose first platform you find
    getPlatform(&platform);
    
    // Choose first device on first platform
    getDevice(platform, &device);
	getDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, 0);
    
    // Create context on first platform/device discovered
    context = createContext(platform, &device, (cl_uint) 1);
    
    program = createProgram("kmeans.cl", context, &device, (cl_uint) 1);
	
	// Create Kernel Objects for add, sub, mult, div
    createKernels(kernels, program);
    
	// Kernel specific vars
	int workItems = getNearestPowOf2((flowers.size()/MAX_WORK_GROUP_SIZE));
	
	// Now to create memory objects for:
	// 1. euclid_dist: calculate euclidean distance between each Centroid and flower
	// ** NOTE: DONT NEED TO SET ANYTHING FOR THIS FUNCTION!!! automatically called by kmeans_assingment **
	//		- Flower array
	//		- Centroid array
	// 2. kmeans_assignment: assigns each Flower to the closest centroid
	//		- Flower array
	//		- Centroid array
	//		- number of flowers in flower array
	//		- number of centroids
	int * array_sizes = new int[3];
	array_sizes[0] = num_flowers;
	array_sizes[1] = num_centroids;
	array_sizes[2] = num_centroids * MAX_WORK_GROUP_SIZE;
	int * intArgs	= new int[3]; // integer arguments for kernels
	intArgs[0] = num_flowers;
	intArgs[1] = num_centroids;
	intArgs[2] = MAX_WORK_GROUP_SIZE;

	bool success = createMemReadWrite(context, memReadWrite, 3, flowers_h, centroids_h, array_sizes);
	if(!success) {
		clReleaseKernel(kernels[0]);
		clReleaseKernel(kernels[1]);
    clReleaseKernel(kernels[2]);
    return 1;
	}
	
	queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &errNum);
	checkErr(errNum, "clCreateCommandQueue");
	
	// set kernel arguments for kmeans assignment, sum, and kmeans_update
	setArgsBlocking(kernels, memReadWrite,intArgs, workItems);
	
	// run kernels
	runKernelBlocking(queue, kernels, memReadWrite, num_flowers, workItems, numIterations, num_centroids); 
	
	// Cleanup
	// Encapsulate this in "Cleanup" method
	// ================================================================
    if (context != 0) {
        clReleaseContext(context);
	}
	
   if (program != 0) {
        clReleaseProgram(program);
	}
		
	
    clReleaseKernel(kernels[0]);
    clReleaseKernel(kernels[1]);
    clReleaseKernel(kernels[2]);
    delete [] memReadWrite;
    delete [] kernels;
    delete [] array_sizes;
    delete [] flowers_h;
    delete [] centroids_h;
	// ================================================================
	
    return 0;
}
