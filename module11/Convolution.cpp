//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//


// Convolution.cpp
//
//    This is a simple example that demonstrates OpenCL platform, device, and context
//    use.

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <ctime>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#if !defined(CL_CALLBACK)
#define CL_CALLBACK
#endif

// Constants
const unsigned int defaultInputSignalWidth  = 49;
const unsigned int defaultInputSignalHeight = 49;

/*cl_uint inputSignal[defaultInputSignalWidth][defaultInputSignalHeight] =
{
	{3, 1, 1, 4, 8, 2, 1, 3},
	{4, 2, 1, 1, 2, 1, 2, 3},
	{4, 4, 4, 4, 3, 2, 2, 2},
	{9, 8, 3, 8, 9, 0, 0, 0},
	{9, 3, 3, 9, 0, 0, 0, 0},
	{0, 9, 0, 8, 0, 0, 0, 0},
	{3, 0, 8, 8, 9, 4, 4, 4},
	{5, 9, 8, 1, 8, 1, 1, 1}
};

cl_uint inputSignal[defaultInputSignalWidth][defaultInputSignalHeight] =
{
	{1, 1, 1, 1, 1, 1, 1, 1},
	{1, 1, 1, 1, 1, 1, 1, 1},
	{1, 1, 1, 1, 1, 1, 1, 1},
	{1, 1, 1, 1, 1, 1, 1, 1},
	{1, 1, 1, 1, 1, 1, 1, 1},
	{1, 1, 1, 1, 1, 1, 1, 1},
	{1, 1, 1, 1, 1, 1, 1, 1},
	{1, 1, 1, 1, 1, 1, 1, 1},
};*/
const unsigned int defaultOutputSignalWidth  = 49;
const unsigned int defaultOutputSignalHeight = 49;

cl_uint outputSignal[defaultOutputSignalWidth][defaultOutputSignalHeight];

const unsigned int defaultMaskWidth  = 7;
const unsigned int defaultMaskHeight = 7;

const unsigned int mask3Width = 3;
const unsigned int mask3Height = 3;

cl_uint identityMask3[mask3Width][mask3Height] =
{
    {0, 0, 0}, {0, 1, 0}, {0, 0, 0},
};

cl_uint mask3[mask3Width][mask3Height] =
{
    {1, 1, 1}, {1, 0, 1}, {1, 1, 1},
};
/*cl_uint mask[defaultMaskWidth][defaultMaskHeight] =
{
	{1, 1, 1}, {1, 0, 1}, {1, 1, 1},
};*/
cl_uint mask7[defaultMaskWidth][defaultMaskHeight] =
{
	{0, 0, 1, 2, 1, 0, 0},
    {0, 1, 2, 3, 2, 1, 0},
    {1, 2, 3, 4, 3, 2, 1},
    {2, 3, 4, 4, 4, 3, 2},
    {1, 2, 3, 4, 3, 2, 1},
    {0, 1, 2, 3, 2, 1, 0},
    {0, 0, 1, 2, 1, 0, 0},
};
cl_uint identityMask7[defaultMaskWidth][defaultMaskHeight] =
{
    {0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 1, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0},
};
// Print signal to console
void printSignal(cl_uint * signal, int size, const char * signalName) {
    std::cout << signalName << ": " << std::endl;
    for (int x = 0; x < size; x++)
	{
		for (int y = 0; y < size; y++)
		{
            std::cout << signal[size * x + y] << " ";
		}
		std::cout << std::endl;
	}
}

// TODO rename arg
void selectMask(cl_uint * mask, int maskSelector) {
    int maskSize;
    switch(maskSelector) {
        // 3 x 3 original mask
        case 0:
            maskSize = 3;
            for(int i = 0; i < 3; i ++) {
                for (int j = 0; j < 3; j++) {
                    mask[3*i + j] = mask3[i][j];
                }
            }
            break;
        
        // 3 x 3 identity mask
        case 1:
            maskSize = 3;
            for(int i = 0; i < 3; i ++) {
                for (int j = 0; j < 3; j++) {
                    mask[3*i + j] = identityMask3[i][j];
                }
            }
            break;
        
        // 7 x 7 assignment mask
        case 2:
            maskSize = 7;
            for(int i = 0; i < 7; i ++) {
                for (int j = 0; j < 7; j++) {
                    mask[7*i + j] = mask7[i][j];
                }
            }
            break;
        
        // 7 x 7 identity mask
        case 3:
            maskSize = 7;
            for(int i = 0; i < 7; i ++) {
                for (int j = 0; j < 7; j++) {
                    mask[7*i + j] = identityMask7[i][j];
                }
            }
            break;
    }
    printSignal(mask, maskSize, "Mask");
}
///
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

void generateInputSignalDynamic(cl_uint * input, unsigned int width, unsigned int height) {
    for (int i = 0; i < width * height; i++) {
        input[i] = (cl_uint) (rand() % 10);
    }
    printSignal(input, width, "Input Signal");
}

cl_uint getNumPlatforms() {
    cl_uint errNum;
    cl_uint numPlatforms;
    
    errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkErr((errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS),
             "clGetPlatformIDs");
    return numPlatforms;
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
// Generate OpenCL memory objects (i.e. buffers)
// TODO: This assumes equal height and width for mask and input
// might be better to make this dynamic
void createMemObjects(cl_context context, cl_mem memObjects[3],
                      cl_uint * input, unsigned int inputSize,
                      cl_uint * mask, unsigned int maskSize)
{
    cl_int errNum;
    
    memObjects[0] = clCreateBuffer(
                                   context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(cl_uint) * inputSize * inputSize,
                                   (input),
                                   &errNum);
	checkErr(errNum, "clCreateBuffer(inputSignal)");
    
	memObjects[1] = clCreateBuffer(
                                   context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(cl_uint) * maskSize * maskSize,
                                   (mask),
                                   &errNum);
	checkErr(errNum, "clCreateBuffer(mask)");
    
    unsigned int outputSize = inputSize - (inputSize % maskSize);
	memObjects[2] = clCreateBuffer(
                                   context,
                                   CL_MEM_WRITE_ONLY,
                                   sizeof(cl_uint) * outputSize * outputSize,
                                   NULL,
                                   &errNum);
	checkErr(errNum, "clCreateBuffer(outputSignal)");
}

// Clean up memory objects (both opencl and c++ memory objects)
void Cleanup(cl_context context, cl_command_queue commandQueue,
             cl_program program, cl_kernel kernel, cl_mem memObjects[3],
             cl_uint * input, cl_uint * mask, cl_uint * output)
{
    for (int i = 0; i < 3; i++)
    {
        if (memObjects[i] != 0)
        clReleaseMemObject(memObjects[i]);
    }
    if (commandQueue != 0)
    clReleaseCommandQueue(commandQueue);
    
    if (kernel != 0)
    clReleaseKernel(kernel);
    
    if (program != 0)
    clReleaseProgram(program);
    
    if (context != 0)
    clReleaseContext(context);
    
    delete [] input;
    delete [] mask;
    delete [] output;
    
}

///
//	main() for Convoloution example
//
int main(int argc, char** argv)
{
    unsigned int input_width = defaultInputSignalWidth;
    unsigned int input_height = defaultInputSignalHeight;
    unsigned int mask_width = defaultMaskWidth;
    unsigned int mask_height = defaultMaskHeight;
    cl_uint normalize = 0;
    int maskSelector = 2;
    if(argc == 2) {
        maskSelector = atoi(argv[1]);
    }
    if(maskSelector == 0 || maskSelector == 1) {
        input_width = 8;
        input_height = 8;
        mask_width = 3;
        mask_height = 3;
    } else {
        input_width = 49;
        input_height = 49;
        mask_width = 7;
        mask_height = 7;
        if(maskSelector == 2) {
            normalize = 1;
        }
    }
    unsigned int output_size = input_width - (input_width % mask_width);
    cl_int errNum;
    
    cl_platform_id platform = 0;
    cl_device_id device = 0;
    cl_context context = NULL;
    cl_program program = NULL;
    cl_mem memObjects[3] = { 0, 0, 0 };
    
	cl_command_queue queue;
	cl_kernel kernel;
    
    cl_uint* input = new cl_uint[input_width*input_height];
    cl_uint* mask = new cl_uint[mask_width*mask_height];
    cl_uint* output = new cl_uint[output_size*output_size];
    
    srand(time(0));
    generateInputSignalDynamic(input, input_width, input_height);
    selectMask(mask, maskSelector);
    
    // First, select an OpenCL platform to run on.  
    getPlatform(&platform);
    getDevice(platform, &device);
    context = createContext(platform, &device, (cl_uint) 1);
    program = createProgram("Convolution.cl", context, &device, (cl_uint) 1);
    
	// Create kernel object
	kernel = clCreateKernel(
		program,
		"convolve",
		&errNum);
	checkErr(errNum, "clCreateKernel");

	// Now allocate buffers
    createMemObjects(context, memObjects, input, input_width, mask, mask_width);
    
	// Pick the first device and create command queue.
	queue = clCreateCommandQueue(
		context,
		device,
		CL_QUEUE_PROFILING_ENABLE,
		&errNum);
	checkErr(errNum, "clCreateCommandQueue");
    
    // Set Kernel Arguments
    errNum  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[0]);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[1]);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[2]);
	errNum |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &input_width);
	errNum |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &mask_width);
	errNum |= clSetKernelArg(kernel, 5, sizeof(cl_uint), &normalize);
	checkErr(errNum, "clSetKernelArg");
    
    const size_t globalWorkSize[1] = { output_size * output_size};
    const size_t localWorkSize[1]  = { 1 };

    // Create event for timing
    cl_event prof_event;
    
    // Queue the kernel up for execution across the array
    errNum = clEnqueueNDRangeKernel(
        queue,
		kernel, 
		1, 
		NULL,
        globalWorkSize, 
		localWorkSize,
        0, 
		NULL, 
		&prof_event);
	checkErr(errNum, "clEnqueueNDRangeKernel");
    
    // Calculate execution time for kernel:
    clWaitForEvents(1, &prof_event );
    cl_ulong ev_start_time=(cl_ulong)0;
    cl_ulong ev_end_time=(cl_ulong)0;
    size_t return_bytes;
    errNum = clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong),
                                     &ev_start_time, &return_bytes);
    errNum = clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong),
                                     &ev_end_time, &return_bytes);
    double run_time =(double)(ev_end_time - ev_start_time);
    
	errNum = clEnqueueReadBuffer(
		queue, 
		memObjects[2],
		CL_TRUE,
        0, 
		sizeof(cl_uint) * output_size * output_size,
		output,
        0, 
		NULL, 
		NULL);
	checkErr(errNum, "clEnqueueReadBuffer");
    
    // Output the result buffer
    printSignal(output, output_size, "Output Signal");
    
    // Clean up memory objects
    Cleanup(context, queue, program, kernel, memObjects, input, mask, output);
    printf("Convolution over %d x %d signal with a %d x %d mask took %f ms\n", input_width, input_height, mask_width, mask_height, run_time*1e-6);
	return 0;
}
