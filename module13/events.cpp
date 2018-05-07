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

// Constant
const int ARRAY_SIZE = 1000;

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

void createKernels(cl_kernel * kernels, cl_program program) {
    cl_int errNum;
    kernels[0] = clCreateKernel(program, "add", &errNum);
    kernels[1] = clCreateKernel(program, "sub", &errNum);
    kernels[2] = clCreateKernel(program, "mult", &errNum);
    kernels[3] = clCreateKernel(program, "div", &errNum);
    checkErr(errNum, "clCreateKernel");
}

bool createMemReadOnly(cl_context context, cl_mem * memObjects,
                       float ** hostObjects, int numObjects, int array_size) {
    for (int i = 0; i < numObjects; i++) {
        memObjects[i] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(float) * array_size, hostObjects[i], NULL);
        if(memObjects[i] == NULL) {
            std::cerr << "Error creating memory objects." << std::endl;
            return false;
        }
    }
    return true;
    
}
bool createMemReadWrite(cl_context context, cl_mem * memObjects,
                       int numObjects, int array_size) {
    for (int i = 0; i < numObjects; i++) {
        memObjects[i] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                       sizeof(float) * array_size, NULL, NULL);
        if(memObjects[i] == NULL) {
            std::cerr << "Error creating memory objects." << std::endl;
            return false;
        }
    }
    return true;
    
}

///
//  Create memory objects used as the arguments to the kernel
//  The kernel takes three arguments: result (output), a (input),
//  and b (input)
//
bool CreateMemObjects(cl_context context, cl_mem memObjects[3],
                      float *a, float *b, int array_size)
{
    memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * array_size, a, NULL);
    memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * array_size, b, NULL);
    memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * array_size, NULL, NULL);
    
    if (memObjects[0] == NULL || memObjects[1] == NULL || memObjects[2] == NULL)
    {
        std::cerr << "Error creating memory objects." << std::endl;
        return false;
    }
    
    return true;
}

// Sets arguments but reuses kernel argument results computed from previous computations
// in the next computation
// e.g. result from add used as argument to subtract
void setArgsBlocking(cl_kernel * kernels, cl_mem * memReadOnly, cl_mem * memReadWrite) {
    // "add" kernel args
    checkErr(clSetKernelArg(kernels[0], 0, sizeof(cl_mem), &memReadOnly[0]), "set kernel arg");
    checkErr(clSetKernelArg(kernels[0], 1, sizeof(cl_mem), &memReadOnly[1]), "set kernel arg");
    checkErr(clSetKernelArg(kernels[0], 2, sizeof(cl_mem), &memReadWrite[0]), "set kernel arg");
    
    // "sub" kernel args
    checkErr(clSetKernelArg(kernels[1], 0, sizeof(cl_mem), &memReadWrite[0]), "set kernel arg");
    checkErr(clSetKernelArg(kernels[1], 1, sizeof(cl_mem), &memReadOnly[1]), "set kernel arg");
    checkErr(clSetKernelArg(kernels[1], 2, sizeof(cl_mem), &memReadWrite[1]), "set kernel arg");
    
    // "mult" kernel args
    checkErr(clSetKernelArg(kernels[2], 0, sizeof(cl_mem), &memReadWrite[1]), "set kernel arg");
    checkErr(clSetKernelArg(kernels[2], 1, sizeof(cl_mem), &memReadOnly[1]), "set kernel arg");
    checkErr(clSetKernelArg(kernels[2], 2, sizeof(cl_mem), &memReadWrite[2]), "set kernel arg");
    
    // "div" kernel args
    checkErr(clSetKernelArg(kernels[3], 0, sizeof(cl_mem), &memReadWrite[2]), "set kernel arg");
    checkErr(clSetKernelArg(kernels[3], 1, sizeof(cl_mem), &memReadOnly[1]), "set kernel arg");
    checkErr(clSetKernelArg(kernels[3], 2, sizeof(cl_mem), &memReadWrite[3]), "set kernel arg");
}

// Sets arguments and reuses "a" and "b" for each operation
// No independence between operations
void setArgsNonBlocking(cl_kernel * kernels, cl_mem * memReadOnly, cl_mem * memReadWrite) {
    // "add" kernel args
    checkErr(clSetKernelArg(kernels[0], 0, sizeof(cl_mem), &memReadOnly[0]), "set kernel arg");
    checkErr(clSetKernelArg(kernels[0], 1, sizeof(cl_mem), &memReadOnly[1]), "set kernel arg");
    checkErr(clSetKernelArg(kernels[0], 2, sizeof(cl_mem), &memReadWrite[0]), "set kernel arg");
    
    // "sub" kernel args
    checkErr(clSetKernelArg(kernels[1], 0, sizeof(cl_mem), &memReadOnly[0]), "set kernel arg");
    checkErr(clSetKernelArg(kernels[1], 1, sizeof(cl_mem), &memReadOnly[1]), "set kernel arg");
    checkErr(clSetKernelArg(kernels[1], 2, sizeof(cl_mem), &memReadWrite[1]), "set kernel arg");
    
    // "mult" kernel args
    checkErr(clSetKernelArg(kernels[2], 0, sizeof(cl_mem), &memReadOnly[0]), "set kernel arg");
    checkErr(clSetKernelArg(kernels[2], 1, sizeof(cl_mem), &memReadOnly[1]), "set kernel arg");
    checkErr(clSetKernelArg(kernels[2], 2, sizeof(cl_mem), &memReadWrite[2]), "set kernel arg");
    
    // "div" kernel args
    checkErr(clSetKernelArg(kernels[3], 0, sizeof(cl_mem), &memReadOnly[0]), "set kernel arg");
    checkErr(clSetKernelArg(kernels[3], 1, sizeof(cl_mem), &memReadOnly[1]), "set kernel arg");
    checkErr(clSetKernelArg(kernels[3], 2, sizeof(cl_mem), &memReadWrite[3]), "set kernel arg");
}

// setArgsNonBlocking this will be different from above because we will re-use a and b for all inputs
void setKernelArgs(cl_kernel kernel, cl_mem memObjects[3]) {
    checkErr(clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[0]), "set kernel arg 0");
    checkErr(clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[1]), "set kernel arg 1");
    checkErr(clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[2]), "set kernel arg 2");
}

///
//  Cleanup any created OpenCL resources
//
void Cleanup(cl_context context, cl_command_queue commandQueue,
             cl_program program, cl_mem memObjects[3])
{
    for (int i = 0; i < 3; i++)
    {
        if (memObjects[i] != 0)
            clReleaseMemObject(memObjects[i]);
    }
    if (commandQueue != 0)
        clReleaseCommandQueue(commandQueue);
    
    /*if (kernel != 0)
     clReleaseKernel(kernel);*/
    
    if (program != 0)
        clReleaseProgram(program);
    
    if (context != 0)
        clReleaseContext(context);
    
}

void populateInputs(float * a, float * b, int array_size) {
    for(int i = 0; i < array_size; i++) {
        a[i] = (float)(i + 1);
        b[i] = (float)((i + 1) * 2);
    }
    printf("populated arrays!\n");
}

// Options:
// 1) Blocking: all kernels wait for the previous kernel to conclude
// 2) Non Blocking: no kernels wait for previous kernel to conclude... all just execute when they can
// 3) Semi-blocking: 2 kernels execute, 2nd 2 kernels wait for first 2 to conclude
void runKernelBlocking(cl_command_queue queue, cl_kernel * kernels, int array_size) {
    size_t globalWorkSize[1];
    globalWorkSize[0] = array_size;
    size_t localWorkSize[1] = {1};
    
    // Events to enforce blocking
    cl_event events[4];
    cl_ulong ev_start_time;
    cl_ulong ev_end_time;
    size_t return_bytes;
    // Queue the kernel up for execution across the array

    /*
     cl_int clEnqueueNDRangeKernel (
     cl_command_queue command_queue,
     cl_kernel kernel,
     cl_uint work_dim,
     const size_t *global_work_offset,
     const size_t *global_work_size,
     const size_t *local_work_size,
     cl_uint num_events_in_wait_list,
     const cl_event *event_wait_list,
     cl_event *event)
     */
    
    // execute add
    checkErr(clEnqueueNDRangeKernel(queue, kernels[0], 1, NULL,
                                    globalWorkSize, localWorkSize,
                                    0, NULL, &events[0]), "enqueue kernel");
    printf("executed add\n");
    // execute sub with result from add
    checkErr(clEnqueueNDRangeKernel(queue, kernels[1], 1, NULL,
                                    globalWorkSize, localWorkSize,
                                    1, &events[0], &events[1]), "enqueue kernel");
    printf("executed sub\n");
    
    // execute mult with result from sub
    checkErr(clEnqueueNDRangeKernel(queue, kernels[2], 1, NULL,
                                    globalWorkSize, localWorkSize,
                                    1, &events[1], &events[2]), "enqueue kernel");
    printf("executed mult\n");
    
    // execute div with result from mult
    checkErr(clEnqueueNDRangeKernel(queue, kernels[3], 1, NULL,
                                    globalWorkSize, localWorkSize,
                                    1, &events[2], &events[3]), "enqueue kernel");
    printf("executed div\n");
    clWaitForEvents(1, &events[0] );
    clWaitForEvents(1, &events[3] );
    checkErr(clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &ev_start_time, &return_bytes), "evt prfiling");
    checkErr(clGetEventProfilingInfo(events[3], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &ev_end_time, &return_bytes), "evt prfiling");
    double run_time =(double)(ev_end_time - ev_start_time);
    
    printf("Blocking kernel runs on arrays of length %d took %f ms\n",array_size, run_time*1e-6);

    
    // read sub result back
    
    // read mult result back
    
    // read div result back
    
}

void runKernelNonBlocking(cl_command_queue queue, cl_kernel * kernels, int array_size) {
    size_t globalWorkSize[1];
    globalWorkSize[0] = array_size;
    size_t localWorkSize[1] = {1};
    
    // Events for profiling
    cl_event events[2];
    cl_ulong ev_start_time;
    cl_ulong ev_end_time;
    size_t return_bytes;
    
    // execute add
    checkErr(clEnqueueNDRangeKernel(queue, kernels[0], 1, NULL,
                                    globalWorkSize, localWorkSize,
                                    0, NULL, &events[0]), "enqueue kernel");
    printf("executed add\n");
    // execute sub with result from add
    checkErr(clEnqueueNDRangeKernel(queue, kernels[1], 1, NULL,
                                    globalWorkSize, localWorkSize,
                                    0, NULL, NULL), "enqueue kernel");
    printf("executed sub\n");
    
    // execute mult with result from sub
    checkErr(clEnqueueNDRangeKernel(queue, kernels[2], 1, NULL,
                                    globalWorkSize, localWorkSize,
                                    0, NULL, NULL), "enqueue kernel");
    printf("executed mult\n");
    
    // execute div with result from mult
    checkErr(clEnqueueNDRangeKernel(queue, kernels[3], 1, NULL,
                                    globalWorkSize, localWorkSize,
                                    0, NULL, &events[1]), "enqueue kernel");
    printf("executed div\n");
    clWaitForEvents(1, &events[0] );
    clWaitForEvents(1, &events[1] );
    checkErr(clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &ev_start_time, &return_bytes), "evt prfiling");
    checkErr(clGetEventProfilingInfo(events[1], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &ev_end_time, &return_bytes), "evt prfiling");
    double run_time =(double)(ev_end_time - ev_start_time);
    
    printf("Non Blocking kernel runs on arrays of length %d took %f ms\n",array_size, run_time*1e-6);

}

void getResults(cl_command_queue queue, cl_mem * memReadWrite, float ** readWrites, int array_size) {
    // read add result back
    checkErr(clEnqueueReadBuffer(queue, memReadWrite[0], CL_TRUE,
                                 0, array_size * sizeof(float), readWrites[0],
                                 0, NULL, NULL), "reading buffer");
    // read sub result back
    checkErr(clEnqueueReadBuffer(queue, memReadWrite[1], CL_TRUE,
                                 0, array_size * sizeof(float), readWrites[1],
                                 0, NULL, NULL), "reading buffer");
    // read mult result back
    checkErr(clEnqueueReadBuffer(queue, memReadWrite[2], CL_TRUE,
                                 0, array_size * sizeof(float), readWrites[2],
                                 0, NULL, NULL), "reading buffer");
    // read div result back
    checkErr(clEnqueueReadBuffer(queue, memReadWrite[3], CL_TRUE,
                                 0, array_size * sizeof(float), readWrites[3],
                                 0, NULL, NULL), "reading buffer");
}
// Specify 0, 1, 2, 3 for "add", "subtract", "multiply", "divide"
int main(int argc, char** argv) {
    // Default variables that can be specified by command line
    int blocking = 1;
    int array_size = 1000;
    
    if(argc == 3) {
        blocking = atoi(argv[1]);
        array_size = atoi(argv[2]);
    }
    
    // Host boilerplate vars
    float * a = new float[array_size];
    float * b = new float[array_size];
    float * result = new float[array_size];
    float * result2 = new float[array_size];
    float * result3 = new float[array_size];
    float * result4 = new float[array_size];
    float ** readOnlys = new float*[2];
    float ** readWrites = new float*[4];
    
    // allocate memory for read only host memory
    for (int i = 0; i < 2; i++) {
        readOnlys[i] = new float[array_size];
    }
    
    // allocate memory for read/write host memory
    for (int i = 0; i < 4; i++) {
        readWrites[i] = new float[array_size];
    }
    
    
    cl_int errNum;
    
    // OpenCL boilerplate vars
    cl_platform_id platform = 0;
    cl_device_id device = 0;
    cl_context context = NULL;
    cl_program program = NULL;
    cl_mem memObjects[3] = {0, 0, 0};
    cl_mem * memReadOnly = new cl_mem[2];
    cl_mem * memReadWrite = new cl_mem[4];
    
    cl_uint num_events_in_waitlist = 0;
    
    
    cl_command_queue queue;
	cl_kernel * kernels = new cl_kernel[3];
    
    // Choose first platform you find
    getPlatform(&platform);
    
    // Choose first device on first platform
    getDevice(platform, &device);
    
    // Create context on first platform/device discovered
    context = createContext(platform, &device, (cl_uint) 1);
    
    // create program for simple kernel operations "add", "sub", etc...
    program = createProgram("simple_ops.cl", context, &device, (cl_uint) 1);
    
    // Create Kernel Objects for add, sub, mult, div
    createKernels(kernels, program);
    printf("Set up platform, device, context, program and kernels\n");
    // populate host memory objects
    populateInputs(a, b, array_size);
    
    for (int i = 0; i < array_size; i++) {
        readOnlys[0][i] = a[i];
        readOnlys[1][i] = b[i];
    }
    /*readOnlys[0] = a;
    readOnlys[1] = b;*/
    // Create mem objects
    /*cl_context context, cl_mem * memObjects,
     float ** hostObjects, int numObjects, int array_size*/
    
    bool success = createMemReadOnly(context, memReadOnly, readOnlys, 2, array_size);
    success = createMemReadWrite(context, memReadWrite, 4, array_size);
    if(!success) {
        clReleaseKernel(kernels[0]);
        clReleaseKernel(kernels[1]);
        clReleaseKernel(kernels[2]);
        clReleaseKernel(kernels[3]);
        return 1;
    }
    /*if(!CreateMemObjects(context, memObjects, a, b, array_size)) {
        clReleaseKernel(kernels[0]);
        clReleaseKernel(kernels[1]);
        clReleaseKernel(kernels[2]);
        clReleaseKernel(kernels[3]);
        return 1;
    }*/
    
    // Pick the first device and create command queue
    queue = clCreateCommandQueue(
                                 context,
                                 device,
                                 CL_QUEUE_PROFILING_ENABLE,
                                 &errNum);
	checkErr(errNum, "clCreateCommandQueue");
    
    // Set kernel arguments for add, sub, mult, div
    if(blocking == 1) {
        setArgsBlocking(kernels, memReadOnly, memReadWrite);
        runKernelBlocking(queue, kernels, array_size);
    } else if(blocking == 0) {
        setArgsNonBlocking(kernels, memReadOnly, memReadWrite);
        runKernelNonBlocking(queue, kernels, array_size);
    }
    
    // Set kernel arguments
    //setKernelArgs(kernels[0], memObjects2)
    // PUT RUN KERNEL FUNCTION HERE
    getResults(queue, memReadWrite, readWrites, array_size);
    /*size_t globalWorkSize[1];
    globalWorkSize[0] = array_size;
    size_t localWorkSize[1] = {1};
    
    printf("BLAHHH\n");
    cl_event prof_event;
    // Queue the kernel up for execution across the array
    checkErr(clEnqueueNDRangeKernel(queue, kernels[0], 1, NULL,
                                    globalWorkSize, localWorkSize,
                                    0, NULL, &prof_event), "enqueue kernel");
    printf("enqueued kernel\n");
    
    checkErr(clWaitForEvents(1, &prof_event), "wait for events");
    cl_ulong ev_start_time = (cl_ulong)0;
    cl_ulong ev_end_time = (cl_ulong)0;
    size_t return_bytes;
    checkErr(clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &ev_start_time, &return_bytes), "evt prfiling");
    checkErr(clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &ev_end_time, &return_bytes), "evt prfiling");
    double run_time =(double)(ev_end_time - ev_start_time);
    // Read the output buffer back to the Host
    // Initialize timing mechanism with cl_event
    checkErr(clEnqueueReadBuffer(queue, memObjects[2], CL_TRUE,
                                 0, array_size * sizeof(float), result,
                                 0, NULL, &prof_event), "reading buffer");*/
    /*for(int i = 0; i < 4; i++) {
        for(int j = 0; j < array_size; j++) {
            printf("%f ", readWrites[i][j]);
        }
        printf("\n");
    }*/
    Cleanup(context, queue, program, memObjects);
    clReleaseKernel(kernels[0]);
    clReleaseKernel(kernels[1]);
    clReleaseKernel(kernels[2]);
    clReleaseKernel(kernels[3]);
    delete [] a;
    delete [] b;
    delete [] result;
    delete [] result2;
    delete [] result3;
    delete [] result4;
    for(int i = 0; i < 2; i++) {
        delete [] readOnlys[i];
    }
    delete [] readOnlys;
    for(int i = 0; i < 4; i++) {
        delete [] readWrites[i];
    }
    delete [] readWrites;
    delete [] kernels;
    printf("all done!\n");
    
}