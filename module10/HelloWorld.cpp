//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//

// HelloWorld.cpp
//
//    This is a simple example that demonstrates basic OpenCL setup and
//    use.

#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <cmath>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

///
//  Constants
//
const int ARRAY_SIZE = 1000;

///
//  Create an OpenCL context on the first available platform using
//  either a GPU or CPU depending on what is available.
//
cl_context CreateContext()
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;
    cl_context context = NULL;

    // First, select an OpenCL platform to run on.  For this example, we
    // simply choose the first available platform.  Normally, you would
    // query for all available platforms and select the most appropriate one.
    errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0)
    {
        std::cerr << "Failed to find any OpenCL platforms." << std::endl;
        return NULL;
    }

    // Next, create an OpenCL context on the platform.  Attempt to
    // create a GPU-based context, and if that fails, try to create
    // a CPU-based context.
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)firstPlatformId,
        0
    };
    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
                                      NULL, NULL, &errNum);
    if (errNum != CL_SUCCESS)
    {
        std::cout << "Could not create GPU context, trying CPU..." << std::endl;
        context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
                                          NULL, NULL, &errNum);
        if (errNum != CL_SUCCESS)
        {
            std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
            return NULL;
        }
    }

    return context;
}

///
//  Create a command queue on the first device available on the
//  context
//
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device)
{
    cl_int errNum;
    cl_device_id *devices;
    cl_command_queue commandQueue = NULL;
    size_t deviceBufferSize = -1;

    // First get the size of the devices buffer
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
        return NULL;
    }

    if (deviceBufferSize <= 0)
    {
        std::cerr << "No devices available.";
        return NULL;
    }

    // Allocate memory for the devices buffer
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
    if (errNum != CL_SUCCESS)
    {
        delete [] devices;
        std::cerr << "Failed to get device IDs";
        return NULL;
    }

    // In this example, we just choose the first available device.  In a
    // real program, you would likely use all available devices or choose
    // the highest performance device based on OpenCL device queries
    commandQueue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, NULL);
    if (commandQueue == NULL)
    {
        delete [] devices;
        std::cerr << "Failed to create commandQueue for device 0";
        return NULL;
    }

    *device = devices[0];
    delete [] devices;
    return commandQueue;
}

///
//  Create an OpenCL program from the kernel source file
//
cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName)
{
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

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);
        return NULL;
    }

    return program;
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

int addSerial(float * a, float * b, float * result, int size) {
    for(int i = 0; i < size; i++) {
        if((a[i] + b[i]) != result[i]) {
            printf("Addition yielded an incorrect result at index %d\n", i);
            return 1;
        }
    }
    return 0;

}
int subSerial(float * a, float * b, float * result, int size) {
    for(int i = 0; i < size; i++) {
        if((a[i] - b[i]) != result[i]) {
            printf("Addition yielded an incorrect result at index %d\n", i);
            return 1;
        }
    }
    return 0;
    
}
int multSerial(float * a, float * b, float * result, int size) {
    for(int i = 0; i < size; i++) {
        if((a[i] * b[i]) != result[i]) {
            printf("Multiplication yielded an incorrect result at index %d\n", i);
            return 1;
        }
    }
    return 0;
    
}
int divSerial(float * a, float * b, float * result, int size) {
    for(int i = 0; i < size; i++) {
        if(abs((a[i] / b[i]) - result[i]) > 0.001) {
            printf("Division yielded an incorrect result at index %d\n", i);
            return 1;
        }
    }
    return 0;
    
}
int powerSerial(float * a, float power, float * result, int size) {
    for(int i = 0; i < size; i++) {
        if(abs(powf(a[i],power) - result[i]) > 0.001) {
            printf("Power yielded an incorrect result at index %d\n", i);
            printf("expected %f, received %f\n", powf(a[i], power), result[i]);
            return 1;
        }
    }
    return 0;
    
}

int testResult(float * a, float * b, float * result, int size, int kernelOp) {
    int errCode = 0;
    switch(kernelOp) {
        case 0: errCode = addSerial(a, b, result, size);
            break;
        case 1: errCode = subSerial(a, b, result, size);
            break;
        case 2: errCode = multSerial(a, b, result, size);
            break;
        case 3: errCode = divSerial(a, b, result, size);
            break;
        case 4: errCode = powerSerial(a, 3.0f, result, size);
            break;
    }
    return errCode;
}

int runKernel(cl_context context, cl_command_queue commandQueue, cl_program program,
              cl_kernel kernel, cl_mem memObjects[3], int array_size, int kernelOp) {
    std::string kernelName;
    // Create OpenCL kernel
    switch(kernelOp) {
        case 0:
            kernelName = "add";
            kernel = clCreateKernel(program, "add", NULL);
            break;
        case 1:
            kernelName = "sub";
            kernel = clCreateKernel(program, "sub", NULL);
            break;
        case 2:
            kernelName = "mult";
            kernel = clCreateKernel(program, "mult", NULL);
            break;
        case 3:
            kernelName = "div";
            kernel = clCreateKernel(program, "div", NULL);
            break;
        case 4:
            kernelName = "power";
            kernel = clCreateKernel(program, "power", NULL);
            break;
        default:
            kernelName = "add";
            kernel = clCreateKernel(program, "add", NULL);
            break;
        
    }
    if (kernel == NULL)
    {
        std::cerr << "Failed to create kernel" << std::endl;
        if (kernel != 0)
            clReleaseKernel(kernel);
        return 1;
    }
    
    // Create memory objects that will be used as arguments to
    // kernel.  First create host memory arrays that will be
    // used to store the arguments to the kernel
    float result[array_size];
    float a[array_size];
    float b[array_size];
    if(kernelOp != 4) {
        for (int i = 0; i < array_size; i++)
        {
            a[i] = (float)(i+1);
            b[i] = (float)((i+1) * 2);
        }
    } else {
        for (int i = 0; i < array_size; i++)
        {
            a[i] = (float)(i+1);
        }
    }
    
    if (!CreateMemObjects(context, memObjects, a, b, array_size))
    {
        clReleaseKernel(kernel);
        return 1;
    }
    
    // Set the kernel arguments (result, a, b)
    int errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[0]);
    if(kernelOp != 4) {
        errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[1]);
    } else {
        static const float power = 3.0f;
        errNum |= clSetKernelArg (kernel, 1, sizeof (float), &power);
    }
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[2]);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error setting kernel arguments." << std::endl;
        clReleaseKernel(kernel);
        return 1;
    }
    
    size_t globalWorkSize[1];
    globalWorkSize[0]= array_size;
    size_t localWorkSize[1] = { 1 };
    
    cl_event prof_event;
    // Queue the kernel up for execution across the array
    errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                    globalWorkSize, localWorkSize,
                                    0, NULL, &prof_event);
    
    // set up program, kernel, memory objects (not shown)
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error queuing kernel for execution." << std::endl;
        clReleaseKernel(kernel);
        return 1;
    }
    
    errNum = clWaitForEvents(1, &prof_event );
    cl_ulong ev_start_time=(cl_ulong)0;
    cl_ulong ev_end_time=(cl_ulong)0;
    size_t return_bytes;
    errNum = clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong),
                                  &ev_start_time, &return_bytes);
    errNum = clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong),
                                  &ev_end_time, &return_bytes);
    double run_time =(double)(ev_end_time - ev_start_time);
    // Read the output buffer back to the Host
    // Initialize timing mechanism with cl_event
    errNum = clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE,
                                 0, array_size * sizeof(float), result,
                                 0, NULL, &prof_event);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error reading result buffer." << std::endl;
        clReleaseKernel(kernel);
        return 1;
    }
    
    int errCode = testResult(a, b, result, array_size, kernelOp);
    std::cout << std::endl;
    if(errCode == 1) {
        return errCode;
    }
    std::cout << "Executed " << kernelName << " program succesfully in " << run_time*1e-6 << " ms" << std::endl;
    
    return errCode;

}
///
//	main() for HelloWorld example
//
int main(int argc, char** argv)
{
    cl_context context = 0;
    cl_command_queue commandQueue = 0;
    cl_program program = 0;
    cl_device_id device = 0;
    cl_kernel kernel = 0;
    cl_mem memObjects[3] = { 0, 0, 0 };
    
    int array_size = 1000;
    if(argc == 2) {
        array_size = atoi(argv[1]);
    }

    // Create an OpenCL context on first available platform
    context = CreateContext();
    if (context == NULL)
    {
        std::cerr << "Failed to create OpenCL context." << std::endl;
        return 1;
    }

    // Create a command-queue on the first device available
    // on the created context
    commandQueue = CreateCommandQueue(context, &device);
    if (commandQueue == NULL)
    {
        if (kernel != 0)
            clReleaseKernel(kernel);
        Cleanup(context, commandQueue, program, memObjects);
        return 1;
    }

    // Create OpenCL program from HelloWorld.cl kernel source
    program = CreateProgram(context, device, "/Users/patrickmclaughlin/Desktop/GradSchool/GPU_Programming/course_repo/EN605.417.FA/module10/HelloWorld.cl");
    if (program == NULL)
    {
        if (kernel != 0)
            clReleaseKernel(kernel);
        Cleanup(context, commandQueue, program, memObjects);
        return 1;
    }

    int errCode  = runKernel(context, commandQueue, program, kernel, memObjects,array_size, 0);
    errCode = runKernel(context, commandQueue, program, kernel, memObjects, array_size, 1);
    errCode = runKernel(context, commandQueue, program, kernel, memObjects, array_size, 2);
    errCode = runKernel(context, commandQueue, program, kernel, memObjects, array_size, 3);
    errCode = runKernel(context, commandQueue, program, kernel, memObjects, array_size, 4);
    return errCode;
    Cleanup(context, commandQueue, program, memObjects);

}
