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

int main(argc, char** argv) {
    cl_platform_id platform = 0;
    cl_device_id device = 0;
    cl_context context = NULL;
    cl_program program = NULL;
    
    // 1. Parse iris.csv into vector of "Flower" structs
    
    // 2. Set up platform, device, context, kernels, program
    //      - Must stream kernels and make sure they all wait for preceeding to complete
    return 0;
}