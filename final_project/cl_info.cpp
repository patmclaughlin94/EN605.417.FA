#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <map>
#include <iostream>
using namespace std;

map<string, cl_platform_info> platformInfoParams;
map<string, cl_device_info> deviceInfoUINTParams;
map<string, cl_device_info> deviceInfoBOOLParams;
map<string, cl_device_info> deviceInfoFPCONFIGParams;
map<string, cl_device_info> deviceInfoExecCapabilitiesParams;
map<string, cl_device_info> deviceInfoCharArrParams;
map<string, cl_device_info> deviceInfoULongParams;

void setPlatformParams() {
    platformInfoParams["CL_PLATFORM_PROFILE"] = CL_PLATFORM_PROFILE;
    platformInfoParams["CL_PLATFORM_VENDOR"] = CL_PLATFORM_VENDOR;
    platformInfoParams["CL_PLATFORM_VERSION"] = CL_PLATFORM_VERSION;
    platformInfoParams["CL_PLATFORM_EXTENSIONS"] = CL_PLATFORM_EXTENSIONS;
}

void setDeviceParams(int deviceType) {
    deviceInfoParams["CL_DEVICE_TYPE"] = CL_DEVICE_TYPE;
    deviceInfoParams["CL_DEVICE_VENDOR_ID"] = CL_DEVICE_TYPE;
    deviceInfoParams["CL_DEVICE_MAX_COMPUTE_UNITS"] = CL_DEVICE_MAX_COMPUTE_UNITS;
    deviceInfoParams["CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS"] = CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS;
    deviceInfoParams["CL_DEVICE_MAX_WORK_ITEM_SIZES"] = CL_DEVICE_MAX_WORK_ITEM_SIZES;
    deviceInfoParams["CL_DEVICE_MAX_WORK_GROUP_SIZE"] = CL_DEVICE_MAX_WORK_GROUP_SIZE;
    deviceInfoParams["CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR"] = CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR;
    deviceInfoParams["CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT"] = CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT;
    deviceInfoParams["CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT"] = CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT;
    deviceInfoParams["CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG"] = CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG;
    deviceInfoParams["CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT"] = CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT;
    deviceInfoParams["CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE"] = CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE;
    
#ifdef CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF
    deviceInfoParams["CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF"] = CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF;
    deviceInfoParams["CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR"] = CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR;
    deviceInfoParams["CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT"] = CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT;
    deviceInfoParams["CL_DEVICE_NATIVE_VECTOR_WIDTH_INT"] = CL_DEVICE_NATIVE_VECTOR_WIDTH_INT;
    deviceInfoParams["CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG"] = CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG;
    deviceInfoParams["CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT"] = CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT;
    deviceInfoParams["CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE"] = CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE;
    deviceInfoParams["CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF"] = CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF;
#endif
    deviceInfoParams["CL_DEVICE_MAX_CLOCK_FREQUENCY"] = CL_DEVICE_MAX_CLOCK_FREQUENCY;
    deviceInfoParams["CL_DEVICE_ADDRESS_BITS"] = CL_DEVICE_ADDRESS_BITS;
    deviceInfoParams["CL_DEVICE_MAX_MEM_ALLOC_SIZE"] = CL_DEVICE_MAX_MEM_ALLOC_SIZE;
    deviceInfoParams["CL_DEVICE_IMAGE_SUPPORT"] = CL_DEVICE_IMAGE_SUPPORT;
    deviceInfoParams["CL_DEVICE_MAX_READ_IMAGE_ARGS"] = CL_DEVICE_MAX_READ_IMAGE_ARGS;
    deviceInfoParams["CL_DEVICE_MAX_WRITE_IMAGE_ARGS"] = CL_DEVICE_MAX_WRITE_IMAGE_ARGS;
    deviceInfoParams["CL_DEVICE_IMAGE2D_MAX_WIDTH"] = CL_DEVICE_IMAGE2D_MAX_WIDTH;
    deviceInfoParams["CL_DEVICE_IMAGE2D_MAX_HEIGHT"] = CL_DEVICE_IMAGE2D_MAX_HEIGHT;
    deviceInfoParams["CL_DEVICE_IMAGE3D_MAX_WIDTH"] = CL_DEVICE_IMAGE3D_MAX_WIDTH;
    deviceInfoParams["CL_DEVICE_IMAGE3D_MAX_HEIGHT"] = CL_DEVICE_IMAGE3D_MAX_HEIGHT;
    deviceInfoParams["CL_DEVICE_IMAGE3D_MAX_DEPTH"] = CL_DEVICE_IMAGE3D_MAX_DEPTH;
    deviceInfoParams["CL_DEVICE_MAX_SAMPLERS"] = CL_DEVICE_MAX_SAMPLERS;
    deviceInfoParams["CL_DEVICE_MAX_PARAMETER_SIZE"] = CL_DEVICE_MAX_PARAMETER_SIZE;
    deviceInfoParams["CL_DEVICE_MEM_BASE_ADDR_ALIGN"] = CL_DEVICE_MEM_BASE_ADDR_ALIGN;
    deviceInfoParams["CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE"] = CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE;
    deviceInfoParams["CL_DEVICE_SINGLE_FP_CONFIG"] = CL_DEVICE_SINGLE_FP_CONFIG;
    deviceInfoParams["CL_DEVICE_GLOBAL_MEM_CACHE_TYPE"] = CL_DEVICE_GLOBAL_MEM_CACHE_TYPE;
    deviceInfoParams["CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE"] = CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE;
    deviceInfoParams["CL_DEVICE_GLOBAL_MEM_CACHE_SIZE"] = CL_DEVICE_GLOBAL_MEM_CACHE_SIZE;
    deviceInfoParams["CL_DEVICE_GLOBAL_MEM_SIZE"] = CL_DEVICE_GLOBAL_MEM_SIZE;
    deviceInfoParams["CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE"] = CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE;
    deviceInfoParams["CL_DEVICE_MAX_CONSTANT_ARGS"] = CL_DEVICE_MAX_CONSTANT_ARGS;
    deviceInfoParams["CL_DEVICE_LOCAL_MEM_TYPE"] = CL_DEVICE_LOCAL_MEM_TYPE;
    deviceInfoParams["CL_DEVICE_LOCAL_MEM_SIZE"] = CL_DEVICE_LOCAL_MEM_SIZE;
    deviceInfoParams["CL_DEVICE_ERROR_CORRECTION_SUPPORT"] = CL_DEVICE_ERROR_CORRECTION_SUPPORT;
#ifdef CL_DEVICE_HOST_UNIFIED_MEMORY
    deviceInfoParams["CL_DEVICE_HOST_UNIFIED_MEMORY"] = CL_DEVICE_HOST_UNIFIED_MEMORY;
#endif
    deviceInfoParams["CL_DEVICE_PROFILING_TIMER_RESOLUTION"] = CL_DEVICE_PROFILING_TIMER_RESOLUTION;
    deviceInfoParams["CL_DEVICE_ENDIAN_LITTLE"] = CL_DEVICE_ENDIAN_LITTLE;
    deviceInfoParams["CL_DEVICE_AVAILABLE"] = CL_DEVICE_AVAILABLE;
    deviceInfoParams["CL_DEVICE_COMPILER_AVAILABLE"] = CL_DEVICE_COMPILER_AVAILABLE;
    deviceInfoParams["CL_DEVICE_EXECUTION_CAPABILITIES"] = CL_DEVICE_EXECUTION_CAPABILITIES;
    deviceInfoParams["CL_DEVICE_QUEUE_PROPERTIES"] = CL_DEVICE_QUEUE_PROPERTIES;
    deviceInfoParams["CL_DEVICE_PLATFORM"] = CL_DEVICE_PLATFORM;
    deviceInfoParams["CL_DEVICE_NAME"] = CL_DEVICE_NAME;
    deviceInfoParams["CL_DEVICE_VENDOR"] = CL_DEVICE_VENDOR;
    deviceInfoParams["CL_DRIVER_VERSION"] = CL_DRIVER_VERSION;
    deviceInfoParams["CL_DEVICE_PROFILE"] = CL_DEVICE_PROFILE;
    deviceInfoParams["CL_DEVICE_VERSION"] = CL_DEVICE_VERSION;
#ifdef CL_DEVICE_OPENCL_C_VERSION
    deviceInfoParams["CL_DEVICE_OPENCL_C_VERSION"] = CL_DEVICE_OPENCL_C_VERSION;
#endif
    deviceInfoParams["CL_DEVICE_EXTENSIONS"] = CL_DEVICE_EXTENSIONS;

}

void checkErr(cl_int err, const char * errContext) {
    if(err != CL_SUCCESS) {
        cerr << "ERROR: " << errContext << " (" << err << ")" << endl;
        exit(EXIT_FAILURE);
    }
}

// Display information for a platform
void DisplayPlatformInfo(
    cl_platform_id id,
    cl_platform_info name,
    string str)
{
    size_t paramValueSize;
    
    checkErr(clGetPlatformInfo(id, name, 0, NULL, &paramValueSize), "Failed to find OpenCL platform");
    
    char * info = (char *) malloc(sizeof(char) * paramValueSize);
    checkErr(clGetPlatformInfo(id, name, paramValueSize, info, NULL), "Failed to get info parameter");
    
    cout << "\t" << str << ":\t" << info << endl;
    
}

// Display information for a device

// Select Platform we will work with

// Select device that we will work with

int main(int argc, char** argv) {
    cout << "HELLO WORLD!" << endl;
    return 0;
}