#include <stdio.h>
#include <iostream>
#include <cstdlib>


using namespace std;
int inline openCLExists() {
#ifdef _WIN32
    FILE *f = fopen("$WINDIR\\system32\\OpenCL.dll", "r");
    if(f != NULL) {
        fclose(f);
        return 0;
    }
    return -1;
#elif __APPLE__
    FILE *f = fopen("/System/Library/Frameworks/OpenCL.framework/OpenCL", "r");
    if(f != NULL) {
        fclose(f);
        return 1;
    }
    return -1;
#elif __linux__
    return -1;
#endif
    
}

void getGPUInfo(int sdkSelector, const char * compileString) {
    switch (sdkSelector) {
        // sdkSelector == 0 --> OpenCL
        case 0:
            system(compileString);
            system("./cl_info");
            break;
            
        // sdkSelector == 1 --> CUDA
        case 1: break;
        
        // sdkSelector == 2 --> Serial implementation
        case 2: break;
    }
    
}

int main(int argc, char** argv) {
    int openCLSelector = openCLExists();
    if(openCLSelector == 1) {
        cout << "Houston, we have opencl!" << endl;
        // Handle exception when this does not work
        getGPUInfo(0,"g++ -o cl_info OpenCLInfo.cpp -framework OpenCL");
    }
}