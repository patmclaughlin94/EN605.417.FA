#include <stdio.h>
#include <iostream>
#include <cstdlib>
using namespace std;
#include<string>
#include "kmeans.h"


using namespace std;
string OCLCompatibilityCMD;
string OCLCompileKMeans;
string OCLRunKMeans = "./kmeans_host_opencl";
string CUDACompatibilityCMD = "nvcc -o CUDAInfo CUDAInfo.cu && ./CUDAInfo";
string CUDACompileKMeans = "nvcc -std=c++11 -o kmeans_host_cuda kmeans.cpp kmeans_host_cuda.cu";
string CUDARunKMeans = "./kmeans_host_cuda";
string welcomeMessage = "Welcome to the Heterogeneous Computing Machine Learning Toolkit!\n"
												"Let us start by analyzing your hardware...\n";

void inline setCompileStrings() {
	#ifdef _WIN32
		OCLCompatibilityCMD = "g++ -o cl_info OpenCLInfo.cpp -lOpenCL && ./cl_info";
		OCLCompileKMeans = "g++ -std=c++11 kmeans.cpp kmeans_host_opencl.cpp -lOpenCL -o kmeans_host_opencl";
	#elif __APPLE__
		OCLCompatibilityCMD = "g++ -o cl_info OpenCLInfo.cpp -framework OpenCL && ./cl_info";
		OCLCompileKMeans = "g++ -std=c++11 kmeans.cpp kmeans_host_opencl.cpp -framework OpenCL -o kmeans_host_opencl";
	#elif __linux__
		OCLCompatibilityCMD = "g++ -w -o cl_info OpenCLInfo.cpp -lOpenCL && ./cl_info";
		OCLCompileKMeans = "g++ -w -std=c++11 kmeans.cpp kmeans_host_opencl.cpp -lOpenCL -o kmeans_host_opencl";
	#endif
}

bool checkSysCall(int retVal, string context) {
	if(retVal != 0) {
		printf("%s\n", context.c_str());
		return false;
	}
	return true;
}

int getGPUInfo() {
	bool hasOCL = true;
	bool hasCUDA = true;;
	// Try OpenCL
	printf("OCLCompatibilityCMD: %s\n", OCLCompatibilityCMD.c_str());
	if(!(checkSysCall(system(OCLCompatibilityCMD.c_str()), "Cannot compile and run cl_info"))) hasOCL = false;
	if(!(checkSysCall(system(CUDACompatibilityCMD.c_str()), "Cannot compile and run cudaDeviceQuery"))) hasCUDA = false;
	
	if(hasOCL && hasCUDA) {
		printf("Compatible with both OpenCL and CUDA!\n");
		return 3;
	} else if(hasOCL && !hasCUDA) {
		printf("Only compatible with both OpenCL!\n");
		return 2;
	} else if(!hasOCL && hasCUDA) {
		printf("Only compatible with both CUDA!\n");
		return 1;
	} else {
		printf("Compatible with neither OpenCL nor CUDA!\n");
		return 0;
	}
	
}
/*void getGPUInfo(int sdkSelector, const char * compileCommand) {
    switch (sdkSelector) {
        // sdkSelector == 0 --> OpenCL
        case 0:
        	try {
            if(!(checkSysCall(system(compileCommand), "Cannot compile cl_info"))) return;
            if(!(checkSysCall(system("./cl_info"), "Cannot run cl_info"))) return;
					}
					catch (std::exception& ex) {
						printf("Not Compatible with OpenCL");
					}
            break;
            
        // sdkSelector == 1 --> CUDA
        case 1: break;
        
        // sdkSelector == 2 --> Serial implementation
        case 2: break;
    }
    
}*/

bool checkFile(string file) {
	if(file.compare("Iris.csv") != 0) return false;
	return true;
}
KMeans initKMeans(string data_file) {
	KMeans kmeans;
  // 1. Parse iris.csv into vector of "Flower" structs
  // Hyperparameters are hardcoded for now
	int numClusters = 3;
	int clusterInitType = 0; // Initiate cluster at random sample of dataset
	
  kmeans.parseDataset(data_file.c_str());
    
  // Print datafile
  //kmeans.printDataset();
	
	// Initialize centroids
	kmeans.initClusters(clusterInitType, 3);
	//kmeans.printCentroids();
	
	return kmeans;
}
int generateUserInterface(int compatibility) {
	string load_data = "\nPlease give me a source file to load your data: ";
	string load_clusters = "\nHow many clusters would you like to use to initialize KMeans? ";
	string load_iterations = "\nFor how many iterations would you like to run KMeans? ";
	switch(compatibility) {
   		case 0: 
   			{
   				printf("Unfortunately, you have neither a CUDA nor OpenCL enabled device... BUMMER!\n");
   				return(1);
   			}
   			break;
   		case 1: 
   			{
  				// System Detected only CUDA so compile only CUDA KMeans applications
  				system(CUDACompileKMeans.c_str());
  				
  				// Create menu for user options
  				string optionMenu = "Your device is only CUDA enabled... Let's get started!\n"
  				"What would you like to do?\n\t"
  				"1. Run KMeans on CUDA\n\t2. Exit\n";
  				string option;
  				string data_set;
  				string numClusters;
  				string numIterations;
  				while(true) {
  					cout << optionMenu;
  					getline (std::cin, option);
  					if(option.compare("2") != 0 && option.compare("1") != 0) {
  							cout << "Please type either 1 or 2 and press enter" << endl;
  					} else if(option.compare("2") == 0) {
  						break;
  					} else {
  						// Assuming the user input a good value and does not wish to exit, continue processing
  						cout << "Great! Now I need some information..." << endl;
  						cout << load_data;	// Request datafile
  						getline(std::cin, data_set); 
  							
  						cout << load_clusters;	// Request number of clusters to initialize KMeans
  						getline(std::cin, numClusters);	
  							
  						cout << load_iterations;	// Request number of iterations to execute KMeans
  						getline(std::cin, numIterations);
  							
  						// Generate Run Strings to launch OpenCL and/or CUDA application
  						CUDARunKMeans = CUDARunKMeans + " " + data_set + " " + numClusters + " " + numIterations;
  						cout << "Executing KMeans in OpenCL given specified parameters...\n" << endl;
  						system(OCLRunKMeans.c_str());	// Execute KMeans in OpenCL
  						
  					}  
  				}
  				break;
  			}
  		case 2: 
  			{
  				// System Detected only OpenCL so compile only OpenCL KMeans applications
  				system(OCLCompileKMeans.c_str());
  				
  				// Create menu for user options
  				string optionMenu = "Your device is only OpenCL enabled... Let's get started!\n"
  				"What would you like to do?\n\t"
  				"1. Run KMeans on OpenCL\n\t2. Exit\n";
  				string option;
  				string data_set;
  				string numClusters;
  				string numIterations;
  				while(true) {
  					cout << optionMenu;
  					getline (std::cin, option);
  					if(option.compare("2") != 0 && option.compare("1") != 0) {
  							cout << "Please type either 1 or 2 and press enter" << endl;
  					} else if(option.compare("2") == 0) {
  						break;
  					} else {
  						// Assuming the user input a good value and does not wish to exit, continue processing
  						cout << "Great! Now I need some information..." << endl;
  						cout << load_data;	// Request datafile
  						getline(std::cin, data_set); 
  							
  						cout << load_clusters;	// Request number of clusters to initialize KMeans
  						getline(std::cin, numClusters);	
  							
  						cout << load_iterations;	// Request number of iterations to execute KMeans
  						getline(std::cin, numIterations);
  							
  						// Generate Run Strings to launch OpenCL and/or CUDA application
  						OCLRunKMeans = OCLRunKMeans + " " + data_set + " " + numClusters + " " + numIterations;
  						cout << "Executing KMeans in OpenCL given specified parameters...\n" << endl;
  						system(OCLRunKMeans.c_str());	// Execute KMeans in OpenCL
  						
  					}  
  				}
  			}
  				break;
  		case 3: 
  			{
  				// System Detected OpenCL and CUDA so compile both OpenCL and CUDA KMeans applications
  				system(OCLCompileKMeans.c_str());
  				system(CUDACompileKMeans.c_str());
  				
  				// Create menu for user options
  				string optionMenu = "Your device is both OpenCL and CUDA enabled... Let's get started!\n"
  				"What would you like to do?\n\t"
  				"1. Run KMeans on OpenCL\n\t2. Run KMeans in CUDA\n\t"
  				"3. Run KMeans in both OpenCL and CUDA\n\t4. Exit\n";
  				string option;
  				string data_set;
  				string numClusters;
  				string numIterations;
  				while(true) {
  					cout << optionMenu;
  					getline (std::cin, option);
  					if(option.compare("4") != 0 && option.compare("3") != 0 && option.compare("2") != 0 && option.compare("1") != 0) {
  							cout << "Please type an integer between 1 and 4 and press enter" << endl;
  					} else if(option.compare("4") == 0) {
  						break;
  					} else {
  							// Assuming the user input a good value and does not wish to exit, continue processing
  							cout << "Great! Now I need some information..." << endl;
  							cout << load_data;	// Request datafile
  							getline(std::cin, data_set); 
  							
  							cout << load_clusters;	// Request number of clusters to initialize KMeans
  							getline(std::cin, numClusters);	
  							
  							cout << load_iterations;	// Request number of iterations to execute KMeans
  							getline(std::cin, numIterations);
  							
  							// Generate Run Strings to launch OpenCL and/or CUDA application
  							OCLRunKMeans = OCLRunKMeans + " " + data_set + " " + numClusters + " " + numIterations;
  						 	CUDARunKMeans = CUDARunKMeans + " " + data_set + " " + numClusters + " " + numIterations;
  						 	
  							// Run KMeans in both OpenCL and CUDA
  						 	if(option.compare("3") == 0) {
  						 		
  						 		cout << "Executing KMeans in OpenCL given specified parameters...\n" << endl;
  						 		system(OCLRunKMeans.c_str());	// Execute KMeans in OpenCL
  						 		
  						 		cout << "Executing KMeans in OpenCL given specified parameters...\n" << endl;
  						 		system(CUDARunKMeans.c_str());	// Execute KMeans in CUDA 
  						 	
  							} 
  							// Run KMeans in only CUDA
  							else if(option.compare("2") == 0) {
  								cout << "Executing KMeans in OpenCL given specified parameters...\n" << endl;
  						 		system(CUDARunKMeans.c_str());	// Execute KMeans in CUDA
  							} 
  							// Run KMeans in only OpenCL
  							else {
  								cout << "Executing KMeans in OpenCL given specified parameters...\n" << endl;
  						 		system(OCLRunKMeans.c_str());	// Execute KMeans in OpenC
  							}
  					}  
  				}
  			}
  			break;
  		default: break;
   }
   printf("Goodbye!\n");
   return 0;

}

int main(int argc, char** argv) {
	 printf("%s", welcomeMessage.c_str());
   setCompileStrings();
   if(generateUserInterface(getGPUInfo()) != 0) return(EXIT_SUCCESS);
   
}
