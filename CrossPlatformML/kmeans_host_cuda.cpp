#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include "kmeans.h"
using namespace std;

int main(int argc, char** argv) {
	
    KMeans kmeans;
    // 1. Parse iris.csv into vector of "Flower" structs
	// Initial factors will likely be specified via commandline
	// For now they are hardcoded
    const char * data_file = "iris.csv";
	int numClusters = 3;
	int clusterInitType = 0; // Initiate cluster at random sample of dataset
	
    kmeans.parseDataset(data_file);
    
    // Print datafile
    kmeans.printDataset();
	
	// Initialize centroids
	kmeans.initClusters(clusterInitType, 3);
	kmeans.printCentroids();
	
    return 0;
}
