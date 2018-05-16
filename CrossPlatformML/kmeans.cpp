#include "kmeans.h"
#include <iostream>
using namespace std;
#include <string>
#include <fstream>
#include <ctime>
#include <cstdlib>

void KMeans::parseDataset(const char * dataFile)
{
    	ifstream iris_data(dataFile);
    if(!iris_data.is_open()) {
     cout << "ERROR: File Open" << '\n';
     cout << "Gonna close..." << endl;
     iris_data.close();
     return;
    }
    int i = 0;
    
    while(iris_data.good()) {
        Flower flower;
            
        // temp storage for getline input
        string id;
        string sepal_length;
        string sepal_width;
        string petal_length;
        string petal_width;
        string species;
            
        // Parse file
        getline(iris_data, id, ',');
        getline(iris_data, sepal_length, ',');
        getline(iris_data, sepal_width, ',');
        getline(iris_data, petal_length, ',');
        getline(iris_data, petal_width, ',');
        getline(iris_data, species, '\n');
        
        // ignore title line
        if(i > 0) {
            flower.id = stoi(id);
            flower.sepal_length = stof(sepal_length);
            flower.sepal_width = stof(sepal_width);
            flower.petal_length = stof(petal_length);
            flower.petal_width = stof(petal_width);
            // convert species string to int for OpenCL struct compatibility
            if(species.compare("Iris-setosa") == 0) {
            	flower.species = 0;
            }
            else if(species.compare("Iris-versicolor") == 0) {
            	flower.species = 1;
            }
            else if(species.compare("Iris-virginica") == 0) {
            	flower.species = 2;
            }
            
            // Might be better way to initialize clust... idk
            flower.clust = 0;
            
            KMeans::flowers.push_back(flower);
        }
        i++;
    }
}

void KMeans::initClusters(int initType, int numClusters){
    if(!KMeans::flowers.empty()) {
        KMeans::k = numClusters;
        switch (initType) {
            // Random sample from
            case 0:
                srand(time(0));
                for (int i = 0; i < k; i++) {
                    Centroid centroid;
                    Flower flower  = KMeans::flowers[(rand() % (KMeans::flowers.size()))];
                    centroid.sepal_length = flower.sepal_length;
                    centroid.sepal_width = flower.sepal_width;
                    centroid.petal_length = flower.petal_length;
                    centroid.petal_width = flower.petal_width;
										KMeans::centroids.push_back(centroid);
                }
                
                
                break;
            default: break;
        }
    } else {
        printf("No dataset has been read in yet...\n");
    }
}

void KMeans::printDataset(){
    if(!KMeans::flowers.empty()) {
        for(int i = 0; i < KMeans::flowers.size(); i++) {
            printf("Sepal length: %f\nSepal width: %f\nPetal length: "
               "%f\nPetal width: %f\nSpecies: %d\nCluster: "
               "%d\n\n=================================\n\n",
               KMeans::flowers.at(i).sepal_length, KMeans::flowers.at(i).sepal_width,
               KMeans::flowers.at(i).petal_length, KMeans::flowers.at(i).petal_width,
               KMeans::flowers.at(i).species, KMeans::flowers.at(i).clust);
        }
    } else {
        printf("No dataset has been read in yet...\n");
    }
}


void KMeans::printCentroids(){
	if(!KMeans::centroids.empty()) {
		printf("Current Centroids:\n\n");
		for (int i = 0; i < KMeans::k; i++) {
			 printf("Sepal length: %f\nSepal width: %f\nPetal length: "
               "%f\nPetal width: %f\nCluster: "
               "%d\n\n=================================\n\n",
               KMeans::centroids.at(i).sepal_length, KMeans::centroids.at(i).sepal_width,
               KMeans::centroids.at(i).petal_length, KMeans::centroids.at(i).petal_width,
			   i);
		}
	
	} else {
		printf("Centroids have not been initialized yet...\n");
	}
}

vector<Flower> KMeans::getFlowers() {
	return KMeans::flowers;
}

vector<Centroid> KMeans::getCentroids() {
	return KMeans::centroids;
}

