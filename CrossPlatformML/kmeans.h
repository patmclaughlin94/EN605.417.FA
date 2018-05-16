#ifndef KMEANS_H
#define KMEANS_H
// Not sure if all includes should go here... double check
#include "flower.h"
using namespace std;
#include <vector>

class KMeans
{
    int k;
    vector<Flower> flowers;
    vector<Centroid> centroids;
    public:
    void parseDataset(const char * dataFile);
    void initClusters(int initType, int numClusters);
    void printDataset();
    void printCentroids();
    vector<Flower> getFlowers();
    vector<Centroid> getCentroids();
};
#endif
