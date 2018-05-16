using namespace std;
#include<string>
typedef struct {
    int id;
    float sepal_length;
    float sepal_width;
    float petal_length;
    float petal_width;
    int species;
    int clust;
} Flower;

typedef struct {
    float sepal_length;
    float sepal_width;
    float petal_length;
    float petal_width;
} Centroid;

typedef struct {
    float s_l_sum;
    float s_w_sum;
    float p_l_sum;
    float p_w_sum;
    int num_points;
} Sum;
