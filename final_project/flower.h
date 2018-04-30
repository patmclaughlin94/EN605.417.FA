typedef struct {
    float s_l;
    float s_w;
    float p_l;
    float p_w;
    int clust;
} Flower;

typedef struct {
    float s_l;
    float s_w;
    float p_l;
    float p_w;
} Centroid;

typedef struct {
    float s_l_sum;
    float s_w_sum;
    float p_l_sum;
    float p_w_sum;
    int num_points;
} Sum;