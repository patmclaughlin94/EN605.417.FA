// hello
typedef struct {
    int id;
    float sepal_length;
    float sepal_width;
    float petal_length;
    float petal_width;
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
static float euclid_dist(__global Flower* f, __global Centroid* c) {
    float d_s_l = (*f).sepal_length - (*c).sepal_length;
    float d_s_w = (*f).sepal_width - (*c).sepal_width;
    float d_p_l = f->petal_length - c->petal_length;
    float d_p_w = f->petal_width- c->petal_width;
    return sqrt((d_s_l*d_s_l) + (d_s_w*d_s_w) + (d_p_l*d_p_l) + (d_p_w*d_p_w));
}
__kernel void add(__global const float *a,
                  __global const float *b,
                  __global float *result)
{
    int gid = get_global_id(0);
    
    result[gid] = a[gid] + b[gid];
}

__kernel void sub(__global const float *a,
                  __global const float *b,
                  __global float *result)
{
    int gid = get_global_id(0);
    
    result[gid] = a[gid] - b[gid];
}
__kernel void mult(__global const float *a,
                   __global const float *b,
                   __global float *result)
{
    int gid = get_global_id(0);
    
    result[gid] = a[gid] * b[gid];
}
__kernel void div(__global const float *a,
                  __global const float *b,
                  __global float *result)
{
    int gid = get_global_id(0);
    
    result[gid] = a[gid] / b[gid];
}
__kernel void power(__global const float *a,
                    int b,
                    __global float *result)
{
    int gid = get_global_id(0);
    
    result[gid] = pow(a[gid], b);
}


