// reference: https://github.com/andreaferretti/kmeans/blob/master/opencl/kmeans.cl
// hello
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
__kernel void kmeans_assignment(__global Flower* flowers,
                                __global Centroid* centroids,
                                int num_points, int num_centroids) {
    int idx = get_global_id(0);
    float min_dist = 1000000.0f;
    float d = 0.0f;
    int i = 0;
    if(idx < num_points) {
        for(i = 0; i < num_centroids; i++) {
            d = euclid_dist(flowers + idx, centroids + i);
            
            if(d < min_dist) {
                min_dist = d;
                flowers[idx].clust = i;
            }
        }
    }
}

__kernel void sum(__global Flower* flowers, __global Sum* sum, __local Sum* scratch, int num_points, int num_centroids) {
    int lid = get_local_id(0);
    int wid = get_group_id(0);
    int gid = get_global_id(0);
    int pos = lid * num_centroids;
    int s;
    int j;
    
    for(s = pos; s < pos + num_centroids; s++) {
        scratch[s].s_l_sum = 0.0f;
        scratch[s].s_w_sum = 0.0f;
        scratch[s].p_l_sum = 0.0f;
        scratch[s].p_w_sum = 0.0f;
        scratch[s].num_points = 0;
    }
    
    if(gid < num_points) {
        int cluster = flowers[gid].clust;
        scratch[pos + cluster].s_l_sum = flowers[gid].sepal_length;
        scratch[pos + cluster].s_w_sum = flowers[gid].sepal_width;
        scratch[pos + cluster].p_l_sum = flowers[gid].petal_length;
        scratch[pos + cluster].p_w_sum = flowers[gid].petal_width;
        scratch[pos + cluster].num_points = 1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for(s = get_local_size(0)/2; s > 0; s = s/2) {
        if(lid < s) {
            for(j = 0; j < num_centroids; j++) {
                int dst = pos+j;
                int src = pos + j + s * num_centroids;
                scratch[dst].s_l_sum += scratch[src].s_l_sum;
                scratch[dst].s_w_sum += scratch[src].s_w_sum;
                scratch[dst].p_l_sum += scratch[src].p_l_sum;
                scratch[dst].p_w_sum += scratch[src].p_w_sum;
                scratch[dst].num_points += scratch[src].num_points;
                
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
   if(lid == 0) {
        for(j = 0; j < num_centroids; j++) {
            int h = wid * num_centroids + j;
            
            sum[h].s_l_sum = scratch[pos + j].s_l_sum;
            sum[h].s_w_sum = scratch[pos + j].s_w_sum;
            sum[h].p_l_sum = scratch[pos + j].p_l_sum;
            sum[h].p_w_sum = scratch[pos + j].p_w_sum;
            sum[h].num_points = scratch[pos + j].num_points;
        }
    }
}

__kernel void kmeans_update(__global Sum* sum, __global Centroid* centroids, int work_groups, int num_centroids) {
    int gid = get_global_id(0);
    float s_l_sum = 0.0f;
    float s_w_sum = 0.0f;
    float p_l_sum = 0.0f;
    float p_w_sum = 0.0f;
    int num_points = 0;
    int i;
    
    if(gid < num_centroids) {
        for(i = 0; i < work_groups; i++) {
            int h = i * num_centroids + gid;
            s_l_sum += sum[h].s_l_sum;
            s_w_sum += sum[h].s_w_sum;
            p_l_sum += sum[h].p_l_sum;
            p_w_sum += sum[h].p_w_sum;
            num_points += sum[h].num_points;
        }
        if(num_points > 0) {
            
            centroids[gid].sepal_length = s_l_sum/(float)num_points;
            centroids[gid].sepal_width = s_w_sum/(float)num_points;
            centroids[gid].petal_length = p_l_sum/(float)num_points;
            centroids[gid].petal_width = p_w_sum/(float)num_points;
        }
    }
}
