typedef struct {
    float x;
    float y;
    float z;
} Point;

__kernel void compute_distance(__global float* output, int length, Point p)
{
    int i = get_global_id(0);

    if (i >= length)
    {
        return;
    }
    
    float dist = sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
    output[i] = dist;
}