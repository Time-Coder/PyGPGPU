// vecops.cu -- the CUDA half of a three-backend module imported through kernel_import.
typedef struct
{
    float x;
    float y;
} Pt;

// Same parameter list as the OpenCL and HIP declarations: one kernel, three implementations.
__global__ void add_offset(
    const float* a,
    const float* b,
    float* out,
    float offset,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[i] + b[i] + offset;
}

// Double precision here, float in vecops.cl and vecops.hip: a second overload.
__global__ void shift(
    double* out,
    double factor,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = out[i] + factor;
}

// Same parameter list as the other two files.
__global__ void in_circle(
    const Pt* points,
    int* results,
    Pt center,
    float radius,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float dx = points[i].x - center.x;
    float dy = points[i].y - center.y;
    results[i] = (dx * dx + dy * dy <= radius * radius) ? 1 : 0;
}
