// vecadd_cuda.cu -- handwritten CUDA kernel, importable like a Python module.
// Demonstrates the same thing as vecadd.cl, but through cudaimport.
__global__ void cuda_add_offset(
    const float* a,
    const float* b,
    float* out,
    float offset)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = a[i] + b[i] + offset;
}