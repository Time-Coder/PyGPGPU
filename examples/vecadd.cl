// vecadd.cl -- handwritten OpenCL kernels, importable like a Python module.
// Lives in examples/, next to 06_handwritten_kernels.py.
// Struct typedefs map automatically to ctypes types; see 06_handwritten_kernels.py.
typedef struct
{
    float x;
    float y;
} Pt;

// Automatically derived work sizes may round the global size up to a multiple of
// the local size, so a kernel must clamp against the real element count. It cannot
// use get_global_size(0) for that: that reports the (possibly rounded) launch size.
__kernel void add_offset(
    __global const float* a,
    __global const float* b,
    __global float* out,
    const float offset,
    const int n)
{
    int i = get_global_id(0);
    if (i >= n) return;
    out[i] = a[i] + b[i] + offset;
}

__kernel void in_circle(
    __global const Pt* points,
    __global int* results,
    const Pt center,
    const float radius,
    const int n)
{
    int i = get_global_id(0);
    if (i >= n) return;
    float dx = points[i].x - center.x;
    float dy = points[i].y - center.y;
    results[i] = (dx * dx + dy * dy <= radius * radius) ? 1 : 0;
}