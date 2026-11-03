// vecops.cl -- the OpenCL half of a three-backend module imported through kernel_import.
// vecops.cu and vecops.hip sit beside this file: same module name, one backend each.
typedef struct
{
    float x;
    float y;
} Pt;

// Declared in all three files with exactly these parameters, so the merged module declares
// a single __call__ for it and each backend runs its own copy.
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

// Float here, double in vecops.cu: the parameter lists differ, so the merged module declares
// two @overloads -- one per parameter list -- and the call picks the one it fits.
__kernel void shift(
    __global float* out,
    const float factor,
    const int n)
{
    int i = get_global_id(0);
    if (i >= n) return;
    out[i] = out[i] + factor;
}

// The same parameters in all three files, with a struct argument and a struct array.
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

// Deliberately absent from vecops.cu: asking for it on CUDA raises KeyError at the bracket.
__kernel void negate(
    __global float* out,
    const int n)
{
    int i = get_global_id(0);
    if (i >= n) return;
    out[i] = -out[i];
}
