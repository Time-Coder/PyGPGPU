__kernel void test_kernel(__global int* a, int size)
{
    int idx = get_global_id(0);
    if (idx >= size)
    {
        return;
    }
    
    a[idx] = idx;
}