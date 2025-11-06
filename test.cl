__kernel void test_kernel(__global int* a)
{
    int idx = get_global_id(0);
    a[idx] = idx + missing_variable;
}