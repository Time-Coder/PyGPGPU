__kernel void test_kernel(__global int* a, int2 shape)
{
    int x = get_global_id(0);
    if (x >= shape.x)
    {
        return;
    }

    int y = get_global_id(1);
    if (y >= shape.y)
    {
        return;
    }

    int idx = x * shape.y + y;
    a[idx] = idx;
}