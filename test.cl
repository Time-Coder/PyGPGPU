__kernel void flip_y(__read_only image2d_t src_image, __write_only image2d_t dest_image, sampler_t sampler)
{
    int2 dest_coord = (int2)(get_global_id(1), get_global_id(0));

    int width = get_image_width(src_image);
    int height = get_image_height(src_image);

    if(dest_coord.x >= width || dest_coord.y >= height)
    {
        return;
    }

    int2 src_coord = (int2)(dest_coord.x, height - 1 - dest_coord.y);
    float4 result = read_imagef(src_image, sampler, src_coord);
    
    write_imagef(dest_image, dest_coord, result);
}

__kernel void test(__global int2* a, int length)
{
    int i = get_global_id(0);
    if (i < length)
    {
        a[i] = (int2)(i, 0);
    }
}