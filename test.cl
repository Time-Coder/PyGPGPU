__kernel void flipY(__read_only image2d_t src_image, __write_only image2d_t dest_image, sampler_t s)
{
    int x = get_global_id(1);
    int y = get_global_id(0);

    int width = get_image_width(src_image);
    int height = get_image_height(src_image);

    if (x >= width || y >= height)
    {
        return;
    }

    int flippedY = height - y - 1;

    float4 pixel = read_imagef(src_image, s, (int2)(x, flippedY));

    write_imagef(dest_image, (int2)(x, y), pixel);
}