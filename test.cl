__kernel void gaussian_blur(__read_only image2d_t src_image, __write_only image2d_t dest_image, sampler_t sampler)
{
    int2 coord = (int2)(get_global_id(1), get_global_id(0));

    int width = get_image_width(src_image);
    int height = get_image_height(src_image);

    if(coord.x >= width || coord.y >= height)
    {
        return;
    }

    float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    float factorSum = 0.0f;

    const float gaussianKernel[5][5] = {
        {1,  4,  6,  4,  1},
        {4, 16, 24, 16,  4},
        {6, 24, 36, 24,  6},
        {4, 16, 24, 16,  4},
        {1,  4,  6,  4,  1}
    };

    for (int y = -2; y <= 2; y++)
    {
        for (int x = -2; x <= 2; x++)
        {
            int currentX = coord.x + x;
            int currentY = coord.y + y;

            if (currentX >= 0 && currentX < width && currentY >= 0 && currentY < height)
            {
                float4 color = read_imagef(src_image, sampler, (int2)(currentX, currentY));
                float kernelValue = gaussianKernel[y+2][x+2];
                sum += color * kernelValue;
                factorSum += kernelValue;
            }
        }
    }

    float4 result = sum / factorSum;
    write_imagef(dest_image, coord, result);
}