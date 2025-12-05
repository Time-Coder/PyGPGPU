typedef struct
{
    float x;
    float y;
} Point;

typedef struct
{
    Point center;
    float radius;
} Circle;

struct Test
{
    int a, b;
    int c[2];
    int *d, *e;
    int f[2][3];
    const int g;
    const int* h;
    __global const volatile int* i;
};

bool isInside(const Point p, const Circle c)
{
    float dx = p.x - c.center.x;
    float dy = p.y - c.center.y;
    float dist_sq = dx * dx + dy * dy;
    return dist_sq <= c.radius * c.radius;
}

__kernel void check_points_in_circle(
    __global const Point* points,
    __global int* results,
    const Circle circle,
    const int num_points
)
{
    int gid = get_global_id(0);
    if (gid >= num_points) return;

    Point p = points[gid];
    results[gid] = isInside(p, circle) ? 1 : 0;
}

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