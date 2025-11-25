typedef struct {
    float x, y, z;
} Point3D;

typedef struct {
    Point3D position;
    Point3D velocity;
    int id;
    float mass;
} Particle;

__kernel void update_particle(
    __global float4* output,
    Particle p
) {
    int gid = get_global_id(0);
    if (gid != 0) return;

    float dt = 0.01f;
    float new_x = p.position.x + p.velocity.x * dt;
    float new_y = p.position.y + p.velocity.y * dt;
    float new_z = p.position.z + p.velocity.z * dt;

    output[gid] = (float4)(new_x, new_y, new_z, p.velocity.x);
}