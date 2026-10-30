/*
 * Quaternion arithmetic for generated kernels.
 *
 * Included through a generated shim, which defines four macros first:
 *   PYGPGPU_QUAT_NAME    the C name of the quaternion structure (a generated `_st_<hash>`)
 *   PYGPGPU_QUAT_SCALAR  float or double
 *   PYGPGPU_QUAT_VEC     the vector type a rotation produces (`float3`, `double3`, ...)
 *   PYGPGPU_QUAT_QUAL    empty, or `__device__` where the backend demands it
 *
 * The structure itself is declared by the generated source, ahead of this include, because
 * its C name depends on its layout; this file only adds the arithmetic.
 *
 * The bodies mirror pygf's `genQuat._op` exactly:
 *   q * q  -> Hamilton product
 *   q * v  -> q * (0, v) * conj(q) / |q|, normalised by q's magnitude
 *   q * s  -> per-component scale
 *   conj   -> (w, -x, -y, -z)
 * so the device answer matches the interpreter.
 */
#ifndef PYGPGPU_QUAT_H
#define PYGPGPU_QUAT_H

#ifndef PYGPGPU_QUAT_QUAL
#define PYGPGPU_QUAT_QUAL
#endif

#define PYGPGPU_QUAT_CAT_(a, b) a##b
#define PYGPGPU_QUAT_CAT(a, b) PYGPGPU_QUAT_CAT_(a, b)
#define PYGPGPU_QUAT_FN(suffix) PYGPGPU_QUAT_CAT(PYGPGPU_QUAT_NAME, suffix)

/* q * r: compose two rotations. */
PYGPGPU_QUAT_QUAL static PYGPGPU_QUAT_NAME PYGPGPU_QUAT_FN(_mul)
    (PYGPGPU_QUAT_NAME a, PYGPGPU_QUAT_NAME b)
{
    PYGPGPU_QUAT_NAME r;
    r.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z;
    r.x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y;
    r.y = a.w * b.y + a.y * b.w + a.z * b.x - a.x * b.z;
    r.z = a.w * b.z + a.z * b.w + a.x * b.y - a.y * b.x;
    return r;
}

/* q + r: component-wise, which is what the base class does. */
PYGPGPU_QUAT_QUAL static PYGPGPU_QUAT_NAME PYGPGPU_QUAT_FN(_add)
    (PYGPGPU_QUAT_NAME a, PYGPGPU_QUAT_NAME b)
{
    PYGPGPU_QUAT_NAME r;
    r.w = a.w + b.w;
    r.x = a.x + b.x;
    r.y = a.y + b.y;
    r.z = a.z + b.z;
    return r;
}

/* q - r: component-wise. */
PYGPGPU_QUAT_QUAL static PYGPGPU_QUAT_NAME PYGPGPU_QUAT_FN(_sub)
    (PYGPGPU_QUAT_NAME a, PYGPGPU_QUAT_NAME b)
{
    PYGPGPU_QUAT_NAME r;
    r.w = a.w - b.w;
    r.x = a.x - b.x;
    r.y = a.y - b.y;
    r.z = a.z - b.z;
    return r;
}

/* conjugate(q) */
PYGPGPU_QUAT_QUAL static PYGPGPU_QUAT_NAME PYGPGPU_QUAT_FN(_conjugate)
    (PYGPGPU_QUAT_NAME a)
{
    PYGPGPU_QUAT_NAME r;
    r.w = a.w;
    r.x = -a.x;
    r.y = -a.y;
    r.z = -a.z;
    return r;
}

/* q * scalar */
PYGPGPU_QUAT_QUAL static PYGPGPU_QUAT_NAME PYGPGPU_QUAT_FN(_scale)
    (PYGPGPU_QUAT_NAME a, PYGPGPU_QUAT_SCALAR s)
{
    PYGPGPU_QUAT_NAME r;
    r.w = a.w * s;
    r.x = a.x * s;
    r.y = a.y * s;
    r.z = a.z * s;
    return r;
}

/* q * v: rotate the vector, normalised by q's magnitude. */
PYGPGPU_QUAT_QUAL static PYGPGPU_QUAT_VEC PYGPGPU_QUAT_FN(_rotate)
    (PYGPGPU_QUAT_NAME q, PYGPGPU_QUAT_VEC v)
{
    PYGPGPU_QUAT_SCALAR pw = (PYGPGPU_QUAT_SCALAR)0;
    PYGPGPU_QUAT_SCALAR px = v.x, py = v.y, pz = v.z;
    PYGPGPU_QUAT_NAME conj;
    conj.w = q.w; conj.x = -q.x; conj.y = -q.y; conj.z = -q.z;

    PYGPGPU_QUAT_SCALAR aw = q.w * pw - q.x * px - q.y * py - q.z * pz;
    PYGPGPU_QUAT_SCALAR ax = q.w * px + q.x * pw + q.y * pz - q.z * py;
    PYGPGPU_QUAT_SCALAR ay = q.w * py + q.y * pw + q.z * px - q.x * pz;
    PYGPGPU_QUAT_SCALAR az = q.w * pz + q.z * pw + q.x * py - q.y * px;

    PYGPGPU_QUAT_SCALAR rx = aw * conj.x + ax * conj.w + ay * conj.z - az * conj.y;
    PYGPGPU_QUAT_SCALAR ry = aw * conj.y + ay * conj.w + az * conj.x - ax * conj.z;
    PYGPGPU_QUAT_SCALAR rz = aw * conj.z + az * conj.w + ax * conj.y - ay * conj.x;

    PYGPGPU_QUAT_SCALAR length =
        (PYGPGPU_QUAT_SCALAR)sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);

    PYGPGPU_QUAT_VEC out;
    out.x = rx / length;
    out.y = ry / length;
    out.z = rz / length;
    return out;
}

#endif /* PYGPGPU_QUAT_H */
