"""The fixed quaternion header, and the per-type shim that names a structure for it.

The arithmetic lives once, in ordinary C, in a header that is copied into a shared folder
beside the caches. A generated source gets a quaternion type whose C name is a content hash
(`_st_<digest>`), so the header cannot spell that name itself; it takes the name, the scalar
type, the vector type and the execution-space qualifier from macros, which a small generated
**shim** defines before including the header. One shim per (quaternion type, vector type,
backend), and the generated source includes it with one line.

`#include` is expanded by the C preprocessor *before* the backend compiler runs
(`macros_expand` with `-I{path}`), so the header is plain text folded into the generated file.
The driver never resolves an include path of its own, which is why the header goes next to
that generated file rather than somewhere the compiler would have to be told about.

The bodies mirror `genQuat._op` line for line. A rotation is `q * v * conj(q) / |q|`,
normalised by **q**'s magnitude rather than the product's -- that is what makes the device
answer match the interpreter, and getting it wrong showed up as a factor-of-three difference.
"""
from __future__ import annotations

import os
from typing import Any, Optional, Tuple

import numpy as np

__all__ = [
    "quat_member_names",
    "is_quaternion",
    "HEADER_NAME",
    "install_header",
    "include_line",
]

# Component names in the order `pygf` declares them.
_MEMBERS:Tuple[str, ...] = ("w", "x", "y", "z")

HEADER_NAME:str = "pygpgpu_quat.h"

_HEADER_BODY:str = """/*
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
"""


def quat_member_names(dtype:np.dtype)->Optional[Tuple[str, ...]]:
    """The component names of a quaternion dtype, or None when it is not a quaternion.

    A quaternion is recognised by its layout -- four floats named w, x, y, z -- because that
    is all a dtype can say. A caller's own `{float w; float x; float y; float z;}` structure
    with no quaternion meaning would match too, but it would also have no arithmetic to ask
    for, so nothing is gained by distinguishing them.
    """
    if tuple(dtype.names or ()) != _MEMBERS:
        return None
    for name in _MEMBERS:
        field_dtype:np.dtype = np.dtype(dtype.fields[name][0])
        if field_dtype.name not in ("float32", "float64"):
            return None
    return _MEMBERS


def is_quaternion(type_name:Any)->Optional[np.dtype]:
    """The quaternion dtype behind a device type name, or None."""
    if not isinstance(type_name, str):
        return None
    from .structs import dtype_for  # noqa: PLC0415 (import cycle)

    dtype:Optional[np.dtype] = dtype_for(type_name)
    if dtype is None:
        return None
    return dtype if quat_member_names(dtype) is not None else None


def _scalar_name(dtype:np.dtype)->str:
    return "double" if np.dtype(dtype.fields["w"][0]).name == "float64" else "float"


def _write_if_changed(path:str, text:str)->None:
    existing:Optional[str] = None
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as file:
            existing = file.read()
    if existing != text:
        with open(path, "w", encoding="utf-8", newline="\n") as file:
            file.write(text)


def install_header(directory:str)->str:
    """Write the fixed header into a shared folder and return its path."""
    os.makedirs(directory, exist_ok=True)
    path:str = os.path.join(directory, HEADER_NAME)
    _write_if_changed(path, _HEADER_BODY)
    return path


def include_line(quat_name:str, dtype:np.dtype, vector_name:str, backend:str,
                 directory:str)->str:
    """Write the per-type shim into ``directory`` and return the `#include` line for it.

    The shim is what names a generated structure for the fixed header: four macros, then the
    include. The header it names is always installed first, so the pair is consistent.
    """
    qualifier:str = "__device__" if backend in ("cuda", "hip") else ""
    install_header(directory)
    body:str = (
        f"/* Generated: names a quaternion structure for {HEADER_NAME}. Do not edit. */\n"
        f"#define PYGPGPU_QUAT_NAME {quat_name}\n"
        f"#define PYGPGPU_QUAT_SCALAR {_scalar_name(dtype)}\n"
        f"#define PYGPGPU_QUAT_VEC {vector_name}\n"
        f"#define PYGPGPU_QUAT_QUAL {qualifier}\n"
        f'#include "{HEADER_NAME}"\n')
    name:str = f"pygpgpu_quat_{quat_name}_{vector_name or 'scalar'}_{backend}.h"
    _write_if_changed(os.path.join(directory, name), body)
    return f'#include "{os.path.basename(directory)}/{name}"'
