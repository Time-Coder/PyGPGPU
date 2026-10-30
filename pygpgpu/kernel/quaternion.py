"""Quaternion arithmetic on the device.

`pygf` defines a quaternion as a `ctypes.Structure` of four floats and gives it arithmetic --
`q * q` composes two rotations, `q * v` rotates a vector, `conjugate(q)` flips the axis.
Those meanings cannot be expressed as ordinary C operators on a structure, so each operation
becomes a small generated device function, and the translator emits a call to it instead of an
operator. The bodies below mirror `genQuat._op` exactly: a rotation is
`q * v * conjugate(q) / |q|`, which is what the host computes, so the two sides agree bit for
bit up to float rounding.

Only the operations with a device meaning are generated: compose, conjugate, rotate a vector,
scale by a scalar, and the component-wise cases that fall through to the base class. Anything
else is refused by the caller.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

__all__ = [
    "quat_member_names",
    "is_quaternion",
    "device_function",
]


# Component names in the order `pygf` declares them.
_MEMBERS:Tuple[str, ...] = ("w", "x", "y", "z")


def quat_member_names(dtype:np.dtype)->Optional[Tuple[str, ...]]:
    """The component names of a quaternion dtype, or None when it is not a quaternion.

    A quaternion is recognised by its layout -- four floats named w, x, y, z -- because that
    is all a dtype can say. A caller's own `{float w; float x; float y; float z;}` structure
    with no quaternion meaning would match too, but it would also have no arithmetic to ask
    for, so nothing is gained by distinguishing them.
    """
    names:Tuple[str, ...] = tuple(dtype.names or ())
    if names != _MEMBERS:
        return None
    for name in names:
        field_dtype:np.dtype = np.dtype(dtype.fields[name][0])
        if field_dtype.name not in ("float32", "float64"):
            return None
    return names


def is_quaternion(type_name:Any)->Optional[np.dtype]:
    """The quaternion dtype behind a device type name, or None."""
    from ..kernel.structs import dtype_for  # noqa: PLC0415 (import cycle)

    if not isinstance(type_name, str):
        return None
    dtype:Optional[np.dtype] = dtype_for(type_name)
    if dtype is None:
        return None
    return dtype if quat_member_names(dtype) is not None else None


def _component(dtype:np.dtype)->str:
    return "double" if np.dtype(dtype.fields["w"][0]).name == "float64" else "float"


def _hamilton(name:str, dtype:np.dtype, qual:str)->str:
    return f"""{qual}static {name} {name}_mul({name} a, {name} b) {{
    {name} r;
    r.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z;
    r.x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y;
    r.y = a.w * b.y + a.y * b.w + a.z * b.x - a.x * b.z;
    r.z = a.w * b.z + a.z * b.w + a.x * b.y - a.y * b.x;
    return r;
}}"""


def _conjugate(name:str, dtype:np.dtype, qual:str)->str:
    return f"""{qual}static {name} {name}_conjugate({name} a) {{
    {name} r;
    r.w = a.w;
    r.x = -a.x;
    r.y = -a.y;
    r.z = -a.z;
    return r;
}}"""


def _scale(name:str, dtype:np.dtype, qual:str)->str:
    t:str = _component(dtype)
    return f"""{qual}static {name} {name}_scale({name} a, {t} s) {{
    {name} r;
    r.w = a.w * s;
    r.x = a.x * s;
    r.y = a.y * s;
    r.z = a.z * s;
    return r;
}}"""


def _rotate(name:str, dtype:np.dtype, vec:str, qual:str)->str:
    """`q * v` as `q * (0, v) * conjugate(q) / |q|`, exactly what the host computes."""
    t:str = _component(dtype)
    vname:str = vec
    return f"""{qual}static {vname} {name}_rotate({name} q, {vname} v) {{
    {t} pw = ({t})0, px = v.x, py = v.y, pz = v.z;
    {name} conj;
    conj.w = q.w; conj.x = -q.x; conj.y = -q.y; conj.z = -q.z;
    {t} aw = q.w * pw - q.x * px - q.y * py - q.z * pz;
    {t} ax = q.w * px + q.x * pw + q.y * pz - q.z * py;
    {t} ay = q.w * py + q.y * pw + q.z * px - q.x * pz;
    {t} az = q.w * pz + q.z * pw + q.x * py - q.y * px;
    {t} rx = aw * conj.x + ax * conj.w + ay * conj.z - az * conj.y;
    {t} ry = aw * conj.y + ay * conj.w + az * conj.x - ax * conj.z;
    {t} rz = aw * conj.z + az * conj.w + ax * conj.y - ay * conj.x;
    {t} length = ({t})sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    {vname} out;
    out.x = rx / length;
    out.y = ry / length;
    out.z = rz / length;
    return out;
}}"""


_CACHE:Dict[Tuple[str, str, str, str], str] = {}

# CUDA needs the execution space spelled out on a plain function; OpenCL has no such
# annotation, and a `__device__` there would not compile.
_QUALIFIER:str = ""


def device_function(operation:str, quat_name:str, dtype:np.dtype,
                    vec_name:str = "", backend:str = "opencl") -> Tuple[str, str]:
    """The device function for one quaternion operation: (its name, its definition).

    ``operation`` is one of ``mul``, ``conjugate``, ``scale``, ``rotate``.
    """
    # The key includes the backend: the generated body carries that backend's execution-space
    # qualifier, so a definition made for one must not be handed to the other.
    key:Tuple[str, str, str, str] = (operation, quat_name, vec_name, backend)
    cached:Optional[str] = _CACHE.get(key)
    function_name:str = f"{quat_name}_{operation}"
    if cached is not None:
        return function_name, cached
    qual:str = "__device__ " if backend in ("cuda", "hip") else ""
    if operation == "mul":
        body:str = _hamilton(quat_name, dtype, qual)
    elif operation == "conjugate":
        body = _conjugate(quat_name, dtype, qual)
    elif operation == "scale":
        body = _scale(quat_name, dtype, qual)
    elif operation == "rotate":
        body = _rotate(quat_name, dtype, vec_name, qual)
    else:
        raise ValueError(f"no quaternion device function for '{operation}'")

    _CACHE[key] = body
    return function_name, body
