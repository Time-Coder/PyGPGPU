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
    "HEADER_SOURCE",
    "header_source",
    "install_header",
    "include_line",
]

# Component names in the order `pygf` declares them.
_MEMBERS:Tuple[str, ...] = ("w", "x", "y", "z")

HEADER_NAME:str = "pygpgpu_quat.h"

# The fixed header is a real file beside this module, so it can be read and edited like any
# other source. install_header copies it into the cache folder next to the generated sources,
# because a quoted #include resolves relative to the including file.
HEADER_SOURCE:str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "headers", HEADER_NAME)


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


def header_source()->str:
    """The fixed header's text, read from the file beside this module.

    Read on demand rather than cached: the file is a normal source a user may edit while
    debugging, and a stale copy would make the cache silently disagree with it.
    """
    with open(HEADER_SOURCE, "r", encoding="utf-8") as file:
        return file.read()


def install_header(directory:str)->str:
    """Copy the fixed header into a folder beside the generated sources.

    A copy is required, not a reference: `#include "...` resolves relative to the file that
    contains it, so the header has to sit where the generated source can see it.
    """
    os.makedirs(directory, exist_ok=True)
    path:str = os.path.join(directory, HEADER_NAME)
    _write_if_changed(path, header_source())
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
