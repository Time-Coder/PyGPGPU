"""Quaternion structures and arithmetic on the device.

`pygf` gives a quaternion arithmetic the C operators cannot express: `q * q` composes two
rotations, `q * v` rotates a vector, `conjugate(q)` flips the axis. Those meanings live in one
fixed C header (`include/quat.h`) that declares both quaternion structures and every operation
on them, so a generated source that includes it needs no declaration of its own.

The structures carry **fixed** C names (`quatf` / `quatd`) rather than the content hashes other
structures get. That is deliberate: a hash cannot be spelled in advance, and the header has to
declare the type it operates on. Two quaternion layouts exist, so two fixed names suffice, and
the translator maps a quaternion dtype onto the right one.

A generated source chooses which blocks it needs through two macros:
`QUAT_NEED_FLOAT` / `QUAT_NEED_DOUBLE`. The guards matter because a `double` block only
compiles where the width does -- an OpenCL device without `cl_khr_fp64` rejects a `double`
variable even inside an unused function.

`#include` is expanded by the C preprocessor *before* the backend compiler runs, so the header
is plain text folded into the generated file. The driver never resolves an include path of its
own, which is why the header is copied into the cache beside the sources that include it.
"""
from __future__ import annotations

import os
from typing import Any, Optional, Tuple

import numpy as np

__all__ = [
    "quat_member_names",
    "is_quaternion",
    "quat_type_name",
    "needs",
    "HEADER_NAME",
    "HEADER_SOURCE",
    "header_source",
    "include_directory",
    "install_header",
    "include_line",
]

# Component names in the order `pygf` declares them.
_MEMBERS:Tuple[str, ...] = ("w", "x", "y", "z")

HEADER_NAME:str = "quat.h"

# The header is a real file beside this module, so it can be read and edited like any other
# source. `install_header` copies it into the cache's include folder, because a quoted
# `#include` resolves relative to the including file.
HEADER_SOURCE:str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "include", HEADER_NAME)

# The folder a cache keeps its device headers in.
INCLUDE_FOLDER:str = "include"

# The two fixed C names the header declares, keyed by the scalar width.
_TYPE_NAMES:dict = {"float32": "quatf", "float64": "quatd"}

# The macro that enables each block, keyed by the same widths.
_NEED_MACROS:dict = {"float32": "QUAT_NEED_FLOAT",
                     "float64": "QUAT_NEED_DOUBLE"}


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


def _scalar_width(dtype:np.dtype)->Optional[str]:
    if quat_member_names(dtype) is None:
        return None
    return np.dtype(dtype.fields["w"][0]).name


def quat_type_name(dtype:np.dtype)->Optional[str]:
    """The fixed C type name a quaternion dtype translates to, or None.

    A quaternion gets a name the header already knows instead of the content hash every other
    structure gets, which is what lets the header declare it.
    """
    width:Optional[str] = _scalar_width(dtype)
    return _TYPE_NAMES.get(width) if width is not None else None


def needs(dtype:np.dtype)->Optional[str]:
    """The macro a generated source defines to compile this quaternion's block."""
    width:Optional[str] = _scalar_width(dtype)
    return _NEED_MACROS.get(width) if width is not None else None


def mixed_needs()->tuple:
    """Both macros, for the mixed float/double block.

    A mixed operation widens to the double form, so its body touches both widths and the header
    only compiles it when neither name is missing -- which is also when it is legal, since a
    device without fp64 cannot do either.
    """
    return tuple(_NEED_MACROS[name] for name in ("float32", "float64"))


def double_dtype()->np.dtype:
    """The double quaternion dtype, which is what a mixed operation answers."""
    return np.dtype([(name, "<f8") for name in _MEMBERS])


def wider(first:Optional[np.dtype], second:Optional[np.dtype])->Optional[np.dtype]:
    """The quaternion dtype a mixed pair promotes to; None when neither operand is one.

    `pygf` answers the double form for every mixed combination, so this only ever widens: a
    float quaternion meeting a double operand becomes the double one.
    """
    for candidate in (first, second):
        if candidate is not None and _scalar_width(candidate) == "float64":
            return candidate
    for candidate in (first, second):
        if candidate is not None:
            return candidate
    return None


def matches_vector(dtype:np.dtype, vector_name:str)->bool:
    """Whether a quaternion's width is the same as a vector type's.

    A float quaternion rotates a `float3` directly and a double one rotates a `double3`; a
    mixed pair goes through the header's widened function, which is a separate name.
    """
    width:Optional[str] = _scalar_width(dtype)
    if width is None:
        return False
    return width == ("float64" if vector_name.startswith("double") else "float32")


def _write_if_changed(path:str, text:str)->None:
    existing:Optional[str] = None
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as file:
            existing = file.read()
    if existing != text:
        with open(path, "w", encoding="utf-8", newline="\n") as file:
            file.write(text)


def header_source()->str:
    """The header's text, read from the file beside this module.

    Read on demand rather than cached: the file is a normal source a user may edit while
    debugging, and a stale copy would make the cache silently disagree with it.
    """
    with open(HEADER_SOURCE, "r", encoding="utf-8") as file:
        return file.read()


def include_directory(source_directory:str)->str:
    """The `include/` folder beside a generated source, where its headers live."""
    return os.path.join(source_directory, INCLUDE_FOLDER)


def install_header(source_directory:str)->str:
    """Put the header in the cache's `include/` folder and return its path.

    Refreshed whenever the package copy changed, so editing the header and re-running a
    kernel picks the new text up instead of compiling a stale one.
    """
    target:str = include_directory(source_directory)
    os.makedirs(target, exist_ok=True)
    path:str = os.path.join(target, HEADER_NAME)
    _write_if_changed(path, header_source())
    return path


def include_line(backends_needed:Tuple[str, ...], backend:str, directory:str)->str:
    """Return the lines a generated source needs to reach the header.

    Here is the whole of it: the enable macros for the widths in use, then one include. The
    structure declarations and the arithmetic both come from the header, so a source that
    includes it has nothing else to say about a quaternion.

    The caller passes the **cache directory**; this function places the file, so do not
    pre-append `include/` at the call site or the folder nests twice.
    """
    install_header(directory)
    qualifier:str = "__device__" if backend in ("cuda", "hip") else ""
    macros:str = "".join(f"#define {need}\n" for need in backends_needed)
    return (f"#define QUAT_QUAL {qualifier}\n"
            f"{macros}"
            f'#include "{INCLUDE_FOLDER}/{HEADER_NAME}"')
