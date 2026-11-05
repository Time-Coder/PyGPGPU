"""Complex numbers on the device.

Numpy carries a complex value as a pair of reals, and device code has to do the same: neither
OpenCL C, CUDA C, MSL nor plain C has a complex type. The two structures live in one fixed C
header (`include/complex.h`) that declares both widths and every operation on them, so a
generated source that includes it needs no declaration of its own.

The structures carry **fixed** C names (`c64` / `c128`) rather than the content hashes other
structures get. That is deliberate, for the same reason the quaternion header does it: a hash
cannot be spelled in advance, and the header has to declare the type it operates on. There is
a second reason here, and it is the important one: the member names are numpy's own layout
(`re`, `im`), so a `complex64` array binds to a `c64*` parameter with no conversion at all --
the two layouts are byte-identical. A dtype with any other member names would need a copy.

A generated source chooses which blocks it needs through two macros:
`COMPLEX_NEED_FLOAT` / `COMPLEX_NEED_DOUBLE`. The guards matter because a `double` block only
compiles where the width does -- an OpenCL device without `cl_khr_fp64` rejects a `double`
variable even inside an unused function.

`#include` is expanded by the C preprocessor *before* the backend compiler runs, so the header
is plain text folded into the generated file. The driver never resolves an include path of its
own, which is why the header is copied into the cache beside the sources that include it.

The *host* side of these names is not here: it is ``pygpgpu.vec_types.funcs`` (pygf's element-
wise helpers, which take a complex scalar), and this module only says how a complex value is
spelled on a device.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np

__all__ = [
    "is_complex",
    "complex_dtype_of",
    "complex_type_name",
    "is_complex_type",
    "needs",
    "mixed_needs",
    "wider",
    "wider_dtype",
    "member_names",
    "scalar_width",
    "HEADER_NAME",
    "HEADER_SOURCE",
    "header_source",
    "include_directory",
    "install_header",
    "include_line",
]

# The member names numpy's complex layout has, in memory order.
_MEMBERS:tuple = ("re", "im")

HEADER_NAME:str = "complex.h"

# The header is a real file beside the quaternion one, so it can be read and edited like any
# other source. `install_header` copies it into the cache's include folder, because a quoted
# `#include` resolves relative to the including file.
HEADER_SOURCE:str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "include", HEADER_NAME)

# The folder a cache keeps its device headers in.
INCLUDE_FOLDER:str = "include"

# The two fixed C names the header declares, keyed by the numpy dtype name.
_TYPE_NAMES:Dict[str, str] = {"complex64": "c64", "complex128": "c128"}

# The macro that enables each block, keyed by the same dtype names.
_NEED_MACROS:Dict[str, str] = {"complex64": "COMPLEX_NEED_FLOAT",
                               "complex128": "COMPLEX_NEED_DOUBLE"}

# The dtype each fixed name declares.
_DTYPES:Dict[str, np.dtype] = {"complex64": np.dtype(np.complex64),
                                "complex128": np.dtype(np.complex128)}

# The real width of each complex width, which is what a member read answers.
_REAL_NAMES:Dict[str, str] = {"complex64": "float", "complex128": "double"}


def is_complex(dtype:Any) -> bool:
    """True when a dtype is one of the two complex widths this package supports."""
    if not isinstance(dtype, np.dtype):
        try:
            dtype = np.dtype(dtype)
        except (TypeError, ValueError):
            return False
    return dtype.name in _TYPE_NAMES


def complex_dtype_of(dtype:Any) -> Optional[np.dtype]:
    """The complex dtype itself, when `dtype` is one; None otherwise."""
    return np.dtype(dtype) if is_complex(dtype) else None


def complex_type_name(dtype:Any) -> Optional[str]:
    """The fixed C type name a complex dtype translates to, or None.

    A complex value gets a name the header already knows instead of the content hash every
    other structure gets, which is what lets the header declare it.
    """
    if not is_complex(dtype):
        return None
    return _TYPE_NAMES[np.dtype(dtype).name]


def dtype_for_type_name(name:Any) -> Optional[np.dtype]:
    """The complex dtype behind a fixed C name (`c64` / `c128`), or None."""
    if not isinstance(name, str):
        return None
    for dtype_name, type_name in _TYPE_NAMES.items():
        if type_name == name:
            return _DTYPES[dtype_name]
    return None


def is_complex_type(type_name:Any) -> bool:
    """True when a translated device type name is one of the fixed complex names."""
    return dtype_for_type_name(type_name) is not None


def needs(dtype:Any) -> Optional[str]:
    """The macro a generated source defines to compile this complex width's block."""
    if not is_complex(dtype):
        return None
    return _NEED_MACROS[np.dtype(dtype).name]


def mixed_needs() -> tuple:
    """Both macros, for the mixed float/double block.

    A mixed operation widens to the double form, so its body touches both widths and the
    header only compiles it when neither name is missing -- which is also when it is legal,
    since a device without fp64 cannot do either side.
    """
    return tuple(_NEED_MACROS[name] for name in ("complex64", "complex128"))


def wider(first:Any, second:Any) -> Optional[np.dtype]:
    """The complex dtype a mixed pair promotes to; None when neither operand is one.

    The rule is numpy's: a double operand widens the whole operation, so a `complex64` meeting
    a `complex128` (or a double scalar) answers a `complex128`, and nothing is ever rounded
    down. A `complex64` meeting a float scalar stays a `complex64`, which is why the answer
    depends on both operands rather than on the complex one alone.
    """
    if not is_complex(first) and not is_complex(second):
        return None
    if _any_double(first, second):
        return _DTYPES["complex128"]
    return _DTYPES["complex64"]


def _is_wide(dtype:Any) -> bool:
    """True for a complex128."""
    return is_complex(dtype) and np.dtype(dtype).name == "complex128"


def _any_double(first:Any, second:Any) -> bool:
    """True when either operand carries the double width, complex or real."""
    for candidate in (first, second):
        if candidate is None:
            continue
        try:
            name:str = np.dtype(candidate).name
        except (TypeError, ValueError):
            continue
        if name == "complex128" or name == "float64":
            return True
    return False


def wider_dtype(dtype:Any) -> Optional[np.dtype]:
    """The double form of one complex dtype, or None when it is not a complex width."""
    return _DTYPES["complex128"] if is_complex(dtype) else None


def member_names() -> tuple:
    """The component names of a complex value, in memory order."""
    return _MEMBERS


def scalar_width(dtype:Any) -> Optional[str]:
    """The translator's scalar type name of one component (`float` / `double`)."""
    if not is_complex(dtype):
        return None
    return _REAL_NAMES[np.dtype(dtype).name]


def _write_if_changed(path:str, text:str) -> None:
    existing:Optional[str] = None
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as file:
            existing = file.read()
    if existing != text:
        with open(path, "w", encoding="utf-8", newline="\n") as file:
            file.write(text)


def header_source() -> str:
    """The header's text, read from the file beside this module.

    Read on demand rather than cached: the file is a normal source a user may edit while
    debugging, and a stale copy would make the cache silently disagree with it.
    """
    with open(HEADER_SOURCE, "r", encoding="utf-8") as file:
        return file.read()


def include_directory(source_directory:str) -> str:
    """The `include/` folder beside a generated source, where its headers live."""
    return os.path.join(source_directory, INCLUDE_FOLDER)


def install_header(source_directory:str) -> str:
    """Put the header in the cache's `include/` folder and return its path.

    Refreshed whenever the package copy changed, so editing the header and re-running a
    kernel picks the new text up instead of compiling a stale one.
    """
    target:str = include_directory(source_directory)
    os.makedirs(target, exist_ok=True)
    path:str = os.path.join(target, HEADER_NAME)
    _write_if_changed(path, header_source())
    return path


def include_line(backends_needed:tuple, backend:str, directory:str) -> str:
    """Return the lines a generated source needs to reach the header.

    Here is the whole of it: the enable macros for the widths in use, then one include. The
    structure declarations and the arithmetic both come from the header, so a source that
    includes it has nothing else to say about a complex number.

    The caller passes the **cache directory**; this function places the file, so do not
    pre-append `include/` at the call site or the folder nests twice.
    """
    install_header(directory)
    qualifier:str = "__device__" if backend in ("cuda", "hip") else ""
    macros:str = "".join(f"#define {need}\n" for need in backends_needed)
    return (f"#define COMPLEX_QUAL {qualifier}\n"
            f"{macros}"
            f'#include "{INCLUDE_FOLDER}/{HEADER_NAME}"')
