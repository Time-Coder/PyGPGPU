# The scalar layer: how a python scalar becomes the device's own bytes, and the
# MSL size table the encoder reads. The comments about the device's widths and
# the complex records moved verbatim with the code.
from __future__ import annotations
import ctypes
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from ..buffer import Buffer
import numpy as np



# A bracket side: one axis count, or one per axis.
Extents = Union[int, Tuple[int, ...]]

# What `_bind_args` produces per argument: the index to bind at, the buffer a pointer
# argument uploaded (or the bytes a scalar occupies), and whether it is a buffer.
BoundEntry = Tuple[int, Union["Buffer", bytes], bool]


# MTLSize is a three-NSUInteger struct, passed by value to the dispatch calls.
class MTLSize(ctypes.Structure):
    # `MTLSize`'s three fields are `NSUInteger`, which is pointer-sized -- 8 bytes on
    # every Mac. `c_ulong` is that size on macOS but 4 bytes on Windows, so naming
    # `c_size_t` keeps the struct the right size regardless of the host.
    _fields_ = [("width", ctypes.c_size_t), ("height", ctypes.c_size_t),
                ("depth", ctypes.c_size_t)]


def _axis_extents(value:Extents, what:str)->Tuple[int, int, int]:
    """One side of the bracket as Metal's three axes, padded with 1.

    `MTLSize` is always three-dimensional, while a bracket may name one axis (`4`)
    or three (`(4, 1, 1)`). A scalar becomes `(4, 1, 1)` and a shorter tuple is
    padded, because a missing axis means one group along it -- zero would ask for
    an empty dispatch, which is never what a caller wrote.
    """
    if isinstance(value, int):
        return (int(value), 1, 1)
    extents:Tuple[int, ...] = tuple(int(v) for v in value)
    if len(extents) > 3:
        raise ValueError(
            f"the {what} side of a work size bracket has {len(extents)} axes; "
            f"Metal dispatches at most three")
    return extents + (1,) * (3 - len(extents))


def _mtl_size(extents:Tuple[int, ...])->MTLSize:
    """One, two or three axes as the struct Metal's dispatch calls take."""
    values:List[int] = [int(v) for v in extents] + [1, 1, 1]
    return MTLSize(values[0], values[1], values[2])


# The bytes each scalar kernel parameter occupies, in MSL's own layout. This is
# what `setBytes:length:atIndex:` has to copy, so it must be the *device's* size
# rather than the host's: a python `int` is 8 bytes, while `int` in MSL is 4.
_SCALAR_SIZES:Dict[str, int] = {
    "bool": 1, "char": 1, "uchar": 1,
    "short": 2, "ushort": 2, "half": 2,
    "int": 4, "uint": 4, "float": 4,
    "long": 8, "ulong": 8, "double": 8,
}

# A complex value is a *structure* of two reals rather than a scalar, and it is bound the way
# a scalar is: the generated signature declares `c64 value`, a value in MSL's thread address
# space, so the host writes the record's bytes with `setBytes:`. The two names are the fixed
# ones the complex header declares (see `program_info.COMPLEX_TYPE_NAMES`), and each spells
# the format of *one* component -- `re` then `im`, in declaration order, which is numpy's own
# complex layout, so a `complex64`/`complex128` array argument and this scalar share one
# memory shape.
_COMPLEX_FORMATS:Dict[str, str] = {"c64": "<f", "c128": "<d"}


def _scalar_bytes(device_type:str, value:object)->bytes:
    """One scalar argument as the bytes the device's own type would hold.

    A kernel's extent parameter is declared `const int`, so the host has to send
    four bytes even though python's `int` is wider -- the length passed to
    `setBytes:` is what the device copies, and a mismatch makes the pipeline read
    half a value.
    """
    if device_type in _COMPLEX_FORMATS:
        return _complex_bytes(device_type, value)
    size:int = _SCALAR_SIZES.get(device_type, 4)
    if device_type in ("float",):
        return _pack_real(device_type, "<f", float(value), 3.4028235e38)
    if device_type in ("double",):
        return _pack_real(device_type, "<d", float(value), None)
    if device_type in ("half",):
        # `<e` is IEEE half in little endian, which is what `setBytes:` expects and
        # what Metal's own storage layout is. `numpy.float16(...).tobytes()` answers
        # the *host's* byte order, so on a big-endian host it would disagree with
        # every other branch here -- all of which spell the order explicitly.
        return _pack_real(device_type, "<e", float(value), 65504.0)
    if device_type in ("bool", "uchar", "char"):
        return _pack_integer(device_type, size, int(bool(value) if device_type == "bool"
                                                    else value),
                             signed=device_type == "char")
    return _pack_integer(device_type, size, int(value),
                         signed=device_type in ("char", "short", "int", "long"))


# The inclusive range each declared scalar type can hold. `to_bytes` raises a bare
# `OverflowError` ("int too big to convert") for a value outside it, which says
# nothing about which parameter was wrong or what the type accepts.
_SCALAR_RANGES:Dict[str, Tuple[int, int]] = {
    "bool": (0, 1),
    "char": (-128, 127), "uchar": (0, 255),
    "short": (-32768, 32767), "ushort": (0, 65535),
    "int": (-2147483648, 2147483647), "uint": (0, 4294967295),
    "long": (-9223372036854775808, 9223372036854775807),
    "ulong": (0, 18446744073709551615),
}


def _pack_real(device_type:str, code:str, value:float,
               largest:Optional[float])->bytes:
    """One floating-point scalar, with an overflow spelled out.

    `struct.pack` raises `OverflowError: float too large to pack with f format -- the
    format letter names neither the kernel parameter's type nor the largest value it
    holds. ±infinity and NaN are accepted (they are valid IEEE values, not overflow).
    """
    import math
    import struct
    if largest is not None and math.isfinite(value) and abs(value) > largest:
        raise OverflowError(
            f"a kernel parameter declared {device_type} cannot hold {value!r}: "
            f"its largest finite value is {largest}")
    return struct.pack(code, value)


def _complex_bytes(device_type:str, value:object)->bytes:
    """One complex argument as the two real components MSL's record holds.

    The components are packed through `_pack_real`, so a `c64` refuses a value its `float`
    component cannot hold with the same message a `float` parameter gives (and keeps ±inf
    and NaN, which IEEE allows). The record is what `setBytes:` copies, so its length is the
    structure's -- 8 bytes for a `c64`, 16 for a `c128`.
    """
    code:str = _COMPLEX_FORMATS[device_type]
    largest:Optional[float] = 3.4028235e38 if device_type == "c64" else None
    real:float = float(np.real(value))
    imaginary:float = float(np.imag(value))
    return (_pack_real(device_type, code, real, largest)
            + _pack_real(device_type, code, imaginary, largest))

def _pack_integer(device_type:str, size:int, value:int, signed:bool)->bytes:
    """One integer scalar as `size` little-endian bytes, with the range checked."""
    limits:Optional[Tuple[int, int]] = _SCALAR_RANGES.get(device_type)
    if limits is not None and not (limits[0] <= value <= limits[1]):
        raise OverflowError(
            f"a kernel parameter declared {device_type} cannot hold {value}: "
            f"its range is {limits[0]}..{limits[1]}")
    return value.to_bytes(size, "little", signed=signed)
