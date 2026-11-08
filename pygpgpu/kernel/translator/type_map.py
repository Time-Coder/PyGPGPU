# The type system: what a python value is carried as on the device.
from __future__ import annotations
from typing import (Dict, Optional, Set, Tuple)
import numpy as np



INT_TYPES:Set[str] = {"bool", "char", "uchar", "short", "ushort", "int", "uint", "long", "ulong"}
FLOAT_TYPES:Set[str] = {"float", "double"}
SCALAR_TYPES:Set[str] = INT_TYPES | FLOAT_TYPES

DTYPE_MAP:Dict[str, str] = {
    "bool": "bool",
    "int8": "char", "int16": "short", "int32": "int", "int64": "long",
    "uint8": "uchar", "uint16": "ushort", "uint32": "uint", "uint64": "ulong",
    "float16": "half", "float32": "float", "float64": "double",
    # A complex value is one of the fixed structures the complex header declares, not a
    # scalar: the entry here is the type *name* a signature and a subscript use, exactly as
    # a structure name is. `is_complex_type` is what tells the two apart.
    "complex64": "c64", "complex128": "c128",
}

# The ctypes scalar classes a caller may name for a python `int`/`float`, and the device type
# each stands for. The package already reads ctypes classes this way for a vector component
# (`_c_scalar_name`), so this is the same mapping given a caller-facing name: `_c_scalar_name`
# is the vector component's *name*, and this is the advice of what a scalar argument becomes.
CTYPES_SCALAR_TYPES:Dict[object, str] = {}

# The python type a width is requested for, and the only names a `type_map` may carry. Two
# names only, because these are the two python builtins whose width is chosen rather than
# carried: every numpy scalar already names its own width (`np.int64`, `np.float64`), so a
# mapping for those would be a way to override what the caller spelled, which is a different
# (and much larger) feature.
TYPE_MAP_KEYS:Set[str] = {"int", "float"}

# What `int=` may name: the eight integer widths, signed and unsigned, 8 through 64 bits. `bool`
# is not among them -- it is not an integer width a caller would pick for a counter -- and
# neither is `half`, which the device has no scalar for on CUDA and HIP.
INTEGER_MAPPING_TARGETS:Set[str] = {
    "char", "uchar", "short", "ushort", "int", "uint", "long", "ulong",
}

# What `float=` may name: the two floating-point widths. `half` is deliberately absent
# (`DTYPE_MAP` has an entry for it, but NVRTC on this machine cannot resolve `cuda_fp16.h`
# and the CUDA/HIP path refuses it), so the mapping would only move the failure later.
FLOAT_MAPPING_TARGETS:Set[str] = {"float", "double"}

# The integer widths that cannot hold a negative value. A python `int` mapped to one of these
# is refused when it is negative, instead of wrapping to a huge positive number.
UNSIGNED_INTEGER_TYPES:Set[str] = {"uchar", "ushort", "uint", "ulong"}


def _ctypes_scalar_types()->Dict[object, str]:
    """Build the ctypes class table once: the scalar classes a caller may name."""
    global CTYPES_SCALAR_TYPES
    if CTYPES_SCALAR_TYPES:
        return CTYPES_SCALAR_TYPES
    import ctypes as _ctypes

    CTYPES_SCALAR_TYPES = {
        _ctypes.c_bool: "bool",
        _ctypes.c_int8: "char", _ctypes.c_int16: "short",
        _ctypes.c_int32: "int", _ctypes.c_int64: "long",
        _ctypes.c_uint8: "uchar", _ctypes.c_uint16: "ushort",
        _ctypes.c_uint32: "uint", _ctypes.c_uint64: "ulong",
        _ctypes.c_float: "float", _ctypes.c_double: "double",
    }
    return CTYPES_SCALAR_TYPES


def ctypes_device_type_name(expected:object)->Optional[str]:
    """The device type name a backend's scalar ctypes class stands for, or None.

    `kernel_import` parses a handwritten source and gets each parameter as the backend's own
    `ctypes` class; to range-check a value against it, the class has to name the device type
    the same way a `@kernel` argument does. Only scalar classes answer; an array, a structure
    or an unknown class answers None, and the caller then falls back to its own test.
    """
    import ctypes as _ctypes

    if not isinstance(expected, type) or not issubclass(expected, _ctypes._SimpleCData):
        return None                  # noqa: SLF001 (the documented base)
    return _ctypes_scalar_types().get(expected)


def resolve_device_type(value:object, key:str)->str:
    """The device type name a caller's type object stands for, for the given mapping key.

    A `@kernel(int=..., float=...)` mapping is given with a **type object**, because that is
    the spelling a reader of numpy already knows and it carries its width and signedness in
    the name rather than in a string this package would have to parse:

    * a numpy dtype or scalar type -- `np.int64`, `np.dtype("uint32")`, `np.dtype(np.float64)`;
    * a ctypes scalar class -- `ctypes.c_int64`, `ctypes.c_double`;
    * the device type name itself -- `"long"`, `"double"`.

    **The choice is a closed set, and it is not the same set for each key.** `int` may be
    any of the eight integer widths (signed and unsigned, 8 through 64 bits); `float` may be
    `float32` or `float64` and nothing else. The alternative -- accepting whatever the device
    happens to have -- was wrong in both directions: `bool` is not an integer width at all,
    and `float16` (`half`) is out of scope on CUDA and HIP, so a kernel broadened to one of
    them would fail to compile instead of failing here with a reason.

    A string is accepted only when it names one of those widths, so `"int8"` is refused while
    `"char"` is accepted: a string is the *device* type name, and the numpy spelling exists
    because `np.int8` is the spelling a caller actually writes. Every refusal is a
    `ValueError`: a wrong type is a wrong value for this argument, not a syntax error.
    """
    import ctypes as _ctypes

    allowed:Dict[str, Set[str]] = INTEGER_MAPPING_TARGETS if key == "int" else FLOAT_MAPPING_TARGETS

    # A numpy dtype or scalar type, given as the type (`np.int64`) or a dtype (`np.dtype("int8")`).
    if isinstance(value, np.dtype) or (isinstance(value, type)
                                       and issubclass(value, np.generic)):
        dtype:np.dtype = np.dtype(value)
        # A numpy scalar names a width; map it to the device spelling and then to the boards
        # of the closed set, so `np.int8` is accepted only when `char` is (it is).
        device_name:Optional[str] = DTYPE_MAP.get(dtype.name)
        if device_name is None or device_name not in allowed:
            raise ValueError(_mapping_refusal(key, value)) from None
        return device_name

    # A ctypes scalar class.
    if isinstance(value, type) and issubclass(value, _ctypes._SimpleCData):
        device_name = _ctypes_scalar_types().get(value)
        if device_name is not None and device_name in allowed:
            return device_name
        raise ValueError(_mapping_refusal(key, value)) from None

    # A device type name, spelled the way a signature spells it.
    if isinstance(value, str):
        if value in allowed:
            return value
        raise ValueError(_mapping_refusal(key, value)) from None

    raise ValueError(_mapping_refusal(key, value)) from None


def _mapping_refusal(key:str, value:object)->str:
    """The message for a type the mapping key does not accept."""
    if key == "int":
        np_names:str = ("np.int8, np.int16, np.int32, np.int64, "
                        "np.uint8, np.uint16, np.uint32, np.uint64")
        others:str = "A bool, a float and a complex value are not integer widths."
        widths:str = ("char (int8), short (int16), int (int32), long (int64), "
                      "uchar (uint8), ushort (uint16), uint (uint32), ulong (uint64)")
    else:
        np_names = "np.float32, np.float64"
        others = "An integer is not a floating-point width, and `half` is out of scope."
        widths = "float (float32), double (float64)"
    return (f"`{key}=` cannot be {value!r}: a python {key} travels as one of "
            f"{widths} -- written as {np_names}, or as the matching ctypes type. {others}")


def validate_type_map(type_map:Optional[Dict[str, object]])->Dict[str, str]:
    """Normalize a caller's `int=`/`float=` mapping to device type names.

    The keys are the two python builtins whose width is otherwise chosen here, and every
    entry is resolved through `resolve_device_type`, so the decorator and the global setter
    accept exactly the same spellings and reject exactly the same mistakes. Every refusal
    here is a `ValueError`, because a mapping is a value the caller supplied.
    """
    if not type_map:
        return {}
    resolved:Dict[str, str] = {}
    for key, value in type_map.items():
        if key not in TYPE_MAP_KEYS:
            raise ValueError(
                f"unknown type-mapping key: {key!r}; only 'int' and 'float' can be mapped "
                f"(a numpy scalar already names its own width)") from None
        resolved[key] = resolve_device_type(value, key)
    return resolved


# The device type names that are complex values rather than structures. Unlike a structure
# they travel by value, their arithmetic is generated (not refused), and their two members
# are read with the header's own accessors.
COMPLEX_TYPE_NAMES:Set[str] = {"c64", "c128"}

# A python `complex` has no width of its own, so its width follows the `float` mapping: a
# kernel that asks for 64-bit reals asks for 64-bit complex values beside them, and the
# default 32-bit real pairs with the single-precision complex. The two widths are the only
# ones a complex value has, so the table is written out for both of the mapping's own names
# and anything else is read as the single-precision pair.
_COMPLEX_WIDTH_PER_REAL:Dict[str, str] = {"float": "c64", "double": "c128"}


def python_complex_type(float_type:str)->str:
    """The complex width a python `complex` travels as beside a real of this width.

    A python `complex` has no width of its own -- the same rule a python `float` follows --
    so the two are always chosen together: `float=` naming a double gives `c128`, and the
    32-bit default (`float`, or no mapping at all) gives `c64`. This is the single place that
    decides it, which is what keeps a python complex value from being the one value in a body
    whose width disagrees with the reals beside it: the scalar path (`infer_ctype`), a body
    literal and `complex(re, im)` all read their width from here, so a `float=` mapping moves
    all three together.
    """
    return _COMPLEX_WIDTH_PER_REAL.get(float_type, "c64")

# The range of every integer the device can hold, as (minimum, maximum). A python `int` is
# arbitrary precision, so a value outside the width it is about to travel as used to wrap
# silently: `k(2 ** 31 + 5, out)` reached the device as -2147483648, and the kernel then
# computed with a number the caller never wrote. Refusing it is the only honest answer, and
# the message names the spelling that carries the width the caller actually needs.
_INTEGER_RANGES:Dict[str, Tuple[int, int]] = {
    "char": (-(2 ** 7), 2 ** 7 - 1),
    "uchar": (0, 2 ** 8 - 1),
    "short": (-(2 ** 15), 2 ** 15 - 1),
    "ushort": (0, 2 ** 16 - 1),
    "int": (-(2 ** 31), 2 ** 31 - 1),
    "uint": (0, 2 ** 32 - 1),
    "long": (-(2 ** 63), 2 ** 63 - 1),
    "ulong": (0, 2 ** 64 - 1),
}


def check_integer_range(value:int, device_type:str, where:str='')->None:
    """Public entry point for the width check, shared by every path that chooses a type.

    A scalar argument, a structure member, and an imported kernel's declared parameter all
    have to agree on what "does not fit" means. `where` names the member when the caller has
    one, so the message points at it rather than at the kernel as a whole.
    """
    _check_integer_fits(value, device_type, where)


def _check_integer_fits(value:int, device_type:str, where:str='')->None:
    """Refuse a python `int` the device type it travels as cannot represent."""
    bounds:Optional[Tuple[int, int]] = _INTEGER_RANGES.get(device_type)
    if bounds is None:
        return                      # `bool`: python bools are not ints for this rule
    low, high = bounds
    if low <= value <= high:
        return
    # The suggestion is the numpy spelling that holds this value, and only when one does: an
    # integer beyond 64 bits has no device type at all, so pointing at `np.int64` there would
    # be pointing at a second failure.
    if 0 <= value <= 2 ** 64 - 1 and value > 2 ** 63 - 1:
        suggestion = f" Pass it as `np.uint64({value})` to keep that width."
    elif -(2 ** 63) <= value <= 2 ** 63 - 1:
        suggestion = (f" Pass it as `np.int64({value})` to keep that width."
                      if value < 0 else
                      f" Pass it as `np.int64({value})` or `np.uint64({value})` "
                      f"to keep that width.")
    else:
        suggestion = " The device has no integer type that wide."
    raise ValueError(
        f"{where + ': ' if where else ''}the integer {value} does not fit the device "
        f"type '{device_type}' "
        f"({low}..{high}); a python int has no width of its own, so it travels as the "
        f"width this kernel chose for `int`.{suggestion}") from None

# The numpy scalars whose device type is the *narrow* one, as it always was: an 8- or
# 16-bit value is an `int` on the device, and a float64 is a `float`. A caller who wants
# a width of its own says so (`np.uint32`, `np.int64`), and that width is kept.
NARROWED_SCALARS:Dict[str, str] = {
    "int8": "int", "int16": "int", "uint8": "int", "uint16": "int",
    "float16": "float", "float64": "float",
}
