# The value slots: outer scope, scalars, and the return coercion.
from __future__ import annotations
from typing import (
    Any,
    Dict,
)
import numpy as np
from ..translator import (
    DeviceType,
)



def _outer_slot_dtype(t:DeviceType)->"np.dtype[Any]":
    """The numpy dtype of a one-element buffer carrying a declared outer scalar."""
    return np.dtype(_SCALAR_SLOT_DTYPES[t])


def _scalar_slot_value(value:object, t:DeviceType)->object:
    """An outer scalar as the element its buffer stores."""
    if t == "bool":
        return np.uint8(1 if value else 0)
    return value


def _outer_scope_value(raw:object, t:DeviceType)->object:
    """A declared outer scalar's buffer element as the python value to store back."""
    if t == "bool":
        return bool(int(raw))
    if t in ("char", "uchar", "short", "ushort", "int", "uint", "long", "ulong"):
        return int(raw)
    if t in ("c64", "c128"):
        # A complex value comes back as a complex number, never as the record the buffer
        # holds: the caller assigned a `complex`, so that is what the scope gets again.
        return complex(raw)
    return float(raw)


_SCALAR_SLOT_DTYPES:Dict[str, object] = {
    "bool": np.uint8,
    "char": np.int8,
    "uchar": np.uint8,
    "short": np.int16,
    "ushort": np.uint16,
    "int": np.int32,
    "uint": np.uint32,
    "long": np.int64,
    "ulong": np.uint64,
    "float": np.float32,
    "double": np.float64,
    # A complex value travels as a one-element complex buffer, which is the same bytes as
    # the fixed structure the device declares -- so a slot and its read-back agree.
    "c64": np.complex64,
    "c128": np.complex128,
}


def _kernel_scalar_argument(value:object, t:DeviceType)->object:
    """A kernel argument in the type the device signature declares for it.

    Only `bool` needs converting: the translator passes it as a byte, because a real
    bool is rejected as a kernel parameter by OpenCL and has no CUDA binding entry. The
    conversion happens on the argument handed to the launch, not on the caller's object,
    so the values a return site hands back are still the caller's own objects.
    """
    if t == "bool":
        return np.uint8(1 if value else 0)
    return value


def _return_value(raw:object, dtype:object, kind:str)->object:
    """One value slot's stored element as the python value it represents.

    ``kind`` is the translator's type name for the returned position, which is how a
    byte-valued slot is turned back into the bool or the char it came from.
    """
    global _VEC_RETURN_TYPES
    if not _VEC_RETURN_TYPES:
        # The host's own classes: a value slot holds the host layout (a 3-component vector
        # has three components there), which is what the caller gets back.
        from ...vec_types.layout import _host_classes

        for name, cls in _host_classes().items():
            _VEC_RETURN_TYPES.setdefault(str(np.dtype(cls.numpy_dtype())), cls)
    vector:object = _VEC_RETURN_TYPES.get(str(np.dtype(dtype)))
    if vector is not None:
        return vector.from_buffer_copy(np.asarray(raw).tobytes())
    value:object = np.asarray(raw).reshape(())[()]
    if isinstance(dtype, np.dtype) and dtype.fields is not None:
        # A structure slot holds a record: the caller gets the record itself, and reads its
        # members exactly as the kernel body did (`result["i"]`).
        return value
    if kind == "c64" or kind == "c128":
        # A complex slot holds the complex layout, so the value is already a complex number.
        return complex(value)
    if kind == "bool":
        return bool(int(value))
    if kind in ("char", "uchar", "short", "ushort", "int", "uint", "long", "ulong"):
        return int(value)
    return float(value)


_VEC_RETURN_TYPES:Dict[str, object] = {}
