# The vector registry: the one place the vector types are looked up.
from __future__ import annotations
from .notation import Translated
from typing import (Dict, List, Tuple)
import numpy as np



_C_FIELD_TYPES:Dict[object, str] = {}

# Declared here (and filled on first use) because importing the driver tables eagerly would
# load a GPU runtime at `import pygpgpu` time.
_VEC_REGISTRY:Dict[str, object]

def _vec_registry()->Dict[str, object]:
    """vec type name -> ctypes class (e.g. float4), built lazily."""
    global _VEC_REGISTRY
    try:
        return _VEC_REGISTRY
    except NameError:
        pass
    from ...opencl.driver import CLInfo
    from .. import structs

    _VEC_REGISTRY = dict(CLInfo.vec_types)
    for name, cls in _VEC_REGISTRY.items():
        for field_name, field_type in cls._fields_:
            _C_FIELD_TYPES[(name, field_name)] = _c_scalar_name(field_type)
        # A vector is a `ctypes.Structure` with an aligned dtype, which is the evidence
        # `is_declared_structure` uses for a structure; claiming the dtype first keeps the
        # vector reading when a caller's class happens to have the same layout.
        try:
            structs.claim_vector_dtype(np.dtype(cls._fields_))
        except Exception:
            pass
    return _VEC_REGISTRY

def _c_scalar_name(ctypes_type:object)->str:
    import ctypes as _ctypes
    sizes = {_ctypes.c_float: "float", _ctypes.c_double: "double",
             _ctypes.c_int8: "char", _ctypes.c_int16: "short", _ctypes.c_int32: "int",
             _ctypes.c_int64: "long", _ctypes.c_uint8: "uchar", _ctypes.c_uint16: "ushort",
             _ctypes.c_uint32: "uint", _ctypes.c_uint64: "ulong"}
    return sizes.get(ctypes_type, "float")

def is_vec(t:object)->bool:
    return isinstance(t, str) and t in _vec_registry()

# A 3-component vector is padded to 16 bytes for the device ABI, so its ctypes struct carries
# a trailing `_` field that is not a component. Everything that names components counts them
# through this helper, so the padding never reaches a constructor, a swizzle or a print.
def _vec_components(vec:str)->List[Translated]:
    """The real (field name, ctypes type) components of a vector type, in order."""
    return [field for field in _vec_registry()[vec]._fields_ if field[0] != "_"]

def _vec_field_names(cls_or_name:object)->List[str]:
    """The component names of a vector type, given the class or its name.

    Everything that expands a vector into per-component code goes through this, so the
    padding field of a 3-component vector never becomes a member the device does not have.
    """
    fields:Tuple[Tuple[str, type], ...] = (cls_or_name._fields_
                  if isinstance(cls_or_name, type)
                  else _vec_registry()[cls_or_name]._fields_)
    return [name for name, _ in fields if name != "_"]

# Colour-style component names, as in `color.rgb`. OpenCL C accepts them natively but CUDA's
# vector types only have x/y/z/w, so the translator normalizes them for every backend.
_VEC_COLOR_ALIASES:Dict[str, str] = {"r": "x", "g": "y", "b": "z", "a": "w"}


def _vec_component_name(text:str)->str:
    """`rgb` -> `xyz`: the device spelling of a component or swizzle."""
    return "".join(_VEC_COLOR_ALIASES.get(char, char) for char in text)
