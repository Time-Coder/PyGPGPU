# The module-level caches, deliberately in a module with no imports of its own
# so the other parts can share them without a cycle. They are mutated in place
# and must never be re-created.
from __future__ import annotations
import ast
import ctypes
from typing import Dict, List, Tuple



_LIFTED_TREES:Dict[str, ast.Module] = {}


_LIFTED_TREES:Dict[str, ast.Module] = {}
_LIFTED_VALUES:Dict[str, Dict[str, object]] = {}


_LIFTED_VALUES:Dict[str, Dict[str, object]] = {}
# class -> the members a lifted construction supplied, in the constructor order.
_LIFTED_MEMBERS:Dict[type, List[Tuple[str, type]]] = {}



# The structures built for values a body constructs, keyed by the class: the translator asks
# this when it reaches the constructor call, and the scan above is what filled it in.
_CONSTRUCTED:Dict[type, type] = {}



_CONSTRUCTED_PARAMETERS:Dict[type, List[str]] = {}



# `int` widens to `float`; nothing else mixes.
_PROMOTIONS:Dict[Tuple[type, type], type] = {
    (ctypes.c_int32, ctypes.c_float): ctypes.c_float,
    (ctypes.c_int32, ctypes.c_double): ctypes.c_double,
    (ctypes.c_float, ctypes.c_double): ctypes.c_double,
}



_STRUCT_CACHE:Dict[Tuple[object, ...], type] = {}



# The ctypes class each device type name stands for. A synthesized structure member is
# declared with one of these, so a mapping's device type name has to come back as the class
# that has that width -- `long` is `c_int64`, not `c_int32`.
_DEVICE_SCALAR_CLASSES:Dict[str, type] = {
    "bool": ctypes.c_bool,
    "char": ctypes.c_int8, "short": ctypes.c_int16,
    "int": ctypes.c_int32, "long": ctypes.c_int64,
    "uchar": ctypes.c_uint8, "ushort": ctypes.c_uint16,
    "uint": ctypes.c_uint32, "ulong": ctypes.c_uint64,
    "float": ctypes.c_float, "double": ctypes.c_double,
}



# The member a python `int`/`float` becomes when the kernel declares no mapping. These are
# the widths every existing kernel already has, so the default is unchanged.
_DEFAULT_INT_MEMBER:type = ctypes.c_int32


_DEFAULT_INT_MEMBER:type = ctypes.c_int32
_DEFAULT_FLOAT_MEMBER:type = ctypes.c_float



# The prefix an inline construction is lifted to. It never reaches the generated source --
# only the scan's copy of the tree -- so it only has to be a name the body cannot spell.
INLINE_PREFIX:str = "_inline_"
