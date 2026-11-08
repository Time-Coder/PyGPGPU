# What a python value becomes on the device.
from __future__ import annotations
from .notation import IMAGE, PTR, DeviceType
from .type_map import (
    COMPLEX_TYPE_NAMES,
    DTYPE_MAP,
    FLOAT_TYPES,
    INT_TYPES,
    NARROWED_SCALARS,
    _check_integer_fits,
    python_complex_type,
)
from .vectors import _vec_registry, is_vec
from typing import (Dict, List, Optional)
import numpy as np
from ..structs import dtype_for as structs_dtype_for



def is_int(t:object)->bool:
    return isinstance(t, str) and t in INT_TYPES

def is_float(t:object)->bool:
    return isinstance(t, str) and t in FLOAT_TYPES

def is_complex(t:object)->bool:
    """True for a device type that is a complex value (`c64` / `c128`).

    A complex value is a *value* like a scalar -- it is passed by value, returned in a slot,
    and stored in a local -- but its arithmetic is generated rather than a C operator, and
    its two members are read through the header's accessors. So the predicates that ask
    "can this take part in an ordinary scalar operator" must answer False for it, and the
    predicates that ask "is this a pointer/aggregate" must answer False as well.
    """
    return isinstance(t, str) and t in COMPLEX_TYPE_NAMES

def is_aggregate(t:object)->bool:
    """True for a value the device carries as a *structure* (`c64`/`c128` included).

    The difference from `structs_dtype_for` is that this answers from the type name alone,
    without the registry: a complex type is a structure on the device even though it is not
    a numpy record dtype here.
    """
    return isinstance(t, str) and (t in COMPLEX_TYPE_NAMES
                                   or structs_dtype_for(t) is not None)

def is_ptr(t:object)->bool:
    return isinstance(t, tuple) and len(t) == 2 and t[0] == PTR

def is_image(t:object)->bool:
    """True for the marker a backend-neutral Image argument gets."""
    return isinstance(t, tuple) and len(t) == 3 and t[0] == IMAGE

def image_channels(t:object)->int:
    return int(t[1])

def image_scalar(t:object)->str:
    """The C scalar type of one image channel."""
    return t[2]

def promote(t1:DeviceType, t2:DeviceType)->DeviceType:
    if is_ptr(t1) or is_ptr(t2):
        raise SyntaxError("cannot mix arrays with scalars in arithmetic") from None
    if is_vec(t1) or is_vec(t2):
        raise SyntaxError("vector arithmetic is handled separately") from None
    if "double" in (t1, t2):
        return "double"
    if is_float(t1) or is_float(t2):
        return "float"
    rank:List[str] = ["bool", "char", "uchar", "short", "ushort", "int", "uint", "long", "ulong"]
    if t1 in rank and t2 in rank:
        return t1 if rank.index(t1) >= rank.index(t2) else t2
    return "int"

def infer_ctype(value:object, type_map:Optional[Dict[str, str]]=None)->DeviceType:
    """Actual call argument -> C type.

    `type_map` is the caller's `int=`/`float=` override (already normalized to device type
    names by `validate_type_map`): a python `int`/`float` has no width of its own, so the
    decorator chooses one here. It applies to **nothing else** -- a numpy scalar names its own
    width and keeps it, which is what makes the mapping an override of the two builtins
    rather than a second, competing type system.
    """
    from ...vec_types.layout import _vector_dtype_name
    from .. import structs

    registry:Dict[str, object] = _vec_registry()
    mapping:Dict[str, str] = type_map or {}
    int_type:str = mapping.get("int", "int")
    float_type:str = mapping.get("float", "float")

    # A backend-neutral Image is opaque storage: the kernel only samples or stores
    # through it, so the parameter carries its channel count and scalar type.
    from ...image import Image
    if isinstance(value, Image):
        channel_type = DTYPE_MAP.get(value.dtype.name)
        if channel_type is None:
            raise SyntaxError(
                "Image of dtype %s cannot be sampled: unsupported element type"
                % value.dtype.name) from None
        return (IMAGE, value.channels, channel_type)

    # A `ctypes.Structure` the caller declared is a structure value, before any of the
    # vector lookups: its dtype compares equal to a vector of the same layout
    # (`{float x; float y;}` *is* `float2`), and only the dtype object's alignment flag says
    # which one the caller meant.
    declared:Optional[np.dtype] = structs.structure_dtype_of(value)
    if declared is not None and not isinstance(value, np.ndarray):
        from ...vec_types.layout import _vector_dtype_name, _host_classes

        declared_class:type = type(value)
        is_vector_value:bool = (
            _vector_dtype_name(declared) is not None
            and (declared_class in _host_classes().values()
                 or any(issubclass(declared_class, host)
                        for host in _host_classes().values())))
        if is_vector_value:
            # A vector type is a ctypes.Structure with a vector layout: it travels by
            # value, because the device reads its components directly. A caller-declared
            # class with a vector-identical layout (Vec2Like) is NOT one of the package's
            # vectors, so it keeps the structure treatment.
            return structs.register(declared)
        if isinstance(value, type):
            # The class itself (a spelling for the layout, not a value): it names the
            # structure, and travels by value like any type.
            return structs.register(declared)
        # A structure argument always travels as a pointer: copying a struct per launch is
        # work the device never needs, a write through it reaches the caller's object, and
        # one spelling covers both read-only (the signature adds `const`) and written.
        return (PTR, structs.register(declared))

    if isinstance(value, np.void):
        # A structured scalar is either a vector element or a structure value: both are
        # described completely by the dtype, so both become a device type.
        if structs.is_declared_structure(value.dtype):
            # An element of a ctypes-declared array reaches here as a record, and the
            # declaration wins over the coincidental vector layout.
            return (PTR, structs.register(value.dtype))
        vec_name:Optional[str] = _vector_dtype_name(value.dtype)
        if vec_name is not None:
            return vec_name
        if structs.is_record(value.dtype):
            return (PTR, structs.register(value.dtype))
        raise SyntaxError(f"unsupported numpy structured scalar: {value.dtype}") from None

    for vec_name, cls in registry.items():
        if isinstance(value, cls):
            return vec_name

    # A vector instance from the host library (`pygf`) is not one of the driver's classes,
    # but its bases name the same device type: `point3f` -> `float3`, `color3d` -> `double3`.
    # A numpy scalar is never a vector, even where the class name collides: `np.int8` is
    # `numpy.int8`, while the registry has an eight-component `int8` vector, and a scalar
    # reaching that lookup became a vector parameter the device could not bind.
    if not isinstance(value, np.generic):
        for base in type(value).__mro__:
            if base.__name__ in registry:
                return base.__name__

    if isinstance(value, np.ndarray):
        if value.dtype.fields is not None:
            if structs.is_declared_structure(value.dtype):
                # An array of a caller's `ctypes.Structure`: structured, whatever layout.
                return (PTR, structs.register(value.dtype))
            vec_name = _vector_dtype_name(value.dtype)
            if vec_name is not None:
                return (PTR, vec_name)
            if structs.is_record(value.dtype):
                return (PTR, structs.register(value.dtype))
            raise SyntaxError(f"unsupported numpy structured dtype: {value.dtype}") from None
        if value.dtype.name not in DTYPE_MAP:
            raise SyntaxError(f"unsupported numpy dtype: {value.dtype.name}") from None
        return (PTR, DTYPE_MAP[value.dtype.name])
    if isinstance(value, (bool, np.bool_)):
        return "bool"
    # A numpy scalar carries its own width and signedness, and `np.uint32(x)` is how a
    # caller says "unsigned 32-bit" without an annotation the translator would ignore.
    # Collapsing every integer to `int` made an unsigned mix (`x * 0x9E3779B9`, `x >> 16`)
    # signed, so the multiply overflowed and the shift filled with ones.
    #
    # The narrow widths keep their old answer: an 8- or 16-bit numpy scalar is `int` on the
    # device, as it always was (see tests/test_kernel_narrow_arguments.py), and a float64
    # is `float`. The widths a caller reaches for *deliberately* -- `np.uint32` for an
    # unsigned mix, `np.int64`/`np.uint64` for a wide counter -- are kept as they are,
    # because there would be no reason to spell them otherwise. A python `int` has no
    # width of its own and stays `int`.
    if isinstance(value, np.integer):
        dtype_name:str = np.dtype(type(value)).name
        if dtype_name in NARROWED_SCALARS:
            return NARROWED_SCALARS[dtype_name]
        if dtype_name in DTYPE_MAP:
            return DTYPE_MAP[dtype_name]
        return int_type
    if isinstance(value, int):
        _check_integer_fits(value, int_type)
        return int_type
    if isinstance(value, np.floating):
        dtype_name = np.dtype(type(value)).name
        if dtype_name in NARROWED_SCALARS:
            return NARROWED_SCALARS[dtype_name]
        if dtype_name in DTYPE_MAP:
            return DTYPE_MAP[dtype_name]
        return float_type
    if isinstance(value, float):
        return float_type
    # A complex scalar. `np.complex64` and `np.complex128` are numpy scalars (they are
    # `np.complexfloating`, not `np.floating`), and a python `complex` has no width of its
    # own -- the same rule a python `float` follows, which is why the two widths are chosen
    # together rather than one of them staying fixed: a kernel that maps `float` to a double
    # is asking for 64-bit real arithmetic, and a python complex beside it should not be the
    # single value in the body that silently lost half its precision. The default is the
    # other way round and just as important: a python float travels as `float`, so a python
    # complex beside it travels as `c64`. A python complex is never one of the numpy scalars,
    # so `isinstance(complex(1, 2), complex)` is checked on its own.
    if isinstance(value, np.complexfloating):
        dtype_name = np.dtype(type(value)).name
        return DTYPE_MAP.get(dtype_name, "c128")
    if isinstance(value, complex):
        return python_complex_type(float_type)
    if _is_device_handle(value):
        # A texture or surface reaches the device as an opaque 64-bit handle, which
        # the kernel reads through tex2D or writes through surf2Dwrite.
        return _handle_device_type(value)
    raise SyntaxError(f"unsupported argument type: {type(value).__name__} (only numpy arrays, int/float/bool/complex scalars and texture handles are supported)") from None


TEXTURE_HANDLE_TYPES = ("TextureObject", "cudaTextureObject_t")
SURFACE_HANDLE_TYPES = ("SurfaceObject", "cudaSurfaceObject_t")


def _is_device_handle(value:object)->bool:
    """True for the objects a kernel receives as an opaque texture or surface handle."""
    return type(value).__name__ in TEXTURE_HANDLE_TYPES + SURFACE_HANDLE_TYPES


def _handle_device_type(value:object)->str:
    """The device-side type name a handle argument takes in a kernel signature."""
    if type(value).__name__ in SURFACE_HANDLE_TYPES:
        return "cudaSurfaceObject_t"
    return "cudaTextureObject_t"
